// EngineLoopV2+MTPFinalize.swift
//
// Finalize-time target-authoritative acceptance, streaming, and KV rollback.

import Foundation
import MLX

extension EngineLoopV2 {
    /// Runs before ordinary token finalization and any deferred request frees.
    /// Sample-token readback alone need not fence independent assistant roots
    /// submitted later. Only explicitly opted-in drafters pay this extra fence.
    func finishStatefulMTPEvaluation(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound,
            let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter,
            stateful.requiresCommittedObservationFence
        else { return }
        let observations = round.committedObservationRows.map { ($0.id, $0.assistantState) }
        let verifying = (round.verify?.rows ?? []).compactMap { row in
            row.assistantState.map { (row.id, $0) }
        }
        var seen = Set<ObjectIdentifier>()
        for (id, state) in observations + verifying {
            guard seen.insert(ObjectIdentifier(state)).inserted else { continue }
            var requiredCompletionFinished = false
            var requiredFailureReason: CBv2NativeShutdownFault.Reason = .nativeWorkFailed
            do {
                let roots = stateful.evaluationTargets(for: state)
                if let handle = retainNativeWork(roots, owners: [state]) {
                    step.nativeRootIDs.append(handle)
                }
                try requireNativeWork()
                if !roots.isEmpty {
                    try withError { error in
                        eval(roots)
                        try error.check()
                    }
                    CBv2CoreInstrumentation.recordHostSync()
                }
                if nativeShutdownState != nil {
                    try requireNativeWork()
                    guard let split = stateful as? any CBv2NativeMTPCompletionSplitting else {
                        throw CBv2NativeShutdownError.unsupportedConsumer
                    }
                    // The exact same existing assistant construction fence.
                    // Never hold the first-winner lock while synchronizing.
                    requiredFailureReason = .capturedFenceFailed
                    try nativeRequiredAssistantFenceForTesting?(state)
                    try split.fenceRequestStateForNativeCompletion(state)
                    requiredCompletionFinished = true
                    nativeShutdownState?.afterAssistantFenceForTesting?(state)
                    guard
                        try nativeCommitThrowing({
                            try split.commitRequestStateNativeCompletion(state)
                        })
                    else { return }
                } else {
                    try stateful.requestStateDidFinishEvaluation(state)
                }
            } catch {
                if let tracking = nativeShutdownState {
                    guard tracking.mayExecute else { return }
                    guard requiredCompletionFinished else {
                        // A failed REQUIRED eval/fence cannot be rehabilitated
                        // by a later successful general cleanup fence. Preserve
                        // the original failure and every real owner permanently.
                        failNativeCompletion(requiredFailureReason)
                        return
                    }
                    // Only post-completion metadata/measurement refusal reaches
                    // the existing drained error-cleanup path below.
                    do { try tracking.fenceCapturedStreams() } catch {
                        failNativeCompletion(.capturedFenceFailed)
                        return
                    }
                    guard tracking.beginCommit() else { return }
                }
                defer { nativeShutdownState?.endCommit() }
                // A recoverable graph/ownership failure must not return capacity
                // while earlier submitted work can still reference the state.
                if nativeShutdownState == nil {
                    Stream.gpu.synchronize()
                    Stream.cpu.synchronize()
                }
                step.discard.insert(id)
                finishRequest(id, reason: .error("assistant evaluation ownership: \(error)"))
            }
        }
    }

    /// Minimum target top-K probability mass (parts-per-million) at the
    /// carry position before the next draft may score only the shortlist
    /// rows. Below this the shortlist would too often miss the token the
    /// target actually produces next, costing acceptance for no byte win —
    /// the round simply reads the full head instead. 0.90 keeps greedy
    /// production traffic (mass typically ≥0.99 at K=256) shortlisted while
    /// flat/uncertain positions fall back.
    static let mtpShortlistMassThresholdPPM: Int32 = 900_000

    /// Runs at the step's existing host-sync boundary after ordinary sampled
    /// rows finalize and before deferred KV releases.
    func finalizeMTPRound(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound else { return }
        // Assistant history and carries keep views of these arrays after this
        // finished step; they need not keep its Metal event.
        defer {
            releaseMTPHistoryEvents(
                round.committedObservationArrays + [round.seedHidden].compactMap { $0 }
                    + (round.verify.map { [$0.draftIDs, $0.lastHidden] } ?? []))
        }
        var nativeCommitHeld = false
        if let tracking = nativeShutdownState {
            guard tracking.beginCommit() else { return }
            nativeCommitHeld = true
        }
        defer { if nativeCommitHeld { nativeShutdownState?.endCommit() } }

        // Plain prompt/decode observations may own request-state arrays that
        // the just-fenced target graph referenced. Restore or release them
        // only now, never at cancellation time while evaluation is live.
        for observation in round.committedObservationRows {
            if step.discard.contains(observation.id)
                || scheduler.record(for: observation.id) == nil
            {
                round.deferredAssistantReleases.append(
                    observation.assistantState)
            } else {
                mtp.restoreAssistantState(
                    observation.assistantState, for: observation.id)
            }
        }

        // The ordinary finalize loop has confirmed each seed row's bonus.
        if let seedHidden = round.seedHidden {
            if nativeCommitHeld {
                nativeShutdownState?.endCommit()
                nativeCommitHeld = false
            }
            var seedPolicyTopTwo: [Float]?
            guard
                completedNativeReadback({
                    seedPolicyTopTwo = round.seedPolicyTopTwoValues?.asArray(Float.self)
                })
            else { return }
            if seedPolicyTopTwo != nil { CBv2CoreInstrumentation.recordHostSync() }
            if let tracking = nativeShutdownState {
                guard tracking.beginCommit() else { return }
                nativeCommitHeld = true
            }
            for (id, decodeIndex) in round.seedRows {
                guard !step.discard.contains(id), !step.nonFiniteFailures.contains(id),
                    let rec = scheduler.record(for: id)
                else { continue }
                let margin = seedPolicyTopTwo.map { values in
                    let base = decodeIndex * 2
                    return Double(values[base] - values[base + 1])
                }
                mtp.storeCarry(
                    id: id, token: rec.tokens.last!,
                    hidden: seedHidden[decodeIndex ..< (decodeIndex + 1), 0..., 0...],
                    previousTopTwoMargin: margin,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
                round.finalizedSeedIDs.insert(id)
            }
        }

        // The ordinary finalize loop has confirmed each finished prompt's
        // first token; its last prompt hidden row is that token's carry.
        // Prefill carries never run under native completion tracking.
        if !round.prefillCarries.isEmpty {
            let prefillTopTwo = round.prefillCarryTopTwoValues?.asArray(Float.self)
            if prefillTopTwo != nil { CBv2CoreInstrumentation.recordHostSync() }
            for (index, carry) in round.prefillCarries.enumerated() {
                guard !step.discard.contains(carry.id),
                    let rec = scheduler.record(for: carry.id), rec.pendingSamples == 0
                else { continue }
                mtp.storeCarry(
                    id: carry.id, token: rec.tokens.last!, hidden: carry.hidden,
                    previousTopTwoMargin: prefillTopTwo.map {
                        Double($0[index * 2] - $0[index * 2 + 1])
                    },
                    tokensCount: rec.tokens.count, kvOffset: rec.numComputedTokens)
            }
        }

        guard let verify = round.verify else { return }
        let k = verify.k
        // Host readbacks of the MTP round, each counted: an MTP-round
        // finalize adds up to three syncs to the step's one (seed policy
        // margin above, acceptance packet, verify policy margin). Serial
        // target verification adds one blocking eval per column at launch
        // (`EngineLoopV2+MTPTargetVerification`); a logprob segment adds
        // three readbacks (`CBv2Logprobs.assemble`); a round whose capture
        // could not be fenced adds one blocking eval (`CBv2MTPCaptureFence`
        // fallback in `EngineLoopV2+MTPExecution`).
        if nativeCommitHeld {
            nativeShutdownState?.endCommit()
            nativeCommitHeld = false
        }
        var host: [Int32] = []
        var policyTopTwoHost: [Float]?
        let acceptanceStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        defer {
            if CBv2StepProfiler.enabled {
                CBv2StepProfiler.record(
                    "v2.mtp.finalize.after.acceptance",
                    seconds: CFAbsoluteTimeGetCurrent() - acceptanceStart)
            }
        }
        guard
            completedNativeReadback({
                host = verify.acceptancePacket.asArray(Int32.self)
                CBv2CoreInstrumentation.recordHostSync()
                policyTopTwoHost = verify.policyTopTwoValues?.asArray(Float.self)
            })
        else { return }
        engineWorkInterval.hostWorkBegan()
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.acceptance.readback", seconds: CFAbsoluteTimeGetCurrent() - acceptanceStart)
        }
        if policyTopTwoHost != nil { CBv2CoreInstrumentation.recordHostSync() }
        if let tracking = nativeShutdownState {
            guard tracking.beginCommit() else { return }
            nativeCommitHeld = true
        }
        let draftCount = verify.rows.count * k
        let targetWidth = 1 + k
        // Typical acceptance appends its [B, k] mask after every other part.
        let acceptBase: Int? =
            verify.hasAcceptMask
            ? draftCount + verify.rows.count * targetWidth
                + (verify.shortlistIDs != nil ? verify.rows.count * targetWidth : 0)
            : nil
        var anyRejected = false

        struct RowOutcome {
            let batchIndex: Int
            let metadata: CBv2MTPRoundInFlight.VerifyRow
            let rec: CBv2ScheduledRequest
            /// The target's own token at every window position.
            var targets: [Int]
            /// What the row commits: the accepted drafts as proposed, then
            /// the target token at the first rejection or bonus position.
            /// Equals `targets` position-for-position under exact acceptance.
            var committed: [Int]
            /// Verify slot indices `[0, ...]` corresponding to `committed`.
            var path: [Int]
            var accepted: Int
            /// Accepted drafts plus one, cut at the length bound and a stop token.
            var naturalEmitted: Int
            var branchHit = false
            var constraintFailure: String? = nil
            var nonFiniteAfterPrefix = false
        }

        let verifyTree: CBv2VerifyTree? =
            verify.hasDynamicTree && !host.isEmpty
            ? CBv2VerifyTree.fromPackedParents(UInt32(bitPattern: host[host.count - 1]), k: k)
            : verify.verifyTree

        // Resolve each row's natural target-authoritative prefix, then choose
        // one committed width for the rectangular step, unless the drafter's
        // target lets every row commit its own prefix.
        let perRowWidth = mtp.drafter.commitsAcceptedPrefixPerRow
        var outcomes: [RowOutcome] = []
        outcomes.reserveCapacity(verify.rows.count)
        var commonEmitted = targetWidth

        func releaseDiscarded(_ metadata: CBv2MTPRoundInFlight.VerifyRow) {
            let id = metadata.id
            if let evaluations = verify.recurrentEvaluations[id] {
                do {
                    for evaluation in evaluations.reversed() { try evaluation.rollback() }
                } catch {
                    preconditionFailure(
                        "CBv2 recurrent MTP discard rollback failed for \(id): \(error)")
                }
            }
            if let state = metadata.assistantState {
                round.deferredAssistantReleases.append(state)
            }
            anyRejected = true
        }

        for (batchIndex, metadata) in verify.rows.enumerated() {
            let id = metadata.id
            if step.discard.contains(id) || scheduler.record(for: id) == nil {
                releaseDiscarded(metadata)
                continue
            }
            let rec = scheduler.record(for: id)!
            if metadata.freeSpan != nil, let failure = sampler.tokenConstraintFailure(for: id) {
                step.discard.insert(id)
                releaseDiscarded(metadata)
                finishRequest(id, reason: .error(failure))
                continue
            }
            let drafts = (0 ..< k).map { Int(host[batchIndex * k + $0]) }
            let targets = (0 ..< targetWidth).map {
                Int(host[draftCount + batchIndex * targetWidth + $0])
            }

            // Grammar verification owns constrained rows outside an opted-in
            // free span; free-span rows keep their pure same-state previews.
            if rec.request.tokenConstraint != nil, metadata.freeSpan == nil {
                let limit = min(targetWidth, rec.request.maxTokens - rec.generatedTokenCount)
                commonEmitted = min(commonEmitted, limit)
                outcomes.append(
                    RowOutcome(
                        batchIndex: batchIndex, metadata: metadata, rec: rec,
                        targets: targets, committed: [], path: Array(0 ..< targetWidth),
                        accepted: 0, naturalEmitted: limit))
                continue
            }

            var accepted = 0
            var path: [Int] = []
            var committed: [Int] = []
            var branchHit = false
            if let tree = verifyTree, metadata.freeSpan == nil, acceptBase == nil {
                var cur = 0
                path.append(0)
                while true {
                    let want = targets[cur]
                    committed.append(want)
                    if let child = tree.children[cur].first(where: { drafts[$0 - 1] == want }) {
                        if child != cur + 1 { branchHit = true }
                        path.append(child)
                        cur = child
                    } else {
                        break
                    }
                }
                accepted = path.count - 1
            } else {
                if let span = metadata.freeSpan {
                    while accepted < k, targets[accepted] == drafts[accepted],
                        span.continues(with: drafts[accepted])
                    {
                        accepted += 1
                    }
                } else if let acceptBase {
                    while accepted < k, host[acceptBase + batchIndex * k + accepted] != 0 {
                        accepted += 1
                    }
                } else {
                    while accepted < k, targets[accepted] == drafts[accepted] { accepted += 1 }
                }
                committed = (0 ..< targetWidth).map { $0 < accepted ? drafts[$0] : targets[$0] }
                path = Array(0 ..< targetWidth)
            }
            var naturalEmitted = accepted + 1
            naturalEmitted = min(
                naturalEmitted,
                rec.request.maxTokens - rec.generatedTokenCount)
            if let stopIndex = committed[..<naturalEmitted].firstIndex(where: {
                rec.request.stopTokens.contains($0)
            }) {
                naturalEmitted = stopIndex + 1
            }
            // A position this row would commit was decided by non-finite
            // target logits: commit nothing and fail the request exactly like
            // any other finalize-time verify failure.
            if path.prefix(naturalEmitted).contains(where: {
                targets[$0] == Int(CBv2NonFiniteLogits.invalidTarget)
            }) {
                nonFiniteVerifyFailures += 1
                step.discard.insert(id)
                releaseDiscarded(metadata)
                finishRequest(id, reason: CBv2NonFiniteLogits.finishReason)
                continue
            }
            commonEmitted = min(commonEmitted, naturalEmitted)
            outcomes.append(
                RowOutcome(
                    batchIndex: batchIndex,
                    metadata: metadata,
                    rec: rec,
                    targets: targets,
                    committed: committed,
                    path: path,
                    accepted: accepted,
                    naturalEmitted: naturalEmitted,
                    branchHit: branchHit))
        }

        if let logits = verify.constraintLogits {
            let grammarRows = outcomes.filter {
                $0.rec.request.tokenConstraint != nil && $0.metadata.freeSpan == nil
            }
            let prefixes = mtpSampleAndConfirmConstrainedPrefixes(
                logits: logits,
                rows: grammarRows.map {
                    CBv2MTPGrammarRow(
                        batchIndex: $0.batchIndex, context: Self.samplerRow($0.rec),
                        stopTokens: $0.rec.request.stopTokens,
                        nonFinite: $0.targets.map { $0 == Int(CBv2NonFiniteLogits.invalidTarget) })
                },
                draftPacket: Array(host.prefix(draftCount)), depth: k,
                commonLimit: commonEmitted, independentWidths: perRowWidth)
            for index in outcomes.indices.reversed() {
                guard let prefix = prefixes[outcomes[index].metadata.id] else { continue }
                if prefix.nonFinite {
                    nonFiniteVerifyFailures += 1
                    if prefix.tokens.isEmpty {
                        let outcome = outcomes.remove(at: index)
                        step.discard.insert(outcome.metadata.id)
                        releaseDiscarded(outcome.metadata)
                        finishRequest(outcome.metadata.id, reason: CBv2NonFiniteLogits.finishReason)
                        continue
                    }
                    outcomes[index].nonFiniteAfterPrefix = true
                }
                outcomes[index].committed = prefix.tokens
                outcomes[index].targets = prefix.tokens
                outcomes[index].accepted = prefix.accepted
                outcomes[index].naturalEmitted = prefix.tokens.count
                outcomes[index].constraintFailure = prefix.failure
            }
            commonEmitted = outcomes.map(\.naturalEmitted).min() ?? 0
        }

        round.finalizedVerifyIDs = Set(outcomes.map { $0.metadata.id })
        round.claimedSeedCostNanos = mtp.claimPendingSeedCost(
            decodeRowBucket: mtp.planDecodeRowBucket,
            finalizedVerifyIDs: round.finalizedVerifyIDs,
            measurement: step.mtpMeasurement)

        if perRowWidth {
            // Each row commits its own prefix, so the controller learns every
            // row's acceptance rather than the step's minimum.
            for outcome in outcomes {
                let accepted = min(outcome.accepted, outcome.naturalEmitted)
                mtp.recordStepAcceptance(
                    drafted: k, accepted: accepted,
                    observedDrafts: outcome.naturalEmitted <= accepted
                        ? outcome.naturalEmitted : min(k, accepted + 1),
                    decodeRowBucket: mtp.planDecodeRowBucket,
                    measurement: step.mtpMeasurement)
            }
        } else if !outcomes.isEmpty {
            let stepAccepted = outcomes.map { min($0.accepted, commonEmitted) }.min() ?? 0
            let observedDrafts =
                commonEmitted <= stepAccepted
                ? commonEmitted : min(k, stepAccepted + 1)
            mtp.recordStepAcceptance(
                drafted: k,
                accepted: stepAccepted,
                observedDrafts: observedDrafts,
                decodeRowBucket: mtp.planDecodeRowBucket,
                measurement: step.mtpMeasurement)
        }

        for outcome in outcomes {
            let batchIndex = outcome.batchIndex
            let metadata = outcome.metadata
            let id = metadata.id
            let rec = outcome.rec
            let accepted = outcome.accepted
            mtp.observeHistoryOutcome(for: id, committed: outcome.committed)
            let emitted = Array(
                outcome.committed.prefix(perRowWidth ? outcome.naturalEmitted : commonEmitted))

            // Confirm in order with the same stop and length semantics as the
            // ordinary finalize loop.
            let detokenizer = detokenizers[id]
            let hasStopStrings = !rec.request.stopStrings.isEmpty
            var kept: [Int] = []
            var textPieces: [String] = []
            var finishReason: CBv2FinishReason?
            for (index, token) in emitted.enumerated() {
                if metadata.freeSpan != nil {
                    sampler.confirmSampledTokens([token], requestIDs: [id])
                    if let failure = sampler.tokenConstraintFailure(for: id) {
                        finishReason = .error(failure)
                        break
                    }
                }
                scheduler.recordSampled(id: id, token: token)
                kept.append(token)
                if index == emitted.count - 1, let failure = outcome.constraintFailure {
                    finishReason = .error(failure)
                    break
                }
                if rec.request.stopTokens.contains(token) {
                    finishReason = .stop
                    break
                }
                if hasStopStrings {
                    var hostLoan: UUID?
                    if let tracking = nativeShutdownState {
                        guard
                            let loan = try? tracking.beginLoan(
                                owner: detokenizer, duringDrain: true)
                        else { return }
                        hostLoan = loan
                        if nativeCommitHeld {
                            tracking.endCommit()
                            nativeCommitHeld = false
                        }
                    }
                    let text = detokenizer?.push([token]) ?? ""
                    if let tracking = nativeShutdownState {
                        guard tracking.beginCommit() else { return }
                        nativeCommitHeld = true
                        if let hostLoan { tracking.endLoan(hostLoan) }
                    }
                    textPieces.append(text)
                    if detokenizer?.matchedStopString == true {
                        finishReason = .stop
                        break
                    }
                }
                if rec.generatedTokenCount >= rec.request.maxTokens {
                    finishReason = .length
                    break
                }
            }
            if finishReason == nil && outcome.nonFiniteAfterPrefix {
                finishReason = CBv2NonFiniteLogits.finishReason
            }

            // Correct KV and scheduler state before any terminal release.
            let confirmed = kept.count
            round.committedVerifyTokenCount +=
                kept.filter {
                    !rec.request.stopTokens.contains($0)
                }.count
            for packet in verify.diagnostics where packet.requestID == id {
                let drafts = (0 ..< k).map { Int(host[batchIndex * k + $0]) }
                packet.reconcile(
                    accepted: accepted, confirmed: confirmed, drafts: drafts,
                    targets: outcome.targets)
            }
            let rejected = (1 + k) - confirmed
            if let evaluations = verify.recurrentEvaluations[id] {
                if evaluations.count == 1, evaluations[0].isCaptured {
                    // Capture-verify: one transaction spans the window. The
                    // accepted prefix commits by device-side selection of the
                    // captured state at position `confirmed`; zero confirmed
                    // tokens restore the pre-verify snapshot by rollback.
                    do {
                        if confirmed > 0 {
                            try evaluations[0].commit(keepPositions: confirmed)
                        } else {
                            try evaluations[0].rollback()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 captured MTP finalization failed for \(id): \(error)")
                    }
                } else {
                    precondition(
                        evaluations.count == 1 + k,
                        "CBv2 recurrent MTP verification generation count mismatch")
                    do {
                        for evaluation in evaluations.suffix(rejected).reversed() {
                            try evaluation.rollback()
                        }
                        for evaluation in evaluations.prefix(confirmed) {
                            try evaluation.commit()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP finalization failed for \(id): \(error)")
                    }
                }
            }
            if outcome.branchHit && confirmed >= 2 {
                var touchedGroups: [ObjectIdentifier: PagedKVGroup] = [:]
                GPU.withCommandBufferLimits(GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)) {
                    for dstIdx in 1 ..< confirmed {
                        let srcSlot = outcome.path[dstIdx]
                        guard srcSlot != dstIdx else { continue }
                        var byGroup: [ObjectIdentifier: (group: PagedKVGroup, pairs: [(Int32, Int32)])] = [:]
                        for sequence in metadata.storageRows {
                            guard let paged = sequence as? PagedSequenceKV else { continue }
                            let treeBase = paged.absoluteOffset - targetWidth
                            let srcPos = treeBase + srcSlot
                            let dstPos = treeBase + dstIdx
                            if let direct = paged.resolveDirectCopySlots(from: srcPos, to: dstPos) {
                                let gkey = ObjectIdentifier(direct.group)
                                touchedGroups[gkey] = direct.group
                                if byGroup[gkey] != nil {
                                    byGroup[gkey]!.pairs.append((direct.src, direct.dst))
                                } else {
                                    byGroup[gkey] = (direct.group, [(direct.src, direct.dst)])
                                }
                            } else if let _ = paged.copyTokenSlot(from: srcPos, to: dstPos) {
                                let g = paged.kvGroupForCopy
                                touchedGroups[ObjectIdentifier(g)] = g
                            }
                        }
                        for entry in byGroup.values {
                            PagedSegmentTransfers.copySlotsInPlace(
                                group: entry.group, pairs: entry.pairs)
                        }
                    }
                    let copyFences = touchedGroups.values.map(\.writeFence)
                    if !copyFences.isEmpty {
                        if nativeShutdownState == nil && finishReason == nil {
                            asyncEval(copyFences)
                        } else {
                            eval(copyFences)
                        }
                    }
                }
            }
            if rejected > 0 {
                for sequence in metadata.storageRows { sequence.rollback(rejected) }
                anyRejected = true
            }
            for sequence in metadata.storageRows { sequence.commitSpeculativeWrite() }
            let committedDraftCount = min(accepted, max(0, confirmed - 1))
            if confirmed > 0, let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter,
                let state = metadata.assistantState
            {
                // Target KV/recurrent truth is committed first. The assistant
                // receives only accepted draft inputs and their trusted target
                // hidden rows; every speculative head suffix is discarded.
                let committedDraftsArray: MLXArray =
                    outcome.branchHit
                    ? MLXArray(kept.prefix(committedDraftCount).map { Int32($0) })
                        .reshaped([1, committedDraftCount])
                    : verify.draftIDs[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount]
                let committedHidden: MLXArray
                if outcome.branchHit, committedDraftCount > 0,
                    !outcome.path.prefix(committedDraftCount).elementsEqual(0 ..< committedDraftCount)
                {
                    let rowIdx = MLXArray(
                        outcome.path.prefix(committedDraftCount).map { Int32($0) })
                    committedHidden = verify.lastHidden[
                        batchIndex ..< batchIndex + 1, 0..., 0...
                    ].take(rowIdx, axis: 1)
                } else {
                    committedHidden = verify.lastHidden[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount,
                        0...]
                }
                stateful.finalizeRound(
                    requestState: state,
                    confirmedInputTokens: 1 + committedDraftCount,
                    committedDraftTokens: committedDraftsArray,
                    committedTargetHidden: committedHidden)
            }
            if rejected > 0 {
                scheduler.discardPendingSamples(id: id, count: rejected)
                scheduler.rollbackComputed(id: id, tokens: rejected)
            }
            // The speculative suffix is now reconciled in BOTH the page tables
            // and scheduler record. Publish only the accepted frontier; doing
            // this before rollback would make a rejected block cache-visible.
            let launchedEnd = step.computedRanges[id]?.upperBound ?? rec.numComputedTokens
            publishFinalizedResidentBlocks(
                requestID: id,
                safeComputedEnd: min(launchedEnd, rec.numComputedTokens))

            if hasStopStrings {
                stream(for: id)?.emit(
                    .delta(text: textPieces.joined(), tokens: kept, logprobs: nil))
            } else {
                let stream = stream(for: id)
                stream?.reserveEmission()
                let endsWithStopToken = finishReason == .stop
                let visibleTokens = outcome.constraintFailure == nil ? kept : Array(kept.dropLast())
                let pushTokens = endsWithStopToken ? Array(visibleTokens.dropLast()) : visibleTokens
                let allTokens = visibleTokens
                let trackedLoop = nativeShutdownState == nil ? nil : self
                enqueueDetokenization(owner: detokenizer) {
                    let text = pushTokens.isEmpty ? "" : (detokenizer?.push(pushTokens) ?? "")
                    let publish: () -> Void = {
                        stream?.emit(
                            .delta(text: text, tokens: allTokens, logprobs: nil),
                            consumingReservation: true)
                    }
                    if let trackedLoop { _ = trackedLoop.nativeCommit(publish) } else { publish() }
                }
            }

            let observedAccepted = min(accepted, confirmed)
            // Per-request timing: this verify row confirmed at the step's
            // readback-done instant (already read by `finalize`).
            rec.recordStepParticipation(step: step, batchRows: step.tokenProducingRows)
            rec.recordMTPRound(drafted: k, accepted: observedAccepted)
            if confirmed > 0 {
                rec.stampTokenConfirmation(readbackDoneNanos: step.readbackDoneNanos)
                step.forwardShapes?.confirmTokens(
                    row: ObjectIdentifier(rec), firstToken: false,
                    count: confirmed, nanos: step.readbackDoneNanos)
                rec.timing.decodeSteps &+= 1
                decodeRowsTotal = Self.saturatingAdd(decodeRowsTotal, 1)
            }
            // engine-local: per-round audit record (benchmark free-run audit)
            mtp.recordRound(
                drafted: k, accepted: observedAccepted, emitted: confirmed,
                audit: CBv2MTPRoundAuditRecord(
                    requestID: id.raw, drafted: k,
                    accepted: accepted, committed: confirmed))
            let useOldPol = CBv2DraftRegime.treeOldAcceptancePolicy
            let terminalBranchHit =
                !useOldPol && verifyTree != nil && outcome.branchHit
                && outcome.path.count > accepted
                && verifyTree!.children[outcome.path[accepted]].isEmpty
            let effectiveDraftedDepth: Int
            let rejectionObserved: Bool
            let acceptanceTruncated: Bool
            if useOldPol {
                effectiveDraftedDepth = k
                rejectionObserved = accepted < k && confirmed > accepted
                acceptanceTruncated =
                    !rejectionObserved && confirmed <= accepted && confirmed < k
            } else if terminalBranchHit {
                effectiveDraftedDepth = accepted
                rejectionObserved = false
                acceptanceTruncated = true
            } else {
                effectiveDraftedDepth = verifyTree != nil ? verifyTree!.chainDepth : k
                rejectionObserved = accepted < effectiveDraftedDepth && confirmed > accepted
                acceptanceTruncated =
                    !rejectionObserved && confirmed <= accepted && confirmed < effectiveDraftedDepth
            }
            mtp.observeRequestAcceptance(
                id: id,
                draftedDepth: effectiveDraftedDepth,
                acceptedDepth: observedAccepted,
                committedTokens: confirmed,
                rejectionObserved: rejectionObserved,
                endedByTruncation: acceptanceTruncated)
            mtp.updateTreeCooldown(
                for: id, wasTree: verifyTree != nil,
                chainDepth: verifyTree?.chainDepth ?? k,
                k: k,
                accepted: observedAccepted, branchHit: outcome.branchHit)

            if let finishReason {
                if let state = metadata.assistantState {
                    mtp.releaseDetachedAssistantState(state)
                }
                finishRequest(id, reason: finishReason)
            } else {
                if let state = metadata.assistantState {
                    mtp.restoreAssistantState(state, for: id)
                }
                // No inline deadline check: an MTP row that just confirmed
                // tokens is making progress and its decode lease is refreshed
                // in `refreshProgressLeases` (run at the end of the enclosing
                // `finalize`, after this round). Lease expiry is evaluated
                // centrally in `processLeaseExpiry` — identical typed-terminal
                // semantics to the ordinary decode path.
                let hiddenColumn =
                    verifyTree != nil
                    ? outcome.path[confirmed - 1]
                    : CBv2MTPHiddenIndex.carryColumn(
                        targetOutputIndex: confirmed - 1, draftDepth: k)
                // Shortlist coverage gate: hand the accepted position's
                // target top-K ids to the next draft only when their
                // probability mass (packet tail, parts-per-million) clears
                // the threshold; otherwise the draft falls back to the full
                // head for that round.
                var carryShortlist: MLXArray?
                if let shortlistIDs = verify.shortlistIDs {
                    let massBase = draftCount + verify.rows.count * targetWidth
                    let mass = host[massBase + batchIndex * targetWidth + hiddenColumn]
                    if mass >= Self.mtpShortlistMassThresholdPPM {
                        carryShortlist = shortlistIDs[batchIndex, hiddenColumn]
                    }
                }
                let previousTopTwoMargin: Double? = policyTopTwoHost.map { values in
                    let base =
                        (batchIndex * targetWidth + hiddenColumn) * 2
                    return Double(values[base] - values[base + 1])
                }
                let carryHidden = verify.lastHidden[
                    batchIndex ..< (batchIndex + 1),
                    hiddenColumn ..< (hiddenColumn + 1), 0...]
                mtp.storeCarry(
                    id: id, token: kept[confirmed - 1],
                    hidden: carryHidden,
                    shortlist: carryShortlist,
                    previousTopTwoMargin: previousTopTwoMargin,
                    needsHistoryTransition: mtp.tracksPersistentHistory,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
                // The carry is verify row `accepted` after `accepted` committed
                // drafts: an outcome a post-verify speculation covers. The
                // drafter keeps it only for this carry object; its all-accepted
                // block also needs the carry token to be the verify's own
                // bonus target.
                if let speculator = mtp.drafter as? any CBv2MTPPostVerifySpeculating,
                    let state = metadata.assistantState
                {
                    let branchSpecCovered = outcome.branchHit && verify.hasDynamicTree
                    if (!outcome.branchHit || branchSpecCovered),
                        confirmed == accepted + 1, committedDraftCount == accepted,
                        outcome.path.count >= confirmed,
                        (outcome.branchHit || hiddenColumn == accepted), accepted <= k
                    {
                        speculator.confirmSpeculation(
                            carryHidden: carryHidden, acceptedDrafts: accepted,
                            acceptedPath: Array(outcome.path.prefix(confirmed)),
                            carryTokenIsTarget: kept[accepted] == outcome.targets[hiddenColumn],
                            requestState: state)
                    } else {
                        speculator.dropSpeculation(requestState: state)
                    }
                }
            }
        }

        // Verify rows emitted tokens without passing through the sampler;
        // drop their configured row state so the next ordinary sample
        // reconfigures from confirmed history (exact penalty counts and
        // per-request RNG step indices).
        if !outcomes.isEmpty {
            sampler.mtpRoundDidCommit(requestIDs: outcomes.map { $0.metadata.id })
        }

        // Rejected suffixes advanced eager device offsets past host truth.
        if anyRejected {
            eagerCompositionStale = true
        }
    }
}

// MARK: - Post-verify speculation

/// A request-stateful drafter that can build the next round's draft index 0
/// work behind a submitted verify without reading the acceptance count: work
/// every outcome shares (the context commit of the verify's rows), and work
/// for the outcome in which every draft is accepted.
///
/// Contract: building writes no request state the engine or any other call
/// observes (context, pending rows, carries); it only holds the speculative
/// arrays. The drafter hands them out at the next round's draft index 0 only
/// when its state is exactly the one the normal path would then have, with
/// every handed-out value equal to what the normal path computes; any other
/// call on the state discards them first.
public protocol CBv2MTPPostVerifySpeculating: CBv2MTPRequestStatefulDrafter {
    /// Right after a one-row round's verify is submitted. `targetTokens`
    /// `[1, depth + 1]` are the verify's per-position target tokens and
    /// `targetHidden` `[1, depth + 1, H]` its hidden rows, both lazy.
    /// `draftsBlock`: the all-accepted outcome's next block may be drafted
    /// too (work every outcome shares is always eligible).
    /// Returns the arrays to submit; empty when it does not speculate.
    /// `draftTokens` `[1, depth]`, when given: draft the next block for the
    /// outcome this verify decides (accept count read on the device).
    func speculateNextRound(
        targetTokens: MLXArray, targetHidden: MLXArray, depth: Int, draftsBlock: Bool,
        draftTokens: MLXArray?, requestState: any CBv2MTPRequestState) -> [MLXArray]
    func speculateNextRound(
        targetTokens: MLXArray, targetHidden: MLXArray, depth: Int, draftsBlock: Bool,
        draftTokens: MLXArray?, dynTreeMeta: MLXArray?,
        requestState: any CBv2MTPRequestState) -> [MLXArray]
    /// Finalize committed `acceptedDrafts` drafts of the round and stored
    /// `carryHidden` (this object, verify row `acceptedDrafts`) as the row's
    /// carry; `carryTokenIsTarget` when its token is the verify's own target
    /// at that row.
    func confirmSpeculation(
        carryHidden: MLXArray, acceptedDrafts: Int, carryTokenIsTarget: Bool,
        requestState: any CBv2MTPRequestState)
    func confirmSpeculation(
        carryHidden: MLXArray, acceptedDrafts: Int, acceptedPath: [Int], carryTokenIsTarget: Bool,
        requestState: any CBv2MTPRequestState)
    /// Forget any speculation (its submission failed).
    func dropSpeculation(requestState: any CBv2MTPRequestState)
    /// `[n]` float32, lazy: the confirmed speculation's per-row top-1
    /// probability of each proposal (row `i` drafts position `i + 1`), or nil
    /// without a confirmed speculation. Advisory: it may size the next
    /// verify window and never changes a proposal or a committed token.
    func speculatedDraftConfidence(requestState: any CBv2MTPRequestState) -> MLXArray?
}

extension CBv2MTPPostVerifySpeculating {
    public func speculateNextRound(
        targetTokens: MLXArray, targetHidden: MLXArray, depth: Int, draftsBlock: Bool,
        draftTokens: MLXArray?, dynTreeMeta: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        speculateNextRound(
            targetTokens: targetTokens, targetHidden: targetHidden, depth: depth,
            draftsBlock: draftsBlock, draftTokens: draftTokens, requestState: requestState)
    }

    public func confirmSpeculation(
        carryHidden: MLXArray, acceptedDrafts: Int, acceptedPath: [Int], carryTokenIsTarget: Bool,
        requestState: any CBv2MTPRequestState
    ) {
        if acceptedPath.elementsEqual(0 ... acceptedDrafts) {
            confirmSpeculation(
                carryHidden: carryHidden, acceptedDrafts: acceptedDrafts,
                carryTokenIsTarget: carryTokenIsTarget, requestState: requestState)
        } else {
            dropSpeculation(requestState: requestState)
        }
    }

    public func speculatedDraftConfidence(requestState: any CBv2MTPRequestState) -> MLXArray? {
        nil
    }
}

extension EngineLoopV2 {
    /// The fewest predicted probability that a round accepts every draft
    /// (the acceptance window's per-position estimates, multiplied) for the
    /// all-accepted next block to be drafted behind its verify. A wrong
    /// block occupies the device past the host gap and delays the next
    /// round; the shared commit work is always submitted. 0 disables the
    /// gate.
    static var mtpSpeculativeBlockMinimumFullAcceptance: Double { 0.5 }
    /// Device-outcome post-verify drafting on deep-low-acceptance rounds: the next
    /// block is drafted behind the verify for whichever accept count the
    /// verify decides, so the device goes from the verify straight into the
    /// next round while the host reads the packet and finalizes.
    /// `CBV2_MTP_OUTCOME_SPEC=0` turns it off. It needs the drafter's ring
    /// fence to stay on the round's own writes (`FastInPlaceContext.arrays`):
    /// with a fence that follows the speculative writes the host waits for
    /// the whole block before it reads the packet, and the round is longer.
    static let mtpOutcomeSpeculation =
        ProcessInfo.processInfo.environment["CBV2_MTP_OUTCOME_SPEC"] != "0"

    /// Submit the drafter's next-round work behind a verify.
    /// One-row rounds only (a batched round's drafter pass has other
    /// shapes); no grammar, shortlist or typical-acceptance rounds, whose
    /// carry token or drafter inputs are not the verify's own target row.
    /// Never under native completion tracking, whose root accounting this
    /// optional work would join.
    func mtpSpeculateAfterVerify(
        _ verify: CBv2MTPRoundInFlight.Verify, driver mtp: CBv2MTPRoundDriver
    ) {
        guard nativeShutdownState == nil, verify.rows.count == 1, verify.k > 0,
            !verify.hasAcceptMask, verify.verifyTree == nil,
            verify.shortlistIDs == nil, verify.constraintLogits == nil,
            let speculator = mtp.drafter as? any CBv2MTPPostVerifySpeculating,
            let state = verify.rows[0].assistantState
        else { return }
        let k = verify.k
        let id = verify.rows[0].id
        // Packet: [1, k] drafts, then [1, k + 1] targets.
        let targets = verify.acceptancePacket[k ..< (2 * k + 1)].reshaped([1, k + 1])
        let draftsBlock =
            mtp.predictedFullAcceptance(for: id, depth: k).map {
                $0 >= Self.mtpSpeculativeBlockMinimumFullAcceptance
            } ?? true
        // A round that starts with one token of the request's budget left ends
        // the request, and with two it usually does: a block drafted behind
        // such a verify would be device work queued behind the last tokens.
        let budgetLeft =
            scheduler.record(for: id).map { $0.request.maxTokens - $0.generatedTokenCount } ?? 0
        // Deep-low rounds only: the regime whose drafter forms the outcome block
        // was built for, and one a high-acceptance request does not enter, so
        // its rounds keep the proposals and the timing they have without it.
        let deepLow = mtp.inDeepLowAcceptanceRegime(id)
        let draftTokens: MLXArray? =
            Self.mtpOutcomeSpeculation && budgetLeft > 2 && deepLow
            ? verify.acceptancePacket[0 ..< k].reshaped([1, k]) : nil
        CBv2DraftRegime.xnorDraftHeadEligible = deepLow
        defer { CBv2DraftRegime.xnorDraftHeadEligible = false }
        let roots = speculator.speculateNextRound(
            targetTokens: targets, targetHidden: verify.lastHidden, depth: k,
            draftsBlock: draftsBlock, draftTokens: draftTokens,
            dynTreeMeta: verify.dynTreeMeta, requestState: state)
        guard !roots.isEmpty else { return }
        do {
            try withError { fault in
                asyncEval(roots)
                try fault.check()
            }
        } catch {
            speculator.dropSpeculation(requestState: state)
        }
    }
}
