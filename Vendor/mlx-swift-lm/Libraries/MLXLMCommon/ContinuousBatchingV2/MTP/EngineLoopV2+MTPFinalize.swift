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
            var accepted: Int
            /// Accepted drafts plus one, cut at the length bound and a stop token.
            var naturalEmitted: Int
            var constraintFailure: String? = nil
            var nonFiniteAfterPrefix = false
            /// Draft tree: the committed leaf's verify column (the round
            /// commits `accepted` drafts, the leaf token and the target there).
            var treeLeafColumn: Int? = nil
            /// Draft tree: that leaf's K/V move, planned before anything commits.
            var treeCopy: [(group: PagedKVGroup, plan: [(PagedKVSegment, [Int32])])]? = nil
        }

        // Draft tree (CBV2_VERIFY_TREE): the chain is the window's first
        // `chain` drafts; leaves commit only from a valid forward, after an
        // exactness probe has passed, and never past a failed one.
        // Same-width round: its layout, chosen on device, read back with the
        // packet (meta [4], rowpos [k + 1]). A layout this design cannot verify
        // exactly commits its leading chain columns only.
        var roundTree = verify.tree
        if var tree = roundTree, tree.sameWidth, let base = tree.sameWidthBase,
            base + 4 + k + 1 <= host.count
        {
            let chain = Int(host[base])
            let leaves = Int(host[base + 1])
            let probe = host[base + 2] != 0
            let rowPositions = Array(host[(base + 4) ..< (base + 4 + k + 1)])
            if CBv2SameWidthTree.layoutIsValid(
                rowPositions: rowPositions, chain: chain, leaves: leaves, drafts: k)
            {
                tree.chain = chain
                if leaves > 0 {
                    tree.layout = CBv2VerifyTree.Layout(
                        chain: chain,
                        leaves: (0 ..< leaves).map { j in
                            CBv2VerifyTree.Leaf(
                                row: Int(rowPositions[chain + 1 + j]) - 1,
                                rank: probe && j == leaves - 1 ? 1 : 2)
                        },
                        probe: probe ? leaves - 1 : nil)
                } else if tree.why.isEmpty {
                    tree.why = "sw_chain"
                }
            } else {
                var chain = 0
                while chain < k, rowPositions.count > chain + 1,
                    rowPositions[chain + 1] == Int32(chain + 1)
                {
                    chain += 1
                }
                tree.chain = chain
                tree.valid = false
                tree.why = "layout_invalid"
            }
            roundTree = tree
        }
        let treeChain = roundTree?.chain ?? k
        var treeHitsAllowed = false
        var treeProbe: Bool? = nil
        if let tree = roundTree, let layout = tree.layout {
            CBv2VerifyTree.counters.armedRounds += 1
            CBv2VerifyTree.counters.leaves += layout.leaves.count - (layout.probe == nil ? 0 : 1)
            if !tree.valid { CBv2VerifyTree.counters.invalidRounds += 1 }
            if tree.valid, let probe = layout.probe, let base = tree.probeBase, base < host.count {
                let column = layout.column(ofLeaf: probe)
                let chainColumn = layout.leaves[probe].row + 1
                let t = draftCount + column
                let c = draftCount + chainColumn
                let ok = host[base] == 1 && t < host.count && host[t] == host[c]
                treeProbe = ok
                CBv2VerifyTree.probesRun += 1
                CBv2VerifyTree.counters.probeRounds += 1
                if ok {
                    CBv2VerifyTree.probePassed = true
                } else {
                    CBv2VerifyTree.probeFailed = true
                    CBv2VerifyTree.counters.probeFailures += 1
                    FileHandle.standardError.write(
                        Data(
                            ("[cbv2-tree] PROBE FAILED: leaf column \(column) differs from chain "
                                + "column \(chainColumn) (hidden bits equal: \(host[base] == 1)); "
                                + "leaves off for this process\n").utf8))
                }
            }
            treeHitsAllowed =
                tree.valid && !CBv2VerifyTree.probeFailed
                && (CBv2VerifyTree.probePassed || CBv2VerifyTree.probeRounds == 0)
        }

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
                        targets: targets, committed: [], accepted: 0, naturalEmitted: limit))
                continue
            }

            var accepted = 0
            if let span = metadata.freeSpan {
                while accepted < treeChain, targets[accepted] == drafts[accepted],
                    span.continues(with: drafts[accepted])
                {
                    accepted += 1
                }
            } else if let acceptBase {
                while accepted < treeChain, host[acceptBase + batchIndex * k + accepted] != 0 {
                    accepted += 1
                }
            } else {
                while accepted < treeChain, targets[accepted] == drafts[accepted] { accepted += 1 }
            }
            var committed = (0 ..< targetWidth).map { $0 < accepted ? drafts[$0] : targets[$0] }
            var naturalEmitted = accepted + 1
            // Draft tree: a leaf at the first rejected row that carries the
            // target's token there commits, with the target at its column.
            var treeLeafColumn: Int? = nil
            var treeCopy: [(group: PagedKVGroup, plan: [(PagedKVSegment, [Int32])])]? = nil
            // (A leaf at row `chain` is the alternative after a fully
            // accepted chain; a rank >= 2 token equal to the target there
            // implies the rank-1 draft would have been rejected.)
            if treeHitsAllowed, let layout = roundTree?.layout, metadata.freeSpan == nil,
                acceptBase == nil, accepted <= layout.chain
            {
                for (index, leaf) in layout.leaves.enumerated()
                where index != layout.probe && leaf.row == accepted {
                    let column = layout.column(ofLeaf: index)
                    guard drafts[column - 1] == targets[accepted],
                        targets[column] != Int(CBv2NonFiniteLogits.invalidTarget)
                    else { continue }
                    let paged = metadata.storageRows.compactMap { $0 as? PagedSequenceKV }
                    let seed = (paged.first?.absoluteOffset ?? 0) - targetWidth
                    if paged.count == metadata.storageRows.count,
                        paged.allSatisfy({ $0.absoluteOffset - targetWidth == seed }),
                        let plan = PagedSequenceKV.treeLeafCopyPlan(
                            rows: paged, source: seed + column, destination: seed + accepted + 1)
                    {
                        treeLeafColumn = column
                        treeCopy = plan
                        committed = Array(drafts[0 ..< accepted]) + [drafts[column - 1], targets[column]]
                        naturalEmitted = accepted + 2
                    } else {
                        CBv2VerifyTree.counters.notArmed["copy_unavailable", default: 0] += 1
                    }
                    break
                }
            }
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
            // A committed leaf: the positions it commits are targets 0 ... a
            // and the target at its own column (not chain column a + 1).
            let decisiveTargets =
                treeLeafColumn.map { column in
                    Array(targets[0 ... accepted])
                        + (naturalEmitted > accepted + 1 ? [targets[column]] : [])
                } ?? Array(targets[..<naturalEmitted])
            if decisiveTargets.contains(Int(CBv2NonFiniteLogits.invalidTarget)) {
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
                    accepted: accepted,
                    naturalEmitted: naturalEmitted,
                    treeLeafColumn: treeLeafColumn,
                    treeCopy: treeCopy))
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
                    drafted: treeChain, accepted: accepted,
                    observedDrafts: outcome.naturalEmitted <= accepted
                        ? outcome.naturalEmitted : min(treeChain, accepted + 1),
                    decodeRowBucket: mtp.planDecodeRowBucket,
                    measurement: step.mtpMeasurement)
            }
        } else if !outcomes.isEmpty {
            let stepAccepted = outcomes.map { min($0.accepted, commonEmitted) }.min() ?? 0
            let observedDrafts =
                commonEmitted <= stepAccepted
                ? commonEmitted : min(treeChain, stepAccepted + 1)
            mtp.recordStepAcceptance(
                drafted: treeChain,
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
            let emitted = Array(
                outcome.committed.prefix(perRowWidth ? outcome.naturalEmitted : commonEmitted))
            // CBV2_TREE_TRACE (measurement only): the plan's inputs and the
            // drafter's block, read before this row commits anything.
            let traceHead =
                CBv2TreeTrace.enabled && nativeShutdownState == nil
                ? mtpTraceRoundHead(rec: rec, assistantState: metadata.assistantState, driver: mtp)
                : nil

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
            // Draft tree: the leaf and the target at its column both
            // committed. (A leaf cut by a stop or the length bound commits
            // only the leaf token, which equals the chain's target there: a
            // chain outcome, with nothing to move.)
            let leafColumn: Int? =
                outcome.treeLeafColumn.flatMap { confirmed == accepted + 2 ? $0 : nil }
            if leafColumn != nil, let copy = outcome.treeCopy {
                // The leaf's K/V onto its logical slot, bitwise, before the
                // rollback releases its column's slot.
                PagedSequenceKV.runTreeLeafCopy(copy)
                CBv2VerifyTree.counters.copies += 1
            }
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
            if rejected > 0 {
                for sequence in metadata.storageRows { sequence.rollback(rejected) }
                anyRejected = true
            }
            for sequence in metadata.storageRows { sequence.commitSpeculativeWrite() }
            let committedDraftCount =
                leafColumn != nil ? accepted + 1 : min(accepted, max(0, confirmed - 1))
            if confirmed > 0, let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter,
                let state = metadata.assistantState
            {
                // Target KV/recurrent truth is committed first. The assistant
                // receives only accepted draft inputs and their trusted target
                // hidden rows; every speculative head suffix is discarded.
                // A committed leaf: inputs seed, d1 ... da, leaf; the rows of
                // seed ... da are verify columns 0 ... a, the leaf's row is the
                // carry.
                if leafColumn != nil, let treeDrafter = mtp.drafter as? any CBv2MTPTreeDrafter {
                    treeDrafter.treePrepareLeafCommit(
                        stagedRows: committedDraftCount, requestState: state)
                }
                let draftTokens =
                    leafColumn.map { column in
                        concatenated(
                            [
                                verify.draftIDs[batchIndex ..< batchIndex + 1, 0 ..< accepted],
                                verify.draftIDs[
                                    batchIndex ..< batchIndex + 1, (column - 1) ..< column],
                            ], axis: 1)
                    }
                    ?? verify.draftIDs[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount]
                stateful.finalizeRound(
                    requestState: state,
                    confirmedInputTokens: 1 + committedDraftCount,
                    committedDraftTokens: draftTokens,
                    committedTargetHidden: verify.lastHidden[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount,
                        0...])
                // The 8-row tree's speculation covered chain outcomes only; a
                // same-width round's is tree-positioned and is confirmed for
                // the leaf's rows with the carry below.
                if leafColumn != nil, roundTree?.sameWidth != true,
                    let treeDrafter = mtp.drafter as? any CBv2MTPTreeDrafter
                {
                    treeDrafter.treeLeafCommitted(requestState: state)
                }
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
            // A committed leaf counts as one more accepted draft in the
            // round's records (committed = accepted + 1, accepted <= drafted).
            let recordedAccepted = leafColumn != nil ? accepted + 1 : observedAccepted
            // Per-request timing: this verify row confirmed at the step's
            // readback-done instant (already read by `finalize`).
            rec.recordStepParticipation(step: step, batchRows: step.tokenProducingRows)
            rec.recordMTPRound(drafted: k, accepted: recordedAccepted)
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
                drafted: k, accepted: recordedAccepted, emitted: confirmed,
                audit: CBv2MTPRoundAuditRecord(
                    requestID: id.raw, drafted: k,
                    accepted: leafColumn != nil ? accepted + 1 : accepted, committed: confirmed))
            // The request's acceptance history and regime observe the chain
            // only (`treeChain == k` outside an armed tree round); the
            // ceiling still counts every committed token.
            let rejectionObserved = accepted < treeChain && confirmed > accepted
            let acceptanceTruncated =
                !rejectionObserved && confirmed <= accepted && confirmed < treeChain
            if let tree = roundTree {
                // Sibling hit rates learn from every row whose drafter top-1
                // the target rejected: the chain's first rejection, or the row
                // after a fully accepted chain (its bonus target is known).
                if let base = tree.topBase, accepted < tree.topRows, accepted < outcome.targets.count {
                    let row = base + accepted * 3
                    let truth = outcome.targets[accepted]
                    if row + 2 < host.count, Int(host[row]) != truth,
                        truth != Int(CBv2NonFiniteLogits.invalidTarget)
                    {
                        mtp.treeObserveRejection(
                            id, row: accepted, rank2: Int(host[row + 1]) == truth,
                            rank3: Int(host[row + 2]) == truth)
                    }
                }
                if leafColumn != nil { CBv2VerifyTree.counters.hits += 1 }
                CBv2VerifyTree.counters.eligibleRounds += tree.topBase != nil ? 1 : 0
                let armedLeaves = tree.layout.map { $0.leaves.count - ($0.probe == nil ? 0 : 1) } ?? 0
                mtp.treeRecord(
                    id, armed: tree.layout != nil, leaves: armedLeaves, hit: leafColumn != nil,
                    invalid: tree.layout != nil && !tree.valid,
                    generated: rec.generatedTokenCount)
                if CBv2TreeTrace.enabled {
                    mtpEmitTraceTree(
                        tree, id: id, k: k, drafts: (0 ..< k).map { Int(host[batchIndex * k + $0]) },
                        accepted: accepted, confirmed: confirmed, leafColumn: leafColumn,
                        probe: treeProbe, rates: mtp.treeSiblingRates(id))
                }
            }
            if let traceHead {
                mtpEmitTraceRound(
                    traceHead, id: id, k: k,
                    drafts: (0 ..< k).map { Int(host[batchIndex * k + $0]) },
                    targets: outcome.targets, accepted: accepted, kept: kept,
                    extra: roundTree.map { "\"chain\":\($0.chain)" })
            }
            mtp.observeRequestAcceptance(
                id: id,
                draftedDepth: treeChain,
                acceptedDepth: observedAccepted,
                committedTokens: confirmed,
                rejectionObserved: rejectionObserved,
                endedByTruncation: acceptanceTruncated)

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
                // A committed leaf's own row is the carry.
                let hiddenColumn =
                    leafColumn
                    ?? CBv2MTPHiddenIndex.carryColumn(
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
                if confirmed == accepted + 1, committedDraftCount == accepted,
                    hiddenColumn == accepted, accepted <= k,
                    let speculator = mtp.drafter as? any CBv2MTPPostVerifySpeculating,
                    let state = metadata.assistantState
                {
                    speculator.confirmSpeculation(
                        carryHidden: carryHidden, acceptedDrafts: accepted,
                        carryTokenIsTarget: kept[accepted]
                            == Int(host[draftCount + batchIndex * targetWidth + accepted]),
                        requestState: state)
                } else if let leafColumn, roundTree?.sameWidth == true,
                    let treeDrafter = mtp.drafter as? any CBv2MTPTreeDrafter,
                    let state = metadata.assistantState
                {
                    // Same-width leaf commit: inputs seed, d1 ... da, leaf are
                    // verify rows 0 ... a and the leaf's column.
                    treeDrafter.treeConfirmSpeculation(
                        carryHidden: carryHidden, rows: Array(0 ... accepted) + [leafColumn],
                        requestState: state)
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
    func speculateNextRound(
        targetTokens: MLXArray, targetHidden: MLXArray, depth: Int, draftsBlock: Bool,
        requestState: any CBv2MTPRequestState) -> [MLXArray]
    /// Finalize committed `acceptedDrafts` drafts of the round and stored
    /// `carryHidden` (this object, verify row `acceptedDrafts`) as the row's
    /// carry; `carryTokenIsTarget` when its token is the verify's own target
    /// at that row.
    func confirmSpeculation(
        carryHidden: MLXArray, acceptedDrafts: Int, carryTokenIsTarget: Bool,
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
            !verify.hasAcceptMask, verify.shortlistIDs == nil, verify.constraintLogits == nil,
            let speculator = mtp.drafter as? any CBv2MTPPostVerifySpeculating,
            let state = verify.rows[0].assistantState
        else { return }
        let k = verify.k
        // Draft tree round: the drafter staged the chain only, and only its
        // outcomes are speculated (a leaf commit drops the speculation).
        let depth = verify.tree?.layout != nil ? (verify.tree?.chain ?? k) : k
        // Packet: [1, k] drafts, then [1, k + 1] targets.
        let targets = verify.acceptancePacket[k ..< (k + depth + 1)].reshaped([1, depth + 1])
        let hidden =
            depth == k ? verify.lastHidden : verify.lastHidden[0..., 0 ..< (depth + 1), 0...]
        let draftsBlock =
            mtp.predictedFullAcceptance(for: verify.rows[0].id, depth: depth).map {
                $0 >= Self.mtpSpeculativeBlockMinimumFullAcceptance
            } ?? true
        CBv2DraftRegime.treeProposals =
            verify.tree?.topBase != nil || verify.tree?.sameWidth == true
        defer { CBv2DraftRegime.treeProposals = false }
        // Same-width round: the speculation's context keys at the verify
        // columns' tree depths (any outcome then appends its own rows).
        if let rowPositions = verify.treeRowPositions,
            let treeDrafter = mtp.drafter as? any CBv2MTPTreeDrafter
        {
            treeDrafter.setSpeculationRowPositions(rowPositions, requestState: state)
        }
        let roots = speculator.speculateNextRound(
            targetTokens: targets, targetHidden: hidden, depth: depth,
            draftsBlock: draftsBlock, requestState: state)
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
