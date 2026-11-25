// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// What the drafter reads from its target. The drafter has no embedding
/// table and no output head of its own.
public protocol MuseGlimmerDFlashTarget: AnyObject {
    /// The identity the engine's model reports as `mtpTargetIdentity`.
    var dflashTargetIdentity: ObjectIdentifier { get }
    var dflashHiddenSize: Int { get }
    var dflashVocabularySize: Int { get }
    /// Width of the hidden rows the target hands to the MTP seam: the
    /// residual stream after each tapped layer, concatenated.
    var dflashContextWidth: Int { get }
    /// The layers whose residual streams make those rows, in order. The
    /// drafter was trained on one set and refuses any other.
    var dflashTappedLayerIDs: [Int] { get }
    /// Rows of the token embedding table WITHOUT the target's embedding
    /// norm: `[..., hiddenSize]`.
    func dflashTokenEmbeddings(_ tokens: MLXArray) -> MLXArray
    /// The output head alone, with no logit multiplier and no softcap:
    /// `[..., vocabulary]`. Both are increasing maps, so the argmax is the
    /// target's.
    func dflashHeadProduct(_ hidden: MLXArray) -> MLXArray
    /// The target's readout after its head product (logit multiplier, then
    /// softcap), so a head product becomes the logits the target samples
    /// from. Only coupled proposals read it: their race needs the target's
    /// scale, while an argmax does not.
    func dflashReadout(_ headProduct: MLXArray) -> MLXArray
    /// True when one forward over a verify window gives every row the bytes
    /// of the decode step it replaces (`rectangularExact`).
    var dflashVerifiesWindowsExactly: Bool { get }
}

extension MuseGlimmerDFlashTarget {
    public var dflashVerifiesWindowsExactly: Bool { false }
    public func dflashReadout(_ headProduct: MLXArray) -> MLXArray { headProduct }
}

/// The Muse Glimmer DFlash block drafter as a request-stateful CBv2 drafter.
///
/// Bound to one target instance. Each request owns the keys and values of its
/// last `sliding_window` committed target positions. The first draft step of
/// a round runs one block forward and keeps its proposals; later steps of the
/// round return them. Proposals are the argmax of the target's own head (or,
/// with `couplesSampledProposals`, a sampled row's race with its verify
/// noise), and the engine keeps one only when it equals the token the target
/// itself chose, so they change speed and never the output.
///
/// With `batchesRequests`, the requests of one step share their passes
/// (`CBv2MTPBatchedRequestDrafter`): one context commit for every observed
/// row, and one block forward and head product for every row of a round.
/// A round's accepted rows are then committed with the next round's carry,
/// in that round's pass.
public final class MuseGlimmerDFlashDrafter: CBv2MTPAdvisoryHistoryDrafter,
    CBv2MTPCoupledProposalDrafter, CBv2MTPBatchedRequestDrafter, CBv2MTPPostVerifySpeculating,
    @unchecked Sendable
{
    /// `startRound` builds the complete block and every later `draftStep`
    /// merely exposes a memo slice.  The proposal token roots the block and
    /// head product; returned hidden slices are not chained into it.
    public let requiresDraftHiddenEvaluation = false

    public struct Settings: Sendable, Equatable {
        /// Draft tokens per round, at most the engine's tested window of 7
        /// and the block's `blockSize - 1`.
        public var maximumDraftTokens: Int
        /// How the target scores a round; nil keeps the engine's setting.
        public var verification: CBv2MTPVerificationMode?
        /// Requests one round may hold. More than one, with rectangular
        /// verification, needs a target whose rows do not depend on each
        /// other.
        public var maximumSpeculativeRows: Int
        /// True only for a target whose verify arithmetic for one row never
        /// depends on the other rows of the step.
        public var commitsAcceptedPrefixPerRow: Bool
        /// Separately admitted allowance for allocator and backend scratch.
        public var backendScratchBytes: Int
        /// A rectangular round attends its (row, column) pairs as rows of one
        /// paged decode dispatch per layer instead of one per column.
        public var batchesVerifyColumnsAsDecodeRows: Bool
        /// Sampled rows propose the race of the draft's row with the noise
        /// of the target's verify draw at each position, instead of the
        /// argmax (`CBv2MTPCoupledProposalDrafter`). Greedy rows keep the
        /// argmax. Output is unchanged either way.
        public var couplesSampledProposals: Bool
        /// The requests of one step share the drafter's passes: observed
        /// rows are committed in one projection, and a round's rows draft
        /// their blocks and head rows in one forward
        /// (`CBv2MTPBatchedRequestDrafter`). Proposals may move in low bits;
        /// output is unchanged either way.
        public var batchesRequests: Bool

        public init(
            maximumDraftTokens: Int = CBv2MTPConfig.testedMaxDraftTokens,
            verification: CBv2MTPVerificationMode? = nil, maximumSpeculativeRows: Int = 1,
            commitsAcceptedPrefixPerRow: Bool = false, backendScratchBytes: Int = 0,
            batchesVerifyColumnsAsDecodeRows: Bool = false, couplesSampledProposals: Bool = false,
            batchesRequests: Bool = false
        ) {
            self.maximumDraftTokens = maximumDraftTokens
            self.verification = verification
            self.maximumSpeculativeRows = maximumSpeculativeRows
            self.commitsAcceptedPrefixPerRow = commitsAcceptedPrefixPerRow
            self.backendScratchBytes = backendScratchBytes
            self.batchesVerifyColumnsAsDecodeRows = batchesVerifyColumnsAsDecodeRows
            self.couplesSampledProposals = couplesSampledProposals
            self.batchesRequests = batchesRequests
        }

        /// One-forward exact verification on a target with exact windows,
        /// each row committing its own prefix and attending its columns as
        /// decode rows; serial verification otherwise.
        public static func verifying(
            _ target: any MuseGlimmerDFlashTarget, maximumSpeculativeRows: Int = 1,
            backendScratchBytes: Int = 0, batchesVerifyColumnsAsDecodeRows: Bool = true,
            couplesSampledProposals: Bool = false, batchesRequests: Bool = false
        ) -> Settings {
            let exact = target.dflashVerifiesWindowsExactly
            return Settings(
                verification: exact ? .rectangularExact : .serialTarget,
                maximumSpeculativeRows: exact ? maximumSpeculativeRows : 1,
                commitsAcceptedPrefixPerRow: exact, backendScratchBytes: backendScratchBytes,
                batchesVerifyColumnsAsDecodeRows: exact && batchesVerifyColumnsAsDecodeRows,
                couplesSampledProposals: couplesSampledProposals, batchesRequests: batchesRequests)
        }
    }

    public let target: any MuseGlimmerDFlashTarget
    public let model: MuseGlimmerDFlashDraftModel
    public let settings: Settings
    /// `[1, hiddenSize]`, the target's raw embedding of the mask token.
    private let maskEmbedding: MLXArray
    /// The target's NVFP4 `lm_head` when it has one. Draft rows take MLX's
    /// own few-row NVFP4 product of it (the NAX `qmv` route on an M5) rather
    /// than the target's row-exact verify kernel: the same E2M1 x E4M3
    /// weights, FP32 accumulation and `weight_scale_2`, in another order.
    /// Proposals only decide acceptance; the target verifies every token.
    private let draftHead: ModelOptWeightOnlyNVFP4Linear?
    private let fusedGateUp: [QuantizedLinear?]
    private let fusedQKV: [QuantizedLinear?]
    private let fusedKV: [QuantizedLinear?]

    public init(
        target: any MuseGlimmerDFlashTarget, model: MuseGlimmerDFlashDraftModel,
        settings: Settings = Settings()
    ) throws {
        let c = model.configuration
        guard target.dflashHiddenSize == c.hiddenSize else {
            throw MuseGlimmerDFlashError.target("hidden size")
        }
        guard target.dflashContextWidth == c.contextWidth else {
            throw MuseGlimmerDFlashError.target("context width")
        }
        guard target.dflashTappedLayerIDs == c.targetLayerIDs else {
            throw MuseGlimmerDFlashError.target("tapped layers")
        }
        guard c.maskTokenID < target.dflashVocabularySize else {
            throw MuseGlimmerDFlashError.target("mask token")
        }
        guard
            (1 ... min(CBv2MTPConfig.testedMaxDraftTokens, c.draftTokensPerBlock)).contains(
                settings.maximumDraftTokens),
            (1 ... CBv2MTPConfig.testedMaxSpeculativeBatch).contains(
                settings.maximumSpeculativeRows),
            settings.backendScratchBytes >= 0
        else { throw MuseGlimmerDFlashError.configuration("settings") }
        guard settings.verification != .rectangularExact || target.dflashVerifiesWindowsExactly
        else { throw MuseGlimmerDFlashError.target("exact verify windows") }
        self.target = target
        self.model = model
        self.settings = settings
        let mask = target.dflashTokenEmbeddings(MLXArray([Int32(c.maskTokenID)]))
        guard mask.shape == [1, c.hiddenSize] else {
            throw MuseGlimmerDFlashError.target("embedding shape")
        }
        maskEmbedding = mask.asType(model.activationDType)
        draftHead = (target as? MuseGlimmerTextModel)?.lmHead as? ModelOptWeightOnlyNVFP4Linear
        Self.requantize8BitLayers(model)
        fusedGateUp = model.layers.map { layer in
            Self.fuseQuantizedGroup([layer.mlp.gate, layer.mlp.up])
        }
        let lastLayerIndex = model.layers.count - 1
        fusedQKV = model.layers.enumerated().map { index, layer in
            guard index < lastLayerIndex else { return nil }
            return Self.fuseQuantizedGroup([
                layer.attention.qProj, layer.attention.kProj, layer.attention.vProj,
            ])
        }
        fusedKV = model.layers.map { layer in
            Self.fuseQuantizedGroup([layer.attention.kProj, layer.attention.vProj])
        }
        try MLX.withError { errors in
            var warm = [maskEmbedding]
            if let textTarget = target as? MuseGlimmerTextModel,
                model.activationDType == .bfloat16
            {
                let dummyLogits = MLXArray.zeros(
                    [1, target.dflashVocabularySize], dtype: .bfloat16)
                let (t, conf) = Self.fusedArgMaxAndConfidence(
                    dummyLogits,
                    outputMultiplier: textTarget.configuration.outputMultiplier,
                    softcap: textTarget.configuration.finalLogitSoftcapping)
                warm += [t, conf, Self.fastArgMax(dummyLogits)]
            }
            eval(warm)
            try errors.check()
        }
    }

    // MARK: Engine policy

    public var mtpTargetIdentity: ObjectIdentifier? { target.dflashTargetIdentity }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { settings.verification }
    public var maximumDraftTokens: Int? { settings.maximumDraftTokens }
    public var maximumSpeculativeBatch: Int? { settings.maximumSpeculativeRows }
    public var commitsAcceptedPrefixPerRow: Bool { settings.commitsAcceptedPrefixPerRow }
    public var batchesVerifyColumnsAsDecodeRows: Bool {
        settings.verification == .rectangularExact && settings.batchesVerifyColumnsAsDecodeRows
    }
    /// Exact acceptance only (decision D6): typical acceptance can emit a
    /// draft, and the advisory history then loses its observation positions.
    public var requiresExactAcceptance: Bool { true }
    /// A proposal is kept only when it equals the token pre-sampled from the
    /// target's own distribution, and that token is what is emitted, so
    /// sampled rows speculate too.
    public var supportsTargetPrefixAcceptance: Bool { true }
    public var couplesSampledProposals: Bool { settings.couplesSampledProposals }
    public var batchesRequests: Bool { settings.batchesRequests }
    /// Observation builds lazy writes into the context ring; they must be
    /// evaluated with the step that produced the observed rows.
    public var requiresCommittedObservationFence: Bool { true }
    /// The prompt's observation commits every prompt row, the last one
    /// included, and awaits no carry row; the first round's block then
    /// anchors on the first token, exactly as after a seed step, which only
    /// moved both one position on. Proposals change speed, never the output.
    public var seedsFirstRoundFromPrefill: Bool { true }
    /// The ring is a fixed charge (`boundedRequestAllocation`), not a rate.
    public var requestStateBytesPerToken: Int { 0 }
    /// The context only decides what is proposed. It is not part of a prefix
    /// checkpoint; after a restore it starts at the first observed position.
    public var requestStateIsAdvisory: Bool { true }

    // MARK: Request state

    final class State: CBv2MTPRequestState {
        let owner: ObjectIdentifier
        let context: MuseGlimmerDFlashContextState
        var released = false
        /// The last committed position's row has not arrived yet: a verify
        /// round hands it over as the next carry.
        var awaitsCarry = false
        /// This round's proposals `[maximumDraftTokens]` and their hidden
        /// rows `[maximumDraftTokens, hiddenSize]`.
        var memo: (tokens: MLXArray, hidden: MLXArray)?
        /// This round's coupled-proposal rule; nil proposes the argmax.
        var selection: CBv2MTPProposalSelection?
        /// `startRounds` already drafted this round's block.
        var prestarted = false
        /// The last round's accepted target rows `[rows, contextWidth]`,
        /// not yet in the ring (`batchesRequests` only). They are the next
        /// positions and are committed before any other row.
        var pending: MLXArray?
        var committedInputCount = 0
        var stagedInputCount = 0
        /// Why this request cannot continue; reported at the next fence.
        var refusal: String?
        /// The next round's index-0 work, built behind this round's verify
        /// (`speculateNextRound`): the commit for every outcome, the block
        /// for the all-accepted one. Not in `arrays`: the round's fence
        /// never waits on it.
        var speculation: Speculation?
        /// The last finalized round accepted every draft.
        var lastRoundFullyAccepted = false

        init(owner: ObjectIdentifier, context: MuseGlimmerDFlashContextState) {
            self.owner = owner
            self.context = context
        }

        var arrays: [MLXArray] { context.arrays + (memo.map { [$0.tokens, $0.hidden] } ?? []) }
        var materializedBytes: Int { arrays.reduce(0) { $0 + $1.nbytes } }
    }

    /// What `speculateNextRound` built and what the normal path must match
    /// for it to be handed out.
    struct Speculation {
        let depth: Int
        /// The context's `nextPosition` and `firstPosition` when built.
        let anchor: Int
        let first: Int
        /// Keys and values `[layers, kvHeads, depth + 1, headDimensions]` of
        /// every verify row at `anchor ..< anchor + depth + 1`. An outcome
        /// with `a` accepted drafts commits the first `a + 1` of them.
        let keys: MLXArray
        let values: MLXArray
        /// The next block for the all-accepted outcome, when built; dropped
        /// at confirmation for any other outcome.
        var memo: (tokens: MLXArray, hidden: MLXArray)? = nil
        /// `[maximumDraftTokens]` float32: each block row's top-1
        /// probability under the target's readout, evaluated with the
        /// speculation (`speculatedDraftConfidence`). Set with `memo`.
        var confidence: MLXArray? = nil
        /// The accepted draft count `finalizeRound` saw, and the pending rows
        /// it set for it (nil when it accepted none).
        var accepted: Int? = nil
        var pending: MLXArray? = nil
        /// The carry the engine stored for that outcome.
        var carry: MLXArray? = nil
    }

    /// What `takeSpeculation` applied at a round's draft index 0.
    private enum TakenSpeculation {
        /// Nothing: the normal path runs.
        case normal
        /// The commit: the block still has to be drafted.
        case committed
        /// The commit and the all-accepted block's memo.
        case drafted
    }

    private final class Prepared: CBv2MTPPreparedCapture {}

    private func state(_ value: any CBv2MTPRequestState) -> State {
        guard let state = value as? State, state.owner == ObjectIdentifier(self), !state.released
        else { preconditionFailure("Muse Glimmer DFlash: foreign or released request state") }
        return state
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        State(owner: ObjectIdentifier(self), context: model.makeContextState())
    }

    /// The engine's rule for this round's proposals. Ignored unless
    /// `couplesSampledProposals`; cleared with the round.
    public func setProposalSelection(
        _ selection: CBv2MTPProposalSelection?, requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        s.selection = settings.couplesSampledProposals ? selection : nil
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Muse Glimmer DFlash requires request-owned state")
    }

    /// Target hidden rows `[1, rows, contextWidth]` as feature rows; nil for
    /// any other shape.
    private func features(_ hidden: MLXArray) -> MLXArray? {
        guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) > 0,
            hidden.dim(2) == model.configuration.contextWidth
        else { return nil }
        return hidden.reshaped([hidden.dim(1), hidden.dim(2)])
    }

    /// Commit the rows of one observation as the next context positions.
    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        observe(observation, at: s.context.nextPosition, state: s)
    }

    /// The same, with the absolute position of the observation's first row.
    /// A position after the held rows restarts the context there; rows
    /// already held are skipped.
    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, startingAt position: Int,
        requestState: any CBv2MTPRequestState
    ) {
        observe(observation, at: position, state: state(requestState))
    }

    private func observe(
        _ observation: CBv2MTPCommittedTargetObservation, at position: Int, state s: State
    ) {
        s.speculation = nil
        guard s.refusal == nil else { return }
        guard s.stagedInputCount == 0, position >= 0, let rows = features(observation.hidden)
        else {
            s.refusal = "observation"
            return
        }
        if s.pending == nil {
            model.commit(features: rows, at: position, to: s.context)
        } else {
            model.commit(takePending(s) + [.init(features: rows, at: position, to: s.context)])
        }
        // Whatever arrived, the context now ends at its newest row.
        s.awaitsCarry = false
        s.committedInputCount += rows.dim(0)
    }

    /// The state's pending rows as the first commit, at its next position.
    private func takePending(_ s: State) -> [MuseGlimmerDFlashDraftModel.ContextCommit] {
        defer { s.pending = nil }
        return s.pending.map { [.init(features: $0, at: nil, to: s.context)] } ?? []
    }

    /// Every entry as `observeCommittedTarget` would take it, with one
    /// context commit for all of them.
    public func observeCommittedTargets(_ observations: [CBv2MTPRequestObservation]) {
        var commits: [MuseGlimmerDFlashDraftModel.ContextCommit] = []
        var observed: [(state: State, rows: Int)] = []
        for entry in observations {
            let s = state(entry.requestState)
            s.speculation = nil
            guard s.refusal == nil else { continue }
            guard s.stagedInputCount == 0, entry.position.map({ $0 >= 0 }) ?? true,
                let rows = features(entry.observation.hidden)
            else {
                s.refusal = "observation"
                continue
            }
            commits += takePending(s)
            commits.append(.init(features: rows, at: entry.position, to: s.context))
            observed.append((s, rows.dim(0)))
        }
        model.commit(commits)
        for (s, rows) in observed {
            s.awaitsCarry = false
            s.committedInputCount += rows
        }
    }

    /// Draft index 0 commits a carry row that is still missing, runs one
    /// block forward and keeps its proposals. Later indices return them.
    ///
    /// A call this drafter cannot serve never stops the engine thread: it
    /// returns a well-formed proposal and the request ends with an error at
    /// the round's fence (`requestStateDidFinishEvaluation`).
    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let s = state(requestState)
        let c = model.configuration
        let index = s.stagedInputCount
        s.stagedInputCount += 1
        if s.refusal == nil, index == 0 {
            if s.prestarted {
                s.prestarted = false
            } else {
                s.refusal = startRound(s, tokens: tokens, hidden: hidden)
            }
        }
        guard s.refusal == nil, shortlist == nil, index < settings.maximumDraftTokens,
            let memo = s.memo
        else {
            s.refusal = s.refusal ?? "draft step"
            s.memo = nil
            return (
                MLXArray([Int32(0)]),
                MLXArray.zeros([1, 1, c.hiddenSize], dtype: model.activationDType)
            )
        }
        return (
            memo.tokens[index ..< (index + 1)],
            memo.hidden[index ..< (index + 1), 0...].reshaped([1, 1, c.hiddenSize])
        )
    }

    /// Builds the round's block and memo. Returns why it could not.
    private func startRound(_ s: State, tokens: MLXArray, hidden: MLXArray) -> String? {
        let c = model.configuration
        let speculation = s.speculation
        s.speculation = nil
        guard tokens.shape == [1, 1] else { return "seed token" }
        let taken = speculation.map { takeSpeculation($0, state: s, hidden: hidden) } ?? .normal
        if case .drafted = taken { return nil }
        if case .committed = taken {
            // The pending rows and the carry are in the ring: draft the block.
        } else if settings.batchesRequests {
            guard let carry = commitCarry(s, hidden: hidden) else { return "carry" }
            if !carry.isEmpty { model.commit(carry) }
        } else if s.awaitsCarry {
            guard let row = features(hidden), row.dim(0) == 1 else { return "carry" }
            model.commit(features: row, at: s.context.nextPosition, to: s.context)
            s.awaitsCarry = false
            s.committedInputCount += 1
        }
        let anchor = target.dflashTokenEmbeddings(tokens.asType(.int32))
        guard anchor.size == c.hiddenSize else { return "embedding" }
        let visible = s.context.visibleContext()
        let rows = fastDraftBlockRows(
            embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding),
            at: s.context.nextPosition,
            context: visible.map { ($0.keys, $0.values) }, mask: visible?.mask,
            draftCount: settings.maximumDraftTokens)
        let logits = draftHeadProduct(rows, readout: s.selection != nil)
        guard logits.shape == [settings.maximumDraftTokens, target.dflashVocabularySize] else {
            return "head"
        }
        s.memo = (proposals(logits, selection: s.selection), rows)
        return nil
    }

    /// A round's commit before its block: the pending rows and the carry row
    /// still missing, as one entry at the next position. Nil when the carry
    /// is not one feature row.
    private func commitCarry(_ s: State, hidden: MLXArray)
        -> [MuseGlimmerDFlashDraftModel.ContextCommit]?
    {
        var rows = s.pending.map { [$0] } ?? []
        s.pending = nil
        if s.awaitsCarry {
            guard let row = features(hidden), row.dim(0) == 1 else { return nil }
            rows.append(row)
            s.awaitsCarry = false
            s.committedInputCount += 1
        }
        guard !rows.isEmpty else { return [] }
        let features = rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
        return [.init(features: features, at: s.context.nextPosition, to: s.context)]
    }

    /// Draft index 0 of every seed's round in one pass: one context commit
    /// for every carry (and pending rows), one block forward and one head
    /// product for every block. A seed this cannot serve is left to its own
    /// index-0 call, which refuses it as before.
    public func startRounds(_ seeds: [CBv2MTPRoundSeed]) {
        guard settings.batchesRequests else { return }
        let c = model.configuration
        var started: [(state: State, tokens: MLXArray)] = []
        var commits: [MuseGlimmerDFlashDraftModel.ContextCommit] = []
        for seed in seeds {
            let s = state(seed.requestState)
            s.speculation = nil
            guard s.refusal == nil, s.stagedInputCount == 0, !s.prestarted, seed.shortlist == nil,
                seed.tokens.shape == [1, 1]
            else { continue }
            guard let carry = commitCarry(s, hidden: seed.hidden) else {
                s.refusal = "carry"
                continue
            }
            commits += carry
            started.append((s, seed.tokens))
        }
        guard !started.isEmpty else { return }
        model.commit(commits)
        var blocks: [MLXArray] = []
        for (s, tokens) in started {
            let anchor = target.dflashTokenEmbeddings(tokens.asType(.int32))
            guard anchor.size == c.hiddenSize else {
                s.refusal = "embedding"
                continue
            }
            blocks.append(model.blockEmbeddings(anchor: anchor, mask: maskEmbedding))
        }
        let states = started.map(\.state).filter { $0.refusal == nil }
        guard !states.isEmpty else { return }
        let k = settings.maximumDraftTokens
        let rows: [MLXArray]
        if states.count == 1 {
            let ctx = states[0].context
            let visible = ctx.visibleContext()
            rows = [
                fastDraftBlockRows(
                    embeddings: blocks[0], at: ctx.nextPosition,
                    context: visible.map { ($0.keys, $0.values) }, mask: visible?.mask,
                    draftCount: k)
            ]
        } else {
            rows = model.draftBlocks(embeddings: blocks, states: states.map(\.context)).map {
                $0[1 ..< (1 + k), 0...]
            }
        }
        let logits = draftHeadProduct(
            rows.count == 1 ? rows[0] : concatenated(rows, axis: 0),
            readout: states.contains { $0.selection != nil })
        guard logits.shape == [states.count * k, target.dflashVocabularySize] else {
            for s in states { s.refusal = "head" }
            return
        }
        for (index, s) in states.enumerated() {
            s.memo = (
                proposals(logits[(index * k) ..< ((index + 1) * k), 0...], selection: s.selection),
                rows[index]
            )
            s.prestarted = true
        }
    }

    // MARK: Post-verify speculation

    /// The served round shape the speculation reproduces: batched requests,
    /// argmax proposals.
    var speculatesAfterVerify: Bool {
        settings.batchesRequests && !settings.couplesSampledProposals
    }

    /// Builds the next round's index-0 work from the verify's lazy rows,
    /// behind the submitted verify, before the acceptance count is known.
    ///
    /// For every outcome the next index 0 first commits one entry at the
    /// context's next position: `finalizeRound`'s pending rows (verify rows
    /// `0 ..< a` for `a` accepted drafts) and the engine's carry (verify row
    /// `a`). Those are the first `a + 1` verify rows whatever `a` is, so this
    /// projects all `depth + 1` rows once (`model.commit`'s projection, no
    /// duplicates, no restart, nothing unread) and the next index 0 appends
    /// the first `a + 1` of them instead of projecting them on the critical
    /// path. Only the projection moves: the ring, the pending rows and the
    /// carry are untouched until `takeSpeculation` confirms the outcome.
    ///
    /// With `draftsBlock`, after a round that accepted every draft, it also
    /// drafts the next block for the all-accepted outcome, whose seed is the
    /// verify's bonus target `targetTokens[depth]`: the append it implies is
    /// written into new wrappers of the ring arrays (the slots `append`
    /// would write).
    ///
    /// Only for the served one-row batched argmax round.
    public func speculateNextRound(
        targetTokens: MLXArray, targetHidden: MLXArray, depth k: Int, draftsBlock: Bool,
        requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        let s = state(requestState)
        s.speculation = nil
        let c = model.configuration
        let context = s.context
        let window = context.window
        let n = k + 1
        let ring = context.arrays
        guard speculatesAfterVerify, s.refusal == nil, s.memo != nil,
            s.selection == nil, !s.prestarted, s.pending == nil, !s.awaitsCarry, k >= 1,
            k == s.stagedInputCount,
            n <= window, context.count > 0, ring.count == 2,
            targetTokens.shape == [1, n], targetHidden.ndim == 3, targetHidden.dim(0) == 1,
            targetHidden.dim(1) == n, targetHidden.dim(2) == c.contextWidth
        else { return [] }

        // The verify rows as finalizeRound and the engine slice them; the
        // all-accepted outcome's entry is all of them.
        let features = targetHidden[0 ..< 1, 0 ..< n, 0...].reshaped([n, c.contextWidth])
        let start = context.nextPosition
        let projected = fusedContextKeyValues(
            model.projectContext(features), at: start)
        var speculation = Speculation(
            depth: k, anchor: start, first: context.firstPosition, keys: projected.keys,
            values: projected.values)
        guard draftsBlock, s.lastRoundFullyAccepted else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }

        let next = start + n
        let first = max(context.firstPosition, next - window)
        let count = next - first

        // startRound's block and memo from the bonus target as the seed.
        let seed = maximum(targetTokens[0 ..< 1, k ..< n].asType(.int32), MLXArray(Int32(0)))
        let anchor = target.dflashTokenEmbeddings(seed)
        guard anchor.size == c.hiddenSize else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }
        let rows: MLXArray
        if context.count + n <= window {
            // Fast path: ring has not wrapped yet — concatenate existing slice
            // and projected rows instead of copying the entire 40 MB ring.
            let existingCount = context.count
            let ctxKeys = concatenated([
                ring[0][0..., 0..., 0 ..< existingCount, 0...],
                projected.keys.asType(context.dtype),
            ], axis: 2)
            let ctxValues = concatenated([
                ring[1][0..., 0..., 0 ..< existingCount, 0...],
                projected.values.asType(context.dtype),
            ], axis: 2)
            let held = Array(first ..< next)
            let mask = MuseGlimmerDFlashContextState.visibility(
                anchor: next, blockSize: context.blockSize, window: window, contextPositions: held
            ).map { MLXArray($0, [context.blockSize, count + context.blockSize]) }
            rows = fastDraftBlockRows(
                embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding), at: next,
                context: (ctxKeys, ctxValues), mask: mask,
                draftCount: settings.maximumDraftTokens)
        } else {
            // context.append on new wrappers. Slot zero holds a position
            // congruent to the ring's base position modulo the window: the
            // first held one before the ring wraps, the base lap's slot zero
            // after (`heldPositions`).
            let slotZero = context.heldPositions[0]
            func slot(_ position: Int) -> Int { ((position - slotZero) % window + window) % window }
            let keys = ring[0].sharingArray()
            let values = ring[1].sharingArray()
            var written = 0
            while written < n {
                let at = slot(start + written)
                let run = min(n - written, window - at)
                let source = written ..< (written + run)
                keys[0..., 0..., at ..< (at + run), 0...] =
                    projected.keys[0..., 0..., source, 0...].asType(context.dtype)
                values[0..., 0..., at ..< (at + run), 0...] =
                    projected.values[0..., 0..., source, 0...].asType(context.dtype)
                written += run
            }
            // visibleContext of the appended ring.
            let held: [Int]
            if count == window {
                let oldest = slot(first)
                held = (0 ..< window).map { first + ($0 - oldest + window) % window }
            } else {
                held = Array(first ..< next)
            }
            let mask = MuseGlimmerDFlashContextState.visibility(
                anchor: next, blockSize: context.blockSize, window: window, contextPositions: held
            ).map { MLXArray($0, [context.blockSize, count + context.blockSize]) }
            rows = fastDraftBlockRows(
                embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding), at: next,
                context: (keys[0..., 0..., 0 ..< count, 0...], values[0..., 0..., 0 ..< count, 0...]),
                mask: mask, draftCount: settings.maximumDraftTokens)
        }
        // startRound's head: the speculated memo must equal the normal one.
        let logits = draftHeadProduct(rows, readout: false)
        guard logits.shape == [settings.maximumDraftTokens, target.dflashVocabularySize] else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }
        let memo: (tokens: MLXArray, hidden: MLXArray)
        let confidence: MLXArray
        if let textTarget = target as? MuseGlimmerTextModel, logits.dtype == .bfloat16 {
            let (tokens, conf) = Self.fusedArgMaxAndConfidence(
                logits,
                outputMultiplier: textTarget.configuration.outputMultiplier,
                softcap: textTarget.configuration.finalLogitSoftcapping)
            memo = (tokens: tokens, hidden: rows)
            confidence = conf
        } else {
            memo = (tokens: proposals(logits, selection: nil), hidden: rows)
            // Per-row top-1 probability of the proposal, softmax over the
            // target's readout scale: exp(max - logsumexp). Advisory only: it
            // sizes the next verify window and never changes a proposal.
            let scores = target.dflashReadout(logits).asType(.float32)
            confidence = exp(scores.max(axis: -1) - scores.logSumExp(axis: -1))
        }
        speculation.memo = memo
        speculation.confidence = confidence
        s.speculation = speculation
        return [memo.tokens, memo.hidden, projected.keys, projected.values, confidence]
    }

    /// The confirmed speculation's per-row draft confidence, still lazy (the
    /// caller reads it only once it is available); nil otherwise.
    public func speculatedDraftConfidence(requestState: any CBv2MTPRequestState) -> MLXArray? {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released,
            let speculation = s.speculation, speculation.carry != nil, speculation.memo != nil,
            speculation.accepted == speculation.depth, speculation.pending != nil
        else { return nil }
        return speculation.confidence
    }

    /// Finalize stored `carryHidden` (this object) as the row's carry after
    /// `acceptedDrafts` accepted drafts, the carry being verify row
    /// `acceptedDrafts`. `carryTokenIsTarget`: its token is the verify's
    /// own target at that row.
    public func confirmSpeculation(
        carryHidden: MLXArray, acceptedDrafts: Int, carryTokenIsTarget: Bool,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        guard var speculation = s.speculation, speculation.carry == nil,
            let accepted = speculation.accepted, accepted == acceptedDrafts,
            (accepted == 0) == (speculation.pending == nil)
        else {
            s.speculation = nil
            return
        }
        // The block assumed every draft accepted and the bonus target as the
        // seed; any other outcome keeps the commit only.
        if accepted != speculation.depth || !carryTokenIsTarget {
            speculation.memo = nil
            speculation.confidence = nil
        }
        speculation.carry = carryHidden
        s.speculation = speculation
    }

    public func dropSpeculation(requestState: any CBv2MTPRequestState) {
        state(requestState).speculation = nil
    }

    /// Index 0 of the round after a speculated one: when the state is the
    /// one `speculateNextRound` assumed for the outcome finalize confirmed
    /// (that carry object, the pending rows `finalizeRound` set for it, the
    /// context unchanged since), apply commitCarry's transitions and append
    /// that outcome's rows, and hand out the block's memo when it was built
    /// for this outcome. `.normal` leaves the state as it was.
    private func takeSpeculation(_ speculation: Speculation, state s: State, hidden: MLXArray)
        -> TakenSpeculation
    {
        guard settings.batchesRequests, s.awaitsCarry,
            let carry = speculation.carry, carry === hidden,
            let accepted = speculation.accepted, accepted >= 0, accepted <= speculation.depth,
            s.context.nextPosition == speculation.anchor,
            s.context.firstPosition == speculation.first
        else { return .normal }
        if let expected = speculation.pending {
            guard let pending = s.pending, pending === expected else { return .normal }
        } else {
            guard accepted == 0, s.pending == nil else { return .normal }
        }
        let memo = s.selection == nil ? speculation.memo : nil
        s.pending = nil
        s.awaitsCarry = false
        s.committedInputCount += 1
        if accepted == speculation.depth {
            s.context.append(keys: speculation.keys, values: speculation.values)
        } else {
            let rows = 0 ..< (accepted + 1)
            s.context.append(
                keys: speculation.keys[0..., 0..., rows, 0...],
                values: speculation.values[0..., 0..., rows, 0...])
        }
        guard let memo else { return .committed }
        s.memo = memo
        return .drafted
    }

    /// The head product of draft rows `[rows, hiddenSize]`: MLX's NVFP4
    /// product of the target's head when it is NVFP4, else the target's own.
    /// Without `readout` (argmax proposals only) the NVFP4 product skips
    /// `weight_scale_2`: a positive scalar, so the row argmax is the same up
    /// to bf16 rounding ties. A coupled selection reads the target's readout
    /// of the product and needs the scale.
    private func draftHeadProduct(_ rows: MLXArray, readout: Bool) -> MLXArray {
        guard let draftHead, rows.dim(-1) == draftHead.inputDimensions else {
            return target.dflashHeadProduct(rows)
        }
        return readout ? draftHead(rows) : draftHead.unscaledProduct(rows)
    }

    private static let requantLock = NSLock()
    nonisolated(unsafe) private static var requantizedModels = Set<ObjectIdentifier>()

    /// Re-quantizes the drafter's 8-bit linear projections (`encoder.fc` and
    /// `self_attn.{q,k,v,o}_proj`, ~597 MB) in memory on load to 4-bit
    /// `groupSize: 32` (~335 MB), leaving the 5-bit MLP weights untouched.
    /// Permitted by the track contract (drafter only proposes tokens; the
    /// pinned target verifies every emitted token).
    private static func requantize8BitLayers(_ model: MuseGlimmerDFlashDraftModel) {
        let key = ObjectIdentifier(model)
        requantLock.lock()
        let claimed = requantizedModels.insert(key).inserted
        requantLock.unlock()
        guard claimed else { return }

        let updates = model.leafModules().compactMapValues { m -> Module? in
            guard let ql = m as? QuantizedLinear, ql.mode == .affine,
                ql.bits == 8 || ql.bits == 5
            else {
                return nil
            }
            let targetBits = ql.bits == 8 ? 4 : 3
            let targetGroupSize = ql.bits == 8 ? 32 : 64
            let dense = MLX.dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
            let replacement = QuantizedLinear(
                weight: dense, bias: ql.bias, groupSize: targetGroupSize, bits: targetBits,
                mode: .affine)
            if let biases = replacement.biases {
                eval(replacement.weight, replacement.scales, biases)
            } else {
                eval(replacement.weight, replacement.scales)
            }
            return replacement
        }
        model.update(modules: updates)
    }

    private static func fuseQuantizedGroup(_ linears: [Linear]) -> QuantizedLinear? {
        guard linears.count >= 2 else { return nil }
        var quantized: [QuantizedLinear] = []
        quantized.reserveCapacity(linears.count)
        for linear in linears {
            guard let ql = linear as? QuantizedLinear,
                ql.bias == nil, ql.globalScale == nil
            else { return nil }
            quantized.append(ql)
        }
        let first = quantized[0]
        let hasBiases = first.biases != nil
        for ql in quantized.dropFirst() {
            guard ql.bits == first.bits, ql.groupSize == first.groupSize, ql.mode == first.mode,
                (ql.biases != nil) == hasBiases
            else { return nil }
        }
        let w = concatenated(quantized.map { $0.weight }, axis: 0)
        let s = concatenated(quantized.map { $0.scales }, axis: 0)
        let bi: MLXArray? =
            if hasBiases {
                concatenated(quantized.compactMap { $0.biases }, axis: 0)
            } else {
                nil
            }
        if let bi {
            eval(w, s, bi)
        } else {
            eval(w, s)
        }
        return QuantizedLinear(
            weight: w, bias: nil, scales: s, biases: bi,
            groupSize: first.groupSize, bits: first.bits, mode: first.mode)
    }

    private func fusedContextKeyValues(
        _ context: MLXArray, at position: Int
    ) -> (keys: MLXArray, values: MLXArray) {
        let c = model.configuration
        let rows = context.dim(0)
        let shape = [1, rows, c.kvHeads, c.headDimensions]
        let kvDim = c.kvHeads * c.headDimensions
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        keys.reserveCapacity(model.layers.count)
        values.reserveCapacity(model.layers.count)
        for (index, layer) in model.layers.enumerated() {
            let attention = layer.attention
            if let kvProj = fusedKV[index] {
                let kv = kvProj(context)
                let key = attention.kNorm(
                    kv[0..., 0 ..< kvDim].reshaped(shape).transposed(0, 2, 1, 3))
                keys.append(
                    MLXFast.RoPE(
                        key, dimensions: c.headDimensions, traditional: false,
                        base: c.ropeTheta, scale: 1, offset: position))
                values.append(
                    kv[0..., kvDim...].reshaped(shape).transposed(0, 2, 1, 3))
            } else {
                let key = attention.kNorm(
                    attention.kProj(context).reshaped(shape).transposed(0, 2, 1, 3))
                keys.append(
                    MLXFast.RoPE(
                        key, dimensions: c.headDimensions, traditional: false,
                        base: c.ropeTheta, scale: 1, offset: position))
                values.append(
                    attention.vProj(context).reshaped(shape).transposed(0, 2, 1, 3))
            }
        }
        return (concatenated(keys, axis: 0), concatenated(values, axis: 0))
    }

    /// One block forward returning only the `draftCount` active draft rows
    /// (`1 ..< 1 + draftCount`). Layers `0 ..< last` run on all `blockSize`
    /// rows (with `compiledSiluProduct` in the MLP). The final layer projects
    /// K/V on all `blockSize` rows so attention sees the full block, then
    /// slices queries, attention, O-proj, MLP and final norm to only the
    /// `draftCount` active rows (7 instead of 16), cutting final-layer work
    /// in half with mathematically identical proposals.
    private func fastDraftBlockRows(
        embeddings: MLXArray, at position: Int,
        context: (keys: MLXArray, values: MLXArray)?, mask: MLXArray?,
        draftCount: Int
    ) -> MLXArray {
        let c = model.configuration
        let block = c.blockSize
        let scale = 1 / Float(c.headDimensions).squareRoot()
        let act = embeddings.dtype == model.activationDType
            ? embeddings : embeddings.asType(model.activationDType)
        var hidden = act.reshaped([1, block, c.hiddenSize])
        let lastIndex = model.layers.count - 1
        for (index, layer) in model.layers.enumerated() {
            let attention = layer.attention
            let x = layer.inputNorm(hidden)
            var queries: MLXArray?
            var keys: MLXArray
            var values: MLXArray
            if index < lastIndex, let qkvProj = fusedQKV[index] {
                let qkv = qkvProj(x)
                let qDim = c.attentionHeads * c.headDimensions
                let kvDim = c.kvHeads * c.headDimensions
                queries = MLXFast.RoPE(
                    attention.qNorm(
                        qkv[0..., 0..., 0 ..< qDim]
                            .reshaped([1, block, c.attentionHeads, c.headDimensions])
                            .transposed(0, 2, 1, 3)),
                    dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: position)
                keys = MLXFast.RoPE(
                    attention.kNorm(
                        qkv[0..., 0..., qDim ..< (qDim + kvDim)]
                            .reshaped([1, block, c.kvHeads, c.headDimensions])
                            .transposed(0, 2, 1, 3)),
                    dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: position)
                values = qkv[0..., 0..., (qDim + kvDim)...]
                    .reshaped([1, block, c.kvHeads, c.headDimensions])
                    .transposed(0, 2, 1, 3)
            } else if let kvProj = fusedKV[index] {
                let kv = kvProj(x)
                let kvDim = c.kvHeads * c.headDimensions
                keys = MLXFast.RoPE(
                    attention.kNorm(
                        kv[0..., 0..., 0 ..< kvDim]
                            .reshaped([1, block, c.kvHeads, c.headDimensions])
                            .transposed(0, 2, 1, 3)),
                    dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: position)
                values = kv[0..., 0..., kvDim...]
                    .reshaped([1, block, c.kvHeads, c.headDimensions])
                    .transposed(0, 2, 1, 3)
            } else {
                keys = MLXFast.RoPE(
                    attention.kNorm(
                        attention.kProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                            .transposed(0, 2, 1, 3)),
                    dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: position)
                values = attention.vProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                    .transposed(0, 2, 1, 3)
            }
            if let context {
                keys = concatenated([context.keys[index ..< index + 1], keys], axis: 2)
                values = concatenated([context.values[index ..< index + 1], values], axis: 2)
            }
            if index == lastIndex {
                let active = 1 ..< (1 + draftCount)
                let xActive = x[0..., active, 0...]
                let activeQueries = MLXFast.RoPE(
                    attention.qNorm(
                        attention.qProj(xActive).reshaped([
                            1, draftCount, c.attentionHeads, c.headDimensions,
                        ]).transposed(0, 2, 1, 3)),
                    dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: position + 1)
                let activeMask = mask.map { $0[active, 0...] }
                let attended = MLXFast.scaledDotProductAttention(
                    queries: activeQueries, keys: keys, values: values, scale: scale,
                    mask: activeMask)
                var h = hidden[0..., active, 0...]
                    + attention.oProj(
                        attended.transposed(0, 2, 1, 3).reshaped([
                            1, draftCount, c.attentionHeads * c.headDimensions,
                        ]))
                let mx = layer.postAttentionNorm(h)
                let mlpAct: MLXArray
                if let gu = fusedGateUp[index] {
                    let gateUp = gu(mx)
                    let inter = c.intermediateSize
                    mlpAct = compiledSiluProduct(
                        gateUp[0..., 0..., 0 ..< inter], gateUp[0..., 0..., inter...])
                } else {
                    mlpAct = compiledSiluProduct(layer.mlp.gate(mx), layer.mlp.up(mx))
                }
                h = h + layer.mlp.down(mlpAct)
                return model.norm(h).reshaped([draftCount, c.hiddenSize])
            }
            let blockQueries =
                queries
                ?? MLXFast.RoPE(
                    attention.qNorm(
                        attention.qProj(x).reshaped([1, block, c.attentionHeads, c.headDimensions])
                            .transposed(0, 2, 1, 3)),
                    dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: position)
            let attended = MLXFast.scaledDotProductAttention(
                queries: blockQueries, keys: keys, values: values, scale: scale, mask: mask)
            hidden =
                hidden
                + attention.oProj(
                    attended.transposed(0, 2, 1, 3).reshaped([
                        1, block, c.attentionHeads * c.headDimensions,
                    ]))
            let mx = layer.postAttentionNorm(hidden)
            let mlpAct: MLXArray
            if let gu = fusedGateUp[index] {
                let gateUp = gu(mx)
                let inter = c.intermediateSize
                mlpAct = compiledSiluProduct(
                    gateUp[0..., 0..., 0 ..< inter], gateUp[0..., 0..., inter...])
            } else {
                mlpAct = compiledSiluProduct(layer.mlp.gate(mx), layer.mlp.up(mx))
            }
            hidden = hidden + layer.mlp.down(mlpAct)
        }
        return model.norm(hidden).reshaped([block, c.hiddenSize])[1 ..< (1 + draftCount), 0...]
    }

    /// Fused 2-stage split-V argmax + softcapped-readout top-1 probability.
    /// Replaces 9 separate dispatches (~50 MB traffic) over `[M, V]` logits
    /// with one read of `logits` (2.8 MB) and a 64-way partial reduction.
    private static let argMaxConfStage1 = MLXFast.metalKernel(
        name: "dflash_argmax_conf_s1",
        inputNames: ["logits", "params"],
        outputNames: ["p_raw", "p_idx", "p_sc", "p_sum"],
        source: """
            constexpr int V = V_;
            constexpr int CHUNKS = 64;
            constexpr int chunk_size = (V + CHUNKS - 1) / CHUNKS;
            const uint chunk = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;

            const int start = int(chunk) * chunk_size;
            const int end = min(V, start + chunk_size);
            const device bfloat16_t* row_ptr = logits + size_t(row) * size_t(V);
            const bfloat16_t b_mult = bfloat16_t(params[0]);
            const bfloat16_t b_cap = bfloat16_t(params[1]);
            const bool has_cap = params[1] > 0.0f;

            float best_raw = -INFINITY;
            int best_idx = V;
            float best_sc = -INFINITY;
            float sum_exp = 0.0f;
            for (int i = start + int(lid); i < end; i += 256) {
              const bfloat16_t raw_bf = row_ptr[i];
              const float raw_f = float(raw_bf);
              if (raw_f > best_raw || (raw_f == best_raw && i < best_idx)) {
                best_raw = raw_f;
                best_idx = i;
              }
              bfloat16_t s_bf = raw_bf * b_mult;
              if (has_cap) {
                s_bf = bfloat16_t(metal::precise::tanh(float(s_bf / b_cap))) * b_cap;
              }
              const float sc = float(s_bf);
              if (sc > best_sc) {
                sum_exp = (sum_exp > 0.0f) ? (sum_exp * metal::exp(best_sc - sc) + 1.0f) : 1.0f;
                best_sc = sc;
              } else {
                sum_exp += metal::exp(sc - best_sc);
              }
            }

            const float sg_raw = simd_max(best_raw);
            const int sg_idx = simd_min((best_raw == sg_raw) ? best_idx : V);
            const float sg_sc = simd_max(best_sc);
            const float sg_sum = simd_sum(
                (sum_exp > 0.0f) ? (sum_exp * metal::exp(best_sc - sg_sc)) : 0.0f);

            threadgroup float sh_raw[8];
            threadgroup int sh_idx[8];
            threadgroup float sh_sc[8];
            threadgroup float sh_sum[8];
            if (lane == 0) {
              sh_raw[sg] = sg_raw;
              sh_idx[sg] = sg_idx;
              sh_sc[sg] = sg_sc;
              sh_sum[sg] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float v_raw = (lane < 8) ? sh_raw[lane] : -INFINITY;
              const int v_idx = (lane < 8) ? sh_idx[lane] : V;
              const float v_sc = (lane < 8) ? sh_sc[lane] : -INFINITY;
              const float v_sum = (lane < 8) ? sh_sum[lane] : 0.0f;

              const float tg_raw = simd_max(v_raw);
              const int tg_idx = simd_min((v_raw == tg_raw) ? v_idx : V);
              const float tg_sc = simd_max(v_sc);
              const float tg_sum = simd_sum(
                  (v_sum > 0.0f) ? (v_sum * metal::exp(v_sc - tg_sc)) : 0.0f);

              if (lane == 0) {
                const uint out_i = row * uint(CHUNKS) + chunk;
                p_raw[out_i] = tg_raw;
                p_idx[out_i] = tg_idx;
                p_sc[out_i] = tg_sc;
                p_sum[out_i] = tg_sum;
              }
            }
            """)

    private static let argMaxConfStage2 = MLXFast.metalKernel(
        name: "dflash_argmax_conf_s2",
        inputNames: ["p_raw", "p_idx", "p_sc", "p_sum"],
        outputNames: ["tokens", "confidence"],
        source: """
            constexpr int V = V_;
            constexpr int CHUNKS = 64;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint idx = row * uint(CHUNKS) + lid;

            const float v_raw = p_raw[idx];
            const int v_idx = p_idx[idx];
            const float v_sc = p_sc[idx];
            const float v_sum = p_sum[idx];

            const float sg_raw = simd_max(v_raw);
            const int sg_idx = simd_min((v_raw == sg_raw) ? v_idx : V);
            const float sg_sc = simd_max(v_sc);
            const float sg_sum = simd_sum(
                (v_sum > 0.0f) ? (v_sum * metal::exp(v_sc - sg_sc)) : 0.0f);

            threadgroup float sh_raw[2];
            threadgroup int sh_idx[2];
            threadgroup float sh_sc[2];
            threadgroup float sh_sum[2];
            if (lane == 0) {
              sh_raw[sg] = sg_raw;
              sh_idx[sg] = sg_idx;
              sh_sc[sg] = sg_sc;
              sh_sum[sg] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float f_raw = (lane < 2) ? sh_raw[lane] : -INFINITY;
              const int f_idx = (lane < 2) ? sh_idx[lane] : V;
              const float f_sc = (lane < 2) ? sh_sc[lane] : -INFINITY;
              const float f_sum = (lane < 2) ? sh_sum[lane] : 0.0f;

              const float row_raw = simd_max(f_raw);
              const int row_idx = simd_min((f_raw == row_raw) ? f_idx : V);
              const float row_sc = simd_max(f_sc);
              const float row_sum = simd_sum(
                  (f_sum > 0.0f) ? (f_sum * metal::exp(f_sc - row_sc)) : 0.0f);

              if (lane == 0) {
                tokens[row] = row_idx;
                confidence[row] = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
              }
            }
            """)

    private static func fusedArgMaxAndConfidence(
        _ logits: MLXArray, outputMultiplier: Float, softcap: Float
    ) -> (tokens: MLXArray, confidence: MLXArray) {
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        let chunks = 64
        let params = MLXArray([outputMultiplier, softcap])
        let partials = argMaxConfStage1(
            [logits, params],
            template: [("V_", vocab)],
            grid: (256 * chunks, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, chunks], [rows, chunks], [rows, chunks], [rows, chunks]],
            outputDTypes: [.float32, .int32, .float32, .float32])
        let out = argMaxConfStage2(
            partials,
            template: [("V_", vocab)],
            grid: (chunks, rows, 1), threadGroup: (chunks, 1, 1),
            outputShapes: [[rows], [rows]],
            outputDTypes: [.int32, .float32])
        return (out[0], out[1])
    }

    private static let fastArgMaxStage1 = MLXFast.metalKernel(
        name: "dflash_fast_argmax_s1",
        inputNames: ["logits"],
        outputNames: ["p_val", "p_idx"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint chunk = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint vocab = uint(V_);
            const uint chunk_size = (vocab + CHUNKS - 1) / CHUNKS;
            const uint start = chunk * chunk_size;
            const uint end = min(vocab, start + chunk_size);
            const device bfloat16_t* row_ptr = logits + ulong(row) * ulong(vocab);

            float best_val = -INFINITY;
            int best_idx = int(vocab);
            const uint start4 = (start + 3u) & ~3u;
            const uint end4 = end & ~3u;
            if (lid == 0) {
              for (uint i = start; i < min(start4, end); i++) {
                const float v = float(row_ptr[i]);
                if (v > best_val || (v == best_val && int(i) < best_idx)) {
                  best_val = v;
                  best_idx = int(i);
                }
              }
            }
            if (end4 > start4) {
              const device vec<bfloat16_t, 4>* src4 =
                  (const device vec<bfloat16_t, 4>*)(row_ptr + start4);
              const uint count4 = (end4 - start4) >> 2;
              for (uint j = lid; j < count4; j += 256u) {
                const vec<bfloat16_t, 4> vv = src4[j];
                const uint base_i = start4 + (j << 2);
                #pragma unroll
                for (uint k = 0; k < 4; k++) {
                  const float v = float(vv[k]);
                  const int idx = int(base_i + k);
                  if (v > best_val || (v == best_val && idx < best_idx)) {
                    best_val = v;
                    best_idx = idx;
                  }
                }
              }
            }
            if (lid == 0) {
              const uint tail_start = max(start4, end4);
              for (uint i = tail_start; i < end; i++) {
                const float v = float(row_ptr[i]);
                if (v > best_val || (v == best_val && int(i) < best_idx)) {
                  best_val = v;
                  best_idx = int(i);
                }
              }
            }

            const float sg_val = simd_max(best_val);
            const int sg_idx = simd_min((best_val == sg_val) ? best_idx : int(vocab));
            threadgroup float sh_val[8];
            threadgroup int sh_idx[8];
            if (lane == 0) {
              sh_val[sg] = sg_val;
              sh_idx[sg] = sg_idx;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float v = (lane < 8) ? sh_val[lane] : -INFINITY;
              const int idx = (lane < 8) ? sh_idx[lane] : int(vocab);
              const float tg_val = simd_max(v);
              const int tg_idx = simd_min((v == tg_val) ? idx : int(vocab));
              if (lane == 0) {
                const uint out_i = row * CHUNKS + chunk;
                p_val[out_i] = tg_val;
                p_idx[out_i] = tg_idx;
              }
            }
            """)

    private static let fastArgMaxStage2 = MLXFast.metalKernel(
        name: "dflash_fast_argmax_s2",
        inputNames: ["p_val", "p_idx"],
        outputNames: ["tokens"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint vocab = uint(V_);
            const uint idx = row * CHUNKS + lid;
            const float v = p_val[idx];
            const int id = p_idx[idx];
            const float sg_val = simd_max(v);
            const int sg_idx = simd_min((v == sg_val) ? id : int(vocab));
            threadgroup float sh_val[2];
            threadgroup int sh_idx[2];
            if (lane == 0) {
              sh_val[sg] = sg_val;
              sh_idx[sg] = sg_idx;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float fv = (lane < 2) ? sh_val[lane] : -INFINITY;
              const int fid = (lane < 2) ? sh_idx[lane] : int(vocab);
              const float rv = simd_max(fv);
              const int rid = simd_min((fv == rv) ? fid : int(vocab));
              if (lane == 0) {
                tokens[row] = rid;
              }
            }
            """)

    private static func fastArgMax(_ logits: MLXArray) -> MLXArray {
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        let chunks = 64
        let partials = fastArgMaxStage1(
            [logits],
            template: [("V_", vocab)],
            grid: (256 * chunks, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, chunks], [rows, chunks]],
            outputDTypes: [.float32, .int32])
        return fastArgMaxStage2(
            partials,
            template: [("V_", vocab)],
            grid: (chunks, rows, 1), threadGroup: (chunks, 1, 1),
            outputShapes: [[rows]],
            outputDTypes: [.int32])[0]
    }

    /// The argmax of the head products, or a selection's proposals over the
    /// target's readout of them when it gives one per row.
    private func proposals(_ logits: MLXArray, selection: CBv2MTPProposalSelection?) -> MLXArray {
        if let selection, let coupled = selection.propose(target.dflashReadout(logits)),
            coupled.shape == [logits.dim(0)]
        {
            return coupled.asType(.int32)
        }
        if logits.ndim == 2, logits.dtype == .bfloat16, logits.dim(1) >= 256 {
            return Self.fastArgMax(logits)
        }
        return argMax(logits, axis: -1).asType(.int32)
    }
    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return [] }
        return s.arrays
    }

    /// Ends a request whose target rows or round calls did not fit.
    public func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { throw MuseGlimmerDFlashError.state("foreign or released") }
        if let refusal = s.refusal { throw MuseGlimmerDFlashError.state(refusal) }
    }

    /// The accepted drafts' trusted target rows become context. The row of
    /// the last committed position follows as the next carry.
    public func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        let s = state(requestState)
        defer { clearRound(s) }
        let speculation = s.speculation
        s.speculation = nil
        s.lastRoundFullyAccepted = false
        let pendingBefore = s.pending
        guard s.refusal == nil else { return }
        let rows = confirmedInputTokens - 1
        guard s.stagedInputCount > 0, rows >= 0, rows <= s.stagedInputCount,
            committedTargetHidden.ndim == 3, committedTargetHidden.dim(1) == rows
        else {
            s.refusal = "finalize"
            return
        }
        if rows > 0 {
            guard let accepted = features(committedTargetHidden) else {
                s.refusal = "accepted rows"
                return
            }
            if settings.batchesRequests {
                // Committed with the next round's carry, in its pass.
                s.pending = s.pending.map { concatenated([$0, accepted], axis: 0) } ?? accepted
            } else {
                model.commit(features: accepted, at: s.context.nextPosition, to: s.context)
            }
        }
        s.awaitsCarry = true
        s.committedInputCount += rows
        s.lastRoundFullyAccepted = rows == s.stagedInputCount
        // Keep the speculation for this outcome: the pending rows exactly
        // this round's first `rows` verify rows (none when `rows` is 0).
        if var kept = speculation, settings.batchesRequests, pendingBefore == nil,
            kept.depth == s.stagedInputCount, rows <= kept.depth,
            (rows == 0) == (s.pending == nil)
        {
            kept.accepted = rows
            kept.pending = s.pending
            s.speculation = kept
        }
    }

    /// Only the memo belongs to a round: a block never writes the context.
    public func discardRound(requestState: any CBv2MTPRequestState) {
        let s = state(requestState)
        s.speculation = nil
        clearRound(s)
    }

    private func clearRound(_ s: State) {
        s.memo = nil
        s.selection = nil
        s.prestarted = false
        s.stagedInputCount = 0
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return }
        clearRound(s)
        s.speculation = nil
        s.pending = nil
        s.context.release()
        s.awaitsCarry = false
        s.committedInputCount = 0
        s.released = true
    }
}

// MARK: - Admission

extension MuseGlimmerDFlashDrafter: CBv2MTPBoundedAllocationProviding {
    /// Explicit arrays of one context commit of `rows` rows: the feature rows
    /// in the drafter dtype, the projected context before and after its norm,
    /// and each layer's keys (projected, normed, rotated) and values.
    func commitBytes(rows: Int) -> Int {
        let c = model.configuration
        let size = model.activationDType.size
        let keyWidth = c.kvHeads * c.headDimensions
        return rows * size * (c.contextWidth + 2 * c.hiddenSize + c.hiddenLayers * 4 * keyWidth)
    }

    /// Explicit arrays of one block forward and its head rows. Per layer: the
    /// keys and values handed to attention (held rows and the block), float32
    /// scores, queries, the attention output and the MLP's three products.
    /// Then float32 head rows and their argmax, and the coupled race when
    /// it is on.
    var blockBytes: Int {
        let c = model.configuration
        let size = model.activationDType.size
        let keys = c.slidingWindow + c.blockSize
        let layer =
            2 * c.kvHeads * keys * c.headDimensions * size
            + c.attentionHeads * c.blockSize * keys * 4
            + c.blockSize * size
            * (3 * c.attentionHeads * c.headDimensions + 6 * c.hiddenSize + 3 * c.intermediateSize)
        let head =
            settings.maximumDraftTokens * (target.dflashVocabularySize * 4 + c.hiddenSize * 4)
        return c.hiddenLayers * layer + head + coupledProposalBytes
            + c.blockSize * (keys + c.hiddenSize * size)
    }

    /// The coupled race over the head rows (`couplesSampledProposals`), all
    /// float32 at worst: the readout's four steps (multiplier, scale, tanh,
    /// scale back) and the race's seven arrays (cast, temperature, softmax,
    /// noise, their ratio, the above-winner mask and its masked mass).
    var coupledProposalBytes: Int {
        settings.couplesSampledProposals
            ? 11 * settings.maximumDraftTokens * target.dflashVocabularySize * 4 : 0
    }

    /// A round's accepted rows held for the next round's pass
    /// (`batchesRequests`): a view of the round's target rows, which can keep
    /// every row of that round alive, charged at float32.
    var pendingBytes: Int {
        settings.batchesRequests
            ? settings.maximumSpeculativeRows * settings.maximumDraftTokens
                * model.configuration.contextWidth * 4 : 0
    }

    /// Per request: the ring's two buffers, one copy of each for a write MLX
    /// cannot make in place, and one commit of at most `window` rows (older
    /// rows of a longer chunk are never projected), plus the pending rows
    /// when requests are batched. Once per engine: one block forward for
    /// each request a round may hold. The ring does not grow with the
    /// request, so nothing is charged per token.
    public func boundedRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        let c = model.configuration
        guard limits.maximumPrefillTokens > 0, limits.maximumDraftTokens > 0 else { return nil }
        let buffer =
            MuseGlimmerDFlashContextState.residentBytes(
                configuration: c, dtype: model.activationDType) / 2
        let commitRows = min(max(limits.maximumPrefillTokens, c.blockSize), c.slidingWindow)
        // A speculative block can still be in flight when its round's
        // normal block is built.
        var owner = [
            CBv2MTPFixedBufferSpec(
                logicalBytes: blockBytes,
                allocationCount: settings.maximumSpeculativeRows + (speculatesAfterVerify ? 1 : 0))
        ]
        if speculatesAfterVerify {
            // The speculation's float32 readout of its head products and one
            // elementwise temporary, for the draft confidence.
            owner.append(
                .init(
                    logicalBytes: settings.maximumDraftTokens * target.dflashVocabularySize * 4,
                    allocationCount: 2))
        }
        if settings.backendScratchBytes > 0 {
            owner.append(.init(logicalBytes: settings.backendScratchBytes))
        }
        var resident = [CBv2MTPFixedBufferSpec(logicalBytes: buffer, allocationCount: 2)]
        if pendingBytes > 0 { resident.append(.init(logicalBytes: pendingBytes)) }
        return CBv2MTPBoundedAllocationSpec(
            resident: resident,
            // The speculation's copy of the ring (`speculateNextRound`) is a
            // second pair of copies next to an in-flight commit's.
            working: [
                .init(logicalBytes: buffer, allocationCount: speculatesAfterVerify ? 4 : 2),
                .init(logicalBytes: commitBytes(rows: commitRows)),
            ],
            hostBytes: c.blockSize * (c.slidingWindow + c.blockSize),
            ownerWorking: owner)
    }
}
