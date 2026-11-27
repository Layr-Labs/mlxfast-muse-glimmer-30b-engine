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
    /// Low-acceptance rounds' MLP: 3-bit group-32 copies of each layer's pinned 5-bit
    /// gate, up and down (`CBV2_DRAFTER_LOW_MLP_BITS`, 5 disables). Empty
    /// when disabled or when a projection is not the pinned affine form.
    private let lowMLP: [(gate: QuantizedLinear, up: QuantizedLinear, down: QuantizedLinear)]
    /// The target's NVFP4 `lm_head` when it has one. Draft rows take MLX's
    /// own few-row NVFP4 product of it (the NAX `qmv` route on an M5) rather
    /// than the target's row-exact verify kernel: the same E2M1 x E4M3
    /// weights, FP32 accumulation and `weight_scale_2`, in another order.
    /// Proposals only decide acceptance; the target verifies every token.
    private let draftHead: ModelOptWeightOnlyNVFP4Linear?

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
        lowMLP = Self.lowAcceptanceMLP(model)
        contextKeyEpilogue = ContextKeyEpilogue(model)
        if CBv2SameWidthTree.enabled, model.activationDType == .bfloat16 {
            // Same-width tree: the fused argmax + top-4 kernel and the
            // tree-positioned context keys, checked bit for bit at load.
            let textTarget = target as? MuseGlimmerTextModel
            let epilogue = contextKeyEpilogue
            let ok =
                (try? MLX.withError { () -> Bool in
                    guard let textTarget,
                        Self.top4SelfTest(
                            vocabulary: target.dflashVocabularySize,
                            outputMultiplier: textTarget.configuration.outputMultiplier,
                            softcap: textTarget.configuration.finalLogitSoftcapping)
                    else { return false }
                    guard let epilogue else { return true }
                    let rows = 6
                    let width = epilogue.kvHeads * 128
                    let projections = (0 ..< 5).map {
                        MLXRandom.normal([rows, width], key: MLXRandom.key(UInt64(50 + $0)))
                            .asType(.bfloat16)
                    }
                    let record = epilogue.apply(projections, rows: rows, position: 333)
                    let identity = epilogue.apply(
                        projections, rows: rows, position: 333,
                        rowPositions: MLXArray((0 ..< rows).map { Int32($0) }))
                    let same = all(record.view(dtype: .uint16) .== identity.view(dtype: .uint16))
                    eval(same)
                    return same.item(Bool.self)
                }) ?? false
            if !ok {
                CBv2VerifyTree.kernelSelfTestFailed = true
                FileHandle.standardError.write(
                    Data("[cbv2-tree] top-4 / context-key self-test failed; tree off\n".utf8))
            }
        }
        if CBv2VerifyTree.enabled, model.activationDType == .bfloat16,
            !((try? MLX.withError { () -> Bool in
                Self.top3SelfTest(vocabulary: target.dflashVocabularySize)
            }) ?? false)
        {
            CBv2VerifyTree.kernelSelfTestFailed = true
            FileHandle.standardError.write(Data("[cbv2-tree] top-3 self-test failed; leaves off\n".utf8))
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
            if let epilogue = contextKeyEpilogue {
                let key = MLXArray.zeros([1, c.kvHeads * c.headDimensions], dtype: .bfloat16)
                warm.append(epilogue.apply(Array(repeating: key, count: 5), rows: 1, position: 0))
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
        /// `CBV2_TREE_TRACE` only: this round's block top entries, lazy.
        var trace: ProposalTraceArrays?
        /// `CBV2_VERIFY_TREE` only: this round's top-3 ids per block row
        /// `[maximumDraftTokens, 3]` int32 (column 0 equals the proposals).
        var memoTop: MLXArray?
        /// Same-width tree: this round's top-4 ids and raw head products per
        /// block row `[maximumDraftTokens, 4]` (lazy; `cbv2_tree_select` reads them).
        var memoTop4: (ids: MLXArray, raw: MLXArray)?
        /// Same-width tree: the verify columns' depths for the next
        /// `speculateNextRound` (set by the engine, consumed there).
        var speculationRowPositions: MLXArray?

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
        /// `CBV2_TREE_TRACE` only: the speculated block's top entries.
        var trace: ProposalTraceArrays? = nil
        /// `CBV2_VERIFY_TREE` only: the speculated block's top-3 ids.
        var memoTop: MLXArray? = nil
        /// Same-width tree: the speculated block's top-4.
        var memoTop4: (ids: MLXArray, raw: MLXArray)? = nil
        /// Same-width tree: `keys` were rotated at each verify row's depth, so a
        /// leaf outcome can append rows `0 ... a` and the leaf's row (`rowMap`).
        var treePositioned = false
        var rowMap: [Int]? = nil
    }

    /// `CBV2_TREE_TRACE` only: a block's top `CBv2TreeTrace.width` entries
    /// per row, lazy; read back once, at the round's finalize.
    struct ProposalTraceArrays {
        /// `[rows, width]` int32, unordered within a row.
        let ids: MLXArray
        /// `[rows, width]` float32: the head product the proposals rank.
        let raw: MLXArray
        /// `[rows, width]` float32: the same entries at the readout scale.
        let scores: MLXArray
        /// `[rows]` float32: log-sum-exp of the readout scores.
        let logSumExp: MLXArray
        let source: String
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
        if CBv2DraftRegime.treeProposals, CBv2SameWidthTree.enabled, s.selection == nil,
            let textTarget = target as? MuseGlimmerTextModel,
            let out = Self.fusedArgMaxConfTop4(
                logits, outputMultiplier: textTarget.configuration.outputMultiplier,
                softcap: textTarget.configuration.finalLogitSoftcapping)
        {
            // Same-width tree round: the argmax (the record's total order) and
            // the top 4 for `cbv2_tree_select` in one pass.
            s.memo = (out.tokens, rows)
            s.memoTop4 = (out.ids, out.raw)
            s.memoTop = out.ids[0..., 0 ..< 3]
        } else if CBv2DraftRegime.treeProposals, s.selection == nil,
            let top = Self.fusedTop3(logits)
        {
            // Draft tree round: one pass gives the argmax (column 0, the same
            // total order as `fastArgMax`) and the siblings.
            s.memo = (top[0..., 0], rows)
            s.memoTop = top
        } else {
            s.memo = (proposals(logits, selection: s.selection), rows)
        }
        if CBv2TreeTrace.tracesProposals {
            s.trace = proposalTraceArrays(
                logits, unscaled: headProductIsUnscaled(rows, readout: s.selection != nil),
                source: "start")
        }
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
        let readout = states.contains { $0.selection != nil }
        let logits = draftHeadProduct(
            rows.count == 1 ? rows[0] : concatenated(rows, axis: 0),
            readout: readout)
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
            if CBv2TreeTrace.tracesProposals {
                s.trace = proposalTraceArrays(
                    logits[(index * k) ..< ((index + 1) * k), 0...],
                    unscaled: headProductIsUnscaled(rows[index], readout: readout),
                    source: "batched")
            }
        }
    }

    // MARK: Post-verify speculation

    /// The served round shape the speculation reproduces: batched requests,
    /// argmax proposals.
    var speculatesAfterVerify: Bool {
        settings.batchesRequests && !settings.couplesSampledProposals
    }

    private let contextKeyEpilogue: ContextKeyEpilogue?

    /// The record kernel's source (unchanged text; the tree variant below
    /// replaces one expression).
    private static let contextKeysSource = """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint rows = uint(dims[0]);
            const uint row = group % rows;
            const uint head = (group / rows) % HEADS_;
            const uint layer = group / (rows * HEADS_);
            const device bfloat16_t* src;
            const device bfloat16_t* weight;
            switch (layer) {
                case 0: src = k0; weight = w0; break;
                case 1: src = k1; weight = w1; break;
                case 2: src = k2; weight = w2; break;
                case 3: src = k3; weight = w3; break;
                default: src = k4; weight = w4; break;
            }
            src += (size_t(row) * HEADS_ + head) * 128 + lane * 4;
            weight += lane * 4;
            device bfloat16_t* dst = keys + size_t(group) * 128 + lane * 4;

            // Match rms_single_row: four adjacent channels per lane, then two SIMD sums.
            float x[4];
            float sum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                x[i] = src[i];
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            sum = simd_sum(lane == 0 ? sum : 0.0f);
            const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
            const bool low = lane < 16;
            const float pos = float(dims[1] + int(row));
            for (uint i = 0; i < 4; ++i) {
                const bfloat16_t norm = weight[i] * static_cast<bfloat16_t>(x[i] * inv);
                const float v = static_cast<float>(norm);
                const float other = simd_shuffle_xor(v, ushort(16));
                const float d = float((lane % 16) * 4 + i) / 64.0f;
                const float theta = pos * metal::exp2(-d * params[1]);
                const float c = metal::fast::cos(theta);
                const float s = metal::fast::sin(theta);
                const float x1 = low ? v : other;
                const float x2 = low ? other : v;
                dst[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
            }
            """

    private static let contextKeysKernel = MLXFast.metalKernel(
        name: "dflash_context_keys",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "w0", "w1", "w2", "w3", "w4", "dims", "params",
        ],
        outputNames: ["keys"],
        source: contextKeysSource)

    /// Same-width tree speculation: each verify row's key rotated at
    /// `position + rowpos[row]` (its tree depth) instead of `position + row`;
    /// the same float expression for the same integer position.
    private static let contextKeysTreeKernel: MLXFast.MLXFastKernel = {
        let record = "const float pos = float(dims[1] + int(row));"
        let tree = "const float pos = float(dims[1] + rowpos[row]);"
        precondition(contextKeysSource.components(separatedBy: record).count == 2)
        return MLXFast.metalKernel(
            name: "dflash_context_keys_rowpos",
            inputNames: [
                "k0", "k1", "k2", "k3", "k4", "w0", "w1", "w2", "w3", "w4", "dims", "params",
                "rowpos",
            ],
            outputNames: ["keys"],
            source: contextKeysSource.replacingOccurrences(of: record, with: tree))
    }()

    private struct ContextKeyEpilogue {
        let kvHeads: Int
        let weights: [MLXArray]
        let coefficients: MLXArray

        init?(_ model: MuseGlimmerDFlashDraftModel) {
            let c = model.configuration
            guard model.layers.count == 5, c.headDimensions == 128,
                model.activationDType == .bfloat16, Device.defaultDevice().deviceType == .gpu
            else { return nil }
            var weights: [MLXArray] = []
            weights.reserveCapacity(5)
            for layer in model.layers {
                let norm = layer.attention.kNorm
                guard norm.weight.dtype == .bfloat16, norm.weight.shape == [128],
                    norm.eps == c.rmsNormEps
                else { return nil }
                weights.append(norm.weight)
            }
            self.kvHeads = c.kvHeads
            self.weights = weights
            self.coefficients = MLXArray([c.rmsNormEps, log2(c.ropeTheta)])
        }

        func apply(_ projections: [MLXArray], rows: Int, position: Int) -> MLXArray {
            var inputs = projections
            inputs.reserveCapacity(12)
            inputs.append(contentsOf: weights)
            inputs.append(MLXArray([Int32(rows), Int32(position)]))
            inputs.append(coefficients)
            return MuseGlimmerDFlashDrafter.contextKeysKernel(
                inputs, template: [("HEADS_", kvHeads)],
                grid: (32, 5 * kvHeads * rows, 1), threadGroup: (32, 1, 1),
                outputShapes: [[5, kvHeads, rows, 128]], outputDTypes: [.bfloat16])[0]
        }

        /// `apply` with row `i` rotated at `position + rowPositions[i]`.
        func apply(_ projections: [MLXArray], rows: Int, position: Int, rowPositions: MLXArray)
            -> MLXArray
        {
            var inputs = projections
            inputs.reserveCapacity(13)
            inputs.append(contentsOf: weights)
            inputs.append(MLXArray([Int32(rows), Int32(position)]))
            inputs.append(coefficients)
            inputs.append(rowPositions.asType(.int32))
            return MuseGlimmerDFlashDrafter.contextKeysTreeKernel(
                inputs, template: [("HEADS_", kvHeads)],
                grid: (32, 5 * kvHeads * rows, 1), threadGroup: (32, 1, 1),
                outputShapes: [[5, kvHeads, rows, 128]], outputDTypes: [.bfloat16])[0]
        }
    }

    private func speculativeContextKeyValues(
        _ context: MLXArray, at position: Int, rowPositions: MLXArray? = nil
    ) -> (keys: MLXArray, values: MLXArray)? {
        let c = model.configuration
        let rows = context.dim(0)
        guard let epilogue = contextKeyEpilogue, context.dtype == .bfloat16, rows > 0,
            position >= 0, position <= Int(Int32.max) - rows
        else {
            // Tree positions need the fused epilogue.
            return rowPositions == nil ? model.contextKeyValues(context, at: position) : nil
        }
        let shape = [1, rows, c.kvHeads, c.headDimensions]
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        keys.reserveCapacity(model.layers.count)
        values.reserveCapacity(model.layers.count)
        for layer in model.layers {
            keys.append(layer.attention.kProj(context))
            values.append(layer.attention.vProj(context).reshaped(shape).transposed(0, 2, 1, 3))
        }
        guard keys.allSatisfy({ $0.dtype == .bfloat16 }) else {
            return rowPositions == nil ? model.contextKeyValues(context, at: position) : nil
        }
        if let rowPositions {
            guard rowPositions.size == rows else { return nil }
            return (
                epilogue.apply(keys, rows: rows, position: position, rowPositions: rowPositions),
                concatenated(values, axis: 0))
        }
        return (
            epilogue.apply(keys, rows: rows, position: position),
            concatenated(values, axis: 0))
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
        // Same-width tree positions are for this call only.
        let rowPositions = s.speculationRowPositions
        s.speculationRowPositions = nil
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
        // Same-width tree round: each verify row's key at its tree depth, so
        // any outcome (chain or leaf) can append the rows it committed.
        guard
            let projected = speculativeContextKeyValues(
                model.projectContext(features), at: start, rowPositions: rowPositions)
        else { return [] }
        var speculation = Speculation(
            depth: k, anchor: start, first: context.firstPosition, keys: projected.keys,
            values: projected.values)
        speculation.treePositioned = rowPositions != nil
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
        if CBv2TreeTrace.tracesProposals {
            speculation.trace = proposalTraceArrays(
                logits, unscaled: headProductIsUnscaled(rows, readout: false), source: "spec")
        }
        let memo: (tokens: MLXArray, hidden: MLXArray)
        let confidence: MLXArray
        if CBv2DraftRegime.treeProposals, CBv2SameWidthTree.enabled,
            let textTarget = target as? MuseGlimmerTextModel,
            let out = Self.fusedArgMaxConfTop4(
                logits, outputMultiplier: textTarget.configuration.outputMultiplier,
                softcap: textTarget.configuration.finalLogitSoftcapping)
        {
            memo = (tokens: out.tokens, hidden: rows)
            confidence = out.confidence
            speculation.memoTop4 = (out.ids, out.raw)
            speculation.memoTop = out.ids[0..., 0 ..< 3]
        } else if let textTarget = target as? MuseGlimmerTextModel, logits.dtype == .bfloat16 {
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
        if CBv2DraftRegime.treeProposals, speculation.memoTop == nil {
            speculation.memoTop = Self.fusedTop3(logits)
        }
        s.speculation = speculation
        return [memo.tokens, memo.hidden, projected.keys, projected.values, confidence]
            + (speculation.memoTop.map { [$0] } ?? [])
            + (speculation.memoTop4.map { [$0.ids, $0.raw] } ?? [])
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
        if let rowMap = speculation.rowMap {
            // Same-width tree leaf outcome: rows 0 ... a and the leaf's row,
            // each already rotated at its tree depth.
            let rows = MLXArray(rowMap.map { Int32($0) })
            s.context.append(
                keys: take(speculation.keys, rows, axis: 2),
                values: take(speculation.values, rows, axis: 2))
        } else if accepted == speculation.depth {
            s.context.append(keys: speculation.keys, values: speculation.values)
        } else {
            let rows = 0 ..< (accepted + 1)
            s.context.append(
                keys: speculation.keys[0..., 0..., rows, 0...],
                values: speculation.values[0..., 0..., rows, 0...])
        }
        guard let memo else { return .committed }
        s.memo = memo
        s.trace = speculation.trace
        s.memoTop = speculation.memoTop
        s.memoTop4 = speculation.memoTop4
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

    /// Whether `draftHeadProduct(rows, readout:)` skipped `weight_scale_2`.
    private func headProductIsUnscaled(_ rows: MLXArray, readout: Bool) -> Bool {
        guard !readout, let draftHead else { return false }
        return rows.dim(-1) == draftHead.inputDimensions
    }

    /// `CBV2_TREE_TRACE` only: each row's top entries of a block's head
    /// products `[rows, vocabulary]`, lazy. Off the proposal path: nothing
    /// that proposes, verifies or commits reads these arrays.
    private func proposalTraceArrays(_ logits: MLXArray, unscaled: Bool, source: String)
        -> ProposalTraceArrays?
    {
        let width = CBv2TreeTrace.width
        guard CBv2TreeTrace.tracesProposals, logits.ndim == 2, logits.dim(1) > width else {
            return nil
        }
        let raw = logits.asType(.float32)
        let top = argPartition(-raw, kth: width - 1, axis: -1)[0..., 0 ..< width]
        var product = logits
        if unscaled, let draftHead { product = draftHead.scaled(logits) }
        let scores = target.dflashReadout(product).asType(.float32)
        return ProposalTraceArrays(
            ids: top.asType(.int32), raw: takeAlong(raw, top, axis: -1),
            scores: takeAlong(scores, top, axis: -1), logSumExp: scores.logSumExp(axis: -1),
            source: source)
    }

    /// Layer `index`'s MLP: the 3-bit copy while a low-acceptance round
    /// drafts (`CBv2DraftRegime.lowAcceptance`), the pinned one otherwise.
    private func draftMLP(_ index: Int, _ mlp: MuseGlimmerDFlashMLP, _ x: MLXArray) -> MLXArray {
        if CBv2DraftRegime.lowAcceptance, index < lowMLP.count {
            let low = lowMLP[index]
            return low.down(compiledSiluProduct(low.gate(x), low.up(x)))
        }
        return mlp.down(compiledSiluProduct(mlp.gate(x), mlp.up(x)))
    }

    private static func lowAcceptanceMLP(_ model: MuseGlimmerDFlashDraftModel)
        -> [(gate: QuantizedLinear, up: QuantizedLinear, down: QuantizedLinear)]
    {
        let bits = ProcessInfo.processInfo.environment["CBV2_DRAFTER_LOW_MLP_BITS"]
            .flatMap { Int($0) }.map { min(max($0, 2), 5) } ?? 3
        guard bits < 5 else { return [] }
        func low(_ linear: Linear) -> QuantizedLinear? {
            guard let ql = linear as? QuantizedLinear, ql.mode == .affine, ql.bits > bits else {
                return nil
            }
            let dense = MLX.dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
            let copy = QuantizedLinear(
                weight: dense, bias: ql.bias, groupSize: 32, bits: bits, mode: .affine)
            eval([copy.weight, copy.scales] + (copy.biases.map { [$0] } ?? []))
            return copy
        }
        var copies: [(gate: QuantizedLinear, up: QuantizedLinear, down: QuantizedLinear)] = []
        for layer in model.layers {
            guard let gate = low(layer.mlp.gate), let up = low(layer.mlp.up),
                let down = low(layer.mlp.down)
            else { return [] }
            copies.append((gate, up, down))
        }
        // Compile the copies' kernels at load for the block's row counts
        // (16 rows; 7 active rows in the last layer), not in a timed round.
        if let first = copies.first {
            let width = model.configuration.hiddenSize
            var warm = [MLXArray]()
            for rows in [model.configuration.blockSize, 7] {
                let x = MLXArray.zeros([1, rows, width], dtype: model.activationDType)
                warm.append(first.down(compiledSiluProduct(first.gate(x), first.up(x))))
            }
            eval(warm)
        }
        return copies
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
            guard let ql = m as? QuantizedLinear, ql.bits == 8, ql.mode == .affine else {
                return nil
            }
            let dense = MLX.dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
            let replacement = QuantizedLinear(
                weight: dense, bias: ql.bias, groupSize: 32, bits: 4, mode: .affine)
            if let biases = replacement.biases {
                eval(replacement.weight, replacement.scales, biases)
            } else {
                eval(replacement.weight, replacement.scales)
            }
            return replacement
        }
        model.update(modules: updates)
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
            var keys = MLXFast.RoPE(
                attention.kNorm(
                    attention.kProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                        .transposed(0, 2, 1, 3)),
                dimensions: c.headDimensions, traditional: false,
                base: c.ropeTheta, scale: 1, offset: position)
            var values = attention.vProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                .transposed(0, 2, 1, 3)
            if let context {
                keys = concatenated([context.keys[index ..< index + 1], keys], axis: 2)
                values = concatenated([context.values[index ..< index + 1], values], axis: 2)
            }
            if index == lastIndex {
                let active = 1 ..< (1 + draftCount)
                let xActive = x[0..., active, 0...]
                let queries = MLXFast.RoPE(
                    attention.qNorm(
                        attention.qProj(xActive).reshaped([
                            1, draftCount, c.attentionHeads, c.headDimensions,
                        ]).transposed(0, 2, 1, 3)),
                    dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: position + 1)
                let activeMask = mask.map { $0[active, 0...] }
                let attended = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: activeMask)
                var h = hidden[0..., active, 0...]
                    + attention.oProj(
                        attended.transposed(0, 2, 1, 3).reshaped([
                            1, draftCount, c.attentionHeads * c.headDimensions,
                        ]))
                let mx = layer.postAttentionNorm(h)
                h = h + draftMLP(index, layer.mlp, mx)
                return model.norm(h).reshaped([draftCount, c.hiddenSize])
            }
            let queries = MLXFast.RoPE(
                attention.qNorm(
                    attention.qProj(x).reshaped([1, block, c.attentionHeads, c.headDimensions])
                        .transposed(0, 2, 1, 3)),
                dimensions: c.headDimensions, traditional: false,
                base: c.ropeTheta, scale: 1, offset: position)
            let attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
            hidden =
                hidden
                + attention.oProj(
                    attended.transposed(0, 2, 1, 3).reshaped([
                        1, block, c.attentionHeads * c.headDimensions,
                    ]))
            let mx = layer.postAttentionNorm(hidden)
            hidden =
                hidden
                + draftMLP(index, layer.mlp, mx)
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

    // MARK: Same-width tree: argmax + confidence + top-4 in one pass

    /// `argMaxConfStage1` with each thread also keeping its top 4 raw values
    /// (the argmax's total order: larger first, ties to the lower id, NaN never
    /// enters); four pop rounds merge the threadgroup's lists. The argmax and
    /// the softcapped sum are the record kernel's expressions.
    private static let argMaxConfTop4Stage1 = MLXFast.metalKernel(
        name: "dflash_argmax_conf_top4_s1",
        inputNames: ["logits", "params"],
        outputNames: ["p_sc", "p_sum", "p_raw4", "p_idx4"],
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

            float lv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
            int li[4] = {V, V, V, V};
            float best_sc = -INFINITY;
            float sum_exp = 0.0f;
            for (int i = start + int(lid); i < end; i += 256) {
              const bfloat16_t raw_bf = row_ptr[i];
              const float raw_f = float(raw_bf);
              if (raw_f > lv[3] || (raw_f == lv[3] && i < li[3])) {
                int at = 3;
                while (at > 0 && (raw_f > lv[at - 1] || (raw_f == lv[at - 1] && i < li[at - 1]))) {
                  lv[at] = lv[at - 1];
                  li[at] = li[at - 1];
                  at -= 1;
                }
                lv[at] = raw_f;
                li[at] = i;
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

            const float sg_sc = simd_max(best_sc);
            const float sg_sum = simd_sum(
                (sum_exp > 0.0f) ? (sum_exp * metal::exp(best_sc - sg_sc)) : 0.0f);
            threadgroup float sh_sc[8];
            threadgroup float sh_sum[8];
            threadgroup float sh_val[8];
            threadgroup int sh_idx[8];
            if (lane == 0) {
              sh_sc[sg] = sg_sc;
              sh_sum[sg] = sg_sum;
            }
            int ptr = 0;
            for (uint k = 0; k < 4; k++) {
              const float hv = ptr < 4 ? lv[ptr] : -INFINITY;
              const int hi = ptr < 4 ? li[ptr] : V;
              const float s_val = simd_max(hv);
              const int s_idx = simd_min((hv == s_val) ? hi : V);
              if (lane == 0) {
                sh_val[sg] = s_val;
                sh_idx[sg] = s_idx;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              const float v8 = (lane < 8) ? sh_val[lane] : -INFINITY;
              const int i8 = (lane < 8) ? sh_idx[lane] : V;
              const float t_val = simd_max(v8);
              const int t_idx = simd_min((v8 == t_val) ? i8 : V);
              if (lid == 0) {
                const uint o = (row * uint(CHUNKS) + chunk) * 4 + k;
                p_raw4[o] = t_val;
                p_idx4[o] = t_idx;
              }
              if (ptr < 4 && hv == t_val && hi == t_idx) ptr++;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (sg == 0) {
              const float v_sc = (lane < 8) ? sh_sc[lane] : -INFINITY;
              const float v_sum = (lane < 8) ? sh_sum[lane] : 0.0f;
              const float tg_sc = simd_max(v_sc);
              const float tg_sum = simd_sum(
                  (v_sum > 0.0f) ? (v_sum * metal::exp(v_sc - tg_sc)) : 0.0f);
              if (lane == 0) {
                const uint out_i = row * uint(CHUNKS) + chunk;
                p_sc[out_i] = tg_sc;
                p_sum[out_i] = tg_sum;
              }
            }
            """)

    /// Stage 2: per row, the 64 chunks' top 4 merged (four pop rounds) and the
    /// record kernel's confidence merge.
    private static let argMaxConfTop4Stage2 = MLXFast.metalKernel(
        name: "dflash_argmax_conf_top4_s2",
        inputNames: ["p_sc", "p_sum", "p_raw4", "p_idx4"],
        outputNames: ["tokens", "confidence", "ids", "raw"],
        source: """
            constexpr int V = V_;
            constexpr int CHUNKS = 64;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint idx = row * uint(CHUNKS) + lid;

            const float v_sc = p_sc[idx];
            const float v_sum = p_sum[idx];
            const float sg_sc = simd_max(v_sc);
            const float sg_sum = simd_sum(
                (v_sum > 0.0f) ? (v_sum * metal::exp(v_sc - sg_sc)) : 0.0f);
            threadgroup float sh_sc[2];
            threadgroup float sh_sum[2];
            threadgroup float sh_val[2];
            threadgroup int sh_idx[2];
            if (lane == 0) {
              sh_sc[sg] = sg_sc;
              sh_sum[sg] = sg_sum;
            }
            float lv[4];
            int li[4];
            for (int k = 0; k < 4; k++) {
              lv[k] = p_raw4[idx * 4 + k];
              li[k] = p_idx4[idx * 4 + k];
            }
            int ptr = 0;
            for (uint k = 0; k < 4; k++) {
              const float hv = ptr < 4 ? lv[ptr] : -INFINITY;
              const int hi = ptr < 4 ? li[ptr] : V;
              const float s_val = simd_max(hv);
              const int s_idx = simd_min((hv == s_val) ? hi : V);
              if (lane == 0) {
                sh_val[sg] = s_val;
                sh_idx[sg] = s_idx;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              const float v2 = (lane < 2) ? sh_val[lane] : -INFINITY;
              const int i2 = (lane < 2) ? sh_idx[lane] : V;
              const float r_val = simd_max(v2);
              const int r_idx = simd_min((v2 == r_val) ? i2 : V);
              if (lid == 0) {
                ids[row * 4 + k] = r_idx;
                raw[row * 4 + k] = r_val;
                if (k == 0) tokens[row] = r_idx;
              }
              if (ptr < 4 && hv == r_val && hi == r_idx) ptr++;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (sg == 0) {
              const float f_sc = (lane < 2) ? sh_sc[lane] : -INFINITY;
              const float f_sum = (lane < 2) ? sh_sum[lane] : 0.0f;
              const float row_sc = simd_max(f_sc);
              const float row_sum = simd_sum(
                  (f_sum > 0.0f) ? (f_sum * metal::exp(f_sc - row_sc)) : 0.0f);
              if (lane == 0) {
                confidence[row] = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
              }
            }
            """)

    /// Tokens (the argmax, `fastArgMax`'s total order), the softcapped top-1
    /// probability, and the top 4 ids and raw values per row of bf16
    /// `[rows, V]` head products. Nil for another shape or dtype.
    static func fusedArgMaxConfTop4(_ logits: MLXArray, outputMultiplier: Float, softcap: Float)
        -> (tokens: MLXArray, confidence: MLXArray, ids: MLXArray, raw: MLXArray)?
    {
        guard logits.ndim == 2, logits.dtype == .bfloat16, logits.dim(1) >= 256 else { return nil }
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        let chunks = 64
        let params = MLXArray([outputMultiplier, softcap])
        let partials = argMaxConfTop4Stage1(
            [logits, params],
            template: [("V_", vocab)],
            grid: (256 * chunks, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, chunks], [rows, chunks], [rows, chunks * 4], [rows, chunks * 4]],
            outputDTypes: [.float32, .float32, .float32, .int32])
        let out = argMaxConfTop4Stage2(
            partials,
            template: [("V_", vocab)],
            grid: (chunks, rows, 1), threadGroup: (chunks, 1, 1),
            outputShapes: [[rows], [rows], [rows, 4], [rows, 4]],
            outputDTypes: [.int32, .float32, .int32, .float32])
        return (out[0], out[1], out[2], out[3])
    }

    /// Load-time check of `fusedArgMaxConfTop4` against `fastArgMax` (tokens)
    /// and a host top 4 in the same total order (ids and raw values).
    static func top4SelfTest(vocabulary: Int, outputMultiplier: Float, softcap: Float) -> Bool {
        let rows = 7
        var host = (0 ..< rows * vocabulary).map { _ in Float.random(in: -8 ... 8) }
        for row in 0 ..< rows {
            let base = row * vocabulary
            host[base + (row * 7919) % vocabulary] = 30
            host[base + (row * 104729 + 13) % vocabulary] = 30
            host[base + (row * 1299709 + 101) % vocabulary] = 29
            host[base + (row * 15485863 + 7) % vocabulary] = 29
        }
        let logits = MLXArray(host, [rows, vocabulary]).asType(.bfloat16)
        guard
            let out = fusedArgMaxConfTop4(
                logits, outputMultiplier: outputMultiplier, softcap: softcap)
        else { return false }
        let argmax = fastArgMax(logits)
        let rounded = logits.asType(.float32)
        eval(out.tokens, out.ids, out.raw, argmax, rounded)
        let ids = out.ids.asArray(Int32.self)
        let raw = out.raw.asArray(Float.self)
        let tokens = out.tokens.asArray(Int32.self)
        let first = argmax.asArray(Int32.self)
        let values = rounded.asArray(Float.self)
        for row in 0 ..< rows {
            let base = row * vocabulary
            var best: [(value: Float, id: Int)] = []
            for id in 0 ..< vocabulary {
                let v = values[base + id]
                guard !v.isNaN else { continue }
                if best.count == 4, !(v > best[3].value) { continue }
                var at = best.count
                while at > 0, v > best[at - 1].value { at -= 1 }
                best.insert((v, id), at: at)
                if best.count > 4 { best.removeLast() }
            }
            guard best.count == 4, tokens[row] == first[row], ids[row * 4] == first[row] else {
                return false
            }
            for k in 0 ..< 4 {
                guard ids[row * 4 + k] == Int32(best[k].id), raw[row * 4 + k] == best[k].value
                else { return false }
            }
        }
        return true
    }

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

    // MARK: Draft tree top-3 (CBV2_VERIFY_TREE)

    /// Stage 1: per (row, chunk), the chunk's top 3 entries in the argmax's
    /// total order (larger value first, ties to the lower id; NaN never
    /// enters, as in `fastArgMaxStage1`). Each thread keeps its own sorted
    /// top 3, then three pop rounds merge the threadgroup's lists.
    private static let top3Stage1 = MLXFast.metalKernel(
        name: "dflash_top3_s1",
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

            float lv[3] = {-INFINITY, -INFINITY, -INFINITY};
            int li[3] = {int(vocab), int(vocab), int(vocab)};
            for (uint i = start + lid; i < end; i += 256u) {
              const float v = float(row_ptr[i]);
              const int idx = int(i);
              if (v > lv[2] || (v == lv[2] && idx < li[2])) {
                if (v > lv[1] || (v == lv[1] && idx < li[1])) {
                  lv[2] = lv[1]; li[2] = li[1];
                  if (v > lv[0] || (v == lv[0] && idx < li[0])) {
                    lv[1] = lv[0]; li[1] = li[0];
                    lv[0] = v; li[0] = idx;
                  } else {
                    lv[1] = v; li[1] = idx;
                  }
                } else {
                  lv[2] = v; li[2] = idx;
                }
              }
            }

            threadgroup float sh_val[8];
            threadgroup int sh_idx[8];
            int ptr = 0;
            for (uint k = 0; k < 3; k++) {
              const float hv = ptr < 3 ? lv[ptr] : -INFINITY;
              const int hi = ptr < 3 ? li[ptr] : int(vocab);
              const float sg_val = simd_max(hv);
              const int sg_idx = simd_min((hv == sg_val) ? hi : int(vocab));
              if (lane == 0) {
                sh_val[sg] = sg_val;
                sh_idx[sg] = sg_idx;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              const float v8 = (lane < 8) ? sh_val[lane] : -INFINITY;
              const int i8 = (lane < 8) ? sh_idx[lane] : int(vocab);
              const float tg_val = simd_max(v8);
              const int tg_idx = simd_min((v8 == tg_val) ? i8 : int(vocab));
              if (lid == 0) {
                const uint out_i = (row * CHUNKS + chunk) * 3 + k;
                p_val[out_i] = tg_val;
                p_idx[out_i] = tg_idx;
              }
              if (ptr < 3 && hv == tg_val && hi == tg_idx) ptr++;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """)

    /// Stage 2: per row, merge the 64 chunks' sorted top 3 the same way.
    private static let top3Stage2 = MLXFast.metalKernel(
        name: "dflash_top3_s2",
        inputNames: ["p_val", "p_idx"],
        outputNames: ["ids"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint vocab = uint(V_);
            const uint base = (row * CHUNKS + lid) * 3;
            float lv[3] = {p_val[base], p_val[base + 1], p_val[base + 2]};
            int li[3] = {p_idx[base], p_idx[base + 1], p_idx[base + 2]};
            threadgroup float sh_val[2];
            threadgroup int sh_idx[2];
            int ptr = 0;
            for (uint k = 0; k < 3; k++) {
              const float hv = ptr < 3 ? lv[ptr] : -INFINITY;
              const int hi = ptr < 3 ? li[ptr] : int(vocab);
              const float sg_val = simd_max(hv);
              const int sg_idx = simd_min((hv == sg_val) ? hi : int(vocab));
              if (lane == 0) {
                sh_val[sg] = sg_val;
                sh_idx[sg] = sg_idx;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              const float v2 = (lane < 2) ? sh_val[lane] : -INFINITY;
              const int i2 = (lane < 2) ? sh_idx[lane] : int(vocab);
              const float r_val = simd_max(v2);
              const int r_idx = simd_min((v2 == r_val) ? i2 : int(vocab));
              if (lid == 0) {
                ids[row * 3 + k] = r_idx;
              }
              if (ptr < 3 && hv == r_val && hi == r_idx) ptr++;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """)

    /// `[rows, 3]` int32: each row's top-3 ids of bf16 `[rows, V]` logits,
    /// column 0 equal to `fastArgMax`. Nil for another shape or dtype.
    static func fusedTop3(_ logits: MLXArray) -> MLXArray? {
        guard logits.ndim == 2, logits.dtype == .bfloat16, logits.dim(1) >= 256 else { return nil }
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        let chunks = 64
        let partials = top3Stage1(
            [logits],
            template: [("V_", vocab)],
            grid: (256 * chunks, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, chunks * 3], [rows, chunks * 3]],
            outputDTypes: [.float32, .int32])
        return top3Stage2(
            partials,
            template: [("V_", vocab)],
            grid: (chunks, rows, 1), threadGroup: (chunks, 1, 1),
            outputShapes: [[rows, 3]],
            outputDTypes: [.int32])[0]
    }

    /// Load-time check of `fusedTop3` against `fastArgMax` and a host sort in
    /// the same total order, on random bf16 rows with planted ties at the
    /// top. False disables the tree for the process.
    static func top3SelfTest(vocabulary: Int) -> Bool {
        let rows = 7
        var host = (0 ..< rows * vocabulary).map { _ in Float.random(in: -8 ... 8) }
        for row in 0 ..< rows {
            let base = row * vocabulary
            // Ties: a top value at a high and a low id, a second-place tie.
            let a = (row * 7919) % vocabulary
            let b = (row * 104729 + 13) % vocabulary
            let c = (row * 1299709 + 101) % vocabulary
            host[base + a] = 30
            host[base + b] = 30
            host[base + c] = 29
            if row == 3 { host[base + (a + 1) % vocabulary] = .nan }
        }
        let logits = MLXArray(host, [rows, vocabulary]).asType(.bfloat16)
        guard let top = fusedTop3(logits) else { return false }
        let argmax = fastArgMax(logits)
        let rounded = logits.asType(.float32)
        eval(top, argmax, rounded)
        let ids = top.asArray(Int32.self)
        let first = argmax.asArray(Int32.self)
        let values = rounded.asArray(Float.self)
        for row in 0 ..< rows {
            let base = row * vocabulary
            // Host top 3 in the same total order (ids ascend, so a later
            // equal value never displaces an earlier one).
            var best: [(value: Float, id: Int)] = []
            for id in 0 ..< vocabulary {
                let v = values[base + id]
                guard !v.isNaN else { continue }
                if best.count == 3, !(v > best[2].value) { continue }
                var at = best.count
                while at > 0, v > best[at - 1].value { at -= 1 }
                best.insert((v, id), at: at)
                if best.count > 3 { best.removeLast() }
            }
            guard best.count == 3, ids[row * 3] == first[row] else { return false }
            for k in 0 ..< 3 where ids[row * 3 + k] != Int32(best[k].id) { return false }
        }
        return true
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
        s.trace = nil
        s.memoTop = nil
        s.memoTop4 = nil
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

// MARK: - Draft tree

extension MuseGlimmerDFlashDrafter: CBv2MTPTreeDrafter {
    public func treeLeafTokens(
        _ leaves: [(row: Int, rank: Int)], requestState: any CBv2MTPRequestState
    ) -> MLXArray? {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released,
            let top = s.memoTop, top.ndim == 2, !leaves.isEmpty,
            leaves.allSatisfy({ $0.row >= 0 && $0.row < top.dim(0) && (1 ... top.dim(1)).contains($0.rank) })
        else { return nil }
        let width = top.dim(1)
        let indices = MLXArray(leaves.map { Int32($0.row * width + $0.rank - 1) })
        return take(top.reshaped([-1]), indices, axis: 0)
    }

    public func treeTopIDs(requestState: any CBv2MTPRequestState) -> MLXArray? {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return nil }
        return s.memoTop?.reshaped([-1])
    }

    public func treePrepareLeafCommit(stagedRows: Int, requestState: any CBv2MTPRequestState) {
        let s = state(requestState)
        // A leaf after a fully accepted chain commits one input past the
        // chain's drafted positions; finalizeRound checks against this count.
        if stagedRows > s.stagedInputCount, stagedRows <= settings.maximumDraftTokens {
            s.stagedInputCount = stagedRows
        }
    }

    public func treeTop4(requestState: any CBv2MTPRequestState) -> (ids: MLXArray, raw: MLXArray)? {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return nil }
        return s.memoTop4
    }

    public func setSpeculationRowPositions(
        _ rowPositions: MLXArray?, requestState: any CBv2MTPRequestState
    ) {
        state(requestState).speculationRowPositions = rowPositions
    }

    /// A same-width tree round committed a leaf: `rows` are the verify rows of
    /// its committed inputs (0 ... a, then the leaf's row). Keeps the
    /// tree-positioned speculation's commit for exactly those rows; any other
    /// speculation is dropped. True when kept.
    @discardableResult
    public func treeConfirmSpeculation(
        carryHidden: MLXArray, rows: [Int], requestState: any CBv2MTPRequestState
    ) -> Bool {
        let s = state(requestState)
        s.lastRoundFullyAccepted = false
        guard var speculation = s.speculation, speculation.treePositioned,
            speculation.carry == nil, let accepted = speculation.accepted,
            accepted == rows.count - 1, rows.allSatisfy({ $0 >= 0 && $0 <= speculation.depth }),
            (accepted == 0) == (speculation.pending == nil)
        else {
            s.speculation = nil
            return false
        }
        speculation.memo = nil
        speculation.confidence = nil
        speculation.rowMap = rows
        speculation.carry = carryHidden
        s.speculation = speculation
        return true
    }

    public func treeLeafCommitted(requestState: any CBv2MTPRequestState) {
        let s = state(requestState)
        // The post-verify speculation covered the chain outcomes only, and the
        // round did not accept every draft.
        s.speculation = nil
        s.lastRoundFullyAccepted = false
    }
}

// MARK: - Tree trace

extension MuseGlimmerDFlashDrafter: CBv2MTPProposalTracing {
    /// The current round's block trace, read back (waits for the device),
    /// each row best first with ties to the lower id. Nil unless
    /// `CBV2_TREE_TRACE` built one for this round. Clears it.
    public func takeProposalTrace(requestState: any CBv2MTPRequestState)
        -> CBv2MTPProposalTrace?
    {
        guard CBv2TreeTrace.tracesProposals, let s = requestState as? State,
            s.owner == ObjectIdentifier(self), !s.released, let trace = s.trace
        else { return nil }
        s.trace = nil
        let memoTokens = s.memo?.tokens
        var arrays = [trace.ids, trace.raw, trace.scores, trace.logSumExp]
        if let memoTokens { arrays.append(memoTokens) }
        eval(arrays)
        let rows = trace.ids.dim(0)
        let width = trace.ids.dim(1)
        let ids = trace.ids.asArray(Int32.self)
        let raw = trace.raw.asArray(Float.self)
        let scores = trace.scores.asArray(Float.self)
        guard ids.count == rows * width, raw.count == ids.count, scores.count == ids.count
        else { return nil }
        var sortedIDs: [Int32] = []
        var sortedRaw: [Float] = []
        var sortedScores: [Float] = []
        sortedIDs.reserveCapacity(ids.count)
        sortedRaw.reserveCapacity(ids.count)
        sortedScores.reserveCapacity(ids.count)
        for row in 0 ..< rows {
            let order = (0 ..< width).map { row * width + $0 }.sorted { a, b in
                raw[a] != raw[b] ? raw[a] > raw[b] : ids[a] < ids[b]
            }
            for index in order {
                sortedIDs.append(ids[index])
                sortedRaw.append(raw[index])
                sortedScores.append(scores[index])
            }
        }
        let proposals = memoTokens.map { $0.asType(.int32).asArray(Int32.self) } ?? []
        return CBv2MTPProposalTrace(
            rows: rows, width: width, ids: sortedIDs, raw: sortedRaw, scores: sortedScores,
            logSumExp: trace.logSumExp.asArray(Float.self), proposals: proposals,
            source: trace.source)
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

// MARK: - Parameter residency

/// The drafter's own buffers (pinned and re-quantized layers, the
/// low-acceptance MLP copies) in a residency set beside the target's; the
/// draft head is the target's and already registered. Memory placement only.
extension MuseGlimmerDFlashDrafter: CBv2ResidentParametersProviding {
    public var residentParameterKey: ObjectIdentifier { ObjectIdentifier(model) }
    public func residentParameterArrays() -> [MLXArray] {
        var arrays = model.parameters().flattened().map(\.1)
        for low in lowMLP {
            for linear in [low.gate, low.up, low.down] {
                arrays += [linear.weight, linear.scales]
                if let biases = linear.biases { arrays.append(biases) }
            }
        }
        return arrays + [maskEmbedding]
    }
}
