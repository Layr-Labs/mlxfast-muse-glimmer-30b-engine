// CBv2MTPRoundDriver.swift
//
// ContinuousBatchingV2 — MTP (speculative decoding) round state.
//
// The driver is a passive state holder owned by `EngineLoopV2`: the drafter
// binding, per-row drafter carries, plan-scoped speculation marks, and the
// cumulative metrics. All round LOGIC (eligibility, execution, finalize)
// lives in `EngineLoopV2+MTP.swift` — it needs the loop's row state
// (`kvStates`, `multimodalByID`) and step-building helpers.
//
// Threading: everything except `metricsSnapshot()` is engine-thread
// confined. Metrics are mutated on the engine thread and snapshotted under
// a lock so the provider can poll them from any thread
// (`EngineV2.mtpMetricsSnapshot()`).

import Cmlx
import Foundation
import MLX

/// Explicit attention-only opt-in: while native request-stateful text MTP is
/// installed, media rows can use this SAME target's ordinary causal path. This
/// does not qualify media drafting or change other models' cohort policy.
public protocol CBv2MTPRequestScopedMediaFallback:
    CBv2MultimodalSteppableModel, CBv2MTPSteppableModel
{}

/// Recurrent counterpart of `CBv2MTPRequestScopedMediaFallback` for a target
/// served through `CBv2SteppableLanguageModelAdapter`. Media requests run this
/// same target's ordinary recurrent token and embedding forwards for their
/// whole lifetime: they never draft, verify, carry or own assistant state,
/// while text requests keep speculating. A decode step with both kinds of rows
/// splits them into two groups; the result is exact only because the target's
/// decode is row-exact (no row depends on the other rows of its forward), so
/// adopt this only on such a target. It does not qualify media drafting.
public protocol CBv2RecurrentMTPRequestScopedMediaFallback:
    CBv2RecurrentMTPForwardable, CBv2PositionedRecurrentEmbeddingForwardable
{}

/// Steppable view of `CBv2RecurrentMTPRequestScopedMediaFallback`, answered
/// at runtime by adapters over arbitrary language models.
protocol CBv2RecurrentMTPRequestScopedMediaFallbackProviding: CBv2RecurrentMTPSteppableModel {
    var supportsRecurrentRequestScopedMediaFallback: Bool { get }
}

// MARK: - Drafter carry

/// One row's drafter carry: the newest confirmed-but-unfed token (the round
/// seed) plus the target's pre-norm hidden at the position BEFORE it — the
/// pair the Gemma-4 drafter chains from. `tokensCount`/`kvOffset` fingerprint
/// the row state the carry was captured against; any mismatch at plan time
/// (a plain step appended a token, preemption reset progress, an id was
/// reused) invalidates the carry, costing exactly one seed step (prefer
/// simple and correct over carry salvage).
struct CBv2MTPCarry {
    let token: Int
    /// [1, 1, H], lazy slice of an already-evaluated step output.
    let hidden: MLXArray
    /// [K] int32 target top-K token ids at the carry position, lazy slice of
    /// an already-evaluated verify shortlist. Non-nil only when the drafter
    /// opted in AND the captured top-K probability mass cleared the coverage
    /// threshold at finalize; nil ⇒ the next draft scores the full head.
    let shortlist: MLXArray?
    /// Target-logit top-two margin selected at the carry position. It is
    /// finalized at the round's existing host readback and used only by the
    /// next adaptive depth decision.
    let previousTopTwoMargin: Double?
    /// True only for a verify-produced carry whose terminal transition was
    /// intentionally left outside committed assistant history.
    let needsHistoryTransition: Bool

    /// `rec.tokens.count` at capture — the carry token must still be
    /// `tokens.last` with the same count.
    let tokensCount: Int
    /// `rec.numComputedTokens` at capture (== the row's KV absoluteOffset,
    /// the round anchor).
    let kvOffset: Int
}

/// Scalar-only observation of an actual stored carry. No arrays, state owners
/// or replacement values escape through this test seam.
struct CBv2MTPCarryObservationForTesting {
    let id: CBv2RequestID
    let token: Int
    let previousTopTwoMargin: Double?
    let tokensCount: Int
    let kvOffset: Int
    let hiddenShape: [Int]
}

// MARK: - In-flight round payload

/// MTP payload riding a `CBv2InFlightStep`. Its presence marks the step as
/// an MTP round step: the chained-decode fast path must never build on top
/// of it (the chained finalize/deferred-release machinery assumes exactly
/// one sample per row), and `finalize` runs the accept-walk for the verify
/// rows at the step's one host-sync boundary.
final class CBv2MTPRoundInFlight {

    struct VerifyRow {
        let id: CBv2RequestID
        /// Storage-owning sequence states at launch, for finalize-time
        /// `rollback` + `commitSpeculativeWrite` (identical objects to
        /// `kvStates[id]` unless the row finished mid-flight, in which case
        /// the whole state is released via the deferred-release fence and
        /// this list is not touched).
        let storageRows: [CBv2SequenceKV]
        /// Request-owned autoregressive assistant state moved out of the
        /// driver for the duration of this in-flight round. This fences its
        /// release behind target/assistant evaluation on cancellation.
        let assistantState: (any CBv2MTPRequestState)?
        var freeSpan: CBv2MTPFreeSpan? = nil
    }

    struct CommittedObservationRow {
        let id: CBv2RequestID
        /// Request-owned state detached until the target graph is fenced.
        let assistantState: any CBv2MTPRequestState
    }

    struct Verify {
        /// Draft tokens per row this round (uniform across the batch).
        let k: Int
        /// Verify-batch rows, in batch row order.
        let rows: [VerifyRow]
        /// Lazy flattened int32 packet: all [B, k] draft ids followed by all
        /// [B, 1+k] target argmaxes (`CBv2NonFiniteLogits.invalidTarget`
        /// where that position's target logits were non-finite), then — iff
        /// `shortlistIDs` is non-nil — all [B, 1+k] shortlist probability
        /// masses in parts-per-million, then — iff `hasAcceptMask` — all
        /// [B, k] typical-acceptance decisions (1 = keep the draft). One
        /// `asArray` at finalize reads everything, preserving the single
        /// host-sync boundary.
        let acceptancePacket: MLXArray
        /// Lazy [B,k] draft ids retained for exact accepted-prefix slicing
        /// into stateful assistant finalization.
        let draftIDs: MLXArray
        /// Lazy [B, 1+k, H] pre-norm hidden — the next carry is gathered
        /// from it at the finalize sync (index = accepted position).
        let lastHidden: MLXArray
        /// Raw target window retained only when a row has a grammar. Its
        /// committed prefix is sampled with the ordinary sampler at finalize.
        var constraintLogits: MLXArray? = nil
        /// Lazy [B, 1+k, K] int32 target top-K ids per verify position, for
        /// shortlisted drafting next round. nil when the drafter did not opt
        /// in or verification ran on the serial oracle.
        let shortlistIDs: MLXArray?
        /// Serial recurrent target generations in target-column order. Empty
        /// for attention-only targets.
        let recurrentEvaluations: [CBv2RequestID: [CBv2RecurrentStateEvaluation]]
        /// Lazy exact Qwen policy top-two values [B, 1+k, 2], float32.
        /// IDs feed greedy scoring on device; values are read only after the
        /// existing acceptance-packet fence.
        let policyTopTwoValues: MLXArray?
        /// True when the packet ends with the [B, k] typical-acceptance mask
        /// (`CBv2MTPAcceptance.typical` on a target-prefix batch). Finalize
        /// then walks the mask instead of comparing drafts to targets.
        let hasAcceptMask: Bool
        var diagnostics: [CBv2LogitDiagnosticPacket] = []
        var includesAssistantPrefill = false
        /// Active static verify tree for this round (nil for linear or dynamic).
        var verifyTree: CBv2VerifyTree? = nil
        /// True when the round used a GPU-dynamic tree whose packed parent word
        /// rides as the final Int32 of `acceptancePacket`.
        var hasDynamicTree = false
        /// GPU `[18]` int32 tree descriptor for post-verify outcome speculation.
        var dynTreeMeta: MLXArray? = nil

    }

    /// nil when this round only seeded (no row had a valid carry yet).
    let verify: Verify?
    /// Seed rows: (request, row index into seedHidden/seedPolicyTopTwoValues).
    /// The opt-in media fallback compacts these to REAL hidden-producing text
    /// rows; ordinary legacy cohorts retain their original decode indices. Their
    /// bonus token rides the step's normal `sampledTokens`; the carry hidden
    /// is sliced from `seedHidden` at finalize.
    let seedRows: [(id: CBv2RequestID, decodeIndex: Int)]
    /// Lazy [B_hidden, 1, H] native target hidden (MiMo: post-final-norm).
    /// (non-nil iff `seedRows` is non-empty).
    let seedHidden: MLXArray?
    /// Lazy float32 [B_hidden, 1, 2] policy values for adaptive seed and
    /// temporary depth-zero carries.
    let seedPolicyTopTwoValues: MLXArray?
    /// Plain prompt/decode target observations whose request-owned assistant
    /// states stay detached until this step's target graph is fenced.
    let committedObservationRows: [CommittedObservationRow]
    /// Those observations' token and hidden arrays (and carries), all
    /// evaluated with this step.
    let committedObservationArrays: [MLXArray]
    /// Final prompt chunks whose last pre-norm hidden row [1, 1, H] becomes
    /// the request's first carry once its first token is confirmed.
    let prefillCarries: [(id: CBv2RequestID, hidden: MLXArray)]
    /// Lazy float32 [N, 1, 2] policy values for `prefillCarries`, in order;
    /// nil when no prefill carry is expected to speculate next.
    let prefillCarryTopTwoValues: MLXArray?

    /// Finalization outcomes used by host-only controller attribution. These
    /// are populated at the existing host-sync boundary.
    var finalizedSeedIDs: Set<CBv2RequestID> = []
    var finalizedVerifyIDs: Set<CBv2RequestID> = []
    var claimedSeedCostNanos: UInt64 = 0
    /// Outputs actually kept after common-width, EOS, stop and budget handling.
    var committedVerifyTokenCount = 0
    /// Cancellation-owned assistant states released only after the target KV
    /// and recurrent deferred-release fence has retired.
    var deferredAssistantReleases: [any CBv2MTPRequestState] = []

    init(
        verify: Verify?,
        seedRows: [(id: CBv2RequestID, decodeIndex: Int)],
        seedHidden: MLXArray?,
        seedPolicyTopTwoValues: MLXArray?,
        committedObservationRows: [CommittedObservationRow],
        committedObservationArrays: [MLXArray],
        prefillCarries: [(id: CBv2RequestID, hidden: MLXArray)],
        prefillCarryTopTwoValues: MLXArray?
    ) {
        self.verify = verify
        self.seedRows = seedRows
        self.seedHidden = seedHidden
        self.seedPolicyTopTwoValues = seedPolicyTopTwoValues
        self.committedObservationRows = committedObservationRows
        self.committedObservationArrays = committedObservationArrays
        self.prefillCarries = prefillCarries
        self.prefillCarryTopTwoValues = prefillCarryTopTwoValues
    }
}

/// A seed step belongs to the row cohort it prepared, not to the depth that
/// happened to be selected on that plan. A later verification may run at a
/// different depth; exact-cohort matching prevents cancelled or invalidated
/// seed work from leaking into an unrelated probe.
struct CBv2MTPSeedCostLedger {
    private struct Pending {
        var nanos: UInt64
        let requestIDs: Set<CBv2RequestID>
    }

    private var byBucket: [Int: Pending] = [:]

    mutating func record(
        decodeRowBucket: Int, requestIDs: Set<CBv2RequestID>, nanos: UInt64
    ) {
        guard decodeRowBucket > 0, !requestIDs.isEmpty, nanos > 0 else { return }
        if var pending = byBucket[decodeRowBucket], pending.requestIDs == requestIDs {
            pending.nanos &+= nanos
            byBucket[decodeRowBucket] = pending
        } else {
            byBucket[decodeRowBucket] = Pending(nanos: nanos, requestIDs: requestIDs)
        }
    }

    /// Claim seed cost before verify-row completion can retire request ids.
    /// An overlapping but non-identical cohort is discarded rather than
    /// partially charging work whose rectangular batch shape changed.
    mutating func take(
        decodeRowBucket: Int, requestIDs: Set<CBv2RequestID>
    ) -> UInt64 {
        guard let pending = byBucket[decodeRowBucket], !requestIDs.isEmpty else { return 0 }
        if pending.requestIDs == requestIDs {
            byBucket.removeValue(forKey: decodeRowBucket)
            return pending.nanos
        }
        if !pending.requestIDs.isDisjoint(with: requestIDs) {
            byBucket.removeValue(forKey: decodeRowBucket)
        }
        return 0
    }

    mutating func invalidate(_ id: CBv2RequestID) {
        let matching = byBucket.compactMap { bucket, pending in
            pending.requestIDs.contains(id) ? bucket : nil
        }
        for bucket in matching {
            byBucket.removeValue(forKey: bucket)
        }
    }

    mutating func removeAll() {
        byBucket.removeAll(keepingCapacity: false)
    }

    var count: Int { byBucket.count }
}

// MARK: - Driver

/// Engine-side MTP state: drafter binding, carries, plan marks, metrics.
final class CBv2MTPRoundDriver {

    let config: CBv2MTPConfig
    let drafter: any CBv2MTPDrafter
    /// The engine's model, downcast once at build (verify forwards go
    /// through `forwardWithHidden`).
    let model: any CBv2MTPSteppableModel
    let captureLayers: CBv2MTPCaptureLayers?
    /// Media requests run target-only beside speculating text. Fixed by the
    /// drafter and model at build; see ``requestScopedMediaFallback(model:drafter:)``.
    let supportsRequestScopedMediaFallback: Bool
    /// Any MTP-ineligible row (stop strings, penalties, logit bias, logprobs)
    /// decodes one ordinary target token per step beside speculating rows
    /// instead of keeping every row target-only. Only the recurrent fallback
    /// qualifies: its decode is row-exact, so the mixed step changes no row.
    let ineligibleRowsDecodeBesideSpeculation: Bool
    /// True when acceptance may use target-prefix pre-sampling (drafter
    /// opt-in). The planner additionally requires the installed sampler's
    /// `supportsMTPTargetPrefix` before lifting the greedy gate.
    let targetPrefixAcceptance: Bool
    /// The stateful drafter opted into ordinary-decode history without the
    /// committed-observation fence, and the target exposes the recurrent
    /// hidden-returning decode seam. Fixed at build.
    private let supportsOrdinaryDecodeHistory: Bool
    /// The final prompt row may seed the first carry: either ordinary-decode
    /// history above, or a stateful drafter on a target with no recurrent
    /// state that opted in (`seedsFirstRoundFromPrefill`). Fixed at build.
    private let supportsPrefillCarry: Bool
    private let depthController: CBv2MTPDepthController
    private var committedDecodeClock = CBv2MTPCommittedDecodeClock()
    private var committedGoodputClock = CBv2MTPCommittedGoodputClock()
    private var goodputPlanRows: [CBv2RequestID] = []
    /// Numeric request IDs can be reused. Stamp launch measurements with the
    /// current lifetime so late chained finalization cannot train a new one.
    private(set) var workloadGeneration: UInt64 = 0
    private var workloadInvalidated = false

    // Engine-thread confined.
    private var carries: [CBv2RequestID: CBv2MTPCarry] = [:]
    private var assistantStates: [CBv2RequestID: any CBv2MTPRequestState] = [:]
    var admittedMTPBindingIssuer: CBv2MTPAdmissionIssuer?
    /// Immutable request-lifetime exclusion, not the lifetime of evaluated
    /// feature arrays. Kept through preemption; cleared at actual finish.
    private var targetOnlyMediaRequests: Set<CBv2RequestID> = []
    /// Qwen acceptance is request-owned; raw nonchained wall cost is shared
    /// only within a compatible decode-row bucket.
    private var requestAcceptance: [CBv2RequestID: CBv2MTPRequestAcceptanceState] = [:]
    /// Optional per-request tree cooldown set when `CBV2_TREE_COOLDOWN=1`.
    private var treeCooldownRequests: Set<CBv2RequestID> = []
    /// Plan-scoped: the confirmed post-verify speculation's lazy draft
    /// confidence for each row whose window this plan offered it to, kept
    /// until that round's finalize calibrates with it. Cleared every plan.
    private var planDraftConfidence: [CBv2RequestID: MLXArray] = [:]
    /// Windows sized by an already-available draft confidence, and windows
    /// that fell back to the acceptance history because it was still in
    /// flight. Diagnostics only.
    private(set) var draftConfidenceWindowHits = 0
    private(set) var draftConfidenceWindowMisses = 0
    private var rawCostEstimators: [Int: CBv2MTPRawCostEstimator] = [:]

    /// Plan-scoped: rows the planner offered a 1+k round this plan.
    private(set) var roundMarks: [CBv2RequestID: Int] = [:]
    /// Plan-scoped: eligible rows without a valid carry — their decode step
    /// this plan is a SEED step (eager forwardWithHidden, hidden captured).
    private(set) var seedMarks: Set<CBv2RequestID> = []
    /// Step-global seed fallback without destroying persistent request
    /// history. Used when one rectangular row lacks a carry.
    private(set) var forceSeedPlan = false

    /// One selection for the whole scheduler plan. Every speculating row in
    /// that plan uses this depth, so target verification stays rectangular.
    private(set) var controllerDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)
    private(set) var planDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)
    private(set) var controllerMeasurementEligible = false

    private let metricsLock = NSLock()
    private var metrics = CBv2MTPMetrics()

    private var pendingSeedCosts = CBv2MTPSeedCostLedger()
    /// Install on the engine queue before submitting requests. Called only
    /// after the real carry assignment; never performs an evaluation itself.
    var carryStoredObserverForTesting: ((CBv2MTPCarryObservationForTesting) -> Void)?
    /// Numeric-test-only, never lifecycle/residency/timing evidence. A test may
    /// retain at most one genuine seed carry view and compare it AFTER engine
    /// drain. The callback must not eval/read back/copy arrays or mutate state.
    /// Nil in ordinary tests and production; no parameter/eligibility override.
    var carryHiddenObserverForTesting: ((CBv2RequestID, MLXArray) -> Void)?

    private init(
        config: CBv2MTPConfig, drafter: any CBv2MTPDrafter,
        model: any CBv2MTPSteppableModel, captureLayers: CBv2MTPCaptureLayers?,
        supportsOrdinaryDecodeChaining: Bool
    ) {
        var config = config
        if let required = drafter.requiredVerificationMode {
            config.verificationMode = required
            if required != .rectangular { config.maxAutomaticRectangularTokens = 0 }
        }
        if let maximum = drafter.maximumDraftTokens {
            config.maxDraftTokens = min(config.maxDraftTokens, max(0, maximum))
            config.fixedDraftTokens = config.fixedDraftTokens.map {
                min($0, config.maxDraftTokens)
            }
        }
        if let maximum = drafter.maximumSpeculativeBatch {
            config.maxSpeculativeBatch = min(config.maxSpeculativeBatch, max(1, maximum))
        }
        let requestStatefulRecurrent =
            drafter is any CBv2MTPRequestStatefulDrafter
            && (model as? any CBv2RecurrentMTPSteppableModel)?.recurrentStateSpec != nil
        if requestStatefulRecurrent {
            config.maxDraftTokens = min(
                config.maxDraftTokens, CBv2MTPMarginalDepthPolicy.maximumDepth)
            config.fixedDraftTokens = config.fixedDraftTokens.map {
                min($0, config.maxDraftTokens)
            }
        }
        if requestStatefulRecurrent {
            // Production recurrent verification requires the captured-window
            // seam. If it is absent, clamp to target-only before any assistant
            // state or draft graph exists; never turn a planned positive-depth
            // Qwen round into multiple serial target forwards.
            let capturedWindow =
                (model as? any CBv2RecurrentMTPSteppableModel)?
                .supportsCapturedVerifyWindow ?? false
            if config.verificationMode != .serialTarget && !capturedWindow {
                config.maxDraftTokens = 0
                config.fixedDraftTokens = config.fixedDraftTokens.map { _ in 0 }
                config.maxAutomaticRectangularTokens = 0
            }
            if config.verificationMode == .serialTarget {
                // The serial column loop is additionally the proven B=1
                // shape: one request-local assistant proposal and one
                // recurrent target transaction per column.
                config.maxSpeculativeBatch = 1
            }
        }
        self.targetPrefixAcceptance = drafter.supportsTargetPrefixAcceptance
        // Ordinary steps have no committed-observation fence (only MTP steps
        // run `finishStatefulMTPEvaluation`); drafters needing one keep MTP steps.
        let stateful = drafter as? any CBv2MTPRequestStatefulDrafter
        let supportsOrdinaryDecodeHistory =
            requestStatefulRecurrent && stateful?.supportsOrdinaryDecodeHistory == true
            && stateful?.requiresCommittedObservationFence == false
        self.supportsOrdinaryDecodeHistory = supportsOrdinaryDecodeHistory
        // A target with no recurrent state observes its prompt rows in the
        // MTP step's own prefill branch, fenced with that step; the last
        // row's hidden produced the first token, so it is the seed's carry.
        let recurrentTarget =
            (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec != nil
        self.supportsPrefillCarry =
            supportsOrdinaryDecodeHistory
            || (!recurrentTarget && stateful?.seedsFirstRoundFromPrefill == true)
        self.config = config
        self.drafter = drafter
        self.model = model
        self.captureLayers = captureLayers
        let requestScopedFallback = Self.requestScopedMediaFallback(model: model, drafter: drafter)
        self.supportsRequestScopedMediaFallback = requestScopedFallback
        self.ineligibleRowsDecodeBesideSpeculation =
            requestScopedFallback
            && (model as? any CBv2RecurrentMTPSteppableModel)?.recurrentStateSpec != nil
        self.depthController = CBv2MTPDepthController(
            maxDepth: self.config.maxDraftTokens, fixedDepth: self.config.fixedDraftTokens,
            fixedDepthByDecodeRows: self.config.fixedDraftTokensByDecodeRows,
            useCommittedDecodeBaseline: supportsOrdinaryDecodeChaining
                && drafter.supportsTargetPrefixAcceptance
                && !(drafter is any CBv2MTPRequestStatefulDrafter))
        self.metrics.verificationMode = self.config.verificationMode
        self.metrics.acceptance = self.config.acceptance
        self.metrics.maxAutomaticRectangularTokens = self.config.maxAutomaticRectangularTokens
    }

    /// Build the driver, or nil when MTP cannot activate: config off (or the
    /// `DARKBLOOM_CBV2_MTP` kill switch), no drafter, or a model that cannot
    /// drive rounds. nil ⇒ the engine is byte-identical to MTP-less builds.
    static func build(
        model: CBv2SteppableModel, drafter: (any CBv2MTPDrafter)?,
        config: CBv2MTPConfig,
        supportsRectangularCacheBank: Bool = true,
        supportsOrdinaryDecodeChaining: Bool = true
    ) -> CBv2MTPRoundDriver? {
        guard config.effectiveEnabled, let drafter else { return nil }
        guard !drafter.requiresExactAcceptance || config.acceptance == .exact else { return nil }
        guard let mtpModel = model as? (any CBv2MTPSteppableModel) else { return nil }
        let stateful = drafter is any CBv2MTPRequestStatefulDrafter
        let recurrentTarget = model as? any CBv2RecurrentMTPSteppableModel
        let recurrent = recurrentTarget?.recurrentStateSpec != nil
        let captureLayers = mtpModel.mtpCaptureLayers
        guard
            (stateful && mtpModel.supportsRequestStatefulMTP
                && (recurrentTarget == nil || recurrent))
                || (!stateful && !recurrent && captureLayers != nil)
        else { return nil }
        guard let modelTarget = mtpModel.mtpTargetIdentity,
            let drafterTarget = drafter.mtpTargetIdentity,
            modelTarget == drafterTarget
        else { return nil }
        var config = config
        let effectiveVerification =
            drafter.requiredVerificationMode
            ?? config.verificationMode
        if stateful, effectiveVerification != .serialTarget,
            !supportsRectangularCacheBank
        {
            config.maxDraftTokens = 0
            config.fixedDraftTokens = config.fixedDraftTokens.map { _ in 0 }
            config.maxAutomaticRectangularTokens = 0
        }
        let driver = CBv2MTPRoundDriver(
            config: config, drafter: drafter, model: mtpModel, captureLayers: captureLayers,
            supportsOrdinaryDecodeChaining: supportsOrdinaryDecodeChaining)
        if driver.usesMarginalPolicy {
            guard mtpModel is any CBv2MTPPolicyTopTwoProviding else { return nil }
            if let availability =
                mtpModel as? any CBv2MTPPolicyTopTwoCapabilityProviding,
                !availability.cbv2MTPPolicyTopTwoAvailable
            {
                return nil
            }
        }
        return driver
    }

    // MARK: Plan-scoped marks

    /// Reset speculation marks. Called immediately before every
    /// `scheduler.plan()` so marks can never leak across plans (a rolled-
    /// back plan's marks must not classify the next plan's rows).
    func beginPlan(
        plannedDecodeRows: Int, canSpeculate: Bool,
        rowIDs: [CBv2RequestID]? = nil
    ) {
        if let rowIDs, depthController.usesCommittedDecodeBaseline {
            let changedCohort = Set(rowIDs) != Set(goodputPlanRows)
            if changedCohort {
                workloadGeneration &+= 1
                committedDecodeClock = CBv2MTPCommittedDecodeClock()
                pendingSeedCosts.removeAll()
            }
            if !canSpeculate || changedCohort {
                committedGoodputClock.reset()
                depthController.cancelCommittedWindow(
                    decodeRowBucket: CBv2MTPDepthController.decodeRowBucket(goodputPlanRows.count))
            }
            goodputPlanRows = rowIDs
            workloadInvalidated = false
            depthController.beginWorkload(rowIDs: rowIDs)
        }
        if !roundMarks.isEmpty { roundMarks = [:] }
        if !seedMarks.isEmpty { seedMarks = [] }
        forceSeedPlan = false
        controllerMeasurementEligible = canSpeculate
        controllerDecision = depthController.select(
            plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate)
        let offered =
            usesMarginalPolicy && canSpeculate && !controllerDecision.isExploration
            ? CBv2MTPDepthDecision(
                depth: config.maxDraftTokens,
                decodeRowBucket: controllerDecision.decodeRowBucket,
                reason: "marginal_offer",
                isExploration: false)
            : controllerDecision
        planDecision = adaptiveSerialLimitedDecision(
            verificationLimitedDecision(offered, plannedDecodeRows: plannedDecodeRows))
        guard plannedDecodeRows > 0 else { return }
        metricsLock.lock()
        metrics.selectedDepth = planDecision.depth
        metrics.decodeRowBucket = planDecision.decodeRowBucket
        metrics.depthSelections[planDecision.depth, default: 0] += 1
        metrics.controllerFallbacks[planDecision.reason, default: 0] += 1
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    func previewDecision(
        plannedDecodeRows: Int, canSpeculate: Bool
    ) -> CBv2MTPDepthDecision {
        adaptiveSerialLimitedDecision(
            verificationLimitedDecision(
                depthController.preview(
                    plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate),
                plannedDecodeRows: plannedDecodeRows))
    }

    /// See `CBv2MTPConfig.allowsAdaptiveSerialRounds`.
    var suppressesAdaptiveSerialRounds: Bool {
        !config.allowsAdaptiveSerialRounds && config.verificationMode == .serialTarget
            && config.fixedDraftTokens == nil
    }

    /// Set once by the engine before any request when the configured strategy
    /// cannot run exactly (for example, native MiMo rectangular verification
    /// without its admitted scalar-dense scratch). Every plan, fixed or
    /// adaptive, then stays target-only; persistent history stays live.
    private(set) var speculativeRoundsSuppression: String?

    func suppressSpeculativeRounds(reason: String) {
        speculativeRoundsSuppression = reason
    }

    private func adaptiveSerialLimitedDecision(
        _ decision: CBv2MTPDepthDecision
    ) -> CBv2MTPDepthDecision {
        guard decision.depth > 0 else { return decision }
        if let reason = speculativeRoundsSuppression {
            return CBv2MTPDepthDecision(
                depth: 0, decodeRowBucket: decision.decodeRowBucket,
                reason: reason, isExploration: false)
        }
        guard suppressesAdaptiveSerialRounds else { return decision }
        return CBv2MTPDepthDecision(
            depth: 0, decodeRowBucket: decision.decodeRowBucket,
            reason: "adaptive_serial_suppressed", isExploration: false)
    }

    func maximumAutomaticDepth(plannedDecodeRows: Int) -> Int {
        guard config.verificationMode == .automatic, plannedDecodeRows > 0 else {
            return config.maxDraftTokens
        }
        let maxWidth = config.maxAutomaticRectangularTokens / plannedDecodeRows
        return min(config.maxDraftTokens, max(0, maxWidth - 1))
    }

    /// Pure ceiling on any plan depth with `decodeRows` decode-ready rows:
    /// zero beyond the batch gate, otherwise the automatic envelope. Deadline
    /// projection mirrors it without touching controller state.
    func speculationDepthCeiling(decodeRows: Int) -> Int {
        decodeRows > config.maxSpeculativeBatch
            ? 0 : maximumAutomaticDepth(plannedDecodeRows: decodeRows)
    }

    private func verificationLimitedDecision(
        _ decision: CBv2MTPDepthDecision, plannedDecodeRows: Int
    ) -> CBv2MTPDepthDecision {
        let limit = maximumAutomaticDepth(plannedDecodeRows: plannedDecodeRows)
        guard decision.depth > limit else { return decision }
        return CBv2MTPDepthDecision(
            depth: limit, decodeRowBucket: decision.decodeRowBucket,
            reason: "automatic_rectangular_limit", isExploration: false)
    }

    func requiresNonChainedDepthZeroProbe(_ decision: CBv2MTPDepthDecision) -> Bool {
        depthController.requiresNonChainedDepthZeroProbe(decision)
    }

    var planDepth: Int { planDecision.depth }
    var planDecodeRowBucket: Int { planDecision.decodeRowBucket }

    /// A plan boundary can discover that one otherwise eligible row cannot
    /// complete the selected full round (typically a max-token tail). Mixing
    /// that row's L=1 target decode with neighbors' L=1+k verify changes the
    /// target batch shape. Clamp the whole plan to depth zero so target-only
    /// and MTP execute the same rectangular tail batch.
    func clampPlanDepth(to requestedDepth: Int, reason: String) {
        let newDepth = min(max(requestedDepth, 0), planDecision.depth)
        guard newDepth != planDecision.depth else { return }
        let oldDepth = planDecision.depth
        depthController.cancelCommittedWindow(decodeRowBucket: planDecision.decodeRowBucket)
        planDecision = CBv2MTPDepthDecision(
            depth: newDepth, decodeRowBucket: planDecision.decodeRowBucket,
            reason: reason, isExploration: false)
        metricsLock.lock()
        metrics.selectedDepth = newDepth
        if let count = metrics.depthSelections[oldDepth], count > 0 {
            if count == 1 {
                metrics.depthSelections.removeValue(forKey: oldDepth)
            } else {
                metrics.depthSelections[oldDepth] = count - 1
            }
        }
        metrics.depthSelections[newDepth, default: 0] += 1
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func markRound(_ id: CBv2RequestID, k: Int) {
        guard !targetOnlyMediaRequests.contains(id) else { return }
        roundMarks[id] = k
    }
    func markSeed(_ id: CBv2RequestID) {
        guard !targetOnlyMediaRequests.contains(id) else { return }
        seedMarks.insert(id)
    }
    func roundMark(for id: CBv2RequestID) -> Int? { roundMarks[id] }

    /// Whether the fixed-depth window currently treats `id` as a
    /// low-acceptance request.
    func inLowAcceptanceRegime(_ id: CBv2RequestID) -> Bool {
        usesFixedDepthAcceptanceWindow
            && (requestAcceptance[id] ?? initialRequestAcceptance).inLowAcceptanceRegime
    }

    /// Stricter low-acceptance check (`observedRegimeRounds >= 3 && fraction < 0.55`)
    /// so approximate Stage-1 draft-head screening never fires on high-acceptance
    /// requests after a single early low round.
    func inDeepLowAcceptanceRegime(_ id: CBv2RequestID) -> Bool {
        usesFixedDepthAcceptanceWindow
            && (requestAcceptance[id] ?? initialRequestAcceptance).isDeepLowAcceptanceRegime
    }

    func shouldUseVerifyTree(for id: CBv2RequestID) -> Bool {
        inLowAcceptanceRegime(id)
            && (!CBv2DraftRegime.treeCooldownEnabled || !treeCooldownRequests.contains(id))
    }

    func updateTreeCooldown(
        for id: CBv2RequestID, wasTree: Bool, chainDepth: Int, k: Int, accepted: Int, branchHit: Bool
    ) {
        guard CBv2DraftRegime.treeCooldownEnabled else { return }
        if wasTree && !branchHit && chainDepth < k && accepted >= chainDepth {
            treeCooldownRequests.insert(id)
        } else if !wasTree && accepted < 4 {
            treeCooldownRequests.remove(id)
        }
    }
    func isSeedMarked(_ id: CBv2RequestID) -> Bool { seedMarks.contains(id) }

    func forceSynchronizedSeed() {
        forceSeedPlan = true
    }
    /// True when this plan produced any MTP work (round or seed).
    var planHasMTPWork: Bool { !roundMarks.isEmpty || !seedMarks.isEmpty }

    // MARK: Carries

    enum CarryStatus {
        case valid(CBv2MTPCarry)
        /// A carry existed but no longer matches the row (plain step,
        /// preemption, id reuse) — dropped by `validatedCarry`.
        case stale
        case none
    }

    /// Pure check (no mutation) — the chained-path pre-check uses it.
    func hasValidCarry(for rec: CBv2ScheduledRequest) -> Bool {
        guard !targetOnlyMediaRequests.contains(rec.id) else { return false }
        guard let carry = carries[rec.id] else { return false }
        return carryMatches(carry, rec: rec)
    }

    /// Validate and return the row's carry; a stale carry is removed here
    /// (invalidate-on-mismatch — one seed step re-establishes it).
    func validatedCarry(for rec: CBv2ScheduledRequest) -> CarryStatus {
        guard !targetOnlyMediaRequests.contains(rec.id) else { return .none }
        guard let carry = carries[rec.id] else { return .none }
        guard carryMatches(carry, rec: rec) else {
            carries.removeValue(forKey: rec.id)
            return .stale
        }
        return .valid(carry)
    }

    private func carryMatches(_ carry: CBv2MTPCarry, rec: CBv2ScheduledRequest) -> Bool {
        rec.pendingSamples == 0
            && carry.tokensCount == rec.tokens.count
            && carry.token == rec.tokens.last
            && carry.kvOffset == rec.numComputedTokens
    }

    /// Take the row's carry for a launching round (a fresh one is stored at
    /// the round's finalize, or the row seeds again).
    func consumeCarry(for id: CBv2RequestID) -> CBv2MTPCarry? {
        guard !targetOnlyMediaRequests.contains(id) else { return nil }
        return carries.removeValue(forKey: id)
    }

    /// Return a carry taken by `consumeCarry` for a round refused before any
    /// graph was built (`CBv2StepScratchRefusal`). The row's KV, tokens and
    /// assistant history are unchanged, so the carry still describes it; the
    /// next plan's `validatedCarry` drops it if that ever stops holding.
    func restoreConsumedCarry(_ carry: CBv2MTPCarry, for id: CBv2RequestID) {
        guard !targetOnlyMediaRequests.contains(id), carries[id] == nil else { return }
        carries[id] = carry
    }

    func storeCarry(
        id: CBv2RequestID, token: Int, hidden: MLXArray,
        shortlist: MLXArray? = nil, previousTopTwoMargin: Double? = nil,
        needsHistoryTransition: Bool = false,
        tokensCount: Int, kvOffset: Int
    ) {
        guard !targetOnlyMediaRequests.contains(id) else { return }
        if tracksPersistentHistory,
            let stateful = drafter as? any CBv2MTPRequestStatefulDrafter,
            assistantStates[id] == nil
        {
            assistantStates[id] = stateful.makeRequestState()
        }
        carries[id] = CBv2MTPCarry(
            token: token, hidden: hidden, shortlist: shortlist,
            previousTopTwoMargin: previousTopTwoMargin,
            needsHistoryTransition: needsHistoryTransition,
            tokensCount: tokensCount, kvOffset: kvOffset)
        if let observer = carryStoredObserverForTesting {
            observer(
                .init(
                    id: id, token: token, previousTopTwoMargin: previousTopTwoMargin,
                    tokensCount: tokensCount, kvOffset: kvOffset, hiddenShape: hidden.shape))
        }
        carryHiddenObserverForTesting?(id, hidden)
    }

    var tracksPersistentHistory: Bool {
        config.maxDraftTokens > 0
            && usesRequestStatefulDrafter && config.fixedDraftTokens != 0
    }

    /// The drafter declared its request state advisory
    /// (`CBv2MTPAdvisoryHistoryDrafter`) and acceptance is exact, so no
    /// emitted token depends on that state.
    var historyIsAdvisory: Bool {
        config.acceptance == .exact
            && (drafter as? any CBv2MTPAdvisoryHistoryDrafter)?.requestStateIsAdvisory == true
    }

    /// Persistent assistant history that a prefix checkpoint must carry for
    /// the request that adopts it. Advisory history never has to.
    var checkpointsRequireAssistantHistory: Bool {
        tracksPersistentHistory && !historyIsAdvisory
    }

    /// Rows that cannot speculate may run ordinary chainable decode; the
    /// engine feeds their trusted hidden rows to history at finalize.
    var feedsHistoryFromOrdinaryDecode: Bool {
        config.ordinaryDecodeHistory && tracksPersistentHistory && supportsOrdinaryDecodeHistory
    }

    /// The final prompt row seeds the first carry (no seed step).
    var seedsCarryFromPrefill: Bool {
        config.prefillCarry && tracksPersistentHistory && supportsPrefillCarry
    }

    /// Whether the current plan's decode rows passed the batch and
    /// eligibility gates (controller depth may still be zero).
    var planCanSpeculate: Bool { controllerMeasurementEligible }

    /// Drop a carry made stale by an in-flight successor, or whose history
    /// transition was just observed, without touching assistant history.
    func dropCarry(_ id: CBv2RequestID) {
        carries.removeValue(forKey: id)
    }

    /// A request-stateful drafter over an attention target that adopts
    /// `CBv2MTPRequestScopedMediaFallback`, or over a recurrent target whose
    /// adapter reports `CBv2RecurrentMTPRequestScopedMediaFallback`. A recurrent
    /// target never qualifies through the attention-only protocol.
    static func requestScopedMediaFallback(
        model: any CBv2MTPSteppableModel, drafter: any CBv2MTPDrafter
    ) -> Bool {
        guard drafter is any CBv2MTPRequestStatefulDrafter else { return false }
        guard (model as? any CBv2RecurrentMTPSteppableModel)?.recurrentStateSpec != nil else {
            return model is any CBv2MTPRequestScopedMediaFallback
        }
        return (model as? any CBv2RecurrentMTPRequestScopedMediaFallbackProviding)?
            .supportsRecurrentRequestScopedMediaFallback == true
    }
    func requiresTargetOnlyMedia(_ request: CBv2Request) -> Bool {
        supportsRequestScopedMediaFallback && request.multimodal?.spans.isEmpty == false
    }
    func registerTargetOnlyMedia(_ request: CBv2Request) {
        guard requiresTargetOnlyMedia(request) else { return }
        if targetOnlyMediaRequests.insert(request.id).inserted {
            precondition(
                assistantStates[request.id] == nil && carries[request.id] == nil,
                "immutable media request acquired unobserved assistant history")
        }
    }
    func isTargetOnlyMediaForTesting(_ id: CBv2RequestID) -> Bool {
        targetOnlyMediaRequests.contains(id)
    }

    func takeOrMakeAssistantState(
        for id: CBv2RequestID, maximumSequenceLength: Int,
        historicalPrefixPromptTokens: [Int]? = nil
    ) throws -> (any CBv2MTPRequestState)? {
        guard !targetOnlyMediaRequests.contains(id), tracksPersistentHistory,
            let stateful = drafter as? any CBv2MTPRequestStatefulDrafter
        else { return nil }
        let existing = assistantStates.removeValue(forKey: id)
        let state = existing ?? stateful.makeRequestState()
        do {
            try configureAssistantState(state, id: id, maximumSequenceLength: maximumSequenceLength)
            if existing == nil, let historicalPrefixPromptTokens {
                guard let historical = drafter as? any CBv2HistoricalMTPPrefixCheckpointCoding
                else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                try historical.installPrefixCaptureContext(
                    requestState: state, promptTokens: historicalPrefixPromptTokens)
            }
        } catch {
            // Configuration detached an existing owner from the map. Put it
            // back before propagating so the fenced cohort retirement path
            // remains the single authority that releases request state.
            assistantStates[id] = state
            throw error
        }
        return state
    }

    func configureAssistantState(
        _ state: any CBv2MTPRequestState, id: CBv2RequestID,
        maximumSequenceLength: Int
    ) throws {
        guard let stateful = drafter as? any CBv2MTPRequestStatefulDrafter else { return }
        if let admitted = stateful as? any CBv2AdmittedMTPRequestStatefulDrafter {
            guard let issuer = admittedMTPBindingIssuer else {
                throw CBv2MTPBindingError.missingAdmission
            }
            try admitted.configureRequestState(
                state,
                admission: issuer.context(id: id, maximumInputs: maximumSequenceLength))
        } else {
            try stateful.configureRequestState(state, maximumSequenceLength: maximumSequenceLength)
        }
    }

    var usesMarginalPolicy: Bool {
        config.maxDraftTokens > 0
            && usesRequestStatefulDrafter && config.fixedDraftTokens == nil
    }

    var shouldApplyMarginalPolicyToPlan: Bool {
        usesMarginalPolicy && !planDecision.isExploration
    }
    /// `position` is the absolute target position of the observation's first
    /// hidden row. Only an advisory drafter receives it.
    func observeCommittedTarget(
        id: CBv2RequestID,
        observation: CBv2MTPCommittedTargetObservation,
        startingAt position: Int? = nil,
        detachedState: any CBv2MTPRequestState
    ) {
        precondition(
            !targetOnlyMediaRequests.contains(id),
            "media must never observe trained assistant state")
        guard let stateful = drafter as? any CBv2MTPRequestStatefulDrafter else {
            preconditionFailure("CBv2 MTP committed observation reached a stateless drafter")
        }
        if let position, historyIsAdvisory,
            let advisory = stateful as? any CBv2MTPAdvisoryHistoryDrafter
        {
            advisory.observeCommittedTarget(
                observation, startingAt: position, requestState: detachedState)
        } else {
            stateful.observeCommittedTarget(observation, requestState: detachedState)
        }
    }

    /// `observeCommittedTarget` for several requests in one batched call.
    /// Positions reach only an advisory drafter, as above.
    func observeCommittedTargets(
        _ observations: [(id: CBv2RequestID, observation: CBv2MTPRequestObservation)],
        drafter: any CBv2MTPBatchedRequestDrafter
    ) {
        precondition(
            observations.allSatisfy { !targetOnlyMediaRequests.contains($0.id) },
            "media must never observe trained assistant state")
        let advisory = historyIsAdvisory && drafter is any CBv2MTPAdvisoryHistoryDrafter
        drafter.observeCommittedTargets(
            observations.map {
                advisory || $0.observation.position == nil
                    ? $0.observation
                    : CBv2MTPRequestObservation(
                        $0.observation.observation, startingAt: nil,
                        requestState: $0.observation.requestState)
            })
    }

    func marginalDepth(
        for id: CBv2RequestID,
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int,
        decodeRowBucket: Int
    ) -> Int {
        guard usesRequestStatefulDrafter, config.fixedDraftTokens == nil else {
            return min(max(offeredDepth, 0), config.maxDraftTokens)
        }
        let acceptance = requestAcceptance[id] ?? CBv2MTPRequestAcceptanceState()
        let cost = rawCostEstimators[decodeRowBucket] ?? CBv2MTPRawCostEstimator()
        let cappedOffer = min(
            offeredDepth, CBv2MTPMarginalDepthPolicy.maximumDepth)
        if cost.needsSteadyStateProbe(depth: 1) {
            return CBv2MTPMarginalDepthPolicy.boundedProbeDepth(
                offeredDepth: cappedOffer,
                remainingTokens: remainingTokens,
                verificationLimit: verificationLimit)
        }
        return CBv2MTPMarginalDepthPolicy.selectDepth(
            offeredDepth: cappedOffer,
            remainingTokens: remainingTokens,
            verificationLimit: verificationLimit,
            acceptanceProbabilities: acceptance.probabilities,
            previousTargetTopTwoMargin: carries[id]?.previousTopTwoMargin,
            headStepCostRatio: cost.headStepCostRatio)
    }

    /// A fixed positive depth with a request-stateful drafter under exact
    /// acceptance: the declared depth is the ceiling and each plan's verify
    /// window is clamped by the request's own acceptance history
    /// (`CBv2MTPMarginalDepthPolicy.fixedWindowDepth`). Exact acceptance
    /// keeps every emitted token the target's own, so the window changes how
    /// many rows are verified, never which tokens are committed. Scoped to
    /// advisory-history drafters (`historyIsAdvisory`, which implies exact
    /// acceptance): their request state reaches neither output nor prefix
    /// checkpoints, so other fixed-depth drafters keep their exact shapes.
    var usesFixedDepthAcceptanceWindow: Bool {
        guard let fixed = config.fixedDraftTokens else { return false }
        return fixed > 0 && config.maxDraftTokens > 0
            && usesRequestStatefulDrafter && historyIsAdvisory
    }

    private var initialRequestAcceptance: CBv2MTPRequestAcceptanceState {
        usesFixedDepthAcceptanceWindow
            ? CBv2MTPRequestAcceptanceState(
                flatPrior: CBv2MTPMarginalDepthPolicy.fixedWindowFlatPrior)
            : CBv2MTPRequestAcceptanceState()
    }

    /// The acceptance window's probability that a round of `depth` drafts
    /// accepts every one: the product of the request's per-position
    /// acceptance estimates. Nil when the window does not apply. Host
    /// arithmetic over history already read back; no device work.
    func predictedFullAcceptance(for id: CBv2RequestID, depth: Int) -> Double? {
        guard usesFixedDepthAcceptanceWindow, depth > 0 else { return nil }
        let probabilities = (requestAcceptance[id] ?? initialRequestAcceptance).probabilities
        guard depth <= probabilities.count else { return nil }
        return probabilities.prefix(depth).reduce(1, *)
    }

    /// The verify window for one row under a fixed declared depth; at most
    /// `offeredDepth` and at least one draft while `offeredDepth > 0`.
    func fixedDepthAcceptanceWindow(
        for id: CBv2RequestID,
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int
    ) -> Int {
        let ceiling = min(max(offeredDepth, 0), config.maxDraftTokens)
        guard usesFixedDepthAcceptanceWindow, ceiling > 0 else { return ceiling }
        let acceptance = requestAcceptance[id] ?? initialRequestAcceptance
        // Read the round's draft confidence in either regime so its
        // calibration keeps learning while the declared depth runs.
        // The window itself never depends on whether the device has already
        // finished that speculation: that is a host/device timing race, so the
        // same request got a different verify width on a slower host.
        _ = availableDraftConfidence(for: id)
        // High-acceptance regime: always the declared depth.
        guard acceptance.inLowAcceptanceRegime else { return ceiling }
        var depth = CBv2MTPMarginalDepthPolicy.fixedWindowDepth(
            offeredDepth: ceiling,
            remainingTokens: remainingTokens,
            verificationLimit: verificationLimit,
            acceptanceProbabilities: acceptance.probabilities)
        if CBv2DraftRegime.treeAdditiveSlot,
            CBv2DraftRegime.verifyTreeMode != 0, CBv2DraftRegime.dynamicTreeEnabled,
            shouldUseVerifyTree(for: id)
        {
            let leaves = max(1, CBv2DraftRegime.dynamicTreeMaxLeaves)
            let target = max(CBv2DraftRegime.dynamicTreeMinWidth, depth + leaves)
            depth = min(ceiling, min(remainingTokens - 1, min(verificationLimit, target)))
        }
        return max(1, depth)
    }

    /// The acceptance ceiling's draft count for one row (see
    /// `CBv2MTPRequestAcceptanceState.acceptanceCeilingDepth`). Same scope
    /// as the fixed-depth acceptance window; host arithmetic over the
    /// request's finalized rounds only.
    func acceptanceCeilingDepth(for id: CBv2RequestID, offeredDepth: Int) -> Int {
        guard usesFixedDepthAcceptanceWindow, offeredDepth > 0 else { return offeredDepth }
        return (requestAcceptance[id] ?? initialRequestAcceptance)
            .acceptanceCeilingDepth(offeredDepth: offeredDepth)
    }

    /// Start of a plan: forget the previous plan's draft confidences.
    func beginDraftConfidencePlan() {
        if !planDraftConfidence.isEmpty { planDraftConfidence.removeAll(keepingCapacity: true) }
    }

    /// The next block's per-position draft confidence, when the drafter
    /// built that block behind the previous verify (a confirmed post-verify
    /// speculation) AND the device has already finished it. Never waits and
    /// never schedules: an unfinished speculation leaves the history window
    /// in charge, so the window adds no host-device sync to the round.
    private func availableDraftConfidence(for id: CBv2RequestID) -> [Double]? {
        guard let speculator = drafter as? any CBv2MTPPostVerifySpeculating,
            let state = assistantStates[id],
            let array = speculator.speculatedDraftConfidence(requestState: state)
        else { return nil }
        planDraftConfidence[id] = array
        guard let values = Self.hostValuesIfAvailable(array) else {
            draftConfidenceWindowMisses &+= 1
            return nil
        }
        draftConfidenceWindowHits &+= 1
        return values
    }

    /// Host copy of `array` only when its evaluation has already completed
    /// (`_mlx_array_is_available` checks the completion event without
    /// waiting); nil while it is unscheduled or in flight.
    private static func hostValuesIfAvailable(_ array: MLXArray) -> [Double]? {
        var available = false
        guard _mlx_array_is_available(&available, array.ctx) == 0, available,
            array.ndim == 1, array.dtype == .float32
        else { return nil }
        return array.asArray(Float.self).map(Double.init)
    }

    func observeRequestAcceptance(
        id: CBv2RequestID,
        draftedDepth: Int,
        acceptedDepth: Int,
        committedTokens: Int,
        rejectionObserved: Bool,
        endedByTruncation: Bool
    ) {
        guard usesRequestStatefulDrafter,
            config.fixedDraftTokens == nil || usesFixedDepthAcceptanceWindow
        else { return }
        var state = requestAcceptance[id] ?? initialRequestAcceptance
        state.observeCommitted(draftedDepth: draftedDepth, committedTokens: committedTokens)
        state.observe(
            draftedDepth: draftedDepth,
            acceptedDepth: acceptedDepth,
            rejectionObserved: rejectionObserved,
            endedByTruncation: endedByTruncation)
        // The round's block confidence finished before its verify did, so it
        // is available here without waiting.
        if let array = planDraftConfidence.removeValue(forKey: id),
            let confidence = Self.hostValuesIfAvailable(array)
        {
            state.observeConfidence(
                confidence,
                draftedDepth: draftedDepth,
                acceptedDepth: acceptedDepth,
                rejectionObserved: rejectionObserved)
        }
        requestAcceptance[id] = state
    }

    func pendingHistoryCarry(for id: CBv2RequestID) -> CBv2MTPCarry? {
        guard !targetOnlyMediaRequests.contains(id) else { return nil }
        guard let carry = carries[id], carry.needsHistoryTransition else { return nil }
        return carry
    }

    func takeAssistantState(for id: CBv2RequestID) -> (any CBv2MTPRequestState)? {
        assistantStates.removeValue(forKey: id)
    }

    func restoreAssistantState(
        _ state: any CBv2MTPRequestState, for id: CBv2RequestID
    ) {
        precondition(
            !targetOnlyMediaRequests.contains(id), "media must never restore assistant state")
        precondition(assistantStates[id] == nil, "duplicate CBv2 MTP assistant state")
        assistantStates[id] = state
    }

    func canInstallHistoricalAssistant(for id: CBv2RequestID) -> Bool {
        tracksPersistentHistory && !targetOnlyMediaRequests.contains(id)
            && assistantStates[id] == nil && carries[id] == nil
    }

    func releaseDetachedAssistantState(_ state: any CBv2MTPRequestState) {
        (drafter as? any CBv2MTPRequestStatefulDrafter)?.releaseRequestState(state)
    }

    func assistantStateCountsForTesting(
        _ id: CBv2RequestID
    ) -> (committed: Int, staged: Int)? {
        assistantStates[id].map { ($0.committedInputCount, $0.stagedInputCount) }
    }

    func materializedAssistantBytes(
        detachedStates: [any CBv2MTPRequestState] = []
    ) -> Int {
        var seen = Set<ObjectIdentifier>()
        var total = 0
        for state in Array(assistantStates.values) + detachedStates {
            guard seen.insert(ObjectIdentifier(state)).inserted else { continue }
            let (next, overflow) = total.addingReportingOverflow(state.materializedBytes)
            total = overflow ? Int.max : next
        }
        return total
    }

    var usesRequestStatefulDrafter: Bool {
        drafter is any CBv2MTPRequestStatefulDrafter
    }

    func hasUnmeasuredAssistantResidency(
        detachedStates: [any CBv2MTPRequestState] = []
    ) -> Bool {
        (Array(assistantStates.values) + detachedStates).contains {
            ($0 as? any CBv2MTPRequestResidencyReporting)?.hasUnmeasuredResidency == true
        }
    }

    private func releaseAssistantState(_ id: CBv2RequestID) {
        guard let state = assistantStates.removeValue(forKey: id),
            let stateful = drafter as? any CBv2MTPRequestStatefulDrafter
        else { return }
        stateful.releaseRequestState(state)
    }

    /// Preemption / membership hygiene: the structural fingerprint would
    /// catch these lazily, but dropping eagerly keeps no stale device
    /// arrays alive.
    func invalidateCarry(_ id: CBv2RequestID) {
        carries.removeValue(forKey: id)
        releaseAssistantState(id)
        pendingSeedCosts.invalidate(id)
        requestAcceptance.removeValue(forKey: id)
        planDraftConfidence.removeValue(forKey: id)
        treeCooldownRequests.remove(id)
    }

    /// The request left the engine for good — ids are legally reusable, so
    /// every per-id trace must go (a reused id must never inherit a carry).
    func requestDidFinish(_ id: CBv2RequestID) {
        targetOnlyMediaRequests.remove(id)
        if goodputPlanRows.contains(id) { invalidateWorkload() }
        carries.removeValue(forKey: id)
        releaseAssistantState(id)
        roundMarks.removeValue(forKey: id)
        seedMarks.remove(id)
        pendingSeedCosts.invalidate(id)
        requestAcceptance.removeValue(forKey: id)
        planDraftConfidence.removeValue(forKey: id)
        treeCooldownRequests.remove(id)
    }

    /// Invalidate only adaptive stateless learning. Other live rows retain
    /// their carries; the next plan calibrates its new request cohort.
    private func invalidateWorkload() {
        guard depthController.usesCommittedDecodeBaseline else { return }
        workloadGeneration &+= 1
        workloadInvalidated = true
        goodputPlanRows.removeAll(keepingCapacity: true)
        committedDecodeClock = CBv2MTPCommittedDecodeClock()
        committedGoodputClock.reset()
        pendingSeedCosts.removeAll()
        depthController.invalidateWorkload()
        metricsLock.lock()
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    private func acceptsWorkloadMeasurement(_ measurement: CBv2MTPStepMeasurement?) -> Bool {
        guard depthController.usesCommittedDecodeBaseline else { return true }
        guard !workloadInvalidated else { return false }
        // Older host-only tests construct unstamped measurements. Production
        // always stamps at plan capture and preserves the stamp at attachment.
        return measurement?.workloadGeneration.map { $0 == workloadGeneration } ?? true
    }

    /// Drain/shutdown drops every request trace and adaptive workload while
    /// retaining cumulative metrics for a final poll.
    func removeAllRequestState() {
        invalidateWorkload()
        carries.removeAll(keepingCapacity: false)
        rawCostEstimators.removeAll(keepingCapacity: false)
        for id in Array(assistantStates.keys) { releaseAssistantState(id) }
        roundMarks.removeAll(keepingCapacity: false)
        seedMarks.removeAll(keepingCapacity: false)
        pendingSeedCosts.removeAll()
        requestAcceptance.removeAll(keepingCapacity: false)
        planDraftConfidence.removeAll(keepingCapacity: false)
        treeCooldownRequests.removeAll(keepingCapacity: false)
    }

    var requestStateCountForTesting: Int {
        carries.count + assistantStates.count + requestAcceptance.count
            + roundMarks.count + seedMarks.count + pendingSeedCosts.count
    }

    // MARK: Metrics (lock-protected; polled cross-thread)

    func recordSkip(_ reason: String) {
        metricsLock.lock()
        metrics.skippedRows[reason, default: 0] += 1
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func recordControllerFallback(_ reason: String) {
        metricsLock.lock()
        metrics.controllerFallbacks[reason, default: 0] += 1
        metricsLock.unlock()
    }

    func recordSeedSteps(_ count: Int) {
        guard count > 0 else { return }
        metricsLock.lock()
        metrics.seedSteps += count
        // engine-local: per-round audit record (benchmark free-run audit)
        metrics.seedRounds += 1
        metricsLock.unlock()
    }

    func recordEarlyDraftSubmission() {
        metricsLock.lock()
        metrics.earlyDraftSubmissions += 1
        metricsLock.unlock()
    }

    func recordRound(
        drafted: Int, accepted: Int, emitted: Int,
        audit: CBv2MTPRoundAuditRecord? = nil
    ) {
        metricsLock.lock()
        metrics.rounds += 1
        metrics.draftedTokens += drafted
        metrics.acceptedTokens += accepted
        metrics.emittedTokens += emitted
        // engine-local: per-round audit record (benchmark free-run audit)
        if let audit {
            metrics.roundAudits.append(audit)
        }
        if metrics.perPositionAccepted.count < drafted {
            metrics.perPositionAccepted.append(
                contentsOf: Array(
                    repeating: 0, count: drafted - metrics.perPositionAccepted.count))
        }
        for position in 0 ..< accepted {
            metrics.perPositionAccepted[position] += 1
        }
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    /// The controller optimizes the synchronized rectangular step, so it
    /// learns the minimum accepted prefix that every participating verify
    /// row can commit together, once per step (not once per row), unless the
    /// drafter commits each row's own prefix: then once per row.
    func recordStepAcceptance(
        drafted: Int, accepted: Int, observedDrafts: Int,
        decodeRowBucket: Int,
        measurement: CBv2MTPStepMeasurement? = nil
    ) {
        guard acceptsWorkloadMeasurement(measurement) else { return }
        depthController.observeAcceptance(
            decodeRowBucket: decodeRowBucket,
            drafted: observedDrafts,
            accepted: accepted)
        metricsLock.lock()
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    func claimPendingSeedCost(
        decodeRowBucket: Int, finalizedVerifyIDs: Set<CBv2RequestID>,
        measurement: CBv2MTPStepMeasurement? = nil
    ) -> UInt64 {
        guard acceptsWorkloadMeasurement(measurement) else { return 0 }
        return pendingSeedCosts.take(
            decodeRowBucket: decodeRowBucket, requestIDs: finalizedVerifyIDs)
    }

    /// The ordinary alternative to stateless MTP can pipeline target decode.
    /// Measure its actual commit cadence, with no extra clock or GPU readback.
    /// Reset on every nonqualifying finalize to exclude idle/cohort transitions.
    func recordCommittedDecodeBaseline(
        measurement: CBv2MTPStepMeasurement?, completedAtNanos: UInt64,
        sampledRows: [CBv2RequestID], finalizedPlainRowCount: Int,
        hasChainedSuccessor: Bool
    ) {
        guard depthController.usesCommittedDecodeBaseline,
            acceptsWorkloadMeasurement(measurement)
        else { return }
        let eligible =
            measurement.map {
                $0.costEligible && $0.chained && !$0.seedOnly
                    && $0.actualDepth == 0 && $0.decision.depth == 0
                    && $0.decision.decodeRowBucket
                        == CBv2MTPDepthController.decodeRowBucket(sampledRows.count)
            } == true && hasChainedSuccessor && sampledRows.count == finalizedPlainRowCount
        guard
            let elapsed = committedDecodeClock.observe(
                completedAtNanos: completedAtNanos, rowIDs: sampledRows, eligible: eligible),
            let measurement
        else { return }
        depthController.observeCommittedDecodeInterval(
            decodeRowBucket: measurement.decision.decodeRowBucket, wallTimeNanos: elapsed)
    }

    func recordStepCost(
        _ measurement: CBv2MTPStepMeasurement,
        wallTimeNanos: UInt64,
        finalizedPlainWork: Bool,
        finalizedSeedIDs: Set<CBv2RequestID>,
        finalizedVerification: Bool,
        claimedSeedCostNanos: UInt64,
        completedAtNanos: UInt64 = 0,
        committedRows: [CBv2RequestID] = [],
        committedTokenCount: Int = 0
    ) {
        guard wallTimeNanos > 0 else { return }
        let decision = measurement.decision
        if depthController.usesCommittedDecodeBaseline {
            guard acceptsWorkloadMeasurement(measurement) else {
                // Finishing may precede this step's measurement callback.
                // Preserve executed-work telemetry without reviving learning.
                if measurement.actualDepth > 0, !measurement.seedOnly, finalizedVerification {
                    metricsLock.lock()
                    metrics.totalRoundWallTimeNanos &+= wallTimeNanos &+ claimedSeedCostNanos
                    metricsLock.unlock()
                }
                return
            }
            let sample = committedGoodputClock.observe(
                measurement: measurement, completedAtNanos: completedAtNanos,
                isolatedWallTimeNanos: wallTimeNanos, rowIDs: committedRows,
                committedTokens: committedTokenCount)
            if measurement.actualDepth > 0, !measurement.seedOnly,
                finalizedVerification
            {
                if let sample {
                    depthController.recordCommittedVerification(
                        decision: decision, wallTimeNanos: sample.wallTimeNanos,
                        committedTokens: sample.committedTokens, rowCount: committedRows.count)
                } else {
                    depthController.cancelCommittedWindow(decodeRowBucket: decision.decodeRowBucket)
                }
                metricsLock.lock()
                // Warmup is real work even when excluded from steady EWMA.
                metrics.totalRoundWallTimeNanos &+=
                    sample?.wallTimeNanos
                    ?? (wallTimeNanos &+ claimedSeedCostNanos)
                refreshControllerMetricsLocked()
                metricsLock.unlock()
                return
            }
        }
        if measurement.seedOnly, decision.depth > 0 {
            guard measurement.costEligible, !finalizedSeedIDs.isEmpty else {
                depthController.cancelCommittedWindow(decodeRowBucket: decision.decodeRowBucket)
                return
            }
            pendingSeedCosts.record(
                decodeRowBucket: decision.decodeRowBucket,
                requestIDs: finalizedSeedIDs,
                nanos: wallTimeNanos)
            return
        }
        if measurement.actualDepth == 0 {
            depthController.cancelCommittedWindow(decodeRowBucket: decision.decodeRowBucket)
        }
        let rawCostEligible =
            usesMarginalPolicy
            && measurement.costEligible && !measurement.chained
            && measurement.actualDepth >= 0
            && (measurement.actualDepth == 0
                ? finalizedPlainWork : finalizedVerification)
        var steadyRawCostRecorded = true
        if rawCostEligible {
            var estimator =
                rawCostEstimators[decision.decodeRowBucket]
                ?? CBv2MTPRawCostEstimator()
            steadyRawCostRecorded = estimator.observe(
                depth: measurement.actualDepth,
                rawWallTimeNanos: Double(wallTimeNanos),
                chained: false)
            rawCostEstimators[decision.decodeRowBucket] = estimator
        }
        if rawCostEligible, measurement.actualDepth > 0,
            !steadyRawCostRecorded
        {
            // Keep the controller depth unsampled too: its next selection is
            // an immediate bounded retry that obtains steady-state C1.
            return
        }
        let attributed = wallTimeNanos &+ claimedSeedCostNanos
        let recorded = depthController.recordFinalizedStep(
            decision: decision,
            actualDepth: measurement.actualDepth,
            wallTimeNanos: attributed,
            costEligible: measurement.costEligible,
            chained: measurement.chained,
            finalizedPlainWork: finalizedPlainWork,
            finalizedVerification: finalizedVerification)
        guard recorded else { return }
        metricsLock.lock()
        if measurement.actualDepth > 0 {
            metrics.totalRoundWallTimeNanos &+= attributed
        }
        refreshControllerMetricsLocked()
        metricsLock.unlock()
    }

    var pendingSeedCostCountForTesting: Int { pendingSeedCosts.count }

    func activeDepthForTesting(decodeRowBucket: Int) -> Int {
        depthController.activeDepthForTesting(decodeRowBucket: decodeRowBucket)
    }

    func probeIntervalForTesting(decodeRowBucket: Int) -> Int {
        depthController.probeIntervalForTesting(decodeRowBucket: decodeRowBucket)
    }

    func metricsSnapshot() -> CBv2MTPMetrics {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        return metrics
    }

    func recordVerificationStrategy(rectangular: Bool) {
        metricsLock.lock()
        if rectangular {
            metrics.rectangularVerificationRounds += 1
        } else {
            metrics.serialVerificationRounds += 1
        }
        metricsLock.unlock()
    }

    private func refreshControllerMetricsLocked() {
        let snapshot = depthController.snapshot()
        metrics.conditionalAcceptance = snapshot.conditionalAcceptance
        metrics.costInputs = snapshot.costInputs
    }
}

/// Official Gemma candidate generation carries the target hidden state that
/// predicted the newest accepted/unfed token. In a verify tensor whose input
/// columns are `[seed, d1, ...]`, that is exactly column `acceptedDrafts`.
/// Kept as a pure seam so indexing is deterministic and fixture-independent.
enum CBv2MTPHiddenIndex {
    static func carryColumn(targetOutputIndex: Int, draftDepth: Int) -> Int {
        precondition(targetOutputIndex >= 0 && targetOutputIndex <= draftDepth)
        return targetOutputIndex
    }
}

/// Multi-leaf verification tree strictly within `M <= 8` (`k <= 7` drafted
/// tokens) for low-acceptance verify rounds. Every leaf's ancestor prefix lies
/// on contiguous slots `0 ..< depth`, and every tree node's attention maps
/// logical ancestor positions onto physical KV slots in exact serial order
/// (0-ULP bit-identical to serial greedy).
public struct CBv2VerifyTree: Sendable {
    public let k: Int
    public let chainDepth: Int
    public let posDeltas: [Int]
    public let parent: [Int]
    public let children: [[Int]]
    public let ancPacked: [UInt32]
    public let packedPosDeltas: UInt32
    public let sourcePositions: [Int]
    public let sourceRanks: [Int]

    public var totalSlots: Int { 1 + k }

    public static func make(mode: Int, k: Int) -> CBv2VerifyTree? {
        guard mode > 0, (3 ... 7).contains(k) else { return nil }
        let chainDepth: Int
        let parents: [Int]
        let srcPos: [Int]
        let srcRank: [Int]
        switch (mode, k) {
        case (1, 7):
            chainDepth = 4
            parents = [-1, 0, 1, 2, 3, 0, 0, 1]
            srcPos = [0, 1, 2, 3, 0, 0, 1]
            srcRank = [0, 0, 0, 0, 1, 2, 1]
        case (2, 7):
            chainDepth = 3
            parents = [-1, 0, 1, 2, 0, 0, 1, 2]
            srcPos = [0, 1, 2, 0, 0, 1, 2]
            srcRank = [0, 0, 0, 1, 2, 1, 1]
        case (3, 7):
            chainDepth = 5
            parents = [-1, 0, 1, 2, 3, 4, 0, 1]
            srcPos = [0, 1, 2, 3, 4, 0, 1]
            srcRank = [0, 0, 0, 0, 0, 1, 1]
        case (_, 6):
            chainDepth = 5
            parents = [-1, 0, 1, 2, 3, 4, 0]
            srcPos = [0, 1, 2, 3, 4, 0]
            srcRank = [0, 0, 0, 0, 0, 1]
        case (_, 5):
            chainDepth = 4
            parents = [-1, 0, 1, 2, 3, 0]
            srcPos = [0, 1, 2, 3, 0]
            srcRank = [0, 0, 0, 0, 1]
        case (_, 4):
            chainDepth = 3
            parents = [-1, 0, 1, 2, 0]
            srcPos = [0, 1, 2, 0]
            srcRank = [0, 0, 0, 1]
        case (_, 3):
            chainDepth = 2
            parents = [-1, 0, 1, 0]
            srcPos = [0, 1, 0]
            srcRank = [0, 0, 1]
        default:
            return nil
        }
        let n = 1 + k
        var posDeltas = [Int](repeating: 0, count: n)
        var children = [[Int]](repeating: [], count: n)
        var ancPacked = [UInt32](repeating: 0, count: n)
        for s in 1 ..< n {
            let p = parents[s]
            posDeltas[s] = posDeltas[p] + 1
            children[p].append(s)
        }
        var packedPos: UInt32 = 0
        for s in 0 ..< n {
            packedPos |= UInt32(posDeltas[s] & 0xF) << (4 * s)
            var chain: [Int] = []
            var cur = s
            while cur >= 0 {
                chain.append(cur)
                cur = parents[cur]
            }
            chain.reverse()
            var packed: UInt32 = 0
            for (d, slot) in chain.enumerated() {
                packed |= UInt32(slot & 0xF) << (4 * d)
            }
            ancPacked[s] = packed
        }
        return CBv2VerifyTree(
            k: k, chainDepth: chainDepth, posDeltas: posDeltas, parent: parents,
            children: children, ancPacked: ancPacked, packedPosDeltas: packedPos,
            sourcePositions: srcPos, sourceRanks: srcRank)
    }

    /// Reconstructs a verify tree from the GPU-packed parent word written into
    /// `treeMeta[17]`: bits `0..27` hold 4-bit parents for slots `1...k`, and
    /// bits `28..31` hold `chainDepth`.
    public static func fromPackedParents(_ packed: UInt32, k: Int) -> CBv2VerifyTree {
        let n = 1 + k
        var parents = [Int](repeating: 0, count: n)
        parents[0] = -1
        for s in 1 ..< n {
            let p = Int((packed >> (4 * (s - 1))) & 0xF)
            parents[s] = min(p, s - 1)
        }
        let rawChain = Int((packed >> 28) & 0xF)
        let chainDepth = max(1, min(k, rawChain == 0 ? k : rawChain))
        var posDeltas = [Int](repeating: 0, count: n)
        var children = [[Int]](repeating: [], count: n)
        var ancPacked = [UInt32](repeating: 0, count: n)
        for s in 1 ..< n {
            let p = parents[s]
            posDeltas[s] = posDeltas[p] + 1
            children[p].append(s)
        }
        var packedPos: UInt32 = 0
        for s in 0 ..< n {
            packedPos |= UInt32(posDeltas[s] & 0xF) << (4 * s)
            var chain: [Int] = []
            var cur = s
            while cur >= 0 {
                chain.append(cur)
                cur = parents[cur]
            }
            chain.reverse()
            var anc: UInt32 = 0
            for (d, slot) in chain.enumerated() {
                anc |= UInt32(slot & 0xF) << (4 * d)
            }
            ancPacked[s] = anc
        }
        return CBv2VerifyTree(
            k: k, chainDepth: chainDepth, posDeltas: posDeltas, parent: parents,
            children: children, ancPacked: ancPacked, packedPosDeltas: packedPos,
            sourcePositions: [], sourceRanks: [])
    }
}

/// Set by the engine while a low-acceptance round's drafts and verify are built.
public enum CBv2DraftRegime {
    nonisolated(unsafe) public static var lowAcceptance = false
    /// Stricter deep-low-acceptance flag for approximate Stage-1 draft-head screening.
    nonisolated(unsafe) public static var xnorDraftHeadEligible = false
    /// Active static multi-leaf tree for the current low-acceptance round (nil = linear or dynamic).
    nonisolated(unsafe) public static var activeVerifyTree: CBv2VerifyTree? = nil
    /// Active GPU-dynamic tree draft count (`3...7`), or `0` when inactive.
    nonisolated(unsafe) public static var activeDynamicTreeK: Int = 0
    /// GPU `[18]` int32 tree descriptor emitted by the drafter for the active round:
    /// `[0] = packedPosDeltas`, `[1...8] = ancPacked[0...7]`, `[9...16] = depth[0...7]`,
    /// `[17] = packedParents`.
    nonisolated(unsafe) public static var activeDynamicTreeMeta: MLXArray? = nil
    /// `CBV2_TREE_LOW=0` disables tree verification on low-acceptance rounds.
    public static let verifyTreeMode: Int = {
        ProcessInfo.processInfo.environment["CBV2_TREE_LOW"].flatMap(Int.init) ?? 1
    }()
    /// `CBV2_DYN_TREE=0` disables GPU-dynamic tree selection (falls back to static tree).
    public static let dynamicTreeEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_DYN_TREE"] != "0"
    /// `CBV2_LOW_ONLY_VERIFY=0` applies verify-path micro-speedups on high-acceptance rounds too.
    /// Default (`true`) confines them to low-acceptance rounds so high-acceptance windows match
    /// baseline latency below the decode speedup ceiling.
    public static let lowOnlyVerifySpeedups: Bool =
        ProcessInfo.processInfo.environment["CBV2_LOW_ONLY_VERIFY"] != "0"
    /// `CBV2_TREE_OLDPOL=1` forces k-wide accept/reject updates on tree rounds.
    /// Default (`false`) uses terminal-branch and chain-depth semantics so tree rounds preserve width.
    public static let treeOldAcceptancePolicy: Bool =
        ProcessInfo.processInfo.environment["CBV2_TREE_OLDPOL"] == "1"
    /// Minimum verify draft count `k` eligible for dynamic tree (`CBV2_TREE_MIN_WIDTH`, default 4).
    public static let dynamicTreeMinWidth: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_TREE_MIN_WIDTH"],
            let v = Int(raw), (2 ... 7).contains(v)
        {
            return v
        }
        return 4
    }()
    /// Maximum verify draft count `k` eligible for dynamic tree (`CBV2_TREE_MAX_WIDTH`, default 7).
    public static let dynamicTreeMaxWidth: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_TREE_MAX_WIDTH"],
            let v = Int(raw), (2 ... 7).contains(v)
        {
            return v
        }
        return 7
    }()
    /// Maximum branch leaves placed for `k < 5` (`CBV2_DYN_TREE_MAX_LEAVES`, default 5).
    public static let dynamicTreeMaxLeaves: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_MAX_LEAVES"],
            let v = Int(raw), (0 ... 6).contains(v)
        {
            return v
        }
        return 5
    }()
    /// Maximum branch slots placed for `k == 5` (`CBV2_DYN_TREE_MAX_LEAVES_5`, defaults to `dynamicTreeMaxLeaves`).
    public static let dynamicTreeMaxLeavesAt5: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_MAX_LEAVES_5"],
            let v = Int(raw), (0 ... 6).contains(v)
        {
            return v
        }
        return dynamicTreeMaxLeaves
    }()
    /// Maximum branch slots placed for `k >= 6` (`CBV2_DYN_TREE_MAX_LEAVES_6`, defaults to `dynamicTreeMaxLeavesAt5`).
    public static let dynamicTreeMaxLeavesAt6: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_MAX_LEAVES_6"],
            let v = Int(raw), (0 ... 6).contains(v)
        {
            return v
        }
        return dynamicTreeMaxLeavesAt5
    }()
    /// Maximum branch length after forking (`CBV2_DYN_TREE_BRANCH_LEN`, default 1; 1 = leaves only).
    public static let dynamicTreeMaxBranchLen: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_BRANCH_LEN"],
            let v = Int(raw), (1 ... 4).contains(v)
        {
            return v
        }
        return 1
    }()
    /// Conditional probability multiplier for depth-2+ branch continuations (`CBV2_DYN_TREE_CONT_DECAY`, default 0.75).
    public static let dynamicTreeContDecay: Float = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_CONT_DECAY"],
            let v = Float(raw), v.isFinite, v >= 0.0, v <= 2.0
        {
            return v
        }
        return 0.75
    }()
    /// Maximum parent chain depth allowed to host branch leaves (`CBV2_DYN_TREE_MAX_DEPTH`, default 5).
    public static let dynamicTreeMaxLeafDepth: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_MAX_DEPTH"],
            let v = Int(raw), (1 ... 6).contains(v)
        {
            return v
        }
        return 5
    }()
    /// Softmax temperature for GPU-dynamic tree scoring (`CBV2_DYN_TREE_TEMP`, default 0.17).
    public static let dynamicTreeTemp: Float = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_TEMP"],
            let v = Float(raw), v.isFinite, v > 0.05, v <= 10.0
        {
            return v
        }
        return 0.17
    }()
    /// Chain EV multiplier (`CBV2_DYN_TREE_BONUS`, default 1.0; `0.0` forces best EV leaf(s)).
    public static let dynamicTreeChainBonus: Float = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_BONUS"],
            let v = Float(raw), v.isFinite, v >= 0.0, v <= 5.0
        {
            return v
        }
        return 1.0
    }()
    /// Leaf purity discount at depth >= 1 (`CBV2_DYN_TREE_PURITY=1` multiplies by `surv[r]`).
    public static let dynamicTreePurityMode: Int = {
        ProcessInfo.processInfo.environment["CBV2_DYN_TREE_PURITY"].flatMap(Int.init) ?? 0
    }()
    /// Survival decay multiplier per depth (`CBV2_DYN_TREE_DECAY`, default 1.0).
    public static let dynamicTreeDecay: Float = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_DECAY"],
            let v = Float(raw), v.isFinite, v > 0.1, v <= 2.0
        {
            return v
        }
        return 1.0
    }()
    /// Exact global DP tree selection over chain length L (`CBV2_DYN_TREE_DP`, default 1).
    public static let dynamicTreeDPMode: Int = {
        ProcessInfo.processInfo.environment["CBV2_DYN_TREE_DP"].flatMap(Int.init) ?? 1
    }()
    /// Candidates per position considered in dynamic tree (`CBV2_DYN_TREE_TOPK`, 3...5, default 5).
    public static let dynamicTreeTopK: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_TOPK"],
            let v = Int(raw), (3 ... 5).contains(v)
        {
            return v
        }
        return 5
    }()
    /// Per-depth temperature slope `T(r) = max(0.05, T_0 + r * slope)` (`CBV2_DYN_TREE_TEMP_SLOPE`, default 0.0).
    public static let dynamicTreeTempSlope: Float = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_TEMP_SLOPE"],
            let v = Float(raw), v.isFinite, v >= -0.2, v <= 1.0
        {
            return v
        }
        return 0.0
    }()
    /// Per-depth leaf discount `pow(leafDecay, r)` (`CBV2_DYN_TREE_LEAF_DECAY`, default 1.0).
    public static let dynamicTreeLeafDecay: Float = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_LEAF_DECAY"],
            let v = Float(raw), v.isFinite, v > 0.1, v <= 2.0
        {
            return v
        }
        return 1.0
    }()
    /// Continuous leaf purity exponent `pow(surv[r], purityExp)` for `r >= 1` (`CBV2_DYN_TREE_PURITY_EXP`, default 0.5).
    public static let dynamicTreePurityExp: Float = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_DYN_TREE_PURITY_EXP"],
            let v = Float(raw), v.isFinite, v >= 0.0, v <= 3.0
        {
            return v
        }
        return 0.5
    }()
    /// Optional host-side cooldown after full-chain tree hit (`CBV2_TREE_COOLDOWN=1`).
    public static let treeCooldownEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_TREE_COOLDOWN"] == "1"
    /// Reserve dynamic-tree leaf slots additively instead of shrinking chain depth (`CBV2_TREE_ADDITIVE != "0"`, default on).
    public static let treeAdditiveSlot: Bool =
        ProcessInfo.processInfo.environment["CBV2_TREE_ADDITIVE"] != "0"
}
