// CBv2MTPDepthController.swift
//
// Step-global adaptive depth selection for rectangular MTP verification.
// One controller belongs to one EngineV2, so model build, assistant revision,
// chip class, and target/assistant quantization are naturally isolated by the
// loaded engine. Within that engine, learned state is keyed by the planned
// decode-row bucket. Adaptive stateless workload estimates reset across
// request generations; exact verification-shape warmup persists.

import Foundation

struct CBv2MTPDepthDecision: Equatable {
    let depth: Int
    let decodeRowBucket: Int
    let reason: String
    let isExploration: Bool
}

struct CBv2MTPControllerSnapshot {
    let selectedDepth: Int
    let decodeRowBucket: Int
    let conditionalAcceptance: [Double]
    let costInputs: [CBv2MTPCostInput]
}

/// Cost attribution attached to one launched engine step. The timestamp is
/// host-only; observing it at finalize adds no MLX synchronization.
struct CBv2MTPStepMeasurement {
    let decision: CBv2MTPDepthDecision
    let actualDepth: Int
    let costEligible: Bool
    /// True when this interval overlaps either predecessor finalization or
    /// successor construction because the step participated in a chain.
    var chained: Bool
    let seedOnly: Bool
    /// Captured at launch so finalizing an older chained step cannot train a reused ID.
    var workloadGeneration: UInt64? = nil

    func excludingAssistantPrefill(_ includesPrefill: Bool) -> Self {
        .init(
            decision: decision, actualDepth: actualDepth,
            costEligible: costEligible && !includesPrefill,
            chained: chained, seedOnly: seedOnly,
            workloadGeneration: workloadGeneration)
    }
}

final class CBv2MTPDepthController {
    private static let costAlpha = 0.3
    private static let costClampFraction = 0.25
    private static let acceptanceAlpha = 0.1
    private static let acceptanceMinSamples = 10
    private static let hysteresisFraction = 0.05
    private static let baseProbeInterval = 8
    private static let maxProbeInterval = 256
    private static let committedBaselineMinSamples = 3

    private struct CostState {
        var samples = 0
        var ewmaNanos = 0.0
        var totalNanos: UInt64 = 0

        mutating func observe(_ nanos: UInt64, clampInnovation: Bool = true) {
            guard nanos > 0 else { return }
            let sample = Double(nanos)
            if samples == 0 {
                ewmaNanos = sample
            } else {
                let limit = SelfLimit.fraction * ewmaNanos
                let innovation =
                    clampInnovation
                    ? min(max(sample - ewmaNanos, -limit), limit) : sample - ewmaNanos
                ewmaNanos += CBv2MTPDepthController.costAlpha * innovation
            }
            samples += 1
            totalNanos &+= nanos
        }

        private enum SelfLimit {
            static let fraction = CBv2MTPDepthController.costClampFraction
        }
    }

    private struct AcceptanceState {
        /// Index zero is unused so the draft-position math is 1-based.
        var rates: [Double] = [0]
        var seen: [Int] = [0]

        mutating func observe(drafted: Int, accepted: Int) {
            guard drafted > 0 else { return }
            for position in 1 ... drafted {
                if accepted < position - 1 { break }
                grow(to: position)
                let outcome = accepted >= position ? 1.0 : 0.0
                if seen[position] == 0 {
                    rates[position] = outcome
                } else {
                    rates[position] +=
                        CBv2MTPDepthController.acceptanceAlpha
                        * (outcome - rates[position])
                }
                seen[position] += 1
            }
        }

        func rate(at position: Int) -> Double {
            if position > 0, position < seen.count,
                seen[position] >= CBv2MTPDepthController.acceptanceMinSamples
            {
                return rates[position]
            }
            guard position > 1 else { return 1 }
            for prior in stride(from: position - 1, through: 1, by: -1) {
                if prior < seen.count,
                    seen[prior] >= CBv2MTPDepthController.acceptanceMinSamples
                {
                    return rates[prior]
                }
            }
            return 1
        }

        func expectedCommitted(depth: Int) -> Double {
            guard depth > 0 else { return 1 }
            var total = 1.0
            var prefixProbability = 1.0
            for position in 1 ... depth {
                prefixProbability *= rate(at: position)
                total += prefixProbability
            }
            return total
        }

        var frontier: Int {
            var result = 0
            guard seen.count > 1 else { return result }
            for position in 1 ..< seen.count {
                guard seen[position] >= CBv2MTPDepthController.acceptanceMinSamples else {
                    break
                }
                result = position
            }
            return result
        }

        var trustedRates: [Double] {
            guard frontier > 0 else { return [] }
            return (1 ... frontier).map { rates[$0] }
        }

        private mutating func grow(to position: Int) {
            while seen.count <= position {
                seen.append(0)
                rates.append(0)
            }
        }
    }

    private struct BucketState {
        var costs: [Int: CostState] = [:]
        var committedBaseline = CostState()
        var committedTokensPerRow: [Int: Double] = [:]
        var committedWindow: CBv2MTPCommittedWindow?
        var acceptance = AcceptanceState()
        var activeDepth = 0
        var probeInterval = CBv2MTPDepthController.baseProbeInterval
        var roundsSinceProbe = 0
    }

    let maxDepth: Int
    let fixedDepth: Int?
    /// Fixed depth by planned decode rows (`CBv2MTPConfig.fixedDraftTokensByDecodeRows`).
    let fixedDepthByDecodeRows: [Int]?
    let usesCommittedDecodeBaseline: Bool
    private var buckets: [Int: BucketState] = [:]
    private struct VerificationShape: Hashable {
        let rowCount: Int
        let depth: Int
    }

    private var warmedVerificationShapes: Set<VerificationShape> = []
    private var workloadRows: [CBv2RequestID] = []
    private var lastDecision = CBv2MTPDepthDecision(
        depth: 0, decodeRowBucket: 0, reason: "inactive", isExploration: false)

    init(
        maxDepth: Int, fixedDepth: Int?, fixedDepthByDecodeRows: [Int]? = nil,
        useCommittedDecodeBaseline: Bool = false
    ) {
        let resolvedMax = min(max(maxDepth, 0), CBv2MTPConfig.testedMaxDraftTokens)
        self.maxDepth = resolvedMax
        self.fixedDepth = fixedDepth.map { min(max($0, 0), resolvedMax) }
        // Only a fixed depth takes a width table; an empty table is none.
        let table =
            fixedDepth == nil || fixedDepthByDecodeRows?.isEmpty != false
            ? nil : fixedDepthByDecodeRows
        self.fixedDepthByDecodeRows = table.map { $0.map { min(max($0, 0), resolvedMax) } }
        self.usesCommittedDecodeBaseline = useCommittedDecodeBaseline && fixedDepth == nil
    }

    static func decodeRowBucket(_ rows: Int) -> Int {
        guard rows > 0 else { return 0 }
        var bucket = 1
        while bucket < rows { bucket *= 2 }
        return bucket
    }

    func preview(plannedDecodeRows: Int, canSpeculate: Bool) -> CBv2MTPDepthDecision {
        decide(plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate, mutate: false)
    }

    func select(plannedDecodeRows: Int, canSpeculate: Bool) -> CBv2MTPDepthDecision {
        decide(plannedDecodeRows: plannedDecodeRows, canSpeculate: canSpeculate, mutate: true)
    }

    /// Request membership changes require a fresh ordinary baseline and fresh
    /// workload profitability. Keep only shape warmup knowledge across requests.
    func beginWorkload(rowIDs: [CBv2RequestID]) {
        guard usesCommittedDecodeBaseline, Set(rowIDs) != Set(workloadRows) else { return }
        workloadRows = rowIDs
        let bucket = Self.decodeRowBucket(rowIDs.count)
        guard bucket > 0 else { return }
        let previous = buckets[bucket] ?? BucketState()
        var fresh = BucketState()
        fresh.costs[0] = previous.costs[0]
        buckets[bucket] = fresh
    }

    /// Finishing a request ends this workload generation even if the caller
    /// later reuses its numeric ID. Old in-flight observations must be dropped
    /// by the driver; only shape knowledge and isolated depth-zero warmup survive.
    func invalidateWorkload() {
        guard usesCommittedDecodeBaseline else { return }
        let bucket = Self.decodeRowBucket(workloadRows.count)
        workloadRows.removeAll(keepingCapacity: true)
        guard bucket > 0 else { return }
        var fresh = BucketState()
        fresh.costs[0] = buckets[bucket]?.costs[0]
        buckets[bucket] = fresh
    }

    /// Invalid/cohort-changing work cannot extend a contiguous observation.
    /// Discard incomplete learning windows; all executed work remains in the
    /// driver's cumulative telemetry, and ordinary request limits still win.
    func cancelCommittedWindow(decodeRowBucket: Int) {
        guard usesCommittedDecodeBaseline else { return }
        buckets[decodeRowBucket]?.committedWindow = nil
    }

    /// Update profitability only after a bounded contiguous window. Both
    /// seed and verify outputs accompany their time, and a lone rejected
    /// token cannot replace an otherwise profitable active-mode estimate.
    @discardableResult
    func recordCommittedVerification(
        decision: CBv2MTPDepthDecision, wallTimeNanos: UInt64,
        committedTokens: Int, rowCount: Int
    ) -> Bool {
        guard usesCommittedDecodeBaseline, decision.depth > 0,
            decision.depth <= maxDepth, wallTimeNanos > 0,
            committedTokens > 0, rowCount > 0,
            Self.decodeRowBucket(rowCount) == decision.decodeRowBucket
        else {
            cancelCommittedWindow(decodeRowBucket: decision.decodeRowBucket)
            return false
        }
        var state = buckets[decision.decodeRowBucket] ?? BucketState()
        var window =
            state.committedWindow
            ?? CBv2MTPCommittedWindow(decision: decision, rowCount: rowCount)
        guard window.decision.depth == decision.depth, window.rowCount == rowCount else {
            state.committedWindow = nil
            buckets[decision.decodeRowBucket] = state
            return false
        }
        // Buckets group costs, but Metal verifies the exact physical row count.
        // Three and four rows share a bucket without sharing their first compile.
        let warmup = warmedVerificationShapes.insert(
            VerificationShape(rowCount: rowCount, depth: decision.depth)
        ).inserted
        window.observe(
            wallTimeNanos: wallTimeNanos, committedTokens: committedTokens, warmup: warmup)
        guard window.isComplete else {
            state.committedWindow = window
            buckets[decision.decodeRowBucket] = state
            return false
        }
        state.committedWindow = nil
        var wall = state.costs[decision.depth] ?? CostState()
        // Use identical EWMA weights for whole-window time and outputs.
        // Clamping time alone would undercharge seed transitions while
        // crediting every extra output, manufacturing apparent profit.
        wall.observe(window.wallTimeNanos, clampInnovation: false)
        state.costs[decision.depth] = wall
        let tokensPerRow = Double(window.committedTokens) / Double(rowCount)
        let previous = state.committedTokensPerRow[decision.depth] ?? tokensPerRow
        state.committedTokensPerRow[decision.depth] =
            previous + Self.costAlpha * (tokensPerRow - previous)
        complete(window.decision, state: &state)
        buckets[decision.decodeRowBucket] = state
        return true
    }

    func observeAcceptance(decodeRowBucket: Int, drafted: Int, accepted: Int) {
        guard decodeRowBucket > 0, drafted > 0 else { return }
        var state = buckets[decodeRowBucket] ?? BucketState()
        state.acceptance.observe(drafted: drafted, accepted: accepted)
        buckets[decodeRowBucket] = state
    }

    func observeCost(decodeRowBucket: Int, depth: Int, wallTimeNanos: UInt64) {
        guard decodeRowBucket > 0, depth >= 0, depth <= maxDepth, wallTimeNanos > 0 else {
            return
        }
        var state = buckets[decodeRowBucket] ?? BucketState()
        var cost = state.costs[depth] ?? CostState()
        cost.observe(wallTimeNanos)
        state.costs[depth] = cost
        buckets[decodeRowBucket] = state
    }

    /// Record one non-overlapping steady ordinary-decode interval. The
    /// engine supplies commit-to-commit time for an unchanged row cohort,
    /// never a chained step's overlapping launch-to-finalize latency.
    func observeCommittedDecodeInterval(decodeRowBucket: Int, wallTimeNanos: UInt64) {
        guard usesCommittedDecodeBaseline, decodeRowBucket > 0, wallTimeNanos > 0 else { return }
        var state = buckets[decodeRowBucket] ?? BucketState()
        state.committedBaseline.observe(wallTimeNanos)
        buckets[decodeRowBucket] = state
    }

    /// Retain one isolated warmup per bucket. Adaptive stateless target-prefix
    /// serving then calibrates its actual chained alternative separately;
    /// stateful/legacy policies continue using the isolated baseline.
    func requiresNonChainedDepthZeroProbe(_ decision: CBv2MTPDepthDecision) -> Bool {
        guard decision.depth == 0, decision.decodeRowBucket > 0 else { return false }
        return buckets[decision.decodeRowBucket]?.costs[0] == nil
    }

    /// Commit one completed controller sample. Positive depths require a
    /// finalized verification at exactly the requested depth. Chained
    /// depth-zero work advances normal-round cadence but never contributes a
    /// wall-cost sample because its elapsed interval overlaps neighboring
    /// graph construction/finalization.
    @discardableResult
    func recordFinalizedStep(
        decision: CBv2MTPDepthDecision,
        actualDepth: Int,
        wallTimeNanos: UInt64,
        costEligible: Bool,
        chained: Bool,
        finalizedPlainWork: Bool,
        finalizedVerification: Bool
    ) -> Bool {
        guard decision.decodeRowBucket > 0,
            actualDepth == decision.depth,
            actualDepth >= 0,
            actualDepth <= maxDepth
        else { return false }

        if actualDepth > 0 {
            guard finalizedVerification, costEligible, !chained, wallTimeNanos > 0 else {
                return false
            }
        } else {
            guard finalizedPlainWork else { return false }
            if chained {
                var state = buckets[decision.decodeRowBucket] ?? BucketState()
                // A warmup baseline must be measured non-chained. Once it
                // exists, completed chained target work may drive the bounded
                // exploration cadence without polluting the cost curve.
                guard state.costs[0] != nil, !decision.isExploration else { return false }
                complete(decision, state: &state)
                buckets[decision.decodeRowBucket] = state
                return true
            }
            guard costEligible, wallTimeNanos > 0 else { return false }
        }

        var state = buckets[decision.decodeRowBucket] ?? BucketState()
        var cost = state.costs[actualDepth] ?? CostState()
        cost.observe(wallTimeNanos)
        state.costs[actualDepth] = cost
        complete(decision, state: &state)
        buckets[decision.decodeRowBucket] = state
        return true
    }

    func activeDepthForTesting(decodeRowBucket: Int) -> Int {
        buckets[decodeRowBucket]?.activeDepth ?? 0
    }

    func probeIntervalForTesting(decodeRowBucket: Int) -> Int {
        buckets[decodeRowBucket]?.probeInterval ?? Self.baseProbeInterval
    }

    func snapshot() -> CBv2MTPControllerSnapshot {
        var inputs: [CBv2MTPCostInput] = []
        for bucket in buckets.keys.sorted() {
            guard let state = buckets[bucket] else { continue }
            for depth in state.costs.keys.sorted() {
                guard let cost = effectiveCost(depth: depth, state: state) else { continue }
                inputs.append(
                    CBv2MTPCostInput(
                        decodeRowBucket: bucket,
                        depth: depth,
                        samples: cost.samples,
                        ewmaWallTimeNanos: UInt64(max(0, cost.ewmaNanos.rounded())),
                        totalWallTimeNanos: cost.totalNanos,
                        ewmaNanosPerCommittedToken: state.committedTokensPerRow[depth].map {
                            UInt64(max(0, (cost.ewmaNanos / $0).rounded()))
                        }))
            }
        }
        let state = buckets[lastDecision.decodeRowBucket]
        return CBv2MTPControllerSnapshot(
            selectedDepth: lastDecision.depth,
            decodeRowBucket: lastDecision.decodeRowBucket,
            conditionalAcceptance: state?.acceptance.trustedRates ?? [],
            costInputs: inputs)
    }

    private func decide(
        plannedDecodeRows: Int, canSpeculate: Bool, mutate: Bool
    ) -> CBv2MTPDepthDecision {
        let bucket = Self.decodeRowBucket(plannedDecodeRows)
        guard bucket > 0 else {
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: 0, reason: "no_decode_rows",
                    isExploration: false),
                mutate: mutate)
        }
        guard canSpeculate, maxDepth > 0 else {
            if mutate { cancelCommittedWindow(decodeRowBucket: bucket) }
            return finish(
                CBv2MTPDepthDecision(
                    depth: 0, decodeRowBucket: bucket,
                    reason: maxDepth == 0 ? "max_depth_zero" : "ineligible",
                    isExploration: false),
                mutate: mutate)
        }
        if let fixedDepth {
            let depth = fixedDepthByDecodeRows.map { table in
                min(fixedDepth, table[min(plannedDecodeRows, table.count) - 1])
            }
            return finish(
                CBv2MTPDepthDecision(
                    depth: depth ?? fixedDepth, decodeRowBucket: bucket,
                    reason: depth == nil ? "fixed" : "fixed_by_rows", isExploration: false),
                mutate: mutate)
        }

        let state = buckets[bucket] ?? BucketState()
        let limit = min(maxDepth, state.acceptance.frontier + 1)
        let decision: CBv2MTPDepthDecision

        if state.costs[0] == nil {
            decision = CBv2MTPDepthDecision(
                depth: 0, decodeRowBucket: bucket, reason: "warmup_baseline",
                isExploration: true)
        } else if usesCommittedDecodeBaseline,
            state.committedBaseline.samples < Self.committedBaselineMinSamples
        {
            // Short requests simply remain ordinary decode. Calibration
            // cannot manufacture samples from seeds, prefill, or idle time.
            decision = CBv2MTPDepthDecision(
                depth: 0, decodeRowBucket: bucket, reason: "warmup_chained_baseline",
                isExploration: false)
        } else if let window = state.committedWindow {
            decision = CBv2MTPDepthDecision(
                depth: min(window.decision.depth, limit), decodeRowBucket: bucket,
                reason: window.decision.isExploration ? "explore_window" : "goodput_window",
                isExploration: window.decision.isExploration)
        } else if let unsampled = (0 ... limit).first(where: { state.costs[$0] == nil }) {
            decision = CBv2MTPDepthDecision(
                depth: unsampled, decodeRowBucket: bucket, reason: "explore_cost",
                isExploration: true)
        } else {
            let current = min(state.activeDepth, limit)
            let currentGoodput = goodput(depth: current, state: state)
            var best = current
            var bestGoodput = currentGoodput
            for depth in 0 ... limit {
                let candidate = goodput(depth: depth, state: state)
                if candidate > bestGoodput {
                    best = depth
                    bestGoodput = candidate
                }
            }

            var selected = current
            var reason = current == 0 ? "unprofitable" : "goodput"
            if best != current {
                if (usesCommittedDecodeBaseline && best == 0)
                    || currentGoodput <= 0
                    || bestGoodput >= currentGoodput * (1 + Self.hysteresisFraction)
                {
                    selected = best
                    reason = best == 0 ? "unprofitable" : "goodput"
                } else {
                    reason = "hysteresis"
                }
            }

            var explore = false
            let nextRounds = state.roundsSinceProbe + 1
            if nextRounds >= state.probeInterval {
                let probe = min(selected + 1, limit)
                if probe > selected {
                    selected = probe
                    reason = "explore_deeper"
                    explore = true
                }
            }
            decision = CBv2MTPDepthDecision(
                depth: selected, decodeRowBucket: bucket, reason: reason,
                isExploration: explore)
        }

        return finish(decision, mutate: mutate)
    }

    private func complete(
        _ decision: CBv2MTPDepthDecision,
        state: inout BucketState
    ) {
        if decision.isExploration {
            state.roundsSinceProbe = 0
            if decision.reason == "explore_deeper" {
                state.probeInterval = min(
                    state.probeInterval * 2, Self.maxProbeInterval)
            }
            return
        }
        if decision.depth != state.activeDepth {
            state.activeDepth = decision.depth
            state.probeInterval = Self.baseProbeInterval
            state.roundsSinceProbe = 0
        } else {
            state.roundsSinceProbe += 1
        }
    }

    private func effectiveCost(depth: Int, state: BucketState) -> CostState? {
        if depth == 0, usesCommittedDecodeBaseline,
            state.committedBaseline.samples >= Self.committedBaselineMinSamples
        {
            return state.committedBaseline
        }
        return state.costs[depth]
    }

    private func goodput(depth: Int, state: BucketState) -> Double {
        if usesCommittedDecodeBaseline, depth > 0,
            let tokens = state.committedTokensPerRow[depth],
            let cost = state.costs[depth], cost.ewmaNanos > 0
        {
            return tokens / cost.ewmaNanos
        }
        guard let cost = effectiveCost(depth: depth, state: state), cost.ewmaNanos > 0 else {
            return 0
        }
        return state.acceptance.expectedCommitted(depth: depth) / cost.ewmaNanos
    }

    private func finish(
        _ decision: CBv2MTPDepthDecision, mutate: Bool
    ) -> CBv2MTPDepthDecision {
        if mutate { lastDecision = decision }
        return decision
    }
}

/// Request-owned conditional acceptance estimates for the stateful MTP marginal
/// depth policy. Hardware cost observations deliberately do not live here.
struct CBv2MTPRequestAcceptanceState: Equatable {
    /// Bounded by each drafter's qualified captured-window contract.
    static let maximumDepth = 7
    private static let alpha = 0.15

    private(set) var probabilities: [Double] =
        (0 ..< CBv2MTPRequestAcceptanceState.maximumDepth).map {
            0.85 * pow(0.98, Double($0))
        }

    /// Draft-confidence calibration: per position, the mean logit of the
    /// drafter's top-1 probability over rounds whose target outcome at that
    /// position was observed, and how many such rounds there were. Only
    /// rounds whose block confidence was read feed it.
    private(set) var confidenceLogitMeans: [Double] =
        Array(repeating: 0, count: CBv2MTPRequestAcceptanceState.maximumDepth)
    private(set) var confidenceSamples: [Int] =
        Array(repeating: 0, count: CBv2MTPRequestAcceptanceState.maximumDepth)
    /// Observed rounds at a position before its confidence modulates it.
    static let minimumConfidenceSamples = 4
    private static let confidenceLogitLimit = 9.0

    /// Request-level acceptance regime: an EWMA of each observed round's
    /// accepted / drafted fraction. Starts at 1 (a request begins in the
    /// high-acceptance regime). The fixed-depth window may shorten a round
    /// below the declared depth only while this is under
    /// `lowAcceptanceRegimeThreshold`; at or above it the declared depth
    /// runs, so occasional rejections on a request that accepts nearly every
    /// draft never shrink its window.
    private(set) var roundAcceptanceFraction = 1.0
    private(set) var observedRegimeRounds = 0
    private(set) var enteredLowAcceptanceRegime = false
    /// Tree-invariant per-slot acceptance EWMA (`accepted / verifyDraftWidth`) updated on
    /// every non-output-truncated round (including terminal tree branch hits), so tree rounds
    /// with short `chainDepth` never falsely exit low-acceptance regimes.
    private(set) var slotAcceptanceFraction = 1.0
    private(set) var observedSlotRounds = 0
    private(set) var enteredLowSlotRegime = false
    private static let regimeAlpha = 0.15
    static let lowAcceptanceRegimeThreshold = 0.75
    static let lowAcceptanceExitThreshold = 0.85
    static let fastRegimeEntryV2: Bool =
        ProcessInfo.processInfo.environment["CBV2_FAST_REGIME_V2"] != "0"
    static let regimeHysteresisEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_REGIME_HYST"] != "0"
    static let slotRegimeEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_SLOT_REGIME"] != "0"

    var meanCommittedPerRound: Double {
        ceilingRounds > 0 ? Double(ceilingCommittedTokens) / Double(ceilingRounds) : 8.0
    }

    static let deepLowMaxMeanCommitted: Double = {
        ProcessInfo.processInfo.environment["CBV2_DEEP_LOW_MAX_COMMIT"].flatMap(Double.init) ?? 5.0
    }()

    var inLowAcceptanceRegime: Bool {
        if Self.slotRegimeEnabled {
            if Self.regimeHysteresisEnabled && enteredLowSlotRegime {
                return slotAcceptanceFraction < Self.lowAcceptanceExitThreshold
            }
            return slotAcceptanceFraction < Self.lowAcceptanceRegimeThreshold
        }
        if Self.regimeHysteresisEnabled && enteredLowAcceptanceRegime {
            return roundAcceptanceFraction < Self.lowAcceptanceExitThreshold
        }
        return roundAcceptanceFraction < Self.lowAcceptanceRegimeThreshold
    }

    static let deepLowAcceptanceMinRounds: Int = {
        ProcessInfo.processInfo.environment["CBV2_DEEP_LOW_MIN_ROUNDS"].flatMap(Int.init) ?? 3
    }()
    static let deepLowAcceptanceThreshold: Double = {
        ProcessInfo.processInfo.environment["CBV2_DEEP_LOW_THRESHOLD"].flatMap(Double.init) ?? 0.55
    }()

    /// Stricter low-acceptance gate for approximate Stage-1 draft-head screening:
    /// requires at least `deepLowAcceptanceMinRounds` observed rounds, an EWMA slot
    /// acceptance fraction strictly below `deepLowAcceptanceThreshold`, and mean committed
    /// tokens per round strictly below `deepLowMaxMeanCommitted` so high-acceptance windows
    /// never invoke approximate Stage-1 screening.
    var isDeepLowAcceptanceRegime: Bool {
        if Self.slotRegimeEnabled {
            return inLowAcceptanceRegime
                && observedSlotRounds >= Self.deepLowAcceptanceMinRounds
                && slotAcceptanceFraction < Self.deepLowAcceptanceThreshold
                && meanCommittedPerRound < Self.deepLowMaxMeanCommitted
        }
        return observedRegimeRounds >= Self.deepLowAcceptanceMinRounds
            && roundAcceptanceFraction < Self.deepLowAcceptanceThreshold
    }

    static let deepLowVerifyMinRounds: Int = {
        ProcessInfo.processInfo.environment["CBV2_DEEP_VERIFY_MIN_ROUNDS"].flatMap(Int.init) ?? 3
    }()
    static let deepLowVerifyThreshold: Double = {
        ProcessInfo.processInfo.environment["CBV2_DEEP_VERIFY_THRESHOLD"].flatMap(Double.init) ?? 0.62
    }()

    /// Stricter low-acceptance gate for exact verify fast paths so
    /// high-acceptance windows stay strictly below the decode speedup ceiling.
    var isDeepLowVerifyRegime: Bool {
        if Self.slotRegimeEnabled {
            return inLowAcceptanceRegime
                && observedSlotRounds >= Self.deepLowVerifyMinRounds
                && slotAcceptanceFraction < Self.deepLowVerifyThreshold
                && meanCommittedPerRound < Self.deepLowMaxMeanCommitted
        }
        return observedRegimeRounds >= Self.deepLowVerifyMinRounds
            && roundAcceptanceFraction < Self.deepLowVerifyThreshold
    }

    /// Acceptance ceiling: a per-request rate limiter on committed tokens
    /// per speculative round. The request's own running totals (rounds that
    /// drafted, tokens those rounds committed) bound the next round's draft
    /// count so that, even if every draft of every round is accepted, the
    /// request commits at most
    /// `ceilingBudgetHundredths / 100 * rounds + ceilingBurstTokens` tokens.
    /// Inputs are accept/reject history only: integer host arithmetic, no
    /// clock, no device state, no token values. It changes how many drafts a
    /// round verifies, never which tokens are committed.
    private(set) var ceilingRounds = 0
    private(set) var ceilingCommittedTokens = 0
    /// The budget line, one pair of constants: committed tokens after n
    /// speculative rounds stay at or under
    /// `ceilingBudgetHundredths / 100 * n + ceilingBurstTokens`.
    /// It is a rounds-per-window limit: (4.04, 27) allows at most 123
    /// tokens in 24 rounds and exactly 128 in 25, so a 128-token window
    /// takes at least 25 rounds whatever the acceptance. Any budget from 389 to 403 with the same
    /// burst gives 26 rounds; 404 to 420 gives 25, 421 to 439 gives 24,
    /// 440 to 459 gives 23 and 460 to 480 gives 22. The line follows the
    /// verify round's cost: shorter rounds need more of them per window.
    /// Low-acceptance requests commit far fewer tokens per round than the
    /// line allows, so it never binds for them.
    static let ceilingBudgetHundredths: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_MTP_CEILING_BUDGET"],
            let v = Int(raw), v > 0
        {
            return v
        }
        return 404
    }()
    static let fastRegimeEntry: Bool =
        ProcessInfo.processInfo.environment["CBV2_FAST_REGIME"] != "0"
    static let ceilingBurstTokens = 27
    static let ceilingEnabled =
        ProcessInfo.processInfo.environment["CBV2_MTP_ACCEPTANCE_CEILING"] != "0"

    /// Records one finalized speculative round for the acceptance ceiling.
    mutating func observeCommitted(draftedDepth: Int, committedTokens: Int) {
        guard draftedDepth > 0 else { return }
        ceilingRounds += 1
        ceilingCommittedTokens += max(committedTokens, 0)
    }

    /// The largest draft count k <= `offeredDepth` whose full acceptance
    /// (k + 1 committed tokens) keeps the request on or under the budget
    /// line after this round; at least one draft while `offeredDepth > 0`.
    func acceptanceCeilingDepth(offeredDepth: Int) -> Int {
        guard Self.ceilingEnabled, offeredDepth > 0 else { return max(offeredDepth, 0) }
        let allowedAfterRound =
            Self.ceilingBudgetHundredths * (ceilingRounds + 1) / 100 + Self.ceilingBurstTokens
        let room = allowedAfterRound - ceilingCommittedTokens - 1
        return min(offeredDepth, max(1, room))
    }

    init() {}

    /// Every position starts at `flatPrior`. The fixed-depth acceptance
    /// window uses an optimistic prior so a request's first rounds run at
    /// the declared depth, exactly as an unclamped fixed-depth engine does.
    init(flatPrior: Double) {
        let prior = flatPrior.isFinite ? min(max(flatPrior, 0), 1) : 0
        probabilities = Array(repeating: prior, count: Self.maximumDepth)
    }

    /// Records positions whose target outcome was actually observed and, after
    /// a fully accepted round, transfers bounded optimism to the next position.
    /// A truncation (stop, token budget, or common-width clamp) passes
    /// `rejectionObserved: false` and `endedByTruncation: true`, so it never
    /// manufactures either a failure or next-position optimism.
    mutating func observe(
        draftedDepth: Int,
        acceptedDepth: Int,
        rejectionObserved: Bool,
        endedByTruncation: Bool = false,
        verifyDraftWidth: Int? = nil,
        outputTruncated: Bool = false
    ) {
        let drafted = min(max(draftedDepth, 0), Self.maximumDepth)
        let accepted = min(max(acceptedDepth, 0), drafted)
        let slotDrafted = min(max(verifyDraftWidth ?? draftedDepth, 0), Self.maximumDepth)
        let slotAccepted = min(max(acceptedDepth, 0), slotDrafted)

        for position in 0 ..< accepted {
            probabilities[position] += Self.alpha * (1.0 - probabilities[position])
        }
        // Tree-invariant slot acceptance signal (uses actual draft width k on every
        // non-output-truncated round, including terminal tree branch hits).
        if slotDrafted > 0, !outputTruncated {
            let slotFrac = Double(slotAccepted) / Double(slotDrafted)
            let slotWeight: Double
            if Self.fastRegimeEntry {
                let denom = Double(observedSlotRounds + (Self.fastRegimeEntryV2 ? 1 : 2))
                slotWeight = max(Self.regimeAlpha, 1.0 / denom)
            } else {
                slotWeight = Self.regimeAlpha
            }
            slotAcceptanceFraction += slotWeight * (slotFrac - slotAcceptanceFraction)
            if observedSlotRounds < Int.max { observedSlotRounds += 1 }
            if slotAcceptanceFraction < Self.lowAcceptanceRegimeThreshold {
                enteredLowSlotRegime = true
            } else if slotAcceptanceFraction >= Self.lowAcceptanceExitThreshold {
                enteredLowSlotRegime = false
            }
        }
        // Regime signal: only rounds whose outcome was fully observed (a
        // rejection, or every draft accepted) count; a truncation does not.
        if drafted > 0,
            (rejectionObserved && accepted < drafted)
                || (!rejectionObserved && !endedByTruncation && accepted == drafted)
        {
            let fraction = Double(accepted) / Double(drafted)
            let regimeWeight: Double
            if Self.fastRegimeEntry {
                let denom = Double(observedRegimeRounds + (Self.fastRegimeEntryV2 ? 1 : 2))
                regimeWeight = max(Self.regimeAlpha, 1.0 / denom)
            } else {
                regimeWeight = Self.regimeAlpha
            }
            roundAcceptanceFraction +=
                regimeWeight * (fraction - roundAcceptanceFraction)
            if observedRegimeRounds < Int.max { observedRegimeRounds += 1 }
            if roundAcceptanceFraction < Self.lowAcceptanceRegimeThreshold {
                enteredLowAcceptanceRegime = true
            } else if roundAcceptanceFraction >= Self.lowAcceptanceExitThreshold {
                enteredLowAcceptanceRegime = false
            }
        }
        if rejectionObserved, accepted < drafted {
            probabilities[accepted] += Self.alpha * (0.0 - probabilities[accepted])
        } else if !rejectionObserved, !endedByTruncation,
            drafted > 0, accepted == drafted, drafted < Self.maximumDepth,
            probabilities[drafted] < 0.95
        {
            // A fully accepted round is bounded evidence that the hot chain
            // may profitably extend one position farther.
            probabilities[drafted] += Self.alpha * (0.95 - probabilities[drafted])
        }
    }

    /// Records the drafter's per-position top-1 probabilities for a round at
    /// the positions whose outcome the round observed (the same positions
    /// `observe` updates, without the full-acceptance extension).
    mutating func observeConfidence(
        _ confidence: [Double],
        draftedDepth: Int,
        acceptedDepth: Int,
        rejectionObserved: Bool
    ) {
        let drafted = min(max(draftedDepth, 0), Self.maximumDepth, confidence.count)
        let accepted = min(max(acceptedDepth, 0), drafted)
        let observed = rejectionObserved && accepted < drafted ? accepted + 1 : accepted
        for position in 0 ..< observed {
            let value = confidence[position]
            guard value.isFinite else { continue }
            let n = confidenceSamples[position]
            let weight = max(Self.alpha, 1.0 / Double(n + 1))
            confidenceLogitMeans[position] +=
                weight * (Self.clampedLogit(value) - confidenceLogitMeans[position])
            confidenceSamples[position] = n == Int.max ? n : n + 1
        }
    }

    /// The acceptance history shifted, in logit space, by how far this
    /// round's drafter confidence sits from its calibrated mean at each
    /// position. A position with too few calibration rounds, or without a
    /// finite confidence, keeps its history probability, so with no
    /// calibration this is exactly `probabilities`.
    func confidenceAdjustedProbabilities(_ confidence: [Double]) -> [Double] {
        probabilities.indices.map { position in
            guard position < confidence.count, confidence[position].isFinite,
                confidenceSamples[position] >= Self.minimumConfidenceSamples
            else { return probabilities[position] }
            let shifted =
                Self.clampedLogit(probabilities[position])
                + Self.clampedLogit(confidence[position]) - confidenceLogitMeans[position]
            return 1.0 / (1.0 + exp(-shifted))
        }
    }

    private static func clampedLogit(_ probability: Double) -> Double {
        let p = min(max(probability, 0), 1)
        let raw = log(p) - log1p(-p)
        return raw.isFinite
            ? min(max(raw, -confidenceLogitLimit), confidenceLogitLimit)
            : (p >= 0.5 ? confidenceLogitLimit : -confidenceLogitLimit)
    }
}

/// Engine-shared raw, nonchained wall-cost estimates for the marginal policy.
/// Callers record the isolated interval itself: seed-attributed cost has no
/// input in this API and therefore cannot contaminate the inferred slope.
struct CBv2MTPRawCostEstimator {
    static let bootstrapHeadStepCostRatio = 0.18
    private static let maximumDepth = CBv2MTPRequestAcceptanceState.maximumDepth
    private static let alpha = 0.3
    private static let clampFraction = 0.25

    private struct Cost {
        var samples = 0
        var ewmaNanos = 0.0

        mutating func observe(_ sample: Double) {
            if samples == 0 {
                ewmaNanos = sample
            } else {
                let limit = CBv2MTPRawCostEstimator.clampFraction * ewmaNanos
                let innovation = min(max(sample - ewmaNanos, -limit), limit)
                ewmaNanos += CBv2MTPRawCostEstimator.alpha * innovation
            }
            samples += 1
        }
    }

    private var costs: [Int: Cost] = [:]
    private var positiveDepthWarmups: Set<Int> = []

    /// Returns whether the raw sample entered the steady-state estimate.
    /// The first isolated sample at each positive depth is a compile/JIT
    /// warm-up: remember that it occurred, but never anchor the clamped EWMA
    /// to it. Depth zero uses its first sample because the target is already
    /// compiled before the nonchained baseline probe.
    @discardableResult
    mutating func observe(
        depth: Int,
        rawWallTimeNanos: Double,
        chained: Bool = false
    ) -> Bool {
        guard depth >= 0, depth <= Self.maximumDepth,
            !chained, rawWallTimeNanos.isFinite, rawWallTimeNanos > 0
        else { return false }

        if depth > 0, positiveDepthWarmups.insert(depth).inserted {
            return false
        }
        var cost = costs[depth] ?? Cost()
        cost.observe(rawWallTimeNanos)
        costs[depth] = cost
        return true
    }

    /// Sample-count-weighted h_k = max(0, (Ck/C0 - 1)/k). Until both C0 and
    /// at least one positive-depth Ck exist, use the measured bootstrap.
    var headStepCostRatio: Double {
        guard let baseline = costs[0], baseline.ewmaNanos.isFinite,
            baseline.ewmaNanos > 0
        else { return Self.bootstrapHeadStepCostRatio }

        var weightedSlope = 0.0
        var weight = 0
        for depth in 1 ... Self.maximumDepth {
            guard let cost = costs[depth],
                cost.samples > 0, cost.ewmaNanos.isFinite, cost.ewmaNanos > 0
            else {
                continue
            }
            let normalizedIncrement =
                cost.ewmaNanos <= baseline.ewmaNanos
                ? 0
                : (cost.ewmaNanos - baseline.ewmaNanos) / baseline.ewmaNanos
            let rawSlope = normalizedIncrement / Double(depth)
            let slope = rawSlope.isFinite ? rawSlope : Double.greatestFiniteMagnitude
            let newWeight = weight + cost.samples
            let fraction = Double(cost.samples) / Double(newWeight)
            // Incremental weighting cannot overflow for nonnegative finite
            // slopes, unlike accumulating `slope * samples`.
            weightedSlope += (slope - weightedSlope) * fraction
            weight = newWeight
        }
        guard weight > 0 else { return Self.bootstrapHeadStepCostRatio }
        return weightedSlope.isFinite ? weightedSlope : Double.greatestFiniteMagnitude
    }

    /// True until a post-warm-up isolated sample exists at `depth`. The engine
    /// may satisfy this with a bounded one-token probe even when confidence
    /// or the provisional cost estimate would otherwise select zero.
    func needsSteadyStateProbe(depth: Int) -> Bool {
        guard depth > 0, depth <= Self.maximumDepth, costs[0] != nil else {
            return false
        }
        return costs[depth] == nil
    }
    func sampleCount(depth: Int) -> Int {
        costs[depth]?.samples ?? 0
    }
}

/// Pure marginal-cost selector. One request supplies its own acceptance
/// probabilities; a caller may share only `headStepCostRatio` across requests.
enum CBv2MTPMarginalDepthPolicy {
    static let maximumDepth = CBv2MTPRequestAcceptanceState.maximumDepth

    static func selectDepth(
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int,
        acceptanceProbabilities: [Double],
        previousTargetTopTwoMargin: Double?,
        headStepCostRatio: Double
    ) -> Int {
        guard offeredDepth > 0, remainingTokens > 1, verificationLimit > 0 else {
            return 0
        }
        let cap = min(
            maximumDepth,
            min(offeredDepth, min(remainingTokens - 1, verificationLimit)))
        guard cap > 0 else { return 0 }

        let h =
            headStepCostRatio.isFinite && headStepCostRatio >= 0
            ? headStepCostRatio
            : CBv2MTPRawCostEstimator.bootstrapHeadStepCostRatio
        var reach = 1.0
        var expected = 0.0
        var depth = 0
        while depth < cap {
            let rawProbability =
                depth < acceptanceProbabilities.count
                ? acceptanceProbabilities[depth]
                : 0
            let probability = cappedAcceptanceProbability(
                position: depth,
                acceptanceProbability: rawProbability,
                previousMargin: previousTargetTopTwoMargin)
            reach *= probability
            let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)
            guard reach > threshold else { break }
            expected += reach
            depth += 1
        }
        return depth
    }

    /// Fixed-depth acceptance window. A fixed declared depth is a ceiling:
    /// each round verifies `min(declared, window)` drafts, where the window
    /// is the marginal rule above over the request's own acceptance history
    /// (no wall-clock input, so the choice is deterministic for a given
    /// accept/reject sequence). The floor of one draft keeps the history
    /// observing: a depth-zero plan would record no acceptance and could
    /// never reopen the window. Emitted tokens are the target's own either
    /// way; only the number of verify rows changes.
    static let fixedWindowFlatPrior = 0.9
    /// One extra verify row as a fraction of a full target step. Round costs
    /// at declared depths 3/5/7 on a request that accepts every draft give a
    /// slope near 0.11-0.12 of the serial step; 0.15 is deliberately a little
    /// conservative so estimator noise does not open windows that pay for
    /// rejected rows.
    static let fixedWindowHeadStepCostRatio: Double = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_HEAD_STEP_COST"],
            let v = Double(raw), v.isFinite, v >= 0.0, v <= 0.5
        {
            return v
        }
        return 0.065
    }()

    static func fixedWindowDepth(
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int,
        acceptanceProbabilities: [Double]
    ) -> Int {
        guard offeredDepth > 0 else { return 0 }
        let selected = selectDepth(
            offeredDepth: offeredDepth,
            remainingTokens: remainingTokens,
            verificationLimit: verificationLimit,
            acceptanceProbabilities: acceptanceProbabilities,
            previousTargetTopTwoMargin: nil,
            headStepCostRatio: fixedWindowHeadStepCostRatio)
        return min(offeredDepth, max(1, selected))
    }

    /// Draft-confidence window (LibraSpec, arXiv 2608.08721 §3.2-3.3, with a
    /// per-request calibration): the fixed-depth acceptance window evaluated
    /// on the history probabilities shifted by this round's drafter
    /// confidence (`confidenceAdjustedProbabilities`). The marginal rule is
    /// LibraSpec's greedy beneficial extension of E[accepted + 1] / round
    /// cost with cost linear in verify rows. Same ceiling and floor as
    /// `fixedWindowDepth`; equal to it before calibration.
    static func fixedWindowConfidenceDepth(
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int,
        acceptance: CBv2MTPRequestAcceptanceState,
        confidence: [Double]
    ) -> Int {
        fixedWindowDepth(
            offeredDepth: offeredDepth,
            remainingTokens: remainingTokens,
            verificationLimit: verificationLimit,
            acceptanceProbabilities: acceptance.confidenceAdjustedProbabilities(confidence))
    }

    /// A cost-confirmation probe is always at most one draft and obeys every
    /// ordinary capacity limit. Whether a probe is due remains engine cadence
    /// state; this helper keeps its depth choice pure and deterministic.
    static func boundedProbeDepth(
        offeredDepth: Int,
        remainingTokens: Int,
        verificationLimit: Int
    ) -> Int {
        guard offeredDepth > 0, remainingTokens > 1, verificationLimit > 0 else {
            return 0
        }
        return min(1, min(offeredDepth, min(remainingTokens - 1, verificationLimit)))
    }

    private static func cappedAcceptanceProbability(
        position: Int,
        acceptanceProbability: Double,
        previousMargin: Double?
    ) -> Double {
        guard acceptanceProbability.isFinite else { return 0 }
        var probability = min(max(acceptanceProbability, 0), 1)
        guard position == 0 || position == 1,
            let margin = previousMargin, margin.isFinite
        else { return probability }

        let divisor = position == 0 ? 2.0 : 3.0
        probability = min(probability, sigmoid(margin / divisor))
        return probability
    }

    private static func sigmoid(_ value: Double) -> Double {
        if value >= 0 {
            return 1.0 / (1.0 + exp(-value))
        }
        let exponential = exp(value)
        return exponential / (1.0 + exponential)
    }
}
