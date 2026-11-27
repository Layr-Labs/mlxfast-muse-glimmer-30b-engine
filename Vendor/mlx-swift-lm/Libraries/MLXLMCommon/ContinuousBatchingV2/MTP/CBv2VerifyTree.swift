// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// One-level draft tree for low-acceptance requests (`CBV2_VERIFY_TREE=1`).
///
/// A tree round verifies, in ONE rectangular window of `1 + chain + leaves`
/// columns, the request's normal chain `[seed, d1 ... dc]` followed by up to
/// `maximumLeaves` siblings: a leaf at block row `j < chain` carries the
/// drafter's rank-`r` (2 or 3) token for draft position `j`, i.e. an
/// alternative to `d(j+1)` whose parent is the chain prefix `seed, d1 ... dj`.
///
/// Leaves sit at block rows `j <= chain` (and `j < maximumLeafRow`): a leaf
/// at `j < chain` is a sibling of chain draft `d(j+1)`; a leaf at `j == chain`
/// is the alternative to the next draft after a fully accepted chain (its
/// parent is `dc`), e.g. `d4#2` on a 3-draft chain.
///
/// Exactness. A leaf column is computed as the serial step at its logical
/// position `seed + j + 1` on its own prefix:
/// - projections, norms, lm_head and the target argmax are row-exact kernels
///   (each row computed as an M=1 row), so a leaf row's bytes depend only on
///   its own inputs;
/// - RoPE: the tree variant of `VerifyFusedQK` rotates every column at its
///   logical position (chain columns at `seed + t`, exactly the record's);
/// - attention: every column's K/V is written to consecutive slots first, so
///   the leaf's own K/V sits at slot `seed + 1 + chain + leafIndex`. Its decode
///   row carries the SERIAL attend range of position `seed + j + 1` (the same
///   seqinfo as chain column `j + 1`) plus one remap: the last attended key is
///   read from the leaf's slot instead of `seed + j + 1`. Partitions, stride
///   assignment, online-softmax order and merge order are therefore the serial
///   step's. Chain columns attend `<= seed + t <= seed + chain` and never see a
///   leaf slot, so chain bytes are those of the record window.
/// - commit: when the chain accepts `a < chain` drafts and a leaf at row `a`
///   carries the target's token there, the round commits `d1 ... da`, the leaf
///   token and the target's token at the leaf column (`a + 2` tokens); the
///   leaf's K/V are copied bitwise to slot `seed + a + 1` before the rollback,
///   and the leaf's hidden row becomes the carry.
///
/// Fail-closed: a round whose forward did not take the tree RoPE on every
/// RoPE layer and the remap attention on every attention layer, or whose KV
/// copy cannot be expressed, commits the chain only (its leaf columns are
/// rejected like any draft). The first armed rounds of a process also verify
/// a probe leaf (a copy of a chain node at the same position) and compare its
/// hidden row and target bit for bit; a mismatch disables leaves for the
/// process. Every decision is counted (`CBv2VerifyTree.Counters`) and, with
/// `CBV2_TREE_TRACE`, logged per round, because a fail-closed tree is
/// otherwise indistinguishable from a slow one.
///
/// Off (unset): no plan, kernel, column or readback changes.
public enum CBv2VerifyTree {
    private static let environment = ProcessInfo.processInfo.environment

    /// `CBV2_VERIFY_TREE=1` arms the 8-row tree; the same-width tree
    /// (`CBv2SameWidthTree`, on unless `CBV2_SAME_WIDTH_TREE=0`) shares this
    /// machinery and, when on, replaces it: no round adds rows.
    public static let enabled: Bool =
        environment["CBV2_VERIFY_TREE"] == "1" || CBv2SameWidthTree.enabled

    /// One leaf row as a fraction of a round (the marginal rule's cost ratio for
    /// leaves). The chain window keeps its own `fixedWindowHeadStepCostRatio`.
    public static let leafCostRatio: Double = {
        guard let raw = environment["CBV2_TREE_LEAF_H"], let value = Double(raw),
            value.isFinite, value > 0, value < 1
        else { return 0.025 }
        return value
    }()

    /// At most this many leaves per round (and `chain + leaves <= 7`).
    /// Default 2: on the full M2 Pro replay (7 prompts, 280 rounds) with
    /// 1.2 ms per row for rows 5-8 the net speed peaks at 2 leaves; 3-4 only
    /// tie or lose. `CBV2_TREE_MAX_LEAVES` (0...4) raises or lowers it.
    public static let maximumLeaves: Int = {
        guard let raw = environment["CBV2_TREE_MAX_LEAVES"], let value = Int(raw) else { return 2 }
        return min(max(value, 0), 4)
    }()

    /// Leaves only at block rows below this (depth 1...4).
    public static let maximumLeafRow = 4

    /// A request is eligible only while its committed tokens per speculative
    /// round so far stay at or under this. Pair-1-like requests (about 5 per
    /// round under the acceptance ceiling) never qualify; the hard pairs
    /// commit about 2.3-2.9. A running statistic, never a latch.
    public static let maximumMeanCommitted: Double = {
        guard let raw = environment["CBV2_TREE_MAX_MEAN"], let value = Double(raw),
            value.isFinite, value > 0
        else { return 3.5 }
        return value
    }()

    /// Speculative rounds a request must have finalized before it can arm.
    public static let minimumRounds = 3

    /// Armed rounds (per process) that carry the exactness probe leaf; leaves
    /// commit only after the probe has passed. `CBV2_TREE_PROBE_ROUNDS`, 0
    /// skips the probe (not recommended).
    public static let probeRounds: Int = {
        guard let raw = environment["CBV2_TREE_PROBE_ROUNDS"], let value = Int(raw) else { return 4 }
        return max(value, 0)
    }()

    /// Per-row sibling hit-rate priors (rows 0...3 = depths 1...4): rank 2
    /// given rank 1 wrong, and rank 3 given ranks 1 and 2 wrong. Defaults: the
    /// full M2 Pro measurement on the record (11/27, 7 prompts, 280 rounds:
    /// s2 .397/.386/.225/.400 over 73/57/40/25 events, s3 .318/.229/.129/.200
    /// over 44/35/31/15). `CBV2_TREE_S2_PRIOR` / `CBV2_TREE_S3_PRIOR`: 1 or 4
    /// comma-separated values.
    public static let siblingPriorsRank2: [Double] = priors(
        "CBV2_TREE_S2_PRIOR", [0.40, 0.39, 0.23, 0.40])
    public static let siblingPriorsRank3: [Double] = priors(
        "CBV2_TREE_S3_PRIOR", [0.32, 0.23, 0.13, 0.20])
    public static let siblingAlpha = 0.1

    private static func priors(_ key: String, _ fallback: [Double]) -> [Double] {
        guard let raw = environment[key] else { return fallback }
        let values = raw.split(separator: ",").compactMap {
            Double($0.trimmingCharacters(in: .whitespaces))
        }.filter { $0.isFinite && $0 >= 0 && $0 <= 1 }
        if values.count == 1 { return Array(repeating: values[0], count: maximumLeafRow) }
        return values.count == maximumLeafRow ? values : fallback
    }

    /// Per-row sibling hit rates (rank 2, rank 3), `maximumLeafRow` each.
    public struct SiblingRates: Sendable, Equatable {
        public var rank2: [Double]
        public var rank3: [Double]
        public init(rank2: [Double], rank3: [Double]) {
            self.rank2 = rank2
            self.rank3 = rank3
        }
        public static var priors: SiblingRates {
            SiblingRates(
                rank2: CBv2VerifyTree.siblingPriorsRank2, rank3: CBv2VerifyTree.siblingPriorsRank3)
        }
        func rate(row: Int, rank: Int) -> Double {
            let values = rank == 2 ? rank2 : rank3
            guard row >= 0, row < values.count else { return 0 }
            return values[row]
        }
    }

    /// `CBV2_TREE_SUMMARY=0` silences the per-request summary line.
    public static let printsSummary: Bool = enabled && environment["CBV2_TREE_SUMMARY"] != "0"

    // MARK: Layout

    public struct Leaf: Equatable, Sendable {
        /// Block row `j` (0-based): the leaf replaces draft `d(j+1)`.
        public var row: Int
        /// Drafter rank (2 or 3); 1 only for the exactness probe.
        public var rank: Int
        public init(row: Int, rank: Int) {
            self.row = row
            self.rank = rank
        }
    }

    public struct Layout: Equatable, Sendable {
        public var chain: Int
        public var leaves: [Leaf]
        /// Index into `leaves` of the probe leaf (rank 1, a copy of a chain
        /// node), when this round carries one.
        public var probe: Int?

        public var drafts: Int { chain + leaves.count }
        public var columns: Int { 1 + drafts }
        /// Verify column of leaf `index`.
        public func column(ofLeaf index: Int) -> Int { 1 + chain + index }

        /// Each column's position offset from the seed: chain columns `t`,
        /// leaf columns `row + 1`.
        public var rowPositions: [Int32] {
            (0 ... chain).map { Int32($0) } + leaves.map { Int32($0.row + 1) }
        }

        /// Draft-ordered (row, rank) of the leaves, as the drafter gathers them.
        public var gather: [(row: Int, rank: Int)] { leaves.map { ($0.row, $0.rank) } }
    }

    /// The marginal rule over leaves (mirrored by tools/tree_replay.py
    /// `v1_layout`; keep in sync). `probabilities`: the request's per-position
    /// acceptance estimates (the history the window used). The chain is the
    /// record window's. Candidate leaf (row j, rank r) for j <= chain,
    /// j < maximumLeafRow, is worth
    ///   rank 2: reach(j) (1 - p_j) s2_j
    ///   rank 3: reach(j) (1 - p_j) (1 - s2_j) s3_j     (needs rank 2 at j)
    /// with reach(j) = p_0 ... p_(j-1), the expected extra tokens it commits.
    /// Leaves are added best first (ties: lower row, then lower rank) while
    /// the value exceeds `h (1 + E) / (1 + n h)`, `E` the expected tokens so
    /// far and `n` the drafts so far, up to `maximumLeaves` and 7 drafts.
    public static func plan(
        chain: Int, probabilities: [Double], siblingRates: SiblingRates, maximumDrafts: Int
    ) -> Layout {
        var layout = Layout(chain: chain, leaves: [], probe: nil)
        guard chain > 0, maximumLeaves > 0, chain < maximumDrafts, !probabilities.isEmpty else {
            return layout
        }
        func p(_ j: Int) -> Double {
            guard j < probabilities.count, probabilities[j].isFinite else { return 0 }
            return min(max(probabilities[j], 0), 1)
        }
        var reach = [1.0]
        for j in 0 ..< max(chain, maximumLeafRow) { reach.append(reach[j] * p(j)) }
        var expected = reach[1 ... chain].reduce(0, +)
        var candidates: [(value: Double, row: Int, rank: Int)] = []
        for j in 0 ..< min(chain + 1, maximumLeafRow) {
            let miss = reach[j] * (1 - p(j))
            let s2 = siblingRates.rate(row: j, rank: 2)
            let s3 = siblingRates.rate(row: j, rank: 3)
            candidates.append((miss * s2, j, 2))
            candidates.append((miss * (1 - s2) * s3, j, 3))
        }
        candidates.sort { a, b in
            a.value != b.value ? a.value > b.value : (a.row != b.row ? a.row < b.row : a.rank < b.rank)
        }
        var n = chain
        let h = leafCostRatio
        for candidate in candidates {
            guard layout.leaves.count < maximumLeaves, n < maximumDrafts else { break }
            let threshold = h * (1 + expected) / (1 + Double(n) * h)
            guard candidate.value > threshold else { break }
            if candidate.rank == 3,
                !layout.leaves.contains(Leaf(row: candidate.row, rank: 2))
            {
                continue
            }
            layout.leaves.append(Leaf(row: candidate.row, rank: candidate.rank))
            expected += candidate.value
            n += 1
        }
        return layout
    }

    // MARK: Round

    /// What a tree round's finalize needs (on `CBv2MTPRoundInFlight.Verify`).
    public struct RoundTree: Sendable {
        /// The chain's drafts: the window's first `chain` drafts.
        public var chain: Int
        /// The armed layout (leaves follow the chain); nil when the round
        /// only learns sibling hit rates.
        public var layout: Layout?
        /// Every tree variant ran where it had to (meaningful with `layout`).
        public var valid: Bool
        /// Why no leaf is armed, or why the armed forward is invalid.
        public var why: String
        /// Acceptance-packet offset of the round's top-3 ids `[topRows * 3]`.
        public var topBase: Int?
        public var topRows: Int
        /// Acceptance-packet offset of the probe's bit-equality flag.
        public var probeBase: Int?
        /// The forward's tree-variant counts (`Build.summary`).
        public var forward: String
        /// Same-width round: the layout is read back at finalize from the
        /// packet at `sameWidthBase` (meta `[4]`, then rowpos `[k + 1]`).
        public var sameWidth = false
        public var sameWidthBase: Int? = nil
    }

    // MARK: Per-request state

    /// Per request, host only.
    struct RequestState {
        var siblingRates = CBv2VerifyTree.SiblingRates.priors
        /// Finalized verify rounds of the request that had a tree plan, and
        /// its generated tokens after the latest one: the summary's scope is
        /// the request's whole life, not a timed window.
        var rounds = 0
        var generated = 0
        var armedRounds = 0
        var leaves = 0
        var hits = 0
        var invalid = 0
        var notArmed: [String: Int] = [:]
    }

    // MARK: Build context

    /// What the model and the attention read while a tree round's target
    /// forward is built (engine thread, one round at a time).
    public final class Build: @unchecked Sendable {
        public let layout: Layout
        /// `[columns]` int32 position offsets from the seed (RoPE).
        public let rowPositions: MLXArray
        /// Set by the model's fused trunk: layers whose attention / RoPE must
        /// take the tree variants.
        /// Same-width round: `rowDepth` (= `rowPositions`, device) drives the
        /// attention's depth templates; the host layout is the plain chain.
        public let sameWidth: Bool
        public let rowDepth: MLXArray?
        public var expectedAttention = 0
        public var expectedRope = 0
        public var attentionCalls = 0
        public var ropeCalls = 0
        public var failure: String?

        /// The engine thread that builds this round's forward.
        let thread = pthread_self()

        init(layout: Layout) {
            self.layout = layout
            self.rowPositions = MLXArray(layout.rowPositions)
            self.sameWidth = false
            self.rowDepth = nil
        }

        /// A same-width round of `drafts` draft columns whose depths come from
        /// `cbv2_tree_select` on device.
        init(sameWidthDrafts drafts: Int, rowPositions: MLXArray) {
            self.layout = Layout(chain: drafts, leaves: [], probe: nil)
            self.rowPositions = rowPositions
            self.sameWidth = true
            self.rowDepth = rowPositions
        }

        public var columns: Int { layout.columns }

        /// The serial column whose attend range leaf column `column` takes,
        /// and the leaf's index; nil for a chain column.
        public func leaf(atColumn column: Int) -> (index: Int, servesColumn: Int)? {
            let index = column - 1 - layout.chain
            guard index >= 0, index < layout.leaves.count else { return nil }
            return (index, layout.leaves[index].row + 1)
        }

        public func fail(_ reason: String) {
            if failure == nil { failure = reason }
        }

        /// Every tree variant ran where it had to.
        public var valid: Bool {
            failure == nil && expectedAttention > 0 && attentionCalls == expectedAttention
                && ropeCalls == expectedRope
        }

        public var summary: String {
            "attn \(attentionCalls)/\(expectedAttention) rope \(ropeCalls)/\(expectedRope)"
                + (failure.map { " fail \($0)" } ?? "")
        }
    }

    /// The tree round whose target forward is being built, or nil.
    nonisolated(unsafe) public static var current: Build?

    /// The active build when it matches a `columns`-wide forward of one row.
    /// Only on the thread that set it: another engine in the process building a
    /// forward of the same width never sees this round's tree.
    public static func active(batch: Int, columns: Int) -> Build? {
        guard let build = current, batch == 1, build.columns == columns,
            pthread_equal(build.thread, pthread_self()) != 0
        else { return nil }
        return build
    }

    /// Load time (model warm-up): compiles the leaf K/V copy kernel and checks
    /// one in-place copy on a scratch page. False turns leaves off.
    public static func warmCopyKernel(dtype: DType, kvHeads: Int, headDim: Int, pageSize: Int)
        -> Bool
    {
        PagedSegmentTransfers.warmTreeCopy(
            dtype: dtype, kvHeads: kvHeads, headDim: headDim, pageSize: pageSize)
    }

    // MARK: Process-wide state

    /// Leaves may commit only after the probe passed; a failed probe disables
    /// leaves for the process (a correctness fuse, not a policy latch).
    nonisolated(unsafe) static var probePassed = false
    nonisolated(unsafe) static var probeFailed = false
    nonisolated(unsafe) static var probesRun = 0
    /// A load-time kernel self-test failed (`selfTest`).
    nonisolated(unsafe) public static var kernelSelfTestFailed = false

    static var usable: Bool { enabled && !probeFailed && !kernelSelfTestFailed }

    public struct Counters: Sendable {
        public var eligibleRounds = 0
        public var armedRounds = 0
        public var leaves = 0
        public var hits = 0
        public var invalidRounds = 0
        public var probeRounds = 0
        public var probeFailures = 0
        public var copies = 0
        public var notArmed: [String: Int] = [:]
    }
    nonisolated(unsafe) public static var counters = Counters()
}

/// A request-stateful drafter that can hand the draft tree its siblings.
public protocol CBv2MTPTreeDrafter: AnyObject {
    /// `[leaves]` int32, lazy: the round's rank-`rank` proposal at each block
    /// `row`, gathered on device from the block's top-3 (no host sync). Nil
    /// when this round's block has no top-3 (the round then verifies a chain).
    func treeLeafTokens(_ leaves: [(row: Int, rank: Int)], requestState: any CBv2MTPRequestState)
        -> MLXArray?
    /// `[rows * 3]` int32, lazy: the round's top-3 ids per block row, or nil.
    func treeTopIDs(requestState: any CBv2MTPRequestState) -> MLXArray?
    /// Before `finalizeRound` of a round that commits a leaf: the round's
    /// committed inputs reach `stagedRows` drafted positions (a leaf after a
    /// fully accepted chain commits one input past the chain's staged drafts).
    func treePrepareLeafCommit(stagedRows: Int, requestState: any CBv2MTPRequestState)
    /// After `finalizeRound` of a round that committed a leaf.
    func treeLeafCommitted(requestState: any CBv2MTPRequestState)
    /// Same-width tree: this round's top-4 ids `[7, 4]` int32 and raw head
    /// products `[7, 4]` float32 (lazy), or nil.
    func treeTop4(requestState: any CBv2MTPRequestState) -> (ids: MLXArray, raw: MLXArray)?
    /// Same-width tree: the verify columns' depths for the next
    /// `speculateNextRound` of this request (consumed there).
    func setSpeculationRowPositions(_ rowPositions: MLXArray?, requestState: any CBv2MTPRequestState)
    /// Same-width tree leaf commit (after `finalizeRound`): keep the
    /// speculation for verify rows `rows`; true when kept.
    @discardableResult
    func treeConfirmSpeculation(
        carryHidden: MLXArray, rows: [Int], requestState: any CBv2MTPRequestState) -> Bool
}
