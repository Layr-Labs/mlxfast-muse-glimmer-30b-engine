// CBv2VerifyTree.swift
//
// Draft-tree verify window for low-acceptance rounds: a chain plus branches,
// each a drafter rank-1...3 token followed by the block's own proposals
// (static: one depth-1 second-choice leaf; dynamic: chosen per round). The
// leaf column sits after the chain in storage and attends the prefix, the
// seed and itself at logical position 1 (the chain past it is skipped). When
// the chain stops at depth 0 and the target's own token there is the leaf,
// the leaf and the target token after it are committed. Nothing here runs
// on a round that does not take the tree.

import Foundation
import MLX
import MLXFast

public struct CBv2VerifyTreeLayout: Hashable, Sendable {
    /// Chain drafts (verify columns 1 ... chain).
    public let chain: Int
    /// Depth (1 ... chain) where each branch starts, in column order.
    public let branchDepths: [Int]
    /// Columns of each branch (at least 1).
    public let branchLengths: [Int]
    /// Drafter rank (1 ... 3) of each branch's first token; its later
    /// columns are the block's own proposals.
    public let branchRanks: [Int]

    public init(chain: Int, branchDepths: [Int], branchLengths: [Int], branchRanks: [Int]? = nil) {
        let ranks = branchRanks ?? Array(repeating: 1, count: branchDepths.count)
        precondition(
            chain >= 1 && branchDepths.count == branchLengths.count
                && ranks.count == branchDepths.count
                && branchDepths.allSatisfy { (1 ... chain).contains($0) }
                && branchLengths.allSatisfy { $0 >= 1 } && ranks.allSatisfy { (1 ... 3).contains($0) })
        self.chain = chain
        self.branchDepths = branchDepths
        self.branchLengths = branchLengths
        self.branchRanks = ranks
    }

    /// `chain` drafts plus one depth-1 second-choice leaf.
    public static func leafAtDepthOne(chain: Int) -> CBv2VerifyTreeLayout {
        CBv2VerifyTreeLayout(chain: chain, branchDepths: [1], branchLengths: [1])
    }

    /// Draft columns in the window (`k`).
    public var drafts: Int { chain + branchLengths.reduce(0, +) }
    /// Verify columns, seed included.
    public var columns: Int { 1 + drafts }

    /// First verify column of branch `b`.
    public func branchStart(_ b: Int) -> Int {
        1 + chain + branchLengths[0 ..< b].reduce(0, +)
    }

    /// Branch and offset in it of verify column `t` (nil for the chain).
    public func branchPosition(column t: Int) -> (branch: Int, offset: Int)? {
        guard t > chain else { return nil }
        var start = 1 + chain
        for (b, length) in branchLengths.enumerated() {
            if t < start + length { return (b, t - start) }
            start += length
        }
        return nil
    }

    /// Logical position offset of verify column `t` from the seed column.
    public func depth(column t: Int) -> Int {
        guard let p = branchPosition(column: t) else { return t }
        return branchDepths[p.branch] + p.offset
    }

    /// Skipped columns `[from, to)` relative to the seed column: a branch
    /// column skips the chain from its start depth and every earlier branch.
    public func hole(column t: Int) -> (from: Int, to: Int)? {
        guard let p = branchPosition(column: t) else { return nil }
        return (branchDepths[p.branch], branchStart(p.branch))
    }

    /// The branches that start at `depth`, in column order.
    public func branches(depth: Int) -> [Int] {
        branchDepths.indices.filter { branchDepths[$0] == depth }
    }

    /// Block position and drafter rank of the token in verify column `t` of
    /// a branch: offset 0 is the branch's rank at its start depth, offset j
    /// the block's own proposal (rank 0) j positions deeper.
    public func proposalIndex(column t: Int) -> (index: Int, rank: Int)? {
        guard let p = branchPosition(column: t) else { return nil }
        return (branchDepths[p.branch] - 1 + p.offset, p.offset == 0 ? branchRanks[p.branch] : 0)
    }

    /// Best-first draft tree over the drafter's own top-4 probabilities
    /// `probs[position][rank]`: `drafts` nodes picked by the product of their
    /// path's probabilities, each path the chain with at most one deviation
    /// (rank 1 ... 3) followed by the block's own proposals, at most
    /// `depthLimit` deep. Ties go to the lexicographically smaller rank path.
    public static func bestFirst(probs: [[Float]], drafts: Int, depthLimit: Int)
        -> CBv2VerifyTreeLayout?
    {
        let depthLimit = min(depthLimit, probs.count)
        guard drafts >= 1, depthLimit >= 1, probs.allSatisfy({ $0.count >= 4 }) else { return nil }
        // A path: chain to `deviation` (0-based position, nil = chain), its
        // rank there, and its length.
        struct Node: Comparable {
            var score: Float
            var deviation: Int?
            var rank: Int
            var length: Int
            var ranks: [Int] {
                (0 ..< length).map { $0 == deviation ? rank : 0 }
            }
            static func < (a: Node, b: Node) -> Bool {
                if a.score != b.score { return a.score > b.score }
                return a.ranks.lexicographicallyPrecedes(b.ranks)
            }
        }
        var frontier = (0 ..< 4).map {
            Node(score: probs[0][$0], deviation: $0 == 0 ? nil : 0, rank: $0, length: 1)
        }
        var chain = 0
        var branchLength: [Int: Int] = [:]  // key: deviation * 4 + rank
        var picked = 0
        while picked < drafts, let best = frontier.min() {
            frontier.remove(at: frontier.firstIndex(of: best)!)
            picked += 1
            if let d = best.deviation {
                branchLength[d * 4 + best.rank, default: 0] += 1
            } else {
                chain += 1
            }
            guard best.length < depthLimit else { continue }
            let at = best.length
            let ranks = best.deviation == nil ? 0 ..< 4 : 0 ..< 1
            for r in ranks {
                frontier.append(
                    Node(
                        score: best.score * probs[at][r],
                        deviation: best.deviation ?? (r == 0 ? nil : at),
                        rank: best.deviation == nil ? r : best.rank, length: at + 1))
            }
        }
        guard picked == drafts, chain >= 1 else { return nil }
        let keys = branchLength.keys.sorted()
        let depths = keys.map { $0 / 4 + 1 }
        guard depths.allSatisfy({ $0 <= chain }) else { return nil }
        return CBv2VerifyTreeLayout(
            chain: chain, branchDepths: depths, branchLengths: keys.map { branchLength[$0]! },
            branchRanks: keys.map { $0 % 4 })
    }
}

/// The layout of the verify forward being built. Set by the engine around
/// one round's target graph construction; layers read it while building.
public enum CBv2VerifyTree {
    private static let environment = ProcessInfo.processInfo.environment
    /// `CBV2_VERIFY_TREE=0` turns tree windows off.
    public static let enabled = environment["CBV2_VERIFY_TREE"] != "0"
    /// Low-acceptance rounds choose their tree per round from the drafter's
    /// top-4 probabilities (`CBV2_DYNAMIC_TREE=0`: the static depth-1 leaf).
    public static let dynamic = enabled && environment["CBV2_DYNAMIC_TREE"] != "0"
    /// Verify width of a dynamic tree: `max(window, this)` drafts, at most
    /// `maxDrafts` (`CBV2_DYNAMIC_TREE_MIN_N`, default 6).
    public static let dynamicMinimumDrafts = min(
        max(Int(environment["CBV2_DYNAMIC_TREE_MIN_N"] ?? "") ?? 6, 1), maxDrafts)
    /// Dynamic trees are built on the GPU from the drafter's top-4
    /// (`CBv2VerifyTreeBuilder`): no host read between the drafter and the
    /// verify. The width stays host-known, so every shape, seqinfo row and
    /// pipeline is a chain's of that width; only the topology is device data.
    /// `CBV2_DYNAMIC_TREE_ZEROSYNC=0`: the host-synced builder.
    public static let zeroSync = dynamic && environment["CBV2_DYNAMIC_TREE_ZEROSYNC"] != "0"
    /// The device-hole attention variant of every binding class compiled and
    /// held its threadgroup (load-time warm-up).
    nonisolated(unsafe) public private(set) static var zeroSyncReady = false
    public static func setZeroSyncReady(_ ready: Bool) { zeroSyncReady = zeroSync && ready }

    /// A GPU-built window's topology while it builds: per-row hole bounds
    /// relative to the seed `[>= columns, 2]` and RoPE depths `[columns]`,
    /// both int32 device arrays. `current` is then a placeholder chain of the
    /// same width (so seqinfo and every host-built descriptor are a chain's).
    nonisolated(unsafe) public private(set) static var deviceTopology:
        (holes: MLXArray, rowOffsets: MLXArray)?
    /// A GPU-built window over a windowed layer that clips a draft column's
    /// range (fbd80ac's per-branch-column check, topology-free): its branch
    /// rows are not exact, so finalize commits the chain only.
    nonisolated(unsafe) public static var clipped = false

    /// Longest attended context (tokens) a tree window can take: the one-pass
    /// paged kernel's reach, set by its load-time warm-up once the branch
    /// variants of every binding class compiled and held their threadgroup.
    /// nil: trees are off (no warm-up, a failed probe, or an ineligible
    /// shape). Longer contexts attend through the two-pass kernels, which
    /// have no branch variant: such a window could not honor its tree.
    nonisolated(unsafe) public private(set) static var contextLimit: Int?

    /// Widest window (drafts) whose kernels the load-time warm-up builds.
    public static let maxDrafts = 7

    public static func setContextLimit(_ limit: Int) { contextLimit = limit }
    public static func disable() { contextLimit = nil }
    /// Trees are enabled and their kernels are built.
    public static var ready: Bool { enabled && contextLimit != nil }

    /// Whether a window of `drafts` over `contextTokens` committed tokens can
    /// attend through the tree kernels (the last column's range is
    /// `contextTokens + drafts + 1` long).
    public static func fits(contextTokens: Int, drafts: Int) -> Bool {
        guard enabled, drafts <= maxDrafts, let contextLimit else { return false }
        return contextTokens + drafts + 1 <= contextLimit
    }

    nonisolated(unsafe) public private(set) static var current: CBv2VerifyTreeLayout?
    /// A layer built the window without the tree's positions or masks.
    nonisolated(unsafe) public static var failed = false
    /// Attention calls that applied the branch masks.
    nonisolated(unsafe) public static var maskedAttentionCalls = 0
    /// RoPE calls that applied the branch positions.
    nonisolated(unsafe) public static var rowPositionCalls = 0
    /// The model's count of each per window (its attention layers and its
    /// RoPE layers), set at load; `end()` then requires exactly these.
    nonisolated(unsafe) public static var expectedCalls: (maskedAttention: Int, rowPositions: Int)?
    /// Smallest window (drafts) that takes a tree: below 4 drafts the lost
    /// last chain position is worth more than the depth-1 leaf (replay:
    /// k = 2 -0.28, k = 3 -0.07, k = 4 +0.02 tokens per round).
    public static let minimumWindow = 4

    public static func begin(
        _ layout: CBv2VerifyTreeLayout?,
        device: (holes: MLXArray, rowOffsets: MLXArray)? = nil
    ) {
        current = layout
        deviceTopology = layout == nil ? nil : device
        clipped = false
        failed = false
        maskedAttentionCalls = 0
        rowPositionCalls = 0
    }

    /// Ends construction; true when every layer honored the layout: each
    /// attention layer applied the branch masks and each RoPE layer the
    /// branch positions, exactly once (counts > 0 when the model set none).
    public static func end() -> Bool {
        let counted: Bool
        if let expectedCalls {
            counted = maskedAttentionCalls == expectedCalls.maskedAttention
                && rowPositionCalls == expectedCalls.rowPositions
        } else {
            counted = maskedAttentionCalls > 0 && rowPositionCalls > 0
        }
        let ok = current != nil && !failed && counted
        current = nil
        deviceTopology = nil
        return ok
    }

    /// The active layout when a call over `columns` verify columns builds.
    public static func active(columns: Int) -> CBv2VerifyTreeLayout? {
        guard let current, current.columns == columns else { return nil }
        return current
    }

    /// The layouts a declared depth can produce (for load-time kernel warm-up),
    /// plus, for dynamic trees, a multi-branch caterpillar at each width they
    /// verify (two heads at depth 1, one at depth 2, branches past one column).
    public static func reachableLayouts(maxDrafts: Int) -> [CBv2VerifyTreeLayout] {
        guard enabled, maxDrafts >= minimumWindow else { return [] }
        var layouts = (minimumWindow - 1 ..< maxDrafts).map {
            CBv2VerifyTreeLayout.leafAtDepthOne(chain: $0)
        }
        if dynamic {
            for width in max(dynamicMinimumDrafts, 5) ... max(maxDrafts, 5) where width <= maxDrafts {
                layouts.append(
                    CBv2VerifyTreeLayout(
                        chain: 2, branchDepths: [1, 1, 2], branchLengths: [width - 4, 1, 1],
                        branchRanks: [1, 2, 1]))
            }
        }
        return layouts
    }

    /// Builds `body` as if `layout` were the window being verified.
    public static func building<T>(_ layout: CBv2VerifyTreeLayout, _ body: () -> T) -> T {
        let saved = (current, failed, maskedAttentionCalls, rowPositionCalls)
        let savedDevice = (deviceTopology, clipped)
        current = layout
        deviceTopology = nil
        defer {
            (current, failed, maskedAttentionCalls, rowPositionCalls) = saved
            (deviceTopology, clipped) = savedDevice
        }
        return body()
    }

    /// Per-column logical offsets `[columns]` int32.
    public static func rowOffsets(_ layout: CBv2VerifyTreeLayout) -> [Int32] {
        (0 ..< layout.columns).map { Int32(layout.depth(column: $0)) }
    }
}

/// GPU builder of the best-first caterpillar (`cbv2_build_caterpillar_topn`).
///
/// `CBv2VerifyTreeLayout.bestFirst` picks the top `W` of the caterpillar's
/// candidate nodes in the order (score descending, rank path
/// lexicographically ascending): a child's score is its parent's times a
/// probability <= 1 and a parent's path is a prefix of its child's, so every
/// node follows all its ancestors and the heap's first `W` pops are exactly
/// the first `W` nodes of that total order. There are `DL + 3 DL (DL+1)/2`
/// candidates (91 at depth 7): the chain of each length, and each deviation
/// (position d, rank 1...3) continued by top-1 proposals. One threadgroup:
/// thread i forms candidate i's score with bestFirst's own float32 product
/// order, counts the candidates ahead of it, and keeps itself when fewer
/// than `W` are; thread 0 then lays the kept nodes out in bestFirst's column
/// order (chain, then branches by `deviation * 4 + rank`).
///
/// Outputs (int32): `tokens [W]` verify columns 1...W (chain columns take
/// the drafter's own staged chain tokens, branch columns the ranked ids,
/// as the host-built window did), `rowpos [W+1]` RoPE depths, `holes
/// [max(W+1, 4), 2]` seed-relative skipped spans ((0, 0) for the seed and
/// chain), `meta [2W + 1 + R]`: per draft column (branch key or -1, depth),
/// a valid flag, then the block's top-1 ids. Not valid (a non-finite score,
/// or a selection that is not a caterpillar): a chain of `W`, as the host
/// path's nil.
public enum CBv2VerifyTreeBuilder {
    // BEGIN cbv2_build_caterpillar_topn
    static let source = """
        const int tid = int(thread_position_in_threadgroup.x);
        constexpr int NC = DL_ + 3 * DL_ * (DL_ + 1) / 2;
        constexpr int NK = DL_ * 4;
        threadgroup float sc[NC];
        threadgroup int dv[NC];
        threadgroup int rk[NC];
        threadgroup int ln[NC];
        threadgroup int ahead[NC];
        threadgroup int bad[1];
        if (tid == 0) bad[0] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < NC) {
            int i = tid;
            int d = -1, r = 0, len = 0;
            if (i < DL_) {
                len = i + 1;
            } else {
                i -= DL_;
                for (int dd = 0; dd < DL_; dd++) {
                    const int n = DL_ - dd;
                    if (i < 3 * n) {
                        d = dd;
                        r = 1 + i / n;
                        len = dd + 1 + i % n;
                        break;
                    }
                    i -= 3 * n;
                }
            }
            float s = 1.0f;
            for (int j = 0; j < len; j++) {
                s = s * probs[j * 4 + (j == d ? r : 0)];
            }
            // Bit test: fast math may fold isfinite.
            if ((as_type<uint>(s) & 0x7f800000u) == 0x7f800000u) bad[0] = 1;
            sc[tid] = s;
            dv[tid] = d;
            rk[tid] = r;
            ln[tid] = len;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < NC) {
            const float si = sc[tid];
            const int di = dv[tid], ri = rk[tid], li = ln[tid];
            int count = 0;
            for (int j = 0; j < NC; j++) {
                if (j == tid) continue;
                const float sj = sc[j];
                bool before;
                if (sj != si) {
                    before = sj > si;
                } else {
                    const int dj = dv[j], rj = rk[j], lj = ln[j];
                    const int m = min(li, lj);
                    int p = 0;
                    while (p < m && (p == dj ? rj : 0) == (p == di ? ri : 0)) p++;
                    before = p < m ? (p == dj ? rj : 0) < (p == di ? ri : 0) : lj < li;
                }
                count += before ? 1 : 0;
            }
            ahead[tid] = count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            int blen[NK];
            for (int k = 0; k < NK; k++) blen[k] = 0;
            int chain = 0, picked = 0;
            for (int i = 0; i < NC; i++) {
                if (ahead[i] >= W_) continue;
                picked++;
                if (dv[i] < 0) chain++; else blen[dv[i] * 4 + rk[i]]++;
            }
            bool ok = bad[0] == 0 && picked == W_ && chain >= 1;
            for (int i = 0; i < NC; i++) {
                if (ahead[i] >= W_) continue;
                // Prefix-closed: lengths 1...chain, and d+1...d+n per branch.
                if (dv[i] < 0 ? ln[i] > chain : ln[i] - dv[i] > blen[dv[i] * 4 + rk[i]]) ok = false;
            }
            for (int k = 0; k < NK; k++) {
                if (blen[k] > 0 && k / 4 + 1 > chain) ok = false;
            }
            if (!ok) {
                chain = W_;
                for (int k = 0; k < NK; k++) blen[k] = 0;
            }
            rowpos[0] = 0;
            for (int c = 0; c < HR_; c++) {
                holes[2 * c] = 0;
                holes[2 * c + 1] = 0;
            }
            for (int c = 1; c <= chain; c++) {
                tokens[c - 1] = chain_tokens[c - 1];
                rowpos[c] = c;
                meta[2 * (c - 1)] = -1;
                meta[2 * (c - 1) + 1] = c;
            }
            int col = chain + 1;
            for (int k = 0; k < NK; k++) {
                const int n = blen[k];
                const int d = k / 4, r = k % 4, start = col;
                for (int j = 0; j < n; j++, col++) {
                    tokens[col - 1] = j == 0 ? ids[d * 4 + r] : ids[(d + j) * 4];
                    rowpos[col] = d + 1 + j;
                    holes[2 * col] = d + 1;
                    holes[2 * col + 1] = start;
                    meta[2 * (col - 1)] = k;
                    meta[2 * (col - 1) + 1] = d + 1 + j;
                }
            }
            meta[2 * W_] = ok ? 1 : 0;
            for (int i = 0; i < R_; i++) meta[2 * W_ + 1 + i] = ids[i * 4];
        }
        """
    // END cbv2_build_caterpillar_topn

    static let kernel = MLXFast.metalKernel(
        name: "cbv2_build_caterpillar_topn",
        inputNames: ["ids", "probs", "chain_tokens"],
        outputNames: ["tokens", "rowpos", "holes", "meta"],
        source: source)

    public struct Built {
        public let tokens: MLXArray
        public let rowOffsets: MLXArray
        public let holes: MLXArray
        public let meta: MLXArray
        public let width: Int
        public let rows: Int
        /// int32 words `meta` appends to the acceptance packet.
        public var metaCount: Int { 2 * width + 1 + rows }
    }

    /// Lazy: the window of `width` drafts over `ids`/`probs` `[R, 4]` and the
    /// drafter's staged chain tokens `[width]`. Nil for shapes it does not take.
    public static func build(ids: MLXArray, probs: MLXArray, chainTokens: MLXArray, width: Int)
        -> Built?
    {
        guard ids.ndim == 2, ids.dim(1) == 4, probs.shape == ids.shape,
            (1 ... CBv2VerifyTree.maxDrafts).contains(width), ids.dim(0) >= width,
            chainTokens.size == width
        else { return nil }
        let rows = ids.dim(0)
        let holeRows = max(width + 1, 4)
        let out = kernel(
            [ids.asType(.int32), probs.asType(.float32), chainTokens.reshaped([-1]).asType(.int32)],
            template: [("W_", width), ("R_", rows), ("DL_", min(rows, width)), ("HR_", holeRows)],
            grid: (128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[width], [width + 1], [holeRows, 2], [2 * width + 1 + rows]],
            outputDTypes: [.int32, .int32, .int32, .int32])
        return Built(
            tokens: out[0], rowOffsets: out[1], holes: out[2], meta: out[3], width: width,
            rows: rows)
    }

    /// The layout a `meta` packet tail describes, its valid flag and the
    /// block's top-1 ids. A malformed tail reads as a chain of `width`.
    public static func decode(meta: [Int32], width: Int)
        -> (layout: CBv2VerifyTreeLayout, valid: Bool, topOne: [Int])
    {
        let chainOnly = CBv2VerifyTreeLayout(chain: width, branchDepths: [], branchLengths: [])
        guard meta.count >= 2 * width + 1 else { return (chainOnly, false, []) }
        let topOne = meta[(2 * width + 1)...].map { Int($0) }
        guard meta[2 * width] == 1 else { return (chainOnly, false, topOne) }
        var chain = 0
        while chain < width, meta[2 * chain] == -1 { chain += 1 }
        var keys: [Int] = []
        var lengths: [Int] = []
        for c in chain ..< width {
            let key = Int(meta[2 * c])
            if key == keys.last { lengths[lengths.count - 1] += 1 } else {
                keys.append(key)
                lengths.append(1)
            }
        }
        guard chain >= 1, keys.allSatisfy({ $0 >= 0 && $0 % 4 >= 1 && $0 / 4 + 1 <= chain }),
            zip(keys, keys.dropFirst()).allSatisfy({ $0 < $1 })
        else { return (chainOnly, false, topOne) }
        return (
            CBv2VerifyTreeLayout(
                chain: chain, branchDepths: keys.map { $0 / 4 + 1 }, branchLengths: lengths,
                branchRanks: keys.map { $0 % 4 }), true, topOne
        )
    }

    /// Compiles the builder at every width a dynamic tree verifies (load time).
    public static func warm(rows: Int = CBv2VerifyTree.maxDrafts) -> [MLXArray] {
        guard CBv2VerifyTree.zeroSync else { return [] }
        let ids = MLXArray.zeros([rows, 4], type: Int32.self)
        let probs = MLXArray.zeros([rows, 4], type: Float.self)
        var warm: [MLXArray] = []
        for width in 1 ... min(rows, CBv2VerifyTree.maxDrafts) {
            if let b = build(
                ids: ids, probs: probs, chainTokens: MLXArray.zeros([width], type: Int32.self),
                width: width)
            {
                warm += [b.tokens, b.rowOffsets, b.holes, b.meta]
            }
        }
        return warm
    }
}
