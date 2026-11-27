// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Same-width dynamic leaf tree (`CBV2_SAME_WIDTH_TREE`, on unless `0`).
///
/// A low-acceptance round keeps the record window's rows: `1 + k` verify
/// columns, `k` the acceptance window's drafts. The device re-spends the `k`
/// draft columns best first from the drafter's own confidence
/// (`cbv2_tree_select`, one threadgroup, no host sync). The result is a chain of
/// `c' <= k` rank-1 drafts followed by `k - c'` single-token leaves: rank-2
/// siblings at block rows `< 4`, each a child of the chain prefix before it.
/// Children are ranks 1-2; a rank-2 node has none (leaf-only); the path score
/// is the product of per-row probabilities `p = softmax(top-4 raw · scale)`.
///
/// Column order is chain first, so every node's ancestors are the contiguous
/// columns `1 ... depth - 1`. A column at depth `d` is therefore computed as the
/// serial step at position `seed + d`:
/// - RoPE at `seed + d` (`VerifyFusedQK.applyTree` with the device `rowpos`);
/// - attention with chain column `d`'s seqinfo and partition records and its
///   own K/V (written at its own column's slot) as the last key
///   (`CBV2_TREE_DEPTH` one-pass variant, `rowdepth = rowpos`);
/// - projections, norms, lm_head and the argmax are row-exact.
/// Chain columns have `rowpos[t] = t`, which is the record's arithmetic.
///
/// Acceptance, the leaf K/V move and the carry use the round's layout read
/// back with the acceptance packet (no extra sync); see `CBv2VerifyTree`.
public enum CBv2SameWidthTree {
    private static let environment = ProcessInfo.processInfo.environment

    /// On by default in this build; `CBV2_SAME_WIDTH_TREE=0` is the record path.
    public static let enabled: Bool = environment["CBV2_SAME_WIDTH_TREE"] != "0"

    /// Leaves per round (siblings exist only at rows 0...3, so at most 4).
    public static let maximumLeaves: Int = {
        guard let raw = environment["CBV2_SW_MAX_LEAVES"], let value = Int(raw) else { return 4 }
        return min(max(value, 0), 4)
    }()

    /// The raw head product times this is the softmax argument over a row's
    /// top 4 (`weight_scale_2 · output multiplier / τ`, τ = 0.2: E1's 0.000257 / 0.2).
    public static let scale: Float = {
        guard let raw = environment["CBV2_SW_SCALE"], let value = Float(raw), value.isFinite,
            value > 0
        else { return 0.000257 / 0.2 }
        return value
    }()

    /// Fewest drafts a same-width round arms with. The depth attention variant
    /// costs +14-16% at 2 verify rows (1 draft) on the M2 Pro TreeBench and
    /// +1.8-2.2% at 3-8 rows, and a 1-draft window has no room for a leaf:
    /// such rounds always take the record kernels.
    public static let minimumDrafts = 2

    /// Rows of the drafter's block and entries kept per row.
    public static let rows = 7
    public static let width = 4

    // MARK: tree_select

    /// One thread runs E1's `sameWidthPlan` (greedy best-first over at most 7
    /// steps); the other 31 lanes idle. Inputs `top_ids` int32 `[7, 4]` and
    /// `top_raw` float32 `[7, 4]`, each row best first, ties to the lower id.
    /// Outputs `tokens` int32 `[K]` (columns 1...K), `rowpos` int32 `[K + 1]`
    /// (each column's depth; column 0 is the seed), `meta` int32 `[4]` = (chain,
    /// leaves, probe, 0). No valid tree (or leaves 0) gives the plain chain.
    /// `PROBE_`: columns 1...K-1 the chain, column K a rank-1 copy of row 0
    /// (depth 1), which must reproduce column 1 bit for bit.
    static let selectSource = """
        if (thread_position_in_grid.x != 0) return;
        constexpr int K = K_;
        constexpr int R = ROWS_;
        constexpr int W = W_;
        constexpr int ML = MAXLEAF_;
        constexpr int MR = MAXROW_;
        const float sc = params[0];
        float p1[R];
        float p2[R];
        for (int r = 0; r < R; r++) {
          float m = -INFINITY;
          for (int i = 0; i < W; i++) m = max(m, top_raw[r * W + i] * sc);
          float z = 0.0f;
          float e0 = 0.0f;
          float e1 = 0.0f;
          for (int i = 0; i < W; i++) {
            const float e = metal::precise::exp(top_raw[r * W + i] * sc - m);
            z += e;
            if (i == 0) e0 = e;
            if (i == 1) e1 = e;
          }
          p1[r] = z > 0.0f ? e0 / z : 0.0f;
          p2[r] = z > 0.0f ? e1 / z : 0.0f;
        }
        int chain = 0;
        int nleaf = 0;
        int leaf_row[4] = {0, 0, 0, 0};
        float sib_score[4];
        int sib_row[4];
        int sib_id[4];
        bool sib_used[4] = {false, false, false, false};
        int nsib = 0;
        float reach = 1.0f;
        bool ok = !PROBE_;
        if (ok) {
          sib_score[0] = p2[0];
          sib_row[0] = 0;
          sib_id[0] = top_ids[1];
          nsib = 1;
          while (chain + nleaf < K) {
            const float chain_score = chain < R ? reach * p1[chain] : -1.0f;
            const int chain_id = chain < R ? top_ids[chain * W] : 2147483647;
            int best = -1;
            if (nleaf < ML) {
              for (int i = 0; i < nsib; i++) {
                if (sib_used[i]) continue;
                if (best < 0 || sib_score[i] > sib_score[best]
                    || (sib_score[i] == sib_score[best] && sib_id[i] < sib_id[best])) best = i;
              }
            }
            bool take_chain;
            if (chain_score < 0.0f) take_chain = false;
            else if (best < 0) take_chain = true;
            else take_chain = chain_score > sib_score[best]
                || (chain_score == sib_score[best] && chain_id <= sib_id[best]);
            if (take_chain) {
              reach *= p1[chain];
              chain += 1;
              if (chain < R && chain < MR && nsib < 4) {
                sib_score[nsib] = reach * p2[chain];
                sib_row[nsib] = chain;
                sib_id[nsib] = top_ids[chain * W + 1];
                nsib += 1;
              }
            } else if (best >= 0) {
              sib_used[best] = true;
              leaf_row[nleaf] = sib_row[best];
              nleaf += 1;
            } else {
              break;
            }
          }
          ok = nleaf > 0 && chain >= 1 && chain + nleaf == K;
        }
        rowpos[0] = 0;
        if (PROBE_ && K >= 2) {
          for (int t = 1; t < K; t++) {
            tokens[t - 1] = top_ids[(t - 1) * W];
            rowpos[t] = t;
          }
          tokens[K - 1] = top_ids[0];
          rowpos[K] = 1;
          meta[0] = K - 1;
          meta[1] = 1;
          meta[2] = 1;
        } else if (ok) {
          for (int t = 1; t <= chain; t++) {
            tokens[t - 1] = top_ids[(t - 1) * W];
            rowpos[t] = t;
          }
          for (int j = 0; j < nleaf; j++) {
            tokens[chain + j] = top_ids[leaf_row[j] * W + 1];
            rowpos[chain + 1 + j] = leaf_row[j] + 1;
          }
          meta[0] = chain;
          meta[1] = nleaf;
          meta[2] = 0;
        } else {
          for (int t = 1; t <= K; t++) {
            tokens[t - 1] = top_ids[(t - 1) * W];
            rowpos[t] = t;
          }
          meta[0] = K;
          meta[1] = 0;
          meta[2] = 0;
        }
        meta[3] = 0;
        """

    static let selectKernel = MLXFast.metalKernel(
        name: "cbv2_tree_select",
        inputNames: ["top_ids", "top_raw", "params"],
        outputNames: ["tokens", "rowpos", "meta"],
        source: selectSource)

    /// The round's layout on device (no host sync): `tokens` `[K]` for verify
    /// columns 1...K, `rowPositions` `[K + 1]` (also the attention's `rowdepth`),
    /// `meta` `[4]` = (chain, leaves, probe, 0).
    public static func select(topIDs: MLXArray, topRaw: MLXArray, drafts: Int, probe: Bool)
        -> (tokens: MLXArray, rowPositions: MLXArray, meta: MLXArray)
    {
        precondition((1 ... 7).contains(drafts), "tree_select: 1...7 drafts")
        let params = MLXArray([scale, 0, 0, 0, 0, 0, 0, 0])
        let out = selectKernel(
            [topIDs.asType(.int32), topRaw.asType(.float32), params],
            template: [
                ("K_", drafts), ("ROWS_", rows), ("W_", width), ("MAXLEAF_", maximumLeaves),
                ("MAXROW_", CBv2VerifyTree.maximumLeafRow), ("PROBE_", probe),
            ],
            grid: (32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[drafts], [drafts + 1], [4]],
            outputDTypes: [.int32, .int32, .int32])
        return (out[0], out[1], out[2])
    }

    /// Host mirror of `select` (load-time self-test, offline tools).
    public static func selectOnHost(ids: [Int32], raw: [Float], drafts: Int, probe: Bool)
        -> (tokens: [Int32], rowPositions: [Int32], chain: Int, leaves: Int)
    {
        let r = rows
        let w = width
        func plain() -> ([Int32], [Int32], Int, Int) {
            ((0 ..< drafts).map { ids[$0 * w] }, (0 ... drafts).map { Int32($0) }, drafts, 0)
        }
        if probe {
            guard drafts >= 2 else { return plain() }
            let tokens = (0 ..< drafts - 1).map { ids[$0 * w] } + [ids[0]]
            let positions = (0 ..< drafts).map { Int32($0) } + [1]
            return (tokens, positions, drafts - 1, 1)
        }
        var p1 = [Float](repeating: 0, count: r)
        var p2 = [Float](repeating: 0, count: r)
        for row in 0 ..< r {
            var m = -Float.infinity
            for i in 0 ..< w { m = max(m, raw[row * w + i] * scale) }
            var z: Float = 0
            var e0: Float = 0
            var e1: Float = 0
            for i in 0 ..< w {
                let e = Foundation.exp(raw[row * w + i] * scale - m)
                z += e
                if i == 0 { e0 = e }
                if i == 1 { e1 = e }
            }
            p1[row] = z > 0 ? e0 / z : 0
            p2[row] = z > 0 ? e1 / z : 0
        }
        var chain = 0
        var leafRows: [Int] = []
        var siblings: [(score: Float, row: Int, id: Int32, used: Bool)] = [(p2[0], 0, ids[1], false)]
        var reach: Float = 1
        while chain + leafRows.count < drafts {
            let chainScore: Float = chain < r ? reach * p1[chain] : -1
            let chainID: Int32 = chain < r ? ids[chain * w] : Int32.max
            var best = -1
            if leafRows.count < maximumLeaves {
                for i in siblings.indices where !siblings[i].used {
                    if best < 0 || siblings[i].score > siblings[best].score
                        || (siblings[i].score == siblings[best].score
                            && siblings[i].id < siblings[best].id)
                    {
                        best = i
                    }
                }
            }
            let takeChain: Bool
            if chainScore < 0 {
                takeChain = false
            } else if best < 0 {
                takeChain = true
            } else {
                takeChain =
                    chainScore > siblings[best].score
                    || (chainScore == siblings[best].score && chainID <= siblings[best].id)
            }
            if takeChain {
                reach *= p1[chain]
                chain += 1
                if chain < r, chain < CBv2VerifyTree.maximumLeafRow, siblings.count < 4 {
                    siblings.append((reach * p2[chain], chain, ids[chain * w + 1], false))
                }
            } else if best >= 0 {
                siblings[best].used = true
                leafRows.append(siblings[best].row)
            } else {
                break
            }
        }
        guard !leafRows.isEmpty, chain >= 1, chain + leafRows.count == drafts else { return plain() }
        let tokens = (0 ..< chain).map { ids[$0 * w] } + leafRows.map { ids[$0 * w + 1] }
        let positions = (0 ... chain).map { Int32($0) } + leafRows.map { Int32($0 + 1) }
        return (tokens, positions, chain, leafRows.count)
    }

    /// A read-back layout is one this design can verify exactly: chain columns
    /// at depths 1...chain, leaves at depths 1...min(chain + 1, 4) after them,
    /// leaf tokens taken from the leaf's own row. False: commit the chain only.
    public static func layoutIsValid(rowPositions: [Int32], chain: Int, leaves: Int, drafts: Int)
        -> Bool
    {
        guard rowPositions.count == drafts + 1, chain >= 1, leaves >= 0, chain + leaves == drafts,
            rowPositions[0] == 0
        else { return false }
        for t in 1 ... chain where rowPositions[t] != Int32(t) { return false }
        for t in (chain + 1) ..< (drafts + 1) {
            let depth = Int(rowPositions[t])
            guard depth >= 1, depth <= min(chain + 1, CBv2VerifyTree.maximumLeafRow) else {
                return false
            }
        }
        return true
    }

    /// Load-time check of `select` on random top-4 rows: every layout must be
    /// valid (`layoutIsValid`); `mismatches` counts layouts that differ from the
    /// host mirror (float `exp` may round a near-tie the other way, which only
    /// changes speed). False disables the tree.
    public static func selectSelfTest() -> (valid: Bool, mismatches: Int) {
        var generator = SystemRandomNumberGenerator()
        var mismatches = 0
        for trial in 0 ..< 24 {
            var ids: [Int32] = []
            var raw: [Float] = []
            for _ in 0 ..< rows {
                var values = (0 ..< width).map { _ in Float(Int.random(in: 0 ... 4000, using: &generator)) }
                values.sort(by: >)
                for i in 1 ..< width where values[i] == values[i - 1] { values[i] -= Float(i) }
                raw += values
                ids += (0 ..< width).map { _ in Int32.random(in: 0 ..< 200_000, using: &generator) }
            }
            let drafts = 2 + trial % 6
            let probe = trial % 8 == 7
            let host = selectOnHost(ids: ids, raw: raw, drafts: drafts, probe: probe)
            let device = select(
                topIDs: MLXArray(ids, [rows, width]), topRaw: MLXArray(raw, [rows, width]),
                drafts: drafts, probe: probe)
            eval(device.tokens, device.rowPositions, device.meta)
            let meta = device.meta.asArray(Int32.self)
            let positions = device.rowPositions.asArray(Int32.self)
            guard layoutIsValid(
                rowPositions: positions, chain: Int(meta[0]), leaves: Int(meta[1]), drafts: drafts)
            else { return (false, mismatches) }
            if device.tokens.asArray(Int32.self) != host.tokens || positions != host.rowPositions
                || Int(meta[0]) != host.chain || Int(meta[1]) != host.leaves
            {
                mismatches += 1
            }
        }
        return (true, mismatches)
    }
}
