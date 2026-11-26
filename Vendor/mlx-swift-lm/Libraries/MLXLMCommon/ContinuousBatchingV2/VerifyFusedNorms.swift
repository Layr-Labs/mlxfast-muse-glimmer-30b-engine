// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

// Fused block-norm boundaries for exact verify forwards.
//
// A centered RMS norm compiled by MLX runs as three dispatches at verify widths
// (an elementwise region writing `f` and `f * f`, MLX's `row_reduce_looped` sum,
// then the scale region). `fusedAddNormNorm` computes
//   h = residual + centeredRMSNorm(x, w1, eps1)
//   y = centeredRMSNorm(h, w2, eps2)
// in one dispatch, one threadgroup per row. Every output element keeps the
// reference's arithmetic: the sum of squares is accumulated in exactly the
// order `row_reduce_looped` uses for a row below 32 rows (threadgroup size
// from `threadgroup_size_from_row_size`, `REDUCE_N_READS` = 4, per-thread
// sequential adds, `simd_sum`, then `simd_sum` over the simdgroup partials),
// the mean multiplies by the FP32 reciprocal of the width, and the scale and
// cast steps are the compiled region's, operation for operation.

/// Implemented by a model adapter that can run a verify forward with fused
/// kernels. Nil means "not applicable here": the caller runs its ordinary
/// forward. The result must be byte-identical to `forwardWithHidden`.
public protocol CBv2FusedVerifyForwarding: AnyObject {
    func fusedVerifyForwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)?
}

public protocol CBv2FusedVerifyReadoutForwarding: CBv2FusedVerifyForwarding {
    func fusedVerifyForwardScoredWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (
            topTwoIDs: MLXArray, topTwoValues: MLXArray, nonFinite: MLXArray, lastHidden: MLXArray
        )?
}

public enum VerifyFusedNorms {
    /// `DARKBLOOM_VERIFY_FUSED_NORMS=0` turns the fused verify forward off.
    public static let enabled: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_VERIFY_FUSED_NORMS"] != "0"

    static let readsPerThread = 4

    /// MLX's `threadgroup_size_from_row_size` for rows above 1,024.
    static func threadgroupSize(width: Int) -> Int {
        let t = ((width + readsPerThread - 1) / readsPerThread + 31) / 32 * 32
        return min(1024, t)
    }

    /// Widths the kernel reproduces: MLX's large-row branch (above 1,024).
    public static func supports(width: Int, rows: Int, dtype: DType) -> Bool {
        width > 1024 && rows > 0 && rows < 32 && dtype == .bfloat16
    }

    static let header = """
        // Mirrors per_thread_row_reduce + threadgroup_reduce (Sum, float) for one row.
        template <int BLOCKS, int NR>
        METAL_FUNC float verify_row_sum(
            thread const float* sq, int blocks, int extra, uint lid, uint lane, uint sg,
            uint nsg, threadgroup float* shared_vals, threadgroup float* bcast) {
          float total = 0.0f;
          for (int i = 0; i < blocks; i++) {
            for (int j = 0; j < NR; j++) {
              total = sq[i * NR + j] + total;
            }
          }
          int index = int(lid) * NR;
          if (index + NR <= extra) {
            for (int j = 0; j < NR; j++) {
              total = sq[blocks * NR + j] + total;
            }
          } else {
            for (int j = 0; index + j < extra; j++) {
              total = sq[blocks * NR + j] + total;
            }
          }
          total = simd_sum(total);
          if (nsg > 1) {
            if (lane == 0) {
              shared_vals[sg] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float v = (lid < nsg) ? shared_vals[lid] : 0.0f;
            total = simd_sum(v);
          }
          if (lid == 0) {
            bcast[0] = total;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          return bcast[0];
        }
        """

    static let source = """
        constexpr int NR = 4;
        constexpr int MAXV = (BLOCKS_ + 1) * NR;
        threadgroup float shared_a[32];
        threadgroup float shared_b[32];
        threadgroup float bcast_a[1];
        threadgroup float bcast_b[1];
        uint lid = thread_position_in_threadgroup.x;
        uint lsize = threads_per_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint nsg = simdgroups_per_threadgroup;
        uint row = threadgroup_position_in_grid.y;
        const int D = D_;
        const int blocks = D / int(lsize * NR);
        const int extra = D - blocks * int(lsize * NR);
        const float inv = params[0];
        const float eps1 = params[1];
        const float eps2 = params[2];

        const device bfloat16_t* xr = x + size_t(row) * D;
        const device bfloat16_t* rr = residual + size_t(row) * D;
        device bfloat16_t* hr = h + size_t(row) * D;
        device bfloat16_t* yr = y + size_t(row) * D;

        int idx[MAXV];
        bool live[MAXV];
        float f[MAXV];
        float sq[MAXV];
        for (int i = 0; i <= blocks; i++) {
          for (int j = 0; j < NR; j++) {
            int k = i * NR + j;
            int e = i * int(lsize * NR) + int(lid) * NR + j;
            bool ok = (i < blocks) || (int(lid) * NR + j < extra);
            idx[k] = e;
            live[k] = ok;
            f[k] = ok ? static_cast<float>(xr[e]) : 0.0f;
            sq[k] = f[k] * f[k];
          }
        }
        float total1 = verify_row_sum<BLOCKS_, NR>(
            sq, blocks, extra, lid, lane, sg, nsg, shared_a, bcast_a);
        float var1 = total1 * inv;
        float r1 = metal::precise::rsqrt(var1 + eps1);
        float f2[MAXV];
        float sq2[MAXV];
        for (int k = 0; k < MAXV; k++) {
          if (live[k]) {
            int e = idx[k];
            float o = f[k] * r1;
            o = o * (1.0f + static_cast<float>(w1[e]));
            bfloat16_t ob = static_cast<bfloat16_t>(o);
            bfloat16_t hb = rr[e] + ob;
            hr[e] = hb;
            f2[k] = static_cast<float>(hb);
          } else {
            f2[k] = 0.0f;
          }
          sq2[k] = f2[k] * f2[k];
        }
        float total2 = verify_row_sum<BLOCKS_, NR>(
            sq2, blocks, extra, lid, lane, sg, nsg, shared_b, bcast_b);
        float var2 = total2 * inv;
        float r2 = metal::precise::rsqrt(var2 + eps2);
        if constexpr (CAST_FP8_) {
          const float m0 = fp8_mu[0];
          const float m1 = fp8_mu[1];
          const float bound = fp8_bd[0];
          const float nan_value = metal::numeric_limits<float>::quiet_NaN();
          for (int k = 0; k < MAXV; k++) {
            if (live[k]) {
              int e = idx[k];
              float o = f2[k] * r2;
              o = o * (1.0f + static_cast<float>(w2[e]));
              bfloat16_t yb = static_cast<bfloat16_t>(o);
              float scaled = static_cast<float>(yb) * m0;
              uint8_t bits = modelopt_e4m3_bits(scaled);
              float decoded = (bits & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(bits);
              bool invalid = metal::isnan(scaled) || metal::abs(scaled) > bound;
              yr[e] = static_cast<bfloat16_t>(invalid ? nan_value : decoded * m1);
            }
          }
        } else {
          for (int k = 0; k < MAXV; k++) {
            if (live[k]) {
              int e = idx[k];
              float o = f2[k] * r2;
              o = o * (1.0f + static_cast<float>(w2[e]));
              yr[e] = static_cast<bfloat16_t>(o);
            }
          }
        }
        """

    static let kernel = MLXFast.metalKernel(
        name: "verify_add_norm_norm",
        inputNames: ["residual", "x", "w1", "w2", "params", "fp8_mu", "fp8_bd"],
        outputNames: ["h", "y"],
        source: source, header: ModelOptActivationKernels.header + header)

    private struct ParamsKey: Hashable {
        let width: Int
        let eps1Bits: UInt32
        let eps2Bits: UInt32
    }

    private struct NormLaunchKey: Hashable {
        let shape: [Int]
        let dtype: DType
        let width: Int
        let blocks: Int
        let castFP8: Bool
    }

    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var paramsCache: [ParamsKey: MLXArray] = [:]
    nonisolated(unsafe) private static var launchCache: [NormLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static let dummyFP8Mu: MLXArray = {
        let a = MLXArray([Float(1), Float(1)])
        eval(a)
        return a
    }()
    nonisolated(unsafe) private static let dummyFP8Bound: MLXArray = {
        let a = MLXArray([Float(448)])
        eval(a)
        return a
    }()

    private static func cachedLaunch(
        shape: [Int], dtype: DType, width: Int, rows: Int, tg: Int, blocks: Int, castFP8: Bool
    ) -> MLXFast.MLXFastKernel.Launch {
        let key = NormLaunchKey(
            shape: shape, dtype: dtype, width: width, blocks: blocks, castFP8: castFP8)
        if let hit = cacheLock.withLock({ launchCache[key] }) { return hit }
        let made = kernel.prepare(
            template: [("D_", width), ("BLOCKS_", blocks), ("CAST_FP8_", castFP8)],
            grid: (tg, rows, 1), threadGroup: (tg, 1, 1),
            outputShapes: [shape, shape], outputDTypes: [dtype, dtype])
        cacheLock.withLock { launchCache[key] = made }
        return made
    }

    private static func cachedParams(width: Int, eps1: Float, eps2: Float) -> MLXArray {
        let key = ParamsKey(width: width, eps1Bits: eps1.bitPattern, eps2Bits: eps2.bitPattern)
        return cacheLock.withLock {
            if let existing = paramsCache[key] { return existing }
            let arr = MLXArray([Float(1.0 / Double(width)), eps1, eps2])
            eval(arr)
            paramsCache[key] = arr
            return arr
        }
    }

    /// `(residual + centeredRMSNorm(x, w1, eps1), centeredRMSNorm(that, w2, eps2))`.
    /// `residual` and `x` are `[..., width]` BF16 with fewer than 32 rows.
    public static func fusedAddNormNorm(
        residual: MLXArray, x: MLXArray, w1: MLXArray, eps1: Float, w2: MLXArray, eps2: Float
    ) -> (h: MLXArray, y: MLXArray) {
        let width = x.dim(-1)
        let rows = x.size / width
        let tg = threadgroupSize(width: width)
        let blocks = width / (tg * readsPerThread)
        let params = cachedParams(width: width, eps1: eps1, eps2: eps2)
        let launch = cachedLaunch(
            shape: x.shape, dtype: x.dtype, width: width, rows: rows, tg: tg, blocks: blocks,
            castFP8: false)
        let out = launch([residual, x, w1, w2, params, dummyFP8Mu, dummyFP8Bound])
        return (out[0], out[1])
    }

    /// `(residual + centeredRMSNorm(x, w1, eps1), fp8Cast(centeredRMSNorm(that, w2, eps2)))`
    /// in one dispatch, bit-identical to `fusedAddNormNorm` followed by `ModelOptActivationKernels.fp8`.
    public static func fusedAddNormNormCastFP8(
        residual: MLXArray, x: MLXArray, w1: MLXArray, eps1: Float, w2: MLXArray, eps2: Float,
        multipliers: MLXArray, overflowBound: MLXArray
    ) -> (h: MLXArray, yCast: MLXArray) {
        let width = x.dim(-1)
        let rows = x.size / width
        let tg = threadgroupSize(width: width)
        let blocks = width / (tg * readsPerThread)
        let params = cachedParams(width: width, eps1: eps1, eps2: eps2)
        let launch = cachedLaunch(
            shape: x.shape, dtype: x.dtype, width: width, rows: rows, tg: tg, blocks: blocks,
            castFP8: true)
        let out = launch([residual, x, w1, w2, params, multipliers, overflowBound])
        return (out[0], out[1])
    }
}

// MARK: - Q/K norm, query scale and RoPE

/// The attention input chain of an exact verify forward in one dispatch:
/// unit RMS norm of every query and key head (`MLXFast.rmsNorm`'s
/// `rms_single_row`, 32 threads x 4 reads at head_dim 128), the FP32 query
/// scale with its BF16 round trip, then MLX's non-traditional `rope` kernel
/// with per-row array offsets. Each step keeps the reference's expressions
/// and rounding points. Outputs are `[B, L, heads, D]` memory returned as
/// `[B, heads, L, D]` views.
public struct CBv2DirectVerifyKVTarget {
    let group: PagedKVGroup
    let segment: PagedKVSegment
    let triples: [Int32]
    let pageSize: Int
    let kvHeads: Int
    let headDim: Int
}

public enum VerifyFusedQK {
    /// `DARKBLOOM_VERIFY_FUSED_QK=0` keeps the tower's own attention inputs.
    public static let enabled: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_VERIFY_FUSED_QK"] != "0"

    /// `CBV2_FUSED_QK_WRITE=0` disables fused K/V paged-segment writing in `VerifyFusedQK`.
    public static let fusedWriteEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_FUSED_QK_WRITE"] != "0"

    static let source = """
        uint lane = thread_position_in_threadgroup.x;
        uint g = threadgroup_position_in_grid.y;
        constexpr int HT = H_ + KVH_;
        int head = int(g) % HT;
        int tok = int(g) / HT;
        int b = tok / L_;
        uint l = uint(tok % L_);
        bool isq = head < H_;
        const device bfloat16_t* src = isq
            ? q + size_t(tok) * H_ * D_ + size_t(head) * D_
            : k + size_t(tok) * KVH_ * D_ + size_t(head - H_) * D_;
        device bfloat16_t* dst = isq
            ? qo + size_t(tok) * H_ * D_ + size_t(head) * D_
            : ko + size_t(tok) * KVH_ * D_ + size_t(head - H_) * D_;
        const float eps = params[0];
        const float qscale = params[1];
        const float base = params[2];
        const float scale = params[3];
        const uint axis_size = D_;

        float thread_x[4];
        float acc = 0;
        src += lane * 4;
        for (int i = 0; i < 4; i++) {
          thread_x[i] = src[i];
          acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        float part = (lane == 0) ? acc : 0.0f;
        acc = simd_sum(part);
        float inv_mean = metal::precise::rsqrt(acc / axis_size + eps);

        bfloat16_t one = static_cast<bfloat16_t>(1.0f);
        float v[4];
        for (int i = 0; i < 4; i++) {
          bfloat16_t n = one * static_cast<bfloat16_t>(thread_x[i] * inv_mean);
          if (isq) {
            n = static_cast<bfloat16_t>(static_cast<float>(n) * qscale);
          }
          v[i] = static_cast<float>(n);
        }
        dst += lane * 4;
        if (ROPE_) {
          const uint half_dims = D_ / 2;
          bool low = lane < 16;
          int batch_offset = offsets[b * OFFB_];
          const uint pos_packed = DYN_TREE_ ? uint(tree_meta[0]) : uint(TREE_POS_);
          const uint pos_delta = (l < 8u) ? ((pos_packed >> (l * 4u)) & 0xFu) : l;
          float L = scale * static_cast<float>(int(pos_delta) + batch_offset);
          for (int i = 0; i < 4; i++) {
            float other = simd_shuffle_xor(v[i], ushort(16));
            uint px = (low ? lane : lane - 16) * 4 + i;
            float d = static_cast<float>(px) / static_cast<float>(half_dims);
            float inv_freq = metal::exp2(-d * base);
            float theta = L * inv_freq;
            float costheta = metal::fast::cos(theta);
            float sintheta = metal::fast::sin(theta);
            float x1 = low ? v[i] : other;
            float x2 = low ? other : v[i];
            float rx1 = x1 * costheta - x2 * sintheta;
            float rx2 = x1 * sintheta + x2 * costheta;
            dst[i] = static_cast<bfloat16_t>(low ? rx1 : rx2);
          }
        } else {
          for (int i = 0; i < 4; i++) {
            dst[i] = static_cast<bfloat16_t>(v[i]);
          }
        }
        """

    static let writeSource = """
        uint lane = thread_position_in_threadgroup.x;
        uint g = threadgroup_position_in_grid.y;
        constexpr int HT = H_ + KVH_;
        int head = int(g) % HT;
        int tok = int(g) / HT;
        int b = tok / L_;
        uint l = uint(tok % L_);
        bool isq = head < H_;
        const int kvh = head - H_;
        const device bfloat16_t* src = isq
            ? q + size_t(tok) * H_ * D_ + size_t(head) * D_
            : k + size_t(tok) * KVH_ * D_ + size_t(kvh) * D_;
        const float eps = params[0];
        const float qscale = params[1];
        const float base = params[2];
        const float scale = params[3];
        const uint axis_size = D_;

        float thread_x[4];
        float acc = 0;
        src += lane * 4;
        for (int i = 0; i < 4; i++) {
          thread_x[i] = src[i];
          acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        float part = (lane == 0) ? acc : 0.0f;
        acc = simd_sum(part);
        float inv_mean = metal::precise::rsqrt(acc / axis_size + eps);

        bfloat16_t one = static_cast<bfloat16_t>(1.0f);
        float v[4];
        for (int i = 0; i < 4; i++) {
          bfloat16_t n = one * static_cast<bfloat16_t>(thread_x[i] * inv_mean);
          if (isq) {
            n = static_cast<bfloat16_t>(static_cast<float>(n) * qscale);
          }
          v[i] = static_cast<float>(n);
        }
        bfloat16_t out4[4];
        if (ROPE_) {
          const uint half_dims = D_ / 2;
          bool low = lane < 16;
          int batch_offset = offsets[b * OFFB_];
          const uint pos_packed = DYN_TREE_ ? uint(tree_meta[0]) : uint(TREE_POS_);
          const uint pos_delta = (l < 8u) ? ((pos_packed >> (l * 4u)) & 0xFu) : l;
          float L = scale * static_cast<float>(int(pos_delta) + batch_offset);
          for (int i = 0; i < 4; i++) {
            float other = simd_shuffle_xor(v[i], ushort(16));
            uint px = (low ? lane : lane - 16) * 4 + i;
            float d = static_cast<float>(px) / static_cast<float>(half_dims);
            float inv_freq = metal::exp2(-d * base);
            float theta = L * inv_freq;
            float costheta = metal::fast::cos(theta);
            float sintheta = metal::fast::sin(theta);
            float x1 = low ? v[i] : other;
            float x2 = low ? other : v[i];
            float rx1 = x1 * costheta - x2 * sintheta;
            float rx2 = x1 * sintheta + x2 * costheta;
            out4[i] = static_cast<bfloat16_t>(low ? rx1 : rx2);
          }
        } else {
          for (int i = 0; i < 4; i++) {
            out4[i] = static_cast<bfloat16_t>(v[i]);
          }
        }
        if (isq) {
          device bfloat16_t* dst = qo + size_t(tok) * H_ * D_ + size_t(head) * D_ + lane * 4;
          for (int i = 0; i < 4; i++) {
            dst[i] = out4[i];
          }
        } else {
          const int page = records[tok * 3 + 1];
          const int slot = records[tok * 3 + 2];
          if (page > 0) {
            const device bfloat16_t* vsrc =
                vals + size_t(tok) * KVH_ * D_ + size_t(kvh) * D_ + lane * 4;
            const size_t target =
                (((size_t)page * KVH_ + size_t(kvh)) * S_ + size_t(slot)) * D_ + size_t(lane) * 4;
            const size_t vbase = size_t(vbase_arr[0]);
            device bfloat16_t* dst_s = storage;
            for (int i = 0; i < 4; i++) {
              dst_s[target + i] = out4[i];
              dst_s[vbase + target + i] = vsrc[i];
            }
          }
        }
        if (lane == 0 && g == 0) {
          fence[0] = previous[0] + 1;
        }
        """

    static let writeKernel = MLXFast.metalKernel(
        name: "verify_qk_norm_rope_write",
        inputNames: [
            "q", "k", "vals", "offsets", "params", "tree_meta",
            "storage", "records", "vbase_arr", "previous",
        ],
        outputNames: ["qo", "fence"],
        source: writeSource,
        ensureRowContiguous: true,
        mutableInputs: ["storage"])

    private struct QKWriteLaunchKey: Hashable {
        let l: Int
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let pageSize: Int
        let rope: Bool
        let posDeltas: UInt32
        let dynTree: Bool
        let dtype: DType
    }

    nonisolated(unsafe) private static var qkWriteLaunchCache:
        [QKWriteLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var vbaseCache: [Int: MLXArray] = [:]

    private static func cachedVBase(_ valueOffset: Int) -> MLXArray {
        qkLock.withLock {
            if let existing = vbaseCache[valueOffset] { return existing }
            let arr = MLXArray([Int64(valueOffset)])
            eval(arr)
            vbaseCache[valueOffset] = arr
            return arr
        }
    }

    /// Computes normalized/scaled/RoPE queries directly as `[1, L, H, D]` while
    /// writing normalized/RoPE keys and raw values directly into the paged KV segment.
    public static func applyAndWrite(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        heads: Int, kvHeads: Int, headDim: Int,
        eps: Float, queryScale: Float, ropeBase: Float?, ropeScale: Float = 1,
        offsets: MLXArray?, posDeltas: UInt32 = 0x7654_3210, treeMeta: MLXArray? = nil,
        target: CBv2DirectVerifyKVTarget
    ) -> MLXArray {
        let l = queries.dim(1)
        let offs = offsets.map { $0.dtype == .int32 ? $0 : $0.asType(.int32) } ?? zeroOffset
        let params = cachedQKParams(
            eps: eps, queryScale: queryScale, ropeBase: ropeBase, ropeScale: ropeScale)
        let dynTree = treeMeta != nil
        let key = QKWriteLaunchKey(
            l: l, heads: heads, kvHeads: kvHeads, headDim: headDim, pageSize: target.pageSize,
            rope: ropeBase != nil, posDeltas: dynTree ? 0x7654_3210 : posDeltas,
            dynTree: dynTree, dtype: queries.dtype)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = qkLock.withLock({ qkWriteLaunchCache[key] }) {
            launch = hit
        } else {
            let made = writeKernel.prepare(
                template: [
                    ("H_", heads), ("KVH_", kvHeads), ("D_", headDim), ("L_", l),
                    ("S_", target.pageSize), ("ROPE_", ropeBase != nil), ("OFFB_", 0),
                    ("TREE_POS_", Int(dynTree ? 0x7654_3210 : posDeltas)),
                    ("DYN_TREE_", dynTree),
                ],
                grid: (32, l * (heads + kvHeads), 1), threadGroup: (32, 1, 1),
                outputShapes: [[1, l, heads, headDim], [1]],
                outputDTypes: [queries.dtype, .int32])
            qkLock.withLock { qkWriteLaunchCache[key] = made }
            launch = made
        }
        let recArr = PagedSegmentTransfers.records(target.triples)
        let vbase = cachedVBase(target.segment.valueOffset)
        let out = launch([
            queries, keys, values, offs, params, treeMeta ?? dummyTreeMeta,
            target.segment.storage, recArr, vbase, target.group.writeFence,
        ])
        target.group.writeFence = out[1]
        return out[0]
    }

    /// Warms the fused QK+KV-write kernel variants on dummy zero storage (page=0 no-op).
    public static func warmDirectWrite(
        heads: Int, kvHeads: Int, headDim: Int, pageSize: Int, dtype: DType,
        eps: Float, queryScale: Float, bases: [Float?], warmDynamicTree: Bool
    ) -> [MLXArray] {
        guard fusedWriteEnabled else { return [] }
        let dummyStorage = MLXArray.zeros([2 * kvHeads * pageSize * headDim], dtype: dtype)
        let dummyFence = MLXArray([Int32(0)])
        let vbase = cachedVBase(kvHeads * pageSize * headDim)
        let off = MLXArray.zeros([1], type: Int32.self)
        // Must have >= 8 elements (`device` address space) to match live 18-element `activeDynamicTreeMeta`.
        let dynMeta = MLXArray.zeros([18], type: Int32.self)
        var warm = [MLXArray]()
        for l in 2 ... 8 {
            let q = MLXArray.zeros([1, l, heads * headDim], dtype: dtype)
            let k = MLXArray.zeros([1, l, kvHeads * headDim], dtype: dtype)
            let v = MLXArray.zeros([1, l, kvHeads * headDim], dtype: dtype)
            let recArr = PagedSegmentTransfers.records([Int32](repeating: 0, count: l * 3))
            for base in bases {
                let params = cachedQKParams(
                    eps: eps, queryScale: queryScale, ropeBase: base, ropeScale: 1)
                let key = QKWriteLaunchKey(
                    l: l, heads: heads, kvHeads: kvHeads, headDim: headDim, pageSize: pageSize,
                    rope: base != nil, posDeltas: 0x7654_3210, dynTree: false, dtype: dtype)
                let launch: MLXFast.MLXFastKernel.Launch
                if let hit = qkLock.withLock({ qkWriteLaunchCache[key] }) {
                    launch = hit
                } else {
                    let made = writeKernel.prepare(
                        template: [
                            ("H_", heads), ("KVH_", kvHeads), ("D_", headDim), ("L_", l),
                            ("S_", pageSize), ("ROPE_", base != nil), ("OFFB_", 0),
                            ("TREE_POS_", Int(0x7654_3210)), ("DYN_TREE_", false),
                        ],
                        grid: (32, l * (heads + kvHeads), 1), threadGroup: (32, 1, 1),
                        outputShapes: [[1, l, heads, headDim], [1]],
                        outputDTypes: [dtype, .int32])
                    qkLock.withLock { qkWriteLaunchCache[key] = made }
                    launch = made
                }
                let out = launch([
                    q, k, v, off, params, dummyTreeMeta, dummyStorage, recArr, vbase, dummyFence,
                ])
                warm += out
                if warmDynamicTree {
                    let dKey = QKWriteLaunchKey(
                        l: l, heads: heads, kvHeads: kvHeads, headDim: headDim, pageSize: pageSize,
                        rope: base != nil, posDeltas: 0x7654_3210, dynTree: true, dtype: dtype)
                    let dLaunch: MLXFast.MLXFastKernel.Launch
                    if let hit = qkLock.withLock({ qkWriteLaunchCache[dKey] }) {
                        dLaunch = hit
                    } else {
                        let made = writeKernel.prepare(
                            template: [
                                ("H_", heads), ("KVH_", kvHeads), ("D_", headDim), ("L_", l),
                                ("S_", pageSize), ("ROPE_", base != nil), ("OFFB_", 0),
                                ("TREE_POS_", Int(0x7654_3210)), ("DYN_TREE_", true),
                            ],
                            grid: (32, l * (heads + kvHeads), 1), threadGroup: (32, 1, 1),
                            outputShapes: [[1, l, heads, headDim], [1]],
                            outputDTypes: [dtype, .int32])
                        qkLock.withLock { qkWriteLaunchCache[dKey] = made }
                        dLaunch = made
                    }
                    let dOut = dLaunch([
                        q, k, v, off, params, dynMeta, dummyStorage, recArr, vbase, dummyFence,
                    ])
                    warm += dOut
                }
            }
        }
        return warm
    }

    static let kernel = MLXFast.metalKernel(
        name: "verify_qk_norm_rope",
        inputNames: ["q", "k", "offsets", "params", "tree_meta"],
        outputNames: ["qo", "ko"],
        source: source)

    public static func supports(headDim: Int, dtype: DType) -> Bool {
        headDim == 128 && dtype == .bfloat16
    }

    private struct QKParamsKey: Hashable {
        let epsBits: UInt32
        let qScaleBits: UInt32
        let baseBits: UInt32
        let rScaleBits: UInt32
    }

    private struct QKLaunchKey: Hashable {
        let b: Int
        let l: Int
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let rope: Bool
        let offB: Int
        let posDeltas: UInt32
        let dynTree: Bool
        let qDType: DType
        let kDType: DType
    }

    private static let qkLock = NSLock()
    nonisolated(unsafe) private static var qkParamsCache: [QKParamsKey: MLXArray] = [:]
    nonisolated(unsafe) private static var qkLaunchCache: [QKLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static let zeroOffset = MLXArray.zeros([1], type: Int32.self)
    nonisolated(unsafe) private static let dummyTreeMeta: MLXArray = {
        let arr = MLXArray([Int32(bitPattern: 0x7654_3210)])
        eval(arr)
        return arr
    }()

    private static func cachedQKParams(
        eps: Float, queryScale: Float, ropeBase: Float?, ropeScale: Float
    ) -> MLXArray {
        let baseLog = log2(ropeBase ?? 1)
        let key = QKParamsKey(
            epsBits: eps.bitPattern, qScaleBits: queryScale.bitPattern,
            baseBits: baseLog.bitPattern, rScaleBits: ropeScale.bitPattern)
        return qkLock.withLock {
            if let existing = qkParamsCache[key] { return existing }
            let arr = MLXArray([eps, queryScale, baseLog, ropeScale])
            eval(arr)
            qkParamsCache[key] = arr
            return arr
        }
    }

    /// `queries` `[B, L, H*D]`, `keys` `[B, L, KVH*D]` (BF16, straight from
    /// the projections). `ropeBase` nil skips RoPE (NoPE layers).
    public static func apply(
        queries: MLXArray, keys: MLXArray, heads: Int, kvHeads: Int, headDim: Int,
        eps: Float, queryScale: Float, ropeBase: Float?, ropeScale: Float = 1,
        offsets: MLXArray?, posDeltas: UInt32 = 0x7654_3210, treeMeta: MLXArray? = nil
    ) -> (queries: MLXArray, keys: MLXArray) {
        let (b, l) = (queries.dim(0), queries.dim(1))
        let offs = offsets.map { $0.dtype == .int32 ? $0 : $0.asType(.int32) } ?? zeroOffset
        let params = cachedQKParams(
            eps: eps, queryScale: queryScale, ropeBase: ropeBase, ropeScale: ropeScale)
        let offB = offs.size == 1 ? 0 : 1
        let dynTree = treeMeta != nil
        let key = QKLaunchKey(
            b: b, l: l, heads: heads, kvHeads: kvHeads, headDim: headDim,
            rope: ropeBase != nil, offB: offB, posDeltas: dynTree ? 0x7654_3210 : posDeltas,
            dynTree: dynTree, qDType: queries.dtype, kDType: keys.dtype)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = qkLock.withLock({ qkLaunchCache[key] }) {
            launch = hit
        } else {
            let made = kernel.prepare(
                template: [
                    ("H_", heads), ("KVH_", kvHeads), ("D_", headDim), ("L_", l),
                    ("ROPE_", ropeBase != nil), ("OFFB_", offB),
                    ("TREE_POS_", Int(dynTree ? 0x7654_3210 : posDeltas)),
                    ("DYN_TREE_", dynTree),
                ],
                grid: (32, b * l * (heads + kvHeads), 1), threadGroup: (32, 1, 1),
                outputShapes: [[b, l, heads, headDim], [b, l, kvHeads, headDim]],
                outputDTypes: [queries.dtype, keys.dtype])
            qkLock.withLock { qkLaunchCache[key] = made }
            launch = made
        }
        let out = launch([queries, keys, offs, params, treeMeta ?? dummyTreeMeta])
        return (out[0].transposed(0, 2, 1, 3), out[1].transposed(0, 2, 1, 3))
    }
}
