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
        for (int k = 0; k < MAXV; k++) {
          if (live[k]) {
            int e = idx[k];
            float o = f2[k] * r2;
            o = o * (1.0f + static_cast<float>(w2[e]));
            yr[e] = static_cast<bfloat16_t>(o);
          }
        }
        """

    static let kernel = MLXFast.metalKernel(
        name: "verify_add_norm_norm",
        inputNames: ["residual", "x", "w1", "w2", "params"],
        outputNames: ["h", "y"],
        source: source, header: header)

    /// `(residual + centeredRMSNorm(x, w1, eps1), centeredRMSNorm(that, w2, eps2))`.
    /// `residual` and `x` are `[..., width]` BF16 with fewer than 32 rows.
    public static func fusedAddNormNorm(
        residual: MLXArray, x: MLXArray, w1: MLXArray, eps1: Float, w2: MLXArray, eps2: Float
    ) -> (h: MLXArray, y: MLXArray) {
        let width = x.dim(-1)
        let rows = x.size / width
        let tg = threadgroupSize(width: width)
        let blocks = width / (tg * readsPerThread)
        let params = MLXArray([Float(1.0 / Double(width)), eps1, eps2])
        let out = kernel(
            [residual, x, w1, w2, params],
            template: [("D_", width), ("BLOCKS_", blocks)],
            grid: (tg, rows, 1), threadGroup: (tg, 1, 1),
            outputShapes: [x.shape, x.shape], outputDTypes: [x.dtype, x.dtype])
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
public enum VerifyFusedQK {
    /// `DARKBLOOM_VERIFY_FUSED_QK=0` keeps the tower's own attention inputs.
    public static let enabled: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_VERIFY_FUSED_QK"] != "0"

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
          float L = scale * static_cast<float>(l + batch_offset);
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

    static let kernel = MLXFast.metalKernel(
        name: "verify_qk_norm_rope",
        inputNames: ["q", "k", "offsets", "params"],
        outputNames: ["qo", "ko"],
        source: source)

    public static func supports(headDim: Int, dtype: DType) -> Bool {
        headDim == 128 && dtype == .bfloat16
    }

    /// `queries` `[B, L, H*D]`, `keys` `[B, L, KVH*D]` (BF16, straight from
    /// the projections). `ropeBase` nil skips RoPE (NoPE layers).
    public static func apply(
        queries: MLXArray, keys: MLXArray, heads: Int, kvHeads: Int, headDim: Int,
        eps: Float, queryScale: Float, ropeBase: Float?, ropeScale: Float = 1,
        offsets: MLXArray?
    ) -> (queries: MLXArray, keys: MLXArray) {
        let (b, l) = (queries.dim(0), queries.dim(1))
        let offs = offsets.map { $0.dtype == .int32 ? $0 : $0.asType(.int32) }
            ?? MLXArray.zeros([1], type: Int32.self)
        let params = MLXArray([eps, queryScale, log2(ropeBase ?? 1), ropeScale])
        let out = kernel(
            [queries, keys, offs, params],
            template: [
                ("H_", heads), ("KVH_", kvHeads), ("D_", headDim), ("L_", l),
                ("ROPE_", ropeBase != nil), ("OFFB_", offs.size == 1 ? 0 : 1),
            ],
            grid: (32, b * l * (heads + kvHeads), 1), threadGroup: (32, 1, 1),
            outputShapes: [[b, l, heads, headDim], [b, l, kvHeads, headDim]],
            outputDTypes: [queries.dtype, keys.dtype])
        return (out[0].transposed(0, 2, 1, 3), out[1].transposed(0, 2, 1, 3))
    }
}
