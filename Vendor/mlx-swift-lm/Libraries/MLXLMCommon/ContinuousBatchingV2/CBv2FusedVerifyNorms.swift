// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Fused norm, rotary and residual kernels for a speculative verify window
/// (2 to 8 exact rows). Every kernel here reproduces, element for element, the
/// bytes of the MLX operations it replaces; it only removes the dispatches and
/// the intermediate arrays between them.
///
/// * `addNorm`: `residual + centeredRMSNorm(x)` and, in the same dispatch, the
///   centered RMS norm of that sum (the next block's input norm). The squares
///   are summed in the order of MLX's `row_reduce_looped` on a 1024-thread group
///   (4 values per thread, simd_sum, one slot per simdgroup, simd_sum of the 32
///   slots), so the variance is the same float, and every later step is the
///   same float expression on the same rounded values: variance times the
///   host's `1 / width`, plus eps, `precise::rsqrt`, times the row, times
///   `1 + weight`, then one rounding to the activation dtype.
/// * `queryKeyRope`: q/k RMS norm (`rms_single_row`: one simdgroup per head),
///   the query scale in float32 with its own rounding, and RoPE
///   (`rope_impl`, exchanging the two halves of a head across the simdgroup).
///
/// A model runs `selfTest`-style comparisons against the unfused operations
/// before it takes this path (see `MuseGlimmerTextModel+CBv2.swift`).
public enum CBv2FusedVerifyNorms {
    /// Set `DARKBLOOM_MUSE_GLIMMER_FUSED_VERIFY_NORMS=0` to turn the fusion off.
    public static let environmentKey = "DARKBLOOM_MUSE_GLIMMER_FUSED_VERIFY_NORMS"
    public static let enabled: Bool =
        ProcessInfo.processInfo.environment[environmentKey] != "0"

    public static let maximumRows = 8

    /// The norm kernel reproduces MLX's looped row reduction of a 1024-thread
    /// group: widths from 4096 (where MLX picks 1024 threads) that are a
    /// multiple of 4 (whole reads in the tail).
    public static func supportsNorm(dtype: DType, width: Int) -> Bool {
        (dtype == .bfloat16 || dtype == .float16) && width >= 4096 && width % 4 == 0
            && width <= 16384
    }

    /// One simdgroup per head with 4 values per lane.
    public static func supportsAttention(dtype: DType, headDimensions: Int) -> Bool {
        (dtype == .bfloat16 || dtype == .float16) && headDimensions == 128
    }

    // MARK: Kernels

    private static let normHeader = """
    // Verify-window fusion helpers. Each line of arithmetic below repeats the
    // statement shape of the MLX kernel it reproduces (reduce_row.h row_reduce_looped
    // with 1024 threads, rms_norm.metal rms_single_row, rope.metal rope_impl), so the
    // compiler contracts the same expressions; products that MLX computes in a separate
    // kernel are kept in their own statements here.

    // `threadgroup_reduce` of reduce_row.h on a 1024-thread group: simd_sum, one slot
    // per simdgroup, simd_sum of the 32 slots. Every thread returns the total.
    inline float mo_fv_total(
        float acc, threadgroup float* sums, threadgroup float* total, uint lid, uint lane,
        uint sg) {
        acc = simd_sum(acc);
        if (lane == 0) {
            sums[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float value = (lid < 32) ? sums[lid] : 0.0f;
        value = simd_sum(value);
        if (lid == 0) {
            total[0] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return total[0];
    }
    """

    private static let normSource = """
    constexpr int NT = 1024;
    constexpr int NB = K / (NT * 4);
    constexpr int EXTRA = K - NB * (NT * 4);
    constexpr int NV = (NB + 1) * 4;
    const float nrm = scal[0];
    const float eps1 = scal[1];
    const float eps2 = scal[2];
    uint row = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float sums[32];
    threadgroup float total[1];
    const bool tail = (int(lid) * 4 + 4 <= EXTRA);
    const size_t base = size_t(row) * K;

    float v[NV];
    for (int b = 0; b < NB; b++) {
        for (int i = 0; i < 4; i++) {
            v[b * 4 + i] = float(a[base + b * (NT * 4) + lid * 4 + i]);
        }
    }
    if (tail) {
        for (int i = 0; i < 4; i++) {
            v[NB * 4 + i] = float(a[base + NB * (NT * 4) + lid * 4 + i]);
        }
    }
    float acc = 0.0f;
    for (int b = 0; b < NB; b++) {
        for (int i = 0; i < 4; i++) {
            float sq = v[b * 4 + i] * v[b * 4 + i];
            acc = sq + acc;
        }
    }
    if (tail) {
        for (int i = 0; i < 4; i++) {
            float sq = v[NB * 4 + i] * v[NB * 4 + i];
            acc = sq + acc;
        }
    }
    float tot = mo_fv_total(acc, sums, total, lid, lane, sg);
    float var1 = tot * nrm;
    float shifted1 = var1 + eps1;
    float rs1 = metal::precise::rsqrt(shifted1);
    T s[NV];
    for (int b = 0; b < NB + 1; b++) {
        if (b < NB || tail) {
            for (int i = 0; i < 4; i++) {
                size_t col = b * (NT * 4) + lid * 4 + i;
                float o = v[b * 4 + i] * rs1;
                o = o * (1.0f + float(w1[col]));
                T oc = static_cast<T>(o);
                if (HAS_RES) {
                    oc = r[base + col] + oc;
                }
                s[b * 4 + i] = oc;
                out0[base + col] = oc;
            }
        }
    }
    if (HAS_NEXT) {
        float acc2 = 0.0f;
        for (int b = 0; b < NB + 1; b++) {
            if (b < NB || tail) {
                for (int i = 0; i < 4; i++) {
                    float f = float(s[b * 4 + i]);
                    v[b * 4 + i] = f;
                    float sq = f * f;
                    acc2 = sq + acc2;
                }
            }
        }
        float tot2 = mo_fv_total(acc2, sums, total, lid, lane, sg);
        float var2 = tot2 * nrm;
        float shifted2 = var2 + eps2;
        float rs2 = metal::precise::rsqrt(shifted2);
        for (int b = 0; b < NB + 1; b++) {
            if (b < NB || tail) {
                for (int i = 0; i < 4; i++) {
                    size_t col = b * (NT * 4) + lid * 4 + i;
                    float o = v[b * 4 + i] * rs2;
                    o = o * (1.0f + float(w2[col]));
                    out1[base + col] = static_cast<T>(o);
                }
            }
        }
    }
    """

    private static let queryKeySource = """
    // One simdgroup per (row, head): q heads first, then key heads.
    constexpr int N_READS = 4;
    constexpr int HALF = D / 2;
    const float eps = fscal[0];
    const float rope_base = fscal[1];
    const float rope_scale = fscal[2];
    const float query_scale = fscal[3];
    uint item = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    const int heads_total = H + KVH;
    const int r = item / heads_total;
    const int hh = item % heads_total;
    const bool is_q = hh < H;
    const int head = is_q ? hh : hh - H;
    const int b = r / L;
    const int l = r % L;
    const device T* src = is_q ? (q + size_t(r) * H * D + size_t(head) * D)
                               : (k + size_t(r) * KVH * D + size_t(head) * D);
    float thread_x[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
        thread_x[i] = src[lane * N_READS + i];
        acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    float inv_mean = metal::precise::rsqrt(acc / float(D) + eps);
    float val[N_READS];
    for (int i = 0; i < N_READS; i++) {
        T n = static_cast<T>(thread_x[i] * inv_mean);
        if (is_q) {
            float scaled = static_cast<float>(n) * query_scale;
            n = static_cast<T>(scaled);
        }
        val[i] = static_cast<float>(n);
    }
    const int Hn = is_q ? H : KVH;
    device T* dst = is_q ? qo : ko;
    size_t out_row;
    if (ROPE) {
        out_row = ((size_t(b) * Hn + head) * L + l) * D;
    } else {
        out_row = ((size_t(b) * L + l) * Hn + head) * D;
    }
    if (ROPE) {
        const int batch_offset = offs[b * offs_stride_arr[0]];
        const uint pos_y = l;
        float L_pos = rope_scale * static_cast<float>(pos_y + batch_offset);
        const bool first_half = lane < 16;
        for (int i = 0; i < N_READS; i++) {
            float other = simd_shuffle_xor(val[i], ushort(16));
            int pos_x = first_half ? int(lane) * N_READS + i : int(lane) * N_READS + i - HALF;
            float d = static_cast<float>(pos_x) / static_cast<float>(HALF);
            float inv_freq = metal::exp2(-d * rope_base);
            float theta = L_pos * inv_freq;
            float costheta = metal::fast::cos(theta);
            float sintheta = metal::fast::sin(theta);
            float x1 = first_half ? val[i] : other;
            float x2 = first_half ? other : val[i];
            float rx;
            if (first_half) {
                rx = x1 * costheta - x2 * sintheta;
            } else {
                rx = x1 * sintheta + x2 * costheta;
            }
            dst[out_row + size_t(lane * N_READS + i)] = static_cast<T>(rx);
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            dst[out_row + size_t(lane * N_READS + i)] = static_cast<T>(val[i]);
        }
    }
    """

    private final class Kernels: @unchecked Sendable {
        let norm = MLXFast.metalKernel(
            name: "cbv2_fused_verify_norm",
            inputNames: ["a", "r", "w1", "w2", "scal"],
            outputNames: ["out0", "out1"],
            source: CBv2FusedVerifyNorms.normSource,
            header: CBv2FusedVerifyNorms.normHeader)
        let queryKey = MLXFast.metalKernel(
            name: "cbv2_fused_verify_qk_rope",
            inputNames: ["q", "k", "offs", "fscal", "offs_stride_arr"],
            outputNames: ["qo", "ko"],
            source: CBv2FusedVerifyNorms.queryKeySource)
    }

    private static let kernels = Kernels()

    private static let constantsLock = NSLock()
    nonisolated(unsafe) private static var constants = [String: MLXArray]()

    /// Small float32 / int32 parameter arrays, built once per distinct value list.
    private static func constant(_ floats: [Float]) -> MLXArray {
        let key = floats.map { String($0.bitPattern, radix: 16) }.joined(separator: ",")
        return constantsLock.withLock {
            if let found = constants[key] { return found }
            let made = MLXArray(floats)
            constants[key] = made
            return made
        }
    }

    private static func constant(ints: [Int32]) -> MLXArray {
        let key = "i" + ints.map(String.init).joined(separator: ",")
        return constantsLock.withLock {
            if let found = constants[key] { return found }
            let made = MLXArray(ints)
            constants[key] = made
            return made
        }
    }

    // MARK: Residual and block norms

    /// `stream = residual + centeredRMSNorm(x)` (just the norm when `residual` is
    /// nil) and, when `next` is given, `normed = centeredRMSNorm(stream)` with
    /// the next norm's scale and eps. Both are `x`'s shape. `weight` holds the
    /// stored (centered) scale: the kernel applies `1 + weight` in float32.
    public static func addNorm(
        _ x: MLXArray, residual: MLXArray?, weight: MLXArray, eps: Float,
        next: (weight: MLXArray, eps: Float)?
    ) -> (stream: MLXArray, normed: MLXArray?) {
        let width = x.dim(-1)
        let rows = x.size / width
        precondition(supportsNorm(dtype: x.dtype, width: width), "unsupported fused norm")
        let flat = x.reshaped(rows, width)
        let flatResidual = residual?.reshaped(rows, width) ?? flat
        let scal = constant([Float(1.0 / Double(width)), eps, next?.eps ?? eps, 0])
        let outputs = kernels.norm(
            [flat, flatResidual, weight, next?.weight ?? weight, scal],
            template: [
                ("T", x.dtype), ("K", width), ("HAS_RES", residual != nil),
                ("HAS_NEXT", next != nil),
            ],
            grid: (1024, rows, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[rows, width], next != nil ? [rows, width] : [1]],
            outputDTypes: [x.dtype, x.dtype])
        return (
            outputs[0].reshaped(x.shape),
            next != nil ? outputs[1].reshaped(x.shape) : nil
        )
    }

    // MARK: q/k norm, query scale, rotary

    /// The attention input of a verify window from the q and k projections
    /// (`[batch * length, heads * headDimensions]` row-major, any leading shape):
    /// RMS norm per head, queries scaled in float32 and rounded back, then RoPE
    /// from `rope.offsets` (one int32 position per batch row, or one for all).
    /// With a rotary layer both results are `[batch, heads, length, dims]`
    /// contiguous, as MLX's RoPE returns them; without, they are the
    /// `[batch, length, heads, dims]` buffers viewed transposed, as the unfused
    /// code leaves them. Nil when the shapes or the offsets are not the ones the
    /// kernel reproduces.
    public static func queryKeyRope(
        queries: MLXArray, keys: MLXArray, batch: Int, length: Int, heads: Int, kvHeads: Int,
        headDimensions: Int, eps: Float, queryScale: Float,
        rope: (theta: Float, offsets: MLXArray)?
    ) -> (queries: MLXArray, keys: MLXArray)? {
        let dtype = queries.dtype
        guard supportsAttention(dtype: dtype, headDimensions: headDimensions),
            keys.dtype == dtype, queries.size == batch * length * heads * headDimensions,
            keys.size == batch * length * kvHeads * headDimensions
        else { return nil }
        var offsets = constant(ints: [0])
        var stride: Int32 = 0
        var logBase: Float = 0
        if let rope {
            guard rope.offsets.dtype == .int32,
                rope.offsets.size == 1 || rope.offsets.size == batch
            else { return nil }
            offsets = rope.offsets
            stride = rope.offsets.size == 1 ? 0 : 1
            logBase = log2(rope.theta)
        }
        let rows = batch * length
        let items = rows * (heads + kvHeads)
        let qShape =
            rope != nil
            ? [batch, heads, length, headDimensions] : [batch, length, heads, headDimensions]
        let kShape =
            rope != nil
            ? [batch, kvHeads, length, headDimensions] : [batch, length, kvHeads, headDimensions]
        let outputs = kernels.queryKey(
            [
                queries.reshaped(rows, heads * headDimensions),
                keys.reshaped(rows, kvHeads * headDimensions), offsets,
                constant([eps, logBase, 1, queryScale]), constant(ints: [stride]),
            ],
            template: [
                ("T", dtype), ("H", heads), ("KVH", kvHeads), ("D", headDimensions),
                ("L", length), ("ROPE", rope != nil),
            ],
            grid: (32 * items, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [qShape, kShape], outputDTypes: [dtype, dtype])
        if rope != nil { return (outputs[0], outputs[1]) }
        return (outputs[0].transposed(0, 2, 1, 3), outputs[1].transposed(0, 2, 1, 3))
    }
}
