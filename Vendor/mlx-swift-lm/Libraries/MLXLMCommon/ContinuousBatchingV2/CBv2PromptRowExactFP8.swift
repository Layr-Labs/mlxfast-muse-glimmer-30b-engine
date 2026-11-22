// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Row-exact FP8 products for many rows at once (a teacher-forced prompt).
///
/// Every output element is computed with the order of operations of
/// `ModelOptRowExact.product` (the windows schedule: activations cast to FP8
/// in a separate pass, then MLX's one-vector GEMV order): lane `g` of
/// `32 * BN` lanes sums `w * x` over `k = i * 128 * BN + 4 * g + tn` with `i`
/// ascending outside and `tn` in 0...3 inside, from zero; each simdgroup then
/// reduces its 32 lanes with the same `simd_shuffle_down` tree, and with
/// `BN == 8` simdgroup 0 adds simdgroups 1 to 7 in order. So each row's bytes
/// equal `rowExactForward` on any piece that holds it. Only the schedule
/// differs: one thread keeps `VR x NR` accumulators, so a decoded weight
/// serves `VR` rows and a loaded activation serves `NR` outputs, and one
/// dispatch covers every row.
public enum CBv2PromptRowExactFP8 {
    /// Rows per thread, outputs per thread, simdgroups per threadgroup (BN 1).
    nonisolated(unsafe) public static var geometry: (rows: Int, outputs: Int, simdgroups: Int) = {
        let parts = (ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_GEOMETRY"] ?? "")
            .split(separator: ",").compactMap { Int($0) }
        return parts.count == 3 ? (parts[0], parts[1], parts[2]) : (8, 4, 4)
    }()

    /// Decode the E4M3 weights once per call (`fp8DenseWeight`, the kernel's
    /// own `T(decode(code) * scale)`) instead of in the product's inner loop.
    nonisolated(unsafe) public static var predecodes =
        ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_PREDECODE"] != "0"

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_prompt_row_exact_fp8",
        inputNames: ["mat", "scale", "x"],
        outputNames: ["y"],
        source: """
            constexpr int blockN = 128 * BN;
            constexpr int n_iter = K / blockN;
            constexpr int leftover = K - blockN * n_iter;
            threadgroup float tgp[BN > 1 ? S * BN * VR * NR : 1];
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const int sgN = int(simd_gid) % BN;
            const int sgS = int(simd_gid) / BN;
            const int vb = NFAST ? int(threadgroup_position_in_grid.z) : int(threadgroup_position_in_grid.x);
            const int nb = NFAST ? int(threadgroup_position_in_grid.x) : int(threadgroup_position_in_grid.z);
            const int v0 = vb * VR;
            const int n0 = (nb * S + sgS) * NR;
            const float weight_scale = scale[0];
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();

            const device T* xr[VR];
            #pragma clang loop unroll(full)
            for (int v = 0; v < VR; v++) {
                xr[v] = x + size_t(min(v0 + v, M - 1)) * K;
            }
            const device MT* wr[NR];
            #pragma clang loop unroll(full)
            for (int n = 0; n < NR; n++) {
                wr[n] = mat + size_t(min(n0 + n, N - 1)) * K;
            }
            float acc[VR][NR];
            #pragma clang loop unroll(full)
            for (int v = 0; v < VR; v++) {
                #pragma clang loop unroll(full)
                for (int n = 0; n < NR; n++) {
                    acc[v][n] = 0;
                }
            }
            int bn = (32 * sgN + int(simd_lid)) * 4;
            for (int i = 0; i < n_iter; ++i) {
                float xv[VR][4];
                #pragma clang loop unroll(full)
                for (int v = 0; v < VR; v++) {
                    const float4 xs = cbv2_load4(xr[v] + bn);
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; tn++) {
                        xv[v][tn] = xs[tn];
                    }
                }
                #pragma clang loop unroll(full)
                for (int n = 0; n < NR; n++) {
                    T inter[4];
                    if constexpr (DENSE) {
                        cbv2_load_t4(wr[n] + bn, inter);
                    } else {
                        const uchar4 codes = *(const device packed_uchar4*)(wr[n] + bn);
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; tn++) {
                            const uint8_t code = codes[tn];
                            const float decoded =
                                (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                            inter[tn] = static_cast<T>(decoded * weight_scale);
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (int v = 0; v < VR; v++) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; tn++) {
                            acc[v][n] += inter[tn] * xv[v][tn];
                        }
                    }
                }
                bn += blockN;
            }
            if (leftover > 0) {
                float xv[VR][4];
                #pragma clang loop unroll(full)
                for (int v = 0; v < VR; v++) {
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; tn++) {
                        xv[v][tn] = bn + tn < K ? static_cast<float>(xr[v][bn + tn]) : 0.0f;
                    }
                }
                #pragma clang loop unroll(full)
                for (int n = 0; n < NR; n++) {
                    T inter[4];
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < 4; tn++) {
                        if (bn + tn < K) {
                            if constexpr (DENSE) {
                                inter[tn] = static_cast<T>(wr[n][bn + tn]);
                            } else {
                                const uint8_t code = wr[n][bn + tn];
                                const float decoded =
                                    (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                                inter[tn] = static_cast<T>(decoded * weight_scale);
                            }
                        } else {
                            inter[tn] = T(0);
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (int v = 0; v < VR; v++) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < 4; tn++) {
                            acc[v][n] += inter[tn] * xv[v][tn];
                        }
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (int v = 0; v < VR; v++) {
                #pragma clang loop unroll(full)
                for (int n = 0; n < NR; n++) {
                    for (ushort sn = 16; sn >= 1; sn >>= 1) {
                        acc[v][n] += simd_shuffle_down(acc[v][n], sn);
                    }
                }
            }
            if (BN > 1) {
                threadgroup float* base = tgp + sgS * (BN * VR * NR);
                if (simd_lid == 0) {
                    #pragma clang loop unroll(full)
                    for (int v = 0; v < VR; v++) {
                        #pragma clang loop unroll(full)
                        for (int n = 0; n < NR; n++) {
                            base[(sgN * VR + v) * NR + n] = acc[v][n];
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgN == 0 && simd_lid == 0) {
                    #pragma clang loop unroll(full)
                    for (int v = 0; v < VR; v++) {
                        #pragma clang loop unroll(full)
                        for (int n = 0; n < NR; n++) {
                            #pragma clang loop unroll(full)
                            for (int sgn = 1; sgn < BN; sgn++) {
                                acc[v][n] += base[(sgn * VR + v) * NR + n];
                            }
                        }
                    }
                }
            }
            if (sgN == 0 && simd_lid == 0) {
                #pragma clang loop unroll(full)
                for (int v = 0; v < VR; v++) {
                    if (v0 + v >= M) break;
                    #pragma clang loop unroll(full)
                    for (int n = 0; n < NR; n++) {
                        if (n0 + n < N) {
                            y[size_t(v0 + v) * N + n0 + n] = static_cast<T>(acc[v][n]);
                        }
                    }
                }
            }
            """,
        header: ModelOptActivationKernels.header + """

            inline float4 cbv2_load4(const device half* p) {
                return float4(half4(*(const device packed_half4*)p));
            }
            inline float4 cbv2_load4(const device bfloat* p) {
                return float4(as_type<vec<bfloat, 4>>(ushort4(*(const device packed_ushort4*)p)));
            }
            template <typename T>
            inline void cbv2_load_t4(const device T* p, thread T* out) {
                for (int i = 0; i < 4; i++) {
                    out[i] = p[i];
                }
            }
            inline void cbv2_load_t4(const device bfloat* p, thread bfloat* out) {
                const ushort4 u = ushort4(*(const device packed_ushort4*)p);
                const vec<bfloat, 4> b = as_type<vec<bfloat, 4>>(u);
                for (int i = 0; i < 4; i++) {
                    out[i] = b[i];
                }
            }
            inline float4 cbv2_load4(const device float* p) {
                return float4(*(const device packed_float4*)p);
            }
            """)


    /// The matrix schedule: one simdgroup holds an 8-row x 8-output tile for
    /// each of its 32 lanes as a simdgroup-matrix accumulator, and feeds each
    /// lane's chain to `simdgroup_multiply_accumulate` eight terms at a time.
    /// On the GPU generations where that multiply-accumulate equals the
    /// ascending chain of fused multiply-adds (`ModelOptWeightOnlyNVFP4RowExact
    /// .scalarKernelMeasured`), and with products of two 16-bit floats exact
    /// in float32, every lane's partial equals the GEMV's. The lanes are then
    /// summed in the GEMV's `simd_shuffle_down` tree, element by element.
    private static let matrixKernel = MLXFast.metalKernel(
        name: "cbv2_prompt_row_exact_fp8_matrix",
        inputNames: ["mat", "x"],
        outputNames: ["y"],
        source: """
            constexpr int blockN = 128 * BN;
            constexpr int steps = (K + blockN - 1) / blockN;
            constexpr int chain = steps * 4;
            constexpr int chunks = (chain + 7) / 8;
            threadgroup float tgp[BN > 1 ? BN * 64 * TV * TN : 1];
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lid = thread_index_in_simdgroup;
            const short qid = lid / 4;
            const short fm = (qid & 4) + ((lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (lid % 2) * 2;
            const int sgN = BN > 1 ? int(sg) : 0;
            const int v0 = int(threadgroup_position_in_grid.x) * 8 * TV;
            const int n0 = (int(threadgroup_position_in_grid.z) * (BN > 1 ? 1 : S) + (BN > 1 ? 0 : int(sg))) * 8 * TN;
            const device T* xa[TV];
            #pragma clang loop unroll(full)
            for (int tv = 0; tv < TV; tv++) {
                xa[tv] = x + size_t(min(v0 + 8 * tv + fm, M - 1)) * K;
            }
            const device MT* wb[TN][2];
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; tn++) {
                wb[tn][0] = mat + size_t(min(n0 + 8 * tn + fn, N - 1)) * K;
                wb[tn][1] = mat + size_t(min(n0 + 8 * tn + fn + 1, N - 1)) * K;
            }
            // Lanes run in bit-reversed order, so the GEMV's shuffle tree
            // (lane l takes lane l + s for s = 16, 8, 4, 2, 1) folds as a
            // binary counter: level j holds a finished subtree of 2^j lanes.
            float2 stack[TV][TN][6];
            #pragma clang loop unroll(full)
            for (int t = 0; t < 32; t++) {
                const int l = ((t & 1) << 4) | ((t & 2) << 2) | (t & 4) | ((t & 8) >> 2) | ((t & 16) >> 4);
                const int g4 = (32 * sgN + l) * 4;
                float2 C[TV][TN];
                #pragma clang loop unroll(full)
                for (int tv = 0; tv < TV; tv++) {
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < TN; tn++) {
                        C[tv][tn] = float2(0);
                    }
                }
                #pragma clang loop unroll(full)
                for (int c = 0; c < chunks; c++) {
                    const int ja = 8 * c + fn;
                    const int jb = 8 * c + fm;
                    const int ka = (ja / 4) * blockN + (ja % 4) + g4;
                    const int kb = (jb / 4) * blockN + (jb % 4) + g4;
                    const bool av = ja < chain && ka < K;
                    const bool bv = jb < chain && kb < K;
                    float2 A[TV];
                    #pragma clang loop unroll(full)
                    for (int tv = 0; tv < TV; tv++) {
                        A[tv] = av ? cbv2_load2(xa[tv] + ka) : float2(0);
                    }
                    float2 B[TN];
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < TN; tn++) {
                        B[tn] = bv ? float2(static_cast<float>(wb[tn][0][kb]), static_cast<float>(wb[tn][1][kb])) : float2(0);
                    }
                    #pragma clang loop unroll(full)
                    for (int tv = 0; tv < TV; tv++) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < TN; tn++) {
                            cbv2_mma(C[tv][tn], A[tv], B[tn]);
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (int tv = 0; tv < TV; tv++) {
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < TN; tn++) {
                        float2 cur = C[tv][tn];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 5; j++) {
                            if (((t >> j) & 1) == 0) break;
                            cur = stack[tv][tn][j] + cur;
                        }
                        int level = 0;
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 5; j++) {
                            if (((t >> j) & 1) == 0) break;
                            level = j + 1;
                        }
                        stack[tv][tn][level] = cur;
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (int tv = 0; tv < TV; tv++) {
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < TN; tn++) {
                    float2 r = stack[tv][tn][5];
                    const int slot = (tv * TN + tn) * 64 + fm * 8 + fn;
                    if (BN > 1) {
                        tgp[sgN * 64 * TV * TN + slot] = r.x;
                        tgp[sgN * 64 * TV * TN + slot + 1] = r.y;
                    }
                    stack[tv][tn][5] = r;
                }
            }
            if (BN > 1) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgN != 0) return;
            }
            #pragma clang loop unroll(full)
            for (int tv = 0; tv < TV; tv++) {
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < TN; tn++) {
                    float2 r = stack[tv][tn][5];
                    const int slot = (tv * TN + tn) * 64 + fm * 8 + fn;
                    #pragma clang loop unroll(full)
                    for (int sgn = 1; sgn < BN; sgn++) {
                        r.x += tgp[sgn * 64 * TV * TN + slot];
                        r.y += tgp[sgn * 64 * TV * TN + slot + 1];
                    }
                    const int v = v0 + 8 * tv + fm;
                    const int n = n0 + 8 * tn + fn;
                    if (v < M) {
                        if (n < N) y[size_t(v) * N + n] = static_cast<T>(r.x);
                        if (n + 1 < N) y[size_t(v) * N + n + 1] = static_cast<T>(r.y);
                    }
                }
            }
            """,
        header: """
            #include <metal_simdgroup_matrix>
            typedef metal::simdgroup_matrix<float, 8, 8> cbv2_matrix;
            inline float2 cbv2_load2(const device bfloat* p) {
                return float2(as_type<vec<bfloat, 2>>(ushort2(*(const device packed_ushort2*)p)));
            }
            inline float2 cbv2_load2(const device half* p) {
                return float2(*(const device packed_half2*)p);
            }
            inline float2 cbv2_load2(const device float* p) {
                return float2(*(const device packed_float2*)p);
            }
            // C += A x B; each thread holds two neighboring columns of a row.
            inline void cbv2_mma(thread float2& C, float2 A, float2 B) {
                cbv2_matrix D_mat;
                cbv2_matrix A_mat;
                cbv2_matrix B_mat;
                cbv2_matrix C_mat;
                reinterpret_cast<thread float2&>(A_mat.thread_elements()) = A;
                reinterpret_cast<thread float2&>(B_mat.thread_elements()) = B;
                reinterpret_cast<thread float2&>(C_mat.thread_elements()) = C;
                simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
                C = reinterpret_cast<thread float2&>(D_mat.thread_elements());
            }
            """)

    /// The matrix schedule runs where the multiply-accumulate was measured
    /// equal to the fused multiply-add chain.
    nonisolated(unsafe) public static var usesMatrix =
        ModelOptWeightOnlyNVFP4RowExact.scalarKernelMeasured
        && ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_MATRIX"] == "1"
    nonisolated(unsafe) public static var matrixSimdgroups: Int =
        Int(ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_MSG"] ?? "") ?? 4
    nonisolated(unsafe) public static var matrixTiles: (Int, Int) = {
        let parts = (ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_TILES"] ?? "")
            .split(separator: ",").compactMap { Int($0) }
        return parts.count == 2 ? (parts[0], parts[1]) : (2, 2)
    }()

    static func matrixProduct(_ dense: MLXArray, vectors: MLXArray, splitsK: Bool) -> MLXArray {
        let (outputs, inputs) = (dense.dim(0), dense.dim(1))
        let rows = vectors.dim(0)
        let bn = splitsK ? 8 : 1
        let s = splitsK ? 1 : matrixSimdgroups
        let (tv, tn) = matrixTiles
        let nTiles = (outputs + 8 * tn * s - 1) / (8 * tn * s)
        return matrixKernel(
            [dense, vectors],
            template: [
                ("T", vectors.dtype), ("MT", dense.dtype), ("K", inputs), ("N", outputs),
                ("M", rows), ("BN", bn), ("S", s), ("TV", tv), ("TN", tn),
            ],
            grid: ((rows + 8 * tv - 1) / (8 * tv) * 32, splitsK ? 8 : s, nTiles),
            threadGroup: (32, splitsK ? 8 : s, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [vectors.dtype])[0]
    }

    /// The permuted matrix schedule: lane `g`'s chain (its terms `j = 4i + tn`
    /// at `k = i * 128 * BN + 4g + tn`) is laid out contiguously per call, as
    /// `xp [G, M, TP]` and `wp [G, TP, N]` with zeros past `K`, so each eight
    /// terms of a chain feed one `simdgroup_multiply_accumulate` (an ascending
    /// fused multiply-add chain where `scalarKernelMeasured`) from contiguous
    /// pairs. Chains start at +0 and padded terms are 0 x 0, so they add
    /// nothing. Lanes then fold in the GEMV's shuffle tree (bit-reversed
    /// order, binary counter), and simdgroup `b` (lanes `32b...32b + 31`)
    /// joins simdgroup 0's sum in order.
    nonisolated(unsafe) public static var usesPermutedMatrix =
        ModelOptWeightOnlyNVFP4RowExact.scalarKernelMeasured
        && ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_PMMA"] != "0"
    /// Narrow products (k/v, 256 outputs) measured slower on this schedule.
    nonisolated(unsafe) public static var permutedMinimumOutputs = 1024
    nonisolated(unsafe) public static var permutedTiles: (Int, Int) = {
        let parts = (ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_PTILES"] ?? "")
            .split(separator: ",").compactMap { Int($0) }
        return parts.count == 2 ? (parts[0], parts[1]) : (4, 4)
    }()

    private static let permutedKernel = MLXFast.metalKernel(
        name: "cbv2_prompt_row_exact_fp8_permuted",
        inputNames: ["wp", "xp"],
        outputNames: ["y"],
        source: """
            constexpr int blockN = 128 * BN;
            constexpr int CH = TP / 8;
            threadgroup float tgp[BN > 1 ? (BN - 1) * 64 * TV * TN : 1];
            const int sg = int(simdgroup_index_in_threadgroup);
            const uint lid = thread_index_in_simdgroup;
            const short qid = lid / 4;
            const short fm = (qid & 4) + ((lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (lid % 2) * 2;
            const int v0 = int(threadgroup_position_in_grid.x) * 8 * TV;
            const int n0 = int(threadgroup_position_in_grid.z) * 8 * TN;
            int xo[TV];
            #pragma clang loop unroll(full)
            for (int tv = 0; tv < TV; tv++) {
                xo[tv] = min(v0 + 8 * tv + fm, M - 1) * TP + fn;
            }
            int wo[TN];
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; tn++) {
                wo[tn] = fm * N + min(n0 + 8 * tn + fn, N - 2);
            }
            float2 stack[TV][TN][6];
            #pragma clang loop unroll(full)
            for (int t = 0; t < 32; t++) {
                const int l = ((t & 1) << 4) | ((t & 2) << 2) | (t & 4) | ((t & 8) >> 2) | ((t & 16) >> 4);
                const int g = 32 * sg + l;
                const device T* xg = xp + size_t(g) * (M * TP);
                const device T* wg = wp + size_t(g) * (TP * N);
                float2 C[TV][TN];
                #pragma clang loop unroll(full)
                for (int tv = 0; tv < TV; tv++) {
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < TN; tn++) {
                        C[tv][tn] = float2(0);
                    }
                }
                for (int c = 0; c < CH; c++) {
                    if (8 * c >= TERMS || 2 * c * blockN + 4 * g >= K) break;
                    float2 A[TV];
                    #pragma clang loop unroll(full)
                    for (int tv = 0; tv < TV; tv++) {
                        A[tv] = cbv2p_load2(xg + xo[tv] + 8 * c);
                    }
                    float2 B[TN];
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < TN; tn++) {
                        B[tn] = cbv2p_load2(wg + wo[tn] + 8 * c * N);
                    }
                    #pragma clang loop unroll(full)
                    for (int tv = 0; tv < TV; tv++) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < TN; tn++) {
                            cbv2p_mma(C[tv][tn], A[tv], B[tn]);
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (int tv = 0; tv < TV; tv++) {
                    #pragma clang loop unroll(full)
                    for (int tn = 0; tn < TN; tn++) {
                        float2 cur = C[tv][tn];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 5; j++) {
                            if (((t >> j) & 1) == 0) break;
                            cur = stack[tv][tn][j] + cur;
                        }
                        int level = 0;
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 5; j++) {
                            if (((t >> j) & 1) == 0) break;
                            level = j + 1;
                        }
                        stack[tv][tn][level] = cur;
                    }
                }
            }
            if (BN > 1) {
                if (sg > 0) {
                    #pragma clang loop unroll(full)
                    for (int tv = 0; tv < TV; tv++) {
                        #pragma clang loop unroll(full)
                        for (int tn = 0; tn < TN; tn++) {
                            const int slot = (sg - 1) * 64 * TV * TN + (tv * TN + tn) * 64 + fm * 8 + fn;
                            tgp[slot] = stack[tv][tn][5].x;
                            tgp[slot + 1] = stack[tv][tn][5].y;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg != 0) return;
            }
            #pragma clang loop unroll(full)
            for (int tv = 0; tv < TV; tv++) {
                #pragma clang loop unroll(full)
                for (int tn = 0; tn < TN; tn++) {
                    float2 r = stack[tv][tn][5];
                    #pragma clang loop unroll(full)
                    for (int b = 1; b < BN; b++) {
                        const int slot = (b - 1) * 64 * TV * TN + (tv * TN + tn) * 64 + fm * 8 + fn;
                        r.x += tgp[slot];
                        r.y += tgp[slot + 1];
                    }
                    const int v = v0 + 8 * tv + fm;
                    const int n = n0 + 8 * tn + fn;
                    if (v < M && n < N) {
                        y[size_t(v) * N + n] = static_cast<T>(r.x);
                        y[size_t(v) * N + n + 1] = static_cast<T>(r.y);
                    }
                }
            }
            """,
        header: """
            #include <metal_simdgroup_matrix>
            typedef metal::simdgroup_matrix<float, 8, 8> cbv2p_matrix;
            inline float2 cbv2p_load2(const device bfloat* p) {
                return float2(as_type<vec<bfloat, 2>>(*(const device uint*)p));
            }
            inline float2 cbv2p_load2(const device half* p) {
                return float2(*(const device half2*)p);
            }
            inline float2 cbv2p_load2(const device float* p) {
                return *(const device float2*)p;
            }
            inline void cbv2p_mma(thread float2& C, float2 A, float2 B) {
                cbv2p_matrix D_mat;
                cbv2p_matrix A_mat;
                cbv2p_matrix B_mat;
                cbv2p_matrix C_mat;
                reinterpret_cast<thread float2&>(A_mat.thread_elements()) = A;
                reinterpret_cast<thread float2&>(B_mat.thread_elements()) = B;
                reinterpret_cast<thread float2&>(C_mat.thread_elements()) = C;
                simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
                C = reinterpret_cast<thread float2&>(D_mat.thread_elements());
            }
            """)

    /// `wp [G, TP, N]` from E4M3 rows `[N, K]`: term `j = 4i + tn` of lane `g`
    /// is `T(decode(code) * scale)` at `k = i * 128 * BN + 4g + tn` (as
    /// `fp8DenseWeight`), zero past `K`. Per call only, never kept.
    private static let permuteWeightKernel = MLXFast.metalKernel(
        name: "cbv2_prompt_fp8_permute_weight",
        inputNames: ["mat", "scale"],
        outputNames: ["wp"],
        source: """
            constexpr int blockN = 128 * BN;
            threadgroup uchar tile[32][129];
            const int lid = int(thread_index_in_threadgroup);
            const int n0 = int(threadgroup_position_in_grid.x) * 32;
            const int i = int(threadgroup_position_in_grid.y);
            const int gb = int(threadgroup_position_in_grid.z);
            const int k0 = i * blockN + gb * 128;
            #pragma clang loop unroll(full)
            for (int r = 0; r < 4; r++) {
                const int e = (lid + 256 * r) * 4;
                const int nn = e / 128;
                const int kk = e % 128;
                uchar4 c = uchar4(0);
                const int k = k0 + kk;
                if (n0 + nn < N && k < K) {
                    c = *(const device packed_uchar4*)(mat + size_t(n0 + nn) * K + k);
                }
                tile[nn][kk] = c.x;
                tile[nn][kk + 1] = c.y;
                tile[nn][kk + 2] = c.z;
                tile[nn][kk + 3] = c.w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float weight_scale = scale[0];
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            const int nn = lid % 32;
            #pragma clang loop unroll(full)
            for (int r = 0; r < 16; r++) {
                const int q = lid / 32 + 8 * r;
                const int gl = q / 4;
                const int tn = q % 4;
                const int k = k0 + 4 * gl + tn;
                T value = T(0);
                if (k < K) {
                    const uint8_t code = tile[nn][4 * gl + tn];
                    const float decoded =
                        (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                    value = static_cast<T>(decoded * weight_scale);
                }
                if (n0 + nn < N) {
                    wp[(size_t(32 * gb + gl) * TP + 4 * i + tn) * N + n0 + nn] = value;
                }
            }
            """,
        header: ModelOptActivationKernels.header)

    /// `xp [G, M, TP]` from `[M, K]` activations, zero past `K`.
    private static let permuteVectorsKernel = MLXFast.metalKernel(
        name: "cbv2_prompt_fp8_permute_vectors",
        inputNames: ["x"],
        outputNames: ["xp"],
        source: """
            constexpr int blockN = 128 * BN;
            const int j = int(thread_position_in_grid.x);
            const int m = int(thread_position_in_grid.y);
            const int g = int(thread_position_in_grid.z);
            const int k = (j / 4) * blockN + 4 * g + (j % 4);
            xp[(size_t(g) * M + m) * TP + j] = (j < TERMS && k < K) ? x[size_t(m) * K + k] : T(0);
            """)

    /// The permuted schedule's product, or nil where it does not apply.
    static func permutedProduct(
        _ mat: MLXArray, scale: MLXArray, vectors: MLXArray, splitsK: Bool
    ) -> MLXArray? {
        let (outputs, inputs) = (mat.dim(0), mat.dim(1))
        let rows = vectors.dim(0)
        guard mat.dtype == .uint8, outputs % 32 == 0, inputs % 4 == 0, rows > 0 else {
            return nil
        }
        let bn = splitsK ? 8 : 1
        let lanes = 32 * bn
        let blockN = 128 * bn
        let steps = (inputs + blockN - 1) / blockN
        let terms = steps * 4
        let tp = (terms + 7) / 8 * 8
        let wp = permuteWeightKernel(
            [mat, scale],
            template: [("T", vectors.dtype), ("N", outputs), ("K", inputs), ("BN", bn), ("TP", tp)],
            grid: (outputs / 32 * 256, tp / 4, bn), threadGroup: (256, 1, 1),
            outputShapes: [[lanes, tp, outputs]], outputDTypes: [vectors.dtype])[0]
        let xp = permuteVectorsKernel(
            [vectors],
            template: [
                ("T", vectors.dtype), ("M", rows), ("K", inputs), ("BN", bn), ("TP", tp),
                ("TERMS", terms),
            ],
            grid: (tp, rows, lanes), threadGroup: (tp, 1, 1),
            outputShapes: [[lanes, rows, tp]], outputDTypes: [vectors.dtype])[0]
        let (tv, tn) = permutedTiles
        return permutedKernel(
            [wp, xp],
            template: [
                ("T", vectors.dtype), ("K", inputs), ("N", outputs), ("M", rows), ("BN", bn),
                ("TP", tp), ("TERMS", terms), ("TV", tv), ("TN", tn),
            ],
            grid: ((rows + 8 * tv - 1) / (8 * tv) * 32, bn, (outputs + 8 * tn - 1) / (8 * tn)),
            threadGroup: (32, bn, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [vectors.dtype])[0]
    }

    /// `[M, N]` products of FP8-cast `vectors` (`[M, K]`, dtype T) with E4M3
    /// rows `mat` (`[N, K]`, uint8) times `scale` (`[1]` float32); `splitsK`
    /// is the one-row call's choice (8 simdgroups across K).
    public static func product(
        _ mat: MLXArray, scale: MLXArray, vectors: MLXArray, splitsK: Bool
    ) -> MLXArray {
        if usesPermutedMatrix, splitsK, mat.dim(0) >= permutedMinimumOutputs,
            let y = permutedProduct(mat, scale: scale, vectors: vectors, splitsK: splitsK)
        {
            return y
        }
        if usesMatrix {
            return matrixProduct(
                ModelOptActivationKernels.fp8DenseWeight(mat, scale: scale, dtype: vectors.dtype),
                vectors: vectors, splitsK: splitsK)
        }
        let dense = predecodes
        let mat = dense
            ? ModelOptActivationKernels.fp8DenseWeight(mat, scale: scale, dtype: vectors.dtype) : mat
        let (outputs, inputs) = (mat.dim(0), mat.dim(1))
        let rows = vectors.dim(0)
        let bn = splitsK ? 8 : 1
        let (vr, nr, s0) = geometry
        let s = splitsK ? 1 : s0
        let vBlocks = (rows + vr - 1) / vr
        let nFast = ProcessInfo.processInfo.environment["CBV2_PROMPT_FP8_NFAST"] != "0"
        let nTiles = (outputs + nr * s - 1) / (nr * s)
        return kernel(
            [mat, scale, vectors],
            template: [
                ("T", vectors.dtype), ("MT", mat.dtype), ("DENSE", dense), ("K", inputs), ("N", outputs), ("M", rows),
                ("BN", bn), ("VR", vr), ("NR", nr), ("S", s), ("NFAST", nFast),
            ],
            grid: nFast ? (nTiles * 32, bn, vBlocks * s) : (vBlocks * 32, bn, nTiles * s),
            threadGroup: (32, bn, s),
            outputShapes: [[rows, outputs]], outputDTypes: [vectors.dtype])[0]
    }
}

extension ModelOptCalibratedLinear {
    /// The row-exact FP8 product of every row of `input` at once, with each
    /// row's bytes equal to `rowExactForward` on any piece holding it; nil
    /// when this projection has no such product.
    public func promptRowExactForward(_ input: MLXArray) -> MLXArray? {
        guard format == .fp8, execution == .graph, inputDimensions > 64,
            input.dtype == weightDType, input.ndim > 0, input.dim(-1) == inputDimensions,
            let constants = graphConstants,
            case .fp8(let multipliers, let overflowBound) = constants.storage,
            weight.dtype == .uint8, weight.shape == [outputDimensions, inputDimensions]
        else { return nil }
        let rows = input.size / inputDimensions
        guard rows > 1 else { return nil }
        let vectors = ModelOptActivationKernels.fp8(
            input.reshaped(rows, inputDimensions), multipliers: multipliers,
            overflowBound: overflowBound)
        var runs: [(rows: Range<Int>, splitsK: Bool)] = []
        for tile in fp8EagerTiles(rows: 1) {
            let splitsK = inputDimensions >= 16 * tile.count
            if let last = runs.last, last.splitsK == splitsK {
                runs[runs.count - 1].rows = last.rows.lowerBound ..< tile.upperBound
            } else {
                runs.append((tile, splitsK))
            }
        }
        let pieces = runs.map { run in
            CBv2PromptRowExactFP8.product(
                weight[run.rows, 0...], scale: constants.outputScale, vectors: vectors,
                splitsK: run.splitsK)
        }
        let result = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
        return result.reshaped(Array(input.shape.dropLast()) + [outputDimensions])
    }
}
