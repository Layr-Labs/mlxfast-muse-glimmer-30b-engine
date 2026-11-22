// Copyright © 2026 Eigen Labs.
// The kernel source below is ModelOptWeightOnlyNVFP4RowExact's matrix kernel,
// verbatim (MLX `fp_qmv_mma` layout, MIT, Copyright Apple Inc.); only its name
// and the row counts it is dispatched at differ.

import Foundation
import MLX

/// The weight-only NVFP4 row-exact matrix product at 17 to 32 rows per call.
///
/// `ModelOptWeightOnlyNVFP4RowExact` dispatches its matrix kernel at 1 to 16
/// rows, so a 512-row teacher-forced prompt reads every NVFP4 weight 32 times.
/// The kernel is generic in the row count `V`: it keeps `(V + 7) / 8` row
/// fragments per thread and each output element follows ONE order of
/// operations, the same at every `V` (partition `s` of the inputs, ascending
/// blocks, `fma` chains, the fixed partial sum order). A row of a 32-row call
/// is therefore bit-identical to the same row of a 16-row call, or alone. At
/// `V = 32` the cross-simdgroup partials take 28 KiB of threadgroup memory,
/// under the 32 KiB limit; `V = 64` would not fit. Reading each weight once
/// per 32 rows halves the timed prefill's NVFP4 weight traffic.
public enum CBv2WideRowExactNVFP4 {
    public static let maximumRows = 32

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_wide_nvfp4_row_exact_matrix",
        inputNames: ["w", "scales", "x", "tensor_scale"],
        outputNames: ["y"],
        source: """
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr int BN = 32;
            constexpr short TN = BN / 8;
            constexpr short TM = (V + 7) / 8;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            constexpr bool whole_blocks = (K % 32) == 0;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            const int y_col = threadgroup_position_in_grid.y * BN;

            // Rows past V repeat row V - 1; their columns are never stored.
            const device T* xr[TM][2];
            float2 C[TN][TM];
            MO_WO_UNROLL
            for (short mt = 0; mt < TM; mt++) {
                MO_WO_UNROLL
                for (short j = 0; j < 2; j++) {
                    xr[mt][j] = x + size_t(min(8 * mt + fn + j, V - 1)) * K + 4 * fm;
                }
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    C[t][mt] = float2(0);
                }
            }
            const device uint* wl[TN];
            const device uint8_t* sl[TN];
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                const int row = min(y_col + fm + 8 * t, N - 1);
                wl[t] = (const device uint*)(
                    (const device uint8_t*)w + size_t(row) * row_bytes + 2 * fn);
                sl[t] = scales + size_t(row) * row_scales + fn / 4;
            }

            for (int k = simd_gid * BK; k < K; k += WK * BK) {
                MO_WO_UNROLL
                for (short block = 0; block < 4; block++) {
                    const int b = k + 32 * block;
                    // The last block may hold 16 inputs: columns 0 to 3. A
                    // missing column is zero on both sides, which adds nothing.
                    const bool a_valid = whole_blocks ? b < K : b + 4 * fn < K;
                    const bool b_valid = whole_blocks ? b < K : b + 4 * fm < K;
                    if (whole_blocks && !a_valid) {
                        break;
                    }
                    float4 xv[TM][2];
                    MO_WO_UNROLL
                    for (short mt = 0; mt < TM; mt++) {
                        MO_WO_UNROLL
                        for (short j = 0; j < 2; j++) {
                            xv[mt][j] = b_valid ? mo_wo_load4(xr[mt][j] + b) : float4(0);
                        }
                    }
                    uint q[TN];
                    half s[TN];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        q[t] = a_valid ? wl[t][b / 8] : 0u;
                        s[t] = a_valid ? mo_wo_e4m3(sl[t][b / 16]) : 0.0h;
                    }
                    MO_WO_UNROLL
                    for (short i = 0; i < 4; i++) {
                        float2 bf[TM];
                        MO_WO_UNROLL
                        for (short mt = 0; mt < TM; mt++) {
                            bf[mt] = float2(xv[mt][0][i], xv[mt][1][i]);
                        }
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            const float2 a = mo_wo_pair(q[t], i, s[t]);
                            MO_WO_UNROLL
                            for (short mt = 0; mt < TM; mt++) {
                                mo_wo_mma(C[t][mt], a, bf[mt]);
                            }
                        }
                    }
                }
            }

            constexpr int tile_elems = TN * TM * 2;
            threadgroup float partials[(WK - 1) * tile_elems * 32];
            if (simd_gid > 0) {
                threadgroup float* p = partials + (simd_gid - 1) * tile_elems * 32 + simd_lid;
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    MO_WO_UNROLL
                    for (short mt = 0; mt < TM; mt++) {
                        MO_WO_UNROLL
                        for (short j = 0; j < 2; j++) {
                            p[((t * TM + mt) * 2 + j) * 32] = C[t][mt][j];
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_gid == 0) {
                for (int s = 0; s < WK - 1; s++) {
                    const threadgroup float* p = partials + s * tile_elems * 32 + simd_lid;
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        MO_WO_UNROLL
                        for (short mt = 0; mt < TM; mt++) {
                            MO_WO_UNROLL
                            for (short j = 0; j < 2; j++) {
                                C[t][mt][j] += p[((t * TM + mt) * 2 + j) * 32];
                            }
                        }
                    }
                }
                MO_WO_UNROLL
                for (short mt = 0; mt < TM; mt++) {
                    MO_WO_UNROLL
                    for (short j = 0; j < 2; j++) {
                        if (8 * mt + fn + j < V) {
                            device T* yr = y + size_t(8 * mt + fn + j) * N;
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                const int out = y_col + fm + 8 * t;
                                if (out < N) {
                                    yr[out] = mo_wo_result<T>(
                                        C[t][mt][j], SCALED, tensor_scale[0]);
                                }
                            }
                        }
                    }
                }
            }
            """,
        header: ModelOptWeightOnlyNVFP4RowExact.header)

    /// `x` `[..., K]` with 1 to 32 rows in all; the same layout and arguments
    /// as `ModelOptWeightOnlyNVFP4RowExact.product`. The matrix kernel serves
    /// every row count, as it does there wherever the scalar kernel is off
    /// (every generation but 15); callers keep that product for those.
    public static func product(
        _ x: MLXArray, weight: MLXArray, blockScales: MLXArray, tensorScale: MLXArray? = nil
    ) -> MLXArray {
        let inputs = x.dim(-1)
        let rows = x.size / max(1, inputs)
        let outputs = weight.dim(0)
        precondition(
            (1 ... maximumRows).contains(rows) && inputs > 0 && inputs % 16 == 0
                && [.bfloat16, .float16, .float32].contains(x.dtype)
                && weight.dtype == .uint32 && weight.shape == [outputs, inputs / 8]
                && blockScales.dtype == .uint8 && blockScales.shape == [outputs, inputs / 16]
                && weight.size >= 8 && blockScales.size >= 8
                && (tensorScale.map { $0.dtype == .float32 && $0.size == 1 } ?? true),
            "wide weight-only NVFP4 row-exact product needs 1...32 rows")
        let tiles = (outputs + 31) / 32
        let template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("V", rows), ("K", inputs), ("N", outputs),
            ("SCALED", tensorScale != nil),
        ]
        let result = kernel(
            [weight, blockScales, x.reshaped(rows, inputs), tensorScale?.reshaped(1) ?? unit],
            template: template,
            grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    nonisolated(unsafe) private static let unit = MLXArray([Float(1)])
}
