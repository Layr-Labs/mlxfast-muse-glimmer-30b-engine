// Copyright © 2026 Eigen Labs.
// The simdgroup-matrix fragment layout and the paired E2M1 decode are those of
// `ModelOptWeightOnlyNVFP4RowExact`'s matrix kernel (after MLX `fp_qmv_mma`,
// MIT, Copyright Apple Inc.).

import Foundation
import MLX

/// Row-exact weight-only NVFP4 products for many rows at once (a
/// teacher-forced prompt), in one dispatch.
///
/// Every output element runs the order of operations of
/// `ModelOptWeightOnlyNVFP4RowExact.product` (its matrix kernel): partition
/// `s` of 8 takes the 128-input chunks `q % 8 == s` ascending; inside a
/// chunk, four 32-input blocks ascending, and in a block starting at `b` the
/// inputs `b + 4c + i` with `i` outside and `c` inside, eight at a time
/// through `simdgroup_multiply_accumulate` (an ascending fused multiply-add
/// chain on the generations in `scalarKernelGenerations`); then
/// `((acc0 + acc1) + ... + acc7)`, the dtype rounding and the tensor scale.
/// That order does not depend on the rows beside an element, so each row's
/// bytes equal the 16-row pieces' on any piece that holds it. Only the
/// schedule differs: a threadgroup takes `8 * TM` rows and `8 * TN` outputs,
/// and one dispatch covers every row, with no pieces to join. Weights are
/// decoded in the inner loop, as in the pieces, never stored decoded.
public enum CBv2PromptRowExactNVFP4 {
    /// 8-row tiles and 8-output tiles per threadgroup.
    nonisolated(unsafe) public static var tiles: (rows: Int, outputs: Int) = {
        let parts = (ProcessInfo.processInfo.environment["CBV2_PROMPT_NVFP4_TILES"] ?? "")
            .split(separator: ",").compactMap { Int($0) }
        return parts.count == 2 ? (parts[0], parts[1]) : (4, 2)
    }()
    /// Row blocks are the fast grid axis: the threadgroups that share a
    /// weight tile run together, so it is read from DRAM about once.
    nonisolated(unsafe) public static var rowsFast =
        ProcessInfo.processInfo.environment["CBV2_PROMPT_NVFP4_RFAST"] != "0"

    /// Local tuning only: CBV2_PROMPT_NVFP4=0 keeps the 16-row pieces.
    nonisolated(unsafe) public static var enabled =
        ProcessInfo.processInfo.environment["CBV2_PROMPT_NVFP4"] != "0"

    /// Local tuning only: rows per dispatch (0, the default: every row).
    nonisolated(unsafe) public static var chunkRows =
        Int(ProcessInfo.processInfo.environment["CBV2_PROMPT_NVFP4_CHUNK"] ?? "") ?? 0

    nonisolated(unsafe) private static let unit = MLXArray([Float(1)])

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_prompt_row_exact_nvfp4",
        inputNames: ["w", "scales", "x", "tensor_scale"],
        outputNames: ["y"],
        source: """
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr int BN = 8 * TN;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            constexpr bool whole_blocks = (K % 32) == 0;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            const int row_block = RFAST ? int(threadgroup_position_in_grid.y)
                                        : int(threadgroup_position_in_grid.z);
            const int tile = RFAST ? int(threadgroup_position_in_grid.z)
                                   : int(threadgroup_position_in_grid.y);
            const int y_col = tile * BN;
            const int r0 = row_block * 8 * TM;

            // Rows past M repeat row M - 1; their columns are never stored.
            const device T* xr[TM][2];
            float2 C[TN][TM];
            MO_WO_UNROLL
            for (short mt = 0; mt < TM; mt++) {
                MO_WO_UNROLL
                for (short j = 0; j < 2; j++) {
                    xr[mt][j] = x + size_t(min(r0 + 8 * mt + fn + j, M - 1)) * K + 4 * fm;
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
                        const int r = r0 + 8 * mt + fn + j;
                        if (r < M) {
                            device T* yr = y + size_t(r) * N;
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

    /// Whether `product` serves this call: the matrix kernel's arithmetic is
    /// the pieces' only where the 16-row pieces run the matrix kernel too
    /// (no scalar-kernel rows), on a whole number of 32-input blocks.
    public static func supports(rows: Int, inputs: Int, outputs: Int, dtype: DType) -> Bool {
        enabled && rows > ModelOptWeightOnlyNVFP4RowExact.maximumRows
            && ModelOptWeightOnlyNVFP4RowExact.scalarRows == 0
            && ModelOptWeightOnlyNVFP4RowExact.supports(rows: 1, inputs: inputs, dtype: dtype)
            && inputs % 32 == 0 && outputs >= 8
    }

    /// `[M, N]`: `ModelOptWeightOnlyNVFP4RowExact.product` of every row of
    /// `x` (`[..., K]`), bit for bit, in one dispatch.
    public static func product(
        _ x: MLXArray, weight: MLXArray, blockScales: MLXArray, tensorScale: MLXArray? = nil
    ) -> MLXArray {
        let inputs = x.dim(-1)
        let rows = x.size / max(1, inputs)
        let outputs = weight.dim(0)
        precondition(
            supports(rows: rows, inputs: inputs, outputs: outputs, dtype: x.dtype)
                && weight.dtype == .uint32 && weight.shape == [outputs, inputs / 8]
                && blockScales.dtype == .uint8 && blockScales.shape == [outputs, inputs / 16]
                && (tensorScale.map { $0.dtype == .float32 && $0.size == 1 } ?? true),
            "prompt NVFP4 row-exact product: unsupported call")
        if chunkRows > ModelOptWeightOnlyNVFP4RowExact.maximumRows, rows > chunkRows {
            let flat = x.reshaped(rows, inputs)
            let pieces = stride(from: 0, to: rows, by: chunkRows).map {
                product(
                    flat[$0 ..< min(rows, $0 + chunkRows)], weight: weight,
                    blockScales: blockScales, tensorScale: tensorScale)
            }
            return concatenated(pieces, axis: 0)
                .reshaped(Array(x.shape.dropLast()) + [outputs])
        }
        let (tm, tn) = tiles
        let rowBlocks = (rows + 8 * tm - 1) / (8 * tm)
        let outTiles = (outputs + 8 * tn - 1) / (8 * tn)
        let template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("M", rows), ("K", inputs), ("N", outputs), ("TM", tm), ("TN", tn),
            ("RFAST", rowsFast), ("SCALED", tensorScale != nil),
        ]
        ModelOptRowExactPipelines.record("cbv2_prompt_row_exact_nvfp4", template)
        let (a, b) = rowsFast ? (rowBlocks, outTiles) : (outTiles, rowBlocks)
        let result = kernel(
            [weight, blockScales, x.reshaped(rows, inputs), tensorScale?.reshaped(1) ?? unit],
            template: template,
            grid: (32, 8 * a, b), threadGroup: (32, 8, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }
}

extension ModelOptWeightOnlyNVFP4Linear {
    /// The row-exact product of every row of `input` at once, with each row's
    /// bytes equal to the 16-row pieces'; nil where this call has none.
    public func promptRowExactForward(_ input: MLXArray) -> MLXArray? {
        guard input.ndim > 0 else { return nil }
        let inputs = input.dim(-1)
        let rows = inputs == 0 ? 0 : input.size / inputs
        guard
            CBv2PromptRowExactNVFP4.supports(
                rows: rows, inputs: inputs, outputs: weight.dim(0), dtype: input.dtype),
            weight.dtype == .uint32, weight.ndim == 2, weight.dim(1) * 8 == inputs
        else { return nil }
        return CBv2PromptRowExactNVFP4.product(
            input, weight: weight, blockScales: weightScale, tensorScale: weightScale2)
    }
}
