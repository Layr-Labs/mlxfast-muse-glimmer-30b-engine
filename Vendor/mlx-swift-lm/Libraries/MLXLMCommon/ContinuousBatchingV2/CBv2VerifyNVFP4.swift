// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// The weight-only NVFP4 row-exact product of verify windows (1 to 8 rows)
/// on a finer launch geometry.
///
/// Every output element is computed with exactly the arithmetic of
/// `ModelOptWeightOnlyNVFP4RowExact.matrixKernel`: the same weight decode
/// (`mo_wo_e4m3`, `mo_wo_pair`), the same eight K partitions (simdgroup `s`
/// takes the chunks of 128 inputs with `q % 8 == s`, ascending), the same
/// blocks, the same 8x8x8 multiply-accumulate sequence (inputs `b + 4c + i`,
/// `i` outer, `c` inside the MMA), the same join `((acc0 + acc1) + ...) + acc7`
/// through threadgroup memory, the same rounding to the activation dtype and
/// the same tensor scale. Only the schedule differs:
/// - `TN` output fragments of 8 rows per simdgroup (stock: 4), so a
///   threadgroup covers `8 * TN` outputs and the launch has 4x (or 2x) as
///   many threadgroups, which keeps every core busy and fits more of them;
/// - `PF 1`: the next chunk's weight words and scale bytes are loaded into a
///   rolling register buffer right after the current chunk's are consumed
///   (prefetch one chunk ahead), for the long 19,968-input rows;
/// - activations are kept as their raw bf16 bits until each MMA converts them
///   (the same float values the stock kernel's float4 holds).
/// Rows past V repeat row V - 1, as in the stock kernel; their columns are
/// never stored. Groups of sibling projections (MLP gate and up) run in one
/// launch, one segment per projection, as `CBv2GroupedProjection` does.
///
/// The Muse Glimmer fused verify forward routes its NVFP4 MLP here
/// (`DARKBLOOM_MUSE_GLIMMER_VERIFY_PROJ`). Only bfloat16 activations, whole
/// chunks (K % 128 == 0) and GPU generations where the matrix kernel serves
/// the row count are taken; anything else returns nil and the caller keeps
/// the stock product.
public enum CBv2VerifyNVFP4 {
    public static let maximumRows = 8

    /// Output fragments per simdgroup and prefetch mode.
    public struct Geometry: Hashable, Sendable {
        public var fragments: Int
        public var prefetch: Int
        public init(fragments: Int, prefetch: Int) {
            self.fragments = fragments
            self.prefetch = prefetch
        }
    }

    /// Tests and benchmarks pin a geometry here (nil: `geometry(rows:inputs:)`).
    nonisolated(unsafe) public static var geometryOverride: Geometry?

    /// Measured on the M5 Ultra (micro, chained, M = 1 to 8): 8-row tiles
    /// everywhere, except 16-row tiles with prefetch for the 19,968-input
    /// rows (MLP down) at 8 rows, where the activation stream of 8-row tiles
    /// costs more than the extra fragment (113 against 120 µs; tied at 7).
    public static func geometry(rows: Int, inputs: Int) -> Geometry {
        if let geometryOverride { return geometryOverride }
        if inputs >= 16384 && rows >= 8 { return Geometry(fragments: 2, prefetch: 1) }
        return Geometry(fragments: 1, prefetch: 0)
    }

    static let tileHelper = """

        // One tile of 8 * TN outputs of one projection: the stock matrix
        // kernel's arithmetic (see CBv2VerifyNVFP4) on 8 * TN rows.
        template <typename T, int V, int K, int N, int TN, int PF, bool SCALED>
        inline void cbv2_vn_tile(
            const device uint32_t* w, const device uint8_t* scales, const device T* x,
            float tensor_scale0, device T* y, int tile, uint simd_gid, uint simd_lid,
            threadgroup float* partials) {
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr short TM = (V + 7) / 8;
            constexpr int ROWS = 8 * TN;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            static_assert(K % BK == 0, "whole chunks only");
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            const int y_col = tile * ROWS;

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

            // Rolling buffer (PF 1): slot `block` holds this chunk's block and is
            // refilled with the next chunk's block once read.
            uint qb[4][TN];
            uint sb[4][TN];
            if (PF == 1 && int(simd_gid) * BK < K) {
                const int k0 = simd_gid * BK;
                MO_WO_UNROLL
                for (short block = 0; block < 4; block++) {
                    const int b = k0 + 32 * block;
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        qb[block][t] = wl[t][b / 8];
                        sb[block][t] = sl[t][b / 16];
                    }
                }
            }
            for (int k = simd_gid * BK; k < K; k += WK * BK) {
                const int kn = k + WK * BK;
                const bool more = kn < K;
                MO_WO_UNROLL
                for (short block = 0; block < 4; block++) {
                    const int b = k + 32 * block;
                    ushort4 xv[TM][2];
                    MO_WO_UNROLL
                    for (short mt = 0; mt < TM; mt++) {
                        MO_WO_UNROLL
                        for (short j = 0; j < 2; j++) {
                            xv[mt][j] = ushort4(*(const device packed_ushort4*)(xr[mt][j] + b));
                        }
                    }
                    uint q[TN];
                    half s[TN];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        if (PF == 1) {
                            q[t] = qb[block][t];
                            s[t] = mo_wo_e4m3(sb[block][t]);
                        } else {
                            q[t] = wl[t][b / 8];
                            s[t] = mo_wo_e4m3(sl[t][b / 16]);
                        }
                    }
                    if (PF == 1 && more) {
                        const int bn = kn + 32 * block;
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            qb[block][t] = wl[t][bn / 8];
                            sb[block][t] = sl[t][bn / 16];
                        }
                    }
                    MO_WO_UNROLL
                    for (short i = 0; i < 4; i++) {
                        float2 bf[TM];
                        MO_WO_UNROLL
                        for (short mt = 0; mt < TM; mt++) {
                            bf[mt] = float2(
                                float(as_type<bfloat>(xv[mt][0][i])),
                                float(as_type<bfloat>(xv[mt][1][i])));
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
                                    yr[out] = mo_wo_result<T>(C[t][mt][j], SCALED, tensor_scale0);
                                }
                            }
                        }
                    }
                }
            }
        }
        """

    /// Segment `i` takes tiles `[sum of tiles before, + ceil(N_i / (8 * TN)))`
    /// of the grid's y axis.
    private static func kernel(segments: Int) -> MLXFast.MLXFastKernel {
        var inputs = [String](), outputs = [String]()
        var body = """
            constexpr int TILE_ELEMS = TN * ((V + 7) / 8) * 2;
            threadgroup float partials[7 * TILE_ELEMS * 32];
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = "0"
        for i in 0 ..< segments {
            inputs += ["w\(i)", "sc\(i)", "ts\(i)"]
            outputs.append("y\(i)")
            let tiles = "((N\(i) + 8 * TN - 1) / (8 * TN))"
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(offset) + \(tiles)) {
                    cbv2_vn_tile<T, V, K, N\(i), TN, PF, SCALED>(
                        w\(i), sc\(i), x, ts\(i)[0], y\(i), tg - (\(offset)), sg, sl, partials);
                }

                """
            offset += " + \(tiles)"
        }
        inputs.append("x")
        return MLXFast.metalKernel(
            name: "cbv2_verify_nvfp4_row_exact_\(segments)", inputNames: inputs,
            outputNames: outputs, source: body,
            header: ModelOptWeightOnlyNVFP4RowExact.header + tileHelper)
    }

    nonisolated(unsafe) private static let kernels: [Int: MLXFast.MLXFastKernel] = [
        1: kernel(segments: 1), 2: kernel(segments: 2),
    ]

    nonisolated(unsafe) private static let unit = MLXArray([Float(1)])

    /// Whether `product` takes `rows` rows of `inputs` activations of `dtype`.
    /// The tile reads activations as bfloat16 bits, so only bfloat16 is taken.
    public static func supports(rows: Int, inputs: Int, dtype: DType) -> Bool {
        dtype == .bfloat16 && (1 ... maximumRows).contains(rows) && inputs > 0
            && inputs % 128 == 0
            && rows > ModelOptWeightOnlyNVFP4RowExact.scalarRows
            && ModelOptWeightOnlyNVFP4RowExact.supports(rows: rows, inputs: inputs, dtype: dtype)
    }

    private static func takes(
        weight: MLXArray, blockScales: MLXArray, tensorScale: MLXArray?, inputs: Int
    ) -> Bool {
        let outputs = weight.dim(0)
        return outputs > 0 && weight.dtype == .uint32 && weight.shape == [outputs, inputs / 8]
            && weight.size >= 8 && blockScales.dtype == .uint8
            && blockScales.shape == [outputs, inputs / 16] && blockScales.size >= 8
            && (tensorScale.map { $0.dtype == .float32 && $0.size == 1 } ?? true)
    }

    /// `ModelOptWeightOnlyNVFP4RowExact.product(x, weight:, blockScales:,
    /// tensorScale:)` bit for bit, or nil when this shape is not taken.
    public static func product(
        _ x: MLXArray, weight: MLXArray, blockScales: MLXArray, tensorScale: MLXArray? = nil
    ) -> MLXArray? {
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        guard supports(rows: rows, inputs: inputs, dtype: x.dtype),
            takes(weight: weight, blockScales: blockScales, tensorScale: tensorScale, inputs: inputs)
        else { return nil }
        let outputs = weight.dim(0)
        let g = geometry(rows: rows, inputs: inputs)
        let tile = 8 * g.fragments
        let template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("V", rows), ("K", inputs), ("TN", g.fragments), ("PF", g.prefetch),
            ("SCALED", tensorScale != nil), ("N0", outputs),
        ]
        let result = kernels[1]!(
            [weight, blockScales, tensorScale?.reshaped(1) ?? unit, x.reshaped(rows, inputs)],
            template: template,
            grid: (32, 8 * ((outputs + tile - 1) / tile), 1), threadGroup: (32, 8, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    /// `layer.rowExactProduct(x)` of the Muse Glimmer tower for a weight-only
    /// NVFP4 layer, or nil.
    public static func product(_ layer: ModelOptWeightOnlyNVFP4Linear, _ x: MLXArray) -> MLXArray? {
        product(x, weight: layer.weight, blockScales: layer.weightScale, tensorScale: layer.weightScale2)
    }

    /// The products of two sibling NVFP4 layers of one input (MLP gate and
    /// up), one launch, each equal to its layer's own row-exact product; nil
    /// when the pair is not taken.
    public static func products(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray]? {
        guard layers.count == 2, let nv = layers as? [ModelOptWeightOnlyNVFP4Linear] else {
            return nil
        }
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        guard supports(rows: rows, inputs: inputs, dtype: x.dtype) else { return nil }
        for layer in nv {
            guard layer.inputDimensions == inputs,
                takes(weight: layer.weight, blockScales: layer.weightScale,
                    tensorScale: layer.weightScale2, inputs: inputs)
            else { return nil }
        }
        let g = geometry(rows: rows, inputs: inputs)
        let tile = 8 * g.fragments
        var arrays = [MLXArray]()
        var template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("V", rows), ("K", inputs), ("TN", g.fragments), ("PF", g.prefetch),
            ("SCALED", true),
        ]
        var shapes = [[Int]]()
        var tiles = 0
        for (i, layer) in nv.enumerated() {
            let outputs = layer.weight.dim(0)
            arrays += [layer.weight, layer.weightScale, layer.weightScale2.reshaped(1)]
            template.append(("N\(i)", outputs))
            shapes.append([rows, outputs])
            tiles += (outputs + tile - 1) / tile
        }
        arrays.append(x.reshaped(rows, inputs))
        let results = kernels[2]!(
            arrays, template: template, grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
            outputShapes: shapes, outputDTypes: [x.dtype, x.dtype])
        let lead = Array(x.shape.dropLast())
        return zip(results, nv).map { $0.reshaped(lead + [$1.weight.dim(0)]) }
    }

    /// Compiles the kernels for rows 1 to 8 of these MLP layers (gate and up
    /// as a pair, down alone) on zeros, before any timed window.
    public static func warm(
        gateUp: [any UnaryLayer], down: ModelOptWeightOnlyNVFP4Linear?, dtype: DType,
        rows: ClosedRange<Int> = 1 ... maximumRows
    ) {
        var outs = [MLXArray]()
        for r in rows {
            if let first = gateUp.first as? ModelOptWeightOnlyNVFP4Linear {
                let x = MLXArray.zeros([r, first.inputDimensions], dtype: dtype)
                outs += products(gateUp, x) ?? []
            }
            if let down {
                let x = MLXArray.zeros([r, down.inputDimensions], dtype: dtype)
                if let y = product(down, x) { outs.append(y) }
            }
        }
        if !outs.isEmpty { eval(outs) }
    }
}
