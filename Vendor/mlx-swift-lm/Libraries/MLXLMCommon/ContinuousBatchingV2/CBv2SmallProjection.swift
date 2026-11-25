// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// Verify-window projections of the attention block (q, k, v, gate and o)
/// with other schedules of the row-exact products: FP8 `ModelOptRowExact`
/// (separate-cast schedule `.windows`) and weight-only NVFP4
/// `ModelOptWeightOnlyNVFP4RowExact` (matrix kernel).
///
/// FP8: every output element keeps the stock arithmetic: the same lane
/// chains (lane `l` of K-simdgroup `g` owns `k = j * BN * 128 + 128 g + 4 l
/// + t`, fused multiply-adds from zero, `j` then `t` ascending, the leftover
/// step's `+0 * +0` adds included), the same `shuffle_down` tree (16, 8, 4,
/// 2, 1, lower lane on the left) and the same join of the K-simdgroups,
/// `((T0 + T1) + ...) + T7`. Only the schedule changes:
/// - one 4-byte weight load and one 8-byte activation load per row and step,
///   with the BN = 1 rows' weight words loaded 2 to 4 steps ahead;
/// - the tree runs packed: the values a lane reduces move in halves, so one
///   shuffle carries one addition of the stock tree for each of 2 to 16
///   values (same operands, same order) instead of one value per shuffle;
/// - the K-simdgroup join runs on one lane per value;
/// - all eager tiles of a projection (the o projection's BN = 1 rows and its
///   BN = 8 tail) and all siblings of a group run in one dispatch that writes
///   the final outputs, so no concatenation follows; BN = 1 rows run eight
///   simdgroups per threadgroup.
///
/// NVFP4: the stock tile body with eight output rows per threadgroup instead
/// of 32 (four times the threadgroups), all siblings in one dispatch.
///
/// The Muse Glimmer fused verify forward routes its FP8 attention
/// projections, MLP gate/up and MLP down up to 5 rows here
/// (`DARKBLOOM_MUSE_GLIMMER_VERIFY_PROJ`).
public enum CBv2SmallProjection {
    public static let maximumRows = 8

    /// Benchmarks only: rows per thread for (BN = 8, BN = 1) segments, in
    /// place of the default choice. Any value gives the same bytes.
    nonisolated(unsafe) public static var rowsPerThreadOverride: (split: Int, flat: Int)?

    // MARK: FP8

    static let fp8Helper = """

        #define CBV2_SP_UNROLL _Pragma("clang loop unroll(full)")

        // One step of the packed tree at offset OFF with C values per lane:
        // each pair of lanes (l, l ^ OFF) exchanges one half of its values,
        // and each lane adds the stock tree's two operands for the half it
        // keeps, the lower lane's partial on the left.
        template <int C, int OFF>
        inline void cbv2_sp_tree_step(thread float* x, ushort lid) {
            if constexpr (C >= 2) {
                constexpr int H = C / 2;
                const bool up = (lid & OFF) != 0;
                CBV2_SP_UNROLL
                for (int j = 0; j < H; j++) {
                    const float keep = up ? x[j + H] : x[j];
                    const float send = up ? x[j] : x[j + H];
                    const float recv = simd_shuffle_xor(send, ushort(OFF));
                    x[j] = up ? (recv + keep) : (keep + recv);
                }
            } else {
                x[0] = x[0] + simd_shuffle_down(x[0], ushort(OFF));
            }
        }

        // Rows TM of one output tile, K split over 8 simdgroups (SPLIT, the
        // stock BN = 8) or one simdgroup per TM rows (stock BN = 1, eight
        // simdgroups per threadgroup). `out_vec` is the output's first column
        // for this segment; NOUT its row stride.
        template <typename T, int K, int V, int TM, bool SPLIT, int N, int NOUT, bool DV, int PF, bool VO>
        inline void cbv2_sp_fp8(
            const device uint8_t* mat, float weight_scale, const device T* in_vec,
            device T* out_vec, int tile, uint simd_gid, uint simd_lid,
            threadgroup float* tgp) {
            constexpr int BN = SPLIT ? 8 : 1;
            constexpr int TN = 4;
            constexpr int blockN = BN * 32 * TN;
            constexpr int n_iter = K / blockN;
            constexpr int leftover = K - blockN * n_iter;
            constexpr int P = V * TM;
            static_assert(P <= 32, "at most 32 values per lane");
            constexpr int P2 = P <= 1 ? 1 : P <= 2 ? 2 : P <= 4 ? 4 : P <= 8 ? 8 : P <= 16 ? 16 : 32;
            constexpr int SH = P2 == 1 ? 5 : P2 == 2 ? 4 : P2 == 4 ? 3 : P2 == 8 ? 2 : P2 == 16 ? 1 : 0;
            constexpr uint LOW = (1u << SH) - 1u;
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            const float dscale = DV ? weight_scale * 256.0f : weight_scale;
            int out_row = SPLIT ? tile * TM : (tile * 8 + int(simd_gid)) * TM;
            if (!SPLIT && out_row >= N) {
                return;
            }
            out_row = out_row + TM <= N ? out_row : N - TM;
            const int g = SPLIT ? int(simd_gid) : 0;
            int bn = (32 * g + int(simd_lid)) * TN;
            const device uint8_t* rows = mat + size_t(out_row) * K;
            float r[P2];
            CBV2_SP_UNROLL
            for (int i = 0; i < P2; i++) {
                r[i] = 0;
            }
            // Weight words are loaded PD iterations at a time, ahead of
            // their use (more loads in flight; the arithmetic is unchanged).
            constexpr int PD = PF <= 1 ? 1 : (PF >= n_iter ? (n_iter > 0 ? n_iter : 1) : PF);
            for (int i0 = 0; i0 < n_iter; i0 += PD) {
              uint wblk[PD][TM];
              CBV2_SP_UNROLL
              for (int d = 0; d < PD; d++) {
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                    wblk[d][tm] = i0 + d < n_iter
                        ? *(const device uint*)(rows + tm * K + bn + d * blockN) : 0u;
                }
              }
              CBV2_SP_UNROLL
              for (int d = 0; d < PD; d++) {
                if (i0 + d >= n_iter) {
                    break;
                }
                ushort4 av[V];
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                    av[v] = ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn));
                }
                thread const uint* wq = wblk[d];
                // Decoded weights of the four codes of each row's word. DV:
                // the half bits of two codes per bit operation (codes 0 and 2,
                // then 1 and 3) with the sign in the half's sign bit, and the
                // exact factor 256 of `modelopt_e4m3_value` moved into the
                // scale: x * (256 s) and (256 x) * s round the same real
                // product once, and -(256 x) == 256 (-x).
                float dec[TM][TN];
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                    const uint w = wq[tm];
                    if (DV) {
                        const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                        const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                        const half2 h02 = as_type<half2>(lo);
                        const half2 h13 = as_type<half2>(hi);
                        dec[tm][0] = static_cast<float>(h02.x);
                        dec[tm][1] = static_cast<float>(h13.x);
                        dec[tm][2] = static_cast<float>(h02.y);
                        dec[tm][3] = static_cast<float>(h13.y);
                        if ((((w & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) != 0) {
                            CBV2_SP_UNROLL
                            for (int tn = 0; tn < TN; tn++) {
                                if (((w >> (8 * tn)) & 0x7Fu) == 0x7Fu) {
                                    dec[tm][tn] = nan_value;
                                }
                            }
                        }
                    } else {
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const uint8_t code = uint8_t((w >> (8 * tn)) & 0xFF);
                            dec[tm][tn] =
                                (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                        }
                    }
                }
                if (VO) {
                    // Rows of activations outermost: each chain still takes
                    // its four inputs in order, with fewer values live.
                    T inter[TM][TN];
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            inter[tm][tn] = static_cast<T>(dec[tm][tn] * dscale);
                        }
                    }
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const float a = static_cast<float>(as_type<T>(av[v][tn]));
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                r[v * TM + tm] += inter[tm][tn] * a;
                            }
                        }
                    }
                } else {
                CBV2_SP_UNROLL
                for (int tn = 0; tn < TN; tn++) {
                    T inter[TM];
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        inter[tm] = static_cast<T>(dec[tm][tn] * dscale);
                    }
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        const float a = static_cast<float>(as_type<T>(av[v][tn]));
                        CBV2_SP_UNROLL
                        for (int tm = 0; tm < TM; tm++) {
                            r[v * TM + tm] += inter[tm] * a;
                        }
                    }
                }
                }
                bn += blockN;
              }
            }
            if (leftover > 0) {
                thread float v_coeff[V][TN];
                thread T inter[TN];
                for (int v = 0; v < V; v++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] =
                            bn + tn < K ? static_cast<float>(in_vec[v * K + bn + tn]) : 0.0f;
                    }
                }
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        if (bn + tn < K) {
                            const uint8_t code = rows[tm * K + bn + tn];
                            const float decoded =
                                (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                            inter[tn] = static_cast<T>(decoded * weight_scale);
                        } else {
                            inter[tn] = T(0);
                        }
                    }
                    for (int v = 0; v < V; v++) {
                        for (int tn = 0; tn < TN; tn++) {
                            r[v * TM + tm] += inter[tn] * v_coeff[v][tn];
                        }
                    }
                }
            }
            const ushort lid = ushort(simd_lid);
            cbv2_sp_tree_step<P2, 16>(r, lid);
            cbv2_sp_tree_step<(P2 >= 2 ? P2 / 2 : 1), 8>(r, lid);
            cbv2_sp_tree_step<(P2 >= 4 ? P2 / 4 : 1), 4>(r, lid);
            cbv2_sp_tree_step<(P2 >= 8 ? P2 / 8 : 1), 2>(r, lid);
            cbv2_sp_tree_step<(P2 >= 16 ? P2 / 16 : 1), 1>(r, lid);
            // Lane l with its low SH bits clear now holds value l >> SH.
            const bool holds = (simd_lid & LOW) == 0;
            const int i = int(simd_lid >> SH);
            if (SPLIT) {
                if (holds) {
                    tgp[g * 32 + i] = r[0];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (g == 0 && holds && i < P) {
                    float result = r[0];
                    for (int s = 1; s < BN; s++) {
                        result += tgp[s * 32 + i];
                    }
                    out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(result);
                }
            } else if (holds && i < P) {
                out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(r[0]);
            }
        }
        """

    /// One segment of a dispatch: `rows` output rows of weight `weight`
    /// starting at weight row `row0`, read from input `input`, written to
    /// output `output` from column `column0`.
    struct FP8Segment: Hashable {
        var weight: Int
        var row0: Int
        var rows: Int
        var input: Int
        var output: Int
        var column0: Int
        var outputWidth: Int
        var split: Bool
        var tm: Int
        var prefetch: Int

        var tiles: Int { split ? (rows + tm - 1) / tm : (rows + 8 * tm - 1) / (8 * tm) }
    }

    struct FP8Shape: Hashable {
        var vectorDecode: Bool
        var rowsOuter: Bool
        var weights: Int
        var outputs: Int
        var segments: [FP8Segment]
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var fp8Kernels: [FP8Shape: MLXFast.MLXFastKernel] = [:]

    private static func fp8Kernel(_ shape: FP8Shape) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let kernel = fp8Kernels[shape] { return kernel }
        var inputNames = [String]()
        for w in 0 ..< shape.weights { inputNames += ["m\(w)", "ws\(w)"] }
        inputNames.append("xc")
        let outputNames = (0 ..< shape.outputs).map { "y\($0)" }
        var body = """
            threadgroup float sp_tgp[8 * 32];
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = 0
        for (index, s) in shape.segments.enumerated() {
            let end = offset + s.tiles
            body += """
                \(index == 0 ? "" : "else ")if (tg < \(end)) {
                    cbv2_sp_fp8<T, K, V, \(s.tm), \(s.split), \(s.rows), \(s.outputWidth), \(shape.vectorDecode), \(s.prefetch), \(shape.rowsOuter)>(
                        m\(s.weight) + size_t(\(s.row0)) * K, ws\(s.weight)[0],
                        xc + size_t(\(s.input)) * V * K,
                        y\(s.output) + \(s.column0), tg - \(offset), sg, sl, sp_tgp);
                }

                """
            offset = end
        }
        let name = "cbv2_small_fp8_" + stableName(body)
        let kernel = MLXFast.metalKernel(
            name: name, inputNames: inputNames, outputNames: outputNames, source: body,
            header: ModelOptActivationKernels.header + fp8Helper)
        fp8Kernels[shape] = kernel
        return kernel
    }

    /// A name that depends only on the kernel's source (FNV-1a).
    static func stableName(_ text: String) -> String {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in text.utf8 { h = (h ^ UInt64(b)) &* 0x100_0000_01b3 }
        return String(h, radix: 36)
    }

    /// Rows per thread: 2, as the stock schedule (4 measured 1.7x slower at
    /// eight rows: registers).
    static func rowsPerThread(rows: Int, split: Bool) -> Int {
        if let o = rowsPerThreadOverride { return split ? o.split : o.flat }
        return 2
    }

    /// Two codes per bit operation in the weight decode, with the exact
    /// factor 256 moved into the scale. Measured no faster and never used:
    /// in-module experiments only (it is one rounding of the same real
    /// product except where `256 * scale` overflows or is subnormal).
    nonisolated(unsafe) static var vectorDecode = false

    /// Weight words loaded ahead, in iterations: none for K-split tiles
    /// (deeper measured slower: registers), 4 (up to 4 rows) or 2 for the
    /// BN = 1 rows (o projection: 32 iterations per lane).
    static func prefetchDepth(rows: Int, split: Bool) -> Int {
        split ? 1 : (rows <= 4 ? 4 : 2)
    }

    /// Activation rows outermost in the inner loop (same chains).
    nonisolated(unsafe) public static var rowsOuter = false

    /// Benchmarks only: weight words loaded ahead, in iterations (1: none).
    nonisolated(unsafe) public static var prefetchOverride: Int?

    /// The runs of one-row eager tiles that share MLX's K split
    /// (`ModelOptCalibratedLinear.vectorRuns`).
    static func fp8Runs(_ layer: ModelOptCalibratedLinear) -> [(rows: Range<Int>, split: Bool)] {
        var runs: [(rows: Range<Int>, split: Bool)] = []
        for tile in layer.fp8EagerTiles(rows: 1) {
            let split = layer.inputDimensions >= 16 * tile.count
            if let last = runs.last, last.split == split {
                runs[runs.count - 1].rows = last.rows.lowerBound ..< tile.upperBound
            } else {
                runs.append((tile, split))
            }
        }
        return runs
    }

    static func fp8Takes(_ layer: ModelOptCalibratedLinear, inputs: Int, dtype: DType) -> Bool {
        // Activations are read as 16-bit lanes (`as_type<T>(ushort4)`).
        guard dtype == .bfloat16 || dtype == .float16, layer.format == .fp8, layer.execution == .graph, layer.weightDType == dtype,
            layer.inputDimensions == inputs, inputs > 64, inputs % 4 == 0,
            layer.rowExactSchedule == ModelOptRowExact.Schedule.windows,
            ModelOptRowExact.geometryOverride == nil,
            let constants = layer.graphConstants, case .fp8 = constants.storage,
            layer.weight.dtype == .uint8,
            layer.weight.shape == [layer.outputDimensions, inputs],
            constants.outputScale.size == 1, constants.outputScale.dtype == .float32,
            layer.outputDimensions >= 8
        else { return false }
        return true
    }

    private static func fp8Run(
        _ layers: [ModelOptCalibratedLinear], _ flat: MLXArray, cast: MLXArray, rows: Int
    ) -> [MLXArray] {
        let inputs = flat.dim(-1)
        var segments = [FP8Segment]()
        var arrays = [MLXArray]()
        for (i, layer) in layers.enumerated() {
            arrays += [layer.weight, layer.graphConstants!.outputScale]
            for run in fp8Runs(layer) {
                let tm = min(
                    rowsPerThread(rows: rows, split: run.split), run.rows.count, max(1, 32 / rows))
                segments.append(
                    FP8Segment(
                        weight: i, row0: run.rows.lowerBound, rows: run.rows.count,
                        input: layers.count == 1 ? 0 : i, output: i,
                        column0: run.rows.lowerBound, outputWidth: layer.outputDimensions,
                        split: run.split, tm: tm,
                        prefetch: prefetchOverride ?? prefetchDepth(rows: rows, split: run.split)))
            }
        }
        arrays.append(cast)
        let shape = FP8Shape(
            vectorDecode: vectorDecode, rowsOuter: rowsOuter, weights: layers.count, outputs: layers.count,
            segments: segments)
        let tiles = segments.reduce(0) { $0 + $1.tiles }
        return fp8Kernel(shape)(
            arrays,
            template: [("T", flat.dtype), ("K", inputs), ("V", rows)],
            grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
            outputShapes: layers.map { [rows, $0.outputDimensions] },
            outputDTypes: Array(repeating: flat.dtype, count: layers.count))
    }

    /// The activation cast of `ModelOptActivationKernels.fp8` for each layer's
    /// own multipliers and bound, one launch (`CBv2GroupedProjection`'s cast).
    private static func castKernel(segments: Int) -> MLXFast.MLXFastKernel {
        var inputs = ["x"]
        for i in 0 ..< segments { inputs += ["mu\(i)", "bd\(i)"] }
        func select(_ name: String, _ element: Int) -> String {
            var expr = "\(name)\(segments - 1)[\(element)]"
            for i in (0 ..< segments - 1).reversed() {
                expr = "(seg == \(i) ? \(name)\(i)[\(element)] : \(expr))"
            }
            return expr
        }
        let source = """
            const uint i = thread_position_in_grid.x;
            const uint seg = i / COUNT;
            const uint idx = i - seg * COUNT;
            const float m0 = \(select("mu", 0));
            const float m1 = \(select("mu", 1));
            const float bound = \(select("bd", 0));
            const float scaled = static_cast<float>(x[idx]) * m0;
            const uint8_t bits = modelopt_e4m3_bits(scaled);
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            const float decoded = (bits & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(bits);
            const bool invalid = metal::isnan(scaled) || metal::abs(scaled) > bound;
            out[i] = static_cast<T>(invalid ? nan_value : decoded * m1);
            """
        return MLXFast.metalKernel(
            name: "cbv2_small_fp8_activation_\(segments)", inputNames: inputs,
            outputNames: ["out"], source: source, header: ModelOptActivationKernels.header)
    }

    nonisolated(unsafe) private static let castKernels: [Int: MLXFast.MLXFastKernel] = [
        2: castKernel(segments: 2), 4: castKernel(segments: 4),
    ]

    /// The row-exact products of FP8 sibling projections of one activation
    /// (`x`: `[..., K]`, 1 to 8 rows): each equal, bit for bit, to the layer's
    /// own `rowExactForward` (or `graphForward` for one row). nil when a layer
    /// is not one this file takes.
    public static func fp8Products(_ layers: [ModelOptCalibratedLinear], _ x: MLXArray)
        -> [MLXArray]?
    {
        let inputs = x.dim(-1)
        guard inputs > 0, !layers.isEmpty else { return nil }
        let rows = x.size / inputs
        guard (1 ... maximumRows).contains(rows),
            layers.allSatisfy({ fp8Takes($0, inputs: inputs, dtype: x.dtype) })
        else { return nil }
        let flat = x.reshaped(rows, inputs)
        let cast: MLXArray
        if layers.count == 1 {
            guard case .fp8(let mu, let bound) = layers[0].graphConstants!.storage else {
                return nil
            }
            cast = ModelOptActivationKernels.fp8(flat, multipliers: mu, overflowBound: bound)
        } else {
            guard let castKernel = castKernels[layers.count] else { return nil }
            var castInputs: [MLXArray] = [flat]
            for layer in layers {
                guard case .fp8(let mu, let bound) = layer.graphConstants!.storage else {
                    return nil
                }
                castInputs += [mu, bound]
            }
            let total = layers.count * rows * inputs
            cast = castKernel(
                castInputs, template: [("T", x.dtype), ("COUNT", rows * inputs)],
                grid: (total, 1, 1), threadGroup: (min(256, total), 1, 1),
                outputShapes: [[total]], outputDTypes: [x.dtype])[0]
        }
        let lead = Array(x.shape.dropLast())
        return zip(fp8Run(layers, flat, cast: cast, rows: rows), layers).map {
            $0.reshaped(lead + [$1.outputDimensions])
        }
    }

    // MARK: NVFP4

    /// `ModelOptWeightOnlyNVFP4RowExact.matrixKernel`'s tile with TN fragments
    /// of eight output rows (stock: 4) and, with PF, the next chunk's weights,
    /// scales and activations loaded before the current chunk's
    /// multiply-accumulates (a register double buffer). Each output keeps the
    /// stock chain: partition `simd_gid` over chunks `simd_gid + 8 j`, blocks
    /// ascending, the same MMA sequence, the same join and rounding.
    static let nvfp4Helper = """

        template <typename T, int V, int K, int N, int TN, bool PF>
        inline void cbv2_sp_nvfp4_tile(
            const device uint32_t* w, const device uint8_t* scales, const device T* x,
            float tensor_scale0, device T* y, int tile, uint simd_gid, uint simd_lid,
            threadgroup float* partials) {
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr int BN = 8 * TN;
            constexpr short TM = (V + 7) / 8;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            constexpr bool whole_blocks = (K % 32) == 0;
            static_assert(!PF || K % BK == 0, "prefetch needs whole chunks");
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            const int y_col = tile * BN;

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

            if constexpr (PF) {
                uint qa[4][TN], qb[4][TN];
                uint sa[4][TN], sb[4][TN];
                float4 xa[4][TM][2], xb[4][TM][2];
                #define CBV2_SP_NV_LOAD(KK, Q, S, X) \\
                    MO_WO_UNROLL for (short block = 0; block < 4; block++) { \\
                        const int b = (KK) + 32 * block; \\
                        MO_WO_UNROLL for (short t = 0; t < TN; t++) { \\
                            Q[block][t] = wl[t][b / 8]; S[block][t] = sl[t][b / 16]; } \\
                        MO_WO_UNROLL for (short mt = 0; mt < TM; mt++) { \\
                            MO_WO_UNROLL for (short j = 0; j < 2; j++) { \\
                                X[block][mt][j] = mo_wo_load4(xr[mt][j] + b); } } }
                #define CBV2_SP_NV_COMPUTE(Q, S, X) \\
                    MO_WO_UNROLL for (short block = 0; block < 4; block++) { \\
                        half s[TN]; \\
                        MO_WO_UNROLL for (short t = 0; t < TN; t++) { s[t] = mo_wo_e4m3(S[block][t]); } \\
                        MO_WO_UNROLL for (short i = 0; i < 4; i++) { \\
                            float2 bf[TM]; \\
                            MO_WO_UNROLL for (short mt = 0; mt < TM; mt++) { \\
                                bf[mt] = float2(X[block][mt][0][i], X[block][mt][1][i]); } \\
                            MO_WO_UNROLL for (short t = 0; t < TN; t++) { \\
                                const float2 a = mo_wo_pair(Q[block][t], i, s[t]); \\
                                MO_WO_UNROLL for (short mt = 0; mt < TM; mt++) { \\
                                    mo_wo_mma(C[t][mt], a, bf[mt]); } } } }
                int k = simd_gid * BK;
                if (k < K) { CBV2_SP_NV_LOAD(k, qa, sa, xa) }
                for (; k < K; k += 2 * WK * BK) {
                    const int k1 = k + WK * BK;
                    if (k1 < K) { CBV2_SP_NV_LOAD(k1, qb, sb, xb) }
                    CBV2_SP_NV_COMPUTE(qa, sa, xa)
                    if (k1 >= K) { break; }
                    const int k2 = k1 + WK * BK;
                    if (k2 < K) { CBV2_SP_NV_LOAD(k2, qa, sa, xa) }
                    CBV2_SP_NV_COMPUTE(qb, sb, xb)
                }
                #undef CBV2_SP_NV_LOAD
                #undef CBV2_SP_NV_COMPUTE
            } else {
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
                                    yr[out] = mo_wo_result<T>(C[t][mt][j], true, tensor_scale0);
                                }
                            }
                        }
                    }
                }
            }
        }
        """

    struct NVFP4Segment: Hashable {
        var outputs: Int
        var tn: Int
        var prefetch: Bool
        var tiles: Int { (outputs + 8 * tn - 1) / (8 * tn) }
    }

    nonisolated(unsafe) private static var nvfp4Kernels: [[NVFP4Segment]: MLXFast.MLXFastKernel] =
        [:]

    private static func nvfp4Kernel(_ segments: [NVFP4Segment]) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let kernel = nvfp4Kernels[segments] { return kernel }
        var inputs = [String](), outputs = [String]()
        let maxTile = segments.map(\.tn).max()! * 2
        var body = """
            threadgroup float partials[7 * \(maxTile) * ((V + 7) / 8) * 32];
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = 0
        for (i, s) in segments.enumerated() {
            inputs += ["w\(i)", "sc\(i)", "ts\(i)"]
            outputs.append("y\(i)")
            let end = offset + s.tiles
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(end)) {
                    cbv2_sp_nvfp4_tile<T, V, K, \(s.outputs), \(s.tn), \(s.prefetch)>(
                        w\(i), sc\(i), x, ts\(i)[0], y\(i), tg - \(offset), sg, sl, partials);
                }

                """
            offset = end
        }
        inputs.append("x")
        let kernel = MLXFast.metalKernel(
            name: "cbv2_small_nvfp4_" + stableName(body), inputNames: inputs,
            outputNames: outputs, source: body,
            header: ModelOptWeightOnlyNVFP4RowExact.header + nvfp4Helper)
        nvfp4Kernels[segments] = kernel
        return kernel
    }

    /// Benchmarks only: (TN, prefetch) for every segment in place of
    /// `nvfp4Tile`.
    nonisolated(unsafe) public static var nvfp4TileOverride: (tn: Int, prefetch: Bool)?

    /// Tile per segment: eight output rows per threadgroup (stock 32): the
    /// attention group takes 1,088 threadgroups instead of 272 and the o
    /// projection 832 instead of 208, which balances the 64 cores (measured
    /// faster at 1 to 8 rows than 16- or 32-row tiles, with or without
    /// prefetch).
    static func nvfp4Tile(outputs: Int, inputs: Int) -> (tn: Int, prefetch: Bool) {
        if let o = nvfp4TileOverride { return (o.tn, o.prefetch && inputs % 128 == 0) }
        return (1, false)
    }

    /// Row-exact products of weight-only NVFP4 siblings of one activation
    /// (`x`: `[..., K]`, 1 to 8 rows), each equal, bit for bit, to
    /// `ModelOptWeightOnlyNVFP4RowExact.product` with the layer's tensor
    /// scale (matrix kernel); nil when not taken.
    public static func nvfp4Products(_ layers: [ModelOptWeightOnlyNVFP4Linear], _ x: MLXArray)
        -> [MLXArray]?
    {
        let inputs = x.dim(-1)
        guard inputs > 0, !layers.isEmpty else { return nil }
        let rows = x.size / inputs
        guard (1 ... maximumRows).contains(rows),
            rows > ModelOptWeightOnlyNVFP4RowExact.scalarRows,
            ModelOptWeightOnlyNVFP4RowExact.supports(rows: rows, inputs: inputs, dtype: x.dtype)
        else { return nil }
        var arrays = [MLXArray]()
        var segments = [NVFP4Segment]()
        for layer in layers {
            let outputs = layer.outputDimensions
            guard layer.inputDimensions == inputs, outputs > 0,
                layer.weight.dtype == .uint32, layer.weight.shape == [outputs, inputs / 8],
                layer.weight.size >= 8, layer.weightScale.dtype == .uint8,
                layer.weightScale.shape == [outputs, inputs / 16], layer.weightScale.size >= 8,
                layer.weightScale2.dtype == .float32, layer.weightScale2.size == 1
            else { return nil }
            let (tn, prefetch) = nvfp4Tile(outputs: outputs, inputs: inputs)
            segments.append(NVFP4Segment(outputs: outputs, tn: tn, prefetch: prefetch))
            arrays += [layer.weight, layer.weightScale, layer.weightScale2.reshaped(1)]
        }
        arrays.append(x.reshaped(rows, inputs))
        let tiles = segments.reduce(0) { $0 + $1.tiles }
        let results = nvfp4Kernel(segments)(
            arrays, template: [("T", x.dtype), ("V", rows), ("K", inputs)],
            grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
            outputShapes: layers.map { [rows, $0.outputDimensions] },
            outputDTypes: Array(repeating: x.dtype, count: layers.count))
        let lead = Array(x.shape.dropLast())
        return zip(results, layers).map { $0.reshaped(lead + [$1.outputDimensions]) }
    }

    /// Runs the FP8 kernels of `layers` on zeros at every row count in `rows`.
    public static func warmFP8(
        _ layers: [ModelOptCalibratedLinear], dtype: DType, rows: ClosedRange<Int>
    ) {
        guard let inputs = layers.first?.inputDimensions else { return }
        for count in rows {
            let zeros = MLXArray.zeros([1, count, inputs], dtype: dtype)
            if let outputs = fp8Products(layers, zeros) { eval(outputs) }
        }
    }
}

// MARK: FP8 matrix products for low-acceptance verify windows

/// FP8 verify projections on simdgroup matrices: eight output rows, eight
/// activation rows and eight inputs per multiply-accumulate, every weight
/// read once for up to 8 rows. The stock row-exact products run one fused
/// multiply-add chain per row, so their cost grows with the window's rows;
/// these do not. They keep the products' operands (the FP8 activation cast,
/// `T(decode(w) * scale)` weights, float32 sums, one rounding to `T`) but
/// not the row-exact order of the sums, so a row's last bits can differ
/// from the one-row step's. The window's tokens are the argmax of its rows
/// (contract section 11.5); only low-acceptance rounds take them.
extension CBv2SmallProjection {
    static let fp8MatrixEnabled =
        ProcessInfo.processInfo.environment["CBV2_FP8_MMA"] != "0"
    /// Set by the engine while a low-acceptance verify window is built.
    nonisolated(unsafe) public static var fp8MatrixRounds = false

    static let fp8MatrixHelper = """

        #include <metal_simdgroup_matrix>
        #define CBV2_F8M_UNROLL _Pragma("clang loop unroll(full)")
        typedef metal::simdgroup_matrix<float, 8, 8> cbv2_f8m_matrix;

        inline void cbv2_f8m_mma(thread float2& C, float2 A, float2 B) {
            cbv2_f8m_matrix D_mat, A_mat, B_mat, C_mat;
            reinterpret_cast<thread float2&>(A_mat.thread_elements()) = A;
            reinterpret_cast<thread float2&>(B_mat.thread_elements()) = B;
            reinterpret_cast<thread float2&>(C_mat.thread_elements()) = C;
            simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
            C = reinterpret_cast<thread float2&>(D_mat.thread_elements());
        }

        // One 32-output tile of y = x . W^T over all K: eight simdgroups take
        // the 128-input chunks q with q % 8 == simdgroup, then join.
        template <typename T, int V, int K, int N>
        inline void cbv2_f8m_tile(
            const device uint8_t* w, float wscale, const device T* x, device T* y,
            int tile, uint simd_gid, uint simd_lid, threadgroup float* partials) {
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr short TN = 4;
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            const int y_col = tile * 32;
            const float nanv = metal::numeric_limits<float>::quiet_NaN();
            const device T* xr[2];
            CBV2_F8M_UNROLL for (short j = 0; j < 2; j++) {
                xr[j] = x + size_t(min(fn + j, V - 1)) * K + 4 * fm;
            }
            float2 C[TN];
            CBV2_F8M_UNROLL for (short t = 0; t < TN; t++) { C[t] = float2(0); }
            const device uint8_t* wl[TN];
            CBV2_F8M_UNROLL for (short t = 0; t < TN; t++) {
                const int row = min(y_col + fm + 8 * t, N - 1);
                wl[t] = w + size_t(row) * K + 4 * fn;
            }
            for (int k = int(simd_gid) * BK; k < K; k += WK * BK) {
                CBV2_F8M_UNROLL for (short block = 0; block < 4; block++) {
                    const int b = k + 32 * block;
                    float4 xv[2];
                    CBV2_F8M_UNROLL for (short j = 0; j < 2; j++) {
                        xv[j] = float4(*(const device vec<T, 4>*)(xr[j] + b));
                    }
                    uint2 q[TN];
                    CBV2_F8M_UNROLL for (short t = 0; t < TN; t++) {
                        q[t] = *(const device uint2*)(wl[t] + b);
                    }
                    CBV2_F8M_UNROLL for (short i = 0; i < 4; i++) {
                        const float2 bf = float2(xv[0][i], xv[1][i]);
                        CBV2_F8M_UNROLL for (short t = 0; t < TN; t++) {
                            const uint c0 = (q[t].x >> (8 * i)) & 255u;
                            const uint c1 = (q[t].y >> (8 * i)) & 255u;
                            const float d0 = (c0 & 0x7F) == 0x7F ? nanv : modelopt_e4m3_value(uint8_t(c0));
                            const float d1 = (c1 & 0x7F) == 0x7F ? nanv : modelopt_e4m3_value(uint8_t(c1));
                            cbv2_f8m_mma(
                                C[t], float2(float(T(d0 * wscale)), float(T(d1 * wscale))), bf);
                        }
                    }
                }
            }
            if (simd_gid > 0) {
                threadgroup float* p = partials + (simd_gid - 1) * TN * 2 * 32 + simd_lid;
                CBV2_F8M_UNROLL for (short t = 0; t < TN; t++) {
                    CBV2_F8M_UNROLL for (short j = 0; j < 2; j++) { p[(t * 2 + j) * 32] = C[t][j]; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_gid == 0) {
                for (int s = 0; s < WK - 1; s++) {
                    const threadgroup float* p = partials + s * TN * 2 * 32 + simd_lid;
                    CBV2_F8M_UNROLL for (short t = 0; t < TN; t++) {
                        CBV2_F8M_UNROLL for (short j = 0; j < 2; j++) { C[t][j] += p[(t * 2 + j) * 32]; }
                    }
                }
                CBV2_F8M_UNROLL for (short j = 0; j < 2; j++) {
                    if (fn + j < V) {
                        device T* yr = y + size_t(fn + j) * N;
                        CBV2_F8M_UNROLL for (short t = 0; t < TN; t++) {
                            const int out = y_col + fm + 8 * t;
                            if (out < N) { yr[out] = static_cast<T>(C[t][j]); }
                        }
                    }
                }
            }
        }
        """

    private static func fp8MatrixKernel(layers: Int) -> MLXFast.MLXFastKernel {
        var body = """
            threadgroup float partials[7 * 4 * 2 * 32];
            int g = int(threadgroup_position_in_grid.y);

            """
        for i in 0 ..< layers {
            let tiles = "((N\(i) + 31) / 32)"
            let xOffset = layers == 1 ? "0" : "\(i) * V * K"
            body += """
                if (g < \(tiles)) {
                    cbv2_f8m_tile<T, V, K, N\(i)>(
                        w\(i), s\(i)[0], x + \(xOffset), y\(i), g,
                        simdgroup_index_in_threadgroup, thread_index_in_simdgroup, partials);
                    return;
                }
                g -= \(tiles);

                """
        }
        return MLXFast.metalKernel(
            name: "cbv2_fp8_matrix_\(layers)",
            inputNames: (0 ..< layers).map { "w\($0)" } + (0 ..< layers).map { "s\($0)" } + ["x"],
            outputNames: (0 ..< layers).map { "y\($0)" },
            source: body, header: ModelOptActivationKernels.header + fp8MatrixHelper)
    }

    nonisolated(unsafe) private static let fp8MatrixKernels: [Int: MLXFast.MLXFastKernel] = [
        1: fp8MatrixKernel(layers: 1), 2: fp8MatrixKernel(layers: 2),
        4: fp8MatrixKernel(layers: 4),
    ]

    /// FP8 sibling projections of one activation (`x`: `[..., K]`, 2 to 8
    /// rows) on simdgroup matrices; nil unless `fp8MatrixRounds` is set and
    /// every layer is one this path takes (then the caller's route runs).
    public static func fp8MatrixProducts(_ layers: [ModelOptCalibratedLinear], _ x: MLXArray)
        -> [MLXArray]?
    {
        guard fp8MatrixEnabled, fp8MatrixRounds, let kernel = fp8MatrixKernels[layers.count]
        else { return nil }
        let inputs = x.dim(-1)
        guard inputs > 0, inputs % 128 == 0, x.dtype == .bfloat16 || x.dtype == .float16
        else { return nil }
        let rows = x.size / inputs
        guard (2 ... maximumRows).contains(rows),
            layers.allSatisfy({ fp8Takes($0, inputs: inputs, dtype: x.dtype) })
        else { return nil }
        let flat = x.reshaped(rows, inputs)
        let cast: MLXArray
        if layers.count == 1 {
            guard case .fp8(let mu, let bound) = layers[0].graphConstants!.storage else {
                return nil
            }
            cast = ModelOptActivationKernels.fp8(flat, multipliers: mu, overflowBound: bound)
        } else {
            guard let castKernel = castKernels[layers.count] else { return nil }
            var castInputs: [MLXArray] = [flat]
            for layer in layers {
                guard case .fp8(let mu, let bound) = layer.graphConstants!.storage else {
                    return nil
                }
                castInputs += [mu, bound]
            }
            let total = layers.count * rows * inputs
            cast = castKernel(
                castInputs, template: [("T", x.dtype), ("COUNT", rows * inputs)],
                grid: (total, 1, 1), threadGroup: (min(256, total), 1, 1),
                outputShapes: [[total]], outputDTypes: [x.dtype])[0]
        }
        var template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("V", rows), ("K", inputs),
        ]
        var tiles = 0
        for (i, layer) in layers.enumerated() {
            template.append(("N\(i)", layer.outputDimensions))
            tiles += (layer.outputDimensions + 31) / 32
        }
        let outputs = kernel(
            layers.map(\.weight) + layers.map { $0.graphConstants!.outputScale } + [cast],
            template: template,
            grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
            outputShapes: layers.map { [rows, $0.outputDimensions] },
            outputDTypes: Array(repeating: x.dtype, count: layers.count))
        let lead = Array(x.shape.dropLast())
        return zip(outputs, layers).map { $0.reshaped(lead + [$1.outputDimensions]) }
    }
}
