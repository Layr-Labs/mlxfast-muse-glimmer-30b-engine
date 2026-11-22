// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// Sibling projections that read one activation, run in ONE dispatch.
///
/// Every output element is computed by the row-exact kernel of its own
/// projection with the same template values, the same tile geometry and the
/// same order of operations; only which threadgroup of one launch computes it
/// changes. The kernel bodies below are the bodies of
/// `ModelOptWeightOnlyNVFP4RowExact.matrixKernel` and `ModelOptRowExact.kernel`
/// (FP8, separate-cast schedule `.windows`) moved into helper functions, with a
/// segment table choosing weights and output by threadgroup index. Groups are
/// taken for 2 to 8 rows of the exact windows; anything else returns nil and
/// the caller runs the projections one by one.
public enum CBv2GroupedProjection {
    /// Set `DARKBLOOM_MUSE_GLIMMER_GROUPED_VERIFY=0` to turn the grouping off.
    public static let enabled: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_MUSE_GLIMMER_GROUPED_VERIFY"] != "0"

    public static let maximumRows = 8

    // MARK: NVFP4

    private static let nvfp4Helper = """

        // `ModelOptWeightOnlyNVFP4RowExact.matrixKernel`, one output tile of one segment.
        template <typename T, int V, int K, int N>
        inline void mo_gx_nvfp4_tile(
            const device uint32_t* w, const device uint8_t* scales, const device T* x,
            float tensor_scale0, device T* y, int tile, uint simd_gid, uint simd_lid,
            threadgroup float* partials) {
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr int BN = 32;
            constexpr short TN = BN / 8;
            constexpr short TM = (V + 7) / 8;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            constexpr bool whole_blocks = (K % 32) == 0;
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

    /// Segment `i` takes tiles `[sum of tiles before, + ceil(N_i / 32))` of the
    /// grid's y axis.
    private static func nvfp4Kernel(segments: Int) -> MLXFast.MLXFastKernel {
        var inputs = [String](), outputs = [String]()
        var body = """
            constexpr int TILE_ELEMS = 4 * ((V + 7) / 8) * 2;
            threadgroup float partials[7 * TILE_ELEMS * 32];
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = "0"
        for i in 0 ..< segments {
            inputs += ["w\(i)", "sc\(i)", "ts\(i)"]
            outputs.append("y\(i)")
            let tiles = "((N\(i) + 31) / 32)"
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(offset) + \(tiles)) {
                    mo_gx_nvfp4_tile<T, V, K, N\(i)>(
                        w\(i), sc\(i), x, ts\(i)[0], y\(i), tg - (\(offset)), sg, sl, partials);
                }

                """
            offset += " + \(tiles)"
        }
        inputs.append("x")
        return MLXFast.metalKernel(
            name: "cbv2_grouped_nvfp4_row_exact_\(segments)", inputNames: inputs,
            outputNames: outputs, source: body,
            header: ModelOptWeightOnlyNVFP4RowExact.header + nvfp4Helper)
    }

    nonisolated(unsafe) private static let nvfp4Kernels: [Int: MLXFast.MLXFastKernel] = [
        2: nvfp4Kernel(segments: 2), 4: nvfp4Kernel(segments: 4),
    ]

    // MARK: FP8

    private static let fp8Helper = """

        // `ModelOptRowExact.kernel` for FP8 weights, no inline activation cast,
        // BM 1, BN 8, TM 2 (schedule `.windows`, K split over 8 simdgroups),
        // one threadgroup `group` of one segment.
        template <typename T, int K, int N, int V>
        inline void mo_gx_fp8_group(
            const device uint8_t* mat, float weight_scale, const device T* in_vec,
            device T* out_vec, int group, uint simd_gid, uint simd_lid,
            threadgroup float* tgp_memory) {
            constexpr int BM = 1;
            constexpr int BN = 8;
            constexpr int TM = 2;
            constexpr int SM = 1;
            constexpr int SN = 32;
            constexpr int TN = 4;
            constexpr int threadsM = BM * SM;
            constexpr int threadsN = BN * SN;
            constexpr int blockM = threadsM * TM;
            constexpr int blockN = threadsN * TN;
            constexpr int tgp_size = BN > 1 ? BN * (blockM + TM) : 1;
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            thread float result[V][TM];
            for (int v = 0; v < V; v++) {
                for (int tm = 0; tm < TM; tm++) {
                    result[v][tm] = 0;
                }
            }
            thread T inter[TN];
            thread float v_coeff[V][TN];
            const int thrM = 0;
            const int thrN = int(simd_lid);
            const int sgN = BN != 1 ? (simd_gid % BN) : 0;
            const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
            const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
            int bm = (simdM + thrM) * TM;
            int bn = (simdN + thrN) * TN;
            int out_row = group * blockM + bm;
            if (out_row >= N) {
                return;
            }
            out_row = out_row + TM <= N ? out_row : N - TM;
            const device auto* rows = mat + size_t(out_row) * K;
            constexpr int n_iter = K / blockN;
            constexpr int leftover = K - blockN * n_iter;
            for (int i = 0; i < n_iter; ++i) {
                for (int v = 0; v < V; v++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] = static_cast<float>(in_vec[v * K + bn + tn]);
                    }
                }
                int mat_offset = 0;
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        const uint8_t code = rows[mat_offset + bn + tn];
                        const float decoded =
                            (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                        inter[tn] = static_cast<T>(decoded * weight_scale);
                    }
                    for (int v = 0; v < V; v++) {
                        for (int tn = 0; tn < TN; tn++) {
                            result[v][tm] += inter[tn] * v_coeff[v][tn];
                        }
                    }
                    mat_offset += K;
                }
                bn += blockN;
            }
            if (leftover > 0) {
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
                            result[v][tm] += inter[tn] * v_coeff[v][tn];
                        }
                    }
                }
            }
            for (int v = 0; v < V; v++) {
                for (int tm = 0; tm < TM; tm++) {
                    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                        result[v][tm] += simd_shuffle_down(result[v][tm], sn);
                    }
                }
            }
            if (BN > 1) {
                if (thrN == 0) {
                    for (int v = 0; v < V; v++) {
                        threadgroup float* tgp_results =
                            tgp_memory + v * tgp_size + sgN * (blockM + TM) + bm;
                        for (int tm = 0; tm < TM; tm++) {
                            tgp_results[tm] = result[v][tm];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_none);
                    if (sgN == 0) {
                        for (int v = 0; v < V; v++) {
                            threadgroup float* tgp_results =
                                tgp_memory + v * tgp_size + sgN * (blockM + TM) + bm;
                            for (int sgn = 1; sgn < BN; sgn++) {
                                for (int tm = 0; tm < TM; tm++) {
                                    result[v][tm] += tgp_results[sgn * (blockM + TM) + tm];
                                }
                            }
                        }
                    }
                }
            }
            if (simdN == 0 && thrN == 0) {
                for (int v = 0; v < V; v++) {
                    for (int tm = 0; tm < TM; tm++) {
                        out_vec[v * N + out_row + tm] = static_cast<T>(result[v][tm]);
                    }
                }
            }
        }
        """

    private static func fp8Kernel(segments: Int) -> MLXFast.MLXFastKernel {
        var inputs = [String](), outputs = [String]()
        var body = """
            threadgroup float tgp_memory[V * 32];
            const int g = int(threadgroup_position_in_grid.x);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = "0"
        for i in 0 ..< segments {
            inputs += ["m\(i)", "ws\(i)"]
            outputs.append("o\(i)")
            let groups = "((N\(i) + 1) / 2)"
            body += """
                \(i == 0 ? "" : "else ")if (g < \(offset) + \(groups)) {
                    mo_gx_fp8_group<T, K, N\(i), V>(
                        m\(i), ws\(i)[0], xc + \(i) * V * K, o\(i), g - (\(offset)), sg, sl,
                        tgp_memory);
                }

                """
            offset += " + \(groups)"
        }
        inputs.append("xc")
        return MLXFast.metalKernel(
            name: "cbv2_grouped_fp8_row_exact_\(segments)", inputNames: inputs,
            outputNames: outputs, source: body,
            header: ModelOptActivationKernels.header + fp8Helper)
    }

    nonisolated(unsafe) private static let fp8Kernels: [Int: MLXFast.MLXFastKernel] = [
        2: fp8Kernel(segments: 2), 4: fp8Kernel(segments: 4),
    ]

    /// `ModelOptActivationKernels.fp8Kernel` for each segment's own multipliers
    /// and bound, one launch; segment `i` writes `out[i * COUNT ..< (i + 1) * COUNT]`.
    private static func castKernel(segments: Int) -> MLXFast.MLXFastKernel {
        var inputs = ["x"]
        for i in 0 ..< segments { inputs += ["mu\(i)", "bd\(i)"] }
        // Per-segment scalars are chosen by value: the small arrays live in the
        // `constant` address space, so pointers to them cannot share a ternary.
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
            name: "cbv2_grouped_fp8_activation_\(segments)", inputNames: inputs,
            outputNames: ["out"], source: source, header: ModelOptActivationKernels.header)
    }

    nonisolated(unsafe) static let castKernels: [Int: MLXFast.MLXFastKernel] = [
        2: castKernel(segments: 2), 4: castKernel(segments: 4),
    ]

    // MARK: Entry

    /// What a group of projections takes: `nil` when it cannot be grouped.
    private enum Plan {
        case nvfp4([ModelOptWeightOnlyNVFP4Linear])
        case fp8([ModelOptCalibratedLinear])
    }

    private static func plan(_ layers: [any UnaryLayer], rows: Int, dtype: DType) -> Plan? {
        guard enabled, layers.count == 2 || layers.count == 4, (2 ... maximumRows).contains(rows)
        else { return nil }
        if let nv = layers as? [ModelOptWeightOnlyNVFP4Linear] {
            return nvfp4Plan(nv, rows: rows, dtype: dtype)
        }
        if let calibrated = layers as? [ModelOptCalibratedLinear] {
            return fp8Plan(calibrated, rows: rows, dtype: dtype)
        }
        return nil
    }

    private static func nvfp4Plan(
        _ layers: [ModelOptWeightOnlyNVFP4Linear], rows: Int, dtype: DType
    ) -> Plan? {
        let inputs = layers[0].inputDimensions
        guard rows > ModelOptWeightOnlyNVFP4RowExact.scalarRows,
            ModelOptWeightOnlyNVFP4RowExact.supports(rows: rows, inputs: inputs, dtype: dtype)
        else { return nil }
        for layer in layers {
            let outputs = layer.outputDimensions
            guard layer.inputDimensions == inputs, outputs > 0,
                layer.weight.dtype == .uint32, layer.weight.shape == [outputs, inputs / 8],
                layer.weight.size >= 8, layer.weightScale.dtype == .uint8,
                layer.weightScale.shape == [outputs, inputs / 16], layer.weightScale.size >= 8,
                layer.weightScale2.dtype == .float32, layer.weightScale2.size == 1
            else { return nil }
        }
        return .nvfp4(layers)
    }

    private static func fp8Plan(
        _ layers: [ModelOptCalibratedLinear], rows: Int, dtype: DType
    ) -> Plan? {
        let inputs = layers[0].inputDimensions
        guard inputs > 64, ModelOptRowExact.geometryOverride == nil else { return nil }
        for layer in layers {
            guard layer.format == .fp8, layer.execution == .graph, layer.weightDType == dtype,
                layer.inputDimensions == inputs,
                layer.rowExactSchedule == ModelOptRowExact.Schedule.windows,
                let constants = layer.graphConstants,
                case .fp8 = constants.storage,
                layer.weight.dtype == .uint8,
                layer.weight.shape == [layer.outputDimensions, inputs],
                constants.outputScale.size == 1, constants.outputScale.dtype == .float32,
                layer.outputDimensions >= 2
            else { return nil }
            // One dispatch of the single-projection path: every one-row eager
            // tile shares MLX's K split (K >= 16 * tile rows), so one run
            // with 8 simdgroups across K, which is this kernel's geometry.
            let tiles = layer.fp8EagerTiles(rows: 1)
            guard !tiles.isEmpty, tiles.allSatisfy({ inputs >= 16 * $0.count }) else {
                return nil
            }
        }
        return .fp8(layers)
    }

    /// Whether `products` takes this group of `x`'s rows.
    public static func canGroup(_ layers: [any UnaryLayer], _ x: MLXArray) -> Bool {
        plan(layers, rows: x.dim(-1) == 0 ? 0 : x.size / x.dim(-1), dtype: x.dtype) != nil
    }

    /// The products of every layer with `x` (`[..., K]`, 2 to 8 rows), in
    /// order, each equal to the layer's own row-exact product; nil when the
    /// group is not one this file runs in one dispatch.
    public static func products(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray]? {
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        guard let plan = plan(layers, rows: rows, dtype: x.dtype) else { return nil }
        let flat = x.reshaped(rows, inputs)
        let count = layers.count
        let lead = Array(x.shape.dropLast())
        switch plan {
        case .nvfp4(let group):
            var arrays = [MLXArray]()
            var template: [(String, any KernelTemplateArg)] = [
                ("T", x.dtype), ("V", rows), ("K", inputs),
            ]
            var shapes = [[Int]]()
            var tiles = 0
            for (i, layer) in group.enumerated() {
                arrays += [layer.weight, layer.weightScale, layer.weightScale2.reshaped(1)]
                template.append(("N\(i)", layer.outputDimensions))
                shapes.append([rows, layer.outputDimensions])
                tiles += (layer.outputDimensions + 31) / 32
            }
            arrays.append(flat)
            let results = nvfp4Kernels[count]!(
                arrays, template: template, grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                outputShapes: shapes, outputDTypes: Array(repeating: x.dtype, count: count))
            return zip(results, group).map {
                $0.reshaped(lead + [$1.outputDimensions])
            }
        case .fp8(let group):
            var castInputs: [MLXArray] = [flat]
            var arrays = [MLXArray]()
            var template: [(String, any KernelTemplateArg)] = [
                ("T", x.dtype), ("V", rows), ("K", inputs),
            ]
            var shapes = [[Int]]()
            var groups = 0
            for (i, layer) in group.enumerated() {
                guard let constants = layer.graphConstants,
                    case .fp8(let multipliers, let bound) = constants.storage
                else { return nil }
                castInputs += [multipliers, bound]
                arrays += [layer.weight, constants.outputScale]
                template.append(("N\(i)", layer.outputDimensions))
                shapes.append([rows, layer.outputDimensions])
                groups += (layer.outputDimensions + 1) / 2
            }
            let total = count * rows * inputs
            let cast = castKernels[count]!(
                castInputs, template: [("T", x.dtype), ("COUNT", rows * inputs)],
                grid: (total, 1, 1), threadGroup: (min(256, total), 1, 1),
                outputShapes: [[total]], outputDTypes: [x.dtype])[0]
            arrays.append(cast)
            let results = fp8Kernels[count]!(
                arrays, template: template, grid: (groups * 32, 8, 1), threadGroup: (32, 8, 1),
                outputShapes: shapes, outputDTypes: Array(repeating: x.dtype, count: count))
            return zip(results, group).map {
                $0.reshaped(lead + [$1.outputDimensions])
            }
        }
    }

    /// Distinct shape of a group, for warm-up de-duplication; nil if not groupable
    /// at 2 rows.
    public static func shapeKey(_ layers: [any UnaryLayer], dtype: DType) -> String? {
        guard let plan = (2 ... maximumRows).lazy.compactMap({ plan(layers, rows: $0, dtype: dtype) }).first
        else { return nil }
        switch plan {
        case .nvfp4(let g):
            return "nvfp4 \(g[0].inputDimensions) " + g.map { String($0.outputDimensions) }.joined(separator: ",")
        case .fp8(let g):
            return "fp8 \(g[0].inputDimensions) " + g.map { String($0.outputDimensions) }.joined(separator: ",")
        }
    }

    /// Runs the group on zero activations at every row count in `rows` so MLX
    /// compiles each kernel before a verify window needs it.
    public static func warm(_ layers: [any UnaryLayer], dtype: DType, rows: ClosedRange<Int>) {
        guard let inputs = (layers.first as? ModelOptWeightOnlyNVFP4Linear)?.inputDimensions
            ?? (layers.first as? ModelOptCalibratedLinear)?.inputDimensions
        else { return }
        for count in rows {
            let zeros = MLXArray.zeros([1, count, inputs], dtype: dtype)
            if let outputs = products(layers, zeros) { eval(outputs) }
        }
    }
}
