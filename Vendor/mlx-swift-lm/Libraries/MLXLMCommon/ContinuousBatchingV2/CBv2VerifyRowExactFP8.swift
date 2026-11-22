import Foundation
import MLX
import MLXNN

/// Row-exact FP8 products for verify windows of 2 to 8 rows. Each output row
/// is bit-identical to `ModelOptCalibratedLinear.rowExactForward` (the
/// `.windows` schedule) and so to the one-row product of that row: the same
/// one-row eager tiles and K split, the same per-lane chain of FP32 sums, the
/// same simd shuffle tree and threadgroup sum order, the same weight decode
/// and final cast. Only the schedule changes: each weight is decoded once per
/// threadgroup through a 256-entry table holding the stock decode of every
/// code, and one activation load serves `TM` output rows. Sibling projections
/// of one activation (`CBv2GroupedProjection`'s groups) run in one cast and
/// one dispatch, each segment with its own weights, scale and cast.
public enum CBv2VerifyRowExactFP8 {
    /// Off: every call returns nil (the stock path).
    /// `DARKBLOOM_MUSE_GLIMMER_VERIFY_FP8=0` turns it off.
    nonisolated(unsafe) public static var enabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_MUSE_GLIMMER_VERIFY_FP8"] != "0"
    /// Benchmarks only: the layout of every shape in place of `plan`.
    nonisolated(unsafe) public static var geometryOverride: Geometry?

    public static let rowRange = 2 ... 8

    /// How a dispatch lays out work; the per-output order never depends on it.
    public struct Geometry: Sendable {
        /// Output rows per thread (TM) and simdgroup sets per threadgroup (BM).
        public var outputs, groups: Int

        public init(outputs: Int, groups: Int) {
            (self.outputs, self.groups) = (outputs, groups)
        }
    }

    static func plan(splitsK: Bool, outputs: Int) -> Geometry {
        geometryOverride ?? Geometry(outputs: 4, groups: 1)
    }

    /// `layer.rowExactForward(x)` for 2 to 8 rows of a projection whose shape
    /// `prepare` admitted, or nil (the caller then takes the stock path).
    public static func product(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray? {
        guard enabled, let operands = Operands(layer, x),
            Admission.admitted(operands.key) == true
        else { return nil }
        return operands.product(x)
    }

    /// `product` without admission (tests and benchmarks).
    public static func uncheckedProduct(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray? {
        Operands(layer, x)?.product(x)
    }

    /// `layers.map { $0.rowExactForward(x) }` in one cast and one dispatch for
    /// a group `prepare` admitted, or nil.
    public static func groupedProducts(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray]? {
        guard enabled, let group = Group(layers, x), Admission.admitted(group.key) == true
        else { return nil }
        return group.products(x)
    }

    /// `groupedProducts` without admission (tests and benchmarks).
    public static func uncheckedGroupedProducts(_ layers: [any UnaryLayer], _ x: MLXArray)
        -> [MLXArray]?
    {
        Group(layers, x)?.products(x)
    }

    /// Before the first verify window: on the first projection (or group) of
    /// each shape, every row count; on every other one, 8 rows; each against
    /// the stock product on its own weights, byte for byte. This compiles every
    /// template. A shape is admitted only if all its comparisons match. A group
    /// this file cannot run in one dispatch is checked member by member, as
    /// `layers` are. Once per `owner`.
    public static func prepare(
        _ layers: [any UnaryLayer], groups: [[any UnaryLayer]] = [], owner: ObjectIdentifier
    ) {
        guard enabled, Admission.firstPreparation(of: owner) else { return }
        var singles = layers
        var groupShapes = [String: Bool]()
        var checkedGroups = 0
        for members in groups {
            guard let first = members.first as? ModelOptCalibratedLinear,
                let group = Group(
                    members, probe(rows: 8, inputs: first.inputDimensions, dtype: first.weightDType))
            else {
                singles += members
                continue
            }
            let known = Admission.admitted(group.key)
            if known != false {
                Admission.record(group.key, group.check(rows: known == nil ? Array(rowRange) : [8]))
                checkedGroups += 1
            }
            groupShapes[group.key] = Admission.admitted(group.key)
        }
        var shapes = [String: Bool]()
        var slots = 0
        for layer in singles {
            guard let fp8 = layer as? ModelOptCalibratedLinear,
                let operands = Operands(
                    fp8, probe(rows: 8, inputs: fp8.inputDimensions, dtype: fp8.weightDType))
            else { continue }
            let known = Admission.admitted(operands.key)
            if known != false {
                Admission.record(
                    operands.key, operands.check(rows: known == nil ? Array(rowRange) : [8]))
                slots += 1
            }
            shapes[operands.key] = Admission.admitted(operands.key)
        }
        let line = "[cbv2-verify-fp8] checked \(slots) projections, \(checkedGroups) groups; "
            + "admitted \(shapes.values.filter { $0 }.count) of \(shapes.count) shapes, "
            + "\(groupShapes.values.filter { $0 }.count) of \(groupShapes.count) group shapes\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    public static func resetAdmission() { Admission.reset() }

    struct Operands {
        let layer: ModelOptCalibratedLinear
        let multipliers: MLXArray
        let overflowBound: MLXArray
        let scale: MLXArray
        let runs: [(rows: Range<Int>, splitsK: Bool)]

        init?(_ layer: any UnaryLayer, _ x: MLXArray) {
            guard let layer = layer as? ModelOptCalibratedLinear, layer.format == .fp8,
                layer.execution == .graph, layer.rowExactSchedule == .windows,
                layer.inputDimensions > 64, layer.inputDimensions % 4 == 0,
                layer.weightDType == x.dtype, x.ndim >= 1, x.dim(-1) == layer.inputDimensions,
                CBv2VerifyRowExactFP8.rowRange.contains(x.size / layer.inputDimensions),
                let constants = layer.graphConstants,
                case .fp8(let multipliers, let overflowBound) = constants.storage,
                constants.outputScale.size == 1, constants.outputScale.dtype == .float32,
                layer.weight.dtype == .uint8,
                layer.weight.shape == [layer.outputDimensions, layer.inputDimensions]
            else { return nil }
            self.layer = layer
            self.multipliers = multipliers
            self.overflowBound = overflowBound
            self.scale = constants.outputScale
            // The stock one-row runs: adjacent eager tiles that share MLX's K split.
            var runs: [(rows: Range<Int>, splitsK: Bool)] = []
            for tile in layer.fp8EagerTiles(rows: 1) {
                let splitsK = layer.inputDimensions >= 16 * tile.count
                if let last = runs.last, last.splitsK == splitsK {
                    runs[runs.count - 1].rows = last.rows.lowerBound ..< tile.upperBound
                } else {
                    runs.append((tile, splitsK))
                }
            }
            self.runs = runs
        }

        var key: String {
            "\(layer.weightDType) \(layer.inputDimensions) \(layer.outputDimensions)"
        }

        func product(_ x: MLXArray) -> MLXArray {
            let k = layer.inputDimensions
            let rows = x.size / k
            // The `.windows` schedule's separate activation cast.
            let vectors = ModelOptActivationKernels.fp8(
                x.reshaped(rows, k), multipliers: multipliers, overflowBound: overflowBound)
            let pieces = runs.map { run in
                CBv2VerifyRowExactFP8.dispatch(
                    layer.weight[run.rows, 0...], scale: scale, vectors: vectors,
                    splitsK: run.splitsK)
            }
            let joined = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
            return joined.reshaped(Array(x.shape.dropLast()) + [layer.outputDimensions])
        }

        func check(rows: [Int]) -> Bool {
            rows.allSatisfy { rows in
                let x = probe(rows: rows, inputs: layer.inputDimensions, dtype: layer.weightDType)
                guard let stock = try? layer.rowExactForward(x) else { return false }
                return bitwiseEqual(product(x), stock)
            }
        }
    }

    /// Two or four projections of one activation whose one-row products are
    /// each a single K-split run over all outputs: each segment runs the
    /// single-projection kernel's body with its own weights, scale and cast.
    struct Group {
        let members: [Operands]

        init?(_ layers: [any UnaryLayer], _ x: MLXArray) {
            guard layers.count == 2 || layers.count == 4 else { return nil }
            var members = [Operands]()
            for layer in layers {
                guard let operands = Operands(layer, x), operands.runs.count == 1,
                    operands.runs[0].splitsK,
                    operands.runs[0].rows == 0 ..< operands.layer.outputDimensions,
                    operands.layer.outputDimensions >= 4,
                    operands.layer.inputDimensions == members.first?.layer.inputDimensions
                        ?? operands.layer.inputDimensions
                else { return nil }
                members.append(operands)
            }
            self.members = members
        }

        var key: String { "group " + members.map(\.key).joined(separator: " | ") }

        func products(_ x: MLXArray) -> [MLXArray] {
            let k = members[0].layer.inputDimensions
            let rows = x.size / k
            let count = members.count
            let outputs = members.map(\.layer.outputDimensions)
            // Each segment's `.windows` cast with its own multipliers and bound.
            var castInputs = [x.reshaped(rows, k)]
            for m in members { castInputs += [m.multipliers, m.overflowBound] }
            let total = count * rows * k
            let cast = CBv2GroupedProjection.castKernels[count]!(
                castInputs, template: [("T", x.dtype), ("COUNT", rows * k)],
                grid: (total, 1, 1), threadGroup: (min(256, total), 1, 1),
                outputShapes: [[total]], outputDTypes: [x.dtype])[0]
            let g = plan(splitsK: true, outputs: outputs.min()!)
            let tm = min(g.outputs, outputs.min()!)
            let bm = max(1, g.groups)
            var mp = 1
            while mp < rows * tm { mp *= 2 }
            var arrays = [MLXArray]()
            var template: [(String, any KernelTemplateArg)] = [
                ("T", x.dtype), ("BM", bm), ("TM", tm), ("K", k), ("V", rows), ("MP", mp),
            ]
            var groups = 0
            for (i, m) in members.enumerated() {
                arrays += [m.layer.weight, m.scale]
                template.append(("N\(i)", outputs[i]))
                groups += (outputs[i] + bm * tm - 1) / (bm * tm)
            }
            arrays.append(cast)
            let results = groupKernels[count]!(
                arrays, template: template, grid: (groups * 32, 8, bm),
                threadGroup: (32, 8, bm), outputShapes: outputs.map { [rows, $0] },
                outputDTypes: Array(repeating: x.dtype, count: count))
            let lead = Array(x.shape.dropLast())
            return zip(results, outputs).map { $0.reshaped(lead + [$1]) }
        }

        func check(rows: [Int]) -> Bool {
            let k = members[0].layer.inputDimensions
            let dtype = members[0].layer.weightDType
            return rows.allSatisfy { rows in
                let x = probe(rows: rows, inputs: k, dtype: dtype)
                let new = products(x)
                return zip(members, new).allSatisfy { m, p in
                    guard let stock = try? m.layer.rowExactForward(x) else { return false }
                    return bitwiseEqual(p, stock)
                }
            }
        }
    }

    /// `[V, N]` products of cast `vectors` (`[V, K]`) with E4M3 rows `mat`
    /// (`[N, K]`) times `scale`; `splitsK` is the one-row call's K split.
    static func dispatch(_ mat: MLXArray, scale: MLXArray, vectors: MLXArray, splitsK: Bool)
        -> MLXArray
    {
        let (outputs, inputs) = (mat.dim(0), mat.dim(1))
        let rows = vectors.dim(0)
        let bn = splitsK ? 8 : 1
        let g = plan(splitsK: splitsK, outputs: outputs)
        let tm = min(g.outputs, outputs)
        let bm = max(1, min(g.groups, (outputs + tm - 1) / tm))
        let groups = (outputs + bm * tm - 1) / (bm * tm)
        var mp = 1
        while mp < rows * tm { mp *= 2 }
        return kernel(
            [mat, scale, vectors],
            template: [
                ("T", vectors.dtype), ("BM", bm), ("BN", bn), ("TM", tm), ("K", inputs),
                ("N", outputs), ("V", rows), ("MP", mp),
            ],
            grid: (groups * 32, bn, bm), threadGroup: (32, bn, bm),
            outputShapes: [[rows, outputs]], outputDTypes: [vectors.dtype])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_verify_row_exact_fp8",
        inputNames: ["mat", "scale", "in_vec"],
        outputNames: ["out_vec"],
        source: """
            threadgroup float table[256];
            threadgroup float parts[BN > 1 ? BM * BN * MP : 1];
            cbv2v_product<T, BM, BN, TM, K, N, V, MP>(
                mat, scale[0], in_vec, out_vec, int(threadgroup_position_in_grid.x),
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, table, parts);
            """,
        header: header)

    /// Segment `i` takes threadgroups `G0 + ... + G(i-1) ..< ... + Gi`, with
    /// `Gi = ceil(Ni / (BM * TM))`; its cast rows start at `i * V * K` in `xc`.
    private static func groupKernel(segments: Int) -> MLXFast.MLXFastKernel {
        var inputs = [String](), outputs = [String]()
        var body = """
            threadgroup float table[256];
            threadgroup float parts[BM * 8 * MP];
            const int g = int(threadgroup_position_in_grid.x);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = "0"
        for i in 0 ..< segments {
            inputs += ["m\(i)", "ws\(i)"]
            outputs.append("o\(i)")
            let groups = "((N\(i) + BM * TM - 1) / (BM * TM))"
            body += """
                \(i == 0 ? "" : "else ")if (g < \(offset) + \(groups)) {
                    cbv2v_product<T, BM, 8, TM, K, N\(i), V, MP>(
                        m\(i), ws\(i)[0], xc + \(i) * V * K, o\(i), g - (\(offset)), sg, sl,
                        table, parts);
                }

                """
            offset += " + \(groups)"
        }
        inputs.append("xc")
        return MLXFast.metalKernel(
            name: "cbv2_verify_row_exact_fp8_group_\(segments)", inputNames: inputs,
            outputNames: outputs, source: body, header: header)
    }

    nonisolated(unsafe) private static let groupKernels: [Int: MLXFast.MLXFastKernel] = [
        2: groupKernel(segments: 2), 4: groupKernel(segments: 4),
    ]

    // Per output and row: lane l of K slice s (BN slices of 32 lanes) sums
    // inter * vector over K positions i * BN * 128 + s * 128 + 4 l + tn, i then
    // tn ascending; then the simd_shuffle_down 16 to 1 tree; then slice 0 adds
    // slices 1 ..< BN in order; then one cast to T: `modelopt_row_exact_gemv`'s
    // order. Unrolling and loading the next weights early move no sum.
    private static let header = ModelOptActivationKernels.header + """

        #define CBV2V_UNROLL _Pragma("clang loop unroll(full)")

        // One level (distance d) of the shuffle-down tree for LIVE
        // accumulators: the lane with bit d clear adds its partner's value
        // to its own (lower lane first, as the tree does) for the lower half,
        // the other lane for the upper half; one live value takes the plain
        // shuffle-down.
        template <int LIVE>
        inline void cbv2v_level(thread float* acc, uint lane, ushort d) {
            if (LIVE == 1) {
                acc[0] += simd_shuffle_down(acc[0], d);
            } else {
                constexpr int H = LIVE / 2;
                const bool upper = (lane & d) != 0;
                for (int j = 0; j < H; j++) {
                    const float other = simd_shuffle_xor(upper ? acc[j] : acc[j + H], d);
                    acc[j] = upper ? other + acc[j + H] : acc[j] + other;
                }
            }
        }

        // Threadgroup `group` of one product; `table` holds 256 floats and
        // `parts` BM * BN * MP (BN > 1).
        template <typename T, int BM, int BN, int TM, int K, int N, int V, int MP>
        inline void cbv2v_product(
            const device uint8_t* mat, const float weight_scale, const device T* in_vec,
            device T* out_vec, const int group, const uint simd_gid, const uint simd_lid,
            threadgroup float* table, threadgroup float* parts) {
            constexpr int SN = 32;
            constexpr int TN = 4;
            constexpr int blockM = BM * TM;
            constexpr int blockN = BN * SN * TN;
            constexpr int n_iter = K / blockN;
            constexpr int leftover = K - blockN * n_iter;
            constexpr int M = V * TM;
            constexpr int PER = MP >= 32 ? MP / 32 : 1;
            constexpr int SPAN = MP >= 32 ? 1 : 32 / MP;
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            // The stock decode of every code, rounded to T as the stock keeps it.
            for (uint c = simd_gid * 32 + simd_lid; c < 256; c += BN * BM * 32) {
                const uint8_t code = static_cast<uint8_t>(c);
                const float decoded =
                    (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                table[c] = static_cast<float>(static_cast<T>(decoded * weight_scale));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int sgN = BN > 1 ? int(simd_gid % BN) : 0;
            const int simdM = BN > 1 ? int(simd_gid / BN) : int(simd_gid);
            int bn = (sgN * SN + int(simd_lid)) * TN;
            int out_row = group * blockM + simdM * TM;
            const bool live = out_row < N;
            out_row = out_row + TM <= N ? out_row : N - TM;
            const device uint8_t* rows = mat + size_t(out_row) * K;
            thread float acc[MP];
            for (int j = 0; j < MP; j++) {
                acc[j] = 0;
            }
            thread float inter[TM][TN];
            // K is a multiple of 4 and bn of TN, so a lane's leftover is four
            // positions inside K or four past it. Past K the stock adds +0 * +0
            // to an accumulator that started at +0 and so is never -0: an exact
            // no-op, skipped.
            const int iters = n_iter + (leftover > 0 && bn + n_iter * blockN < K ? 1 : 0);
            uint packed[TM];
            if (iters > 0) {
                CBV2V_UNROLL for (int tm = 0; tm < TM; tm++) {
                    packed[tm] = *reinterpret_cast<const device uint*>(rows + tm * K + bn);
                }
            }
            for (int i = 0; i < iters; ++i) {
                CBV2V_UNROLL for (int tm = 0; tm < TM; tm++) {
                    CBV2V_UNROLL for (int tn = 0; tn < TN; tn++) {
                        inter[tm][tn] = table[(packed[tm] >> (8 * tn)) & 0xFF];
                    }
                }
                if (i + 1 < iters) {
                    CBV2V_UNROLL for (int tm = 0; tm < TM; tm++) {
                        packed[tm] =
                            *reinterpret_cast<const device uint*>(rows + tm * K + bn + blockN);
                    }
                }
                CBV2V_UNROLL for (int v = 0; v < V; v++) {
                    const float4 a =
                        float4(*reinterpret_cast<const device vec<T, 4>*>(in_vec + v * K + bn));
                    CBV2V_UNROLL for (int tm = 0; tm < TM; tm++) {
                        CBV2V_UNROLL for (int tn = 0; tn < TN; tn++) {
                            acc[v * TM + tm] += inter[tm][tn] * a[tn];
                        }
                    }
                }
                bn += blockN;
            }
            // The shuffle-down tree for every accumulator, spread over the
            // lanes (MP accumulators, padded to a power of two).
            cbv2v_level<MP>(acc, simd_lid, 16);
            cbv2v_level<(MP > 2 ? MP / 2 : 1)>(acc, simd_lid, 8);
            cbv2v_level<(MP > 4 ? MP / 4 : 1)>(acc, simd_lid, 4);
            cbv2v_level<(MP > 8 ? MP / 8 : 1)>(acc, simd_lid, 2);
            cbv2v_level<(MP > 16 ? MP / 16 : 1)>(acc, simd_lid, 1);
            const bool writer = int(simd_lid) % SPAN == 0;
            const int first = int(simd_lid) / SPAN * PER;
            if (BN > 1) {
                threadgroup float* p = parts + simdM * (BN * MP);
                if (writer) {
                    for (int j = 0; j < PER; j++) {
                        p[sgN * MP + first + j] = acc[j];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgN == 0 && writer) {
                    for (int j = 0; j < PER; j++) {
                        for (int sgn = 1; sgn < BN; sgn++) {
                            acc[j] += p[sgn * MP + first + j];
                        }
                    }
                }
            }
            if (live && sgN == 0 && writer) {
                for (int j = 0; j < PER; j++) {
                    const int index = first + j;
                    if (index < M) {
                        out_vec[(index / TM) * N + out_row + index % TM] = static_cast<T>(acc[j]);
                    }
                }
            }
        }
        """

    // MARK: - Self-check

    /// Per shape: nil before `prepare` saw it, then whether every comparison matched.
    enum Admission {
        nonisolated(unsafe) private static var results = [String: Bool]()
        private static let lock = NSLock()

        static func admitted(_ key: String) -> Bool? { lock.withLock { results[key] } }

        static func record(_ key: String, _ ok: Bool) {
            lock.withLock { results[key] = (results[key] ?? true) && ok }
        }

        nonisolated(unsafe) private static var owners = Set<ObjectIdentifier>()

        static func firstPreparation(of owner: ObjectIdentifier) -> Bool {
            lock.withLock { owners.insert(owner).inserted }
        }

        static func reset() {
            lock.withLock {
                results.removeAll()
                owners.removeAll()
            }
        }
    }

    /// Signed values over a wide exponent range, exact zeros, negative zeros
    /// and values past the FP8 activation bound.
    static func probe(rows: Int, inputs: Int, dtype: DType) -> MLXArray {
        let index = MLX.arange(rows * inputs, dtype: .int32).reshaped(rows, inputs)
        let phase = sin(index.asType(.float32) * 0.7548776662)
        var values = exp(phase * 8.3177661667)
        values = MLX.where((index % 3) .== 0, -values, values)
        values = MLX.where((index % 97) .== 5, MLXArray(Float(0)), values)
        values = MLX.where((index % 89) .== 7, MLXArray(-Float(0)), values)
        values = MLX.where((index % 101) .== 11, MLXArray(Float(1e4)), values)
        return values.asType(dtype)
    }

    static func bitwiseEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        return (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all().item(Bool.self)
    }
}
