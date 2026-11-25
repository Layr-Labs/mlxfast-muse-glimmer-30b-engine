// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// The weight-only NVFP4 row-exact product of 1 to 3 verify rows with scalar
/// fused multiply-add chains instead of 8x8x8 multiply-accumulates.
///
/// Every output element runs the order `ModelOptWeightOnlyNVFP4RowExact`
/// defines: the same weight decode (`mo_wo_pair`, paired e4m3 scales), the
/// same eight K partitions (chunk `q` of 128 inputs goes to partition
/// `q % 8`, ascending), blocks of 32 inputs in ascending order with inputs
/// `b + 4c + i` (`i` outside, `c` inside) as `acc = fma(w, x, acc)` from
/// zero, the join `((acc0 + acc1) + ...) + acc7` through threadgroup memory
/// and the same rounding and tensor scale. The matrix kernel's
/// multiply-accumulate equals that ascending chain only on the GPU
/// generations `ModelOptWeightOnlyNVFP4RowExact.scalarKernelMeasured` names,
/// so only those are taken.
///
/// Schedule: a threadgroup of 4 simdgroups covers 32 outputs; each
/// simdgroup runs 2 partitions, lane `(sub, rl)` runs partition
/// `2 * sg + sub` for outputs `rl` and `rl + 16` of the tile, and each
/// partition's 128-input activation chunk is staged once as float in
/// threadgroup memory. Measured on the M5 Ultra against the 8-row tiles
/// (micro, chained): MLP gate+up -18 / -13 / -9% at 1 / 2 / 3 rows, down
/// -22 / -18 / -8%, attention output -16 / -10 / +2%; from 4 rows on the
/// matrix kernels win, so callers route only up to 3 rows.
/// `MUSE_VROW=0` (or `DARKBLOOM_MUSE_GLIMMER_VERIFY_VROW=0`) turns it off and
/// restores the 8-row tile routes.
public enum CBv2VerifyRows {
    public static let enabled =
        ProcessInfo.processInfo.environment["MUSE_VROW"] != "0"
        && ProcessInfo.processInfo.environment["DARKBLOOM_MUSE_GLIMMER_VERIFY_VROW"] != "0"
    public static let maximumRows = 3

    static let tileHelper = """

        // One tile of R = (32 / P) * RS outputs of one projection, V activation
        // rows. Simdgroup sg runs partitions sg * P ... sg * P + P - 1; lane
        // (sub, rl) runs partition sg * P + sub for rows tile * R + rl + RL * r.
        template <typename T, int V, int K, int N, int RS, int P, bool SCALED>
        inline void cbv2_vrow_tile(
            const device uint32_t* w, const device uint8_t* scales, const device T* x,
            float tensor_scale0, device T* y, int tile, uint sg, uint lane,
            threadgroup float* buf) {
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr int RL = 32 / P;
            constexpr int R = RL * RS;
            constexpr int NCH = K / BK;
            constexpr int XS = V * BK + 4;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            static_assert(K % BK == 0, "whole chunks only");
            const int sub = int(lane) / RL;
            const int rl = int(lane) % RL;
            const int part = int(sg) * P + sub;
            const threadgroup float* xp = buf + part * XS;
            const int row0 = tile * R + rl;

            const device uint4* wr[RS];
            const device ushort* sr16[RS];
            float acc[V][RS];
            MO_WO_UNROLL
            for (int r = 0; r < RS; r++) {
                const int row = min(row0 + r * RL, N - 1);
                wr[r] = (const device uint4*)((const device uint8_t*)w + size_t(row) * row_bytes);
                sr16[r] = (const device ushort*)(scales + size_t(row) * row_scales);
                MO_WO_UNROLL
                for (int v = 0; v < V; v++) {
                    acc[v][r] = 0.0f;
                }
            }

            for (int q0 = int(sg) * P; q0 < NCH; q0 += WK) {
                // Stage this step's chunk of each of the simdgroup's partitions:
                // input (block bl, column c, element e) at e * 32 + bl * 8 + c.
                simdgroup_barrier(mem_flags::mem_threadgroup);
                MO_WO_UNROLL
                for (int sp = 0; sp < P; sp++) {
                    const int qs = q0 + sp;
                    if (qs < NCH) {
                        threadgroup float* d = buf + (int(sg) * P + sp) * XS + lane;
                        MO_WO_UNROLL
                        for (int v = 0; v < V; v++) {
                            const float4 f = mo_wo_load4(x + size_t(v) * K + qs * BK + 4 * lane);
                            MO_WO_UNROLL
                            for (int e = 0; e < 4; e++) {
                                d[v * BK + e * 32] = f[e];
                            }
                        }
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                const int q = q0 + sub;
                if (q < NCH) {
                    // One block's words and scale pair per row at a time.
                    #pragma clang loop unroll(disable)
                    for (int bl = 0; bl < 4; bl++) {
                        uint4 wb[RS];
                        half s0[RS];
                        half s1[RS];
                        MO_WO_UNROLL
                        for (int r = 0; r < RS; r++) {
                            wb[r] = wr[r][q * 4 + bl];
                            const uint sp = sr16[r][q * 4 + bl];
                            const half2 sd = cb_vr_e4m3x2((sp & 0xFFu) | (sp << 8));
                            s0[r] = sd.x;
                            s1[r] = sd.y;
                        }
                        const threadgroup float* xb = xp + bl * 8;
                        MO_WO_UNROLL
                        for (int i = 0; i < 4; i++) {
                            // wf[r][c]: the weight of input b + 4c + i.
                            float wf[RS][8];
                            MO_WO_UNROLL
                            for (int r = 0; r < RS; r++) {
                                MO_WO_UNROLL
                                for (int j = 0; j < 4; j++) {
                                    const float2 wv = mo_wo_pair(wb[r][j], i, j < 2 ? s0[r] : s1[r]);
                                    wf[r][2 * j] = wv.x;
                                    wf[r][2 * j + 1] = wv.y;
                                }
                            }
                            MO_WO_UNROLL
                            for (int v = 0; v < V; v++) {
                                const float4 xa = *(const threadgroup float4*)(xb + v * BK + i * 32);
                                const float4 xc = *(const threadgroup float4*)(xb + v * BK + i * 32 + 4);
                                MO_WO_UNROLL
                                for (int r = 0; r < RS; r++) {
                                    MO_WO_UNROLL
                                    for (int c = 0; c < 8; c++) {
                                        acc[v][r] = fma(wf[r][c], c < 4 ? xa[c] : xc[c - 4], acc[v][r]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Join: ((acc_0 + acc_1) + ...) + acc_7 per output, as the stock kernels.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float* jb = buf;
            MO_WO_UNROLL
            for (int v = 0; v < V; v++) {
                MO_WO_UNROLL
                for (int r = 0; r < RS; r++) {
                    jb[(part * V + v) * R + r * RL + rl] = acc[v][r];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            constexpr int THREADS = 32 * (8 / P);
            const int tid = int(sg) * 32 + int(lane);
            for (int o = tid; o < V * R; o += THREADS) {
                const int v = o / R;
                const int rloc = o % R;
                float total = jb[v * R + rloc];
                for (int s = 1; s < WK; s++) {
                    total += jb[(s * V + v) * R + rloc];
                }
                const int row = tile * R + rloc;
                if (row < N) {
                    y[size_t(v) * N + row] = mo_wo_result<T>(total, SCALED, tensor_scale0);
                }
            }
        }
        """

    // Paired e4m3 scale decode, the same bits as `mo_wo_e4m3` per half.
    static let scaleHelper = """

        inline half2 cb_vr_e4m3x2(uint x) {
            const uint v = x & 0x007F007Fu;
            const uint u = (v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7)
                | ((x & 0x00800080u) << 8);
            return as_type<half2>(u) * half2(256.0h);
        }

        """

    static let rowsPerLane = 2
    static let partitionsPerSimdgroup = 2
    static let rowsPerTile = (32 / partitionsPerSimdgroup) * rowsPerLane

    /// Segment `i` takes tiles `[sum of tiles before, + ceil(N_i / R))` of the
    /// grid's y axis.
    private static func kernel(segments: Int) -> MLXFast.MLXFastKernel {
        var inputs = [String](), outputs = [String]()
        var body = """
            constexpr int R = (32 / P) * RS;
            constexpr int STAGE = 8 * (V * 128 + 4);
            constexpr int JOIN = 8 * V * R;
            threadgroup float buf[STAGE > JOIN ? STAGE : JOIN];
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = "0"
        for i in 0 ..< segments {
            inputs += ["w\(i)", "sc\(i)", "ts\(i)"]
            outputs.append("y\(i)")
            let tiles = "((N\(i) + R - 1) / R)"
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(offset) + \(tiles)) {
                    cbv2_vrow_tile<T, V, K, N\(i), RS, P, SCALED>(
                        w\(i), sc\(i), x, ts\(i)[0], y\(i), tg - (\(offset)), sg, sl, buf);
                }

                """
            offset += " + \(tiles)"
        }
        inputs.append("x")
        return MLXFast.metalKernel(
            name: "cbv2_verify_rows_nvfp4_\(segments)", inputNames: inputs, outputNames: outputs,
            source: body,
            header: ModelOptWeightOnlyNVFP4RowExact.header + scaleHelper + tileHelper)
    }

    nonisolated(unsafe) private static let kernels: [Int: MLXFast.MLXFastKernel] = [
        1: kernel(segments: 1), 2: kernel(segments: 2),
    ]

    nonisolated(unsafe) private static let unit = MLXArray([Float(1)])

    /// Whether `products` takes `rows` rows of `inputs` activations of `dtype`.
    public static func supports(rows: Int, inputs: Int, dtype: DType) -> Bool {
        enabled && dtype == .bfloat16 && (1 ... maximumRows).contains(rows) && inputs > 0
            && inputs % 128 == 0 && ModelOptWeightOnlyNVFP4RowExact.scalarKernelMeasured
            && ModelOptWeightOnlyNVFP4RowExact.supports(rows: rows, inputs: inputs, dtype: dtype)
    }

    private static func takes(_ layer: ModelOptWeightOnlyNVFP4Linear, inputs: Int) -> Bool {
        let outputs = layer.weight.dim(0)
        return outputs > 0 && layer.inputDimensions == inputs && layer.weight.dtype == .uint32
            && layer.weight.shape == [outputs, inputs / 8] && layer.weight.size >= 8
            && layer.weightScale.dtype == .uint8
            && layer.weightScale.shape == [outputs, inputs / 16] && layer.weightScale.size >= 8
            && layer.weightScale2.dtype == .float32 && layer.weightScale2.size == 1
    }

    /// The products of one or two NVFP4 layers of one input (MLP gate and up
    /// as a pair, down or the attention output alone), one launch, each equal
    /// to its layer's own row-exact product; nil when not taken.
    public static func products(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray]? {
        guard (1 ... 2).contains(layers.count), let nv = layers as? [ModelOptWeightOnlyNVFP4Linear]
        else { return nil }
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        guard supports(rows: rows, inputs: inputs, dtype: x.dtype),
            nv.allSatisfy({ takes($0, inputs: inputs) })
        else { return nil }
        var arrays = [MLXArray]()
        var template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("V", rows), ("K", inputs), ("RS", rowsPerLane),
            ("P", partitionsPerSimdgroup), ("SCALED", true),
        ]
        var shapes = [[Int]]()
        var tiles = 0
        for (i, layer) in nv.enumerated() {
            let outputs = layer.weight.dim(0)
            arrays += [layer.weight, layer.weightScale, layer.weightScale2.reshaped(1)]
            template.append(("N\(i)", outputs))
            shapes.append([rows, outputs])
            tiles += (outputs + rowsPerTile - 1) / rowsPerTile
        }
        arrays.append(x.reshaped(rows, inputs))
        let simdgroups = 8 / partitionsPerSimdgroup
        let results = kernels[nv.count]!(
            arrays, template: template, grid: (32, simdgroups * tiles, 1),
            threadGroup: (32, simdgroups, 1), outputShapes: shapes,
            outputDTypes: Array(repeating: x.dtype, count: nv.count))
        let lead = Array(x.shape.dropLast())
        return zip(results, nv).map { $0.reshaped(lead + [$1.weight.dim(0)]) }
    }
}
