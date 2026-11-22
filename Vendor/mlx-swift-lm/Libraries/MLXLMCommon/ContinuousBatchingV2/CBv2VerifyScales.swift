// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// Verify-row projections with the stock row-exact arithmetic on another
/// schedule. Each output element keeps the order of operations and rounding
/// points of `ModelOptWeightOnlyNVFP4RowExact` (NVFP4 matrix kernel) or of
/// `ModelOptRowExact` with the `.windows` schedule (FP8); only which threads
/// compute which outputs and how weights, scales and activations are loaded
/// change.
///
/// NVFP4 (routed configuration: `NB 8`, the other knobs off): eight output
/// rows per threadgroup (stock: 32), and every load of a 128-input chunk
/// (four code words, four scale bytes, eight activation quads per lane)
/// issued before that chunk's multiply-accumulates, which run in the stock
/// order. The block scales are decoded two per 32-bit operation
/// (`cb_sc_e4m3x2`, the same bits as `mo_wo_e4m3`, NaN codes included).
/// The knobs `SV` (one 8-byte scale load), `FS` (float-domain scaling), `PF`
/// (register double buffer), `RG` and `CH` give the same bytes and were
/// measured slower; benchmarks only.
///
/// FP8 (routed for the MLP down projection at 6 to 8 rows): the stock lane
/// chains, shuffle tree and threadgroup join, with one 4-byte weight load and
/// one 8-byte activation load per row and step (`WV`, `AV`) and four row
/// groups per threadgroup on the split-K run (`BM 4`). `PD` (packed decode)
/// gives the same bytes and is benchmarks only.
///
/// The Muse Glimmer fused verify forward routes here
/// (`DARKBLOOM_MUSE_GLIMMER_VERIFY_PROJ`).
public enum CBv2VerifyScales {
    // MARK: NVFP4

    private static let nvfp4Source = """
        constexpr int WK = 8;
        constexpr int BK = 128;
        constexpr int BN = NB;
        constexpr short TN = BN / 8;
        constexpr short TM = (V + 7) / 8;
        constexpr int row_bytes = K / 2;
        constexpr int row_scales = K / 16;
        static_assert(K % BK == 0, "whole chunks only");
        // RG row groups of 8 simdgroups (the partitions) per threadgroup.
        const uint simd_gid = simdgroup_index_in_threadgroup % WK;
        const uint row_group = simdgroup_index_in_threadgroup / WK;
        const uint simd_lid = thread_index_in_simdgroup;
        const short qid = simd_lid / 4;
        const short fm = (qid & 4) + ((simd_lid / 2) % 4);
        const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
        const int y_col = (threadgroup_position_in_grid.z * RG + row_group) * BN;
        // Lanes with fn >= 4 read the odd scale of each block pair.
        const uint sshift = fn >= 4 ? 8u : 0u;

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
        const device uint2* sv[TN];
        MO_WO_UNROLL
        for (short t = 0; t < TN; t++) {
            const int row = min(y_col + fm + 8 * t, N - 1);
            wl[t] = (const device uint*)(
                (const device uint8_t*)w + size_t(row) * row_bytes + 2 * fn);
            sl[t] = scales + size_t(row) * row_scales + fn / 4;
            sv[t] = (const device uint2*)(scales + size_t(row) * row_scales);
        }

        uint qa[4][TN];
        uint sa[TN][2];
        float4 xa[4][TM][2];
        uint qb[4][TN];
        uint sb[TN][2];
        float4 xb[4][TM][2];

        #define CB_SC_LOAD(KK, Q, S, X) \\
            MO_WO_UNROLL for (short block = 0; block < 4; block++) { \\
                const int b = (KK) + 32 * block; \\
                MO_WO_UNROLL for (short t = 0; t < TN; t++) { Q[block][t] = wl[t][b / 8]; } \\
                MO_WO_UNROLL for (short mt = 0; mt < TM; mt++) { \\
                    MO_WO_UNROLL for (short j = 0; j < 2; j++) { X[block][mt][j] = mo_wo_load4(xr[mt][j] + b); } } } \\
            MO_WO_UNROLL for (short t = 0; t < TN; t++) { \\
                if (SV) { \\
                    const uint2 raw = sv[t][(KK) / BK]; \\
                    S[t][0] = (raw.x >> sshift) & 0x00FF00FFu; \\
                    S[t][1] = (raw.y >> sshift) & 0x00FF00FFu; \\
                } else { \\
                    S[t][0] = uint(sl[t][(KK) / 16]) | (uint(sl[t][(KK) / 16 + 2]) << 16); \\
                    S[t][1] = uint(sl[t][(KK) / 16 + 4]) | (uint(sl[t][(KK) / 16 + 6]) << 16); \\
                } }

        #define CB_SC_COMPUTE(Q, S, X) { \\
            half2 hs[TN][2]; \\
            MO_WO_UNROLL for (short t = 0; t < TN; t++) { \\
                MO_WO_UNROLL for (short h = 0; h < 2; h++) { hs[t][h] = SD ? half2(0) : cb_sc_e4m3x2(S[t][h]); } } \\
            MO_WO_UNROLL for (short block = 0; block < 4; block++) { \\
                half s[TN]; \\
                float fs[TN]; \\
                MO_WO_UNROLL for (short t = 0; t < TN; t++) { \\
                    s[t] = SD ? mo_wo_e4m3((S[t][block / 2] >> (16 * (block % 2))) & 0xFFu) : hs[t][block / 2][block % 2]; \\
                    fs[t] = float(s[t]) * 16384.0f; } \\
                MO_WO_UNROLL for (short i = 0; i < 4; i++) { \\
                    float2 bf[TM]; \\
                    MO_WO_UNROLL for (short mt = 0; mt < TM; mt++) { bf[mt] = float2(X[block][mt][0][i], X[block][mt][1][i]); } \\
                    MO_WO_UNROLL for (short t = 0; t < TN; t++) { \\
                        const float2 a = FS ? cb_sc_pair_f(Q[block][t], i, fs[t]) : mo_wo_pair(Q[block][t], i, s[t]); \\
                        MO_WO_UNROLL for (short mt = 0; mt < TM; mt++) { mo_wo_mma(C[t][mt], a, bf[mt]); } } } } }

        if (PF) {
            int k = simd_gid * BK;
            if (k < K) { CB_SC_LOAD(k, qa, sa, xa) }
            for (; k < K; k += 2 * WK * BK) {
                const int k1 = k + WK * BK;
                if (k1 < K) { CB_SC_LOAD(k1, qb, sb, xb) }
                CB_SC_COMPUTE(qa, sa, xa)
                if (k1 >= K) { break; }
                const int k2 = k1 + WK * BK;
                if (k2 < K) { CB_SC_LOAD(k2, qa, sa, xa) }
                CB_SC_COMPUTE(qb, sb, xb)
            }
        } else if (CH == 2) {
            // Two chunks' loads in flight before either chunk's arithmetic.
            for (int k = simd_gid * BK; k < K; k += 2 * WK * BK) {
                const bool second = k + WK * BK < K;
                CB_SC_LOAD(k, qa, sa, xa)
                if (second) { CB_SC_LOAD(k + WK * BK, qb, sb, xb) }
                CB_SC_COMPUTE(qa, sa, xa)
                if (second) { CB_SC_COMPUTE(qb, sb, xb) }
            }
        } else {
            for (int k = simd_gid * BK; k < K; k += WK * BK) {
                CB_SC_LOAD(k, qa, sa, xa)
                CB_SC_COMPUTE(qa, sa, xa)
            }
        }

        constexpr int tile_elems = TN * TM * 2;
        threadgroup float partials_all[RG * (WK - 1) * tile_elems * 32];
        threadgroup float* partials = partials_all + row_group * (WK - 1) * tile_elems * 32;
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
                                yr[out] = mo_wo_result<T>(C[t][mt][j], SCALED, tensor_scale[0]);
                            }
                        }
                    }
                }
            }
        }
        """

    private static let nvfp4Header = ModelOptWeightOnlyNVFP4RowExact.header + """

        // `mo_wo_e4m3` of the low byte of each 16-bit half of `x`.
        inline half2 cb_sc_e4m3x2(uint x) {
            const uint v = x & 0x007F007Fu;
            const uint u = (v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7)
                | ((x & 0x00800080u) << 8);
            return as_type<half2>(u) * half2(256.0h);
        }

        // `mo_wo_pair(q, i, s)` with the scale as `float(s) * 16384`: the same
        // exact product, formed in float.
        inline float2 cb_sc_pair_f(uint q, int i, float fs) {
            uint p;
            if (i == 0) {
                p = ((q & 0x00070007) << 9) | ((q & 0x00080008) << 12);
            } else if (i == 1) {
                p = ((q & 0x00700070) << 5) | ((q & 0x00800080) << 8);
            } else if (i == 2) {
                p = ((q & 0x07000700) << 1) | ((q & 0x08000800) << 4);
            } else {
                p = ((q & 0x70007000) >> 3) | (q & 0x80008000);
            }
            return float2(as_type<half2>(p)) * fs;
        }

        """

    private static let nvfp4Kernel = MLXFast.metalKernel(
        name: "cbv2_verify_scales_nvfp4",
        inputNames: ["w", "scales", "x", "tensor_scale"],
        outputNames: ["y"],
        source: nvfp4Source,
        header: nvfp4Header)

    nonisolated(unsafe) private static let unit = MLXArray([Float(1)])

    /// Whether `nvfp4Product` takes these operands: 1 to 8 rows of K % 128 == 0.
    public static func nvfp4Supports(rows: Int, inputs: Int, outputs: Int, dtype: DType) -> Bool {
        (1 ... 8).contains(rows) && inputs % 128 == 0 && outputs >= 8
            && [.bfloat16, .float16, .float32].contains(dtype)
    }

    /// `ModelOptWeightOnlyNVFP4RowExact.product` for 1 to 8 rows, bit for bit.
    /// `rowsPerTile` (8, 16 or 32) and `prefetch` only schedule the work;
    /// `vectorScales` and `floatScale` choose the load and decode form.
    public static func nvfp4Product(
        _ x: MLXArray, weight: MLXArray, blockScales: MLXArray, tensorScale: MLXArray?,
        rowsPerTile: Int = 8, prefetch: Bool = false, vectorScales: Bool = false,
        floatScale: Bool = false, stockScaleDecode: Bool = false, rowGroups: Int = 1,
        chunksInFlight: Int = 1
    ) -> MLXArray {
        let inputs = x.dim(-1)
        let rows = x.size / max(1, inputs)
        let outputs = weight.dim(0)
        precondition(
            nvfp4Supports(rows: rows, inputs: inputs, outputs: outputs, dtype: x.dtype)
                && [8, 16, 32].contains(rowsPerTile) && [1, 2, 4].contains(rowGroups))
        let tiles = (outputs + rowsPerTile - 1) / rowsPerTile
        let groups = (tiles + rowGroups - 1) / rowGroups
        let template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("V", rows), ("K", inputs), ("N", outputs),
            ("SCALED", tensorScale != nil), ("NB", rowsPerTile), ("PF", prefetch),
            ("SV", vectorScales), ("FS", floatScale), ("SD", stockScaleDecode),
            ("RG", rowGroups), ("CH", chunksInFlight),
        ]
        let result = nvfp4Kernel(
            [weight, blockScales, x.reshaped(rows, inputs), tensorScale?.reshaped(1) ?? unit],
            template: template,
            grid: (32, 8, rowGroups * groups), threadGroup: (32, 8, rowGroups),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    // MARK: FP8

    private static let fp8Source = """
        constexpr int SN = 32;
        constexpr int TN = 4;
        constexpr int TM = 2;
        constexpr int threadsN = BN * SN;
        constexpr int blockM = BM * TM;
        constexpr int blockN = threadsN * TN;
        constexpr int tgp_size = BN > 1 ? BN * (blockM + TM) : 1;
        threadgroup float tgp_memory[V * tgp_size];
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const float weight_scale = scale[0];
        const float weight_scale_256 = weight_scale * 256.0f;
        const float nan_value = metal::numeric_limits<float>::quiet_NaN();
        thread float result[V][TM];
        for (int v = 0; v < V; v++) {
            for (int tm = 0; tm < TM; tm++) {
                result[v][tm] = 0;
            }
        }
        thread T inter[TN];
        thread float v_coeff[V][TN];
        const int thrN = int(simd_lid);
        const int sgN = BN != 1 ? (simd_gid % BN) : 0;
        const int simdM = BN != 1 ? int(simd_gid / BN) : int(simd_gid);
        const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
        int bm = simdM * TM;
        int bn = (simdN + thrN) * TN;
        int out_row = threadgroup_position_in_grid.x * blockM + bm;
        if (out_row >= N) {
            return;
        }
        out_row = out_row + TM <= N ? out_row : N - TM;
        const device auto* rows = mat + size_t(out_row) * K;
        constexpr int n_iter = K / blockN;
        constexpr int leftover = K - blockN * n_iter;
        for (int i = 0; i < n_iter; ++i) {
            for (int v = 0; v < V; v++) {
                if (AV) {
                    const float4 f = float4(as_type<vec<T, 4>>(
                        ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn))));
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] = f[tn];
                    }
                } else {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] = static_cast<float>(in_vec[v * K + bn + tn]);
                    }
                }
            }
            int mat_offset = 0;
            for (int tm = 0; tm < TM; tm++) {
                if (PD) {
                    const uint word = *(const device uint*)(rows + mat_offset + bn);
                    if (((word & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) {
                        for (int tn = 0; tn < TN; tn++) {
                            const uint8_t code = uint8_t(word >> (8 * tn));
                            const float decoded =
                                (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                            inter[tn] = static_cast<T>(decoded * weight_scale);
                        }
                    } else {
                        const float2 even = cb_sc_e4m3_scaled2(word, weight_scale_256);
                        const float2 odd = cb_sc_e4m3_scaled2(word >> 8, weight_scale_256);
                        inter[0] = static_cast<T>(even.x);
                        inter[1] = static_cast<T>(odd.x);
                        inter[2] = static_cast<T>(even.y);
                        inter[3] = static_cast<T>(odd.y);
                    }
                } else {
                    uint word = 0;
                    if (WV) {
                        word = *(const device uint*)(rows + mat_offset + bn);
                    }
                    for (int tn = 0; tn < TN; tn++) {
                        const uint8_t code = WV ? uint8_t(word >> (8 * tn)) : rows[mat_offset + bn + tn];
                        const float decoded =
                            (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                        inter[tn] = static_cast<T>(decoded * weight_scale);
                    }
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
                    v_coeff[v][tn] = bn + tn < K ? static_cast<float>(in_vec[v * K + bn + tn]) : 0.0f;
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
        """

    private static let fp8Header = """
        inline float modelopt_e4m3_value(uint8_t bits) {
            half converted = as_type<half>(static_cast<uint16_t>((bits & 127) << 7));
            converted *= 256.0;
            return static_cast<float>((bits & 128) ? -converted : converted);
        }

        // `modelopt_e4m3_value(c) * scale` for bytes 0 and 2 of `w` (no NaN
        // codes), with `scale256 = scale * 256`: the same real product, so the
        // same float rounding.
        inline float2 cb_sc_e4m3_scaled2(uint w, float scale256) {
            const uint x = w & 0x00FF00FFu;
            const uint h = ((x & 0x007F007Fu) << 7) | ((x & 0x00800080u) << 8);
            return float2(as_type<half2>(h)) * scale256;
        }
        """

    private static let fp8Kernel = MLXFast.metalKernel(
        name: "cbv2_verify_scales_fp8",
        inputNames: ["mat", "scale", "in_vec"],
        outputNames: ["out_vec"],
        source: fp8Source,
        header: fp8Header)

    /// `ModelOptRowExact.product(mat, scale:, vectors:, splitsK:, schedule: .windows)`
    /// for 1 to 8 rows of cast activations, bit for bit. `rowGroups` (BM)
    /// only schedules the work; `packedDecode`, `vectorWeights` and
    /// `vectorActivations` choose the load and decode form.
    public static func fp8Product(
        _ mat: MLXArray, scale: MLXArray, vectors: MLXArray, splitsK: Bool,
        rowGroups: Int = 1, packedDecode: Bool = true, vectorWeights: Bool = true,
        vectorActivations: Bool = true
    ) -> MLXArray {
        let (outputs, inputs) = (mat.dim(0), mat.dim(1))
        let count = vectors.dim(0)
        precondition(
            (1 ... 8).contains(count) && outputs >= 2 && inputs % 4 == 0 && mat.dtype == .uint8
                && vectors.dim(1) == inputs)
        let bn = splitsK ? 8 : 1
        let bm = rowGroups
        let groups = (outputs + bm * 2 - 1) / (bm * 2)
        let template: [(String, any KernelTemplateArg)] = [
            ("T", vectors.dtype), ("BM", bm), ("BN", bn), ("K", inputs), ("N", outputs),
            ("V", count), ("PD", packedDecode), ("WV", vectorWeights), ("AV", vectorActivations),
        ]
        return fp8Kernel(
            [mat, scale, vectors], template: template,
            grid: (groups * 32, bn, bm), threadGroup: (32, bn, bm),
            outputShapes: [[count, outputs]], outputDTypes: [vectors.dtype])[0]
    }

    /// The stock product the FP8 kernel reproduces (tests and benchmarks).
    public static func stockFP8Product(
        _ mat: MLXArray, scale: MLXArray, vectors: MLXArray, splitsK: Bool
    ) -> MLXArray {
        ModelOptRowExact.product(
            mat, scale: scale, vectors: vectors, splitsK: splitsK, schedule: .windows)
    }

    /// The FP8 activation cast pass of the `.windows` schedule.
    public static func fp8Cast(
        _ x: MLXArray, multipliers: MLXArray, overflowBound: MLXArray
    ) -> MLXArray {
        ModelOptActivationKernels.fp8(x, multipliers: multipliers, overflowBound: overflowBound)
    }

    // MARK: Routing

    /// Row groups per threadgroup for an FP8 run: four for the split-K run of
    /// a K = 19,968 projection at 6 to 8 rows (activations shared in L1),
    /// otherwise one. Scheduling only. Four row groups need a run of whole
    /// 8-output blocks, at any row count: the kernel's early return for
    /// outputs past N must stay uniform across a threadgroup's simdgroups.
    static func fp8RowGroups(splitsK: Bool, rows: Int, inputs: Int, runRows: Int) -> Int {
        splitsK && rows >= 6 && inputs >= 16384 && runRows % 8 == 0 ? 4 : 1
    }

    /// The exact product of `layer` for the verify rows `x` through these
    /// kernels: NVFP4 at 1 to 8 rows (8-row tiles), FP8 at 2 to 8 rows.
    /// Nil where they do not apply; the caller then runs the stock product.
    public static func product(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray? {
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        if let l = layer as? ModelOptWeightOnlyNVFP4Linear {
            // The matrix kernel's arithmetic: rows the stock product would
            // give its scalar kernel are left to it.
            guard l.inputDimensions == inputs,
                rows > ModelOptWeightOnlyNVFP4RowExact.scalarRows,
                ModelOptWeightOnlyNVFP4RowExact.supports(rows: rows, inputs: inputs, dtype: x.dtype),
                nvfp4Supports(
                    rows: rows, inputs: inputs, outputs: l.outputDimensions, dtype: x.dtype),
                l.weight.dtype == .uint32, l.weight.shape == [l.outputDimensions, inputs / 8],
                l.weightScale.dtype == .uint8,
                l.weightScale.shape == [l.outputDimensions, inputs / 16],
                l.weightScale2.dtype == .float32, l.weightScale2.size == 1
            else { return nil }
            return nvfp4Product(
                x, weight: l.weight, blockScales: l.weightScale, tensorScale: l.weightScale2,
                rowsPerTile: 8)
        }
        if let l = layer as? ModelOptCalibratedLinear {
            // Measured faster only for the MLP shapes: gate/up (vector loads) and
            // down at 6 to 8 rows (four row groups); attention keeps the stock product.
            let mlp = l.outputDimensions >= 16384 || (inputs >= 16384 && rows >= 6)
            // Activations are read as 16-bit lanes (`as_type<vec<T, 4>>(ushort4)`).
            guard mlp, l.format == .fp8, l.execution == .graph, l.inputDimensions == inputs,
                inputs > 64, inputs % 4 == 0, l.weightDType == x.dtype, (2 ... 8).contains(rows),
                x.dtype == .bfloat16 || x.dtype == .float16,
                l.rowExactSchedule == ModelOptRowExact.Schedule.windows,
                ModelOptRowExact.geometryOverride == nil,
                l.weight.shape == [l.outputDimensions, inputs]
            else { return nil }
            return fp8Forward(
                l, x,
                rowGroups: {
                    fp8RowGroups(splitsK: $0, rows: rows, inputs: inputs, runRows: $1)
                },
                packedDecode: false)
        }
        return nil
    }

    /// `product` for every layer, or nil unless it applies to all of them.
    public static func products(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray]? {
        var out: [MLXArray] = []
        for layer in layers {
            guard let y = product(layer, x) else { return nil }
            out.append(y)
        }
        return out
    }

    /// Runs `product` on zero rows at every count in `rows` so MLX compiles
    /// each kernel before a verify window needs it.
    public static func warm(_ layers: [any UnaryLayer], dtype: DType, rows: ClosedRange<Int>) {
        for layer in layers {
            guard let inputs = (layer as? ModelOptWeightOnlyNVFP4Linear)?.inputDimensions
                ?? (layer as? ModelOptCalibratedLinear)?.inputDimensions
            else { continue }
            for count in rows {
                if let y = product(layer, MLXArray.zeros([1, count, inputs], dtype: dtype)) {
                    eval(y)
                }
            }
        }
    }

    /// The runs of one-row eager tiles that share MLX's K split, as
    /// `ModelOptCalibratedLinear.rowExactForward` dispatches them.
    public static func fp8Runs(_ layer: ModelOptCalibratedLinear) -> [(
        rows: Range<Int>, splitsK: Bool
    )] {
        var runs: [(rows: Range<Int>, splitsK: Bool)] = []
        for tile in layer.fp8EagerTiles(rows: 1) {
            let splitsK = layer.inputDimensions >= 16 * tile.count
            if let last = runs.last, last.splitsK == splitsK {
                runs[runs.count - 1].rows = last.rows.lowerBound ..< tile.upperBound
            } else {
                runs.append((tile, splitsK))
            }
        }
        return runs
    }

    /// `layer.rowExactForward(x)` (schedule `.windows`, 2 to 8 rows) with the
    /// products of `fp8Product`; `stock` runs the stock products instead.
    /// Nil when the layer is not a calibrated FP8 layer.
    public static func fp8Forward(
        _ layer: ModelOptCalibratedLinear, _ x: MLXArray, stock: Bool = false,
        rowGroups: (_ splitsK: Bool, _ runRows: Int) -> Int = { _, _ in 1 },
        packedDecode: Bool = true,
        vectorWeights: Bool = true, vectorActivations: Bool = true
    ) -> MLXArray? {
        guard let constants = layer.graphConstants,
            case .fp8(let multipliers, let overflowBound) = constants.storage,
            layer.weight.dtype == .uint8
        else { return nil }
        let rows = x.size / layer.inputDimensions
        let vectors = fp8Cast(
            x.reshaped(rows, layer.inputDimensions), multipliers: multipliers,
            overflowBound: overflowBound)
        let pieces = fp8Runs(layer).map { run in
            let mat = layer.weight[run.rows, 0...]
            return stock
                ? stockFP8Product(
                    mat, scale: constants.outputScale, vectors: vectors, splitsK: run.splitsK)
                : fp8Product(
                    mat, scale: constants.outputScale, vectors: vectors, splitsK: run.splitsK,
                    rowGroups: rowGroups(run.splitsK, run.rows.count), packedDecode: packedDecode,
                    vectorWeights: vectorWeights, vectorActivations: vectorActivations)
        }
        let result = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
        return result.reshaped(Array(x.shape.dropLast()) + [layer.outputDimensions])
    }
}
