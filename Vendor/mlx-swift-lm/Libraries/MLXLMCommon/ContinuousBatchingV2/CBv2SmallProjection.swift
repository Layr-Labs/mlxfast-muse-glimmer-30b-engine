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
        // DV_MODE: 0 = scalar e4m3 decode, 1 = half2 decode with per-word NaN
        // check, 2 = branchless half2 + float2 SIMD scale (load-time verified
        // NaN-free weights and non-overflowing normal outputScale).
        template <typename T, int K, int V, int TM, bool SPLIT, int N, int NOUT, int DV_MODE, int PF, bool VO>
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
            const float dscale = (DV_MODE != 0) ? weight_scale * 256.0f : weight_scale;
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
                // Decoded weights of the four codes of each row's word. DV_MODE!=0:
                // the half bits of two codes per bit operation (codes 0 and 2,
                // then 1 and 3) with the sign in the half's sign bit, and the
                // exact factor 256 of `modelopt_e4m3_value` moved into the
                // scale: x * (256 s) and (256 x) * s round the same real
                // product once, and -(256 x) == 256 (-x).
                // When DV_MODE == 2, load-time verification proved zero 0x7F/0xFF
                // bytes and a normal non-overflowing scale, so dec[tm][tn] holds
                // the already-scaled float value with zero branches.
                float dec[TM][TN];
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                    const uint w = wq[tm];
                    if constexpr (DV_MODE == 2) {
                        const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                        const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                        const float2 f02 = float2(as_type<half2>(lo)) * dscale;
                        const float2 f13 = float2(as_type<half2>(hi)) * dscale;
                        dec[tm][0] = f02.x;
                        dec[tm][1] = f13.x;
                        dec[tm][2] = f02.y;
                        dec[tm][3] = f13.y;
                    } else if constexpr (DV_MODE == 1) {
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
                            inter[tm][tn] = static_cast<T>(
                                (DV_MODE == 2) ? dec[tm][tn] : (dec[tm][tn] * dscale));
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
                        inter[tm] = static_cast<T>(
                            (DV_MODE == 2) ? dec[tm][tn] : (dec[tm][tn] * dscale));
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
            if constexpr (leftover > 0) {
                if constexpr (DV_MODE == 2 && leftover % 128 == 0) {
                    if (bn < K) {
                        ushort4 av[V];
                        CBV2_SP_UNROLL
                        for (int v = 0; v < V; v++) {
                            av[v] = ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn));
                        }
                        float dec[TM][TN];
                        CBV2_SP_UNROLL
                        for (int tm = 0; tm < TM; tm++) {
                            const uint w = *(const device uint*)(rows + tm * K + bn);
                            const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                            const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                            const float2 f02 = float2(as_type<half2>(lo)) * dscale;
                            const float2 f13 = float2(as_type<half2>(hi)) * dscale;
                            dec[tm][0] = f02.x;
                            dec[tm][1] = f13.x;
                            dec[tm][2] = f02.y;
                            dec[tm][3] = f13.y;
                        }
                        if (VO) {
                            T inter[TM][TN];
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                CBV2_SP_UNROLL
                                for (int tn = 0; tn < TN; tn++) {
                                    inter[tm][tn] = static_cast<T>(dec[tm][tn]);
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
                                    inter[tm] = static_cast<T>(dec[tm][tn]);
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
                    } else {
                        CBV2_SP_UNROLL
                        for (int i = 0; i < P; i++) {
                            r[i] += 0.0f;
                        }
                    }
                } else {
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

    /// `CBV2_FP8_ROWGROUPS=0`: disables Wave-6 `RG=4` + branchless half2 SIMD + streamAV
    /// on pure-SPLIT FP8 `qkvg` (28L) and FP8 `SwiGLU` (7L), strictly `xnorDraftHeadEligible`-gated.
    public static let fp8RowGroupsEnvEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_FP8_ROWGROUPS"] != "0"

    static let fp8W6Helper = """

        inline half2 cbv2_sp_e4m3_lo(uint w) {
            const uint v = w & 0x007F007Fu;
            const uint u = (v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7)
                | ((w & 0x00800080u) << 8);
            return as_type<half2>(u) * half2(256.0h);
        }

        inline half2 cbv2_sp_e4m3_hi(uint w) {
            const uint v = (w >> 1) & 0x3F803F80u;
            const uint u = v | ((v + 0x00800080u) & 0x40004000u) | (w & 0x80008000u);
            return as_type<half2>(u) * half2(256.0h);
        }

        template <typename T>
        inline void cbv2_sp_decode4_bld(uint w, float scale, thread T* out) {
            const half2 h02 = cbv2_sp_e4m3_lo(w);
            const half2 h13 = cbv2_sp_e4m3_hi(w);
            out[0] = static_cast<T>(static_cast<float>(h02.x) * scale);
            out[1] = static_cast<T>(static_cast<float>(h13.x) * scale);
            out[2] = static_cast<T>(static_cast<float>(h02.y) * scale);
            out[3] = static_cast<T>(static_cast<float>(h13.y) * scale);
        }

        template <typename T, int K, int V, int TM, bool SPLIT, int N, int NOUT, int PF>
        inline void cbv2_sp_fp8_w6(
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
                r[i] = 0.0f;
            }
            constexpr int PD = PF <= 1 ? 1 : (PF >= n_iter ? (n_iter > 0 ? n_iter : 1) : PF);
            for (int i0 = 0; i0 < n_iter; i0 += PD) {
                uint wblk[PD][TM];
                CBV2_SP_UNROLL
                for (int d = 0; d < PD; d++) {
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        wblk[d][tm] = i0 + d < n_iter
                            ? *(const device uint*)(rows + size_t(tm) * K + bn + d * blockN) : 0u;
                    }
                }
                CBV2_SP_UNROLL
                for (int d = 0; d < PD; d++) {
                    if (i0 + d >= n_iter) {
                        break;
                    }
                    T inter[TM][TN];
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        cbv2_sp_decode4_bld<T>(wblk[d][tm], weight_scale, inter[tm]);
                    }
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        const ushort4 av = ushort4(*(const device packed_ushort4*)(in_vec + size_t(v) * K + bn));
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const float a = static_cast<float>(as_type<T>(av[tn]));
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                r[v * TM + tm] += inter[tm][tn] * a;
                            }
                        }
                    }
                    bn += blockN;
                }
            }
            if (leftover > 0 && bn < K) {
                T inter[TM][TN];
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                    const uint w = *(const device uint*)(rows + size_t(tm) * K + bn);
                    cbv2_sp_decode4_bld<T>(w, weight_scale, inter[tm]);
                }
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                    const ushort4 av = ushort4(*(const device packed_ushort4*)(in_vec + size_t(v) * K + bn));
                    CBV2_SP_UNROLL
                    for (int tn = 0; tn < TN; tn++) {
                        const float a = static_cast<float>(as_type<T>(av[tn]));
                        CBV2_SP_UNROLL
                        for (int tm = 0; tm < TM; tm++) {
                            r[v * TM + tm] += inter[tm][tn] * a;
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

    static let fp8SwiGLURowGroupHelper = """

        template <typename T, int V, int RPS, int P2>
        inline void cbv2_rg_swiglu_step_w6(
            const device T* xc, int K, int bn, thread const uint* wg, thread const uint* wu,
            float scale_g, float scale_u, thread float* rg, thread float* ru) {
            T ig[RPS][4];
            T iu[RPS][4];
            CBV2_SP_UNROLL
            for (int r = 0; r < RPS; r++) {
                cbv2_sp_decode4_bld<T>(wg[r], scale_g, ig[r]);
                cbv2_sp_decode4_bld<T>(wu[r], scale_u, iu[r]);
            }
            CBV2_SP_UNROLL
            for (int v = 0; v < V; v++) {
                const ushort4 av = ushort4(*(const device packed_ushort4*)(xc + size_t(v) * K + bn));
                CBV2_SP_UNROLL
                for (int tn = 0; tn < 4; tn++) {
                    const float a = static_cast<float>(as_type<T>(av[tn]));
                    CBV2_SP_UNROLL
                    for (int r = 0; r < RPS; r++) {
                        rg[r * P2 + v] += ig[r][tn] * a;
                        ru[r * P2 + v] += iu[r][tn] * a;
                    }
                }
            }
        }
        """

    private static let fp8SwiGLURowGroupSource = """
        constexpr int TN = 4;
        constexpr int blockN = 8 * 32 * TN;
        constexpr int n_iter = K / blockN;
        constexpr int leftover = K - blockN * n_iter;
        constexpr int P = V;
        static_assert(P <= 8, "at most 8 verify rows");
        static_assert(K % 4 == 0, "whole 4-code words");
        constexpr int P2 = P <= 1 ? 1 : P <= 2 ? 2 : P <= 4 ? 4 : 8;
        constexpr int SH = P2 == 1 ? 5 : P2 == 2 ? 4 : P2 == 4 ? 3 : 2;
        constexpr uint LOW = (1u << SH) - 1u;
        const float nan_value = metal::numeric_limits<float>::quiet_NaN();
        const float scale_g = wsg[0];
        const float scale_u = wsu[0];
        const uint sid = simdgroup_index_in_threadgroup;
        const int g = int(sid & 7u);
        const int grp = int(sid >> 3);
        const uint simd_lid = thread_index_in_simdgroup;
        const int row0 = (int(threadgroup_position_in_grid.y) * BM + grp) * RPS;
        int bn = (32 * g + int(simd_lid)) * TN;
        const device uint8_t* row_g[RPS];
        const device uint8_t* row_u[RPS];
        float rg[RPS * P2];
        float ru[RPS * P2];
        CBV2_SP_UNROLL
        for (int r = 0; r < RPS; r++) {
            const int row = min(row0 + r, N - 1);
            row_g[r] = mg + size_t(row) * K;
            row_u[r] = mu + size_t(row) * K;
        }
        CBV2_SP_UNROLL
        for (int i = 0; i < RPS * P2; i++) {
            rg[i] = 0.0f;
            ru[i] = 0.0f;
        }
        for (int i0 = 0; i0 < n_iter; i0 += PF) {
            uint wgb[PF][RPS];
            uint wub[PF][RPS];
            CBV2_SP_UNROLL
            for (int d = 0; d < PF; d++) {
                CBV2_SP_UNROLL
                for (int r = 0; r < RPS; r++) {
                    const bool ok = (PF == 1) || (i0 + d < n_iter);
                    wgb[d][r] = ok ? *(const device uint*)(row_g[r] + bn + d * blockN) : 0u;
                    wub[d][r] = ok ? *(const device uint*)(row_u[r] + bn + d * blockN) : 0u;
                }
            }
            CBV2_SP_UNROLL
            for (int d = 0; d < PF; d++) {
                if (PF > 1 && i0 + d >= n_iter) {
                    break;
                }
                cbv2_rg_swiglu_step_w6<T, V, RPS, P2>(xc, K, bn, wgb[d], wub[d], scale_g, scale_u, rg, ru);
                bn += blockN;
            }
        }
        if (leftover > 0 && bn < K) {
            uint wg1[RPS];
            uint wu1[RPS];
            CBV2_SP_UNROLL
            for (int r = 0; r < RPS; r++) {
                wg1[r] = *(const device uint*)(row_g[r] + bn);
                wu1[r] = *(const device uint*)(row_u[r] + bn);
            }
            cbv2_rg_swiglu_step_w6<T, V, RPS, P2>(xc, K, bn, wg1, wu1, scale_g, scale_u, rg, ru);
        }
        const ushort lid = ushort(simd_lid);
        threadgroup float tgp_g[BM * RPS * 64];
        threadgroup float tgp_u[BM * RPS * 64];
        const bool holds = (simd_lid & LOW) == 0;
        const int i = int(simd_lid >> SH);
        CBV2_SP_UNROLL
        for (int r = 0; r < RPS; r++) {
            thread float* xg = rg + r * P2;
            thread float* xu = ru + r * P2;
            cbv2_sp_tree_step<P2, 16>(xg, lid);
            cbv2_sp_tree_step<(P2 >= 2 ? P2 / 2 : 1), 8>(xg, lid);
            cbv2_sp_tree_step<(P2 >= 4 ? P2 / 4 : 1), 4>(xg, lid);
            cbv2_sp_tree_step<(P2 >= 8 ? P2 / 8 : 1), 2>(xg, lid);
            cbv2_sp_tree_step<1, 1>(xg, lid);
            cbv2_sp_tree_step<P2, 16>(xu, lid);
            cbv2_sp_tree_step<(P2 >= 2 ? P2 / 2 : 1), 8>(xu, lid);
            cbv2_sp_tree_step<(P2 >= 4 ? P2 / 4 : 1), 4>(xu, lid);
            cbv2_sp_tree_step<(P2 >= 8 ? P2 / 8 : 1), 2>(xu, lid);
            cbv2_sp_tree_step<1, 1>(xu, lid);
            if (holds && i < 8) {
                tgp_g[((grp * RPS + r) * 8 + g) * 8 + i] = xg[0];
                tgp_u[((grp * RPS + r) * 8 + g) * 8 + i] = xu[0];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g < RPS && simd_lid < uint(P)) {
            const int r = g;
            const int v = int(simd_lid);
            const int out_row = row0 + r;
            const int base = (grp * RPS + r) * 64;
            float res_g = tgp_g[base + v];
            float res_u = tgp_u[base + v];
            for (int s = 1; s < 8; s++) {
                res_g += tgp_g[base + s * 8 + v];
                res_u += tgp_u[base + s * 8 + v];
            }
            if (out_row < N) {
                const bfloat16_t gv = static_cast<bfloat16_t>(res_g);
                const bfloat16_t uv = static_cast<bfloat16_t>(res_u);
                const bfloat16_t zero_bf = bfloat16_t(0.0f);
                const bfloat16_t one_bf = bfloat16_t(1.0f);
                const bfloat16_t ax = (gv < zero_bf) ? -gv : gv;
                const bfloat16_t ey = static_cast<bfloat16_t>(
                    metal::precise::exp(static_cast<float>(ax)));
                const bfloat16_t sy = one_bf / (one_bf + ey);
                const bfloat16_t sig = (gv < zero_bf) ? sy : (one_bf - sy);
                bfloat16_t act = (gv * sig) * uv;
                if constexpr (CAST_FP8) {
                    const float scaled = static_cast<float>(act) * fp8_mu[0];
                    const uint8_t bits = modelopt_e4m3_bits(scaled);
                    const float decoded =
                        (bits & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(bits);
                    const bool invalid = metal::isnan(scaled) || metal::abs(scaled) > fp8_bd[0];
                    act = static_cast<bfloat16_t>(invalid ? nan_value : decoded * fp8_mu[1]);
                }
                y[size_t(v) * N + out_row] = static_cast<T>(act);
            }
        }
        """

    private static let fp8SwiGLURowGroupKernel = MLXFast.metalKernel(
        name: "cbv2_small_fp8_swiglu_rg_w6",
        inputNames: ["mg", "wsg", "mu", "wsu", "xc", "fp8_mu", "fp8_bd"],
        outputNames: ["y"],
        source: fp8SwiGLURowGroupSource,
        header: ModelOptActivationKernels.header + fp8Helper + fp8W6Helper + fp8SwiGLURowGroupHelper)

    private struct FP8RGLaunchKey: Hashable {
        let shape: FP8Shape
        let dtype: DType
        let inputs: Int
        let rows: Int
        let outputs: [Int]
        let rowGroups: Int
    }

    private struct FP8SwiGLURGLaunchKey: Hashable {
        let dtype: DType
        let inputs: Int
        let outputs: Int
        let rows: Int
        let castFP8: Bool
        let rowGroups: Int
    }

    nonisolated(unsafe) private static var fp8RGKernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var fp8RGLaunchCache:
        [FP8RGLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var fp8SwiGLURGLaunchCache:
        [FP8SwiGLURGLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var fp8RGChecked = false
    nonisolated(unsafe) private static var fp8RGVerified = false
    nonisolated(unsafe) private static var fp8SwiGLURGChecked = false
    nonisolated(unsafe) private static var fp8SwiGLURGVerified = false

    private static func fp8W6RowGroupKernel(_ shape: FP8Shape, rowGroups rg: Int) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        let key = "w6rg\(rg)_" + shape.segments.map {
            "\($0.weight)_\($0.row0)_\($0.rows)_\($0.tm)_\($0.prefetch)"
        }.joined(separator: "_")
        if let existing = fp8RGKernels[key] { return existing }
        var inputs = [String]()
        for i in 0 ..< shape.weights {
            inputs += ["m\(i)", "s\(i)"]
        }
        inputs.append("xc")
        let outputs = (0 ..< shape.outputs).map { "y\($0)" }
        var body = """
            threadgroup float sp_tgp_all[\(rg) * 8 * 32];
            const uint rg_idx = simdgroup_index_in_threadgroup / 8;
            const uint sg = simdgroup_index_in_threadgroup % 8;
            const uint sl = thread_index_in_simdgroup;
            const int tg = int(threadgroup_position_in_grid.y) * \(rg) + int(rg_idx);
            threadgroup float* sp_tgp = sp_tgp_all + rg_idx * 256;

            """
        var offset = 0
        for (i, s) in shape.segments.enumerated() {
            let end = offset + s.tiles
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(end)) {
                    cbv2_sp_fp8_w6<T, K, V, \(s.tm), true, \(s.rows), \(s.outputWidth), \(s.prefetch)>(
                        m\(s.weight) + size_t(\(s.row0)) * K, s\(s.weight)[0], xc,
                        y\(s.output) + \(s.column0), tg - \(offset), sg, sl, sp_tgp);
                }

                """
            offset = end
        }
        let name = "cbv2_small_fp8_w6_rg\(rg)_" + stableName(body)
        let made = MLXFast.metalKernel(
            name: name, inputNames: inputs, outputNames: outputs, source: body,
            header: ModelOptActivationKernels.header + fp8Helper + fp8W6Helper)
        fp8RGKernels[key] = made
        return made
    }

    private static func fp8RGLaunch(
        shape: FP8Shape, dtype: DType, inputs: Int, rows: Int, outputDims: [Int], rowGroups rg: Int
    ) -> MLXFast.MLXFastKernel.Launch {
        let key = FP8RGLaunchKey(
            shape: shape, dtype: dtype, inputs: inputs, rows: rows, outputs: outputDims,
            rowGroups: rg)
        if let hit = lock.withLock({ fp8RGLaunchCache[key] }) { return hit }
        let tiles = shape.segments.reduce(0) { $0 + $1.tiles }
        let groups = (tiles + rg - 1) / rg
        let made = fp8W6RowGroupKernel(shape, rowGroups: rg).prepare(
            template: [("T", dtype), ("K", inputs), ("V", rows)],
            grid: (32, 8 * rg * groups, 1), threadGroup: (32, 8 * rg, 1),
            outputShapes: outputDims.map { [rows, $0] },
            outputDTypes: Array(repeating: dtype, count: outputDims.count))
        lock.withLock { fp8RGLaunchCache[key] = made }
        return made
    }

    private static func ensureFP8RowGroupsVerified(
        shape: FP8Shape, arrays: [MLXArray], dtype: DType, inputs: Int, rows: Int,
        outputDims: [Int], rowGroups rg: Int
    ) -> Bool {
        let (checked, verified) = lock.withLock { (fp8RGChecked, fp8RGVerified) }
        if checked { return verified }
        var vals = [Float]()
        vals.reserveCapacity(rows * inputs)
        for i in 0 ..< (rows * inputs) {
            vals.append(Float(((i * 11 + 5) % 29) - 14) * 0.0625)
        }
        var testArrays = Array(arrays.dropLast())
        testArrays.append(MLXArray(vals, [rows, inputs]).asType(dtype))
        let ok: Bool =
            (try? MLX.withError { () -> Bool in
                let tiles = shape.segments.reduce(0) { $0 + $1.tiles }
                let refLaunch = fp8Kernel(shape).prepare(
                    template: [("T", dtype), ("K", inputs), ("V", rows)],
                    grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                    outputShapes: outputDims.map { [rows, $0] },
                    outputDTypes: Array(repeating: dtype, count: outputDims.count))
                let ref = refLaunch(testArrays)
                let got = fp8RGLaunch(
                    shape: shape, dtype: dtype, inputs: inputs, rows: rows,
                    outputDims: outputDims, rowGroups: rg)(testArrays)
                var totalDiff = MLXArray(Int32(0))
                for (a, b) in zip(got, ref) {
                    totalDiff = totalDiff
                        + (a.view(dtype: .uint16) .!= b.view(dtype: .uint16)).asType(.int32).sum()
                }
                eval(totalDiff)
                return totalDiff.item(Int.self) == 0
            }) ?? false
        lock.withLock {
            fp8RGChecked = true
            fp8RGVerified = ok
        }
        return ok
    }

    private static func fp8SwiGLURGLaunch(
        dtype: DType, inputs: Int, outputs: Int, rows: Int, castFP8: Bool, rowGroups rg: Int
    ) -> MLXFast.MLXFastKernel.Launch {
        let key = FP8SwiGLURGLaunchKey(
            dtype: dtype, inputs: inputs, outputs: outputs, rows: rows, castFP8: castFP8,
            rowGroups: rg)
        if let hit = lock.withLock({ fp8SwiGLURGLaunchCache[key] }) { return hit }
        let threadgroups = (outputs + rg - 1) / rg
        let made = fp8SwiGLURowGroupKernel.prepare(
            template: [
                ("T", dtype), ("K", inputs), ("V", rows), ("N", outputs),
                ("BM", rg), ("RPS", 1), ("PF", 1), ("CAST_FP8", castFP8),
            ],
            grid: (32, 8 * rg * threadgroups, 1), threadGroup: (32, 8 * rg, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [dtype])
        lock.withLock { fp8SwiGLURGLaunchCache[key] = made }
        return made
    }

    private static func ensureFP8SwiGLURowGroupsVerified(
        gate: ModelOptCalibratedLinear, up: ModelOptCalibratedLinear,
        muArr: MLXArray, bdArr: MLXArray, castFP8: Bool,
        dtype: DType, inputs: Int, outputs: Int, rows: Int, rowGroups rg: Int
    ) -> Bool {
        let (checked, verified) = lock.withLock { (fp8SwiGLURGChecked, fp8SwiGLURGVerified) }
        if checked { return verified }
        var vals = [Float]()
        vals.reserveCapacity(rows * inputs)
        for i in 0 ..< (rows * inputs) {
            vals.append(Float(((i * 7 + 3) % 29) - 14) * 0.0625)
        }
        let testX = MLXArray(vals, [rows, inputs]).asType(dtype)
        let args = [
            gate.weight, gate.graphConstants!.outputScale,
            up.weight, up.graphConstants!.outputScale,
            testX, muArr, bdArr,
        ]
        let ok: Bool =
            (try? MLX.withError { () -> Bool in
                let refLaunch = fp8GateUpSiluKernel.prepare(
                    template: [
                        ("T", dtype), ("K", inputs), ("V", rows), ("N", outputs),
                        ("DV_MODE", 0), ("VO", false), ("CAST_FP8", castFP8),
                    ],
                    grid: (32, 8 * outputs, 1), threadGroup: (32, 8, 1),
                    outputShapes: [[rows, outputs]], outputDTypes: [dtype])
                let ref = refLaunch(args)[0]
                let got = fp8SwiGLURGLaunch(
                    dtype: dtype, inputs: inputs, outputs: outputs, rows: rows,
                    castFP8: castFP8, rowGroups: rg)(args)[0]
                let diff = (got.view(dtype: .uint16) .!= ref.view(dtype: .uint16))
                    .asType(.int32).sum()
                eval(diff)
                return diff.item(Int.self) == 0
            }) ?? false
        lock.withLock {
            fp8SwiGLURGChecked = true
            fp8SwiGLURGVerified = ok
        }
        return ok
    }

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
        var vectorDecodeMode: Int
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
                    cbv2_sp_fp8<T, K, V, \(s.tm), \(s.split), \(s.rows), \(s.outputWidth), \(shape.vectorDecodeMode), \(s.prefetch), \(shape.rowsOuter)>(
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

    /// Two codes per bit operation in the weight decode (mode 1: with per-word
    /// NaN check; mode 2: branchless after load-time NaN-byte + scale-range
    /// verification).
    nonisolated(unsafe) static var vectorDecode = false

    /// `CBV2_FP8_BRANCHLESS`: `0` = off, `1` = on for low-acceptance rounds
    /// (default), `2` = always on (requires load-time NaN-free + scale-safe
    /// verification of every layer in the group).
    public static let fp8BranchlessMode: Int =
        ProcessInfo.processInfo.environment["CBV2_FP8_BRANCHLESS"].flatMap(Int.init) ?? 1

    private static let fp8SafeLock = NSLock()
    nonisolated(unsafe) private static var fp8SafeCache: [ObjectIdentifier: Bool] = [:]

    private static func isScaleSafeFor256(_ scale: Float) -> Bool {
        guard scale.isFinite else { return false }
        let a = abs(scale)
        return a == 0 || (a >= Float.leastNormalMagnitude && a <= 1e35)
    }

    /// Batch-verifies frozen FP8 layers once at load time in one GPU sync:
    /// checks that `weight` contains zero `0x7F`/`0xFF` NaN bytes and that
    /// `outputScale * 256.0f` neither overflows nor rounds as a subnormal.
    public static func warmFP8BranchlessSafe(_ layers: [ModelOptCalibratedLinear]) {
        guard fp8BranchlessMode != 0 else { return }
        var pending: [(id: ObjectIdentifier, hasNaN: MLXArray, scale: MLXArray)] = []
        fp8SafeLock.withLock {
            for layer in layers {
                let id = ObjectIdentifier(layer)
                guard fp8SafeCache[id] == nil,
                    layer.format == .fp8, layer.weight.dtype == .uint8,
                    let constants = layer.graphConstants,
                    constants.outputScale.dtype == .float32, constants.outputScale.size == 1
                else { continue }
                let w = layer.weight
                let hasNaN = MLX.any(
                    logicalOr(equal(w, MLXArray(UInt8(0x7F))), equal(w, MLXArray(UInt8(0xFF)))))
                pending.append((id, hasNaN, constants.outputScale))
            }
        }
        guard !pending.isEmpty else { return }
        eval(pending.flatMap { [$0.hasNaN, $0.scale] })
        let results: [(ObjectIdentifier, Bool)] = pending.map { item in
            let safe = !item.hasNaN.item(Bool.self)
                && isScaleSafeFor256(item.scale.item(Float.self))
            return (item.id, safe)
        }
        fp8SafeLock.withLock {
            for (id, safe) in results {
                fp8SafeCache[id] = safe
            }
        }
    }

    /// True when `layer` has been proven NaN-byte-free and scale-safe (checked
    /// once per layer and cached).
    public static func isFP8BranchlessSafe(_ layer: ModelOptCalibratedLinear) -> Bool {
        let id = ObjectIdentifier(layer)
        if let cached = fp8SafeLock.withLock({ fp8SafeCache[id] }) {
            return cached
        }
        guard layer.format == .fp8, layer.weight.dtype == .uint8,
            let constants = layer.graphConstants,
            constants.outputScale.dtype == .float32, constants.outputScale.size == 1
        else {
            fp8SafeLock.withLock { fp8SafeCache[id] = false }
            return false
        }
        let scaleVal = constants.outputScale.item(Float.self)
        guard isScaleSafeFor256(scaleVal) else {
            fp8SafeLock.withLock { fp8SafeCache[id] = false }
            return false
        }
        let w = layer.weight
        let hasNaN = MLX.any(
            logicalOr(equal(w, MLXArray(UInt8(0x7F))), equal(w, MLXArray(UInt8(0xFF))))
        ).item(Bool.self)
        let safe = !hasNaN
        fp8SafeLock.withLock { fp8SafeCache[id] = safe }
        return safe
    }

    /// True when branchless half2 FP8 decode should run for `layers` in the
    /// current regime (low-acc gated by default, load-time NaN/scale verified).
    public static func shouldUseBranchlessFP8(for layers: [ModelOptCalibratedLinear]) -> Bool {
        guard !layers.isEmpty else { return false }
        let modeAllowed: Bool
        switch fp8BranchlessMode {
        case 0: modeAllowed = false
        case 2: modeAllowed = true
        default: modeAllowed = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        }
        guard modeAllowed else { return false }
        return layers.allSatisfy(isFP8BranchlessSafe)
    }

    /// Weight words loaded ahead, in iterations: none for K-split tiles
    /// (deeper measured slower: registers), 4 (up to 4 rows) or 2 for the
    /// BN = 1 rows (o projection: 32 iterations per lane).
    static func prefetchDepth(rows: Int, split: Bool) -> Int {
        split ? 1 : (rows <= 4 ? 4 : 2)
    }

    /// Activation rows outermost in the inner loop (same chains).
    nonisolated(unsafe) public static var rowsOuter = false
    /// `CBV2_FP8_ROWS_OUTER`: `0` = off, `1` = on for `rows >= 5` (default), `2` = always on.
    public static let fp8RowsOuterMode: Int =
        ProcessInfo.processInfo.environment["CBV2_FP8_ROWS_OUTER"].flatMap(Int.init) ?? 1

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

    private static let sharedCastLock = NSLock()
    nonisolated(unsafe) private static var sharedCastCache:
        [[ObjectIdentifier]: (multipliers: MLXArray, overflowBound: MLXArray)?] = [:]

    /// Returns the shared `(multipliers, overflowBound)` when every FP8 layer
    /// in `layers` has bit-identical activation cast parameters (cached once
    /// per layer group; zero per-step host sync).
    public static func sharedFP8CastParams(
        _ layers: [ModelOptCalibratedLinear], inputs: Int, dtype: DType
    ) -> (multipliers: MLXArray, overflowBound: MLXArray)? {
        guard !layers.isEmpty,
            layers.allSatisfy({ fp8Takes($0, inputs: inputs, dtype: dtype) })
        else { return nil }
        let key = layers.map(ObjectIdentifier.init)
        if let cached = sharedCastLock.withLock({ sharedCastCache[key] }) {
            return cached
        }
        guard case .fp8(let mu0, let bd0) = layers[0].graphConstants!.storage else { return nil }
        let refMu = mu0.asArray(Float.self).map(\.bitPattern)
        let refBd = bd0.asArray(Float.self).map(\.bitPattern)
        var allSame = true
        for layer in layers.dropFirst() {
            guard case .fp8(let mu, let bd) = layer.graphConstants!.storage else {
                allSame = false
                break
            }
            if mu.asArray(Float.self).map(\.bitPattern) != refMu
                || bd.asArray(Float.self).map(\.bitPattern) != refBd
            {
                allSame = false
                break
            }
        }
        let result: (multipliers: MLXArray, overflowBound: MLXArray)? = allSame ? (mu0, bd0) : nil
        sharedCastLock.withLock { sharedCastCache[key] = result }
        return result
    }

    private struct FP8LaunchKey: Hashable {
        let shape: FP8Shape
        let dtype: DType
        let inputs: Int
        let rows: Int
        let outputs: [Int]
    }

    private struct CastLaunchKey: Hashable {
        let segments: Int
        let dtype: DType
        let count: Int
    }

    nonisolated(unsafe) private static var fp8LaunchCache:
        [FP8LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var castLaunchCache:
        [CastLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    private static func fp8Run(
        _ layers: [ModelOptCalibratedLinear], _ flat: MLXArray, cast: MLXArray, rows: Int,
        sharedInput: Bool
    ) -> [MLXArray] {
        let inputs = flat.dim(-1)
        var segments = [FP8Segment]()
        var arrays = [MLXArray]()
        arrays.reserveCapacity(layers.count * 2 + 1)
        for (i, layer) in layers.enumerated() {
            arrays += [layer.weight, layer.graphConstants!.outputScale]
            for run in fp8Runs(layer) {
                let tm = min(
                    rowsPerThread(rows: rows, split: run.split), run.rows.count, max(1, 32 / rows))
                segments.append(
                    FP8Segment(
                        weight: i, row0: run.rows.lowerBound, rows: run.rows.count,
                        input: (layers.count == 1 || sharedInput) ? 0 : i, output: i,
                        column0: run.rows.lowerBound, outputWidth: layer.outputDimensions,
                        split: run.split, tm: tm,
                        prefetch: prefetchOverride ?? prefetchDepth(rows: rows, split: run.split)))
            }
        }
        arrays.append(cast)
        let useRowsOuter =
            rowsOuter || fp8RowsOuterMode == 2
            || (fp8RowsOuterMode == 1 && rows >= 5
                && (!CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance))
        let dvMode = shouldUseBranchlessFP8(for: layers) ? 2 : (vectorDecode ? 1 : 0)
        let shape = FP8Shape(
            vectorDecodeMode: dvMode, rowsOuter: useRowsOuter, weights: layers.count,
            outputs: layers.count, segments: segments)
        let outputDims = layers.map(\.outputDimensions)
        let deepLow = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.xnorDraftHeadEligible
        let allSplit = segments.allSatisfy(\.split)
        let totalTiles = segments.reduce(0) { $0 + $1.tiles }
        if deepLow && fp8RowGroupsEnvEnabled && sharedInput && layers.count >= 2 && rows >= 2
            && allSplit && totalTiles >= 4 && totalTiles % 4 == 0
            && ensureFP8RowGroupsVerified(
                shape: shape, arrays: arrays, dtype: flat.dtype, inputs: inputs, rows: rows,
                outputDims: outputDims, rowGroups: 4)
        {
            return fp8RGLaunch(
                shape: shape, dtype: flat.dtype, inputs: inputs, rows: rows,
                outputDims: outputDims, rowGroups: 4)(arrays)
        }
        let launchKey = FP8LaunchKey(
            shape: shape, dtype: flat.dtype, inputs: inputs, rows: rows, outputs: outputDims)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = lock.withLock({ fp8LaunchCache[launchKey] }) {
            launch = hit
        } else {
            let tiles = segments.reduce(0) { $0 + $1.tiles }
            let made = fp8Kernel(shape).prepare(
                template: [("T", flat.dtype), ("K", inputs), ("V", rows)],
                grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                outputShapes: outputDims.map { [rows, $0] },
                outputDTypes: Array(repeating: flat.dtype, count: outputDims.count))
            lock.withLock { fp8LaunchCache[launchKey] = made }
            launch = made
        }
        return launch(arrays)
    }

    /// Runs sibling FP8 projections directly on an already-cast `[..., K]`
    /// buffer shared by all layers in the group (e.g. from
    /// `VerifyFusedNorms.fusedAddNormNormCastFP8`).
    public static func fp8ProductsPreCast(
        _ layers: [ModelOptCalibratedLinear], preCast: MLXArray
    ) -> [MLXArray]? {
        let inputs = preCast.dim(-1)
        guard inputs > 0, !layers.isEmpty else { return nil }
        let rows = preCast.size / inputs
        guard (1 ... maximumRows).contains(rows),
            layers.allSatisfy({ fp8Takes($0, inputs: inputs, dtype: preCast.dtype) })
        else { return nil }
        let flat = preCast.reshaped(rows, inputs)
        let lead = Array(preCast.shape.dropLast())
        return zip(fp8Run(layers, flat, cast: flat, rows: rows, sharedInput: true), layers).map {
            $0.reshaped(lead + [$1.outputDimensions])
        }
    }

    /// Dedicated single-segment FP8 activation cast (no integer division).
    private static let castKernel1 = MLXFast.metalKernel(
        name: "cbv2_small_fp8_activation_1",
        inputNames: ["x", "mu0", "bd0"],
        outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            const float scaled = static_cast<float>(x[i]) * mu0[0];
            const uint8_t bits = modelopt_e4m3_bits(scaled);
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            const float decoded = (bits & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(bits);
            const bool invalid = metal::isnan(scaled) || metal::abs(scaled) > bd0[0];
            out[i] = static_cast<T>(invalid ? nan_value : decoded * mu0[1]);
            """,
        header: ModelOptActivationKernels.header)

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
        1: castKernel1, 2: castKernel(segments: 2), 4: castKernel(segments: 4),
    ]

    private static func preparedFP8Cast(
        _ layers: [ModelOptCalibratedLinear], flat: MLXArray, count: Int
    ) -> MLXArray? {
        guard let kernel = castKernels[layers.count] else { return nil }
        var castInputs: [MLXArray] = [flat]
        castInputs.reserveCapacity(1 + layers.count * 2)
        for layer in layers {
            guard case .fp8(let mu, let bound) = layer.graphConstants!.storage else {
                return nil
            }
            castInputs += [mu, bound]
        }
        let key = CastLaunchKey(segments: layers.count, dtype: flat.dtype, count: count)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = lock.withLock({ castLaunchCache[key] }) {
            launch = hit
        } else {
            let total = layers.count * count
            let template: [(String, any KernelTemplateArg)] =
                layers.count == 1
                ? [("T", flat.dtype)]
                : [("T", flat.dtype), ("COUNT", count)]
            let made = kernel.prepare(
                template: template,
                grid: (total, 1, 1), threadGroup: (min(256, total), 1, 1),
                outputShapes: [[total]], outputDTypes: [flat.dtype])
            lock.withLock { castLaunchCache[key] = made }
            launch = made
        }
        return launch(castInputs)[0]
    }

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
        guard let cast = preparedFP8Cast(layers, flat: flat, count: rows * inputs) else {
            return nil
        }
        let lead = Array(x.shape.dropLast())
        return zip(fp8Run(layers, flat, cast: cast, rows: rows, sharedInput: false), layers).map {
            $0.reshaped(lead + [$1.outputDimensions])
        }
    }

    // MARK: Dense BF16/FP16

    static let denseHelper = """

        template <typename T, int K, int V, int TM, bool SPLIT, int N, int NOUT, int PF, bool VO>
        inline void cbv2_sp_dense(
            const device T* mat, const device T* in_vec, device T* out_vec,
            int tile, uint simd_gid, uint simd_lid, threadgroup float* tgp) {
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
            int out_row = SPLIT ? tile * TM : (tile * 8 + int(simd_gid)) * TM;
            if (!SPLIT && out_row >= N) { return; }
            out_row = out_row + TM <= N ? out_row : N - TM;
            const int g = SPLIT ? int(simd_gid) : 0;
            int bn = (32 * g + int(simd_lid)) * TN;
            const device T* rows = mat + size_t(out_row) * K;
            float r[P2];
            CBV2_SP_UNROLL
            for (int i = 0; i < P2; i++) { r[i] = 0; }
            constexpr int PD = PF <= 1 ? 1 : (PF >= n_iter ? (n_iter > 0 ? n_iter : 1) : PF);
            for (int i0 = 0; i0 < n_iter; i0 += PD) {
              ushort4 wblk[PD][TM];
              CBV2_SP_UNROLL
              for (int d = 0; d < PD; d++) {
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                  wblk[d][tm] = i0 + d < n_iter
                      ? ushort4(*(const device packed_ushort4*)(rows + tm * K + bn + d * blockN))
                      : ushort4(0);
                }
              }
              CBV2_SP_UNROLL
              for (int d = 0; d < PD; d++) {
                if (i0 + d >= n_iter) { break; }
                ushort4 av[V];
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                  av[v] = ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn));
                }
                if (VO) {
                  CBV2_SP_UNROLL
                  for (int v = 0; v < V; v++) {
                    CBV2_SP_UNROLL
                    for (int tn = 0; tn < TN; tn++) {
                      const float a = static_cast<float>(as_type<T>(av[v][tn]));
                      CBV2_SP_UNROLL
                      for (int tm = 0; tm < TM; tm++) {
                        r[v * TM + tm] += as_type<T>(wblk[d][tm][tn]) * a;
                      }
                    }
                  }
                } else {
                  CBV2_SP_UNROLL
                  for (int tn = 0; tn < TN; tn++) {
                    T inter[TM];
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                      inter[tm] = as_type<T>(wblk[d][tm][tn]);
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
                  v_coeff[v][tn] = bn + tn < K ? static_cast<float>(in_vec[v * K + bn + tn]) : 0.0f;
                }
              }
              for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                  inter[tn] = bn + tn < K ? static_cast<T>(rows[tm * K + bn + tn]) : T(0);
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
            const bool holds = (simd_lid & LOW) == 0;
            const int i = int(simd_lid >> SH);
            if (SPLIT) {
              if (holds) { tgp[g * 32 + i] = r[0]; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (g == 0 && holds && i < P) {
                float result = r[0];
                for (int s = 1; s < BN; s++) { result += tgp[s * 32 + i]; }
                out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(result);
              }
            } else if (holds && i < P) {
              out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(r[0]);
            }
        }
        """

    /// `CBV2_DENSE_ROWGROUPS=0`: disables `RG=4` multi-row-group dense BF16/FP16 verify projections
    /// (`V >= 6`, low-acceptance only, 0-ULP verified at first eligible call).
    public static let denseRowGroupsEnvEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_DENSE_ROWGROUPS"] != "0"

    nonisolated(unsafe) private static var denseRowGroupsChecked = false
    nonisolated(unsafe) private static var denseRowGroupsVerified = false

    static let denseRowGroupHelper = """

        template <typename T, int K, int V, int TM, int N, int NOUT, int PF>
        inline void cbv2_sp_dense_rg(
            const device T* mat, const device T* in_vec, device T* out_vec,
            int tile, uint g, uint simd_lid, threadgroup float* tgp) {
            constexpr int BN = 8;
            constexpr int TN = 4;
            constexpr int blockN = BN * 32 * TN;
            constexpr int n_iter = K / blockN;
            constexpr int leftover = K - blockN * n_iter;
            constexpr int P = V * TM;
            static_assert(P <= 32, "at most 32 values per lane");
            constexpr int P2 = P <= 1 ? 1 : P <= 2 ? 2 : P <= 4 ? 4 : P <= 8 ? 8 : P <= 16 ? 16 : 32;
            constexpr int SH = P2 == 1 ? 5 : P2 == 2 ? 4 : P2 == 4 ? 3 : P2 == 8 ? 2 : P2 == 16 ? 1 : 0;
            constexpr uint LOW = (1u << SH) - 1u;
            int out_row = tile * TM;
            if (out_row >= N) { return; }
            out_row = out_row + TM <= N ? out_row : N - TM;
            int bn = (32 * int(g) + int(simd_lid)) * TN;
            const device T* rows = mat + size_t(out_row) * K;
            float r[P2];
            CBV2_SP_UNROLL
            for (int i = 0; i < P2; i++) { r[i] = 0; }
            constexpr int PD = PF <= 1 ? 1 : (PF >= n_iter ? (n_iter > 0 ? n_iter : 1) : PF);
            for (int i0 = 0; i0 < n_iter; i0 += PD) {
              ushort4 wblk[PD][TM];
              CBV2_SP_UNROLL
              for (int d = 0; d < PD; d++) {
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                  wblk[d][tm] = i0 + d < n_iter
                      ? ushort4(*(const device packed_ushort4*)(rows + tm * K + bn + d * blockN))
                      : ushort4(0);
                }
              }
              CBV2_SP_UNROLL
              for (int d = 0; d < PD; d++) {
                if (i0 + d >= n_iter) { break; }
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                  const ushort4 av = ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn));
                  CBV2_SP_UNROLL
                  for (int tn = 0; tn < TN; tn++) {
                    const float a = static_cast<float>(as_type<T>(av[tn]));
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                      r[v * TM + tm] += as_type<T>(wblk[d][tm][tn]) * a;
                    }
                  }
                }
                bn += blockN;
              }
            }
            if (leftover > 0) {
              if (bn + TN <= K) {
                ushort4 wtail[TM];
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                  wtail[tm] = ushort4(*(const device packed_ushort4*)(rows + tm * K + bn));
                }
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                  const ushort4 av = ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn));
                  CBV2_SP_UNROLL
                  for (int tn = 0; tn < TN; tn++) {
                    const float a = static_cast<float>(as_type<T>(av[tn]));
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                      r[v * TM + tm] += as_type<T>(wtail[tm][tn]) * a;
                    }
                  }
                }
              } else if (bn < K) {
                for (int v = 0; v < V; v++) {
                  for (int tn = 0; tn < TN && bn + tn < K; tn++) {
                    const float a = static_cast<float>(in_vec[v * K + bn + tn]);
                    for (int tm = 0; tm < TM; tm++) {
                      r[v * TM + tm] += static_cast<T>(rows[tm * K + bn + tn]) * a;
                    }
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
            const bool holds = (simd_lid & LOW) == 0;
            const int i = int(simd_lid >> SH);
            if (holds) { tgp[g * 32 + i] = r[0]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (g == 0 && holds && i < P) {
              float result = r[0];
              for (int s = 1; s < BN; s++) { result += tgp[s * 32 + i]; }
              out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(result);
            }
        }
        """

    struct DenseSegment: Hashable {
        var rows: Int
        var split: Bool
        var tm: Int
        var prefetch: Int
        var tiles: Int { split ? (rows + tm - 1) / tm : (rows + 8 * tm - 1) / (8 * tm) }
    }

    struct DenseShape: Hashable {
        var rowsOuter: Bool
        var segments: [DenseSegment]
    }

    private struct DenseLaunchKey: Hashable {
        let shape: DenseShape
        let dtype: DType
        let inputs: Int
        let rows: Int
        let rowGroups: Int
    }

    nonisolated(unsafe) private static var denseKernels: [DenseShape: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var denseRGKernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var denseLaunchCache:
        [DenseLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    private static func denseRowGroupKernel(_ shape: DenseShape, rowGroups rg: Int) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        let cacheKey = "\(rg)_" + shape.segments.map { "\($0.rows)_\($0.tm)_\($0.prefetch)" }.joined(separator: "_")
        if let existing = denseRGKernels[cacheKey] { return existing }
        var inputs = (0 ..< shape.segments.count).map { "m\($0)" }
        inputs.append("xc")
        let outputs = (0 ..< shape.segments.count).map { "y\($0)" }
        var body = """
            threadgroup float sp_tgp_all[\(rg) * 8 * 32];
            const uint rg_idx = simdgroup_index_in_threadgroup / 8;
            const uint sg = simdgroup_index_in_threadgroup % 8;
            const uint sl = thread_index_in_simdgroup;
            const int tg = int(threadgroup_position_in_grid.y) * \(rg) + int(rg_idx);
            threadgroup float* sp_tgp = sp_tgp_all + rg_idx * 256;

            """
        var offset = 0
        for (i, s) in shape.segments.enumerated() {
            let end = offset + s.tiles
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(end)) {
                    cbv2_sp_dense_rg<T, K, V, \(s.tm), \(s.rows), \(s.rows), \(s.prefetch)>(
                        m\(i), xc, y\(i), tg - \(offset), sg, sl, sp_tgp);
                }

                """
            offset = end
        }
        let name = "cbv2_small_dense_rg\(rg)_" + stableName(body)
        let made = MLXFast.metalKernel(
            name: name, inputNames: inputs, outputNames: outputs, source: body,
            header: ModelOptActivationKernels.header + fp8Helper + denseHelper + denseRowGroupHelper)
        denseRGKernels[cacheKey] = made
        return made
    }

    private static func denseLaunch(
        shape: DenseShape, dtype: DType, inputs: Int, rows: Int, rowGroups rg: Int
    ) -> MLXFast.MLXFastKernel.Launch {
        let launchKey = DenseLaunchKey(
            shape: shape, dtype: dtype, inputs: inputs, rows: rows, rowGroups: rg)
        if let hit = lock.withLock({ denseLaunchCache[launchKey] }) { return hit }
        let tiles = shape.segments.reduce(0) { $0 + $1.tiles }
        let made: MLXFast.MLXFastKernel.Launch
        if rg > 1 {
            let groups = (tiles + rg - 1) / rg
            made = denseRowGroupKernel(shape, rowGroups: rg).prepare(
                template: [("T", dtype), ("K", inputs), ("V", rows)],
                grid: (32, 8 * rg * groups, 1), threadGroup: (32, 8 * rg, 1),
                outputShapes: shape.segments.map { [rows, $0.rows] },
                outputDTypes: Array(repeating: dtype, count: shape.segments.count))
        } else {
            made = denseKernel(shape).prepare(
                template: [("T", dtype), ("K", inputs), ("V", rows)],
                grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                outputShapes: shape.segments.map { [rows, $0.rows] },
                outputDTypes: Array(repeating: dtype, count: shape.segments.count))
        }
        lock.withLock { denseLaunchCache[launchKey] = made }
        return made
    }

    private static func ensureDenseRowGroupsVerified(
        shape: DenseShape, arrays: [MLXArray], dtype: DType, inputs: Int, rows: Int, rowGroups rg: Int
    ) -> Bool {
        let (checked, verified) = lock.withLock { (denseRowGroupsChecked, denseRowGroupsVerified) }
        if checked { return verified }
        var vals = [Float]()
        vals.reserveCapacity(rows * inputs)
        for i in 0 ..< (rows * inputs) {
            vals.append(Float(((i * 13 + 5) % 29) - 14) * 0.0625)
        }
        let testX = MLXArray(vals, [rows, inputs]).asType(dtype)
        var testArrays = Array(arrays.dropLast())
        testArrays.append(testX)
        let ok: Bool =
            (try? MLX.withError { () -> Bool in
                let ref = denseLaunch(
                    shape: shape, dtype: dtype, inputs: inputs, rows: rows, rowGroups: 1)(testArrays)
                let got = denseLaunch(
                    shape: shape, dtype: dtype, inputs: inputs, rows: rows, rowGroups: rg)(testArrays)
                var totalDiff = MLXArray(Int32(0))
                for (a, b) in zip(got, ref) {
                    totalDiff = totalDiff + (a.view(dtype: .uint16) .!= b.view(dtype: .uint16)).asType(.int32).sum()
                }
                eval(totalDiff)
                return totalDiff.item(Int.self) == 0
            }) ?? false
        lock.withLock {
            denseRowGroupsChecked = true
            denseRowGroupsVerified = ok
        }
        return ok
    }

    private static func denseKernel(_ shape: DenseShape) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let existing = denseKernels[shape] { return existing }
        var inputs = (0 ..< shape.segments.count).map { "m\($0)" }
        inputs.append("xc")
        let outputs = (0 ..< shape.segments.count).map { "y\($0)" }
        var body = """
            threadgroup float sp_tgp[8 * 32];
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = 0
        for (i, s) in shape.segments.enumerated() {
            let end = offset + s.tiles
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(end)) {
                    cbv2_sp_dense<T, K, V, \(s.tm), \(s.split), \(s.rows), \(s.rows), \(s.prefetch), \(shape.rowsOuter)>(
                        m\(i), xc, y\(i), tg - \(offset), sg, sl, sp_tgp);
                }

                """
            offset = end
        }
        let name = "cbv2_small_dense_" + stableName(body)
        let made = MLXFast.metalKernel(
            name: name, inputNames: inputs, outputNames: outputs, source: body,
            header: ModelOptActivationKernels.header + fp8Helper + denseHelper)
        denseKernels[shape] = made
        return made
    }

    /// Grouped row-exact products of dense BF16/FP16 `Linear` siblings (e.g. dense attention q/k/v/gate),
    /// bit-identical to `ModelOptRowExact.matmul` for 1 to 8 rows.
    public static func denseProducts(_ layers: [Linear], _ x: MLXArray) -> [MLXArray]? {
        let inputs = x.dim(-1)
        guard inputs > 64, inputs % 4 == 0, !layers.isEmpty,
            x.dtype == .bfloat16 || x.dtype == .float16
        else { return nil }
        let rows = x.size / inputs
        guard (1 ... maximumRows).contains(rows), ModelOptRowExact.geometryOverride == nil else {
            return nil
        }
        var segments = [DenseSegment]()
        var arrays = [MLXArray]()
        segments.reserveCapacity(layers.count)
        arrays.reserveCapacity(layers.count + 1)
        for layer in layers {
            guard layer.bias == nil, layer.weight.ndim == 2,
                layer.weight.dtype == x.dtype, layer.weight.dim(1) == inputs,
                layer.weight.dim(0) >= 8
            else { return nil }
            let outputs = layer.weight.dim(0)
            let split = inputs >= 16 * outputs
            let tm = min(rowsPerThread(rows: rows, split: split), outputs, max(1, 32 / rows))
            let pf = prefetchOverride ?? prefetchDepth(rows: rows, split: split)
            segments.append(DenseSegment(rows: outputs, split: split, tm: tm, prefetch: pf))
            arrays.append(layer.weight)
        }
        let flat = x.reshaped(rows, inputs)
        arrays.append(flat)
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        let deepLow = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.xnorDraftHeadEligible
        let useRowsOuter =
            rowsOuter || fp8RowsOuterMode == 2
            || (fp8RowsOuterMode == 1 && rows >= 5 && lowVerify)
        let shape = DenseShape(rowsOuter: useRowsOuter, segments: segments)
        let tiles = segments.reduce(0) { $0 + $1.tiles }
        let allSplit = segments.allSatisfy(\.split)
        var rg = 1
        if deepLow && denseRowGroupsEnvEnabled && rows >= 6 && allSplit && tiles >= 4 && tiles % 4 == 0 {
            if ensureDenseRowGroupsVerified(
                shape: shape, arrays: arrays, dtype: flat.dtype, inputs: inputs, rows: rows,
                rowGroups: 4)
            {
                rg = 4
            }
        }
        let launch = denseLaunch(
            shape: shape, dtype: flat.dtype, inputs: inputs, rows: rows, rowGroups: rg)
        let lead = Array(x.shape.dropLast())
        return zip(launch(arrays), segments).map { $0.reshaped(lead + [$1.rows]) }
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

    private struct NVFP4LaunchKey: Hashable {
        let segments: [NVFP4Segment]
        let dtype: DType
        let rows: Int
        let inputs: Int
    }

    nonisolated(unsafe) private static var nvfp4LaunchCache:
        [NVFP4LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

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
        arrays.reserveCapacity(layers.count * 3 + 1)
        segments.reserveCapacity(layers.count)
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
        let launchKey = NVFP4LaunchKey(segments: segments, dtype: x.dtype, rows: rows, inputs: inputs)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = lock.withLock({ nvfp4LaunchCache[launchKey] }) {
            launch = hit
        } else {
            let tiles = segments.reduce(0) { $0 + $1.tiles }
            let made = nvfp4Kernel(segments).prepare(
                template: [("T", x.dtype), ("V", rows), ("K", inputs)],
                grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                outputShapes: layers.map { [rows, $0.outputDimensions] },
                outputDTypes: Array(repeating: x.dtype, count: layers.count))
            lock.withLock { nvfp4LaunchCache[launchKey] = made }
            launch = made
        }
        let results = launch(arrays)
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

    // MARK: - Fused FP8 Gate + Up + SiLU * Up (+ inline FP8 Cast)

    private static let fp8GateUpSiluHeader = ModelOptActivationKernels.header + fp8Helper + """

        inline bfloat16_t cbv2_sp_bf16_sigmoid(bfloat16_t x) {
            bfloat16_t ax = (x < bfloat16_t(0.0f)) ? -x : x;
            bfloat16_t ey = static_cast<bfloat16_t>(metal::precise::exp(static_cast<float>(ax)));
            bfloat16_t one = bfloat16_t(1.0f);
            bfloat16_t y = one / (one + ey);
            return (x < bfloat16_t(0.0f)) ? y : (one - y);
        }

        inline bfloat16_t cbv2_sp_bf16_cast_fp8(
            bfloat16_t yb, float m0, float m1, float bound, float nan_value
        ) {
            float scaled = static_cast<float>(yb) * m0;
            uint8_t bits = modelopt_e4m3_bits(scaled);
            float decoded = (bits & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(bits);
            bool invalid = metal::isnan(scaled) || metal::abs(scaled) > bound;
            return static_cast<bfloat16_t>(invalid ? nan_value : decoded * m1);
        }
        """

    private static let fp8GateUpSiluSource = """
        constexpr int BN = 8;
        constexpr int TN = 4;
        constexpr int TM = 2; // 0 = gate, 1 = up for the same output row
        constexpr int blockN = BN * 32 * TN;
        constexpr int n_iter = K / blockN;
        constexpr int leftover = K - blockN * n_iter;
        constexpr int P = V * TM;
        static_assert(P <= 32, "at most 32 values per lane");
        constexpr int P2 = P <= 1 ? 1 : P <= 2 ? 2 : P <= 4 ? 4 : P <= 8 ? 8 : P <= 16 ? 16 : 32;
        constexpr int SH = P2 == 1 ? 5 : P2 == 2 ? 4 : P2 == 4 ? 3 : P2 == 8 ? 2 : P2 == 16 ? 1 : 0;
        constexpr uint LOW = (1u << SH) - 1u;
        threadgroup float tgp[8 * 32];
        const int out_row = int(threadgroup_position_in_grid.y);
        const uint g = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const float nan_value = metal::numeric_limits<float>::quiet_NaN();
        const float ws_g = wsg[0];
        const float ws_u = wsu[0];
        const float dscale_g = (DV_MODE != 0) ? ws_g * 256.0f : ws_g;
        const float dscale_u = (DV_MODE != 0) ? ws_u * 256.0f : ws_u;
        int bn = (32 * int(g) + int(simd_lid)) * TN;
        const device uint8_t* row_g = mg + size_t(out_row) * K;
        const device uint8_t* row_u = mu + size_t(out_row) * K;
        float r[P2];
        CBV2_SP_UNROLL
        for (int i = 0; i < P2; i++) {
            r[i] = 0;
        }
        for (int i0 = 0; i0 < n_iter; i0++) {
            const uint wg = *(const device uint*)(row_g + bn);
            const uint wu = *(const device uint*)(row_u + bn);
            ushort4 av[V];
            CBV2_SP_UNROLL
            for (int v = 0; v < V; v++) {
                av[v] = ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn));
            }
            float dec[2][TN];
            const uint ws_words[2] = { wg, wu };
            const float dscales[2] = { dscale_g, dscale_u };
            CBV2_SP_UNROLL
            for (int tm = 0; tm < 2; tm++) {
                const uint w = ws_words[tm];
                const float ds = dscales[tm];
                if constexpr (DV_MODE == 2) {
                    const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                    const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                    const float2 f02 = float2(as_type<half2>(lo)) * ds;
                    const float2 f13 = float2(as_type<half2>(hi)) * ds;
                    dec[tm][0] = f02.x;
                    dec[tm][1] = f13.x;
                    dec[tm][2] = f02.y;
                    dec[tm][3] = f13.y;
                } else if constexpr (DV_MODE == 1) {
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
                T inter[2][TN];
                CBV2_SP_UNROLL
                for (int tm = 0; tm < 2; tm++) {
                    const float ds = dscales[tm];
                    CBV2_SP_UNROLL
                    for (int tn = 0; tn < TN; tn++) {
                        inter[tm][tn] = static_cast<T>(
                            (DV_MODE == 2) ? dec[tm][tn] : (dec[tm][tn] * ds));
                    }
                }
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                    CBV2_SP_UNROLL
                    for (int tn = 0; tn < TN; tn++) {
                        const float a = static_cast<float>(as_type<T>(av[v][tn]));
                        r[v * 2 + 0] += inter[0][tn] * a;
                        r[v * 2 + 1] += inter[1][tn] * a;
                    }
                }
            } else {
                CBV2_SP_UNROLL
                for (int tn = 0; tn < TN; tn++) {
                    const T ig = static_cast<T>(
                        (DV_MODE == 2) ? dec[0][tn] : (dec[0][tn] * dscale_g));
                    const T iu = static_cast<T>(
                        (DV_MODE == 2) ? dec[1][tn] : (dec[1][tn] * dscale_u));
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        const float a = static_cast<float>(as_type<T>(av[v][tn]));
                        r[v * 2 + 0] += ig * a;
                        r[v * 2 + 1] += iu * a;
                    }
                }
            }
            bn += blockN;
        }
        if constexpr (leftover > 0) {
            if constexpr (leftover % 128 == 0) {
                if (bn < K) {
                    const uint wg = *(const device uint*)(row_g + bn);
                    const uint wu = *(const device uint*)(row_u + bn);
                    ushort4 av[V];
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        av[v] = ushort4(*(const device packed_ushort4*)(in_vec + v * K + bn));
                    }
                    float dec[2][TN];
                    const uint ws_words[2] = { wg, wu };
                    const float dscales[2] = { dscale_g, dscale_u };
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < 2; tm++) {
                        const uint w = ws_words[tm];
                        const float ds = dscales[tm];
                        if constexpr (DV_MODE == 2) {
                            const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                            const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                            const float2 f02 = float2(as_type<half2>(lo)) * ds;
                            const float2 f13 = float2(as_type<half2>(hi)) * ds;
                            dec[tm][0] = f02.x;
                            dec[tm][1] = f13.x;
                            dec[tm][2] = f02.y;
                            dec[tm][3] = f13.y;
                        } else if constexpr (DV_MODE == 1) {
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
                    CBV2_SP_UNROLL
                    for (int tn = 0; tn < TN; tn++) {
                        const T ig = static_cast<T>(
                            (DV_MODE == 2) ? dec[0][tn] : (dec[0][tn] * dscale_g));
                        const T iu = static_cast<T>(
                            (DV_MODE == 2) ? dec[1][tn] : (dec[1][tn] * dscale_u));
                        CBV2_SP_UNROLL
                        for (int v = 0; v < V; v++) {
                            const float a = static_cast<float>(as_type<T>(av[v][tn]));
                            r[v * 2 + 0] += ig * a;
                            r[v * 2 + 1] += iu * a;
                        }
                    }
                } else {
                    CBV2_SP_UNROLL
                    for (int i = 0; i < P; i++) {
                        r[i] += 0.0f;
                    }
                }
            } else {
                thread float v_coeff[V][TN];
                for (int v = 0; v < V; v++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] =
                            bn + tn < K ? static_cast<float>(in_vec[v * K + bn + tn]) : 0.0f;
                    }
                }
                for (int tn = 0; tn < TN; tn++) {
                    T ig = T(0), iu = T(0);
                    if (bn + tn < K) {
                        const uint8_t cg = row_g[bn + tn];
                        const uint8_t cu = row_u[bn + tn];
                        ig = static_cast<T>(((cg & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(cg)) * ws_g);
                        iu = static_cast<T>(((cu & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(cu)) * ws_u);
                    }
                    for (int v = 0; v < V; v++) {
                        r[v * 2 + 0] += ig * v_coeff[v][tn];
                        r[v * 2 + 1] += iu * v_coeff[v][tn];
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
        const bool holds = (simd_lid & LOW) == 0;
        const int i = int(simd_lid >> SH);
        if (holds) {
            tgp[g * 32 + i] = r[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0) {
            float result = r[0];
            if (holds && i < P) {
                for (int s = 1; s < BN; s++) {
                    result += tgp[s * 32 + i];
                }
            }
            const float peer = simd_shuffle_xor(result, ushort(1u << SH));
            if (holds && (i & 1) == 0 && i < P) {
                const bfloat16_t g_bf = static_cast<bfloat16_t>(result);
                const bfloat16_t u_bf = static_cast<bfloat16_t>(peer);
                bfloat16_t yb = (g_bf * cbv2_sp_bf16_sigmoid(g_bf)) * u_bf;
                if constexpr (CAST_FP8) {
                    yb = cbv2_sp_bf16_cast_fp8(yb, fp8_mu[0], fp8_mu[1], fp8_bd[0], nan_value);
                }
                out_vec[(i >> 1) * N + out_row] = yb;
            }
        }
        """

    private static let fp8GateUpSiluKernel = MLXFast.metalKernel(
        name: "cbv2_small_fp8_gate_up_silu",
        inputNames: ["mg", "wsg", "mu", "wsu", "in_vec", "fp8_mu", "fp8_bd"],
        outputNames: ["out_vec"],
        source: fp8GateUpSiluSource,
        header: fp8GateUpSiluHeader)

    private struct FP8GateUpLaunchKey: Hashable {
        let dtype: DType
        let inputs: Int
        let outputs: Int
        let rows: Int
        let dvMode: Int
        let rowsOuter: Bool
        let castFP8: Bool
    }

    nonisolated(unsafe) private static var fp8GateUpLaunchCache:
        [FP8GateUpLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static let dummyFP8Mu: MLXArray = {
        let a = MLXArray([Float(1), Float(1)])
        eval(a)
        return a
    }()
    nonisolated(unsafe) private static let dummyFP8Bound: MLXArray = {
        let a = MLXArray([Float(448)])
        eval(a)
        return a
    }()

    /// Fuses FP8 `gate_proj` + `up_proj` + `silu(gate) * up` (+ optional inline ModelOpt
    /// FP8 activation cast for `fp8Down`) on an already-cast shared input `[..., K]`
    /// (0-ULP bit-exact, zero intermediate `[V, N]` buffers).
    public static func fp8GateUpSiluPreCast(
        gate: ModelOptCalibratedLinear,
        up: ModelOptCalibratedLinear,
        preCast: MLXArray,
        fp8Down: ModelOptCalibratedLinear? = nil
    ) -> MLXArray? {
        guard CBv2VerifyNVFP4.fusedGateUpSiluEnabled, preCast.dtype == .bfloat16 else { return nil }
        let inputs = preCast.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = preCast.size / inputs
        let outputs = gate.outputDimensions
        guard (1 ... maximumRows).contains(rows),
            up.outputDimensions == outputs,
            fp8Takes(gate, inputs: inputs, dtype: preCast.dtype),
            fp8Takes(up, inputs: inputs, dtype: preCast.dtype)
        else { return nil }
        let gRuns = fp8Runs(gate)
        let uRuns = fp8Runs(up)
        guard gRuns.count == 1, uRuns.count == 1,
            gRuns[0].split, uRuns[0].split,
            gRuns[0].rows == 0 ..< outputs, uRuns[0].rows == 0 ..< outputs
        else { return nil }
        let muArr: MLXArray
        let bdArr: MLXArray
        let castFP8: Bool
        if let fp8Down {
            guard let constants = fp8Down.graphConstants,
                case .fp8(let m, let b) = constants.storage
            else { return nil }
            muArr = m
            bdArr = b
            castFP8 = true
        } else {
            muArr = dummyFP8Mu
            bdArr = dummyFP8Bound
            castFP8 = false
        }
        let deepLow = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.xnorDraftHeadEligible
        if deepLow && fp8RowGroupsEnvEnabled && rows >= 2 && outputs % 8 == 0
            && ensureFP8SwiGLURowGroupsVerified(
                gate: gate, up: up, muArr: muArr, bdArr: bdArr, castFP8: castFP8,
                dtype: preCast.dtype, inputs: inputs, outputs: outputs, rows: rows, rowGroups: 4)
        {
            let res = fp8SwiGLURGLaunch(
                dtype: preCast.dtype, inputs: inputs, outputs: outputs, rows: rows,
                castFP8: castFP8, rowGroups: 4)([
                    gate.weight, gate.graphConstants!.outputScale,
                    up.weight, up.graphConstants!.outputScale,
                    preCast.reshaped(rows, inputs), muArr, bdArr,
                ])[0]
            return res.reshaped(Array(preCast.shape.dropLast()) + [outputs])
        }
        let useRowsOuter =
            rowsOuter || fp8RowsOuterMode == 2
            || (fp8RowsOuterMode == 1 && rows >= 5
                && (!CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance))
        let dvMode = shouldUseBranchlessFP8(for: [gate, up]) ? 2 : (vectorDecode ? 1 : 0)
        let key = FP8GateUpLaunchKey(
            dtype: preCast.dtype, inputs: inputs, outputs: outputs, rows: rows,
            dvMode: dvMode, rowsOuter: useRowsOuter, castFP8: castFP8)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = lock.withLock({ fp8GateUpLaunchCache[key] }) {
            launch = hit
        } else {
            let made = fp8GateUpSiluKernel.prepare(
                template: [
                    ("T", preCast.dtype), ("K", inputs), ("V", rows), ("N", outputs),
                    ("DV_MODE", dvMode), ("VO", useRowsOuter), ("CAST_FP8", castFP8),
                ],
                grid: (32, 8 * outputs, 1), threadGroup: (32, 8, 1),
                outputShapes: [[rows, outputs]], outputDTypes: [preCast.dtype])
            lock.withLock { fp8GateUpLaunchCache[key] = made }
            launch = made
        }
        let res = launch([
            gate.weight, gate.graphConstants!.outputScale,
            up.weight, up.graphConstants!.outputScale,
            preCast.reshaped(rows, inputs), muArr, bdArr,
        ])[0]
        return res.reshaped(Array(preCast.shape.dropLast()) + [outputs])
    }
}
