// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import os

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

    /// The record's (c95a27f5) helper, verbatim: high-acceptance verify rounds
    /// (`CBv2DraftRegime.lowOnlyVerifySpeedups && !CBv2DraftRegime.lowAcceptance`)
    /// launch these kernels so their timing stays the record's.
    static let fp8HelperRec = """

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
        // stock BN = 8; RG row-groups share a threadgroup for L1 activation
        // reuse) or one simdgroup per TM rows (stock BN = 1, eight simdgroups
        // per threadgroup). `out_vec` is the output's first column for this
        // segment; NOUT its row stride.
        // DV_MODE: 0 = scalar e4m3 decode, 1 = NaN-checked half2 decode,
        // 2 = branchless half2 + float2*dscale direct to T inter[TM][TN]
        // (load-time verified NaN-free weights and non-overflowing outputScale).
        // COOP: when !SPLIT && V >= 2, double-buffered 4 KB cooperative s_av
        // across all 8 simdgroups (8x fewer device activation loads, 1 barrier/iter).
        template <typename T, int K, int V, int TM, bool SPLIT, int N, int NOUT, int DV_MODE, int PF, bool VO, int RG, bool COOP = false>
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
            static_assert(!COOP || V <= 8, "coop s_av requires V <= 8");
            constexpr int P2 = P <= 1 ? 1 : P <= 2 ? 2 : P <= 4 ? 4 : P <= 8 ? 8 : P <= 16 ? 16 : 32;
            constexpr int SH = P2 == 1 ? 5 : P2 == 2 ? 4 : P2 == 4 ? 3 : P2 == 8 ? 2 : P2 == 16 ? 1 : 0;
            constexpr uint LOW = (1u << SH) - 1u;
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            constexpr bool DV = (DV_MODE != 0);
            const float dscale = DV ? weight_scale * 256.0f : weight_scale;
            const uint rg = SPLIT ? (simd_gid / 8u) : 0u;
            const uint g = SPLIT ? (simd_gid % 8u) : 0u;
            const int orig_row = SPLIT ? (tile * RG + int(rg)) * TM : (tile * 8 + int(simd_gid)) * TM;
            if constexpr (!SPLIT && !(COOP && V >= 2)) {
                if (orig_row >= N) {
                    return;
                }
            }
            const bool active = orig_row < N;
            const int out_row = active ? (orig_row + TM <= N ? orig_row : N - TM) : (N - TM);
            int bn = (32 * int(g) + int(simd_lid)) * TN;
            const device uint8_t* rows = mat + size_t(out_row) * K;
            float r[P2];
            CBV2_SP_UNROLL
            for (int i = 0; i < P2; i++) {
                r[i] = 0;
            }
            // Weight words are loaded PD iterations at a time, ahead of
            // their use (more loads in flight; the arithmetic is unchanged).
            constexpr int PD = PF <= 1 ? 1 : (PF >= n_iter ? (n_iter > 0 ? n_iter : 1) : PF);
            if constexpr (COOP && !SPLIT && V >= 2 && n_iter > 0) {
                // All 8 simdgroups cooperatively stage V activation rows into
                // double-buffered 4 KB threadgroup SRAM (1 row per SG, 1 barrier/iter).
                threadgroup ushort4* s_av = (threadgroup ushort4*)tgp;
                constexpr int BUF_STRIDE = V * 32;
                if (int(simd_gid) < V) {
                    s_av[simd_gid * 32u + simd_lid] =
                        *(const device ushort4*)(in_vec + simd_gid * K + bn);
                }
                for (int it = 0; it < n_iter; it++) {
                    const int cur = it & 1;
                    const int nxt = cur ^ 1;
                    uint wq[TM];
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        wq[tm] = *(const device uint*)(rows + tm * K + bn);
                    }
                    T inter[TM][TN];
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        const uint w = wq[tm];
                        if constexpr (DV_MODE == 2) {
                            const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                            const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                            const float2 f02 = float2(as_type<half2>(lo)) * dscale;
                            const float2 f13 = float2(as_type<half2>(hi)) * dscale;
                            inter[tm][0] = static_cast<T>(f02.x);
                            inter[tm][1] = static_cast<T>(f13.x);
                            inter[tm][2] = static_cast<T>(f02.y);
                            inter[tm][3] = static_cast<T>(f13.y);
                        } else {
                            CBV2_SP_UNROLL
                            for (int tn = 0; tn < TN; tn++) {
                                const uint8_t code = uint8_t((w >> (8 * tn)) & 0xFF);
                                const float decoded =
                                    (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                                inter[tm][tn] = static_cast<T>(decoded * weight_scale);
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (it + 1 < n_iter && int(simd_gid) < V) {
                        s_av[nxt * BUF_STRIDE + simd_gid * 32u + simd_lid] =
                            *(const device ushort4*)(in_vec + simd_gid * K + bn + blockN);
                    }
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        const ushort4 av_v = s_av[cur * BUF_STRIDE + v * 32 + simd_lid];
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const float a = static_cast<float>(as_type<T>(av_v[tn]));
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                r[v * TM + tm] += inter[tm][tn] * a;
                            }
                        }
                    }
                    bn += blockN;
                }
            } else {
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
                thread const uint* wq = wblk[d];
                if constexpr (VO && (DV_MODE == 0 || DV_MODE == 2)) {
                    // Decode FP8 weights straight into `T inter[TM][TN]` (skipping
                    // the 16-float `dec` buffer) and load activations one row at a time.
                    T inter[TM][TN];
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        const uint w = wq[tm];
                        if constexpr (DV_MODE == 2) {
                            const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                            const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                            const float2 f02 = float2(as_type<half2>(lo)) * dscale;
                            const float2 f13 = float2(as_type<half2>(hi)) * dscale;
                            inter[tm][0] = static_cast<T>(f02.x);
                            inter[tm][1] = static_cast<T>(f13.x);
                            inter[tm][2] = static_cast<T>(f02.y);
                            inter[tm][3] = static_cast<T>(f13.y);
                        } else {
                            CBV2_SP_UNROLL
                            for (int tn = 0; tn < TN; tn++) {
                                const uint8_t code = uint8_t((w >> (8 * tn)) & 0xFF);
                                const float decoded =
                                    (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                                inter[tm][tn] = static_cast<T>(decoded * weight_scale);
                            }
                        }
                    }
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        const ushort4 av_v = *(const device ushort4*)(in_vec + v * K + bn);
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const float a = static_cast<float>(as_type<T>(av_v[tn]));
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                r[v * TM + tm] += inter[tm][tn] * a;
                            }
                        }
                    }
                } else {
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
                    if constexpr (DV_MODE == 2) {
                        const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                        const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                        const half2 h02 = as_type<half2>(lo);
                        const half2 h13 = as_type<half2>(hi);
                        dec[tm][0] = static_cast<float>(h02.x);
                        dec[tm][1] = static_cast<float>(h13.x);
                        dec[tm][2] = static_cast<float>(h02.y);
                        dec[tm][3] = static_cast<float>(h13.y);
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
                if constexpr (VO) {
                    // Rows of activations outermost, loaded one row at a time so
                    // `av` holds 1 ushort4 instead of V (saving 14 regs at V = 8).
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
                        const ushort4 av_v = *(const device ushort4*)(in_vec + v * K + bn);
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const float a = static_cast<float>(as_type<T>(av_v[tn]));
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                r[v * TM + tm] += inter[tm][tn] * a;
                            }
                        }
                    }
                } else {
                ushort4 av[V];
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                    av[v] = *(const device ushort4*)(in_vec + v * K + bn);
                }
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
                }
                bn += blockN;
              }
            }
            }
            if constexpr (leftover > 0) {
                T inter[TM][TN];
                if (bn + TN <= K) {
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        const uint w = *(const device uint*)(rows + tm * K + bn);
                        if constexpr (DV_MODE == 2) {
                            const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                            const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                            const float2 f02 = float2(as_type<half2>(lo)) * dscale;
                            const float2 f13 = float2(as_type<half2>(hi)) * dscale;
                            inter[tm][0] = static_cast<T>(f02.x);
                            inter[tm][1] = static_cast<T>(f13.x);
                            inter[tm][2] = static_cast<T>(f02.y);
                            inter[tm][3] = static_cast<T>(f13.y);
                        } else if constexpr (DV_MODE == 1) {
                            const uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
                            const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
                            const half2 h02 = as_type<half2>(lo);
                            const half2 h13 = as_type<half2>(hi);
                            float d0 = static_cast<float>(h02.x);
                            float d1 = static_cast<float>(h13.x);
                            float d2 = static_cast<float>(h02.y);
                            float d3 = static_cast<float>(h13.y);
                            if ((((w & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) != 0) {
                                if ((w & 0x7Fu) == 0x7Fu) { d0 = nan_value; }
                                if (((w >> 8) & 0x7Fu) == 0x7Fu) { d1 = nan_value; }
                                if (((w >> 16) & 0x7Fu) == 0x7Fu) { d2 = nan_value; }
                                if (((w >> 24) & 0x7Fu) == 0x7Fu) { d3 = nan_value; }
                            }
                            inter[tm][0] = static_cast<T>(d0 * dscale);
                            inter[tm][1] = static_cast<T>(d1 * dscale);
                            inter[tm][2] = static_cast<T>(d2 * dscale);
                            inter[tm][3] = static_cast<T>(d3 * dscale);
                        } else {
                            CBV2_SP_UNROLL
                            for (int tn = 0; tn < TN; tn++) {
                                const uint8_t code = uint8_t((w >> (8 * tn)) & 0xFF);
                                const float decoded =
                                    (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                                inter[tm][tn] = static_cast<T>(decoded * weight_scale);
                            }
                        }
                    }
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        const ushort4 av_v = *(const device ushort4*)(in_vec + v * K + bn);
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const float a = static_cast<float>(as_type<T>(av_v[tn]));
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                r[v * TM + tm] += inter[tm][tn] * a;
                            }
                        }
                    }
                } else {
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            if (bn + tn < K) {
                                const uint8_t code = rows[tm * K + bn + tn];
                                const float decoded =
                                    (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                                inter[tm][tn] = static_cast<T>(decoded * weight_scale);
                            } else {
                                inter[tm][tn] = T(0);
                            }
                        }
                    }
                    CBV2_SP_UNROLL
                    for (int v = 0; v < V; v++) {
                        CBV2_SP_UNROLL
                        for (int tn = 0; tn < TN; tn++) {
                            const float a =
                                bn + tn < K ? static_cast<float>(in_vec[v * K + bn + tn]) : 0.0f;
                            CBV2_SP_UNROLL
                            for (int tm = 0; tm < TM; tm++) {
                                r[v * TM + tm] += inter[tm][tn] * a;
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
                threadgroup float* row_tgp = tgp + rg * 256;
                if (holds) {
                    row_tgp[g * 32 + i] = r[0];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (g == 0 && holds && i < P && active) {
                    float result = r[0];
                    for (int s = 1; s < BN; s++) {
                        result += row_tgp[s * 32 + i];
                    }
                    out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(result);
                }
            } else if (holds && i < P && active) {
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

        func tiles(rowGroups: Int) -> Int {
            split ? (rows + rowGroups * tm - 1) / (rowGroups * tm) : (rows + 8 * tm - 1) / (8 * tm)
        }
    }

    struct FP8Shape: Hashable {
        var dvMode: Int
        var rowsOuter: Bool
        var rowGroups: Int
        var coopOProj: Bool
        var weights: Int
        var outputs: Int
        var segments: [FP8Segment]
        /// The record's kernel source (high-acceptance verify rounds).
        var record: Bool = false
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
        if shape.record {
            precondition(shape.rowGroups == 1, "record kernels have one row group")
            var body = """
                threadgroup float sp_tgp[8 * 32];
                const int tg = int(threadgroup_position_in_grid.y);
                const uint sg = simdgroup_index_in_threadgroup;
                const uint sl = thread_index_in_simdgroup;

                """
            var offset = 0
            for (index, s) in shape.segments.enumerated() {
                let end = offset + s.tiles(rowGroups: 1)
                body += """
                    \(index == 0 ? "" : "else ")if (tg < \(end)) {
                        cbv2_sp_fp8<T, K, V, \(s.tm), \(s.split), \(s.rows), \(s.outputWidth), \(shape.dvMode != 0), \(s.prefetch), \(shape.rowsOuter)>(
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
                header: ModelOptActivationKernels.header + fp8HelperRec)
            fp8Kernels[shape] = kernel
            return kernel
        }
        let rg = shape.rowGroups
        let tgpDecl =
            shape.coopOProj
            ? "alignas(8) threadgroup float sp_tgp[(\(rg) * 8 * 32) > (2 * V * 64) ? (\(rg) * 8 * 32) : (2 * V * 64)];"
            : "threadgroup float sp_tgp[\(rg) * 8 * 32];"
        var body = """
            \(tgpDecl)
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = 0
        for (index, s) in shape.segments.enumerated() {
            let end = offset + s.tiles(rowGroups: rg)
            let segCoop = shape.coopOProj && !s.split
            body += """
                \(index == 0 ? "" : "else ")if (tg < \(end)) {
                    cbv2_sp_fp8<T, K, V, \(s.tm), \(s.split), \(s.rows), \(s.outputWidth), \(shape.dvMode), \(s.prefetch), \(shape.rowsOuter), \(rg), \(segCoop)>(
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
    /// factor 256 moved into the scale.
    nonisolated(unsafe) static var vectorDecode = false
    /// `CBV2_FP8_DV=1` enables vectorized two-code E4M3 decode on low-acceptance verify rows.
    private static let vectorDecodeEnv: Bool =
        ProcessInfo.processInfo.environment["CBV2_FP8_DV"] == "1"
    /// `CBV2_FP8_BL_DV=0` disables branchless half2+float2 direct-to-inter decode on strict-low verify rows.
    public static let branchlessDVEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_FP8_BL_DV"] != "0"
    /// `CBV2_FP8_OPROJ_COOP=0` disables 4 KB double-buffered cooperative s_av on non-SPLIT (o_proj) strict-low verify rows.
    public static let coopOProjEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_FP8_OPROJ_COOP"] != "0"

    /// `CBV2_FP8_RG`: row-groups per threadgroup for pure-SPLIT FP8/dense verify tiles at rows >= 5 (1, 2, or 4; default 4).
    public static let fp8RowGroups: Int = {
        let raw = ProcessInfo.processInfo.environment["CBV2_FP8_RG"].flatMap(Int.init) ?? 4
        return (raw == 1 || raw == 2 || raw == 4) ? raw : 4
    }()

    /// `CBV2_FP8_O_PF`: prefetch depth override for non-SPLIT (`BN = 1`, e.g. `o_proj`) tiles on low-acceptance verify rows.
    private static let oPrefetchEnv: Int? =
        ProcessInfo.processInfo.environment["CBV2_FP8_O_PF"].flatMap(Int.init)

    /// Weight words loaded ahead, in iterations: none for K-split tiles
    /// (deeper measured slower: registers), 4 (up to 4 rows) or 2 for the
    /// BN = 1 rows (o projection: 32 iterations per lane).
    static func prefetchDepth(rows: Int, split: Bool, lowVerify: Bool = false) -> Int {
        if split { return 1 }
        if lowVerify, let oPF = oPrefetchEnv, oPF >= 1 { return oPF }
        return rows <= 4 ? 4 : 2
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

    private struct SharedCastKey: Hashable {
        let id0: ObjectIdentifier
        let idLast: ObjectIdentifier
        let count: Int
        let inputs: Int
        let isFloat16: Bool
    }

    private final class FastFP8CacheBox: @unchecked Sendable {
        var fp8: [FP8GroupPlanKey: FP8GroupPlan] = [:]
        var dense: [DenseGroupPlanKey: DenseGroupPlan] = [:]
        var sharedCast: [SharedCastKey: (multipliers: MLXArray, overflowBound: MLXArray)?] = [:]
        var branchlessSafe: Set<ObjectIdentifier> = []
    }

    private static let fastLock = OSAllocatedUnfairLock(initialState: FastFP8CacheBox())

    /// Batch-verifies once at boot that each FP8 layer's weight matrix has zero
    /// E4M3 NaN bytes (`0x7F`/`0xFF`) and a finite normal `outputScale` that
    /// does not overflow when multiplied by `256.0f` (0 ms per-step overhead).
    public static func warmFP8BranchlessSafe(_ layers: [ModelOptCalibratedLinear]) {
        guard branchlessDVEnabled, !layers.isEmpty else { return }
        var candidates: [(id: ObjectIdentifier, hasNaN: MLXArray, scale: MLXArray)] = []
        candidates.reserveCapacity(layers.count)
        var seen = Set<ObjectIdentifier>()
        for layer in layers {
            let id = ObjectIdentifier(layer)
            guard seen.insert(id).inserted,
                layer.format == .fp8, layer.weight.dtype == .uint8,
                let constants = layer.graphConstants,
                constants.outputScale.size == 1, constants.outputScale.dtype == .float32
            else { continue }
            if fastLock.withLockUnchecked({ $0.branchlessSafe.contains(id) }) { continue }
            let hasNaN = ((layer.weight & MLXArray(UInt8(0x7F))) .== MLXArray(UInt8(0x7F))).any()
            candidates.append((id: id, hasNaN: hasNaN, scale: constants.outputScale))
        }
        guard !candidates.isEmpty else { return }
        var toEval: [MLXArray] = []
        toEval.reserveCapacity(candidates.count * 2)
        for c in candidates {
            toEval.append(c.hasNaN)
            toEval.append(c.scale)
        }
        eval(toEval)
        let maxSafeScale = Float.greatestFiniteMagnitude / 256.0
        var verified = [ObjectIdentifier]()
        verified.reserveCapacity(candidates.count)
        for c in candidates {
            let nanFound = c.hasNaN.item(Bool.self)
            let s = c.scale.item(Float.self)
            if !nanFound && s.isFinite && s.isNormal && abs(s) <= maxSafeScale {
                verified.append(c.id)
            }
        }
        if !verified.isEmpty {
            fastLock.withLockUnchecked { box in
                for id in verified { box.branchlessSafe.insert(id) }
            }
        }
    }

    private static func allBranchlessSafe(_ layers: [ModelOptCalibratedLinear]) -> Bool {
        fastLock.withLockUnchecked { box in
            for layer in layers {
                if !box.branchlessSafe.contains(ObjectIdentifier(layer)) { return false }
            }
            return true
        }
    }

    /// Returns the shared `(multipliers, overflowBound)` when every FP8 layer
    /// in `layers` has bit-identical activation cast parameters (cached once
    /// per layer group; zero per-step host sync).
    public static func sharedFP8CastParams(
        _ layers: [ModelOptCalibratedLinear], inputs: Int, dtype: DType
    ) -> (multipliers: MLXArray, overflowBound: MLXArray)? {
        guard !layers.isEmpty, dtype == .bfloat16 || dtype == .float16 else { return nil }
        let skey = SharedCastKey(
            id0: ObjectIdentifier(layers[0]), idLast: ObjectIdentifier(layers[layers.count - 1]),
            count: layers.count, inputs: inputs, isFloat16: dtype == .float16)
        if ModelOptRowExact.geometryOverride == nil,
            let hit = fastLock.withLockUnchecked({ $0.sharedCast[skey] })
        {
            return hit
        }
        guard layers.allSatisfy({ fp8Takes($0, inputs: inputs, dtype: dtype) }) else {
            if ModelOptRowExact.geometryOverride == nil {
                fastLock.withLockUnchecked { $0.sharedCast[skey] = .some(nil) }
            }
            return nil
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
        if ModelOptRowExact.geometryOverride == nil {
            fastLock.withLockUnchecked { $0.sharedCast[skey] = result }
        }
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

    private struct FP8GroupPlanKey: Hashable {
        let id0: ObjectIdentifier
        let idLast: ObjectIdentifier
        let count: Int
        let inputs: Int
        let rows: Int
        let dtype: DType
        let rowsOuter: Bool
        let rowGroups: Int
        let dvMode: Int
        let coopOProj: Bool
        let lowVerify: Bool
        let sharedInput: Bool
    }

    private struct FP8GroupPlan {
        let launch: MLXFast.MLXFastKernel.Launch
        let weightArrays: [MLXArray]
        let outputDims: [Int]
    }

    private struct DenseGroupPlanKey: Hashable {
        let id0: ObjectIdentifier
        let idLast: ObjectIdentifier
        let count: Int
        let inputs: Int
        let rows: Int
        let dtype: DType
        let rowsOuter: Bool
        let rowGroups: Int
        let lowVerify: Bool
    }

    private struct DenseGroupPlan {
        let launch: MLXFast.MLXFastKernel.Launch
        let weightArrays: [MLXArray]
        let outputDims: [Int]
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
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        let strictLow = lowVerify && CBv2DraftRegime.strictLowVerifyEligible
        let useRowsOuter =
            rowsOuter || fp8RowsOuterMode == 2
            || (fp8RowsOuterMode == 1 && rows >= 5 && lowVerify)
        let useBranchlessDV =
            strictLow && branchlessDVEnabled && allBranchlessSafe(layers)
            && (useRowsOuter || (coopOProjEnabled && rows >= 2))
        let dvMode: Int =
            useBranchlessDV ? 2 : ((vectorDecode || (vectorDecodeEnv && lowVerify)) ? 1 : 0)
        let allowCoopOProj = strictLow && coopOProjEnabled && rows >= 2
        let targetRG = (lowVerify && rows >= 2) ? fp8RowGroups : 1
        let canUseFastPlan =
            lowVerify && rowsPerThreadOverride == nil && prefetchOverride == nil
            && ModelOptRowExact.geometryOverride == nil
        if canUseFastPlan {
            let planKey = FP8GroupPlanKey(
                id0: ObjectIdentifier(layers[0]), idLast: ObjectIdentifier(layers[layers.count - 1]),
                count: layers.count, inputs: inputs, rows: rows, dtype: flat.dtype,
                rowsOuter: useRowsOuter, rowGroups: targetRG, dvMode: dvMode,
                coopOProj: allowCoopOProj, lowVerify: lowVerify, sharedInput: sharedInput)
            if let plan = fastLock.withLockUnchecked({ $0.fp8[planKey] }) {
                var args = plan.weightArrays
                args.append(cast)
                return plan.launch(args)
            }
            var segments = [FP8Segment]()
            var weightArrays = [MLXArray]()
            weightArrays.reserveCapacity(layers.count * 2 + 1)
            for (i, layer) in layers.enumerated() {
                weightArrays += [layer.weight, layer.graphConstants!.outputScale]
                for run in fp8Runs(layer) {
                    let tm = min(
                        rowsPerThread(rows: rows, split: run.split), run.rows.count,
                        max(1, 32 / rows))
                    segments.append(
                        FP8Segment(
                            weight: i, row0: run.rows.lowerBound, rows: run.rows.count,
                            input: (layers.count == 1 || sharedInput) ? 0 : i, output: i,
                            column0: run.rows.lowerBound, outputWidth: layer.outputDimensions,
                            split: run.split, tm: tm,
                            prefetch: prefetchDepth(
                                rows: rows, split: run.split, lowVerify: lowVerify)))
                }
            }
            let rg = segments.allSatisfy(\.split) ? targetRG : 1
            let coopActive = allowCoopOProj && segments.contains(where: { !$0.split })
            let effectiveDVMode = (dvMode == 2 && !useRowsOuter && !coopActive) ? 0 : dvMode
            let shape = FP8Shape(
                dvMode: effectiveDVMode, rowsOuter: useRowsOuter, rowGroups: rg,
                coopOProj: coopActive, weights: layers.count, outputs: layers.count,
                segments: segments)
            let outputDims = layers.map(\.outputDimensions)
            let launchKey = FP8LaunchKey(
                shape: shape, dtype: flat.dtype, inputs: inputs, rows: rows, outputs: outputDims)
            let launch: MLXFast.MLXFastKernel.Launch
            if let hit = lock.withLock({ fp8LaunchCache[launchKey] }) {
                launch = hit
            } else {
                let tiles = segments.reduce(0) { $0 + $1.tiles(rowGroups: rg) }
                let made = fp8Kernel(shape).prepare(
                    template: [("T", flat.dtype), ("K", inputs), ("V", rows)],
                    grid: (32, 8 * rg * tiles, 1), threadGroup: (32, 8 * rg, 1),
                    outputShapes: outputDims.map { [rows, $0] },
                    outputDTypes: Array(repeating: flat.dtype, count: outputDims.count))
                lock.withLock { fp8LaunchCache[launchKey] = made }
                launch = made
            }
            let plan = FP8GroupPlan(
                launch: launch, weightArrays: weightArrays, outputDims: outputDims)
            fastLock.withLockUnchecked { $0.fp8[planKey] = plan }
            var args = weightArrays
            args.append(cast)
            return launch(args)
        }
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
                        prefetch: prefetchOverride
                            ?? prefetchDepth(rows: rows, split: run.split, lowVerify: lowVerify)))
            }
        }
        arrays.append(cast)
        let rg = segments.allSatisfy(\.split) ? targetRG : 1
        let coopActive = allowCoopOProj && segments.contains(where: { !$0.split })
        let effectiveDVMode = (dvMode == 2 && !useRowsOuter && !coopActive) ? 0 : dvMode
        let shape = FP8Shape(
            dvMode: effectiveDVMode, rowsOuter: useRowsOuter, rowGroups: rg,
            coopOProj: coopActive, weights: layers.count, outputs: layers.count,
            segments: segments, record: !lowVerify)
        let outputDims = layers.map(\.outputDimensions)
        let launchKey = FP8LaunchKey(
            shape: shape, dtype: flat.dtype, inputs: inputs, rows: rows, outputs: outputDims)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = lock.withLock({ fp8LaunchCache[launchKey] }) {
            launch = hit
        } else {
            let tiles = segments.reduce(0) { $0 + $1.tiles(rowGroups: rg) }
            let made = fp8Kernel(shape).prepare(
                template: [("T", flat.dtype), ("K", inputs), ("V", rows)],
                grid: (32, 8 * rg * tiles, 1), threadGroup: (32, 8 * rg, 1),
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
        guard (1 ... maximumRows).contains(rows) else { return nil }
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        let strictLow = lowVerify && CBv2DraftRegime.strictLowVerifyEligible
        let useRowsOuter =
            rowsOuter || fp8RowsOuterMode == 2
            || (fp8RowsOuterMode == 1 && rows >= 5 && lowVerify)
        let useBranchlessDV =
            strictLow && branchlessDVEnabled && allBranchlessSafe(layers)
            && (useRowsOuter || (coopOProjEnabled && rows >= 2))
        let dvMode: Int =
            useBranchlessDV ? 2 : ((vectorDecode || (vectorDecodeEnv && lowVerify)) ? 1 : 0)
        let allowCoopOProj = strictLow && coopOProjEnabled && rows >= 2
        let targetRG = (lowVerify && rows >= 2) ? fp8RowGroups : 1
        if rowsPerThreadOverride == nil && prefetchOverride == nil
            && ModelOptRowExact.geometryOverride == nil
        {
            let planKey = FP8GroupPlanKey(
                id0: ObjectIdentifier(layers[0]), idLast: ObjectIdentifier(layers[layers.count - 1]),
                count: layers.count, inputs: inputs, rows: rows, dtype: preCast.dtype,
                rowsOuter: useRowsOuter, rowGroups: targetRG, dvMode: dvMode,
                coopOProj: allowCoopOProj, lowVerify: lowVerify, sharedInput: true)
            if let plan = fastLock.withLockUnchecked({ $0.fp8[planKey] }) {
                let flat = preCast.reshaped(rows, inputs)
                var args = plan.weightArrays
                args.append(flat)
                let raw = plan.launch(args)
                if preCast.ndim == 3 {
                    let b = preCast.dim(0), l = preCast.dim(1)
                    return (0 ..< raw.count).map { raw[$0].reshaped(b, l, plan.outputDims[$0]) }
                }
                let lead = Array(preCast.shape.dropLast())
                return zip(raw, plan.outputDims).map { $0.reshaped(lead + [$1]) }
            }
        }
        guard layers.allSatisfy({ fp8Takes($0, inputs: inputs, dtype: preCast.dtype) })
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

    /// The record's (c95a27f5) helper, verbatim: high-acceptance verify rounds
    /// (`CBv2DraftRegime.lowOnlyVerifySpeedups && !CBv2DraftRegime.lowAcceptance`)
    /// launch these kernels so their timing stays the record's.
    static let denseHelperRec = """

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

    static let denseHelper = """

        template <typename T, int K, int V, int TM, bool SPLIT, int N, int NOUT, int PF, bool VO, int RG>
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
            const uint rg = SPLIT ? (simd_gid / 8u) : 0u;
            const uint g = SPLIT ? (simd_gid % 8u) : 0u;
            const int orig_row = SPLIT ? (tile * RG + int(rg)) * TM : (tile * 8 + int(simd_gid)) * TM;
            if (!SPLIT && orig_row >= N) { return; }
            const bool active = orig_row < N;
            const int out_row = active ? (orig_row + TM <= N ? orig_row : N - TM) : (N - TM);
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
                      ? *(const device ushort4*)(rows + tm * K + bn + d * blockN)
                      : ushort4(0);
                }
              }
              CBV2_SP_UNROLL
              for (int d = 0; d < PD; d++) {
                if (i0 + d >= n_iter) { break; }
                if constexpr (VO) {
                  CBV2_SP_UNROLL
                  for (int v = 0; v < V; v++) {
                    const ushort4 av_v = *(const device ushort4*)(in_vec + v * K + bn);
                    CBV2_SP_UNROLL
                    for (int tn = 0; tn < TN; tn++) {
                      const float a = static_cast<float>(as_type<T>(av_v[tn]));
                      CBV2_SP_UNROLL
                      for (int tm = 0; tm < TM; tm++) {
                        r[v * TM + tm] += as_type<T>(wblk[d][tm][tn]) * a;
                      }
                    }
                  }
                } else {
                  ushort4 av[V];
                  CBV2_SP_UNROLL
                  for (int v = 0; v < V; v++) {
                    av[v] = *(const device ushort4*)(in_vec + v * K + bn);
                  }
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
            if constexpr (leftover > 0) {
              if (bn + TN <= K) {
                ushort4 wv[TM];
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                  wv[tm] = *(const device ushort4*)(rows + tm * K + bn);
                }
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                  const ushort4 av_v = *(const device ushort4*)(in_vec + v * K + bn);
                  CBV2_SP_UNROLL
                  for (int tn = 0; tn < TN; tn++) {
                    const float a = static_cast<float>(as_type<T>(av_v[tn]));
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                      r[v * TM + tm] += as_type<T>(wv[tm][tn]) * a;
                    }
                  }
                }
              } else {
                T inter[TM][TN];
                CBV2_SP_UNROLL
                for (int tm = 0; tm < TM; tm++) {
                  CBV2_SP_UNROLL
                  for (int tn = 0; tn < TN; tn++) {
                    inter[tm][tn] = bn + tn < K ? static_cast<T>(rows[tm * K + bn + tn]) : T(0);
                  }
                }
                CBV2_SP_UNROLL
                for (int v = 0; v < V; v++) {
                  CBV2_SP_UNROLL
                  for (int tn = 0; tn < TN; tn++) {
                    const float a = bn + tn < K ? static_cast<float>(in_vec[v * K + bn + tn]) : 0.0f;
                    CBV2_SP_UNROLL
                    for (int tm = 0; tm < TM; tm++) {
                      r[v * TM + tm] += inter[tm][tn] * a;
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
            if (SPLIT) {
              threadgroup float* row_tgp = tgp + rg * 256;
              if (holds) { row_tgp[g * 32 + i] = r[0]; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (g == 0 && holds && i < P && active) {
                float result = r[0];
                for (int s = 1; s < BN; s++) { result += row_tgp[s * 32 + i]; }
                out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(result);
              }
            } else if (holds && i < P) {
              out_vec[(i / TM) * NOUT + out_row + (i % TM)] = static_cast<T>(r[0]);
            }
        }
        """

    struct DenseSegment: Hashable {
        var rows: Int
        var split: Bool
        var tm: Int
        var prefetch: Int
        func tiles(rowGroups: Int) -> Int {
            split ? (rows + rowGroups * tm - 1) / (rowGroups * tm) : (rows + 8 * tm - 1) / (8 * tm)
        }
    }

    struct DenseShape: Hashable {
        var rowsOuter: Bool
        var rowGroups: Int
        var segments: [DenseSegment]
        /// The record's kernel source (high-acceptance verify rounds).
        var record: Bool = false
    }

    private struct DenseLaunchKey: Hashable {
        let shape: DenseShape
        let dtype: DType
        let inputs: Int
        let rows: Int
    }

    nonisolated(unsafe) private static var denseKernels: [DenseShape: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var denseLaunchCache:
        [DenseLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    private static func denseKernel(_ shape: DenseShape) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let existing = denseKernels[shape] { return existing }
        var inputs = (0 ..< shape.segments.count).map { "m\($0)" }
        inputs.append("xc")
        let outputs = (0 ..< shape.segments.count).map { "y\($0)" }
        if shape.record {
            precondition(shape.rowGroups == 1, "record kernels have one row group")
            var body = """
                threadgroup float sp_tgp[8 * 32];
                const int tg = int(threadgroup_position_in_grid.y);
                const uint sg = simdgroup_index_in_threadgroup;
                const uint sl = thread_index_in_simdgroup;

                """
            var offset = 0
            for (i, s) in shape.segments.enumerated() {
                let end = offset + s.tiles(rowGroups: 1)
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
                header: ModelOptActivationKernels.header + fp8HelperRec + denseHelperRec)
            denseKernels[shape] = made
            return made
        }
        let rg = shape.rowGroups
        var body = """
            threadgroup float sp_tgp[\(rg) * 8 * 32];
            const int tg = int(threadgroup_position_in_grid.y);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint sl = thread_index_in_simdgroup;

            """
        var offset = 0
        for (i, s) in shape.segments.enumerated() {
            let end = offset + s.tiles(rowGroups: rg)
            body += """
                \(i == 0 ? "" : "else ")if (tg < \(end)) {
                    cbv2_sp_dense<T, K, V, \(s.tm), \(s.split), \(s.rows), \(s.rows), \(s.prefetch), \(shape.rowsOuter), \(rg)>(
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
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        let useRowsOuter =
            rowsOuter || fp8RowsOuterMode == 2
            || (fp8RowsOuterMode == 1 && rows >= 5 && lowVerify)
        let targetRG = (lowVerify && rows >= 2) ? fp8RowGroups : 1
        let canUseFastPlan = lowVerify && rowsPerThreadOverride == nil && prefetchOverride == nil
        if canUseFastPlan {
            let planKey = DenseGroupPlanKey(
                id0: ObjectIdentifier(layers[0]), idLast: ObjectIdentifier(layers[layers.count - 1]),
                count: layers.count, inputs: inputs, rows: rows, dtype: x.dtype,
                rowsOuter: useRowsOuter, rowGroups: targetRG, lowVerify: lowVerify)
            if let plan = fastLock.withLockUnchecked({ $0.dense[planKey] }) {
                var args = plan.weightArrays
                args.append(x.reshaped(rows, inputs))
                let raw = plan.launch(args)
                if x.ndim == 3 {
                    let b = x.dim(0), l = x.dim(1)
                    return (0 ..< raw.count).map { raw[$0].reshaped(b, l, plan.outputDims[$0]) }
                }
                let lead = Array(x.shape.dropLast())
                return zip(raw, plan.outputDims).map { $0.reshaped(lead + [$1]) }
            }
        }
        var segments = [DenseSegment]()
        var weightArrays = [MLXArray]()
        segments.reserveCapacity(layers.count)
        weightArrays.reserveCapacity(layers.count + 1)
        for layer in layers {
            guard layer.bias == nil, layer.weight.ndim == 2,
                layer.weight.dtype == x.dtype, layer.weight.dim(1) == inputs,
                layer.weight.dim(0) >= 8
            else { return nil }
            let outputs = layer.weight.dim(0)
            let split = inputs >= 16 * outputs
            let tm = min(rowsPerThread(rows: rows, split: split), outputs, max(1, 32 / rows))
            let pf = prefetchOverride ?? prefetchDepth(rows: rows, split: split, lowVerify: lowVerify)
            segments.append(DenseSegment(rows: outputs, split: split, tm: tm, prefetch: pf))
            weightArrays.append(layer.weight)
        }
        let rg = segments.allSatisfy(\.split) ? targetRG : 1
        let shape = DenseShape(
            rowsOuter: useRowsOuter, rowGroups: rg, segments: segments, record: !lowVerify)
        let launchKey = DenseLaunchKey(shape: shape, dtype: x.dtype, inputs: inputs, rows: rows)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = lock.withLock({ denseLaunchCache[launchKey] }) {
            launch = hit
        } else {
            let tiles = segments.reduce(0) { $0 + $1.tiles(rowGroups: rg) }
            let made = denseKernel(shape).prepare(
                template: [("T", x.dtype), ("K", inputs), ("V", rows)],
                grid: (32, 8 * rg * tiles, 1), threadGroup: (32, 8 * rg, 1),
                outputShapes: segments.map { [rows, $0.rows] },
                outputDTypes: Array(repeating: x.dtype, count: segments.count))
            lock.withLock { denseLaunchCache[launchKey] = made }
            launch = made
        }
        let outputDims = segments.map(\.rows)
        if canUseFastPlan {
            let planKey = DenseGroupPlanKey(
                id0: ObjectIdentifier(layers[0]), idLast: ObjectIdentifier(layers[layers.count - 1]),
                count: layers.count, inputs: inputs, rows: rows, dtype: x.dtype,
                rowsOuter: useRowsOuter, rowGroups: targetRG, lowVerify: lowVerify)
            let plan = DenseGroupPlan(
                launch: launch, weightArrays: weightArrays, outputDims: outputDims)
            fastLock.withLockUnchecked { $0.dense[planKey] = plan }
        }
        var arrays = weightArrays
        arrays.append(x.reshaped(rows, inputs))
        let raw = launch(arrays)
        if x.ndim == 3 {
            let b = x.dim(0), l = x.dim(1)
            return (0 ..< raw.count).map { raw[$0].reshaped(b, l, outputDims[$0]) }
        }
        let lead = Array(x.shape.dropLast())
        return zip(raw, outputDims).map { $0.reshaped(lead + [$1]) }
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
}