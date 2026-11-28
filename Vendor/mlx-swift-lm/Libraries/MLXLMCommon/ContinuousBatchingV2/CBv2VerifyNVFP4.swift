// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import os

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
/// - `TN` output fragments of 8 rows per simdgroup (`8 * TN` outputs per
///   threadgroup);
/// - `WC`: loads all 4 blocks of a 128-input chunk upfront and decodes two
///   E4M3 scales per op (`cbv2_vn_e4m3x2`), optionally via one aligned 8-byte
///   `uint2` scale load (`SV`);
/// - `PF 1`: rolling register buffer prefetching the next chunk's weight words
///   and packed scales for long 19,968-input rows;
/// - activations are kept as their raw bf16 bits until each MMA converts them.
/// Rows past V repeat row V - 1, as in the stock kernel; their columns are
/// never stored. Groups of sibling projections (MLP gate and up; attention
/// q, k, v and gate) run in one launch, one segment per projection.
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

    private static let env = ProcessInfo.processInfo.environment
    private static let prefetchEnv: Int? = env["CBV2_NVFP4_PF"].flatMap(Int.init)
    private static let downPrefetchEnv: Int? = env["CBV2_NVFP4_DOWN_PF"].flatMap(Int.init)
    private static let downTN1Env: Bool = env["CBV2_NVFP4_DOWN_TN1"] == "1"
    private static let peeledPFEnv: Bool = env["CBV2_NVFP4_PEELED_PF"] == "1"
    private static let fusedGUTN1Env: Bool = env["CBV2_NVFP4_FUSED_GU_TN1"] == "1"
    private static let smallTN1Env: Bool = env["CBV2_NVFP4_SMALL_TN1"] == "1"
    public static let fusedGateUpSiluEnabled: Bool = env["CBV2_NVFP4_FUSED_GU_SILU"] != "0"
    private static let wholeChunkEnv: Bool? = env["CBV2_NVFP4_WHOLE_CHUNK"].map { $0 != "0" }
    private static let scaleVecEnv: Bool? = env["CBV2_NVFP4_SV"].map { $0 != "0" }

    /// Tests and benchmarks pin a geometry here (nil: `geometry(rows:inputs:)`).
    nonisolated(unsafe) public static var geometryOverride: Geometry?

    /// Per-shape fragment and prefetch schedule.
    public static func geometry(rows: Int, inputs: Int) -> Geometry {
        if let geometryOverride { return geometryOverride }
        let pf = prefetchEnv ?? (inputs >= 16384 && rows >= 8 ? 1 : 0)
        if inputs >= 16384 && rows >= 7 { return Geometry(fragments: 2, prefetch: pf) }
        return Geometry(fragments: 1, prefetch: pf)
    }

    private static func schedule(
        rows: Int, inputs: Int, lowVerify: Bool
    ) -> (fragments: Int, prefetch: Int, wholeChunk: Bool, scaleVec: Bool) {
        if let o = geometryOverride {
            let wc = wholeChunkEnv ?? lowVerify
            let sv = scaleVecEnv ?? wc
            return (o.fragments, o.prefetch, wc, sv)
        }
        let wc = wholeChunkEnv ?? lowVerify
        let sv = scaleVecEnv ?? wc
        let strictC3 = lowVerify && CBv2DraftRegime.strictLowVerifyC3
        if inputs >= 16384 {
            // Strict-C3 schedule for mlp.down (K=19968, N=6656, 40L = 2.99 GB/rd):
            // TN=2 PF=0 with streamed block-at-a-time x loads + uint<<16 + float-domain
            // pair_f keeps maxT=832, halves duplicate [8, 19968] activation loads, and
            // runs +5.37% faster than TN=2 PF=2 (+8.2% vs TN=1 PF=2) under safe math (0-ULP).
            if strictC3 && rows >= 7 && wc {
                let tn = downTN1Env ? 1 : 2
                let defaultPF = peeledPFEnv ? 2 : 0
                let pf = prefetchEnv ?? (downPrefetchEnv ?? defaultPF)
                return (tn, pf, wc, sv)
            }
            // MLP down (K = 19968): WC + PF=1 + SV=1 on low-acceptance verify rows;
            // keep high-acceptance Pair 1 at PF=0 to stay safely below the decode ceiling.
            let tn = rows >= 7 ? 2 : 1
            let defaultPF = (lowVerify && wc) ? (downPrefetchEnv ?? (rows >= 8 ? 1 : 0)) : 0
            let pf = prefetchEnv ?? defaultPF
            return (tn, pf, wc, sv)
        }
        // Strict-C3 schedule for K < 16384 (qkvg 0.59 GB, o 0.15 GB, fallback gate_up):
        // TN=2 PF=0 with streamed x + pair_f halves duplicate activation loads and runs
        // +4.7..7% faster than TN=1 PF=2 under safe math (0-ULP).
        if strictC3 && rows >= 5 && wc {
            let tn = smallTN1Env ? 1 : 2
            let defaultPF = peeledPFEnv ? 2 : 0
            let pf = prefetchEnv ?? defaultPF
            return (tn, pf, wc, sv)
        }
        let pf = prefetchEnv ?? 0
        return (1, pf, wc, sv)
    }

    static let tileHelper = """

        inline half2 cbv2_vn_e4m3x2(uint x) {
            const uint v = x & 0x007F007Fu;
            const uint u = (v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7)
                | ((x & 0x00800080u) << 8);
            return as_type<half2>(u) * half2(256.0h);
        }

        // Exact float-domain equivalent of `mo_wo_pair(q, i, s)` with `fs = float(s) * 16384.0f`
        // (0 bit diffs across all 16 E2M1 x 254 finite E4M3 pairs; replaces 2 half2 muls with 1 float2 mul).
        inline float2 cbv2_vn_pair_f(uint q, int i, float fs) {
            uint p;
            if (i == 0) {
                p = ((q & 0x00070007u) << 9) | ((q & 0x00080008u) << 12);
            } else if (i == 1) {
                p = ((q & 0x00700070u) << 5) | ((q & 0x00800080u) << 8);
            } else if (i == 2) {
                p = ((q & 0x07000700u) << 1) | ((q & 0x08000800u) << 4);
            } else {
                p = ((q & 0x70007000u) >> 3) | (q & 0x80008000u);
            }
            return float2(as_type<half2>(p)) * fs;
        }

        // One tile of 8 * TN outputs of one projection: the stock matrix
        // kernel's arithmetic (see CBv2VerifyNVFP4).
        template <typename T, int V, int K, int N, int TN, int PF, bool SCALED, bool WC, bool SV>
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
            const uint sg = simd_gid;
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            const uint scale_shift = (fn & 4u) << 1;
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
                sl[t] = scales + size_t(row) * row_scales + (SV ? 0 : (fn / 4));
            }

            if constexpr (WC && PF == 0) {
                for (int k = int(sg) * BK; k < K; k += WK * BK) {
                    uint qa[4][TN];
                    MO_WO_UNROLL
                    for (short block = 0; block < 4; block++) {
                        const int b = k + 32 * block;
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) { qa[block][t] = wl[t][b / 8]; }
                    }
                    half2 hs[TN][2];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        if constexpr (SV) {
                            const uint2 sv = *(const device uint2*)(sl[t] + (k / 16));
                            hs[t][0] = cbv2_vn_e4m3x2(sv.x >> scale_shift);
                            hs[t][1] = cbv2_vn_e4m3x2(sv.y >> scale_shift);
                        } else {
                            const uint sa0 = uint(sl[t][k / 16]) | (uint(sl[t][k / 16 + 2]) << 16);
                            const uint sa1 = uint(sl[t][k / 16 + 4]) | (uint(sl[t][k / 16 + 6]) << 16);
                            hs[t][0] = cbv2_vn_e4m3x2(sa0);
                            hs[t][1] = cbv2_vn_e4m3x2(sa1);
                        }
                    }
                    MO_WO_UNROLL
                    for (short block = 0; block < 4; block++) {
                        const int b = k + 32 * block;
                        ushort4 xa[TM][2];
                        MO_WO_UNROLL
                        for (short mt = 0; mt < TM; mt++) {
                            xa[mt][0] = *(const device ushort4*)(xr[mt][0] + b);
                            xa[mt][1] = *(const device ushort4*)(xr[mt][1] + b);
                        }
                        float fs[TN];
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            fs[t] = float(hs[t][block / 2][block % 2]) * 16384.0f;
                        }
                        MO_WO_UNROLL
                        for (short i = 0; i < 4; i++) {
                            float2 bf[TM];
                            MO_WO_UNROLL
                            for (short mt = 0; mt < TM; mt++) {
                                bf[mt] = float2(
                                    as_type<float>(uint(xa[mt][0][i]) << 16),
                                    as_type<float>(uint(xa[mt][1][i]) << 16));
                            }
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                const float2 a = cbv2_vn_pair_f(qa[block][t], i, fs[t]);
                                MO_WO_UNROLL
                                for (short mt = 0; mt < TM; mt++) {
                                    mo_wo_mma(C[t][mt], a, bf[mt]);
                                }
                            }
                        }
                    }
                }
            } else if constexpr (WC && PF == 2) {
                // Branchless peeled next-chunk weight+scale DRAM prefetch with streamed
                // block-at-a-time activation loads and float-domain E2M1 scaling.
                constexpr int STEP = WK * BK;
                int k = int(sg) * BK;
                if (k < K) {
                    uint qb[4][TN];
                    uint2 sb[TN];
                    MO_WO_UNROLL
                    for (short block = 0; block < 4; block++) {
                        const int b = k + 32 * block;
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) { qb[block][t] = wl[t][b / 8]; }
                    }
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        if constexpr (SV) {
                            sb[t] = *(const device uint2*)(sl[t] + (k / 16));
                        } else {
                            sb[t] = uint2(
                                uint(sl[t][k / 16]) | (uint(sl[t][k / 16 + 2]) << 16),
                                uint(sl[t][k / 16 + 4]) | (uint(sl[t][k / 16 + 6]) << 16));
                        }
                    }
                    for (; k + STEP < K; k += STEP) {
                        const int kn = k + STEP;
                        uint q[4][TN];
                        MO_WO_UNROLL
                        for (short block = 0; block < 4; block++) {
                            const int bn = kn + 32 * block;
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                q[block][t] = qb[block][t];
                                qb[block][t] = wl[t][bn / 8];
                            }
                        }
                        half2 hs[TN][2];
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            if constexpr (SV) {
                                hs[t][0] = cbv2_vn_e4m3x2(sb[t].x >> scale_shift);
                                hs[t][1] = cbv2_vn_e4m3x2(sb[t].y >> scale_shift);
                                sb[t] = *(const device uint2*)(sl[t] + (kn / 16));
                            } else {
                                hs[t][0] = cbv2_vn_e4m3x2(sb[t].x);
                                hs[t][1] = cbv2_vn_e4m3x2(sb[t].y);
                                sb[t] = uint2(
                                    uint(sl[t][kn / 16]) | (uint(sl[t][kn / 16 + 2]) << 16),
                                    uint(sl[t][kn / 16 + 4]) | (uint(sl[t][kn / 16 + 6]) << 16));
                            }
                        }
                        MO_WO_UNROLL
                        for (short block = 0; block < 4; block++) {
                            const int b = k + 32 * block;
                            ushort4 xa[TM][2];
                            MO_WO_UNROLL
                            for (short mt = 0; mt < TM; mt++) {
                                xa[mt][0] = *(const device ushort4*)(xr[mt][0] + b);
                                xa[mt][1] = *(const device ushort4*)(xr[mt][1] + b);
                            }
                            float fs[TN];
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                fs[t] = float(hs[t][block / 2][block % 2]) * 16384.0f;
                            }
                            MO_WO_UNROLL
                            for (short i = 0; i < 4; i++) {
                                float2 bf[TM];
                                MO_WO_UNROLL
                                for (short mt = 0; mt < TM; mt++) {
                                    bf[mt] = float2(
                                        as_type<float>(uint(xa[mt][0][i]) << 16),
                                        as_type<float>(uint(xa[mt][1][i]) << 16));
                                }
                                MO_WO_UNROLL
                                for (short t = 0; t < TN; t++) {
                                    const float2 a = cbv2_vn_pair_f(q[block][t], i, fs[t]);
                                    MO_WO_UNROLL
                                    for (short mt = 0; mt < TM; mt++) {
                                        mo_wo_mma(C[t][mt], a, bf[mt]);
                                    }
                                }
                            }
                        }
                    }
                    // Final peeled chunk (no next-chunk load).
                    {
                        half2 hs[TN][2];
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            if constexpr (SV) {
                                hs[t][0] = cbv2_vn_e4m3x2(sb[t].x >> scale_shift);
                                hs[t][1] = cbv2_vn_e4m3x2(sb[t].y >> scale_shift);
                            } else {
                                hs[t][0] = cbv2_vn_e4m3x2(sb[t].x);
                                hs[t][1] = cbv2_vn_e4m3x2(sb[t].y);
                            }
                        }
                        MO_WO_UNROLL
                        for (short block = 0; block < 4; block++) {
                            const int b = k + 32 * block;
                            ushort4 xa[TM][2];
                            MO_WO_UNROLL
                            for (short mt = 0; mt < TM; mt++) {
                                xa[mt][0] = *(const device ushort4*)(xr[mt][0] + b);
                                xa[mt][1] = *(const device ushort4*)(xr[mt][1] + b);
                            }
                            float fs[TN];
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                fs[t] = float(hs[t][block / 2][block % 2]) * 16384.0f;
                            }
                            MO_WO_UNROLL
                            for (short i = 0; i < 4; i++) {
                                float2 bf[TM];
                                MO_WO_UNROLL
                                for (short mt = 0; mt < TM; mt++) {
                                    bf[mt] = float2(
                                        as_type<float>(uint(xa[mt][0][i]) << 16),
                                        as_type<float>(uint(xa[mt][1][i]) << 16));
                                }
                                MO_WO_UNROLL
                                for (short t = 0; t < TN; t++) {
                                    const float2 a = cbv2_vn_pair_f(qb[block][t], i, fs[t]);
                                    MO_WO_UNROLL
                                    for (short mt = 0; mt < TM; mt++) {
                                        mo_wo_mma(C[t][mt], a, bf[mt]);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if constexpr (WC && PF == 1) {
                uint qb[4][TN];
                uint2 sb[TN];
                if (int(sg) * BK < K) {
                    const int k0 = int(sg) * BK;
                    MO_WO_UNROLL
                    for (short block = 0; block < 4; block++) {
                        const int b = k0 + 32 * block;
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) { qb[block][t] = wl[t][b / 8]; }
                    }
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        if constexpr (SV) {
                            sb[t] = *(const device uint2*)(sl[t] + (k0 / 16));
                        } else {
                            sb[t] = uint2(
                                uint(sl[t][k0 / 16]) | (uint(sl[t][k0 / 16 + 2]) << 16),
                                uint(sl[t][k0 / 16 + 4]) | (uint(sl[t][k0 / 16 + 6]) << 16));
                        }
                    }
                }
                for (int k = int(sg) * BK; k < K; k += WK * BK) {
                    const int kn = k + WK * BK;
                    const bool more = kn < K;
                    half2 hs[TN][2];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        if constexpr (SV) {
                            hs[t][0] = cbv2_vn_e4m3x2(sb[t].x >> scale_shift);
                            hs[t][1] = cbv2_vn_e4m3x2(sb[t].y >> scale_shift);
                        } else {
                            hs[t][0] = cbv2_vn_e4m3x2(sb[t].x);
                            hs[t][1] = cbv2_vn_e4m3x2(sb[t].y);
                        }
                    }
                    if (more) {
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            if constexpr (SV) {
                                sb[t] = *(const device uint2*)(sl[t] + (kn / 16));
                            } else {
                                sb[t] = uint2(
                                    uint(sl[t][kn / 16]) | (uint(sl[t][kn / 16 + 2]) << 16),
                                    uint(sl[t][kn / 16 + 4]) | (uint(sl[t][kn / 16 + 6]) << 16));
                            }
                        }
                    }
                    MO_WO_UNROLL
                    for (short block = 0; block < 4; block++) {
                        const int b = k + 32 * block;
                        ushort4 xv[TM][2];
                        MO_WO_UNROLL
                        for (short mt = 0; mt < TM; mt++) {
                            MO_WO_UNROLL
                            for (short j = 0; j < 2; j++) {
                                xv[mt][j] = *(const device ushort4*)(xr[mt][j] + b);
                            }
                        }
                        uint q[TN];
                        half s[TN];
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            q[t] = qb[block][t];
                            s[t] = hs[t][block / 2][block % 2];
                        }
                        if (more) {
                            const int bn = kn + 32 * block;
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                qb[block][t] = wl[t][bn / 8];
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
            } else {
            // Rolling buffer (PF 1): slot `block` holds this chunk's block and is
            // refilled with the next chunk's block once read.
            uint qb[4][TN];
            uint sb[4][TN];
            if (PF == 1 && int(sg) * BK < K) {
                const int k0 = int(sg) * BK;
                MO_WO_UNROLL
                for (short block = 0; block < 4; block++) {
                    const int b = k0 + 32 * block;
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        qb[block][t] = wl[t][b / 8];
                        sb[block][t] = sl[t][b / 16 + (SV ? (fn / 4) : 0)];
                    }
                }
            }
            for (int k = int(sg) * BK; k < K; k += WK * BK) {
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
                            xv[mt][j] = *(const device ushort4*)(xr[mt][j] + b);
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
                            s[t] = mo_wo_e4m3(sl[t][b / 16 + (SV ? (fn / 4) : 0)]);
                        }
                    }
                    if (PF == 1 && more) {
                        const int bn = kn + 32 * block;
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            qb[block][t] = wl[t][bn / 8];
                            sb[block][t] = sl[t][bn / 16 + (SV ? (fn / 4) : 0)];
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
            }

            constexpr int tile_elems = TN * TM * 2;
            if (sg > 0) {
                threadgroup float* p = partials + (sg - 1) * tile_elems * 32 + simd_lid;
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
            if (sg == 0) {
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

    /// Segment `i` takes tiles `[sum of tiles before, + ceil(N_i / (8 * TN * RG)))`
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
                    cbv2_vn_tile<T, V, K, N\(i), TN, PF, SCALED, WC, SV>(
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
        1: kernel(segments: 1), 2: kernel(segments: 2), 4: kernel(segments: 4),
    ]

    private struct LaunchKey: Hashable {
        let segments: Int
        let dtype: DType
        let rows: Int
        let inputs: Int
        let fragments: Int
        let prefetch: Int
        let scaled: Bool
        let wholeChunk: Bool
        let scaleVec: Bool
        let outputs: [Int]
    }

    private struct PlanKey: Hashable {
        let id0: ObjectIdentifier
        let idLast: ObjectIdentifier
        let count: Int
        let inputs: Int
        let rows: Int
        let dtype: DType
        let fragments: Int
        let prefetch: Int
        let wholeChunk: Bool
        let scaleVec: Bool
    }

    private struct Plan {
        let launch: MLXFast.MLXFastKernel.Launch
        let weightArrays: [MLXArray]
        let outputDims: [Int]
    }

    private final class CacheBox: @unchecked Sendable {
        var launches: [LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
        var plans: [PlanKey: Plan] = [:]
        var fusedGULaunches: [FusedGULaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
        var fusedGUPlans: [FusedGUPlanKey: FusedGUPlan] = [:]
    }

    private struct FusedGULaunchKey: Hashable {
        let rows: Int
        let inputs: Int
        let outputs: Int
        let castFP8: Bool
        let tn: Int
    }

    private struct FusedGUPlanKey: Hashable {
        let gateID: ObjectIdentifier
        let upID: ObjectIdentifier
        let downID: ObjectIdentifier?
        let rows: Int
        let inputs: Int
        let tn: Int
    }

    private struct FusedGUPlan {
        let launch: MLXFast.MLXFastKernel.Launch
        let weightArrays: [MLXArray]
        let fp8Mu: MLXArray
        let fp8Bound: MLXArray
        let outputs: Int
        let castFP8: Bool
    }

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

    static let fusedGateUpSiluHeader = """

        inline bfloat16_t cbv2_vn_bf16_sigmoid(bfloat16_t x) {
            const bfloat16_t ax = (x < bfloat16_t(0.0f)) ? -x : x;
            const bfloat16_t ey = static_cast<bfloat16_t>(metal::precise::exp(static_cast<float>(ax)));
            const bfloat16_t one = bfloat16_t(1.0f);
            const bfloat16_t y = one / (one + ey);
            return (x < bfloat16_t(0.0f)) ? y : (one - y);
        }

        inline bfloat16_t cbv2_vn_bf16_cast_fp8(
            bfloat16_t yb, float m0, float m1, float bound, float nan_value
        ) {
            const float scaled = static_cast<float>(yb) * m0;
            const uint8_t bits = modelopt_e4m3_bits(scaled);
            const float decoded = (bits & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(bits);
            const bool invalid = metal::isnan(scaled) || metal::abs(scaled) > bound;
            return static_cast<bfloat16_t>(invalid ? nan_value : decoded * m1);
        }

        """

    /// Fused NVFP4 `gate_up + SiLU (+ optional FP8 cast)` with `TN` 4g+4u fragments/TG
    /// (`TN=2` default: 8g+8u, 2,496 TGs = 39*64 exact on M5 Max, streamed block-at-a-time
    /// activation loads, `uint<<16` bf16 unpack, float-domain `cbv2_vn_pair_f`, maxT=832,
    /// +7.74% faster than 4g+4u PF=2 under safe math, 0-ULP bit-exact).
    static let fusedGateUpSiluSource = """
        constexpr int WK = 8;
        constexpr int BK = 128;
        constexpr int STEP = WK * BK;
        constexpr int row_bytes = K / 2;
        constexpr int row_scales = K / 16;
        static_assert(K % BK == 0, "whole chunks only");
        static_assert(V >= 1 && V <= 8, "1..8 rows");
        static_assert(TN == 1 || TN == 2, "TN must be 1 or 2");
        threadgroup float partials[7 * TN * 2 * 32];
        const int tile = int(threadgroup_position_in_grid.y);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint sl = thread_index_in_simdgroup;
        const short qid = sl / 4;
        const short fm = (qid & 4) + ((sl / 2) % 4);
        const short fn = (qid & 2) * 2 + (sl % 2) * 2;
        const uint scale_shift = (fn & 4u) << 1;
        const bool is_up = (fm >= 4);
        // Activation address keeps original `fm` so the 8x8x8 MMA covers all 32 K lanes.
        const device bfloat16_t* xr0 = x + size_t(min(int(fn), V - 1)) * K + 4 * fm;
        const device bfloat16_t* xr1 = x + size_t(min(int(fn + 1), V - 1)) * K + 4 * fm;
        const device uint8_t* wbase = (const device uint8_t*)(is_up ? w1 : w0);
        const device uint8_t* sbase = (is_up ? sc1 : sc0);
        const device uint* wl[TN];
        const device uint8_t* scl[TN];
        float2 C[TN];
        MO_WO_UNROLL
        for (short t = 0; t < TN; t++) {
            const int row = min(tile * 4 * TN + t * 4 + (fm & 3), N - 1);
            wl[t] = (const device uint*)(wbase + size_t(row) * row_bytes + 2 * fn);
            scl[t] = sbase + size_t(row) * row_scales;
            C[t] = float2(0);
        }
        for (int k = int(sg) * BK; k < K; k += STEP) {
            uint q[4][TN];
            MO_WO_UNROLL
            for (short block = 0; block < 4; block++) {
                const int b = (k + 32 * block) / 8;
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) { q[block][t] = wl[t][b]; }
            }
            half2 hs[TN][2];
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                const uint2 sv = *(const device uint2*)(scl[t] + (k / 16));
                hs[t][0] = cbv2_vn_e4m3x2(sv.x >> scale_shift);
                hs[t][1] = cbv2_vn_e4m3x2(sv.y >> scale_shift);
            }
            MO_WO_UNROLL
            for (short block = 0; block < 4; block++) {
                const int b = k + 32 * block;
                const ushort4 x0 = *(const device ushort4*)(xr0 + b);
                const ushort4 x1 = *(const device ushort4*)(xr1 + b);
                float fs[TN];
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    fs[t] = float(hs[t][block / 2][block % 2]) * 16384.0f;
                }
                MO_WO_UNROLL
                for (short i = 0; i < 4; i++) {
                    const float2 bf = float2(
                        as_type<float>(uint(x0[i]) << 16),
                        as_type<float>(uint(x1[i]) << 16));
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        mo_wo_mma(C[t], cbv2_vn_pair_f(q[block][t], i, fs[t]), bf);
                    }
                }
            }
        }
        if (sg > 0u) {
            threadgroup float* p = partials + (sg - 1u) * (TN * 64u) + sl;
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                p[t * 64 + 0] = C[t][0];
                p[t * 64 + 32] = C[t][1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            for (int s = 0; s < WK - 1; s++) {
                const threadgroup float* p = partials + s * (TN * 64) + sl;
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    C[t][0] += p[t * 64 + 0];
                    C[t][1] += p[t * 64 + 32];
                }
            }
            const float ts_val = is_up ? ts1[0] : ts0[0];
            const float nan_v = metal::numeric_limits<float>::quiet_NaN();
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                const bfloat16_t v0 = mo_wo_result<bfloat16_t>(C[t][0], true, ts_val);
                const bfloat16_t v1 = mo_wo_result<bfloat16_t>(C[t][1], true, ts_val);
                // Warp-uniform shuffle before restricting stores to the gate half (fm < 4).
                const ushort u0_bits = simd_shuffle_down(as_type<ushort>(v0), ushort(16));
                const ushort u1_bits = simd_shuffle_down(as_type<ushort>(v1), ushort(16));
                const int out_row = tile * 4 * TN + t * 4 + (fm & 3);
                if (!is_up && out_row < N) {
                    if (fn < V) {
                        bfloat16_t o0 =
                            (v0 * cbv2_vn_bf16_sigmoid(v0)) * as_type<bfloat16_t>(u0_bits);
                        if constexpr (CAST_FP8) {
                            o0 = cbv2_vn_bf16_cast_fp8(o0, fp8_mu[0], fp8_mu[1], fp8_bd[0], nan_v);
                        }
                        y[size_t(fn) * N + out_row] = o0;
                    }
                    if (fn + 1 < V) {
                        bfloat16_t o1 =
                            (v1 * cbv2_vn_bf16_sigmoid(v1)) * as_type<bfloat16_t>(u1_bits);
                        if constexpr (CAST_FP8) {
                            o1 = cbv2_vn_bf16_cast_fp8(o1, fp8_mu[0], fp8_mu[1], fp8_bd[0], nan_v);
                        }
                        y[size_t(fn + 1) * N + out_row] = o1;
                    }
                }
            }
        }
        """

    private static let fusedGateUpSiluKernel = MLXFast.metalKernel(
        name: "cbv2_verify_nvfp4_fused_gu_silu_tn",
        inputNames: ["w0", "sc0", "ts0", "w1", "sc1", "ts1", "x", "fp8_mu", "fp8_bd"],
        outputNames: ["y"],
        source: fusedGateUpSiluSource,
        header: ModelOptWeightOnlyNVFP4RowExact.header + ModelOptActivationKernels.header
            + tileHelper + fusedGateUpSiluHeader)

    /// Fused NVFP4 `silu(gate(x)) * up(x)` (optionally pre-cast to FP8 for an FP8
    /// `downProj`), 0-ULP bit-exact to separate `products([gate, up], x)` + `silu(g)*u` (+ FP8 cast).
    /// Active strictly on `strictLowVerifyC3` low-acceptance verify rounds (`rows in 2...8`).
    public static func fusedGateUpSilu(
        gate: ModelOptWeightOnlyNVFP4Linear, up: ModelOptWeightOnlyNVFP4Linear, x: MLXArray,
        fp8Down: ModelOptCalibratedLinear? = nil
    ) -> (act: MLXArray, preCastFP8: Bool)? {
        guard fusedGateUpSiluEnabled, geometryOverride == nil else { return nil }
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        guard lowVerify, CBv2DraftRegime.strictLowVerifyC3 else { return nil }
        let inputs = x.dim(-1)
        guard inputs > 0, x.dtype == .bfloat16 else { return nil }
        let rows = x.size / inputs
        guard (2 ... maximumRows).contains(rows) else { return nil }
        let tn = fusedGUTN1Env ? 1 : 2
        let downID = fp8Down.map(ObjectIdentifier.init)
        let pkey = FusedGUPlanKey(
            gateID: ObjectIdentifier(gate), upID: ObjectIdentifier(up), downID: downID,
            rows: rows, inputs: inputs, tn: tn)
        if let plan = cacheLock.withLockUnchecked({ $0.fusedGUPlans[pkey] }) {
            var args = plan.weightArrays
            args += [x.reshaped(rows, inputs), plan.fp8Mu, plan.fp8Bound]
            let raw = plan.launch(args)[0]
            let act =
                x.ndim == 3
                ? raw.reshaped(x.dim(0), x.dim(1), plan.outputs)
                : raw.reshaped(Array(x.shape.dropLast()) + [plan.outputs])
            return (act, plan.castFP8)
        }
        guard supports(rows: rows, inputs: inputs, dtype: x.dtype),
            gate.inputDimensions == inputs, up.inputDimensions == inputs,
            gate.outputDimensions == up.outputDimensions,
            takes(weight: gate.weight, blockScales: gate.weightScale, tensorScale: gate.weightScale2, inputs: inputs),
            takes(weight: up.weight, blockScales: up.weightScale, tensorScale: up.weightScale2, inputs: inputs)
        else { return nil }
        let fp8Mu: MLXArray
        let fp8Bound: MLXArray
        let castFP8: Bool
        if let fp8Down, fp8Down.format == .fp8,
            let constants = fp8Down.graphConstants,
            case .fp8(let mu, let bd) = constants.storage
        {
            fp8Mu = mu
            fp8Bound = bd
            castFP8 = true
        } else {
            fp8Mu = dummyFP8Mu
            fp8Bound = dummyFP8Bound
            castFP8 = false
        }
        let outputs = gate.weight.dim(0)
        let lkey = FusedGULaunchKey(rows: rows, inputs: inputs, outputs: outputs, castFP8: castFP8, tn: tn)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = cacheLock.withLockUnchecked({ $0.fusedGULaunches[lkey] }) {
            launch = hit
        } else {
            let tiles = (outputs + 4 * tn - 1) / (4 * tn)
            let made = fusedGateUpSiluKernel.prepare(
                template: [
                    ("V", rows), ("K", inputs), ("N", outputs), ("TN", tn), ("CAST_FP8", castFP8),
                ],
                grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                outputShapes: [[rows, outputs]], outputDTypes: [.bfloat16])
            cacheLock.withLockUnchecked { $0.fusedGULaunches[lkey] = made }
            launch = made
        }
        let weightArrays = [
            gate.weight, gate.weightScale, gate.weightScale2.reshaped(1),
            up.weight, up.weightScale, up.weightScale2.reshaped(1),
        ]
        let plan = FusedGUPlan(
            launch: launch, weightArrays: weightArrays, fp8Mu: fp8Mu, fp8Bound: fp8Bound,
            outputs: outputs, castFP8: castFP8)
        cacheLock.withLockUnchecked { $0.fusedGUPlans[pkey] = plan }
        var args = weightArrays
        args += [x.reshaped(rows, inputs), fp8Mu, fp8Bound]
        let raw = launch(args)[0]
        let act =
            x.ndim == 3
            ? raw.reshaped(x.dim(0), x.dim(1), outputs)
            : raw.reshaped(Array(x.shape.dropLast()) + [outputs])
        return (act, castFP8)
    }

    private static let cacheLock = OSAllocatedUnfairLock(initialState: CacheBox())

    private static func preparedLaunch(
        segments: Int, dtype: DType, rows: Int, inputs: Int, fragments: Int, prefetch: Int,
        scaled: Bool, wholeChunk: Bool, scaleVec: Bool, outputs: [Int]
    ) -> MLXFast.MLXFastKernel.Launch {
        let key = LaunchKey(
            segments: segments, dtype: dtype, rows: rows, inputs: inputs, fragments: fragments,
            prefetch: prefetch, scaled: scaled, wholeChunk: wholeChunk, scaleVec: scaleVec,
            outputs: outputs)
        if let hit = cacheLock.withLockUnchecked({ $0.launches[key] }) { return hit }
        let tile = 8 * fragments
        var template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("V", rows), ("K", inputs), ("TN", fragments), ("PF", prefetch),
            ("SCALED", scaled), ("WC", wholeChunk), ("SV", scaleVec),
        ]
        var shapes = [[Int]]()
        shapes.reserveCapacity(outputs.count)
        var tiles = 0
        for (i, n) in outputs.enumerated() {
            template.append(("N\(i)", n))
            shapes.append([rows, n])
            tiles += (n + tile - 1) / tile
        }
        let made = kernels[segments]!.prepare(
            template: template,
            grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
            outputShapes: shapes, outputDTypes: Array(repeating: dtype, count: outputs.count))
        cacheLock.withLockUnchecked { $0.launches[key] = made }
        return made
    }

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
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        if !lowVerify {
            return Record.product(
                x, weight: weight, blockScales: blockScales, tensorScale: tensorScale)
        }
        let s = schedule(rows: rows, inputs: inputs, lowVerify: lowVerify)
        let launch = preparedLaunch(
            segments: 1, dtype: x.dtype, rows: rows, inputs: inputs, fragments: s.fragments,
            prefetch: s.prefetch, scaled: tensorScale != nil,
            wholeChunk: s.wholeChunk, scaleVec: s.scaleVec, outputs: [outputs])
        let result = launch([
            weight, blockScales, tensorScale?.reshaped(1) ?? unit, x.reshaped(rows, inputs),
        ])[0]
        if x.ndim == 3 {
            return result.reshaped(x.dim(0), x.dim(1), outputs)
        }
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    /// `layer.rowExactProduct(x)` of the Muse Glimmer tower for a weight-only
    /// NVFP4 layer, or nil.
    public static func product(_ layer: ModelOptWeightOnlyNVFP4Linear, _ x: MLXArray) -> MLXArray? {
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        if !lowVerify { return Record.product(layer, x) }
        let s = schedule(rows: rows, inputs: inputs, lowVerify: lowVerify)
        let id = ObjectIdentifier(layer)
        let pkey = PlanKey(
            id0: id, idLast: id, count: 1, inputs: inputs, rows: rows, dtype: x.dtype,
            fragments: s.fragments, prefetch: s.prefetch,
            wholeChunk: s.wholeChunk, scaleVec: s.scaleVec)
        if lowVerify && geometryOverride == nil, let plan = cacheLock.withLockUnchecked({ $0.plans[pkey] }) {
            var args = plan.weightArrays
            args.append(x.reshaped(rows, inputs))
            let result = plan.launch(args)[0]
            if x.ndim == 3 {
                return result.reshaped(x.dim(0), x.dim(1), plan.outputDims[0])
            }
            return result.reshaped(Array(x.shape.dropLast()) + [plan.outputDims[0]])
        }
        guard supports(rows: rows, inputs: inputs, dtype: x.dtype),
            layer.inputDimensions == inputs,
            takes(
                weight: layer.weight, blockScales: layer.weightScale,
                tensorScale: layer.weightScale2, inputs: inputs)
        else { return nil }
        let outputs = layer.weight.dim(0)
        let launch = preparedLaunch(
            segments: 1, dtype: x.dtype, rows: rows, inputs: inputs, fragments: s.fragments,
            prefetch: s.prefetch, scaled: true,
            wholeChunk: s.wholeChunk, scaleVec: s.scaleVec, outputs: [outputs])
        let weightArrays = [layer.weight, layer.weightScale, layer.weightScale2.reshaped(1)]
        if lowVerify && geometryOverride == nil {
            let plan = Plan(launch: launch, weightArrays: weightArrays, outputDims: [outputs])
            cacheLock.withLockUnchecked { $0.plans[pkey] = plan }
        }
        var args = weightArrays
        args.append(x.reshaped(rows, inputs))
        let result = launch(args)[0]
        if x.ndim == 3 {
            return result.reshaped(x.dim(0), x.dim(1), outputs)
        }
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    /// The products of sibling NVFP4 layers of one input (MLP gate and
    /// up; attention q, k, v and gate), one launch, each equal to its layer's own
    /// row-exact product; nil when the group is not taken.
    public static func products(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray]? {
        guard kernels[layers.count] != nil, layers.count >= 2,
            let nv = layers as? [ModelOptWeightOnlyNVFP4Linear]
        else {
            return nil
        }
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        let lowVerify = !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance
        if !lowVerify { return Record.products(nv, x) }
        let s = schedule(rows: rows, inputs: inputs, lowVerify: lowVerify)
        let pkey = PlanKey(
            id0: ObjectIdentifier(nv[0]), idLast: ObjectIdentifier(nv[nv.count - 1]),
            count: nv.count, inputs: inputs, rows: rows, dtype: x.dtype,
            fragments: s.fragments, prefetch: s.prefetch,
            wholeChunk: s.wholeChunk, scaleVec: s.scaleVec)
        if lowVerify && geometryOverride == nil, let plan = cacheLock.withLockUnchecked({ $0.plans[pkey] }) {
            var args = plan.weightArrays
            args.append(x.reshaped(rows, inputs))
            let results = plan.launch(args)
            if x.ndim == 3 {
                let b = x.dim(0), l = x.dim(1)
                return (0 ..< results.count).map { results[$0].reshaped(b, l, plan.outputDims[$0]) }
            }
            let lead = Array(x.shape.dropLast())
            return zip(results, plan.outputDims).map { $0.reshaped(lead + [$1]) }
        }
        guard supports(rows: rows, inputs: inputs, dtype: x.dtype) else { return nil }
        for layer in nv {
            guard layer.inputDimensions == inputs,
                takes(weight: layer.weight, blockScales: layer.weightScale,
                    tensorScale: layer.weightScale2, inputs: inputs)
            else { return nil }
        }
        var weightArrays = [MLXArray]()
        weightArrays.reserveCapacity(nv.count * 3 + 1)
        var outputDims = [Int]()
        outputDims.reserveCapacity(nv.count)
        for layer in nv {
            let outputs = layer.weight.dim(0)
            weightArrays += [layer.weight, layer.weightScale, layer.weightScale2.reshaped(1)]
            outputDims.append(outputs)
        }
        let launch = preparedLaunch(
            segments: nv.count, dtype: x.dtype, rows: rows, inputs: inputs,
            fragments: s.fragments, prefetch: s.prefetch, scaled: true,
            wholeChunk: s.wholeChunk, scaleVec: s.scaleVec, outputs: outputDims)
        if lowVerify && geometryOverride == nil {
            let plan = Plan(launch: launch, weightArrays: weightArrays, outputDims: outputDims)
            cacheLock.withLockUnchecked { $0.plans[pkey] = plan }
        }
        var args = weightArrays
        args.append(x.reshaped(rows, inputs))
        let results = launch(args)
        if x.ndim == 3 {
            let b = x.dim(0), l = x.dim(1)
            return (0 ..< results.count).map { results[$0].reshaped(b, l, outputDims[$0]) }
        }
        let lead = Array(x.shape.dropLast())
        return zip(results, outputDims).map { $0.reshaped(lead + [$1]) }
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

// MARK: - Record (c95a27f5) schedule for high-acceptance verify rounds

extension CBv2VerifyNVFP4 {
    /// The record's (c95a27f5) NVFP4 verify tile, launch geometry and host path,
    /// verbatim except for the kernel names. High-acceptance verify rounds
    /// (`CBv2DraftRegime.lowOnlyVerifySpeedups && !CBv2DraftRegime.lowAcceptance`)
    /// take it so their timing stays the record's; low-acceptance rounds keep the
    /// whole-chunk / vectorized-scale tiles.
    enum Record {
        static let tileHelper = """

            inline half2 cbv2_vn_e4m3x2(uint x) {
                const uint v = x & 0x007F007Fu;
                const uint u = (v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7)
                    | ((x & 0x00800080u) << 8);
                return as_type<half2>(u) * half2(256.0h);
            }

            inline void cbv2_vn_transpose4(uint4 v, bool odd, bool hi, thread uint* q) {
                const uint r_lo = simd_shuffle_xor(odd ? v.x : v.y, ushort(1));
                const uint r_hi = simd_shuffle_xor(odd ? v.z : v.w, ushort(1));
                const uint a0 = odd ? r_lo : v.x;
                const uint a1 = odd ? v.y : r_lo;
                const uint b0 = odd ? r_hi : v.z;
                const uint b1 = odd ? v.w : r_hi;
                const uint p0 = simd_shuffle_xor(hi ? a0 : b0, ushort(8));
                const uint p1 = simd_shuffle_xor(hi ? a1 : b1, ushort(8));
                q[0] = hi ? p0 : a0;
                q[1] = hi ? p1 : a1;
                q[2] = hi ? b0 : p0;
                q[3] = hi ? b1 : p1;
            }

            // One tile of 8 * TN outputs of one projection: the stock matrix
            // kernel's arithmetic (see CBv2VerifyNVFP4) on 8 * TN rows.
            template <typename T, int V, int K, int N, int TN, int PF, bool SCALED, bool COAL>
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
                if constexpr (COAL) {
                    const short c = fn >> 1;
                    const bool odd = (simd_lid & 1) != 0;
                    const bool hi = (simd_lid & 8) != 0;
                    const uint sshift = hi ? 8u : 0u;
                    const device uint4* w4l[TN];
                    const device uint* s4l[TN];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        const int row = min(y_col + fm + 8 * t, N - 1);
                        w4l[t] = (const device uint4*)(
                            (const device uint8_t*)w + size_t(row) * row_bytes) + c;
                        s4l[t] = (const device uint*)(
                            scales + size_t(row) * row_scales) + (hi ? 1 : 0);
                    }
                    uint4 w_cur[TN];
                    uint s_cur[TN];
                    if (PF == 1 && int(simd_gid) * BK < K) {
                        const int k0 = int(simd_gid) * BK;
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            w_cur[t] = w4l[t][k0 / 32];
                            s_cur[t] = s4l[t][k0 / 64];
                        }
                    }
                    for (int k = int(simd_gid) * BK; k < K; k += WK * BK) {
                        uint4 w_use[TN];
                        uint s_use[TN];
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            if (PF == 1) {
                                w_use[t] = w_cur[t];
                                s_use[t] = s_cur[t];
                            } else {
                                w_use[t] = w4l[t][k / 32];
                                s_use[t] = s4l[t][k / 64];
                            }
                        }
                        const int kn = k + WK * BK;
                        if (PF == 1 && kn < K) {
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                w_cur[t] = w4l[t][kn / 32];
                                s_cur[t] = s4l[t][kn / 64];
                            }
                        }
                        uint q[TN][4];
                        half2 hs0[TN], hs1[TN];
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            cbv2_vn_transpose4(w_use[t], odd, hi, q[t]);
                            const uint s_peer = simd_shuffle_xor(s_use[t], ushort(8));
                            const uint s_lo = hi ? s_peer : s_use[t];
                            const uint s_hi = hi ? s_use[t] : s_peer;
                            hs0[t] = cbv2_vn_e4m3x2((s_lo >> sshift) & 0x00FF00FFu);
                            hs1[t] = cbv2_vn_e4m3x2((s_hi >> sshift) & 0x00FF00FFu);
                        }
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
                            half s[TN];
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                s[t] = (block < 2) ? hs0[t][block] : hs1[t][block - 2];
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
                                    const float2 a = mo_wo_pair(q[t][block], i, s[t]);
                                    MO_WO_UNROLL
                                    for (short mt = 0; mt < TM; mt++) {
                                        mo_wo_mma(C[t][mt], a, bf[mt]);
                                    }
                                }
                            }
                        }
                    }
                } else {
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

        static func geometry(rows: Int, inputs: Int) -> Geometry {
            if let geometryOverride { return geometryOverride }
            let pf = prefetchEnv ?? (inputs >= 16384 && rows >= 8 ? 1 : 0)
            if inputs >= 16384 && rows >= 7 { return Geometry(fragments: 2, prefetch: pf) }
            return Geometry(fragments: 1, prefetch: pf)
        }

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
                        cbv2_vn_tile<T, V, K, N\(i), TN, PF, SCALED, COAL>(
                            w\(i), sc\(i), x, ts\(i)[0], y\(i), tg - (\(offset)), sg, sl, partials);
                    }

                    """
                offset += " + \(tiles)"
            }
            inputs.append("x")
            return MLXFast.metalKernel(
                name: "cbv2_verify_nvfp4_row_exact_rec_\(segments)", inputNames: inputs,
                outputNames: outputs, source: body,
                header: ModelOptWeightOnlyNVFP4RowExact.header + tileHelper)
        }

        nonisolated(unsafe) static let kernels: [Int: MLXFast.MLXFastKernel] = [
            1: kernel(segments: 1), 2: kernel(segments: 2), 4: kernel(segments: 4),
        ]

        private struct LaunchKey: Hashable {
            let segments: Int
            let dtype: DType
            let rows: Int
            let inputs: Int
            let fragments: Int
            let prefetch: Int
            let scaled: Bool
            let coalesced: Bool
            let outputs: [Int]
        }

        private static let launchLock = NSLock()
        nonisolated(unsafe) private static var launchCache:
            [LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

        private static func preparedLaunch(
            segments: Int, dtype: DType, rows: Int, inputs: Int, fragments: Int, prefetch: Int,
            scaled: Bool, outputs: [Int]
        ) -> MLXFast.MLXFastKernel.Launch {
            let coal = false
            let key = LaunchKey(
                segments: segments, dtype: dtype, rows: rows, inputs: inputs, fragments: fragments,
                prefetch: prefetch, scaled: scaled, coalesced: coal, outputs: outputs)
            if let hit = launchLock.withLock({ launchCache[key] }) { return hit }
            let tile = 8 * fragments
            var template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("V", rows), ("K", inputs), ("TN", fragments), ("PF", prefetch),
                ("SCALED", scaled), ("COAL", coal),
            ]
            var shapes = [[Int]]()
            shapes.reserveCapacity(outputs.count)
            var tiles = 0
            for (i, n) in outputs.enumerated() {
                template.append(("N\(i)", n))
                shapes.append([rows, n])
                tiles += (n + tile - 1) / tile
            }
            let made = kernels[segments]!.prepare(
                template: template, grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                outputShapes: shapes, outputDTypes: Array(repeating: dtype, count: outputs.count))
            launchLock.withLock { launchCache[key] = made }
            return made
        }

        static func product(
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
            let launch = preparedLaunch(
                segments: 1, dtype: x.dtype, rows: rows, inputs: inputs, fragments: g.fragments,
                prefetch: g.prefetch, scaled: tensorScale != nil, outputs: [outputs])
            let result = launch([
                weight, blockScales, tensorScale?.reshaped(1) ?? unit, x.reshaped(rows, inputs),
            ])[0]
            return result.reshaped(Array(x.shape.dropLast()) + [outputs])
        }

        static func product(_ layer: ModelOptWeightOnlyNVFP4Linear, _ x: MLXArray) -> MLXArray? {
            product(
                x, weight: layer.weight, blockScales: layer.weightScale,
                tensorScale: layer.weightScale2)
        }

        static func products(_ nv: [ModelOptWeightOnlyNVFP4Linear], _ x: MLXArray) -> [MLXArray]? {
            guard kernels[nv.count] != nil, nv.count >= 2 else { return nil }
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
            var arrays = [MLXArray]()
            arrays.reserveCapacity(nv.count * 3 + 1)
            var outputDims = [Int]()
            outputDims.reserveCapacity(nv.count)
            for layer in nv {
                let outputs = layer.weight.dim(0)
                arrays += [layer.weight, layer.weightScale, layer.weightScale2.reshaped(1)]
                outputDims.append(outputs)
            }
            arrays.append(x.reshaped(rows, inputs))
            let launch = preparedLaunch(
                segments: nv.count, dtype: x.dtype, rows: rows, inputs: inputs,
                fragments: g.fragments, prefetch: g.prefetch, scaled: true, outputs: outputDims)
            let results = launch(arrays)
            let lead = Array(x.shape.dropLast())
            return zip(results, nv).map { $0.reshaped(lead + [$1.weight.dim(0)]) }
        }
    }
}
