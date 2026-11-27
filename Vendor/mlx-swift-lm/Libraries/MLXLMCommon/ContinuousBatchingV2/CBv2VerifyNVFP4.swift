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

    /// `CBV2_NVFP4_COALESCED=1` enables coalesced `uint4` weight + `uint` scale loads (off by default).
    public static let coalescedEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_NVFP4_COALESCED"] == "1"

    /// `CBV2_NVFP4_B4SV=0` disables the low-acceptance `TN=2` 4-block batched weight
    /// + coalesced `uint2` scale-vector (`b4+SV`) schedule across NVFP4 verify sites.
    public static let b4svEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_NVFP4_B4SV"] != "0"

    private static let prefetchEnv: Int? =
        ProcessInfo.processInfo.environment["CBV2_NVFP4_PF"].flatMap(Int.init)

    /// Tests and benchmarks pin a geometry here (nil: `geometry(rows:inputs:)`).
    nonisolated(unsafe) public static var geometryOverride: Geometry?

    /// Launch geometry for verify NVFP4 projections. High-acceptance rounds keep
    /// the frontier schedule untouched; low-acceptance rounds use `TN=2, PF=2`
    /// (`b4+SV`: 4-block weight batching + coalesced `uint2` scale loads) once
    /// load-time 0-ULP parity has passed.
    public static func geometry(rows: Int, inputs: Int) -> Geometry {
        if let geometryOverride { return geometryOverride }
        if !coalescedEnabled && prefetchEnv == nil && b4svActive
            && (!CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance)
        {
            return Geometry(fragments: 2, prefetch: 2)
        }
        let pf = prefetchEnv ?? (coalescedEnabled ? 1 : (inputs >= 16384 && rows >= 8 ? 1 : 0))
        if inputs >= 16384 && rows >= 7 { return Geometry(fragments: 2, prefetch: pf) }
        return Geometry(fragments: 1, prefetch: pf)
    }

    private static let e4m3x2Helper = """

        inline half2 cbv2_vn_e4m3x2(uint x) {
            const uint v = x & 0x007F007Fu;
            const uint u = (v << 7) | (((v + 0x00010001u) & 0x00800080u) << 7)
                | ((x & 0x00800080u) << 8);
            return as_type<half2>(u) * half2(256.0h);
        }
        """

    static let tileHelper = e4m3x2Helper + """

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
            } else if constexpr (PF == 2) {
                // b4+SV: 4-block batched weight loads + coalesced 64-bit uint2 scale loads
                // (row_scales = K / 16 = (K / 128) * 8 is always 8-byte aligned).
                const uint sshift = fn >= 4 ? 8u : 0u;
                const device uint* wl[TN];
                const device uint2* sv[TN];
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    const int row = min(y_col + fm + 8 * t, N - 1);
                    wl[t] = (const device uint*)(
                        (const device uint8_t*)w + size_t(row) * row_bytes + 2 * fn);
                    sv[t] = (const device uint2*)(scales + size_t(row) * row_scales);
                }
                for (int k = int(simd_gid) * BK; k < K; k += WK * BK) {
                    uint q[4][TN];
                    half2 hs0[TN], hs1[TN];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        const uint2 raw = sv[t][k / BK];
                        hs0[t] = cbv2_vn_e4m3x2((raw.x >> sshift) & 0x00FF00FFu);
                        hs1[t] = cbv2_vn_e4m3x2((raw.y >> sshift) & 0x00FF00FFu);
                    }
                    MO_WO_UNROLL
                    for (short block = 0; block < 4; block++) {
                        const int b = k + 32 * block;
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            q[block][t] = wl[t][b / 8];
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
                                const float2 a = mo_wo_pair(q[block][t], i, s[t]);
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
                    cbv2_vn_tile<T, V, K, N\(i), TN, PF, SCALED, COAL>(
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
        let coalesced: Bool
        let outputs: [Int]
    }

    private static let launchLock = NSLock()
    nonisolated(unsafe) private static var launchCache: [LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    private static func preparedLaunch(
        segments: Int, dtype: DType, rows: Int, inputs: Int, fragments: Int, prefetch: Int,
        scaled: Bool, outputs: [Int]
    ) -> MLXFast.MLXFastKernel.Launch {
        let coal = coalescedEnabled
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
        let launch = preparedLaunch(
            segments: 1, dtype: x.dtype, rows: rows, inputs: inputs, fragments: g.fragments,
            prefetch: g.prefetch, scaled: tensorScale != nil, outputs: [outputs])
        let result = launch([
            weight, blockScales, tensorScale?.reshaped(1) ?? unit, x.reshaped(rows, inputs),
        ])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    /// `layer.rowExactProduct(x)` of the Muse Glimmer tower for a weight-only
    /// NVFP4 layer, or nil.
    public static func product(_ layer: ModelOptWeightOnlyNVFP4Linear, _ x: MLXArray) -> MLXArray? {
        product(x, weight: layer.weight, blockScales: layer.weightScale, tensorScale: layer.weightScale2)
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

    // MARK: Fused Gate + Up + SwiGLU (+ optional inline FP8 activation cast)

    /// `DARKBLOOM_VERIFY_FUSED_SWIGLU=0` disables fused NVFP4 gate+up+SwiGLU.
    public static let swigluEnvEnabled: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_VERIFY_FUSED_SWIGLU"] != "0"

    private static let swigluVerifyLock = NSLock()
    nonisolated(unsafe) private static var b4svVerified = false
    nonisolated(unsafe) private static var swigluVerified = false
    nonisolated(unsafe) private static var swigluFP8Verified = false
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

    public static var b4svActive: Bool {
        b4svEnabled && swigluVerifyLock.withLock { b4svVerified }
    }

    public static var swigluActive: Bool {
        swigluEnvEnabled && swigluVerifyLock.withLock { swigluVerified }
    }

    public static var swigluFP8Active: Bool {
        swigluEnvEnabled && swigluVerifyLock.withLock { swigluFP8Verified }
    }

    private struct SwiGLULaunchKey: Hashable {
        let dtype: DType
        let rows: Int
        let inputs: Int
        let outputs: Int
        let fragments: Int
        let b4sv: Bool
        let scaled: Bool
        let castFP8: Bool
    }

    nonisolated(unsafe) private static var swigluLaunchCache:
        [SwiGLULaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    private static let swigluSource = """
        constexpr int WK = 8;
        constexpr int BK = 128;
        constexpr short TM = (V + 7) / 8;
        constexpr int ROWS = 8 * TN;
        constexpr int row_bytes = K / 2;
        constexpr int row_scales = K / 16;
        static_assert(K % BK == 0, "whole chunks only");
        const int tg = int(threadgroup_position_in_grid.y);
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const short qid = simd_lid / 4;
        const short fm = (qid & 4) + ((simd_lid / 2) % 4);
        const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
        const int y_col = tg * ROWS;

        const device T* xr[TM][2];
        float2 Cg[TN][TM];
        float2 Cu[TN][TM];
        MO_WO_UNROLL
        for (short mt = 0; mt < TM; mt++) {
            MO_WO_UNROLL
            for (short j = 0; j < 2; j++) {
                xr[mt][j] = x + size_t(min(8 * mt + fn + j, V - 1)) * K + 4 * fm;
            }
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                Cg[t][mt] = float2(0);
                Cu[t][mt] = float2(0);
            }
        }
        if constexpr (B4SV) {
            const uint sshift = fn >= 4 ? 8u : 0u;
            const device uint* wlg[TN];
            const device uint2* sgv[TN];
            const device uint* wlu[TN];
            const device uint2* suv[TN];
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                const int row = min(y_col + fm + 8 * t, N - 1);
                wlg[t] = (const device uint*)(
                    (const device uint8_t*)wg + size_t(row) * row_bytes + 2 * fn);
                sgv[t] = (const device uint2*)(scg + size_t(row) * row_scales);
                wlu[t] = (const device uint*)(
                    (const device uint8_t*)wu + size_t(row) * row_bytes + 2 * fn);
                suv[t] = (const device uint2*)(scu + size_t(row) * row_scales);
            }
            for (int k = int(simd_gid) * BK; k < K; k += WK * BK) {
                uint qg[4][TN], qu[4][TN];
                half2 hsg0[TN], hsg1[TN], hsu0[TN], hsu1[TN];
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    const uint2 rg = sgv[t][k / BK];
                    hsg0[t] = cbv2_vn_e4m3x2((rg.x >> sshift) & 0x00FF00FFu);
                    hsg1[t] = cbv2_vn_e4m3x2((rg.y >> sshift) & 0x00FF00FFu);
                    const uint2 ru = suv[t][k / BK];
                    hsu0[t] = cbv2_vn_e4m3x2((ru.x >> sshift) & 0x00FF00FFu);
                    hsu1[t] = cbv2_vn_e4m3x2((ru.y >> sshift) & 0x00FF00FFu);
                }
                MO_WO_UNROLL
                for (short block = 0; block < 4; block++) {
                    const int b = k + 32 * block;
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        qg[block][t] = wlg[t][b / 8];
                        qu[block][t] = wlu[t][b / 8];
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
                            xv[mt][j] = ushort4(*(const device packed_ushort4*)(xr[mt][j] + b));
                        }
                    }
                    half sg[TN], su[TN];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        sg[t] = (block < 2) ? hsg0[t][block] : hsg1[t][block - 2];
                        su[t] = (block < 2) ? hsu0[t][block] : hsu1[t][block - 2];
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
                            const float2 ag = mo_wo_pair(qg[block][t], i, sg[t]);
                            const float2 au = mo_wo_pair(qu[block][t], i, su[t]);
                            MO_WO_UNROLL
                            for (short mt = 0; mt < TM; mt++) {
                                mo_wo_mma(Cg[t][mt], ag, bf[mt]);
                                mo_wo_mma(Cu[t][mt], au, bf[mt]);
                            }
                        }
                    }
                }
            }
        } else {
            const device uint* wlg[TN];
            const device uint8_t* slg[TN];
            const device uint* wlu[TN];
            const device uint8_t* slu[TN];
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                const int row = min(y_col + fm + 8 * t, N - 1);
                wlg[t] = (const device uint*)(
                    (const device uint8_t*)wg + size_t(row) * row_bytes + 2 * fn);
                slg[t] = scg + size_t(row) * row_scales + fn / 4;
                wlu[t] = (const device uint*)(
                    (const device uint8_t*)wu + size_t(row) * row_bytes + 2 * fn);
                slu[t] = scu + size_t(row) * row_scales + fn / 4;
            }
            for (int k = int(simd_gid) * BK; k < K; k += WK * BK) {
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
                    uint qg[TN], qu[TN];
                    half sg[TN], su[TN];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        qg[t] = wlg[t][b / 8];
                        sg[t] = mo_wo_e4m3(slg[t][b / 16]);
                        qu[t] = wlu[t][b / 8];
                        su[t] = mo_wo_e4m3(slu[t][b / 16]);
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
                            const float2 ag = mo_wo_pair(qg[t], i, sg[t]);
                            const float2 au = mo_wo_pair(qu[t], i, su[t]);
                            MO_WO_UNROLL
                            for (short mt = 0; mt < TM; mt++) {
                                mo_wo_mma(Cg[t][mt], ag, bf[mt]);
                                mo_wo_mma(Cu[t][mt], au, bf[mt]);
                            }
                        }
                    }
                }
            }
        }

        constexpr int tile_elems = TN * TM * 4;
        threadgroup float partials[7 * tile_elems * 32];
        if (simd_gid > 0) {
            threadgroup float* p = partials + (simd_gid - 1) * tile_elems * 32 + simd_lid;
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                MO_WO_UNROLL
                for (short mt = 0; mt < TM; mt++) {
                    p[((t * TM + mt) * 4 + 0) * 32] = Cg[t][mt][0];
                    p[((t * TM + mt) * 4 + 1) * 32] = Cg[t][mt][1];
                    p[((t * TM + mt) * 4 + 2) * 32] = Cu[t][mt][0];
                    p[((t * TM + mt) * 4 + 3) * 32] = Cu[t][mt][1];
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
                        Cg[t][mt][0] += p[((t * TM + mt) * 4 + 0) * 32];
                        Cg[t][mt][1] += p[((t * TM + mt) * 4 + 1) * 32];
                        Cu[t][mt][0] += p[((t * TM + mt) * 4 + 2) * 32];
                        Cu[t][mt][1] += p[((t * TM + mt) * 4 + 3) * 32];
                    }
                }
            }
            const float tsg0 = tsg[0];
            const float tsu0 = tsu[0];
            const float m0 = fp8_mu[0];
            const float m1 = fp8_mu[1];
            const float bound = fp8_bd[0];
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            const bfloat16_t zero_bf = bfloat16_t(0.0f);
            const bfloat16_t one_bf = bfloat16_t(1.0f);
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                const int out = y_col + fm + 8 * t;
                if (out < N) {
                    MO_WO_UNROLL
                    for (short mt = 0; mt < TM; mt++) {
                        MO_WO_UNROLL
                        for (short j = 0; j < 2; j++) {
                            if (8 * mt + fn + j < V) {
                                const bfloat16_t gv = mo_wo_result<bfloat16_t>(Cg[t][mt][j], SCALED, tsg0);
                                const bfloat16_t uv = mo_wo_result<bfloat16_t>(Cu[t][mt][j], SCALED, tsu0);
                                const bfloat16_t ax = (gv < zero_bf) ? -gv : gv;
                                const bfloat16_t ey = static_cast<bfloat16_t>(
                                    metal::precise::exp(static_cast<float>(ax)));
                                const bfloat16_t sy = one_bf / (one_bf + ey);
                                const bfloat16_t sig = (gv < zero_bf) ? sy : (one_bf - sy);
                                bfloat16_t act = (gv * sig) * uv;
                                if constexpr (CAST_FP8) {
                                    const float scaled = static_cast<float>(act) * m0;
                                    const uint8_t bits = modelopt_e4m3_bits(scaled);
                                    const float decoded = (bits & 0x7F) == 0x7F
                                        ? nan_value : modelopt_e4m3_value(bits);
                                    const bool invalid =
                                        metal::isnan(scaled) || metal::abs(scaled) > bound;
                                    act = static_cast<bfloat16_t>(
                                        invalid ? nan_value : decoded * m1);
                                }
                                y[size_t(8 * mt + fn + j) * N + out] = static_cast<T>(act);
                            }
                        }
                    }
                }
            }
        }
        """

    private static let swigluKernel = MLXFast.metalKernel(
        name: "cbv2_verify_nvfp4_swiglu",
        inputNames: ["wg", "scg", "tsg", "wu", "scu", "tsu", "x", "fp8_mu", "fp8_bd"],
        outputNames: ["y"],
        source: swigluSource,
        header: ModelOptActivationKernels.header + ModelOptWeightOnlyNVFP4RowExact.header + e4m3x2Helper)

    /// Fused `compiledSiluProduct(gate(x), up(x))` (plus optional inline FP8
    /// activation cast when `fp8Multipliers` and `fp8OverflowBound` are given)
    /// in a single NVFP4 launch: shares activation loads between gate and up
    /// and eliminates intermediate `[..., N]` BF16 gate/up buffers and the
    /// standalone SwiGLU (and FP8 cast) dispatches.
    public static func swiglu(
        gate: ModelOptWeightOnlyNVFP4Linear, up: ModelOptWeightOnlyNVFP4Linear, _ x: MLXArray,
        fp8Multipliers: MLXArray? = nil, fp8OverflowBound: MLXArray? = nil
    ) -> MLXArray? {
        let inputs = x.dim(-1)
        guard inputs > 0 else { return nil }
        let rows = x.size / inputs
        let outputs = gate.weight.dim(0)
        guard supports(rows: rows, inputs: inputs, dtype: x.dtype),
            up.weight.dim(0) == outputs,
            gate.inputDimensions == inputs, up.inputDimensions == inputs,
            takes(
                weight: gate.weight, blockScales: gate.weightScale,
                tensorScale: gate.weightScale2, inputs: inputs),
            takes(
                weight: up.weight, blockScales: up.weightScale,
                tensorScale: up.weightScale2, inputs: inputs)
        else { return nil }
        let castFP8 = fp8Multipliers != nil && fp8OverflowBound != nil
        let useB4SV = b4svActive && (!CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance)
        let fragments = useB4SV ? 2 : 1
        let key = SwiGLULaunchKey(
            dtype: x.dtype, rows: rows, inputs: inputs, outputs: outputs,
            fragments: fragments, b4sv: useB4SV, scaled: true, castFP8: castFP8)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = launchLock.withLock({ swigluLaunchCache[key] }) {
            launch = hit
        } else {
            let tile = 8 * fragments
            let tiles = (outputs + tile - 1) / tile
            let made = swigluKernel.prepare(
                template: [
                    ("T", x.dtype), ("V", rows), ("K", inputs), ("N", outputs),
                    ("TN", fragments), ("B4SV", useB4SV),
                    ("SCALED", true), ("CAST_FP8", castFP8),
                ],
                grid: (32, 8 * tiles, 1), threadGroup: (32, 8, 1),
                outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])
            launchLock.withLock { swigluLaunchCache[key] = made }
            launch = made
        }
        let result = launch([
            gate.weight, gate.weightScale, gate.weightScale2.reshaped(1),
            up.weight, up.weightScale, up.weightScale2.reshaped(1),
            x.reshaped(rows, inputs),
            fp8Multipliers ?? dummyFP8Mu, fp8OverflowBound ?? dummyFP8Bound,
        ])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }

    /// Convenience overload extracting FP8 activation cast parameters from
    /// `fp8Down` inside `MLXLMCommon` (returns nil if `fp8Down` is not FP8).
    public static func swiglu(
        gate: ModelOptWeightOnlyNVFP4Linear, up: ModelOptWeightOnlyNVFP4Linear, _ x: MLXArray,
        fp8Down: ModelOptCalibratedLinear
    ) -> MLXArray? {
        guard let constants = fp8Down.graphConstants,
            case .fp8(let mu, let bd) = constants.storage
        else { return nil }
        return swiglu(gate: gate, up: up, x, fp8Multipliers: mu, fp8OverflowBound: bd)
    }

    /// Load-time 0-ULP parity check and warmup for `b4+SV` (`TN=2, PF=2`) and
    /// `swiglu` (both plain and inline-FP8-cast modes across all verify row
    /// counts `1...8`) on non-zero BF16 inputs. Fails closed (`b4svVerified = false`
    /// / `swigluVerified = false`) on any bit mismatch.
    public static func verifyAndWarmSwiGLU(
        gate: ModelOptWeightOnlyNVFP4Linear, up: ModelOptWeightOnlyNVFP4Linear,
        dtype: DType, rows: ClosedRange<Int> = 1 ... maximumRows
    ) {
        guard (swigluEnvEnabled || b4svEnabled), dtype == .bfloat16 else { return }
        let prevLow = CBv2DraftRegime.lowAcceptance
        CBv2DraftRegime.lowAcceptance = true
        defer { CBv2DraftRegime.lowAcceptance = prevLow }
        let width = gate.inputDimensions
        let (b4svOK, plainOK, fp8OK): (Bool, Bool, Bool) =
            (try? MLX.withError { errors -> (Bool, Bool, Bool) in
                let testRows = 2
                var vals = [Float]()
                vals.reserveCapacity(testRows * width)
                for i in 0 ..< (testRows * width) {
                    vals.append(Float(((i * 7 + 3) % 29) - 14) * 0.125)
                }
                let x = MLXArray(vals, [1, testRows, width]).asType(.bfloat16)

                // 1. Verify b4+SV (TN=2, PF=2) single-projection 0-ULP parity against stock row-exact.
                var b4svPass = false
                if b4svEnabled {
                    let prevOverride = geometryOverride
                    geometryOverride = Geometry(fragments: 2, prefetch: 2)
                    let b4Out = product(gate, x)
                    geometryOverride = prevOverride
                    if let b4Out, let refGate = CBv2VerifyScales.product(gate, x) {
                        let diffB4 = (
                            contiguous(b4Out).view(dtype: .uint16)
                                .!= contiguous(refGate).view(dtype: .uint16)
                        ).asType(.int32).sum()
                        eval(diffB4)
                        try errors.check()
                        b4svPass = diffB4.item(Int.self) == 0
                    }
                }
                if b4svPass {
                    swigluVerifyLock.withLock { b4svVerified = true }
                }

                // 2. Verify fused gate+up+SwiGLU (plain + inline FP8 cast) 0-ULP parity
                // against the independent CBv2VerifyScales reference.
                guard swigluEnvEnabled else { return (b4svPass, false, false) }
                guard let refPair = CBv2VerifyScales.products([gate, up], x),
                    let fused = swiglu(gate: gate, up: up, x)
                else { return (b4svPass, false, false) }
                let refAct = compiledSiluProduct(refPair[0], refPair[1])
                let diff = (
                    contiguous(fused).view(dtype: .uint16)
                        .!= contiguous(refAct).view(dtype: .uint16)
                ).asType(.int32).sum()
                eval(diff)
                try errors.check()
                let plainPass = diff.item(Int.self) == 0
                var fp8Pass = false
                if plainPass {
                    let mu = MLXArray([Float(1.25), Float(0.8)])
                    let bd = MLXArray([Float(448.0)])
                    if let fusedFP8 = swiglu(
                        gate: gate, up: up, x, fp8Multipliers: mu, fp8OverflowBound: bd)
                    {
                        let refFP8 = ModelOptActivationKernels.fp8(
                            refAct, multipliers: mu, overflowBound: bd)
                        let diffFP8 = (
                            contiguous(fusedFP8).view(dtype: .uint16)
                                .!= contiguous(refFP8).view(dtype: .uint16)
                        ).asType(.int32).sum()
                        eval(diffFP8)
                        try errors.check()
                        fp8Pass = diffFP8.item(Int.self) == 0
                    }
                }
                if plainPass {
                    var warm = [MLXArray]()
                    for r in rows {
                        let z = MLXArray.zeros([1, r, width], dtype: dtype)
                        if let y = swiglu(gate: gate, up: up, z) { warm.append(y) }
                        if fp8Pass,
                            let yf = swiglu(
                                gate: gate, up: up, z,
                                fp8Multipliers: dummyFP8Mu, fp8OverflowBound: dummyFP8Bound)
                        {
                            warm.append(yf)
                        }
                    }
                    if !warm.isEmpty { eval(warm) }
                    try errors.check()
                }
                return (b4svPass, plainPass, fp8Pass)
            }) ?? (false, false, false)
        swigluVerifyLock.withLock {
            b4svVerified = b4svOK
            swigluVerified = plainOK
            swigluFP8Verified = fp8OK
        }
    }
}
