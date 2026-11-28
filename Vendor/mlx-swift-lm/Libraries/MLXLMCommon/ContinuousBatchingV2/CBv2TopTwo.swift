// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

// MARK: - Hierarchical top-2

// Metal entry-point names remain stable so extraction preserves pipeline identity.
/// Shared exact ordering for the two-stage candidate-only top-2 reduction:
/// value descending, token id ascending on exact ties, and NaNs last.
private let cbv2TopTwoHeader = """
    struct darkbloom_qwen35_mtp_top2_state {
        float first_value;
        float second_value;
        uint first_id;
        uint second_id;
        uint count;
    };

    inline darkbloom_qwen35_mtp_top2_state darkbloom_qwen35_mtp_top2_empty() {
        darkbloom_qwen35_mtp_top2_state state;
        state.first_value = 0.0f;
        state.second_value = 0.0f;
        state.first_id = 0;
        state.second_id = 0;
        state.count = 0;
        return state;
    }

    inline bool darkbloom_qwen35_mtp_top2_better(
        float candidate_value,
        uint candidate_id,
        float current_value,
        uint current_id
    ) {
        bool candidate_nan = isnan(candidate_value);
        bool current_nan = isnan(current_value);
        if (candidate_nan != current_nan) {
            return !candidate_nan;
        }
        if (candidate_value > current_value) {
            return true;
        }
        if (candidate_value < current_value) {
            return false;
        }
        return candidate_id < current_id;
    }

    inline void darkbloom_qwen35_mtp_top2_insert(
        thread darkbloom_qwen35_mtp_top2_state &state,
        float value,
        uint id
    ) {
        if (state.count > 0 && state.first_id == id) {
            return;
        }
        if (state.count > 1 && state.second_id == id) {
            return;
        }
        if (state.count == 0
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.first_value, state.first_id)) {
            if (state.count > 0) {
                state.second_value = state.first_value;
                state.second_id = state.first_id;
            }
            state.first_value = value;
            state.first_id = id;
            state.count = min(state.count + 1, 2u);
            return;
        }
        if (state.count == 1
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.second_value, state.second_id)) {
            state.second_value = value;
            state.second_id = id;
            state.count = 2;
        }
    }
"""

/// Stage one: 32 threadgroups per row reduce disjoint vocabulary stripes.
private let cbv2TopTwoPartialKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();

        for (uint index = group * 256 + lane;
             index < vocab;
             index += 32 * 256) {
            ulong offset = ulong(row) * ulong(logits_strides[1])
                + ulong(index) * ulong(logits_strides[2]);
            darkbloom_qwen35_mtp_top2_insert(local, float(logits[offset]), index);
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[256];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.second_value, other.second_id);
                }
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint base = (row * 32 + group) * 2;
            uint sentinel_id = vocab + group * 2;
            float sentinel_value = as_type<float>(0x7fc00000u);
            partial_ids[base] = scratch[0].count > 0
                ? int(scratch[0].first_id) : int(sentinel_id);
            partial_ids[base + 1] = scratch[0].count > 1
                ? int(scratch[0].second_id) : int(sentinel_id + 1);
            partial_values[base] = scratch[0].count > 0
                ? scratch[0].first_value : sentinel_value;
            partial_values[base + 1] = scratch[0].count > 1
                ? scratch[0].second_value : sentinel_value;
        }
    """,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Stage two: one 32-lane threadgroup per row merges the partial pairs.
private let cbv2TopTwoFinalizeKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize",
    inputNames: ["partial_ids", "partial_values"],
    outputNames: ["top_ids", "top_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint base = (row * 32 + lane) * 2;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base], uint(partial_ids[base]));
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base + 1], uint(partial_ids[base + 1]));

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[32];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 16; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.first_value, other.first_id);
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.second_value, other.second_id);
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint output_base = row * 2;
            top_ids[output_base] = int(scratch[0].first_id);
            top_ids[output_base + 1] = int(scratch[0].second_id);
            top_values[output_base] = scratch[0].first_value;
            top_values[output_base + 1] = scratch[0].second_value;
        }
    """,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Exact top-2 token ids and logit values for every row of `[1, rows, vocab]`.
///
/// Returns lazy device arrays shaped `[rows, 2]`: ids are `int32`, values are
/// `float32`. No evaluation or host read occurs here. Results are ordered by
/// value descending, then token id ascending on exact ties, with NaNs last.
package func cbv2TopTwoRows(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
    precondition(logits.ndim == 3 && logits.dim(0) == 1)
    let rows = logits.dim(1)
    let vocabularySize = logits.dim(2)
    precondition(rows > 0 && vocabularySize >= 2)

    let partials = cbv2TopTwoPartialKernel(
        [logits],
        grid: (rows * 32 * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, 32, 2], [rows, 32, 2]],
        outputDTypes: [.int32, .float32]
    )
    let outputs = cbv2TopTwoFinalizeKernel(
        partials,
        grid: (rows * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[rows, 2], [rows, 2]],
        outputDTypes: [.int32, .float32]
    )
    return (outputs[0], outputs[1])
}

private let cbv2PackedReadoutHeader = """
    inline bfloat16_t cbv2_readout(
        bfloat16_t raw, bfloat16_t mult, bfloat16_t cap, bool has_cap
    ) {
        bfloat16_t scaled = raw * mult;
        if (has_cap) {
            bfloat16_t q = scaled / cap;
            bfloat16_t t = bfloat16_t(metal::precise::tanh(float(q)));
            scaled = t * cap;
        }
        return scaled;
    }

    // The high half orders rounded BF16 values, the low half breaks ties
    // by smaller chunk-local ID. Key zero denotes an empty candidate.
    inline uint cbv2_readout_key(bfloat16_t value, uint offset, thread uint& nf) {
        uint bits = uint(as_type<ushort>(value));
        uint magnitude = bits & 0x7fffu;
        nf |= uint(magnitude >= 0x7f80u);
        uint rank = (bits & 0x8000u) ? (bits ^ 0xffffu) : (bits ^ 0x8000u);
        rank = magnitude == 0u ? 0x8000u : rank;
        rank = magnitude > 0x7f80u ? 1u : rank;
        return (rank << 16) | (0xffffu - offset);
    }

    inline float cbv2_readout_value(
        uint key, const device bfloat16_t* raw, uint id,
        bfloat16_t mult, bfloat16_t cap, bool has_cap
    ) {
        uint rank = key >> 16;
        // The comparison key intentionally discards zero sign and NaN payload.
        if (rank == 0x8000u || rank == 1u) {
            return float(cbv2_readout(raw[id], mult, cap, has_cap));
        }
        uint bits = rank ^ ((rank & 0x8000u) ? 0x8000u : 0xffffu);
        return float(as_type<bfloat16_t>(ushort(bits)));
    }

    // Conservative lower bound on finite raw value that can reach or tie `key`.
    inline float cbv2_raw_skip_bound(uint key, float inv_mult) {
        uint rank = key >> 16;
        if (rank <= 0x8000u || rank >= 0xff80u) { return -INFINITY; }
        uint bits = rank ^ 0x8000u;
        float v = float(as_type<bfloat16_t>(ushort(bits)));
        return (v * 0.90f) * inv_mult;
    }

    inline void cbv2_insert_readout_key(thread uint2& pair, uint key) {
        pair.y = max(pair.y, min(pair.x, key));
        pair.x = max(pair.x, key);
    }

    inline uint2 cbv2_simd_readout_keys(uint2 pair) {
        uint first = simd_max(pair.x);
        uint second = simd_max(pair.x == first ? pair.y : pair.x);
        return uint2(first, second);
    }
    """

private let cbv2FusedReadoutTopTwoHeader = cbv2TopTwoHeader + """
    inline void cbv2_merge_readout_top2(
        thread darkbloom_qwen35_mtp_top2_state& state,
        darkbloom_qwen35_mtp_top2_state other
    ) {
        if (other.count > 0u) {
            darkbloom_qwen35_mtp_top2_insert(state, other.first_value, other.first_id);
        }
        if (other.count > 1u) {
            darkbloom_qwen35_mtp_top2_insert(state, other.second_value, other.second_id);
        }
    }

    inline void cbv2_reduce_simd_readout_top2(
        thread darkbloom_qwen35_mtp_top2_state& state, uint lane
    ) {
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            darkbloom_qwen35_mtp_top2_state other;
            other.first_value = simd_shuffle_down(state.first_value, offset);
            other.second_value = simd_shuffle_down(state.second_value, offset);
            other.first_id = simd_shuffle_down(state.first_id, offset);
            other.second_id = simd_shuffle_down(state.second_id, offset);
            other.count = simd_shuffle_down(state.count, offset);
            if (lane < offset) {
                cbv2_merge_readout_top2(state, other);
            }
        }
    }
    """

private let cbv2FusedReadoutTopTwoStage1 = MLXFast.metalKernel(
    name: "cbv2_fused_readout_top2_s1",
    inputNames: ["raw_logits", "params"],
    outputNames: ["partial_ids", "partial_values", "partial_nf"],
    source: """
        constexpr uint CHUNKS = uint(C_);
        const uint lid = thread_position_in_threadgroup.x;
        const uint chunk = threadgroup_position_in_grid.x;
        const uint row = threadgroup_position_in_grid.y;
        constexpr uint vocab = uint(V_);
        constexpr uint chunk_size = (((vocab - 1) / CHUNKS + 1) + 3u) & ~3u;
        const uint start = chunk * chunk_size;
        const uint end = min(vocab, start + chunk_size);

        const bfloat16_t mult = bfloat16_t(params[0]);
        const bfloat16_t cap = bfloat16_t(params[1]);
        const bool has_cap = params[1] > 0.0f;
        const float fmult = float(mult);
        const bool can_skip = has_cap && fmult > 1e-6f && fmult < 1e6f;
        const float inv_mult = can_skip ? (1.0f / fmult) : 0.0f;

        const ulong row_base = ulong(row) * ulong(vocab);
        const device bfloat16_t* row_ptr = raw_logits + row_base;
        uint2 keys(0u);
        uint local_nf = 0u;
        float skip_bound = -INFINITY;

        const uint align = vocab % 4u == 0u ? 0u : (4u - uint(row_base & 3ul)) & 3u;
        const uint start4 = start + align;
        const uint count4 = start4 < end ? (end - start4) >> 2 : 0u;
        if (lid == 0) {
            for (uint i = start; i < min(start4, end); i++) {
                const bfloat16_t value = cbv2_readout(row_ptr[i], mult, cap, has_cap);
                cbv2_insert_readout_key(keys, cbv2_readout_key(value, i - start, local_nf));
            }
            if (can_skip) { skip_bound = cbv2_raw_skip_bound(keys.y, inv_mult); }
        }
        if (count4 > 0u) {
            const device vec<bfloat16_t, 4>* src4 =
                (const device vec<bfloat16_t, 4>*)(row_ptr + start4);
            for (uint j = lid; j < count4; j += 256u) {
                const vec<bfloat16_t, 4> v = src4[j];
                const uint base_i = start4 + (j << 2);
                #pragma unroll
                for (uint k = 0; k < 4; k++) {
                    const bfloat16_t raw = v[k];
                    const float fr = float(raw);
                    if (can_skip && fr < skip_bound && fr > -1e20f) {
                        continue;
                    }
                    const bfloat16_t value = cbv2_readout(raw, mult, cap, has_cap);
                    const uint old_y = keys.y;
                    cbv2_insert_readout_key(
                        keys, cbv2_readout_key(value, base_i + k - start, local_nf));
                    if (can_skip && keys.y != old_y) {
                        skip_bound = cbv2_raw_skip_bound(keys.y, inv_mult);
                    }
                }
            }
        }
        if (lid == 0) {
            const uint tail_start = start4 + (count4 << 2);
            for (uint i = tail_start; i < end; i++) {
                const bfloat16_t value = cbv2_readout(row_ptr[i], mult, cap, has_cap);
                cbv2_insert_readout_key(keys, cbv2_readout_key(value, i - start, local_nf));
            }
        }

        const uint lane = lid & 31u;
        const uint group = lid >> 5;
        keys = cbv2_simd_readout_keys(keys);
        local_nf = simd_max(local_nf);
        threadgroup uint2 scratch[8];
        threadgroup uint nf_scratch[8];
        if (lane == 0) {
            scratch[group] = keys;
            nf_scratch[group] = local_nf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (group == 0) {
            keys = cbv2_simd_readout_keys(lane < 8 ? scratch[lane] : uint2(0u));
            local_nf = simd_max(lane < 8 ? nf_scratch[lane] : 0u);
        }

        if (lid == 0) {
            const uint base = (row * CHUNKS + chunk) * 2;
            for (uint j = 0; j < 2; ++j) {
                const uint key = keys[j];
                const uint id = start + (0xffffu - (key & 0xffffu));
                partial_ids[base + j] = key ? int(id) : int(vocab + chunk * 2 + j);
                partial_values[base + j] = key
                    ? cbv2_readout_value(key, row_ptr, id, mult, cap, has_cap)
                    : as_type<float>(0x7fc00000u);
            }
            partial_nf[row * CHUNKS + chunk] = int(local_nf);
        }
    """,
    header: cbv2PackedReadoutHeader
)

private func cbv2MakeReadoutTopTwoStage2(includeTargetIDs: Bool) -> MLXFast.MLXFastKernel {
    let writeTarget = includeTargetIDs
        ? "\n        target_ids[row] = nf != 0u ? \(CBv2NonFiniteLogits.invalidTarget) : int(local.first_id);"
        : ""
    let source = """
        constexpr uint CHUNKS = uint(C_);
        const uint lid = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = lid & 31u;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        uint nf = 0u;
        for (uint chunk = lid; chunk < CHUNKS; chunk += 64u) {
            const uint base = (row * CHUNKS + chunk) * 2;
            darkbloom_qwen35_mtp_top2_insert(
                local, partial_values[base], uint(partial_ids[base]));
            darkbloom_qwen35_mtp_top2_insert(
                local, partial_values[base + 1], uint(partial_ids[base + 1]));
            nf |= uint(partial_nf[row * CHUNKS + chunk]);
        }

        cbv2_reduce_simd_readout_top2(local, lane);
        nf = simd_max(nf);
        threadgroup darkbloom_qwen35_mtp_top2_state second_group[1];
        threadgroup uint second_nf[1];
        if (lid == 32u) {
            second_group[0] = local;
            second_nf[0] = nf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            cbv2_merge_readout_top2(local, second_group[0]);
            nf |= second_nf[0];
            const uint output_base = row * 2;
            top_ids[output_base] = int(local.first_id);
            top_ids[output_base + 1] = int(local.second_id);
            top_values[output_base] = local.first_value;
            top_values[output_base + 1] = local.second_value;
            non_finite[row] = nf != 0u ? 1 : 0;\(writeTarget)
        }
    """
    let outputNames = ["top_ids", "top_values", "non_finite"]
    return MLXFast.metalKernel(
        name: includeTargetIDs ? "cbv2_fused_readout_top2_s2_targets" : "cbv2_fused_readout_top2_s2",
        inputNames: ["partial_ids", "partial_values", "partial_nf"],
        outputNames: includeTargetIDs ? outputNames + ["target_ids"] : outputNames,
        source: source, header: cbv2FusedReadoutTopTwoHeader)
}

private let cbv2FusedReadoutTopTwoStage2 = cbv2MakeReadoutTopTwoStage2(includeTargetIDs: false)
private let cbv2FusedReadoutTargetIDsStage2 = cbv2MakeReadoutTopTwoStage2(includeTargetIDs: true)

package typealias CBv2TopTwoReadout = (
    ids: MLXArray, values: MLXArray, nonFinite: MLXArray, targetIDs: MLXArray?
)

private func cbv2PrepareReadoutStage2(rows: Int, chunks: Int, includeTargetIDs: Bool)
    -> MLXFast.MLXFastKernel.Launch
{
    let kernel = includeTargetIDs ? cbv2FusedReadoutTargetIDsStage2 : cbv2FusedReadoutTopTwoStage2
    var shapes = [[rows, 2], [rows, 2], [rows]]
    var dtypes: [DType] = [.int32, .float32, .int32]
    if includeTargetIDs {
        shapes.append([rows])
        dtypes.append(.int32)
    }
    return kernel.prepare(
        template: [("C_", chunks)], grid: (64, rows, 1), threadGroup: (64, 1, 1),
        outputShapes: shapes, outputDTypes: dtypes)
}

private func cbv2ReadoutOutputs(_ outputs: [MLXArray], batch: Int, width: Int) -> CBv2TopTwoReadout {
    (
        outputs[0].reshaped([batch, width, 2]),
        outputs[1].reshaped([batch, width, 2]),
        outputs[2].reshaped([batch, width]),
        outputs.count == 4 ? outputs[3].reshaped([batch, width]) : nil
    )
}

private let cbv2ReadoutParamsLock = NSLock()
nonisolated(unsafe) private var cbv2ReadoutParamsCache: [UInt64: MLXArray] = [:]

internal func cbv2CachedReadoutParams(outputMultiplier: Float, softcap: Float) -> MLXArray {
    let key = (UInt64(outputMultiplier.bitPattern) << 32) | UInt64(softcap.bitPattern)
    return cbv2ReadoutParamsLock.withLock {
        if let existing = cbv2ReadoutParamsCache[key] { return existing }
        let arr = MLXArray([outputMultiplier, softcap])
        eval(arr)
        cbv2ReadoutParamsCache[key] = arr
        return arr
    }
}

private struct CBv2ReadoutLaunchKey: Hashable {
    let rows: Int
    let vocab: Int
    let chunks: Int
    let includeTargetIDs: Bool
}

private let cbv2ReadoutLaunchLock = NSLock()
nonisolated(unsafe) private var cbv2ReadoutLaunchCache:
    [CBv2ReadoutLaunchKey: (s1: MLXFast.MLXFastKernel.Launch, s2: MLXFast.MLXFastKernel.Launch)] = [:]

private struct CBv2ReadoutStage2Key: Hashable {
    let rows: Int
    let chunks: Int
    let includeTargetIDs: Bool
}

nonisolated(unsafe) private var cbv2ReadoutStage2Cache:
    [CBv2ReadoutStage2Key: MLXFast.MLXFastKernel.Launch] = [:]

internal func cbv2FusedReadoutTopTwoMergePartials(
    partialIDs: MLXArray, partialValues: MLXArray, partialNF: MLXArray,
    batch: Int, width: Int, chunks: Int, includeTargetIDs: Bool
) -> CBv2TopTwoReadout {
    let rows = batch * width
    let key = CBv2ReadoutStage2Key(rows: rows, chunks: chunks, includeTargetIDs: includeTargetIDs)
    let s2: MLXFast.MLXFastKernel.Launch
    if let hit = cbv2ReadoutLaunchLock.withLock({ cbv2ReadoutStage2Cache[key] }) {
        s2 = hit
    } else {
        let made = cbv2PrepareReadoutStage2(
            rows: rows, chunks: chunks, includeTargetIDs: includeTargetIDs)
        cbv2ReadoutLaunchLock.withLock { cbv2ReadoutStage2Cache[key] = made }
        s2 = made
    }
    let outputs = s2([partialIDs, partialValues, partialNF])
    return cbv2ReadoutOutputs(outputs, batch: batch, width: width)
}

public func cbv2FusedReadoutTopTwoAndNonFinite(
    _ rawProduct: MLXArray, outputMultiplier: Float, softcap: Float
) -> (ids: MLXArray, values: MLXArray, nonFinite: MLXArray) {
    let readout = cbv2FusedReadoutTopTwoAndNonFinite(
        rawProduct, outputMultiplier: outputMultiplier, softcap: softcap, includeTargetIDs: false)
    return (readout.ids, readout.values, readout.nonFinite)
}

package func cbv2FusedReadoutTopTwoAndNonFinite(
    _ rawProduct: MLXArray, outputMultiplier: Float, softcap: Float, includeTargetIDs: Bool
) -> CBv2TopTwoReadout {
    precondition(rawProduct.ndim == 3 && rawProduct.dtype == .bfloat16)
    let batch = rawProduct.dim(0)
    let width = rawProduct.dim(1)
    let vocab = rawProduct.dim(2)
    let rows = batch * width
    precondition(rows > 0 && vocab >= 2)
    let chunks = max(64, (vocab - 1) / 65_536 + 1)
    let params = cbv2CachedReadoutParams(outputMultiplier: outputMultiplier, softcap: softcap)
    let flat = rawProduct.reshaped([rows, vocab])
    let key = CBv2ReadoutLaunchKey(
        rows: rows, vocab: vocab, chunks: chunks, includeTargetIDs: includeTargetIDs)
    let launches: (s1: MLXFast.MLXFastKernel.Launch, s2: MLXFast.MLXFastKernel.Launch)
    if let hit = cbv2ReadoutLaunchLock.withLock({ cbv2ReadoutLaunchCache[key] }) {
        launches = hit
    } else {
        let s1 = cbv2FusedReadoutTopTwoStage1.prepare(
            template: [("V_", vocab), ("C_", chunks)],
            grid: (256 * chunks, rows, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, chunks, 2], [rows, chunks, 2], [rows, chunks]],
            outputDTypes: [.int32, .float32, .int32])
        let s2 = cbv2PrepareReadoutStage2(
            rows: rows, chunks: chunks, includeTargetIDs: includeTargetIDs)
        launches = (s1, s2)
        cbv2ReadoutLaunchLock.withLock { cbv2ReadoutLaunchCache[key] = launches }
    }
    let partials = launches.s1([flat, params])
    return cbv2ReadoutOutputs(launches.s2(partials), batch: batch, width: width)
}
