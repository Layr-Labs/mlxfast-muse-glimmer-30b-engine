// Copyright © 2026 Eigen Labs.

import MLX

// MARK: - Hierarchical top-2

// Metal entry-point names remain stable so extraction preserves pipeline identity.
/// Shared exact ordering for the two-stage candidate-only top-2 reduction:
/// value descending, token id ascending on exact ties, and NaNs last.
let cbv2TopTwoHeader = """
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

private let cbv2FusedReadoutTopTwoStage1 = MLXFast.metalKernel(
    name: "cbv2_fused_readout_top2_s1",
    inputNames: ["raw_logits", "params"],
    outputNames: ["partial_ids", "partial_values", "partial_nf"],
    source: """
        constexpr uint CHUNKS = 64;
        const uint lid = thread_position_in_threadgroup.x;
        const uint chunk = threadgroup_position_in_grid.x;
        const uint row = threadgroup_position_in_grid.y;
        const uint vocab = uint(V_);
        const uint chunk_size = (vocab + CHUNKS - 1) / CHUNKS;
        const uint start = chunk * chunk_size;
        const uint end = min(vocab, start + chunk_size);

        const bfloat16_t mult = bfloat16_t(params[0]);
        const bfloat16_t cap = bfloat16_t(params[1]);
        const bool has_cap = params[1] > 0.0f;

        const device bfloat16_t* row_ptr = raw_logits + ulong(row) * ulong(vocab);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        uint local_nf = 0u;

        const uint start4 = (start + 3u) & ~3u;
        const uint end4 = end & ~3u;
        if (lid == 0) {
            for (uint i = start; i < min(start4, end); i++) {
                bfloat16_t s = row_ptr[i] * mult;
                if (has_cap) {
                    bfloat16_t q = s / cap;
                    bfloat16_t t = bfloat16_t(metal::precise::tanh(float(q)));
                    s = t * cap;
                }
                float fv = float(s);
                if (!isfinite(fv)) { local_nf = 1u; }
                darkbloom_qwen35_mtp_top2_insert(local, fv, i);
            }
        }
        if (end4 > start4) {
            const device vec<bfloat16_t, 4>* src4 =
                (const device vec<bfloat16_t, 4>*)(row_ptr + start4);
            const uint count4 = (end4 - start4) >> 2;
            for (uint j = lid; j < count4; j += 256u) {
                const vec<bfloat16_t, 4> v = src4[j];
                const uint base_i = start4 + (j << 2);
                #pragma unroll
                for (uint k = 0; k < 4; k++) {
                    bfloat16_t s = v[k] * mult;
                    if (has_cap) {
                        bfloat16_t q = s / cap;
                        bfloat16_t t = bfloat16_t(metal::precise::tanh(float(q)));
                        s = t * cap;
                    }
                    float fv = float(s);
                    if (!isfinite(fv)) { local_nf = 1u; }
                    darkbloom_qwen35_mtp_top2_insert(local, fv, base_i + k);
                }
            }
        }
        if (lid == 0) {
            const uint tail_start = max(start4, end4);
            for (uint i = tail_start; i < end; i++) {
                bfloat16_t s = row_ptr[i] * mult;
                if (has_cap) {
                    bfloat16_t q = s / cap;
                    bfloat16_t t = bfloat16_t(metal::precise::tanh(float(q)));
                    s = t * cap;
                }
                float fv = float(s);
                if (!isfinite(fv)) { local_nf = 1u; }
                darkbloom_qwen35_mtp_top2_insert(local, fv, i);
            }
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[256];
        threadgroup uint nf_scratch[256];
        scratch[lid] = local;
        nf_scratch[lid] = local_nf;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lid];
                darkbloom_qwen35_mtp_top2_state other = scratch[lid + stride];
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.second_value, other.second_id);
                }
                scratch[lid] = merged;
                nf_scratch[lid] |= nf_scratch[lid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            const uint base = (row * CHUNKS + chunk) * 2;
            const uint sentinel_id = vocab + chunk * 2;
            const float sentinel_value = as_type<float>(0x7fc00000u);
            partial_ids[base] = scratch[0].count > 0
                ? int(scratch[0].first_id) : int(sentinel_id);
            partial_ids[base + 1] = scratch[0].count > 1
                ? int(scratch[0].second_id) : int(sentinel_id + 1);
            partial_values[base] = scratch[0].count > 0
                ? scratch[0].first_value : sentinel_value;
            partial_values[base + 1] = scratch[0].count > 1
                ? scratch[0].second_value : sentinel_value;
            partial_nf[row * CHUNKS + chunk] = int(nf_scratch[0]);
        }
    """,
    header: cbv2TopTwoHeader
)

private let cbv2FusedReadoutTopTwoStage2 = MLXFast.metalKernel(
    name: "cbv2_fused_readout_top2_s2",
    inputNames: ["partial_ids", "partial_values", "partial_nf"],
    outputNames: ["top_ids", "top_values", "non_finite"],
    source: """
        constexpr uint CHUNKS = 64;
        const uint lid = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.y;
        const uint base = (row * CHUNKS + lid) * 2;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base], uint(partial_ids[base]));
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base + 1], uint(partial_ids[base + 1]));
        uint nf = uint(partial_nf[row * CHUNKS + lid]);

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[64];
        threadgroup uint nf_scratch[64];
        scratch[lid] = local;
        nf_scratch[lid] = nf;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (lid < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lid];
                darkbloom_qwen35_mtp_top2_state other = scratch[lid + stride];
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.first_value, other.first_id);
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.second_value, other.second_id);
                scratch[lid] = merged;
                nf_scratch[lid] |= nf_scratch[lid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            const uint output_base = row * 2;
            top_ids[output_base] = int(scratch[0].first_id);
            top_ids[output_base + 1] = int(scratch[0].second_id);
            top_values[output_base] = scratch[0].first_value;
            top_values[output_base + 1] = scratch[0].second_value;
            non_finite[row] = nf_scratch[0] != 0u ? 1 : 0;
        }
    """,
    header: cbv2TopTwoHeader
)

public func cbv2FusedReadoutTopTwoAndNonFinite(
    _ rawProduct: MLXArray, outputMultiplier: Float, softcap: Float
) -> (ids: MLXArray, values: MLXArray, nonFinite: MLXArray) {
    precondition(rawProduct.ndim == 3 && rawProduct.dtype == .bfloat16)
    let batch = rawProduct.dim(0)
    let width = rawProduct.dim(1)
    let vocab = rawProduct.dim(2)
    let rows = batch * width
    precondition(rows > 0 && vocab >= 2)
    let chunks = 64
    let params = MLXArray([outputMultiplier, softcap])
    let flat = rawProduct.reshaped([rows, vocab])
    let partials = cbv2FusedReadoutTopTwoStage1(
        [flat, params],
        template: [("V_", vocab)],
        grid: (256 * chunks, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, chunks, 2], [rows, chunks, 2], [rows, chunks]],
        outputDTypes: [.int32, .float32, .int32])
    let outputs = cbv2FusedReadoutTopTwoStage2(
        partials,
        grid: (chunks, rows, 1),
        threadGroup: (chunks, 1, 1),
        outputShapes: [[rows, 2], [rows, 2], [rows]],
        outputDTypes: [.int32, .float32, .int32])
    return (
        outputs[0].reshaped([batch, width, 2]),
        outputs[1].reshaped([batch, width, 2]),
        outputs[2].reshaped([batch, width])
    )
}

private let cbv2FusedReadoutTopTwoStage2Chunks = MLXFast.metalKernel(
    name: "cbv2_fused_readout_top2_s2_chunks",
    inputNames: ["partial_ids", "partial_values", "partial_nf"],
    outputNames: ["top_ids", "top_values", "non_finite"],
    source: """
        constexpr uint CHUNKS = uint(CHUNKS_);
        const uint lid = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.y;

        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        uint nf = 0u;
        for (uint c = lid; c < CHUNKS; c += 256u) {
            const uint base = (row * CHUNKS + c) * 2;
            darkbloom_qwen35_mtp_top2_insert(
                local, partial_values[base], uint(partial_ids[base]));
            darkbloom_qwen35_mtp_top2_insert(
                local, partial_values[base + 1], uint(partial_ids[base + 1]));
            nf |= uint(partial_nf[row * CHUNKS + c]);
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[256];
        threadgroup uint nf_scratch[256];
        scratch[lid] = local;
        nf_scratch[lid] = nf;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lid];
                darkbloom_qwen35_mtp_top2_state other = scratch[lid + stride];
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.second_value, other.second_id);
                }
                scratch[lid] = merged;
                nf_scratch[lid] |= nf_scratch[lid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            const uint output_base = row * 2;
            top_ids[output_base] = int(scratch[0].first_id);
            top_ids[output_base + 1] = int(scratch[0].second_id);
            top_values[output_base] = scratch[0].first_value;
            top_values[output_base + 1] = scratch[0].second_value;
            non_finite[row] = nf_scratch[0] != 0u ? 1 : 0;
        }
    """,
    header: cbv2TopTwoHeader
)

/// Finalizes chunked fused readout top-2 and non-finite partials into
/// `[batch, width, 2]` IDs/values and `[batch, width]` non-finite flags.
func cbv2FusedReadoutTopTwoFinalizeChunks(
    partialIDs: MLXArray, partialValues: MLXArray, partialNF: MLXArray,
    batch: Int, width: Int, chunks: Int
) -> (ids: MLXArray, values: MLXArray, nonFinite: MLXArray) {
    let rows = batch * width
    let outputs = cbv2FusedReadoutTopTwoStage2Chunks(
        [partialIDs, partialValues, partialNF],
        template: [("CHUNKS_", chunks)],
        grid: (256, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, 2], [rows, 2], [rows]],
        outputDTypes: [.int32, .float32, .int32])
    return (
        outputs[0].reshaped([batch, width, 2]),
        outputs[1].reshaped([batch, width, 2]),
        outputs[2].reshaped([batch, width])
    )
}
