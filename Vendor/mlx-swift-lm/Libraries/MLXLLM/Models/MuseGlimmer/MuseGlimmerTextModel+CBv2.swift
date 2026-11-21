// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

// MARK: - Layer structure

extension MuseGlimmerTextModelConfiguration {
    /// One owning layer per decoder layer: no KV sharing, no sinks.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        (0 ..< hiddenLayers).map {
            CBv2LayerKind(
                attention: isSliding($0) ? .slidingWindow(slidingWindow) : .full,
                headDim: headDimensions, kvHeads: kvHeads, queryHeads: attentionHeads)
        }
    }
}

extension MuseGlimmerTextModel {
    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.cbv2LayerKinds }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { try makeLayerCache($0.offset, $0.element) }
    }
}

// MARK: - Capabilities

extension MuseGlimmerTextModel: CBv2ModelCapabilityProviding {
    /// Attention K/V lives only in segmented paged storage with the observed
    /// layer dtypes; a contiguous backend is refused at engine build. Packed
    /// prefill stays off until row independence of the ModelOpt products over
    /// rectangular prompt cohorts is proven. Prompts are cut on the
    /// `prefillChunkAlignment` grid.
    public var cbv2Capabilities: CBv2ModelCapabilities {
        CBv2ModelCapabilities(
            supportsPrefixReuse: true, supportsPagedKV: true, requiresNativePagedKV: true,
            supportsCompiledDecode: false, supportsPackedPrefill: false, supportsMTP: true,
            requiresPagedKV: true, prefillChunkAlignment: prefillChunkAlignment)
    }
}

// MARK: - Speculative round command buffers

/// DFlash rounds of decode and verification rows (at most
/// `EngineLoopV2.speculativeCommandBufferRowTokens` tokens per row) take the
/// plain-decode limits too. Under MLX's defaults every NVFP4/FP8 projection's
/// weight read crosses the input bound, so the drafter's early submission
/// alone opens more than MLX's ten in-flight command buffers and its
/// `asyncEval` waits for the GPU before the verify graph can be built; the
/// verify forward then commits after nearly every projection. A round's
/// temporaries are small (depth + 1 rows; one 16-row block), and rounds that
/// carry a prompt chunk keep the defaults. Results do not depend on the
/// partitioning.
extension MuseGlimmerCBv2Adapter {
    public var speculativeCommandBufferLimits: GPU.CommandBufferLimits? {
        MuseGlimmerDecodeCommandBuffers.limits
    }
}

// MARK: - Prompt output

/// A prompt chunk runs the whole trunk and writes every layer's K/V; only the
/// vocabulary projection of rows nothing reads is left out. It keeps MLX's
/// arithmetic with exact windows on too.
extension MuseGlimmerTextModel: CBv2LanguageModelPrefillForwardable {
    public func cbv2Prefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        numerics.prompt {
            let hidden = model(inputEmbedding ?? model.inputEmbeddings(inputs), cache: cache)
            switch requirement {
            case .evaluationOnly: return hidden[0..., -1, 0 ..< 1]
            case .lastPositionLogits: return outputLogits(hidden[0..., -1, 0...])
            }
        }
    }
    // Packed prefill keeps the protocol's fail-closed default.
}

// MARK: - Teacher-forced readout

/// A multi-row teacher-forced forward is read at its last position only. Every
/// row runs each layer's attention and writes its K/V; the last layer's
/// feed-forward half, the final norm and the readout run on the last row. Exact
/// windows give a row the bytes it gets alone, so the logits do not change.
///
/// A prompt-length forward submits its trunk every
/// `lastPositionTrunkSubmitEvery` layers, so the GPU runs the early layers
/// while the host builds the later ones. Only the evaluation order changes:
/// the same operations on the same inputs, and the caller's single
/// `asyncEval` still roots the logits and every layer's K/V (each layer's
/// attention reads its cache write, so a submitted layer's K/V goes with it).
extension MuseGlimmerTextModel: CBv2LastPositionLogitsForwardable {
    /// Trunk layers per early submission; `0` builds the whole forward before
    /// the caller's `asyncEval`.
    public static let lastPositionTrunkSubmitEvery = 1
    /// Shorter forwards keep one submission.
    public static let lastPositionTrunkSubmitFromRows = 64

    public func cbv2LastPositionLogits(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        cbv2LastPositionLogits(
            inputs, cache: cache, submittingEvery: Self.lastPositionTrunkSubmitEvery)
    }

    /// `submittingEvery` layers per early trunk submission (`0`: none). The
    /// logits and K/V bytes do not depend on it.
    public func cbv2LastPositionLogits(
        _ inputs: MLXArray, cache: [KVCache]?, submittingEvery stride: Int
    ) -> MLXArray {
        guard let cache, cache.first is any CBv2AttendingLayerCache else {
            let hidden = model(model.inputEmbeddings(inputs), cache: cache)
            return outputLogits(hidden[0..., (hidden.dim(1) - 1)..., 0...])
        }
        var h = model.inputEmbeddings(inputs)
        let last = model.layers.count - 1
        var shared: (positions: [Int], offsets: MLXArray)?
        let submits = stride > 0 && h.dim(1) >= Self.lastPositionTrunkSubmitFromRows
        for (index, layer) in model.layers.enumerated() {
            var ropeOffsets: MLXArray?
            if layer.selfAttention.rope != nil,
                let attending = cache[index] as? any CBv2AttendingLayerCache
            {
                let positions = attending.rows.map(\.absoluteOffset)
                if let shared, shared.positions == positions {
                    ropeOffsets = shared.offsets
                } else {
                    let offsets = attending.positionOffsets + 0
                    shared = (positions, offsets)
                    ropeOffsets = offsets
                }
            }
            guard index == last else {
                h = layer(h, mask: .none, cache: cache[index], ropeOffsets: ropeOffsets)
                if submits, (index + 1) % stride == 0 { asyncEval(h) }
                continue
            }
            let attended = layer.postAttentionLayerNorm.addedTo(
                h,
                layer.selfAttention(
                    layer.inputLayerNorm(h), mask: .none, cache: cache[index],
                    ropeOffsets: ropeOffsets))
            let row = attended[0..., (attended.dim(1) - 1)..., 0...]
            h = layer.postFeedforwardLayerNorm.addedTo(
                row, layer.mlp(layer.preFeedforwardLayerNorm(row)))
        }
        return outputLogits(model.norm(h))
    }
}

// MARK: - Complete prefix checkpoints

// All persistent target state is attention K/V, every layer is causal, and no
// layer borrows another's storage.
extension MuseGlimmerTextModel: CBv2HistoricalAttentionCheckpointProviding {
    public var cbv2SupportsHistoricalAttentionCheckpoint: Bool { true }
}

extension MuseGlimmerTextModel: CBv2CompleteCheckpointKVTypeProviding {
    /// The dtype each layer stores K and V in. Both keep the activation
    /// dtype: the K/V projections return it, and the parameterless norms and
    /// RoPE do not change it. Pass this as the paged pool's `layerDTypes`.
    public var kvStorageDTypes: [DType] {
        Array(repeating: activationDType, count: configuration.hiddenLayers)
    }

    public var cbv2CompleteCheckpointKVDTypes: [DType]? { kvStorageDTypes }
}

// MARK: - Parameter residency

/// The target's parameters, keyed by the target: the runner builds a new
/// adapter for every engine around the same loaded tower.
extension MuseGlimmerCBv2Adapter: CBv2ResidentParametersProviding {
    public var residentParameterKey: ObjectIdentifier { ObjectIdentifier(target) }
    public func residentParameterArrays() -> [MLXArray] {
        target.parameters().flattened().map(\.1)
    }
}
