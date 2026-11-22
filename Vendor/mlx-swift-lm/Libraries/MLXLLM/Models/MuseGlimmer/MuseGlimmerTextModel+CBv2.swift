// Copyright © 2026 Eigen Labs.

import Foundation
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
        // Engine build: compile the fused verify kernels before any window.
        warmFusedVerifyKernels()
        return CBv2ModelCapabilities(
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
extension MuseGlimmerTextModel: CBv2LastPositionLogitsForwardable {
    /// Layers of the multi-row teacher-forced forward that take MLX's prompt
    /// arithmetic (one GEMM per projection over all rows) instead of exact
    /// windows (16-row pieces). Exact windows exist so a verify row gets the
    /// bytes it gets alone; this forward is a prompt, read at its last
    /// position only, and nothing decodes from its K/V. The span stays well
    /// inside the trunk so the prefill gain stays below the fixture's
    /// prefill speedup ceiling.
    static let promptArithmeticLayers = 16 ..< 50

    public func cbv2LastPositionLogits(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache, cache.first is any CBv2AttendingLayerCache else {
            let hidden = model(model.inputEmbeddings(inputs), cache: cache)
            return outputLogits(hidden[0..., (hidden.dim(1) - 1)..., 0...])
        }
        var h = model.inputEmbeddings(inputs)
        let last = model.layers.count - 1
        var shared: (positions: [Int], offsets: MLXArray)?
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
                if Self.promptArithmeticLayers.contains(index) {
                    // A prompt chunk's own arithmetic: the same MLX products
                    // the seed prefill (`prefillWithMTPHidden`) runs.
                    h = numerics.prompt {
                        layer(h, mask: .none, cache: cache[index], ropeOffsets: ropeOffsets)
                    }
                } else {
                    h = layer(h, mask: .none, cache: cache[index], ropeOffsets: ropeOffsets)
                }
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

// MARK: - Fused verify forward

/// The exact verify forward with each block-norm boundary in one dispatch:
/// `postAttention` add-norm with the following `preFeedforward` norm, and
/// `postFeedforward` add-norm with the next layer's `input` norm. Every other
/// operation is the tower's own. Bytes equal `forwardWithMTPHidden`; the
/// kernel reproduces MLX's row reduction below 32 rows
/// (`VerifyFusedNorms`). `DARKBLOOM_VERIFY_FUSED_NORMS=0` turns it off.
extension MuseGlimmerTextModel {
    func fusedVerifyForwardWithMTPHidden(_ tokens: MLXArray, cache: [KVCache])
        -> (logits: MLXArray, hidden: MLXArray)?
    {
        guard VerifyFusedNorms.enabled, !mtpHiddenLayerIDs.isEmpty,
            cache.count == model.layers.count,
            cache.allSatisfy({ $0 is any CBv2AttendingLayerCache })
        else { return nil }
        var h = model.inputEmbeddings(tokens)
        let rows = MuseGlimmerNumerics.rows(h)
        guard numerics.isExact(h),
            VerifyFusedNorms.supports(width: h.dim(-1), rows: rows, dtype: h.dtype)
        else { return nil }

        let layers = model.layers
        let capture = mtpHiddenLayerIDs
        var captured = [Int: MLXArray]()
        var shared: (positions: [Int], offsets: MLXArray)?
        var layerInput = layers[0].inputLayerNorm(h)
        for (index, layer) in layers.enumerated() {
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
            let attention = fusedVerifyAttention(
                layer.selfAttention, layerIndex: index, layerInput,
                cache: cache[index] as! any CBv2AttendingLayerCache, ropeOffsets: ropeOffsets)
            let post = layer.postAttentionLayerNorm
            let pre = layer.preFeedforwardLayerNorm
            let (mid, feedInput) = VerifyFusedNorms.fusedAddNormNorm(
                residual: h, x: attention, w1: post.weight, eps1: post.eps,
                w2: pre.weight, eps2: pre.eps)
            let feed = layer.mlp(feedInput)
            if index + 1 < layers.count {
                let postFF = layer.postFeedforwardLayerNorm
                let next = layers[index + 1].inputLayerNorm
                (h, layerInput) = VerifyFusedNorms.fusedAddNormNorm(
                    residual: mid, x: feed, w1: postFF.weight, eps1: postFF.eps,
                    w2: next.weight, eps2: next.eps)
            } else {
                h = layer.postFeedforwardLayerNorm.addedTo(mid, feed)
            }
            if capture.contains(index) { captured[index] = h }
        }
        let parts = capture.map { captured[$0]! }
        let normed = model.norm(h)
        return (outputLogits(normed), parts.count == 1 ? parts[0] : concatenated(parts, axis: -1))
    }

    /// `MuseGlimmerTextAttention`'s CBv2 branch with the q/k norms, the
    /// query scale and RoPE in one dispatch (`VerifyFusedQK`).
    func fusedVerifyAttention(
        _ a: MuseGlimmerTextAttention, layerIndex: Int, _ x: MLXArray,
        cache attending: any CBv2AttendingLayerCache, ropeOffsets: MLXArray?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let n = numerics
        guard VerifyFusedQK.enabled,
            VerifyFusedQK.supports(headDim: a.headDimensions, dtype: x.dtype)
        else {
            return a(x, mask: .none, cache: attending as? KVCache, ropeOffsets: ropeOffsets)
        }
        let theta = configuration.layerRopeTheta[layerIndex]
        let usesRope = a.rope != nil
        precondition(usesRope == (theta != 0))
        let (queries, keys) = VerifyFusedQK.apply(
            queries: n.project(a.queryProj, x), keys: n.project(a.keyProj, x),
            heads: a.heads, kvHeads: a.kvHeads, headDim: a.headDimensions,
            eps: a.qkNorm.eps, queryScale: a.qkScaleFactor,
            ropeBase: usesRope ? theta : nil,
            offsets: usesRope ? (ropeOffsets ?? (attending.positionOffsets + 0)) : nil)
        let values = n.project(a.valueProj, x)
            .reshaped(batch, length, a.kvHeads, a.headDimensions).transposed(0, 2, 1, 3)
        var output = attending.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: a.scale, sinks: nil)
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        output = MuseGlimmerFusion.sigmoidGate([output, n.project(a.gateProj, x)])[0]
        return n.project(a.outputProj, output)
    }
}

extension MuseGlimmerTextModel {
    nonisolated(unsafe) private static var fusedVerifyWarmed = Set<ObjectIdentifier>()
    private static let fusedVerifyWarmLock = NSLock()

    /// Runs each fused verify kernel once per set of template values (the
    /// q/k kernel's include the row count), on zeros, and drops the results.
    /// MLX compiles a custom kernel at its first dispatch; without this the
    /// first verify window of each width compiles while it is served. Once
    /// per model; changes no weight, cache or numerics state.
    func warmFusedVerifyKernels() {
        Self.fusedVerifyWarmLock.lock()
        let claimed = Self.fusedVerifyWarmed.insert(ObjectIdentifier(self)).inserted
        Self.fusedVerifyWarmLock.unlock()
        guard claimed, VerifyFusedNorms.enabled, let first = model.layers.first else { return }
        let dtype = activationDType
        let width = first.inputLayerNorm.weight.dim(-1)
        guard VerifyFusedNorms.supports(width: width, rows: 1, dtype: dtype) else { return }
        var warm = [MLXArray]()
        let zero = MLXArray.zeros([1, 1, width], dtype: dtype)
        let post = first.postAttentionLayerNorm
        let pre = first.preFeedforwardLayerNorm
        let norms = VerifyFusedNorms.fusedAddNormNorm(
            residual: zero, x: zero, w1: post.weight, eps1: post.eps,
            w2: pre.weight, eps2: pre.eps)
        warm += [norms.h, norms.y]
        let a = first.selfAttention
        if VerifyFusedQK.enabled, VerifyFusedQK.supports(headDim: a.headDimensions, dtype: dtype) {
            let thetas = Set(configuration.layerRopeTheta)
            let rope = thetas.first { $0 != 0 }
            var bases = [Float?]()
            if let rope { bases.append(rope) }
            if thetas.contains(0) { bases.append(nil) }
            for rows in 1 ... 8 {
                for base in bases {
                    // One request (shared offset) and several (per-request offsets).
                    for batch in [1, 2] {
                        let q = MLXArray.zeros([batch, rows, a.heads * a.headDimensions], dtype: dtype)
                        let k = MLXArray.zeros(
                            [batch, rows, a.kvHeads * a.headDimensions], dtype: dtype)
                        let out = VerifyFusedQK.apply(
                            queries: q, keys: k, heads: a.heads, kvHeads: a.kvHeads,
                            headDim: a.headDimensions, eps: a.qkNorm.eps,
                            queryScale: a.qkScaleFactor, ropeBase: base,
                            offsets: base == nil
                                ? nil : MLXArray.zeros([batch], type: Int32.self))
                        warm += [out.queries, out.keys]
                    }
                }
            }
        }
        eval(warm)
    }
}

extension MuseGlimmerCBv2Adapter: CBv2FusedVerifyForwarding {
    public func fusedVerifyForwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray)? {
        let kv = caches.compactMap { $0 as? KVCache }
        guard kv.count == caches.count,
            let result = target.fusedVerifyForwardWithMTPHidden(tokens, cache: kv)
        else { return nil }
        return (result.logits, result.hidden)
    }
}
