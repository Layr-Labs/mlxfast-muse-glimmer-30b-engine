// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

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
extension MuseGlimmerTextModel: CBv2LastPositionLogitsForwardable {
    /// Layers of the multi-row teacher-forced forward that take MLX's prompt
    /// arithmetic (one GEMM per projection over all rows), the arithmetic the
    /// engine's own prompt path uses, instead of exact windows (16-row pieces).
    /// Exact windows exist so a verify row gets the bytes it gets alone; this
    /// forward is a prompt chunk read at its last position.
    static let promptArithmeticLayers = 12 ..< 51

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

// MARK: - Grouped verify projections

/// The verify window (2 to 8 exact rows) with sibling projections that read one
/// activation run in one dispatch each: attention q, k, v and gate, and the
/// feed-forward gate and up. Every output element keeps its own kernel's
/// arithmetic (`CBv2GroupedProjection`), so the bytes equal the plain forward's.
/// Anything the grouping does not take (other formats, other row counts) runs
/// through `numerics.project` as before.
extension MuseGlimmerTextModel {
    private func projections(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray] {
        CBv2GroupedProjection.products(layers, x) ?? layers.map { numerics.project($0, x) }
    }

    private func verifyAttention(
        _ a: MuseGlimmerTextAttention, _ x: MLXArray, cache: KVCache?, ropeOffsets: MLXArray?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let n = numerics
        let p = projections([a.queryProj, a.keyProj, a.valueProj, a.gateProj], x)
        var queries = p[0].reshaped(batch, length, a.heads, a.headDimensions)
        var keys = p[1].reshaped(batch, length, a.kvHeads, a.headDimensions)
        var values = p[2].reshaped(batch, length, a.kvHeads, a.headDimensions)
        queries = a.qkNorm(queries)
        queries = (queries.asType(.float32) * a.qkScaleFactor)
            .asType(queries.dtype).transposed(0, 2, 1, 3)
        keys = a.qkNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        guard let attending = cache as? any CBv2AttendingLayerCache else {
            preconditionFailure("grouped verify needs CBv2 layer caches")
        }
        if let rope = a.rope {
            let offsets = ropeOffsets ?? (attending.positionOffsets + 0)
            queries = rope(queries, offset: offsets)
            keys = rope(keys, offset: offsets)
        }
        var output = attending.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: a.scale, sinks: nil)
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        output = MuseGlimmerFusion.sigmoidGate([output, p[3]])[0]
        return n.project(a.outputProj, output)
    }

    private func verifyLayer(
        _ layer: MuseGlimmerDecoderLayer, _ x: MLXArray, cache: KVCache?, ropeOffsets: MLXArray?
    ) -> MLXArray {
        let h = layer.postAttentionLayerNorm.addedTo(
            x,
            verifyAttention(
                layer.selfAttention, layer.inputLayerNorm(x), cache: cache,
                ropeOffsets: ropeOffsets))
        let m = layer.preFeedforwardLayerNorm(h)
        let gu = projections([layer.mlp.gateProj, layer.mlp.upProj], m)
        let mlp = numerics.project(layer.mlp.downProj, compiledSiluProduct(gu[0], gu[1]))
        return layer.postFeedforwardLayerNorm.addedTo(h, mlp)
    }

    // MARK: Fused norms (`CBv2FusedVerifyNorms`)

    /// `verifyAttention` with the q/k norm, the query scale and RoPE in one
    /// dispatch. Nil when the fused kernel does not take the shapes.
    private func fusedVerifyAttention(
        _ a: MuseGlimmerTextAttention, _ x: MLXArray, cache: KVCache?, ropeOffsets: MLXArray?,
        theta: Float
    ) -> MLXArray? {
        let (batch, length) = (x.dim(0), x.dim(1))
        guard let attending = cache as? any CBv2AttendingLayerCache else { return nil }
        let p = projections([a.queryProj, a.keyProj, a.valueProj, a.gateProj], x)
        var rope: (theta: Float, offsets: MLXArray)?
        if a.rope != nil {
            rope = (theta, ropeOffsets ?? (attending.positionOffsets + 0))
        }
        guard
            let (queries, keys) = CBv2FusedVerifyNorms.queryKeyRope(
                queries: p[0], keys: p[1], batch: batch, length: length, heads: a.heads,
                kvHeads: a.kvHeads, headDimensions: a.headDimensions, eps: a.qkNorm.eps,
                queryScale: a.qkScaleFactor, rope: rope)
        else { return nil }
        let values = p[2].reshaped(batch, length, a.kvHeads, a.headDimensions)
            .transposed(0, 2, 1, 3)
        var output = attending.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: a.scale, sinks: nil)
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        output = MuseGlimmerFusion.sigmoidGate([output, p[3]])[0]
        return numerics.project(a.outputProj, output)
    }

    /// Whether this model's verify windows take the fused norms: the switch is
    /// on, the shapes are the kernels', and the kernels reproduce the unfused
    /// operations bit for bit on this machine, on this model's own norm scales
    /// and attention settings (checked once, then remembered).
    private func fusedVerifyNormsReady() -> Bool {
        guard CBv2FusedVerifyNorms.enabled else { return false }
        let width = configuration.hiddenSize
        let dtype = activationDType
        guard CBv2FusedVerifyNorms.supportsNorm(dtype: dtype, width: width),
            CBv2FusedVerifyNorms.supportsAttention(
                dtype: dtype, headDimensions: configuration.headDimensions),
            !model.layers.isEmpty
        else { return false }
        let id = ObjectIdentifier(self)
        if let known = Self.fusedVerifyLock.withLock({ Self.fusedVerifyVerdicts[id] }) {
            return known
        }
        let verdict = fusedVerifyNormsMatchUnfused()
        Self.fusedVerifyLock.withLock { Self.fusedVerifyVerdicts[id] = verdict }
        if !verdict {
            FileHandle.standardError.write(
                Data(
                    "muse-glimmer: fused verify norms differ from the unfused operations on this machine; using the unfused path\n"
                        .utf8))
        }
        return verdict
    }

    private static let fusedVerifyLock = NSLock()
    nonisolated(unsafe) private static var fusedVerifyVerdicts = [ObjectIdentifier: Bool]()

    /// Runs each fused kernel and the operations it replaces on synthetic rows
    /// (2 to 8 rows, several magnitudes) with the model's real norm scales, eps
    /// and rotary settings, and compares every byte.
    private func fusedVerifyNormsMatchUnfused() -> Bool {
        let width = configuration.hiddenSize
        let dtype = activationDType
        let layers = model.layers
        func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && a.dtype == b.dtype && a.arrayEqual(b).item(Bool.self)
        }
        var seed: UInt64 = 1
        func random(_ shape: [Int], scale: Float) -> MLXArray {
            seed += 1
            return MLXRandom.normal(shape, dtype: dtype, scale: scale, key: MLXRandom.key(seed))
        }
        for (rows, scale) in [(2, Float(1)), (5, 6), (8, 40), (3, 0.01), (8, 1)] {
            for index in Set([0, layers.count / 2, layers.count - 1]).sorted() {
                let layer = layers[index]
                let x = random([1, rows, width], scale: scale)
                let attention = random([1, rows, width], scale: scale)
                // Input norm alone.
                let input = CBv2FusedVerifyNorms.addNorm(
                    x, residual: nil, weight: layer.inputLayerNorm.weight,
                    eps: layer.inputLayerNorm.eps, next: nil)
                // Post-attention add-norm into the pre-feed-forward norm.
                let h = layer.postAttentionLayerNorm.addedTo(x, attention)
                let m = layer.preFeedforwardLayerNorm(h)
                let boundary = CBv2FusedVerifyNorms.addNorm(
                    attention, residual: x, weight: layer.postAttentionLayerNorm.weight,
                    eps: layer.postAttentionLayerNorm.eps,
                    next: (layer.preFeedforwardLayerNorm.weight, layer.preFeedforwardLayerNorm.eps)
                )
                // Post-feed-forward add-norm, alone and into the next input norm.
                let after = layer.postFeedforwardLayerNorm.addedTo(x, attention)
                let nextLayer = layers[min(index + 1, layers.count - 1)]
                let chained = CBv2FusedVerifyNorms.addNorm(
                    attention, residual: x, weight: layer.postFeedforwardLayerNorm.weight,
                    eps: layer.postFeedforwardLayerNorm.eps,
                    next: (nextLayer.inputLayerNorm.weight, nextLayer.inputLayerNorm.eps))
                let ended = CBv2FusedVerifyNorms.addNorm(
                    attention, residual: x, weight: layer.postFeedforwardLayerNorm.weight,
                    eps: layer.postFeedforwardLayerNorm.eps, next: nil)
                guard let boundaryNormed = boundary.normed, let chainedNormed = chained.normed
                else { return false }
                let expectedChained = nextLayer.inputLayerNorm(after)
                let checks: [(MLXArray, MLXArray)] = [
                    (input.stream, layer.inputLayerNorm(x)), (boundary.stream, h),
                    (boundaryNormed, m), (chained.stream, after),
                    (chainedNormed, expectedChained), (ended.stream, after),
                ]
                if !checks.allSatisfy({ same($0.0, $0.1) }) { return false }
            }
        }
        // q/k norm, query scale and RoPE: one block per rotary setting.
        var seen = Set<Bool>()
        for (index, layer) in layers.enumerated() {
            let a = layer.selfAttention
            guard seen.insert(a.rope != nil).inserted else { continue }
            for (batch, length) in [(1, 2), (1, 7), (1, 8), (3, 1), (8, 1), (2, 2)] {
                let rows = batch * length
                for scale in [Float(1), 9] {
                    let q = random([batch, length, a.heads * a.headDimensions], scale: scale)
                    let k = random([batch, length, a.kvHeads * a.headDimensions], scale: scale)
                    let positions = (0 ..< batch).map { Int32(($0 * 977 + length * 131) % 4000) }
                    let offsets =
                        batch == 1 ? MLXArray([Int32(rows * 613 + 5)]) : MLXArray(positions)
                    var queries = q.reshaped(batch, length, a.heads, a.headDimensions)
                    var keys = k.reshaped(batch, length, a.kvHeads, a.headDimensions)
                    queries = a.qkNorm(queries)
                    queries = (queries.asType(.float32) * a.qkScaleFactor)
                        .asType(queries.dtype).transposed(0, 2, 1, 3)
                    keys = a.qkNorm(keys).transposed(0, 2, 1, 3)
                    if let rope = a.rope {
                        queries = rope(queries, offset: offsets)
                        keys = rope(keys, offset: offsets)
                    }
                    guard
                        let fused = CBv2FusedVerifyNorms.queryKeyRope(
                            queries: q, keys: k, batch: batch, length: length, heads: a.heads,
                            kvHeads: a.kvHeads, headDimensions: a.headDimensions,
                            eps: a.qkNorm.eps, queryScale: a.qkScaleFactor,
                            rope: a.rope.map { _ in
                                (configuration.layerRopeTheta[index], offsets)
                            })
                    else { return false }
                    if !(same(fused.queries, queries) && same(fused.keys, keys)) { return false }
                }
            }
        }
        return true
    }

    /// Warm-up state: the kernels of every distinct group shape, compiled once.
    private static let groupedWarmLock = NSLock()
    nonisolated(unsafe) private static var groupedWarmedModels = Set<ObjectIdentifier>()

    private func warmGroupedProjections() {
        let id = ObjectIdentifier(self)
        let first = Self.groupedWarmLock.withLock { Self.groupedWarmedModels.insert(id).inserted }
        guard first else { return }
        var seen = Set<String>()
        let dtype = activationDType
        for layer in model.layers {
            let a = layer.selfAttention
            for group in [
                [a.queryProj, a.keyProj, a.valueProj, a.gateProj],
                [layer.mlp.gateProj, layer.mlp.upProj],
            ] {
                if let key = CBv2GroupedProjection.shapeKey(group, dtype: dtype),
                    seen.insert(key).inserted
                {
                    CBv2GroupedProjection.warm(group, dtype: dtype, rows: 2 ... 8)
                }
            }
        }
    }

    /// `forwardWithMTPHidden` for a verify window; nil outside the exact
    /// 2...8-row CBv2 case.
    func verifyWindowWithMTPHidden(_ tokens: MLXArray, cache: [KVCache]?)
        -> (logits: MLXArray, hidden: MLXArray)?
    {
        guard CBv2GroupedProjection.enabled, let cache, cache.first is any CBv2AttendingLayerCache,
            !mtpHiddenLayerIDs.isEmpty
        else { return nil }
        var h = model.inputEmbeddings(tokens)
        let rows = MuseGlimmerNumerics.rows(h)
        guard numerics.isExact(h), (2 ... CBv2GroupedProjection.maximumRows).contains(rows)
        else { return nil }
        warmGroupedProjections()
        let fused = fusedVerifyNormsReady()
        let layers = model.layers
        // Fused: each block boundary is one dispatch for the residual add, the
        // block's post norm and the next block's input norm.
        var blockInput: MLXArray?
        if fused {
            let first = layers[0].inputLayerNorm
            blockInput = CBv2FusedVerifyNorms.addNorm(
                h, residual: nil, weight: first.weight, eps: first.eps, next: nil
            ).stream
        }
        var captured = [Int: MLXArray]()
        var shared: (positions: [Int], offsets: MLXArray)?
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
            if fused, let input = blockInput,
                let attention = fusedVerifyAttention(
                    layer.selfAttention, input, cache: cache[index], ropeOffsets: ropeOffsets,
                    theta: configuration.layerRopeTheta[index])
            {
                let afterAttention = layer.postAttentionLayerNorm
                let beforeFeedforward = layer.preFeedforwardLayerNorm
                let boundary = CBv2FusedVerifyNorms.addNorm(
                    attention, residual: h, weight: afterAttention.weight, eps: afterAttention.eps,
                    next: (beforeFeedforward.weight, beforeFeedforward.eps))
                let gu = projections([layer.mlp.gateProj, layer.mlp.upProj], boundary.normed!)
                let mlp = numerics.project(layer.mlp.downProj, compiledSiluProduct(gu[0], gu[1]))
                let afterFeedforward = layer.postFeedforwardLayerNorm
                var next: (weight: MLXArray, eps: Float)?
                if index + 1 < layers.count {
                    let following = layers[index + 1].inputLayerNorm
                    next = (following.weight, following.eps)
                }
                let closing = CBv2FusedVerifyNorms.addNorm(
                    mlp, residual: boundary.stream, weight: afterFeedforward.weight,
                    eps: afterFeedforward.eps, next: next)
                h = closing.stream
                blockInput = closing.normed
            } else {
                // Unfused layer; later layers follow it unfused.
                blockInput = nil
                h = verifyLayer(layer, h, cache: cache[index], ropeOffsets: ropeOffsets)
            }
            if mtpHiddenLayerIDs.contains(index) { captured[index] = h }
        }
        let normed = model.norm(h)
        let hiddens = mtpHiddenLayerIDs.map { captured[$0]! }
        let hidden = hiddens.count == 1 ? hiddens[0] : concatenated(hiddens, axis: -1)
        return (outputLogits(normed), hidden)
    }
}

extension MuseGlimmerCBv2Adapter: CBv2VerifyWindowForwarding {
    public func forwardVerifyWindow(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)?
    {
        let kv = caches.compactMap { $0 as? KVCache }
        guard kv.count == caches.count,
            let result = target.verifyWindowWithMTPHidden(tokens, cache: kv)
        else { return nil }
        return (result.logits, result.hidden)
    }
}
