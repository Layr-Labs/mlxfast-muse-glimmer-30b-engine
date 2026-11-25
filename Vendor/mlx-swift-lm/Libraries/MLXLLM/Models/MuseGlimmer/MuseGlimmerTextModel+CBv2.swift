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
        // Engine build: compile the fused verify kernels before any window.
        warmGroupedProjections()
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
    public var prefillCommandBufferLimits: GPU.CommandBufferLimits? {
        GPU.CommandBufferLimits(operations: 60, megabytes: 1 << 20)
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
            let embeddings = inputEmbedding ?? model.inputEmbeddings(inputs)
            if let fused = fusedPrefillResidual(embeddings, cache: cache, capture: []) {
                switch requirement {
                case .evaluationOnly:
                    return fused.hidden[0..., -1, 0 ..< 1]
                case .lastPositionLogits:
                    return outputLogits(model.norm(fused.hidden[0..., -1, 0...]))
                }
            }
            let hidden = model(embeddings, cache: cache)
            switch requirement {
            case .evaluationOnly: return hidden[0..., -1, 0 ..< 1]
            case .lastPositionLogits: return outputLogits(hidden[0..., -1, 0...])
            }
        }
    }
    // Packed prefill keeps the protocol's fail-closed default.

    /// Fused prefill trunk for both plain prefill and MTP hidden-capture
    /// prefill: keeps every projection, norm, attention and MLP operation in
    /// `numerics.prompt` while fusing Q/K norm + FP32 query scale + RoPE into
    /// one head-major kernel (`VerifyFusedQK.applyHeadMajor`, 0-ULP verified
    /// with stock fallback) and applying the attention output gate before
    /// merging heads so no `[B, L, H, D]` transpose copy is materialized.
    private func fusedPrefillResidual(
        _ embeddings: MLXArray, cache: [KVCache]?, capture: [Int]
    ) -> (hidden: MLXArray, captured: [MLXArray])? {
        guard let cache, cache.count == model.layers.count,
            cache.allSatisfy({ $0 is any CBv2AttendingLayerCache })
        else { return nil }
        var h = embeddings
        var captured = [Int: MLXArray]()
        var shared: (positions: [Int], offsets: MLXArray)?
        for (index, layer) in model.layers.enumerated() {
            let attending = cache[index] as! any CBv2AttendingLayerCache
            var ropeOffsets: MLXArray?
            if layer.selfAttention.rope != nil {
                let positions = attending.rows.map(\.absoluteOffset)
                if let shared, shared.positions == positions {
                    ropeOffsets = shared.offsets
                } else {
                    let offsets = attending.positionOffsets + 0
                    shared = (positions, offsets)
                    ropeOffsets = offsets
                }
            }
            let attn = fusedPrefillAttention(
                layer.selfAttention, layerIndex: index, layer.inputLayerNorm(h),
                cache: attending, ropeOffsets: ropeOffsets)
            let mid = layer.postAttentionLayerNorm.addedTo(h, attn)
            h = layer.postFeedforwardLayerNorm.addedTo(
                mid, layer.mlp(layer.preFeedforwardLayerNorm(mid)))
            if capture.contains(index) { captured[index] = h }
        }
        return (h, capture.map { captured[$0]! })
    }

    private func fusedPrefillAttention(
        _ a: MuseGlimmerTextAttention, layerIndex: Int, _ x: MLXArray,
        cache attending: any CBv2AttendingLayerCache, ropeOffsets: MLXArray?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let n = numerics
        let qRaw = n.project(a.queryProj, x)
        let kRaw = n.project(a.keyProj, x)
        let vRaw = n.project(a.valueProj, x)
        let gRaw = n.project(a.gateProj, x)
        let queries: MLXArray
        let keys: MLXArray
        let theta = configuration.layerRopeTheta[layerIndex]
        let usesRope = a.rope != nil
        if VerifyFusedQK.prefillEnabled,
            VerifyFusedQK.supports(headDim: a.headDimensions, dtype: x.dtype)
        {
            precondition(usesRope == (theta != 0))
            let qk = VerifyFusedQK.applyHeadMajor(
                queries: qRaw, keys: kRaw,
                heads: a.heads, kvHeads: a.kvHeads, headDim: a.headDimensions,
                eps: a.qkNorm.eps, queryScale: a.qkScaleFactor,
                ropeBase: usesRope ? theta : nil,
                offsets: usesRope ? (ropeOffsets ?? (attending.positionOffsets + 0)) : nil)
            queries = qk.queries
            keys = qk.keys
        } else {
            var q = a.qkNorm(qRaw.reshaped(batch, length, a.heads, a.headDimensions))
            q = (q.asType(.float32) * a.qkScaleFactor).asType(q.dtype).transposed(0, 2, 1, 3)
            var k = a.qkNorm(kRaw.reshaped(batch, length, a.kvHeads, a.headDimensions))
                .transposed(0, 2, 1, 3)
            if let rope = a.rope {
                let offsets = ropeOffsets ?? (attending.positionOffsets + 0)
                q = rope(q, offset: offsets)
                k = rope(k, offset: offsets)
            }
            queries = q
            keys = k
        }
        let values = vRaw
            .reshaped(batch, length, a.kvHeads, a.headDimensions).transposed(0, 2, 1, 3)
        var output = attending.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: a.scale, sinks: nil)
        let gate = gRaw.reshaped(batch, length, a.heads, a.headDimensions)
        output = MuseGlimmerFusion.sigmoidGate([output.transposed(0, 2, 1, 3), gate])[0]
            .reshaped(batch, length, -1)
        return n.project(a.outputProj, output)
    }

    func fusedPrefillWithMTPHidden(
        _ tokens: MLXArray, cache: [KVCache]?, requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, hidden: MLXArray)? {
        guard !mtpHiddenLayerIDs.isEmpty else { return nil }
        return numerics.prompt {
            guard
                let fused = fusedPrefillResidual(
                    model.inputEmbeddings(tokens), cache: cache, capture: mtpHiddenLayerIDs)
            else { return nil }
            let captured =
                fused.captured.count == 1
                ? fused.captured[0] : concatenated(fused.captured, axis: -1)
            switch requirement {
            case .evaluationOnly:
                return (fused.hidden[0..., -1, 0 ..< 1], captured)
            case .lastPositionLogits:
                return (outputLogits(model.norm(fused.hidden[0..., -1, 0...])), captured)
            }
        }
    }
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
    static let promptArithmeticLayers = 16 ..< 50

    public func cbv2LastPositionLogits(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache, cache.first is any CBv2AttendingLayerCache else {
            let hidden = model(model.inputEmbeddings(inputs), cache: cache)
            return outputLogits(hidden[0..., (hidden.dim(1) - 1)..., 0...])
        }
        // Each finished trunk layer is submitted while the host builds the next,
        // under the decode command-buffer limits. Values do not depend on either.
        return GPU.withCommandBufferLimits(MuseGlimmerDecodeCommandBuffers.limits) {
            submittedLastPositionLogits(inputs, cache: cache)
        }
    }

    private func submittedLastPositionLogits(_ inputs: MLXArray, cache: [KVCache]) -> MLXArray {
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
                asyncEval([h])
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
    private func fusedVerifyTrunk(_ tokens: MLXArray, cache: [KVCache])
        -> (normed: MLXArray, hidden: MLXArray)?
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
            let feed = groupedVerifyFeed(layer.mlp, feedInput)
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
        return (normed, parts.count == 1 ? parts[0] : concatenated(parts, axis: -1))
    }
    func fusedVerifyForwardWithMTPHidden(_ tokens: MLXArray, cache: [KVCache])
        -> (logits: MLXArray, hidden: MLXArray)?
    {
        guard let trunk = fusedVerifyTrunk(tokens, cache: cache) else { return nil }
        return (verifyOutputLogits(trunk.normed), trunk.hidden)
    }

    func fusedVerifyForwardScoredWithMTPHidden(_ tokens: MLXArray, cache: [KVCache])
        -> (
            topTwoIDs: MLXArray, topTwoValues: MLXArray, nonFinite: MLXArray, hidden: MLXArray
        )?
    {
        guard lmHead is ModelOptWeightOnlyNVFP4Linear,
            let trunk = fusedVerifyTrunk(tokens, cache: cache),
            verifyRows(trunk.normed) != nil,
            let product = CBv2VerifyScales.product(lmHead, trunk.normed),
            product.ndim == 3, product.dtype == .bfloat16
        else { return nil }
        checkRouted([product], [lmHead], trunk.normed, kind: "nvfp4.lm_head")
        let scored = cbv2FusedReadoutTopTwoAndNonFinite(
            product,
            outputMultiplier: configuration.outputMultiplier,
            softcap: configuration.finalLogitSoftcapping)
        return (scored.ids, scored.values, scored.nonFinite, trunk.hidden)
    }
    /// `MuseGlimmerTextAttention`'s CBv2 branch with the q/k norms, the
    /// query scale and RoPE in one dispatch (`VerifyFusedQK`).
    func fusedVerifyAttention(
        _ a: MuseGlimmerTextAttention, layerIndex: Int, _ x: MLXArray,
        cache attending: any CBv2AttendingLayerCache, ropeOffsets: MLXArray?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        guard VerifyFusedQK.enabled,
            VerifyFusedQK.supports(headDim: a.headDimensions, dtype: x.dtype)
        else {
            return a(x, mask: .none, cache: attending as? KVCache, ropeOffsets: ropeOffsets)
        }
        let theta = configuration.layerRopeTheta[layerIndex]
        let usesRope = a.rope != nil
        precondition(usesRope == (theta != 0))
        let p = verifyAttentionProjections([a.queryProj, a.keyProj, a.valueProj, a.gateProj], x)
        let (queries, keys) = VerifyFusedQK.apply(
            queries: p[0], keys: p[1],
            heads: a.heads, kvHeads: a.kvHeads, headDim: a.headDimensions,
            eps: a.qkNorm.eps, queryScale: a.qkScaleFactor,
            ropeBase: usesRope ? theta : nil,
            offsets: usesRope ? (ropeOffsets ?? (attending.positionOffsets + 0)) : nil)
        let values = p[2]
            .reshaped(batch, length, a.kvHeads, a.headDimensions).transposed(0, 2, 1, 3)
        var output = attending.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: a.scale, sinks: nil)
        // Gate the `[B, L, H, D]` view of the attention output directly: the
        // elementwise kernel reads it strided and writes contiguous rows, so
        // the merge to `[B, L, H * D]` is free. Reshaping the transposed view
        // first materializes a copy per layer (one more dispatch and a full
        // read and write of the output). Same elements, same expression.
        let gate = p[3].reshaped(batch, length, a.heads, a.headDimensions)
        output = MuseGlimmerFusion.sigmoidGate([output.transposed(0, 2, 1, 3), gate])[0]
            .reshaped(batch, length, -1)
        return verifyOutputProjection(a.outputProj, output)
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
        // One-pass paged verify attention: one variant per binding class.
        PagedOnePassAttention.warm(
            dtype: dtype, queryHeads: a.heads, kvHeads: a.kvHeads, headDim: a.headDimensions,
            pageSize: 16, softcap: false)
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
            if VerifyFusedQK.verifyPrefillParity(
                heads: a.heads, kvHeads: a.kvHeads, headDim: a.headDimensions,
                eps: a.qkNorm.eps, queryScale: a.qkScaleFactor, ropeBase: rope, dtype: dtype)
            {
                let prefillLen = 512
                let q512 = MLXArray.zeros(
                    [1, prefillLen, a.heads * a.headDimensions], dtype: dtype)
                let k512 = MLXArray.zeros(
                    [1, prefillLen, a.kvHeads * a.headDimensions], dtype: dtype)
                for base in bases {
                    let out = VerifyFusedQK.applyHeadMajor(
                        queries: q512, keys: k512, heads: a.heads, kvHeads: a.kvHeads,
                        headDim: a.headDimensions, eps: a.qkNorm.eps,
                        queryScale: a.qkScaleFactor, ropeBase: base,
                        offsets: base == nil ? nil : MLXArray.zeros([1], type: Int32.self))
                    warm += [out.queries, out.keys]
                }
            }
        }
        eval(warm)
        warmVerifyProjections(dtype: dtype)
    }
}

// MARK: - Grouped verify projections

/// Sibling projections of one activation (attention q, k, v and gate; the
/// feed-forward gate and up) in one dispatch each for 2 to 8 exact rows
/// (`CBv2GroupedProjection`): every output element keeps its own kernel's
/// arithmetic, so the bytes equal the separate row-exact products.
/// `DARKBLOOM_MUSE_GLIMMER_GROUPED_VERIFY=0` turns it off; any other shape
/// runs `numerics.project` projection by projection.
extension MuseGlimmerTextModel {
    func groupedProjections(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray] {
        if numerics.isExact(x), let grouped = CBv2GroupedProjection.products(layers, x) {
            return grouped
        }
        return layers.map { numerics.project($0, x) }
    }

    func groupedVerifyFeed(_ mlp: MuseGlimmerTextMLP, _ x: MLXArray) -> MLXArray {
        if MuseGlimmerVerifyProjections.fusedSilu, verifyRows(x) != nil,
            let fused = CBv2VerifyNVFP4.verifiedSiluProduct([mlp.gateProj, mlp.upProj], x)
        {
            if MuseGlimmerVerifyProjections.selfTest {
                let stock = compiledSiluProduct(
                    numerics.project(mlp.gateProj, x), numerics.project(mlp.upProj, x))
                MuseGlimmerVerifyProjections.requireEqualBits(
                    [fused], [stock], kind: "nvfp4.gate_up_silu", rows: MuseGlimmerNumerics.rows(x))
            }
            return verifyDownProjection(mlp.downProj, fused)
        }
        let gu = verifyFeedInputProjections([mlp.gateProj, mlp.upProj], x)
        return verifyDownProjection(mlp.downProj, compiledSiluProduct(gu[0], gu[1]))
    }

    nonisolated(unsafe) private static var groupedWarmed = Set<ObjectIdentifier>()
    private static let groupedWarmLock = NSLock()

    /// Compiles the grouped kernels of every distinct shape group at 2 to 8
    /// rows, on zeros, once per model; a no-op until exact windows are on.
    func warmGroupedProjections() {
        guard CBv2GroupedProjection.enabled, numerics.exactWindows else { return }
        Self.groupedWarmLock.lock()
        let claimed = Self.groupedWarmed.insert(ObjectIdentifier(self)).inserted
        Self.groupedWarmLock.unlock()
        guard claimed else { return }
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

extension MuseGlimmerCBv2Adapter: CBv2FusedMTPPrefillForwarding {
    public func fusedForwardWithHiddenForPrefill(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray)? {
        let kv = caches.compactMap { $0 as? KVCache }
        guard kv.count == caches.count,
            let result = target.fusedPrefillWithMTPHidden(
                tokens, cache: kv, requirement: requirement)
        else { return nil }
        return (result.logits, result.hidden)
    }
}

extension MuseGlimmerCBv2Adapter: CBv2FusedVerifyReadoutForwarding {
    public func fusedVerifyForwardScoredWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (
        topTwoIDs: MLXArray, topTwoValues: MLXArray, nonFinite: MLXArray, lastHidden: MLXArray
    )? {
        let kv = caches.compactMap { $0 as? KVCache }
        guard kv.count == caches.count,
            let result = target.fusedVerifyForwardScoredWithMTPHidden(tokens, cache: kv)
        else { return nil }
        return (result.topTwoIDs, result.topTwoValues, result.nonFinite, result.hidden)
    }
}

// MARK: - Verify projection kernels

/// Exact verify rows (1 to 8 per call) of the trunk projections on finer
/// schedules of the row-exact products. Every output element keeps the stock
/// kernel's arithmetic: the same products, the same lanes summing them in the
/// same order, the same shuffle tree and partial joins, the same rounding
/// points. Only which threads compute which outputs, how many outputs a
/// threadgroup covers and how weights, scales and activations are loaded
/// change:
/// - NVFP4 MLP gate and up: one launch, 8 output rows per threadgroup
///   (`CBv2VerifyNVFP4`); down: the same, 16-row tiles with a one-chunk
///   register prefetch at 7 and 8 rows (`CBv2VerifyNVFP4`), whole-chunk loads
///   below (`CBv2VerifyScales`).
/// - NVFP4 attention q/k/v/gate and o: 8-row tiles with whole-chunk loads
///   (`CBv2VerifyScales`).
/// - FP8 attention q/k/v/gate and MLP gate/up (one cast and one product
///   launch), o and MLP down up to 5 rows (one launch for all eager tiles):
///   vector loads, prefetched weight words and a packed, order-preserving
///   shuffle tree (`CBv2SmallProjection`).
/// - FP8 MLP down at 6 to 8 rows: vector loads and four row groups per
///   threadgroup (`CBv2VerifyScales`).
/// - NVFP4 lm_head: 8-row tiles (`CBv2VerifyScales`), then the stock
///   multiplier and softcap.
/// Anything else (dense layers, more than 8 rows) keeps the frontier path.
/// `DARKBLOOM_MUSE_GLIMMER_VERIFY_PROJ=0` turns the routing off.
/// `DARKBLOOM_MUSE_GLIMMER_VERIFY_PROJ_SELFTEST=1` also computes every
/// routed projection with the stock row-exact product, on the same
/// activations, and stops the process unless every output's bits are equal.
enum MuseGlimmerVerifyProjections {
    static let environment = ProcessInfo.processInfo.environment
    static let enabled = environment["DARKBLOOM_MUSE_GLIMMER_VERIFY_PROJ"] != "0"
    static let selfTest = environment["DARKBLOOM_MUSE_GLIMMER_VERIFY_PROJ_SELFTEST"] == "1"
    /// Single-pass NVFP4 gate/up + SiLU product with one-time runtime 0-ULP
    /// parity verification and automatic stock fallback on any mismatch.
    static let fusedSilu = environment["DARKBLOOM_MUSE_GLIMMER_VERIFY_SILU"] != "0"
    static let maximumRows = 8

    private static let lock = NSLock()
    nonisolated(unsafe) private static var checkedOutputs = 0
    nonisolated(unsafe) private static var checkedElements = 0
    nonisolated(unsafe) private static var checkedByKind = [String: Int]()

    /// Self-test: `routed` and `stock` must match in shape, dtype and every
    /// element's bits (NaN payloads included).
    static func requireEqualBits(
        _ routed: [MLXArray], _ stock: [MLXArray], kind: String, rows: Int
    ) {
        precondition(routed.count == stock.count, "verify-projection self-test: \(kind) count")
        var elements = 0
        for (index, (a, b)) in zip(routed, stock).enumerated() {
            precondition(
                a.shape == b.shape && a.dtype == b.dtype,
                "verify-projection self-test: \(kind) rows \(rows) output \(index): "
                    + "\(a.shape) \(a.dtype) vs \(b.shape) \(b.dtype)")
            let bits: DType = a.dtype.size == 4 ? .uint32 : .uint16
            let differing = (a.view(dtype: bits) .!= b.view(dtype: bits)).asType(.int32).sum()
                .item(Int.self)
            precondition(
                differing == 0,
                "verify-projection self-test: \(kind) rows \(rows) output \(index): "
                    + "\(differing) of \(a.size) elements differ")
            elements += a.size
        }
        let report: String? = lock.withLock {
            checkedOutputs += routed.count
            checkedElements += elements
            let key = "\(kind)@\(rows)"
            let first = checkedByKind[key] == nil
            checkedByKind[key, default: 0] += routed.count
            guard first || checkedOutputs % 2000 < routed.count else { return nil }
            let kinds = checkedByKind.keys.sorted().map { "\($0)=\(checkedByKind[$0]!)" }
            return "[verify-proj-selftest] outputs=\(checkedOutputs) elements=\(checkedElements) "
                + "differing=0 \(kinds.joined(separator: " "))\n"
        }
        if let report { FileHandle.standardError.write(Data(report.utf8)) }
    }
}

extension MuseGlimmerTextModel {
    private var routesVerifyProjections: Bool { MuseGlimmerVerifyProjections.enabled }

    private func verifyRows(_ x: MLXArray) -> Int? {
        guard routesVerifyProjections, numerics.isExact(x) else { return nil }
        let rows = MuseGlimmerNumerics.rows(x)
        return (1 ... MuseGlimmerVerifyProjections.maximumRows).contains(rows) ? rows : nil
    }

    /// The self-test's reference: each layer's own row-exact product.
    private func checkRouted(
        _ routed: [MLXArray], _ layers: [any UnaryLayer], _ x: MLXArray, kind: String
    ) {
        guard MuseGlimmerVerifyProjections.selfTest else { return }
        let stock = layers.map { numerics.project($0, x) }
        MuseGlimmerVerifyProjections.requireEqualBits(
            routed, stock, kind: kind, rows: MuseGlimmerNumerics.rows(x))
    }

    /// Attention q, k, v and gate of one layer: all FP8 through
    /// `CBv2SmallProjection`, all NVFP4 through `CBv2VerifyScales`; otherwise
    /// the frontier's grouped or per-layer products.
    func verifyAttentionProjections(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray] {
        if verifyRows(x) != nil {
            var routed: [MLXArray]?
            var kind = ""
            if let fp8 = layers as? [ModelOptCalibratedLinear] {
                routed = CBv2SmallProjection.fp8Products(fp8, x)
                kind = "fp8.qkvg"
            } else if layers.allSatisfy({ $0 is ModelOptWeightOnlyNVFP4Linear }) {
                routed = CBv2VerifyScales.products(layers, x)
                kind = "nvfp4.qkvg"
            }
            if let routed {
                checkRouted(routed, layers, x, kind: kind)
                return routed
            }
        }
        return groupedProjections(layers, x)
    }

    /// The attention output projection: FP8 through `CBv2SmallProjection`
    /// (all eager tiles in one launch), NVFP4 through `CBv2VerifyScales`.
    func verifyOutputProjection(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray {
        if verifyRows(x) != nil {
            var routed: MLXArray?
            var kind = ""
            if let fp8 = layer as? ModelOptCalibratedLinear {
                routed = CBv2SmallProjection.fp8Products([fp8], x)?.first
                kind = "fp8.o"
            } else if layer is ModelOptWeightOnlyNVFP4Linear {
                routed = CBv2VerifyScales.product(layer, x)
                kind = "nvfp4.o"
            }
            if let routed {
                checkRouted([routed], [layer], x, kind: kind)
                return routed
            }
        }
        return numerics.project(layer, x)
    }

    /// The feed-forward gate and up: an NVFP4 pair in one `CBv2VerifyNVFP4`
    /// launch; an FP8 pair through `CBv2SmallProjection` (one cast and one
    /// product launch).
    func verifyFeedInputProjections(_ layers: [any UnaryLayer], _ x: MLXArray) -> [MLXArray] {
        if verifyRows(x) != nil {
            var routed: [MLXArray]?
            var kind = ""
            if let pair = CBv2VerifyNVFP4.products(layers, x) {
                routed = pair
                kind = "nvfp4.gate_up"
            } else if let fp8 = layers as? [ModelOptCalibratedLinear] {
                routed = CBv2SmallProjection.fp8Products(fp8, x)
                kind = "fp8.gate_up"
            }
            if let routed {
                checkRouted(routed, layers, x, kind: kind)
                return routed
            }
        }
        return groupedProjections(layers, x)
    }

    /// The feed-forward down projection: NVFP4 through `CBv2VerifyNVFP4` at 7
    /// and 8 rows (16-row tiles with prefetch) and `CBv2VerifyScales` below;
    /// FP8 through `CBv2SmallProjection` up to 5 rows and `CBv2VerifyScales`
    /// at 6 to 8 rows (four row groups).
    func verifyDownProjection(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray {
        if let rows = verifyRows(x) {
            var routed: MLXArray?
            var kind = ""
            if let nv = layer as? ModelOptWeightOnlyNVFP4Linear {
                routed =
                    rows >= 7
                    ? CBv2VerifyNVFP4.product(nv, x) : CBv2VerifyScales.product(layer, x)
                kind = "nvfp4.down"
            } else if let fp8 = layer as? ModelOptCalibratedLinear {
                routed =
                    rows >= 6
                    ? CBv2VerifyScales.product(layer, x)
                    : CBv2SmallProjection.fp8Products([fp8], x)?.first
                kind = "fp8.down"
            }
            if let routed {
                checkRouted([routed], [layer], x, kind: kind)
                return routed
            }
        }
        return numerics.project(layer, x)
    }

    private static let readoutKernel = MLXFast.metalKernel(
        name: "muse_glimmer_verify_logit_readout",
        inputNames: ["x", "params"],
        outputNames: ["y"],
        source: """
            const uint idx = thread_position_in_grid.x;
            const uint n4 = uint(N4_);
            if (idx >= n4) { return; }
            const bfloat16_t mult = bfloat16_t(params[0]);
            const bfloat16_t cap = bfloat16_t(params[1]);
            const device vec<bfloat16_t, 4>* src = (const device vec<bfloat16_t, 4>*)x;
            device vec<bfloat16_t, 4>* dst = (device vec<bfloat16_t, 4>*)y;
            const vec<bfloat16_t, 4> v = src[idx];
            vec<bfloat16_t, 4> o;
            #pragma unroll
            for (short i = 0; i < 4; i++) {
              const bfloat16_t s = v[i] * mult;
              const bfloat16_t q = s / cap;
              const bfloat16_t t = bfloat16_t(metal::precise::tanh(float(q)));
              o[i] = t * cap;
            }
            dst[idx] = o;
            """)

    /// `outputLogits` for verify rows: an NVFP4 `lm_head` through
    /// `CBv2VerifyScales` (8-row tiles), then the same multiplier and softcap
    /// fused into one vectorized pass. Other calls run `outputLogits` itself.
    func verifyOutputLogits(_ normedHidden: MLXArray) -> MLXArray {
        guard verifyRows(normedHidden) != nil, lmHead is ModelOptWeightOnlyNVFP4Linear,
            let product = CBv2VerifyScales.product(lmHead, normedHidden)
        else { return outputLogits(normedHidden) }
        checkRouted([product], [lmHead], normedHidden, kind: "nvfp4.lm_head")
        let cap = configuration.finalLogitSoftcapping
        if cap > 0, product.dtype == .bfloat16, product.size % 4 == 0 {
            let n4 = product.size / 4
            let tg = 256
            let gridX = ((n4 + tg - 1) / tg) * tg
            let params = MLXArray([configuration.outputMultiplier, cap])
            return Self.readoutKernel(
                [product, params],
                template: [("N4_", n4)],
                grid: (gridX, 1, 1), threadGroup: (tg, 1, 1),
                outputShapes: [product.shape], outputDTypes: [product.dtype])[0]
        }
        var logits = product * configuration.outputMultiplier
        if cap > 0 { logits = tanh(logits / cap) * cap }
        return logits
    }

    /// Compiles every routed kernel (each distinct projection shape at 1 to
    /// 8 rows) on zeros, so no verify window compiles one while it is served.
    /// Called from `warmFusedVerifyKernels` (once per model); changes no
    /// weight, cache or numerics state.
    func warmVerifyProjections(dtype: DType) {
        guard routesVerifyProjections, numerics.exactWindows else { return }
        func key(_ layers: [any UnaryLayer]) -> String {
            layers.map { layer -> String in
                guard let shape = (layer as? any MuseGlimmerRowExactProjection)?.rowExactShape
                else { return "other" }
                return "\(shape.format):\(shape.inputs)x\(shape.outputs)"
            }.joined(separator: ",")
        }
        func inputs(_ layer: any UnaryLayer) -> Int? {
            (layer as? any MuseGlimmerRowExactProjection)?.rowExactShape.inputs
        }
        var seen = Set<String>()
        for layer in model.layers {
            let a = layer.selfAttention
            let mlp = layer.mlp
            let sites: [(String, [any UnaryLayer])] = [
                ("qkvg", [a.queryProj, a.keyProj, a.valueProj, a.gateProj]),
                ("o", [a.outputProj]),
                ("gate_up", [mlp.gateProj, mlp.upProj]),
                ("down", [mlp.downProj]),
            ]
            for (site, group) in sites {
                guard seen.insert(site + " " + key(group)).inserted,
                    let width = inputs(group[0])
                else { continue }
                var outputs = [MLXArray]()
                for rows in 1 ... MuseGlimmerVerifyProjections.maximumRows {
                    let x = MLXArray.zeros([1, rows, width], dtype: dtype)
                    switch site {
                    case "qkvg": outputs += verifyAttentionProjections(group, x)
                    case "o": outputs.append(verifyOutputProjection(group[0], x))
                    case "gate_up": outputs += verifyFeedInputProjections(group, x)
                    default: outputs.append(verifyDownProjection(group[0], x))
                    }
                }
                eval(outputs)
                if site == "gate_up", MuseGlimmerVerifyProjections.fusedSilu {
                    CBv2VerifyNVFP4.verifyAndWarmSilu(gateUp: group, dtype: dtype)
                }
            }
        }
        if lmHead is ModelOptWeightOnlyNVFP4Linear, let width = inputs(lmHead) {
            eval(
                (1 ... MuseGlimmerVerifyProjections.maximumRows).map {
                    verifyOutputLogits(MLXArray.zeros([1, $0, width], dtype: dtype))
                })
            let fusedWarm = cbv2FusedReadoutTopTwoAndNonFinite(
                MLXArray.zeros([1, 1, configuration.vocabularySize], dtype: .bfloat16),
                outputMultiplier: configuration.outputMultiplier,
                softcap: configuration.finalLogitSoftcapping)
            eval(fusedWarm.ids, fusedWarm.values, fusedWarm.nonFinite)
        }
    }
}
