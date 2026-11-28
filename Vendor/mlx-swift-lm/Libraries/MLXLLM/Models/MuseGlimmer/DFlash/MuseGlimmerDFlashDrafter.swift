// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import os

/// What the drafter reads from its target. The drafter has no embedding
/// table and no output head of its own.
public protocol MuseGlimmerDFlashTarget: AnyObject {
    /// The identity the engine's model reports as `mtpTargetIdentity`.
    var dflashTargetIdentity: ObjectIdentifier { get }
    var dflashHiddenSize: Int { get }
    var dflashVocabularySize: Int { get }
    /// Width of the hidden rows the target hands to the MTP seam: the
    /// residual stream after each tapped layer, concatenated.
    var dflashContextWidth: Int { get }
    /// The layers whose residual streams make those rows, in order. The
    /// drafter was trained on one set and refuses any other.
    var dflashTappedLayerIDs: [Int] { get }
    /// Rows of the token embedding table WITHOUT the target's embedding
    /// norm: `[..., hiddenSize]`.
    func dflashTokenEmbeddings(_ tokens: MLXArray) -> MLXArray
    /// The output head alone, with no logit multiplier and no softcap:
    /// `[..., vocabulary]`. Both are increasing maps, so the argmax is the
    /// target's.
    func dflashHeadProduct(_ hidden: MLXArray) -> MLXArray
    /// The target's readout after its head product (logit multiplier, then
    /// softcap), so a head product becomes the logits the target samples
    /// from. Only coupled proposals read it: their race needs the target's
    /// scale, while an argmax does not.
    func dflashReadout(_ headProduct: MLXArray) -> MLXArray
    /// True when one forward over a verify window gives every row the bytes
    /// of the decode step it replaces (`rectangularExact`).
    var dflashVerifiesWindowsExactly: Bool { get }
}

extension MuseGlimmerDFlashTarget {
    public var dflashVerifiesWindowsExactly: Bool { false }
    public func dflashReadout(_ headProduct: MLXArray) -> MLXArray { headProduct }
}

/// The Muse Glimmer DFlash block drafter as a request-stateful CBv2 drafter.
///
/// Bound to one target instance. Each request owns the keys and values of its
/// last `sliding_window` committed target positions. The first draft step of
/// a round runs one block forward and keeps its proposals; later steps of the
/// round return them. Proposals are the argmax of the target's own head (or,
/// with `couplesSampledProposals`, a sampled row's race with its verify
/// noise), and the engine keeps one only when it equals the token the target
/// itself chose, so they change speed and never the output.
///
/// With `batchesRequests`, the requests of one step share their passes
/// (`CBv2MTPBatchedRequestDrafter`): one context commit for every observed
/// row, and one block forward and head product for every row of a round.
/// A round's accepted rows are then committed with the next round's carry,
/// in that round's pass.
public final class MuseGlimmerDFlashDrafter: CBv2MTPAdvisoryHistoryDrafter,
    CBv2MTPCoupledProposalDrafter, CBv2MTPBatchedRequestDrafter, CBv2MTPPostVerifySpeculating,
    @unchecked Sendable
{
    /// `startRound` builds the complete block and every later `draftStep`
    /// merely exposes a memo slice.  The proposal token roots the block and
    /// head product; returned hidden slices are not chained into it.
    public let requiresDraftHiddenEvaluation = false

    public struct Settings: Sendable, Equatable {
        /// Draft tokens per round, at most the engine's tested window of 7
        /// and the block's `blockSize - 1`.
        public var maximumDraftTokens: Int
        /// How the target scores a round; nil keeps the engine's setting.
        public var verification: CBv2MTPVerificationMode?
        /// Requests one round may hold. More than one, with rectangular
        /// verification, needs a target whose rows do not depend on each
        /// other.
        public var maximumSpeculativeRows: Int
        /// True only for a target whose verify arithmetic for one row never
        /// depends on the other rows of the step.
        public var commitsAcceptedPrefixPerRow: Bool
        /// Separately admitted allowance for allocator and backend scratch.
        public var backendScratchBytes: Int
        /// A rectangular round attends its (row, column) pairs as rows of one
        /// paged decode dispatch per layer instead of one per column.
        public var batchesVerifyColumnsAsDecodeRows: Bool
        /// Sampled rows propose the race of the draft's row with the noise
        /// of the target's verify draw at each position, instead of the
        /// argmax (`CBv2MTPCoupledProposalDrafter`). Greedy rows keep the
        /// argmax. Output is unchanged either way.
        public var couplesSampledProposals: Bool
        /// The requests of one step share the drafter's passes: observed
        /// rows are committed in one projection, and a round's rows draft
        /// their blocks and head rows in one forward
        /// (`CBv2MTPBatchedRequestDrafter`). Proposals may move in low bits;
        /// output is unchanged either way.
        public var batchesRequests: Bool

        public init(
            maximumDraftTokens: Int = CBv2MTPConfig.testedMaxDraftTokens,
            verification: CBv2MTPVerificationMode? = nil, maximumSpeculativeRows: Int = 1,
            commitsAcceptedPrefixPerRow: Bool = false, backendScratchBytes: Int = 0,
            batchesVerifyColumnsAsDecodeRows: Bool = false, couplesSampledProposals: Bool = false,
            batchesRequests: Bool = false
        ) {
            self.maximumDraftTokens = maximumDraftTokens
            self.verification = verification
            self.maximumSpeculativeRows = maximumSpeculativeRows
            self.commitsAcceptedPrefixPerRow = commitsAcceptedPrefixPerRow
            self.backendScratchBytes = backendScratchBytes
            self.batchesVerifyColumnsAsDecodeRows = batchesVerifyColumnsAsDecodeRows
            self.couplesSampledProposals = couplesSampledProposals
            self.batchesRequests = batchesRequests
        }

        /// One-forward exact verification on a target with exact windows,
        /// each row committing its own prefix and attending its columns as
        /// decode rows; serial verification otherwise.
        public static func verifying(
            _ target: any MuseGlimmerDFlashTarget, maximumSpeculativeRows: Int = 1,
            backendScratchBytes: Int = 0, batchesVerifyColumnsAsDecodeRows: Bool = true,
            couplesSampledProposals: Bool = false, batchesRequests: Bool = false
        ) -> Settings {
            let exact = target.dflashVerifiesWindowsExactly
            return Settings(
                verification: exact ? .rectangularExact : .serialTarget,
                maximumSpeculativeRows: exact ? maximumSpeculativeRows : 1,
                commitsAcceptedPrefixPerRow: exact, backendScratchBytes: backendScratchBytes,
                batchesVerifyColumnsAsDecodeRows: exact && batchesVerifyColumnsAsDecodeRows,
                couplesSampledProposals: couplesSampledProposals, batchesRequests: batchesRequests)
        }
    }

    public let target: any MuseGlimmerDFlashTarget
    public let model: MuseGlimmerDFlashDraftModel
    public let settings: Settings
    /// `[1, hiddenSize]`, the target's raw embedding of the mask token.
    private let maskEmbedding: MLXArray
    /// Low-acceptance rounds' MLP: 3-bit group-32 copies of each layer's pinned 5-bit
    /// gate/up (concatenated into `gateUp` when `fusedProjEnabled`) and down
    /// (`CBV2_DRAFTER_LOW_MLP_BITS`, 5 disables). Empty when disabled or when a
    /// projection is not the pinned affine form.
    private let lowMLP:
        [(
            gate: QuantizedLinear?, up: QuantizedLinear?, gateUp: QuantizedLinear?,
            down: QuantizedLinear
        )]
    /// Load-time concatenated 4-bit drafter projections (bit-identical to separate
    /// calls; gated on low-acceptance rounds to keep high-acceptance Pair 1 untouched).
    private static let fusedProjEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_DRAFTER_FUSED_PROJ"] != "0"
    private static let nvfp4DrafterEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_DRAFTER_NVFP4"] != "0"

    private struct NVFP4DrafterState {
        let fc: QuantizedLinear
        let fcEps: Float
        let encOutputNormWeightScaled: MLXArray
        let fusedContextKV: QuantizedLinear
        let blockQKV: [QuantizedLinear]
        let lastKV: QuantizedLinear
        let lastQ: QuantizedLinear
        let oProj: [QuantizedLinear]
        let oInvScale: [Float]
        let gateUp: [QuantizedLinear]
        let down: [QuantizedLinear]
        let downInvScale: [Float]
        let inputNormWeightScaled: [MLXArray]
        let postAttnNormWeightScaled: [MLXArray]

        var residentArrays: [MLXArray] {
            var arr: [MLXArray] = [
                fc.weight, fc.scales, encOutputNormWeightScaled,
                fusedContextKV.weight, fusedContextKV.scales,
                lastKV.weight, lastKV.scales, lastQ.weight, lastQ.scales,
            ]
            if let b = fc.biases { arr.append(b) }
            for ql in blockQKV {
                arr += [ql.weight, ql.scales]
                if let b = ql.biases { arr.append(b) }
            }
            for ql in oProj {
                arr += [ql.weight, ql.scales]
                if let b = ql.biases { arr.append(b) }
            }
            for ql in gateUp {
                arr += [ql.weight, ql.scales]
                if let b = ql.biases { arr.append(b) }
            }
            for ql in down {
                arr += [ql.weight, ql.scales]
                if let b = ql.biases { arr.append(b) }
            }
            arr += inputNormWeightScaled
            arr += postAttnNormWeightScaled
            return arr
        }
    }

    private let fusedContextKV: QuantizedLinear?
    private let fusedBlockQKV: [QuantizedLinear]
    private let fusedLastKV: QuantizedLinear?
    private let nvfp4Drafter: NVFP4DrafterState?
    /// The target's NVFP4 `lm_head` when it has one. Draft rows take MLX's
    /// own few-row NVFP4 product of it (the NAX `qmv` route on an M5) rather
    /// than the target's row-exact verify kernel: the same E2M1 x E4M3
    /// weights, FP32 accumulation and `weight_scale_2`, in another order.
    /// Proposals only decide acceptance; the target verifies every token.
    private let draftHead: ModelOptWeightOnlyNVFP4Linear?
    /// Read-only 2-stage compressed draft head (1-bit sign + g256 L1 scale + exact NVFP4
    /// Top-1024 rescore) used strictly on low-acceptance dynamic-tree rounds.
    private let compressedDraftHead: CompressedDraftHead?

    public init(
        target: any MuseGlimmerDFlashTarget, model: MuseGlimmerDFlashDraftModel,
        settings: Settings = Settings()
    ) throws {
        let c = model.configuration
        guard target.dflashHiddenSize == c.hiddenSize else {
            throw MuseGlimmerDFlashError.target("hidden size")
        }
        guard target.dflashContextWidth == c.contextWidth else {
            throw MuseGlimmerDFlashError.target("context width")
        }
        guard target.dflashTappedLayerIDs == c.targetLayerIDs else {
            throw MuseGlimmerDFlashError.target("tapped layers")
        }
        guard c.maskTokenID < target.dflashVocabularySize else {
            throw MuseGlimmerDFlashError.target("mask token")
        }
        guard
            (1 ... min(CBv2MTPConfig.testedMaxDraftTokens, c.draftTokensPerBlock)).contains(
                settings.maximumDraftTokens),
            (1 ... CBv2MTPConfig.testedMaxSpeculativeBatch).contains(
                settings.maximumSpeculativeRows),
            settings.backendScratchBytes >= 0
        else { throw MuseGlimmerDFlashError.configuration("settings") }
        guard settings.verification != .rectangularExact || target.dflashVerifiesWindowsExactly
        else { throw MuseGlimmerDFlashError.target("exact verify windows") }
        self.target = target
        self.model = model
        self.settings = settings
        let mask = target.dflashTokenEmbeddings(MLXArray([Int32(c.maskTokenID)]))
        guard mask.shape == [1, c.hiddenSize] else {
            throw MuseGlimmerDFlashError.target("embedding shape")
        }
        maskEmbedding = mask.asType(model.activationDType)
        draftHead = (target as? MuseGlimmerTextModel)?.lmHead as? ModelOptWeightOnlyNVFP4Linear
        // Build deep-low `.nvfp4` 4b-g16 prescaled drafter BEFORE `requantize8BitLayers`
        // so `encoder.fc` and `self_attn` quantize from pristine 8-bit weights with zero double-quant.
        nvfp4Drafter = Self.buildNVFP4Drafter(model)
        Self.requantize8BitLayers(model)
        lowMLP = Self.lowAcceptanceMLP(model)
        let fused = Self.buildFusedAttentionProjections(model)
        fusedContextKV = fused.contextKV
        fusedBlockQKV = fused.blockQKV
        fusedLastKV = fused.lastKV
        compressedDraftHead =
            (Device.defaultDevice().deviceType == .gpu && model.activationDType == .bfloat16)
            ? draftHead.flatMap { CompressedDraftHead(head: $0) }
            : nil
        let headScale2: Float = draftHead.map { $0.weightScale2.asType(.float32).item(Float.self) } ?? 1.0
        let outMult: Float = (target as? MuseGlimmerTextModel)?.configuration.outputMultiplier ?? 1.0
        let softcap: Float = (target as? MuseGlimmerTextModel)?.configuration.finalLogitSoftcapping ?? 0.0
        let effScale = headScale2 * outMult
        let dtParams = MLXArray([
            effScale,
            softcap,
            CBv2DraftRegime.dynamicTreeTemp,
            CBv2DraftRegime.dynamicTreeChainBonus,
            CBv2DraftRegime.dynamicTreeDecay,
            Float(CBv2DraftRegime.dynamicTreeMaxLeaves),
            Float(CBv2DraftRegime.dynamicTreeMaxLeafDepth),
            Float(CBv2DraftRegime.dynamicTreeMaxLeavesAt5),
            Float(CBv2DraftRegime.dynamicTreeMaxBranchLen),
            CBv2DraftRegime.dynamicTreeContDecay,
            Float(CBv2DraftRegime.dynamicTreeMaxLeavesAt6),
            Float(CBv2DraftRegime.dynamicTreePurityMode),
            Float(CBv2DraftRegime.dynamicTreeDPMode),
            Float(CBv2DraftRegime.dynamicTreeTopK),
            CBv2DraftRegime.dynamicTreeTempSlope,
            CBv2DraftRegime.dynamicTreeLeafDecay,
            CBv2DraftRegime.dynamicTreePurityExp,
        ])
        eval(dtParams)
        self.dynTreeParams = dtParams
        contextKeyEpilogue = ContextKeyEpilogue(model)
        try MLX.withError { errors in
            var warm = [maskEmbedding]
            if let textTarget = target as? MuseGlimmerTextModel,
                model.activationDType == .bfloat16
            {
                let dummyLogits = MLXArray.zeros(
                    [1, target.dflashVocabularySize], dtype: .bfloat16)
                let (t, conf) = Self.fusedArgMaxAndConfidence(
                    dummyLogits,
                    outputMultiplier: textTarget.configuration.outputMultiplier,
                    softcap: textTarget.configuration.finalLogitSoftcapping)
                warm += [t, conf, Self.fastArgMax(dummyLogits)]
                if CBv2DraftRegime.verifyTreeMode != 0 && CBv2DraftRegime.dynamicTreeEnabled {
                    let dummyRows = MLXArray.zeros(
                        [settings.maximumDraftTokens, target.dflashVocabularySize],
                        dtype: .bfloat16)
                    let p = Self.fastDynamicTreeStage1(dummyRows, params: dtParams)
                    warm += [
                        p.val, p.idx, p.sum,
                        Self.fastDynamicTreeArgMax(
                            p.val, p.idx, vocab: target.dflashVocabularySize),
                    ]
                    if let comp = compressedDraftHead {
                        let dummyH = MLXArray.zeros(
                            [settings.maximumDraftTokens, c.hiddenSize], dtype: .bfloat16)
                        if let cp = comp.dynamicTreePartials(dummyH, params: dtParams) {
                            warm += [
                                cp.val, cp.idx, cp.sum,
                                Self.fastDynamicTreeArgMax(
                                    cp.val, cp.idx, vocab: target.dflashVocabularySize),
                            ]
                        }
                        let maxCk = min(7, settings.maximumDraftTokens)
                        if maxCk >= 2 {
                            for dk in 2 ... maxCk {
                                let dkH = MLXArray.zeros([dk, c.hiddenSize], dtype: .bfloat16)
                                if let cp = comp.dynamicTreePartials(dkH, params: dtParams) {
                                    let dyn = Self.fastDynamicTreeStage2(
                                        cp.val, cp.idx, cp.sum, k: dk,
                                        vocab: target.dflashVocabularySize, params: dtParams)
                                    warm += [cp.val, cp.idx, cp.sum, dyn.tokens, dyn.treeMeta]
                                }
                            }
                        }
                    }
                    let maxDk = min(7, settings.maximumDraftTokens)
                    if maxDk >= 2 {
                        for dk in 2 ... maxDk {
                            let dyn = Self.fastDynamicTreeStage2(
                                p.val, p.idx, p.sum, k: dk,
                                vocab: target.dflashVocabularySize, params: dtParams)
                            let dummyDrafts = MLXArray.zeros([1, dk], type: Int32.self)
                            let dummyTargets = MLXArray.zeros([1, dk + 1], type: Int32.self)
                            let out = Self.fastTreeOutcome(
                                drafts: dummyDrafts, targets: dummyTargets,
                                treeMeta: dyn.treeMeta, k: dk)
                            warm += [
                                dyn.tokens, dyn.treeMeta,
                                out.carrySlot, out.accepted, out.visible,
                            ]
                        }
                    }
                }
            }
            if let epilogue = contextKeyEpilogue {
                let key = MLXArray.zeros([1, c.kvHeads * c.headDimensions], dtype: .bfloat16)
                warm.append(epilogue.apply(Array(repeating: key, count: 5), rows: 1, position: 0))
                if CBv2DraftRegime.verifyTreeMode != 0 && CBv2DraftRegime.dynamicTreeEnabled {
                    let dummyMeta = MLXArray.zeros([18], type: Int32.self)
                    warm.append(
                        epilogue.apply(
                            Array(repeating: key, count: 5), rows: 1, position: 0,
                            treeMeta: dummyMeta))
                }
                if fusedContextKV != nil {
                    let allKV = MLXArray.zeros(
                        [1, 10 * c.kvHeads * c.headDimensions], dtype: .bfloat16)
                    let kv0 = epilogue.applyFusedKV(allKV, rows: 1, position: 0)
                    warm += [kv0.keys, kv0.values]
                    if CBv2DraftRegime.verifyTreeMode != 0 && CBv2DraftRegime.dynamicTreeEnabled {
                        let dummyMeta = MLXArray.zeros([18], type: Int32.self)
                        let kv1 = epilogue.applyFusedKV(
                            allKV, rows: 1, position: 0, treeMeta: dummyMeta)
                        warm += [kv1.keys, kv1.values]
                    }
                }
            }
            let dummyRopeIn = MLXArray.zeros(
                [1, c.kvHeads, c.blockSize, c.headDimensions], dtype: model.activationDType)
            warm.append(
                MLXFast.RoPE(
                    dummyRopeIn, dimensions: c.headDimensions, traditional: false,
                    base: c.ropeTheta, scale: 1, offset: MLXArray(Int32(0))))
            if Device.defaultDevice().deviceType == .gpu {
                if Self.inPlaceRingEnabled {
                    warm.append(
                        FastInPlaceContext.warmRingWriteKernel(
                            layers: c.hiddenLayers, kvHeads: c.kvHeads, window: c.slidingWindow,
                            headDimensions: c.headDimensions, dtype: model.activationDType))
                }
                // Warm branch-hit gather (`take(idx, axis: 2)` for 4D KV and `take(idx, axis: 1)`
                // for 3D target hidden) and device-outcome scalar ops.
                let dummyKV = MLXArray.zeros(
                    [c.hiddenLayers, c.kvHeads, 8, c.headDimensions], dtype: model.activationDType)
                let dummyIdx = MLXArray([Int32(0), Int32(2)])
                warm.append(dummyKV.take(dummyIdx, axis: 2))
                let dummyTargetH = MLXArray.zeros(
                    [1, 8, c.contextWidth], dtype: model.activationDType)
                warm.append(dummyTargetH.take(dummyIdx, axis: 1))
                let dummyDrafts = MLXArray.zeros([1, 7], type: Int32.self)
                let dummyTargets = MLXArray.zeros([1, 8], type: Int32.self)
                let dummyMatch = equal(dummyDrafts, dummyTargets[0..., 0 ..< 7]).asType(.int32)
                let dummyAcc = cumprod(dummyMatch, axis: 1).sum()
                let dummyVis = lessEqual(MLXArray(Array(Int32(0) ..< Int32(8))), dummyAcc)
                let dummySeed = maximum(
                    takeAlong(dummyTargets, dummyAcc.reshaped([1, 1]), axis: 1),
                    MLXArray(Int32(0)))
                warm += [dummyVis, dummySeed]
                // Warm 4-bit context projection, K/V projections, drafter MLP (normal + low-bit),
                // and draft head across every reachable row count `1 ... (maximumDraftTokens + 1)`
                // so MLX's quantized matvec (`affine_qmv_wide` / `fp_qmv`) never compiles inside
                // a timed decode window.
                let prevLow = CBv2DraftRegime.lowAcceptance
                for rows in 1 ... (settings.maximumDraftTokens + 1) {
                    let dummyFeat = MLXArray.zeros(
                        [rows, c.contextWidth], dtype: model.activationDType)
                    let projCtx = model.projectContext(dummyFeat)
                    CBv2DraftRegime.lowAcceptance = false
                    let kvHi = speculativeContextKeyValues(projCtx, at: 0)
                    warm += [kvHi.keys, kvHi.values]
                    if fusedContextKV != nil {
                        CBv2DraftRegime.lowAcceptance = true
                        let kvLo = speculativeContextKeyValues(projCtx, at: 0)
                        warm += [kvLo.keys, kvLo.values]
                    }
                    if rows <= settings.maximumDraftTokens {
                        let dummyH = MLXArray.zeros(
                            [1, rows, c.hiddenSize], dtype: model.activationDType)
                        if let firstLayer = model.layers.first {
                            CBv2DraftRegime.lowAcceptance = false
                            warm.append(draftMLP(0, firstLayer.mlp, dummyH))
                            CBv2DraftRegime.lowAcceptance = true
                            warm.append(draftMLP(0, firstLayer.mlp, dummyH))
                        }
                        let dummyHeadIn = MLXArray.zeros(
                            [rows, c.hiddenSize], dtype: model.activationDType)
                        warm.append(draftHeadProduct(dummyHeadIn, readout: false))
                    }
                }
                warm.append(
                    concatenated(
                        [
                            MLXArray.ones([4], type: Bool.self),
                            MLXArray.ones([4], type: Bool.self),
                        ],
                        axis: 0))
                // Warm masked AND unmasked drafter SDPA for both `kL % 32 == 0` (544) and
                // `kL % 32 != 0` (536) so neither `align_K_t` nor `align_K_n` compiles inside
                // a timed decode window.
                let dummyEmb = model.blockEmbeddings(anchor: maskEmbedding, mask: maskEmbedding)
                let devPos = MLXArray(Int32(512))
                for ctxLen in [520, 528] {
                    let ck = MLXArray.zeros(
                        [c.hiddenLayers, c.kvHeads, ctxLen, c.headDimensions],
                        dtype: model.activationDType)
                    let cv = MLXArray.zeros(
                        [c.hiddenLayers, c.kvHeads, ctxLen, c.headDimensions],
                        dtype: model.activationDType)
                    let cols = MLXArray.ones([ctxLen + c.blockSize], type: Bool.self)
                    let mask2d = broadcast(
                        cols.reshaped([1, ctxLen + c.blockSize]),
                        to: [c.blockSize, ctxLen + c.blockSize])
                    CBv2DraftRegime.lowAcceptance = true
                    let maskedRows = fastDraftBlockRows(
                        embeddings: dummyEmb, at: ctxLen, context: (ck, cv), mask: mask2d,
                        draftCount: settings.maximumDraftTokens, devicePosition: devPos)
                    CBv2DraftRegime.lowAcceptance = false
                    let unmaskedRows = fastDraftBlockRows(
                        embeddings: dummyEmb, at: ctxLen, context: (ck, cv), mask: nil,
                        draftCount: settings.maximumDraftTokens)
                    eval(maskedRows, unmaskedRows)
                }
                if let nv = nvfp4Drafter, let epilogue = contextKeyEpilogue {
                    let prevDeep = CBv2DraftRegime.xnorDraftHeadEligible
                    CBv2DraftRegime.lowAcceptance = true
                    CBv2DraftRegime.xnorDraftHeadEligible = true
                    defer { CBv2DraftRegime.xnorDraftHeadEligible = prevDeep }
                    var nvWarm = [MLXArray]()
                    for rows in 1 ... (settings.maximumDraftTokens + 1) {
                        let df = MLXArray.zeros([rows, c.contextWidth], dtype: .bfloat16)
                        let kv = speculativeProjectAndContextKV(df, at: 0)
                        nvWarm += [kv.keys, kv.values]
                    }
                    let qW = c.attentionHeads * c.headDimensions
                    let dummyRingK = MLXArray.zeros(
                        [c.hiddenLayers, c.kvHeads, c.slidingWindow, 128], dtype: .bfloat16)
                    let dummyRingV = MLXArray.zeros(
                        [c.hiddenLayers, c.kvHeads, c.slidingWindow, 128], dtype: .bfloat16)
                    let dummyPos = MLXArray([Int32(0)])
                    let dummyIP = MLXArray([Int32(0), Int32(0)])
                    let dummyFence = MLXArray.zeros([1], dtype: .int32)
                    if let firstLayer = model.layers.first {
                        let qkvLaunch = Self.preparedBlockQKVRing(
                            qHeads: c.attentionHeads, kvHeads: c.kvHeads,
                            block: c.blockSize, window: c.slidingWindow)
                        let wQKV = MLXArray.zeros(
                            [1, c.blockSize, (c.attentionHeads + 2 * c.kvHeads) * 128],
                            dtype: .bfloat16)
                        let o1 = qkvLaunch([
                            wQKV, firstLayer.attention.qNorm.weight,
                            firstLayer.attention.kNorm.weight, dummyRingK, dummyRingV,
                            dummyPos, epilogue.coefficients, dummyIP, dummyFence,
                        ])
                        let kvLaunch = Self.preparedBlockKVRing(
                            kvHeads: c.kvHeads, block: c.blockSize, window: c.slidingWindow)
                        let wKV = MLXArray.zeros(
                            [1, c.blockSize, 2 * c.kvHeads * 128], dtype: .bfloat16)
                        let o2 = kvLaunch([
                            wKV, firstLayer.attention.kNorm.weight, dummyRingK, dummyRingV,
                            dummyPos, epilogue.coefficients, dummyIP, o1[1],
                        ])
                        nvWarm += [o1[0], o1[1], o2[0]]
                    }
                    let lastIdx = c.hiddenLayers - 1
                    let lastLayer = model.layers[lastIdx]
                    for rows in [c.blockSize, 1, 2, 3, 4, 5, 6, 7] {
                        let dh = MLXArray.zeros([1, rows, c.hiddenSize], dtype: .bfloat16)
                        let dq = MLXArray.zeros([1, rows, qW], dtype: .bfloat16)
                        let mlpIdx = (rows == c.blockSize) ? 0 : lastIdx
                        if rows == c.blockSize {
                            nvWarm.append(nv.blockQKV[0](dh))
                            nvWarm.append(nv.lastKV(dh))
                        } else {
                            let lq = nv.lastQ(dh)
                            nvWarm.append(
                                Self.applyBlockQEpilogue(
                                    lq, qNormWeight: lastLayer.attention.qNorm.weight,
                                    posScalar: dummyPos, coefficients: epilogue.coefficients,
                                    qHeads: c.attentionHeads, rows: rows))
                        }
                        let oOut = nv.oProj[mlpIdx](dq)
                        let (hMid, mx) = Self.fusedAddRMSNorm(
                            residual: dh, delta: oOut, weight: nv.postAttnNormWeightScaled[mlpIdx],
                            eps: lastLayer.postAttentionNorm.eps, deltaScale: nv.oInvScale[mlpIdx])
                        let gu = nv.gateUp[mlpIdx](mx)
                        let downOut = nv.down[mlpIdx](Self.splitSiluMul(gu, scale: nv.downInvScale[mlpIdx]))
                        nvWarm += [hMid, downOut]
                    }
                    eval(nvWarm)
                    // Warm strided-ring SDPA on deep-low NVFP4 path for both 32-aligned and unaligned ctxLen,
                    // covering BOTH unmasked and Boolean-masked contiguous-Q (`boolmask_qnt`) used by live
                    // `speculateOutcomeBlock` so no SDPA Metal PSO compiles during timed decode.
                    for ctxLen in [520, 528] {
                        let tmpCtx = FastInPlaceContext(
                            configuration: c, dtype: .bfloat16, inPlaceRing: true)
                        tmpCtx.append(
                            keys: MLXArray.zeros(
                                [c.hiddenLayers, c.kvHeads, ctxLen, 128], dtype: .bfloat16),
                            values: MLXArray.zeros(
                                [c.hiddenLayers, c.kvHeads, ctxLen, 128], dtype: .bfloat16))
                        let vis = tmpCtx.visibleContext()
                        let rUnmasked = fastDraftBlockRows(
                            embeddings: dummyEmb, at: ctxLen,
                            context: vis.map { ($0.keys, $0.values) }, mask: vis?.mask,
                            draftCount: settings.maximumDraftTokens,
                            ringContext: tmpCtx, ringTailStart: ctxLen)
                        let cols = MLXArray.ones([ctxLen + c.blockSize], type: Bool.self)
                        let boolMask = broadcast(
                            cols.reshaped([1, ctxLen + c.blockSize]),
                            to: [c.blockSize, ctxLen + c.blockSize])
                        let rMasked = fastDraftBlockRows(
                            embeddings: dummyEmb, at: ctxLen,
                            context: vis.map { ($0.keys, $0.values) }, mask: boolMask,
                            draftCount: settings.maximumDraftTokens, devicePosition: devPos,
                            ringContext: tmpCtx, ringTailStart: ctxLen)
                        eval(rUnmasked, rMasked)
                    }
                }
                CBv2DraftRegime.lowAcceptance = prevLow
            }
            eval(warm)
            try errors.check()
        }
    }

    // MARK: Engine policy

    public var mtpTargetIdentity: ObjectIdentifier? { target.dflashTargetIdentity }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { settings.verification }
    public var maximumDraftTokens: Int? { settings.maximumDraftTokens }
    public var maximumSpeculativeBatch: Int? { settings.maximumSpeculativeRows }
    public var commitsAcceptedPrefixPerRow: Bool { settings.commitsAcceptedPrefixPerRow }
    public var batchesVerifyColumnsAsDecodeRows: Bool {
        settings.verification == .rectangularExact && settings.batchesVerifyColumnsAsDecodeRows
    }
    /// Exact acceptance only (decision D6): typical acceptance can emit a
    /// draft, and the advisory history then loses its observation positions.
    public var requiresExactAcceptance: Bool { true }
    /// A proposal is kept only when it equals the token pre-sampled from the
    /// target's own distribution, and that token is what is emitted, so
    /// sampled rows speculate too.
    public var supportsTargetPrefixAcceptance: Bool { true }
    public var couplesSampledProposals: Bool { settings.couplesSampledProposals }
    public var batchesRequests: Bool { settings.batchesRequests }
    /// Observation builds lazy writes into the context ring; they must be
    /// evaluated with the step that produced the observed rows.
    public var requiresCommittedObservationFence: Bool { true }
    /// The prompt's observation commits every prompt row, the last one
    /// included, and awaits no carry row; the first round's block then
    /// anchors on the first token, exactly as after a seed step, which only
    /// moved both one position on. Proposals change speed, never the output.
    public var seedsFirstRoundFromPrefill: Bool { true }
    /// The ring is a fixed charge (`boundedRequestAllocation`), not a rate.
    public var requestStateBytesPerToken: Int { 0 }
    /// The context only decides what is proposed. It is not part of a prefix
    /// checkpoint; after a restore it starts at the first observed position.
    public var requestStateIsAdvisory: Bool { true }

    // MARK: Request state

    private static let inPlaceRingEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_DFLASH_INPLACE_RING"] != "0"

    /// Request-owned zero-copy in-place DFlash ring context (eliminates ~84 MB/round
    /// failed-donation `SliceUpdate` copies while keeping exact `[L, H, W, D]` storage
    /// and dependency ordering via `writeFence` + `depends`).
    final class FastInPlaceContext {
        let layers: Int
        let kvHeads: Int
        let headDimensions: Int
        let window: Int
        let blockSize: Int
        let dtype: DType
        let inPlaceRing: Bool

        private static let ospecZeroCopyEnabled: Bool =
            ProcessInfo.processInfo.environment["CBV2_OSPEC_ZEROCOPY"] != "0"
        private static let ospecPromoteEnabled: Bool =
            ProcessInfo.processInfo.environment["CBV2_OSPEC_PROMOTE"] != "0"

        struct SpeculativeTailTicket {
            let keysID: ObjectIdentifier
            let valuesID: ObjectIdentifier
            let anchor: Int
            let first: Int
            let base: Int
            let startSlot: Int
            let writtenRows: Int
            let generation: UInt64
        }

        private var keys: MLXArray?
        private var values: MLXArray?
        private var writeFence: MLXArray = MLXArray.zeros([1], dtype: .int32)
        private var speculativeReadDeps: [MLXArray] = []
        private var ringGeneration: UInt64 = 0
        private var basePosition = 0
        private(set) var firstPosition = 0
        private(set) var nextPosition = 0

        private static let ringWriteKernel = MLXFast.metalKernel(
            name: "dflash_ring_write_inplace",
            inputNames: ["new_keys", "new_values", "ring_keys", "ring_values", "params", "previous"],
            outputNames: ["fence"],
            source: """
                const uint d = thread_position_in_grid.x;
                const uint r = thread_position_in_grid.y;
                const uint lh = thread_position_in_grid.z;
                const uint rows = uint(params[1]);
                if (d < uint(D_) && r < rows) {
                    const uint l = lh / uint(H_);
                    const uint h = lh % uint(H_);
                    const uint dst_slot = (uint(params[0]) + r) % uint(W_);
                    const size_t k_src =
                        size_t(l) * size_t(new_keys_strides[0])
                        + size_t(h) * size_t(new_keys_strides[1])
                        + size_t(r) * size_t(new_keys_strides[2])
                        + size_t(d) * size_t(new_keys_strides[3]);
                    const size_t v_src =
                        size_t(l) * size_t(new_values_strides[0])
                        + size_t(h) * size_t(new_values_strides[1])
                        + size_t(r) * size_t(new_values_strides[2])
                        + size_t(d) * size_t(new_values_strides[3]);
                    const size_t dst =
                        (size_t(lh) * size_t(W_) + size_t(dst_slot)) * size_t(D_) + size_t(d);
                    device T* rk = ring_keys;
                    device T* rv = ring_values;
                    rk[dst] = new_keys[k_src];
                    rv[dst] = new_values[v_src];
                }
                if (d == 0u && r == 0u && lh == 0u) {
                    fence[0] = previous[0] + 1;
                }
                """,
            ensureRowContiguous: false,
            mutableInputs: ["ring_keys", "ring_values"])

        private struct RingWriteLaunchKey: Hashable {
            let dtype: DType
            let kvHeads: Int
            let window: Int
            let headDimensions: Int
            let rows: Int
            let layers: Int
        }

        private final class RingLaunchBox: @unchecked Sendable {
            var launches: [RingWriteLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
        }

        private static let ringLaunchLock = OSAllocatedUnfairLock(initialState: RingLaunchBox())

        private static func preparedRingWrite(
            dtype: DType, kvHeads: Int, window: Int, headDimensions: Int, rows: Int, layers: Int
        ) -> MLXFast.MLXFastKernel.Launch {
            let key = RingWriteLaunchKey(
                dtype: dtype, kvHeads: kvHeads, window: window, headDimensions: headDimensions,
                rows: rows, layers: layers)
            if let hit = ringLaunchLock.withLockUnchecked({ $0.launches[key] }) { return hit }
            let tgX = min(256, max(32, headDimensions))
            let gridX = ((headDimensions + tgX - 1) / tgX) * tgX
            let made = ringWriteKernel.prepare(
                template: [("T", dtype), ("H_", kvHeads), ("W_", window), ("D_", headDimensions)],
                grid: (gridX, rows, layers * kvHeads),
                threadGroup: (tgX, 1, 1),
                outputShapes: [[1]],
                outputDTypes: [.int32])
            ringLaunchLock.withLockUnchecked { $0.launches[key] = made }
            return made
        }

        /// High-acceptance rounds (`CBv2DraftRegime.lowOnlyVerifySpeedups &&
        /// !CBv2DraftRegime.lowAcceptance`) keep the record's (c95a27f5) per-call
        /// kernel call; low-acceptance rounds take the cached prepared launch.
        private static func ringWrite(
            _ inputs: [MLXArray], dtype: DType, kvHeads: Int, window: Int, headDimensions: Int,
            rows: Int, layers: Int
        ) -> MLXArray {
            if !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance {
                let launch = preparedRingWrite(
                    dtype: dtype, kvHeads: kvHeads, window: window, headDimensions: headDimensions,
                    rows: rows, layers: layers)
                return launch(inputs)[0]
            }
            let tgX = min(256, max(32, headDimensions))
            let gridX = ((headDimensions + tgX - 1) / tgX) * tgX
            return ringWriteKernel(
                inputs,
                template: [
                    ("T", dtype), ("H_", kvHeads), ("W_", window), ("D_", headDimensions),
                ],
                grid: (gridX, rows, layers * kvHeads),
                threadGroup: (tgX, 1, 1),
                outputShapes: [[1]],
                outputDTypes: [.int32])[0]
        }

        static func warmRingWriteKernel(
            layers: Int, kvHeads: Int, window: Int, headDimensions: Int, dtype: DType
        ) -> MLXArray {
            let nk = MLXArray.zeros([layers, kvHeads, 1, headDimensions], dtype: dtype)
            let nv = MLXArray.zeros([layers, kvHeads, 1, headDimensions], dtype: dtype)
            let rk = MLXArray.zeros([layers, kvHeads, window, headDimensions], dtype: dtype)
            let rv = MLXArray.zeros([layers, kvHeads, window, headDimensions], dtype: dtype)
            let params = MLXArray([Int32(0), Int32(1)])
            let prev = MLXArray.zeros([1], dtype: .int32)
            let launch = preparedRingWrite(
                dtype: dtype, kvHeads: kvHeads, window: window, headDimensions: headDimensions,
                rows: 1, layers: layers)
            return launch([nk, nv, rk, rv, params, prev])[0]
        }

        init(configuration c: MuseGlimmerDFlashConfiguration, dtype: DType, inPlaceRing: Bool) {
            layers = c.hiddenLayers
            kvHeads = c.kvHeads
            headDimensions = c.headDimensions
            window = c.slidingWindow
            blockSize = c.blockSize
            self.dtype = dtype
            self.inPlaceRing = inPlaceRing
        }

        var count: Int { nextPosition - firstPosition }

        var arrays: [MLXArray] {
            guard let keys, let values else { return [] }
            return inPlaceRing
                ? depends(inputs: [keys, values], dependencies: [writeFence]) : [keys, values]
        }

        func restart(at position: Int) {
            precondition(position >= 0, "DFlash context position must not be negative")
            ringGeneration &+= 1
            basePosition = position
            firstPosition = position
            nextPosition = position
        }

        func release() {
            keys = nil
            values = nil
            writeFence = MLXArray.zeros([1], dtype: .int32)
            speculativeReadDeps.removeAll(keepingCapacity: false)
            restart(at: 0)
        }

        private func consumeReadBarriers() {
            guard !speculativeReadDeps.isEmpty else { return }
            writeFence = depends(inputs: [writeFence], dependencies: speculativeReadDeps)[0]
            speculativeReadDeps.removeAll(keepingCapacity: true)
        }

        /// Writes unconfirmed verify rows directly into the contiguous unused tail
        /// `[count ..< count + rows)` of the ring without advancing `nextPosition`,
        /// avoiding a full ring copy inside `speculateOutcomeBlock`.
        func writeSpeculativeTail(
            keys newKeys: MLXArray, values newValues: MLXArray
        ) -> (keys: MLXArray, values: MLXArray, ticket: SpeculativeTailTicket)? {
            let rows = newKeys.dim(2)
            guard Self.ospecZeroCopyEnabled, inPlaceRing, let keys, let values,
                rows > 0, firstPosition == basePosition, count + rows <= window
            else { return nil }
            consumeReadBarriers()
            ringGeneration &+= 1
            let nk = newKeys.dtype == dtype ? newKeys : newKeys.asType(dtype)
            let nv = newValues.dtype == dtype ? newValues : newValues.asType(dtype)
            let params = MLXArray([Int32(count), Int32(rows)])
            writeFence = Self.ringWrite(
                [nk, nv, keys, values, params, writeFence], dtype: dtype, kvHeads: kvHeads,
                window: window, headDimensions: headDimensions, rows: rows, layers: layers)
            let pair = depends(inputs: [keys, values], dependencies: [writeFence])
            let total = count + rows
            let ticket = SpeculativeTailTicket(
                keysID: ObjectIdentifier(keys), valuesID: ObjectIdentifier(values),
                anchor: nextPosition, first: firstPosition, base: basePosition,
                startSlot: count, writtenRows: rows, generation: ringGeneration)
            return (
                pair[0][0..., 0..., 0 ..< total, 0...],
                pair[1][0..., 0..., 0 ..< total, 0...],
                ticket
            )
        }

        /// Promotes a verified contiguous linear prefix `[0 ..< prefixCount)` that
        /// `writeSpeculativeTail` already placed at `[count ..< count + prefixCount)`,
        /// avoiding a duplicate `dflash_ring_write_inplace` dispatch on commit.
        func promoteSpeculativePrefix(
            ticket: SpeculativeTailTicket, count prefixCount: Int
        ) -> Bool {
            guard Self.ospecPromoteEnabled, inPlaceRing, let keys, let values,
                ObjectIdentifier(keys) == ticket.keysID,
                ObjectIdentifier(values) == ticket.valuesID,
                ringGeneration == ticket.generation,
                firstPosition == basePosition,
                firstPosition == ticket.first,
                basePosition == ticket.base,
                nextPosition == ticket.anchor,
                count == ticket.startSlot,
                prefixCount > 0, prefixCount <= ticket.writtenRows,
                count + prefixCount <= window
            else { return false }
            nextPosition += prefixCount
            firstPosition = max(firstPosition, nextPosition - window)
            ringGeneration &+= 1
            return true
        }

        func recordSpeculativeRead(_ dep: MLXArray) {
            speculativeReadDeps.append(dep)
        }

        func canWriteBlockTail(at tailStart: Int, rows: Int) -> Bool {
            inPlaceRing && keys != nil && values != nil && firstPosition == basePosition
                && tailStart == count && tailStart > 0 && tailStart + rows <= window
        }

        func canWriteSpeculativeBlockTail(at tailStart: Int, rows: Int) -> Bool {
            inPlaceRing && keys != nil && values != nil && firstPosition == basePosition
                && tailStart >= count && tailStart > 0 && tailStart + rows <= window
        }

        func ringTailBuffers() -> (keys: MLXArray, values: MLXArray, fence: MLXArray)? {
            guard let keys, let values else { return nil }
            consumeReadBarriers()
            return (keys, values, writeFence)
        }

        func commitBlockTail(fence: MLXArray, readDep: MLXArray) {
            writeFence = fence
            speculativeReadDeps.append(readDep)
        }

        func append(keys newKeys: MLXArray, values newValues: MLXArray, skipping: Int = 0) {
            let rows = newKeys.dim(2)
            precondition(
                newKeys.shape == [layers, kvHeads, rows, headDimensions]
                    && newValues.shape == newKeys.shape && rows > 0 && rows <= window
                    && skipping >= 0 && (skipping == 0 || rows == window),
                "DFlash context append does not match the ring")
            if keys == nil {
                let shape = [layers, kvHeads, window, headDimensions]
                keys = MLXArray.zeros(shape, dtype: dtype)
                values = MLXArray.zeros(shape, dtype: dtype)
            }
            consumeReadBarriers()
            let start = nextPosition + skipping
            if skipping > 0 {
                basePosition = start
                firstPosition = start
            }
            if inPlaceRing && (start - basePosition) + rows <= window {
                let startSlot = (start - basePosition) % window
                let nk = newKeys.dtype == dtype ? newKeys : newKeys.asType(dtype)
                let nv = newValues.dtype == dtype ? newValues : newValues.asType(dtype)
                let params = MLXArray([Int32(startSlot), Int32(rows)])
                writeFence = Self.ringWrite(
                    [nk, nv, keys!, values!, params, writeFence], dtype: dtype, kvHeads: kvHeads,
                    window: window, headDimensions: headDimensions, rows: rows, layers: layers)
            } else {
                var written = 0
                while written < rows {
                    let slot = (start + written - basePosition) % window
                    let run = min(rows - written, window - slot)
                    let source = written ..< (written + run)
                    keys![0..., 0..., slot ..< (slot + run), 0...] =
                        newKeys[0..., 0..., source, 0...].asType(dtype)
                    values![0..., 0..., slot ..< (slot + run), 0...] =
                        newValues[0..., 0..., source, 0...].asType(dtype)
                    written += run
                }
            }
            nextPosition = start + rows
            firstPosition = max(firstPosition, nextPosition - window)
            ringGeneration &+= 1
        }

        var heldPositions: [Int] {
            guard count == window else { return Array(firstPosition ..< nextPosition) }
            let oldestSlot = (firstPosition - basePosition) % window
            return (0 ..< window).map { slot in
                firstPosition + (slot - oldestSlot + window) % window
            }
        }

        func visibleContext() -> (keys: MLXArray, values: MLXArray, mask: MLXArray?)? {
            guard count > 0, let keys, let values else { return nil }
            let pair =
                inPlaceRing
                ? depends(inputs: [keys, values], dependencies: [writeFence]) : [keys, values]
            let rows = 0 ..< count
            let visibleKeys = pair[0][0..., 0..., rows, 0...]
            let visibleValues = pair[1][0..., 0..., rows, 0...]
            let mask = MuseGlimmerDFlashContextState.visibility(
                anchor: nextPosition, blockSize: blockSize, window: window,
                contextPositions: heldPositions
            ).map { MLXArray($0, [blockSize, count + blockSize]) }
            return (visibleKeys, visibleValues, mask)
        }
    }

    private struct FastContextCommit {
        let features: MLXArray
        let position: Int?
        let state: FastInPlaceContext
    }

    private func commitToContext(
        features: MLXArray, at position: Int, to context: FastInPlaceContext
    ) {
        let rows = features.dim(0)
        let duplicates = max(0, context.nextPosition - position)
        guard rows > duplicates else { return }
        if position > context.nextPosition { context.restart(at: position) }
        let fresh = rows - duplicates
        let unread = max(0, fresh - context.window)
        let used = features[(duplicates + unread)..., 0...]
        let projected = model.contextKeyValues(
            model.projectContext(used), at: context.nextPosition + unread)
        context.append(keys: projected.keys, values: projected.values, skipping: unread)
    }

    private func commitBatchToContext(_ commits: [FastContextCommit]) {
        struct Planned {
            let state: FastInPlaceContext
            let rows: Range<Int>
            let keyPosition: Int
            let restart: Int?
            let skipping: Int
        }
        var next: [ObjectIdentifier: Int] = [:]
        var planned: [Planned] = []
        var used: [MLXArray] = []
        var total = 0
        for entry in commits {
            let features = entry.features
            let id = ObjectIdentifier(entry.state)
            let held = next[id] ?? entry.state.nextPosition
            let position = entry.position ?? held
            precondition(position >= 0, "DFlash context position must not be negative")
            let rows = features.dim(0)
            let duplicates = max(0, held - position)
            guard rows > duplicates else { continue }
            let restart = position > held ? position : nil
            let start = restart ?? held
            let fresh = rows - duplicates
            let unread = max(0, fresh - entry.state.window)
            used.append(features[(duplicates + unread)..., 0...])
            planned.append(
                Planned(
                    state: entry.state, rows: total ..< (total + fresh - unread),
                    keyPosition: start + unread, restart: restart, skipping: unread))
            total += fresh - unread
            next[id] = start + fresh
        }
        guard !planned.isEmpty else { return }
        let context = model.projectContext(
            used.count == 1 ? used[0] : concatenated(used, axis: 0))
        let projected = model.contextKeyValues(
            context, segments: planned.map { ($0.rows, $0.keyPosition) })
        for (entry, keyValues) in zip(planned, projected) {
            if let restart = entry.restart { entry.state.restart(at: restart) }
            entry.state.append(
                keys: keyValues.keys, values: keyValues.values, skipping: entry.skipping)
        }
    }

    final class State: CBv2MTPRequestState {
        let owner: ObjectIdentifier
        let context: FastInPlaceContext
        var released = false
        /// The last committed position's row has not arrived yet: a verify
        /// round hands it over as the next carry.
        var awaitsCarry = false
        /// This round's proposals `[maximumDraftTokens]` and their hidden
        /// rows `[maximumDraftTokens, hiddenSize]`.
        var memo: (tokens: MLXArray, hidden: MLXArray)?
        /// This round's coupled-proposal rule; nil proposes the argmax.
        var selection: CBv2MTPProposalSelection?
        /// `startRounds` already drafted this round's block.
        var prestarted = false
        /// The last round's accepted target rows `[rows, contextWidth]`,
        /// not yet in the ring (`batchesRequests` only). They are the next
        /// positions and are committed before any other row.
        var pending: MLXArray?
        var committedInputCount = 0
        var stagedInputCount = 0
        /// Why this request cannot continue; reported at the next fence.
        var refusal: String?
        /// The next round's index-0 work, built behind this round's verify
        /// (`speculateNextRound`): the commit for every outcome, the block
        /// for the all-accepted one. Not in `arrays`: the round's fence
        /// never waits on it.
        var speculation: Speculation?
        /// The last finalized round accepted every draft.
        var lastRoundFullyAccepted = false

        init(owner: ObjectIdentifier, context: FastInPlaceContext) {
            self.owner = owner
            self.context = context
        }

        var arrays: [MLXArray] { context.arrays + (memo.map { [$0.tokens, $0.hidden] } ?? []) }
        var materializedBytes: Int { arrays.reduce(0) { $0 + $1.nbytes } }
    }

    /// What `speculateNextRound` built and what the normal path must match
    /// for it to be handed out.
    struct Speculation {
        let depth: Int
        /// The context's `nextPosition` and `firstPosition` when built.
        let anchor: Int
        let first: Int
        /// Keys and values `[layers, kvHeads, depth + 1, headDimensions]` of
        /// every verify row at `anchor ..< anchor + depth + 1`. An outcome
        /// with `a` accepted drafts commits the first `a + 1` of them.
        let keys: MLXArray
        let values: MLXArray
        /// The next block for the all-accepted outcome, when built; dropped
        /// at confirmation for any other outcome.
        var memo: (tokens: MLXArray, hidden: MLXArray)? = nil
        /// Stage-1 top-3 + softmax partials built by `speculateOutcomeBlock` so
        /// `takeSpeculation` can run 3 us Stage-2 tree or argmax reduction for
        /// whichever `k` the next round chooses.
        var dynPartials: (val: MLXArray, idx: MLXArray, sum: MLXArray, hidden: MLXArray)? = nil
        /// The block was drafted for the outcome the verify itself decided
        /// (its accept count read on the device), not for all-accepted.
        var deviceOutcome = false
        /// `[maximumDraftTokens]` float32: each block row's top-1
        /// probability under the target's readout, evaluated with the
        /// speculation (`speculatedDraftConfidence`). Set with `memo`.
        var confidence: MLXArray? = nil
        /// The accepted draft count `finalizeRound` saw, and the pending rows
        /// it set for it (nil when it accepted none).
        var accepted: Int? = nil
        var pending: MLXArray? = nil
        /// The carry the engine stored for that outcome.
        var carry: MLXArray? = nil
        /// Verify slot path `[0, ..., carrySlot]` committed for this outcome
        /// (length `accepted + 1`; nil falls back to linear `0...accepted`).
        var acceptedPath: [Int]? = nil
        /// Ticket proving `[0 ..< depth + 1)` were already written into the
        /// request's in-place ring at `[count ..< count + depth + 1)`, so a
        /// confirmed linear prefix can be promoted without re-copying.
        var tailTicket: FastInPlaceContext.SpeculativeTailTicket? = nil
    }

    /// What `takeSpeculation` applied at a round's draft index 0.
    private enum TakenSpeculation {
        /// Nothing: the normal path runs.
        case normal
        /// The commit: the block still has to be drafted.
        case committed
        /// The commit and the all-accepted block's memo.
        case drafted
    }

    private final class Prepared: CBv2MTPPreparedCapture {}

    private func state(_ value: any CBv2MTPRequestState) -> State {
        guard let state = value as? State, state.owner == ObjectIdentifier(self), !state.released
        else { preconditionFailure("Muse Glimmer DFlash: foreign or released request state") }
        return state
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        let inPlace = Device.defaultDevice().deviceType == .gpu && Self.inPlaceRingEnabled
        return State(
            owner: ObjectIdentifier(self),
            context: FastInPlaceContext(
                configuration: model.configuration, dtype: model.activationDType,
                inPlaceRing: inPlace))
    }

    /// The engine's rule for this round's proposals. Ignored unless
    /// `couplesSampledProposals`; cleared with the round.
    public func setProposalSelection(
        _ selection: CBv2MTPProposalSelection?, requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        s.selection = settings.couplesSampledProposals ? selection : nil
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Muse Glimmer DFlash requires request-owned state")
    }

    /// Target hidden rows `[1, rows, contextWidth]` as feature rows; nil for
    /// any other shape.
    private func features(_ hidden: MLXArray) -> MLXArray? {
        guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) > 0,
            hidden.dim(2) == model.configuration.contextWidth
        else { return nil }
        return hidden.reshaped([hidden.dim(1), hidden.dim(2)])
    }

    /// Commit the rows of one observation as the next context positions.
    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        observe(observation, at: s.context.nextPosition, state: s)
    }

    /// The same, with the absolute position of the observation's first row.
    /// A position after the held rows restarts the context there; rows
    /// already held are skipped.
    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, startingAt position: Int,
        requestState: any CBv2MTPRequestState
    ) {
        observe(observation, at: position, state: state(requestState))
    }

    private func observe(
        _ observation: CBv2MTPCommittedTargetObservation, at position: Int, state s: State
    ) {
        s.speculation = nil
        guard s.refusal == nil else { return }
        guard s.stagedInputCount == 0, position >= 0, let rows = features(observation.hidden)
        else {
            s.refusal = "observation"
            return
        }
        if s.pending == nil {
            commitToContext(features: rows, at: position, to: s.context)
        } else {
            commitBatchToContext(
                takePending(s) + [.init(features: rows, position: position, state: s.context)])
        }
        // Whatever arrived, the context now ends at its newest row.
        s.awaitsCarry = false
        s.committedInputCount += rows.dim(0)
    }

    /// The state's pending rows as the first commit, at its next position.
    private func takePending(_ s: State) -> [FastContextCommit] {
        defer { s.pending = nil }
        return s.pending.map { [.init(features: $0, position: nil, state: s.context)] } ?? []
    }

    /// Every entry as `observeCommittedTarget` would take it, with one
    /// context commit for all of them.
    public func observeCommittedTargets(_ observations: [CBv2MTPRequestObservation]) {
        var commits: [FastContextCommit] = []
        var observed: [(state: State, rows: Int)] = []
        for entry in observations {
            let s = state(entry.requestState)
            s.speculation = nil
            guard s.refusal == nil else { continue }
            guard s.stagedInputCount == 0, entry.position.map({ $0 >= 0 }) ?? true,
                let rows = features(entry.observation.hidden)
            else {
                s.refusal = "observation"
                continue
            }
            commits += takePending(s)
            commits.append(.init(features: rows, position: entry.position, state: s.context))
            observed.append((s, rows.dim(0)))
        }
        commitBatchToContext(commits)
        for (s, rows) in observed {
            s.awaitsCarry = false
            s.committedInputCount += rows
        }
    }

    /// Draft index 0 commits a carry row that is still missing, runs one
    /// block forward and keeps its proposals. Later indices return them.
    ///
    /// A call this drafter cannot serve never stops the engine thread: it
    /// returns a well-formed proposal and the request ends with an error at
    /// the round's fence (`requestStateDidFinishEvaluation`).
    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let s = state(requestState)
        let c = model.configuration
        let index = s.stagedInputCount
        s.stagedInputCount += 1
        if s.refusal == nil, index == 0 {
            if s.prestarted {
                s.prestarted = false
            } else {
                s.refusal = startRound(s, tokens: tokens, hidden: hidden)
            }
        }
        guard s.refusal == nil, shortlist == nil, index < settings.maximumDraftTokens,
            let memo = s.memo, index < memo.tokens.dim(0)
        else {
            s.refusal = s.refusal ?? "draft step"
            s.memo = nil
            return (
                MLXArray([Int32(0)]),
                MLXArray.zeros([1, 1, c.hiddenSize], dtype: model.activationDType)
            )
        }
        return (
            memo.tokens[index ..< (index + 1)],
            memo.hidden[index ..< (index + 1), 0...].reshaped([1, 1, c.hiddenSize])
        )
    }

    /// Builds the round's block and memo. Returns why it could not.
    private func startRound(_ s: State, tokens: MLXArray, hidden: MLXArray) -> String? {
        let c = model.configuration
        let speculation = s.speculation
        s.speculation = nil
        guard tokens.shape == [1, 1] else { return "seed token" }
        let taken = speculation.map { takeSpeculation($0, state: s, hidden: hidden) } ?? .normal
        if case .drafted = taken { return nil }
        if case .committed = taken {
            // The pending rows and the carry are in the ring: draft the block.
        } else if settings.batchesRequests {
            guard let carry = commitCarry(s, hidden: hidden) else { return "carry" }
            if !carry.isEmpty { commitBatchToContext(carry) }
        } else if s.awaitsCarry {
            guard let row = features(hidden), row.dim(0) == 1 else { return "carry" }
            commitToContext(features: row, at: s.context.nextPosition, to: s.context)
            s.awaitsCarry = false
            s.committedInputCount += 1
        }
        let anchor = target.dflashTokenEmbeddings(tokens.asType(.int32))
        guard anchor.size == c.hiddenSize else { return "embedding" }
        let visible = s.context.visibleContext()
        let activeDraftCount =
            (CBv2DraftRegime.activeDynamicTreeK >= 2)
            ? min(settings.maximumDraftTokens, CBv2DraftRegime.activeDynamicTreeK)
            : settings.maximumDraftTokens
        let useBlockRingTail =
            Self.zeroCopyBlockTailEnabled && CBv2DraftRegime.lowAcceptance
            && CBv2DraftRegime.xnorDraftHeadEligible
            && s.context.canWriteBlockTail(at: s.context.count, rows: c.blockSize)
        let rows = fastDraftBlockRows(
            embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding),
            at: s.context.nextPosition,
            context: visible.map { ($0.keys, $0.values) }, mask: visible?.mask,
            draftCount: activeDraftCount,
            ringContext: useBlockRingTail ? s.context : nil,
            ringTailStart: useBlockRingTail ? s.context.count : 0)
        let dynK = CBv2DraftRegime.activeDynamicTreeK
        if s.selection == nil, CBv2DraftRegime.lowAcceptance, CBv2DraftRegime.xnorDraftHeadEligible,
            rows.ndim == 2, rows.dim(0) == activeDraftCount,
            rows.dim(0) <= 7,
            let comp = compressedDraftHead,
            let p = comp.dynamicTreePartials(rows, params: dynTreeParams)
        {
            if dynK >= 2, dynK <= rows.dim(0) {
                let dyn = Self.fastDynamicTreeStage2(
                    p.val, p.idx, p.sum, k: dynK, vocab: target.dflashVocabularySize,
                    params: dynTreeParams)
                CBv2DraftRegime.activeDynamicTreeMeta = dyn.treeMeta
                s.memo = (dyn.tokens, rows)
                return nil
            }
            if dynK == 0 && CBv2DraftRegime.activeVerifyTree == nil {
                let tokens = Self.fastDynamicTreeArgMax(
                    p.val, p.idx, vocab: target.dflashVocabularySize)
                s.memo = (tokens, rows)
                return nil
            }
        }
        let logits = draftHeadProduct(rows, readout: s.selection != nil)
        guard logits.shape == [activeDraftCount, target.dflashVocabularySize] else {
            return "head"
        }
        s.memo = (proposals(logits, selection: s.selection), rows)
        return nil
    }

    /// A round's commit before its block: the pending rows and the carry row
    /// still missing, as one entry at the next position. Nil when the carry
    /// is not one feature row.
    private func commitCarry(_ s: State, hidden: MLXArray) -> [FastContextCommit]? {
        var rows = s.pending.map { [$0] } ?? []
        s.pending = nil
        if s.awaitsCarry {
            guard let row = features(hidden), row.dim(0) == 1 else { return nil }
            rows.append(row)
            s.awaitsCarry = false
            s.committedInputCount += 1
        }
        guard !rows.isEmpty else { return [] }
        let features = rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
        return [.init(features: features, position: s.context.nextPosition, state: s.context)]
    }

    /// Draft index 0 of every seed's round in one pass: one context commit
    /// for every carry (and pending rows), one block forward and one head
    /// product for every block. A seed this cannot serve is left to its own
    /// index-0 call, which refuses it as before.
    public func startRounds(_ seeds: [CBv2MTPRoundSeed]) {
        guard settings.batchesRequests else { return }
        let c = model.configuration
        var started: [(state: State, tokens: MLXArray)] = []
        var commits: [FastContextCommit] = []
        for seed in seeds {
            let s = state(seed.requestState)
            s.speculation = nil
            guard s.refusal == nil, s.stagedInputCount == 0, !s.prestarted, seed.shortlist == nil,
                seed.tokens.shape == [1, 1]
            else { continue }
            guard let carry = commitCarry(s, hidden: seed.hidden) else {
                s.refusal = "carry"
                continue
            }
            commits += carry
            started.append((s, seed.tokens))
        }
        guard !started.isEmpty else { return }
        commitBatchToContext(commits)
        var blocks: [MLXArray] = []
        for (s, tokens) in started {
            let anchor = target.dflashTokenEmbeddings(tokens.asType(.int32))
            guard anchor.size == c.hiddenSize else {
                s.refusal = "embedding"
                continue
            }
            blocks.append(model.blockEmbeddings(anchor: anchor, mask: maskEmbedding))
        }
        let states = started.map(\.state).filter { $0.refusal == nil }
        guard !states.isEmpty else { return }
        let k = settings.maximumDraftTokens
        let rows: [MLXArray]
        if states.count == 1 {
            let ctx = states[0].context
            let visible = ctx.visibleContext()
            rows = [
                fastDraftBlockRows(
                    embeddings: blocks[0], at: ctx.nextPosition,
                    context: visible.map { ($0.keys, $0.values) }, mask: visible?.mask,
                    draftCount: k)
            ]
        } else {
            let visibles = states.map { $0.context.visibleContext() }
            rows = model.blockForwards(
                embeddings: blocks, at: states.map(\.context.nextPosition),
                contexts: visibles.map { $0.map { ($0.keys, $0.values) } },
                masks: visibles.map { $0?.mask }
            ).map { $0[1 ..< (1 + k), 0...] }
        }
        let logits = draftHeadProduct(
            rows.count == 1 ? rows[0] : concatenated(rows, axis: 0),
            readout: states.contains { $0.selection != nil })
        guard logits.shape == [states.count * k, target.dflashVocabularySize] else {
            for s in states { s.refusal = "head" }
            return
        }
        for (index, s) in states.enumerated() {
            s.memo = (
                proposals(logits[(index * k) ..< ((index + 1) * k), 0...], selection: s.selection),
                rows[index]
            )
            s.prestarted = true
        }
    }

    // MARK: Post-verify speculation

    /// The served round shape the speculation reproduces: batched requests,
    /// argmax proposals.
    var speculatesAfterVerify: Bool {
        settings.batchesRequests && !settings.couplesSampledProposals
    }

    private let contextKeyEpilogue: ContextKeyEpilogue?
    private let dynTreeParams: MLXArray

    private static let contextKeysKernel = MLXFast.metalKernel(
        name: "dflash_context_keys",
        inputNames: [
            "k0", "k1", "k2", "k3", "k4", "w0", "w1", "w2", "w3", "w4", "dims", "params",
            "tree_meta",
        ],
        outputNames: ["keys"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint rows = uint(dims[0]);
            const uint row = group % rows;
            const uint head = (group / rows) % HEADS_;
            const uint layer = group / (rows * HEADS_);
            const device bfloat16_t* src;
            const device bfloat16_t* weight;
            switch (layer) {
                case 0: src = k0; weight = w0; break;
                case 1: src = k1; weight = w1; break;
                case 2: src = k2; weight = w2; break;
                case 3: src = k3; weight = w3; break;
                default: src = k4; weight = w4; break;
            }
            src += (size_t(row) * HEADS_ + head) * 128 + lane * 4;
            weight += lane * 4;
            device bfloat16_t* dst = keys + size_t(group) * 128 + lane * 4;

            // Match rms_single_row: four adjacent channels per lane, then two SIMD sums.
            float x[4];
            float sum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                x[i] = src[i];
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            sum = simd_sum(lane == 0 ? sum : 0.0f);
            const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
            const bool low = lane < 16;
            const int row_delta = DYN_TREE_ ? tree_meta[9 + row] : int(row);
            const float pos = float(dims[1] + row_delta);
            for (uint i = 0; i < 4; ++i) {
                const bfloat16_t norm = weight[i] * static_cast<bfloat16_t>(x[i] * inv);
                const float v = static_cast<float>(norm);
                const float other = simd_shuffle_xor(v, ushort(16));
                const float d = float((lane % 16) * 4 + i) / 64.0f;
                const float theta = pos * metal::exp2(-d * params[1]);
                const float c = metal::fast::cos(theta);
                const float s = metal::fast::sin(theta);
                const float x1 = low ? v : other;
                const float x2 = low ? other : v;
                dst[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
            }
            """)

    private static let contextKVFusedKernel = MLXFast.metalKernel(
        name: "dflash_context_kv_fused",
        inputNames: [
            "kv_all", "w0", "w1", "w2", "w3", "w4", "dims", "params", "tree_meta",
        ],
        outputNames: ["keys", "values"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint rows = uint(dims[0]);
            const uint row = group % rows;
            const uint head = (group / rows) % HEADS_;
            const uint layer = group / (rows * HEADS_);
            const device bfloat16_t* weight;
            switch (layer) {
                case 0: weight = w0; break;
                case 1: weight = w1; break;
                case 2: weight = w2; break;
                case 3: weight = w3; break;
                default: weight = w4; break;
            }
            const size_t row_stride = size_t(10u * HEADS_) * 128u;
            const device bfloat16_t* ksrc =
                kv_all + size_t(row) * row_stride + size_t(layer * HEADS_ + head) * 128u + lane * 4u;
            const device bfloat16_t* vsrc =
                kv_all + size_t(row) * row_stride + size_t((5u + layer) * HEADS_ + head) * 128u + lane * 4u;
            const vec<bfloat16_t, 4> wv = *(const device vec<bfloat16_t, 4>*)(weight + lane * 4u);
            *(device vec<bfloat16_t, 4>*)(values + size_t(group) * 128u + lane * 4u) =
                *(const device vec<bfloat16_t, 4>*)vsrc;

            const vec<bfloat16_t, 4> kv4 = *(const device vec<bfloat16_t, 4>*)ksrc;
            float x[4];
            float sum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                x[i] = float(kv4[i]);
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            sum = simd_sum(lane == 0 ? sum : 0.0f);
            const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
            const bool low = lane < 16;
            const int row_delta = DYN_TREE_ ? tree_meta[9 + row] : int(row);
            const float pos = float(dims[1] + row_delta);
            vec<bfloat16_t, 4> out4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const bfloat16_t norm = wv[i] * static_cast<bfloat16_t>(x[i] * inv);
                const float v = static_cast<float>(norm);
                const float other = simd_shuffle_xor(v, ushort(16));
                const float d = float((lane % 16) * 4 + i) / 64.0f;
                const float theta = pos * metal::exp2(-d * params[1]);
                const float c = metal::fast::cos(theta);
                const float s = metal::fast::sin(theta);
                const float x1 = low ? v : other;
                const float x2 = low ? other : v;
                out4[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
            }
            *(device vec<bfloat16_t, 4>*)(keys + size_t(group) * 128u + lane * 4u) = out4;
            """)

    private static let blockQKVEpilogueKernel = MLXFast.metalKernel(
        name: "dflash_block_qkv_epilogue",
        inputNames: ["qkv", "qw", "kw", "pos_in", "params"],
        outputNames: ["q_out", "k_out", "v_out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint b_idx = threadgroup_position_in_grid.z;
            const uint row = group % uint(B_);
            const uint h_all = group / uint(B_);
            const uint total_h = uint(HQ_ + 2 * HKV_);
            const device bfloat16_t* src =
                qkv + ((size_t(b_idx) * size_t(B_) + size_t(row)) * size_t(total_h) + size_t(h_all)) * 128u + lane * 4u;
            if (h_all >= uint(HQ_ + HKV_)) {
                const uint vh = h_all - uint(HQ_ + HKV_);
                device bfloat16_t* dst =
                    v_out + ((size_t(b_idx) * size_t(HKV_) + size_t(vh)) * size_t(B_) + size_t(row)) * 128u + lane * 4u;
                *(device vec<bfloat16_t, 4>*)dst = *(const device vec<bfloat16_t, 4>*)src;
                return;
            }
            const bool is_q = h_all < uint(HQ_);
            const uint h_idx = is_q ? h_all : (h_all - uint(HQ_));
            const vec<bfloat16_t, 4> wv = *(const device vec<bfloat16_t, 4>*)((is_q ? qw : kw) + lane * 4u);
            device bfloat16_t* dst =
                (is_q
                    ? q_out + ((size_t(b_idx) * size_t(HQ_) + size_t(h_idx)) * size_t(B_) + size_t(row)) * 128u
                    : k_out + ((size_t(b_idx) * size_t(HKV_) + size_t(h_idx)) * size_t(B_) + size_t(row)) * 128u)
                + lane * 4u;
            const vec<bfloat16_t, 4> sv = *(const device vec<bfloat16_t, 4>*)src;
            float x[4];
            float sum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                x[i] = float(sv[i]);
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            sum = simd_sum(lane == 0 ? sum : 0.0f);
            const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
            const bool low = lane < 16;
            const float pos = float(pos_in[0] + int(row));
            vec<bfloat16_t, 4> out4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const bfloat16_t norm = wv[i] * static_cast<bfloat16_t>(x[i] * inv);
                const float v = static_cast<float>(norm);
                const float other = simd_shuffle_xor(v, ushort(16));
                const float d = float((lane % 16) * 4 + i) / 64.0f;
                const float theta = pos * metal::exp2(-d * params[1]);
                const float c = metal::fast::cos(theta);
                const float s = metal::fast::sin(theta);
                const float x1 = low ? v : other;
                const float x2 = low ? other : v;
                out4[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
            }
            *(device vec<bfloat16_t, 4>*)dst = out4;
            """)

    private static let blockKVEpilogueKernel = MLXFast.metalKernel(
        name: "dflash_block_kv_epilogue",
        inputNames: ["kv", "kw", "pos_in", "params"],
        outputNames: ["k_out", "v_out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint b_idx = threadgroup_position_in_grid.z;
            const uint row = group % uint(B_);
            const uint h_all = group / uint(B_);
            const uint total_h = uint(2 * HKV_);
            const device bfloat16_t* src =
                kv + ((size_t(b_idx) * size_t(B_) + size_t(row)) * size_t(total_h) + size_t(h_all)) * 128u + lane * 4u;
            if (h_all >= uint(HKV_)) {
                const uint vh = h_all - uint(HKV_);
                device bfloat16_t* dst =
                    v_out + ((size_t(b_idx) * size_t(HKV_) + size_t(vh)) * size_t(B_) + size_t(row)) * 128u + lane * 4u;
                *(device vec<bfloat16_t, 4>*)dst = *(const device vec<bfloat16_t, 4>*)src;
                return;
            }
            const vec<bfloat16_t, 4> wv = *(const device vec<bfloat16_t, 4>*)(kw + lane * 4u);
            device bfloat16_t* dst =
                k_out + ((size_t(b_idx) * size_t(HKV_) + size_t(h_all)) * size_t(B_) + size_t(row)) * 128u + lane * 4u;
            const vec<bfloat16_t, 4> sv = *(const device vec<bfloat16_t, 4>*)src;
            float x[4];
            float sum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                x[i] = float(sv[i]);
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            sum = simd_sum(lane == 0 ? sum : 0.0f);
            const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
            const bool low = lane < 16;
            const float pos = float(pos_in[0] + int(row));
            vec<bfloat16_t, 4> out4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const bfloat16_t norm = wv[i] * static_cast<bfloat16_t>(x[i] * inv);
                const float v = static_cast<float>(norm);
                const float other = simd_shuffle_xor(v, ushort(16));
                const float d = float((lane % 16) * 4 + i) / 64.0f;
                const float theta = pos * metal::exp2(-d * params[1]);
                const float c = metal::fast::cos(theta);
                const float s = metal::fast::sin(theta);
                const float x1 = low ? v : other;
                const float x2 = low ? other : v;
                out4[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
            }
            *(device vec<bfloat16_t, 4>*)dst = out4;
            """)

    private static let addRmsNormKernel = MLXFast.metalKernel(
        name: "dflash_add_rmsnorm",
        inputNames: ["res", "delta", "w", "params"],
        outputNames: ["h_out", "y_out"],
        source: """
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint row = threadgroup_position_in_grid.y;
            constexpr uint D4 = uint(D_) / 4u;
            constexpr uint MAX_IT = (D4 + 255u) / 256u;
            const device vec<bfloat16_t, 4>* r4 =
                (const device vec<bfloat16_t, 4>*)(res + size_t(row) * D_);
            const device vec<bfloat16_t, 4>* d4 =
                (const device vec<bfloat16_t, 4>*)(delta + size_t(row) * D_);
            const device vec<bfloat16_t, 4>* w4 = (const device vec<bfloat16_t, 4>*)w;
            device vec<bfloat16_t, 4>* h4 =
                (device vec<bfloat16_t, 4>*)(h_out + size_t(row) * D_);
            device vec<bfloat16_t, 4>* y4 =
                (device vec<bfloat16_t, 4>*)(y_out + size_t(row) * D_);
            float buf[MAX_IT][4];
            float sum = 0.0f;
            uint it = 0u;
            for (uint i = lid; i < D4; i += 256u) {
                const vec<bfloat16_t, 4> rv = r4[i];
                const vec<bfloat16_t, 4> dv = d4[i];
                vec<bfloat16_t, 4> hv;
                #pragma unroll
                for (uint k = 0; k < 4; ++k) {
                    const bfloat16_t dval = SCALE_DELTA_
                        ? static_cast<bfloat16_t>(float(dv[k]) * params[1]) : dv[k];
                    const bfloat16_t h = rv[k] + dval;
                    hv[k] = h;
                    const float fv = float(h);
                    buf[it][k] = fv;
                    sum += fv * fv;
                }
                h4[i] = hv;
                it++;
            }
            sum = simd_sum(sum);
            threadgroup float sh[8];
            if (lane == 0u) { sh[sg] = sum; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float v = (lane < 8u) ? sh[lane] : 0.0f;
                v = simd_sum(v);
                if (lane == 0u) { sh[0] = metal::precise::rsqrt(v / float(D_) + params[0]); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = sh[0];
            it = 0u;
            for (uint i = lid; i < D4; i += 256u) {
                const vec<bfloat16_t, 4> wv = w4[i];
                vec<bfloat16_t, 4> yv;
                #pragma unroll
                for (uint k = 0; k < 4; ++k) {
                    yv[k] = wv[k] * static_cast<bfloat16_t>(buf[it][k] * inv);
                }
                y4[i] = yv;
                it++;
            }
            """,
        ensureRowContiguous: true)

    private static let splitSiluMulKernel = MLXFast.metalKernel(
        name: "dflash_split_silu_mul",
        inputNames: ["gu", "params"],
        outputNames: ["y"],
        source: """
            const uint col4 = thread_position_in_grid.x;
            if (col4 >= uint(M4_)) { return; }
            const uint row = threadgroup_position_in_grid.y;
            const device vec<bfloat16_t, 4>* g4 = (const device vec<bfloat16_t, 4>*)gu;
            device vec<bfloat16_t, 4>* y4 = (device vec<bfloat16_t, 4>*)y;
            const size_t base = size_t(row) * size_t(2u * M4_) + size_t(col4);
            const vec<bfloat16_t, 4> gv = g4[base];
            const vec<bfloat16_t, 4> uv = g4[base + size_t(M4_)];
            vec<bfloat16_t, 4> out;
            const bfloat16_t one = bfloat16_t(1.0f);
            const float sc = SCALE_ ? params[0] : 1.0f;
            #pragma unroll
            for (uint k = 0; k < 4; ++k) {
                const bfloat16_t g = gv[k];
                const bfloat16_t ax = (g < bfloat16_t(0.0f)) ? -g : g;
                const bfloat16_t ey =
                    static_cast<bfloat16_t>(metal::precise::exp(static_cast<float>(ax)));
                const bfloat16_t s = one / (one + ey);
                const bfloat16_t sig = (g < bfloat16_t(0.0f)) ? s : (one - s);
                const bfloat16_t prod = (g * sig) * uv[k];
                out[k] = SCALE_ ? static_cast<bfloat16_t>(float(prod) * sc) : prod;
            }
            y4[size_t(row) * size_t(M4_) + size_t(col4)] = out;
            """,
        ensureRowContiguous: true)

    private static let zeroCopyBlockTailEnabled: Bool =
        ProcessInfo.processInfo.environment["CBV2_DFLASH_ZEROCOPY_BLK"] != "0"

    private static let blockQEpilogueKernel = MLXFast.metalKernel(
        name: "dflash_block_q_epilogue",
        inputNames: ["q_in", "qw", "pos_in", "params"],
        outputNames: ["q_out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint row = group % uint(R_);
            const uint head = group / uint(R_);
            const device bfloat16_t* src =
                q_in + (size_t(row) * size_t(HQ_) + size_t(head)) * 128u + lane * 4u;
            const vec<bfloat16_t, 4> wv = *(const device vec<bfloat16_t, 4>*)(qw + lane * 4u);
            const vec<bfloat16_t, 4> sv = *(const device vec<bfloat16_t, 4>*)src;
            float x[4];
            float sum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                x[i] = float(sv[i]);
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            sum = simd_sum(lane == 0 ? sum : 0.0f);
            const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
            const bool low = lane < 16;
            const float pos = float(pos_in[0] + 1 + int(row));
            vec<bfloat16_t, 4> out4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const bfloat16_t norm = wv[i] * static_cast<bfloat16_t>(x[i] * inv);
                const float v = static_cast<float>(norm);
                const float other = simd_shuffle_xor(v, ushort(16));
                const float d = float((lane % 16) * 4 + i) / 64.0f;
                const float theta = pos * metal::exp2(-d * params[1]);
                const float c = metal::fast::cos(theta);
                const float s = metal::fast::sin(theta);
                const float x1 = low ? v : other;
                const float x2 = low ? other : v;
                out4[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
            }
            device bfloat16_t* dst =
                q_out + (size_t(head) * size_t(R_) + size_t(row)) * 128u + lane * 4u;
            *(device vec<bfloat16_t, 4>*)dst = out4;
            """,
        ensureRowContiguous: true)

    private static let blockQKVRingKernel = MLXFast.metalKernel(
        name: "dflash_block_qkv_ring",
        inputNames: [
            "qkv", "qw", "kw", "ring_keys", "ring_values", "pos_in", "params", "iparams",
            "prev_fence",
        ],
        outputNames: ["q_out", "fence"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint row = group % uint(B_);
            const uint h_all = group / uint(B_);
            const size_t row_stride = size_t(HQ_ + 2 * HKV_) * 128u;
            const bool low = lane < 16;
            const float pos = float(pos_in[0] + int(row));

            if (h_all < uint(HQ_)) {
                const device bfloat16_t* src =
                    qkv + size_t(row) * row_stride + size_t(h_all) * 128u + lane * 4u;
                const vec<bfloat16_t, 4> wv = *(const device vec<bfloat16_t, 4>*)(qw + lane * 4u);
                const vec<bfloat16_t, 4> sv = *(const device vec<bfloat16_t, 4>*)src;
                float x[4];
                float sum = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    x[i] = float(sv[i]);
                    sum += x[i] * x[i];
                }
                sum = simd_sum(sum);
                sum = simd_sum(lane == 0 ? sum : 0.0f);
                const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
                vec<bfloat16_t, 4> out4;
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const bfloat16_t norm = wv[i] * static_cast<bfloat16_t>(x[i] * inv);
                    const float v = static_cast<float>(norm);
                    const float other = simd_shuffle_xor(v, ushort(16));
                    const float d = float((lane % 16) * 4 + i) / 64.0f;
                    const float theta = pos * metal::exp2(-d * params[1]);
                    const float c = metal::fast::cos(theta);
                    const float s = metal::fast::sin(theta);
                    const float x1 = low ? v : other;
                    const float x2 = low ? other : v;
                    out4[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
                }
                device bfloat16_t* dst =
                    q_out + (size_t(h_all) * size_t(B_) + size_t(row)) * 128u + lane * 4u;
                *(device vec<bfloat16_t, 4>*)dst = out4;
            } else {
                const uint kv_h = h_all - uint(HQ_);
                const device bfloat16_t* src_k =
                    qkv + size_t(row) * row_stride + size_t(HQ_ + kv_h) * 128u + lane * 4u;
                const device bfloat16_t* src_v =
                    qkv + size_t(row) * row_stride + size_t(HQ_ + HKV_ + kv_h) * 128u + lane * 4u;
                const vec<bfloat16_t, 4> wv = *(const device vec<bfloat16_t, 4>*)(kw + lane * 4u);
                const uint layer = uint(iparams[0]);
                const uint dst_slot = uint(iparams[1]) + row;
                const size_t ring_off =
                    ((size_t(layer) * size_t(HKV_) + size_t(kv_h)) * size_t(W_)
                        + size_t(dst_slot)) * 128u + lane * 4u;
                device bfloat16_t* rk = ring_keys;
                device bfloat16_t* rv = ring_values;
                *(device vec<bfloat16_t, 4>*)(rv + ring_off) = *(const device vec<bfloat16_t, 4>*)src_v;
                const vec<bfloat16_t, 4> sk = *(const device vec<bfloat16_t, 4>*)src_k;
                float x[4];
                float sum = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    x[i] = float(sk[i]);
                    sum += x[i] * x[i];
                }
                sum = simd_sum(sum);
                sum = simd_sum(lane == 0 ? sum : 0.0f);
                const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
                vec<bfloat16_t, 4> out4;
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const bfloat16_t norm = wv[i] * static_cast<bfloat16_t>(x[i] * inv);
                    const float v = static_cast<float>(norm);
                    const float other = simd_shuffle_xor(v, ushort(16));
                    const float d = float((lane % 16) * 4 + i) / 64.0f;
                    const float theta = pos * metal::exp2(-d * params[1]);
                    const float c = metal::fast::cos(theta);
                    const float s = metal::fast::sin(theta);
                    const float x1 = low ? v : other;
                    const float x2 = low ? other : v;
                    out4[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
                }
                *(device vec<bfloat16_t, 4>*)(rk + ring_off) = out4;
            }
            if (lane == 0u && group == 0u) {
                fence[0] = prev_fence[0] + 1;
            }
            """,
        ensureRowContiguous: false,
        mutableInputs: ["ring_keys", "ring_values"])

    private static let blockKVRingKernel = MLXFast.metalKernel(
        name: "dflash_block_kv_ring",
        inputNames: [
            "kv", "kw", "ring_keys", "ring_values", "pos_in", "params", "iparams", "prev_fence",
        ],
        outputNames: ["fence"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint group = threadgroup_position_in_grid.y;
            const uint row = group % uint(B_);
            const uint kv_h = group / uint(B_);
            const size_t row_stride = size_t(2 * HKV_) * 128u;
            const device bfloat16_t* src_k =
                kv + size_t(row) * row_stride + size_t(kv_h) * 128u + lane * 4u;
            const device bfloat16_t* src_v =
                kv + size_t(row) * row_stride + size_t(HKV_ + kv_h) * 128u + lane * 4u;
            const vec<bfloat16_t, 4> wv = *(const device vec<bfloat16_t, 4>*)(kw + lane * 4u);
            const uint layer = uint(iparams[0]);
            const uint dst_slot = uint(iparams[1]) + row;
            const size_t ring_off =
                ((size_t(layer) * size_t(HKV_) + size_t(kv_h)) * size_t(W_)
                    + size_t(dst_slot)) * 128u + lane * 4u;
            device bfloat16_t* rk = ring_keys;
            device bfloat16_t* rv = ring_values;
            *(device vec<bfloat16_t, 4>*)(rv + ring_off) = *(const device vec<bfloat16_t, 4>*)src_v;
            const vec<bfloat16_t, 4> sk = *(const device vec<bfloat16_t, 4>*)src_k;
            float x[4];
            float sum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                x[i] = float(sk[i]);
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            sum = simd_sum(lane == 0 ? sum : 0.0f);
            const float inv = metal::precise::rsqrt(sum / 128u + params[0]);
            const bool low = lane < 16;
            const float pos = float(pos_in[0] + int(row));
            vec<bfloat16_t, 4> out4;
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const bfloat16_t norm = wv[i] * static_cast<bfloat16_t>(x[i] * inv);
                const float v = static_cast<float>(norm);
                const float other = simd_shuffle_xor(v, ushort(16));
                const float d = float((lane % 16) * 4 + i) / 64.0f;
                const float theta = pos * metal::exp2(-d * params[1]);
                const float c = metal::fast::cos(theta);
                const float s = metal::fast::sin(theta);
                const float x1 = low ? v : other;
                const float x2 = low ? other : v;
                out4[i] = static_cast<bfloat16_t>(low ? x1 * c - x2 * s : x1 * s + x2 * c);
            }
            *(device vec<bfloat16_t, 4>*)(rk + ring_off) = out4;
            if (lane == 0u && group == 0u) {
                fence[0] = prev_fence[0] + 1;
            }
            """,
        ensureRowContiguous: false,
        mutableInputs: ["ring_keys", "ring_values"])

    private final class DrafterLaunchBox: @unchecked Sendable {
        var ctxKV: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var blkQKV: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var blkKV: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var blkQ: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var blkQKVRing: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var blkKVRing: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var addNorm: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var splitSilu: [UInt64: MLXFast.MLXFastKernel.Launch] = [:]
        var epsScaleArr: [UInt64: MLXArray] = [:]
        var scaleArr: [UInt32: MLXArray] = [:]
    }

    private static let drafterLaunchLock = OSAllocatedUnfairLock(initialState: DrafterLaunchBox())

    private static func preparedBlockQKVRing(
        qHeads: Int, kvHeads: Int, block: Int, window: Int
    ) -> MLXFast.MLXFastKernel.Launch {
        let key =
            (UInt64(qHeads) << 40) | (UInt64(kvHeads) << 28) | (UInt64(block) << 16) | UInt64(window)
        return drafterLaunchLock.withLockUnchecked { box in
            if let hit = box.blkQKVRing[key] { return hit }
            let made = blockQKVRingKernel.prepare(
                template: [
                    ("HQ_", qHeads), ("HKV_", kvHeads), ("B_", block), ("W_", window),
                ],
                grid: (32, (qHeads + kvHeads) * block, 1), threadGroup: (32, 1, 1),
                outputShapes: [[1, qHeads, block, 128], [1]],
                outputDTypes: [.bfloat16, .int32])
            box.blkQKVRing[key] = made
            return made
        }
    }

    private static func preparedBlockKVRing(
        kvHeads: Int, block: Int, window: Int
    ) -> MLXFast.MLXFastKernel.Launch {
        let key = (UInt64(kvHeads) << 28) | (UInt64(block) << 16) | UInt64(window)
        return drafterLaunchLock.withLockUnchecked { box in
            if let hit = box.blkKVRing[key] { return hit }
            let made = blockKVRingKernel.prepare(
                template: [("HKV_", kvHeads), ("B_", block), ("W_", window)],
                grid: (32, kvHeads * block, 1), threadGroup: (32, 1, 1),
                outputShapes: [[1]],
                outputDTypes: [.int32])
            box.blkKVRing[key] = made
            return made
        }
    }

    private static func applyBlockQEpilogue(
        _ qIn: MLXArray, qNormWeight: MLXArray, posScalar: MLXArray, coefficients: MLXArray,
        qHeads: Int, rows: Int
    ) -> MLXArray {
        let key = (UInt64(qHeads) << 16) | UInt64(rows)
        let launch: MLXFast.MLXFastKernel.Launch = drafterLaunchLock.withLockUnchecked { box in
            if let hit = box.blkQ[key] { return hit }
            let made = blockQEpilogueKernel.prepare(
                template: [("HQ_", qHeads), ("R_", rows)],
                grid: (32, qHeads * rows, 1), threadGroup: (32, 1, 1),
                outputShapes: [[1, qHeads, rows, 128]],
                outputDTypes: [.bfloat16])
            box.blkQ[key] = made
            return made
        }
        return launch([qIn, qNormWeight, posScalar, coefficients])[0]
    }

    private static func fusedAddRMSNorm(
        residual: MLXArray, delta: MLXArray, weight: MLXArray, eps: Float,
        deltaScale: Float = 1.0
    ) -> (h: MLXArray, y: MLXArray) {
        let rows = residual.dim(1)
        let d = residual.dim(2)
        let scaled = deltaScale != 1.0
        let key = (UInt64(rows) << 33) | (UInt64(d) << 1) | (scaled ? 1 : 0)
        let pKey = (UInt64(eps.bitPattern) << 32) | UInt64(deltaScale.bitPattern)
        let (launch, paramsArr): (MLXFast.MLXFastKernel.Launch, MLXArray) =
            drafterLaunchLock.withLockUnchecked { box in
                let p: MLXArray
                if let hit = box.epsScaleArr[pKey] {
                    p = hit
                } else {
                    let a = MLXArray([eps, deltaScale])
                    eval(a)
                    box.epsScaleArr[pKey] = a
                    p = a
                }
                if let hit = box.addNorm[key] { return (hit, p) }
                let made = addRmsNormKernel.prepare(
                    template: [("D_", d), ("SCALE_DELTA_", scaled)],
                    grid: (256, rows, 1), threadGroup: (256, 1, 1),
                    outputShapes: [[1, rows, d], [1, rows, d]],
                    outputDTypes: [.bfloat16, .bfloat16])
                box.addNorm[key] = made
                return (made, p)
            }
        let out = launch([residual, delta, weight, paramsArr])
        return (out[0], out[1])
    }

    private static func splitSiluMul(_ gu: MLXArray, scale: Float = 1.0) -> MLXArray {
        let rows = gu.dim(1)
        let mid = gu.dim(2) / 2
        let mid4 = mid / 4
        let scaled = scale != 1.0
        let key = (UInt64(rows) << 33) | (UInt64(mid4) << 1) | (scaled ? 1 : 0)
        let (launch, paramsArr): (MLXFast.MLXFastKernel.Launch, MLXArray) =
            drafterLaunchLock.withLockUnchecked { box in
                let p: MLXArray
                if let hit = box.scaleArr[scale.bitPattern] {
                    p = hit
                } else {
                    let a = MLXArray([scale])
                    eval(a)
                    box.scaleArr[scale.bitPattern] = a
                    p = a
                }
                if let hit = box.splitSilu[key] { return (hit, p) }
                let tg = 256
                let gridX = ((mid4 + tg - 1) / tg) * tg
                let made = splitSiluMulKernel.prepare(
                    template: [("M4_", mid4), ("SCALE_", scaled)],
                    grid: (gridX, rows, 1), threadGroup: (tg, 1, 1),
                    outputShapes: [[1, rows, mid]], outputDTypes: [.bfloat16])
                box.splitSilu[key] = made
                return (made, p)
            }
        return launch([gu, paramsArr])[0]
    }

    private struct ContextKeyEpilogue {
        let kvHeads: Int
        let weights: [MLXArray]
        let coefficients: MLXArray
        let dummyTreeMeta: MLXArray

        init?(_ model: MuseGlimmerDFlashDraftModel) {
            let c = model.configuration
            guard model.layers.count == 5, c.headDimensions == 128,
                model.activationDType == .bfloat16, Device.defaultDevice().deviceType == .gpu
            else { return nil }
            var weights: [MLXArray] = []
            weights.reserveCapacity(5)
            for layer in model.layers {
                let norm = layer.attention.kNorm
                guard norm.weight.dtype == .bfloat16, norm.weight.shape == [128],
                    norm.eps == c.rmsNormEps
                else { return nil }
                weights.append(norm.weight)
            }
            self.kvHeads = c.kvHeads
            self.weights = weights
            self.coefficients = MLXArray([c.rmsNormEps, log2(c.ropeTheta)])
            let dm = MLXArray.zeros([18], type: Int32.self)
            eval(dm)
            self.dummyTreeMeta = dm
        }

        func apply(
            _ projections: [MLXArray], rows: Int, position: Int, treeMeta: MLXArray? = nil
        ) -> MLXArray {
            var inputs = projections
            inputs.reserveCapacity(13)
            inputs.append(contentsOf: weights)
            inputs.append(MLXArray([Int32(rows), Int32(position)]))
            inputs.append(coefficients)
            inputs.append(treeMeta ?? dummyTreeMeta)
            return MuseGlimmerDFlashDrafter.contextKeysKernel(
                inputs, template: [("HEADS_", kvHeads), ("DYN_TREE_", treeMeta != nil)],
                grid: (32, 5 * kvHeads * rows, 1), threadGroup: (32, 1, 1),
                outputShapes: [[5, kvHeads, rows, 128]], outputDTypes: [.bfloat16])[0]
        }

        func applyFusedKV(
            _ allKV: MLXArray, rows: Int, position: Int, treeMeta: MLXArray? = nil
        ) -> (keys: MLXArray, values: MLXArray) {
            var inputs: [MLXArray] = [allKV]
            inputs.reserveCapacity(9)
            inputs.append(contentsOf: weights)
            inputs.append(MLXArray([Int32(rows), Int32(position)]))
            inputs.append(coefficients)
            inputs.append(treeMeta ?? dummyTreeMeta)
            let dyn = treeMeta != nil
            let key = (UInt64(kvHeads) << 33) | (UInt64(rows) << 1) | (dyn ? 1 : 0)
            let launch: MLXFast.MLXFastKernel.Launch =
                MuseGlimmerDFlashDrafter.drafterLaunchLock.withLockUnchecked { box in
                    if let hit = box.ctxKV[key] { return hit }
                    let made = MuseGlimmerDFlashDrafter.contextKVFusedKernel.prepare(
                        template: [("HEADS_", kvHeads), ("DYN_TREE_", dyn)],
                        grid: (32, 5 * kvHeads * rows, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[5, kvHeads, rows, 128], [5, kvHeads, rows, 128]],
                        outputDTypes: [.bfloat16, .bfloat16])
                    box.ctxKV[key] = made
                    return made
                }
            let out = launch(inputs)
            return (out[0], out[1])
        }
    }

    private func speculativeProjectAndContextKV(
        _ features: MLXArray, at position: Int, treeMeta: MLXArray? = nil
    ) -> (keys: MLXArray, values: MLXArray) {
        let rows = features.dim(0)
        if CBv2DraftRegime.xnorDraftHeadEligible, let nv = nvfp4Drafter,
            let epilogue = contextKeyEpilogue, rows > 0, position >= 0,
            position <= Int(Int32.max) - rows
        {
            let act = features.dtype == .bfloat16 ? features : features.asType(.bfloat16)
            let fcOut = nv.fc(act)
            let ctx = MLXFast.rmsNorm(
                fcOut, weight: nv.encOutputNormWeightScaled, eps: nv.fcEps)
            let allKV = nv.fusedContextKV(ctx)
            if allKV.dtype == .bfloat16 {
                return epilogue.applyFusedKV(allKV, rows: rows, position: position, treeMeta: treeMeta)
            }
        }
        return speculativeContextKeyValues(
            model.projectContext(features), at: position, treeMeta: treeMeta)
    }

    private func speculativeContextKeyValues(
        _ context: MLXArray, at position: Int, treeMeta: MLXArray? = nil
    ) -> (keys: MLXArray, values: MLXArray) {
        let c = model.configuration
        let rows = context.dim(0)
        guard let epilogue = contextKeyEpilogue, context.dtype == .bfloat16, rows > 0,
            position >= 0, position <= Int(Int32.max) - rows
        else { return model.contextKeyValues(context, at: position) }
        if Self.fusedProjEnabled
            && (CBv2DraftRegime.lowAcceptance || CBv2DraftRegime.xnorDraftHeadEligible),
            let fusedContextKV
        {
            let allKV = fusedContextKV(context)
            if allKV.dtype == .bfloat16 {
                return epilogue.applyFusedKV(allKV, rows: rows, position: position, treeMeta: treeMeta)
            }
        }
        let shape = [1, rows, c.kvHeads, c.headDimensions]
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        keys.reserveCapacity(model.layers.count)
        values.reserveCapacity(model.layers.count)
        for layer in model.layers {
            keys.append(layer.attention.kProj(context))
            values.append(layer.attention.vProj(context).reshaped(shape).transposed(0, 2, 1, 3))
        }
        guard keys.allSatisfy({ $0.dtype == .bfloat16 }) else {
            return model.contextKeyValues(context, at: position)
        }
        return (
            epilogue.apply(keys, rows: rows, position: position, treeMeta: treeMeta),
            concatenated(values, axis: 0))
    }

    /// Builds the next round's index-0 work from the verify's lazy rows,
    /// behind the submitted verify, before the acceptance count is known.
    ///
    /// For every outcome the next index 0 first commits one entry at the
    /// context's next position: `finalizeRound`'s pending rows (verify rows
    /// `0 ..< a` for `a` accepted drafts) and the engine's carry (verify row
    /// `a`). Those are the first `a + 1` verify rows whatever `a` is, so this
    /// projects all `depth + 1` rows once (`model.commit`'s projection, no
    /// duplicates, no restart, nothing unread) and the next index 0 appends
    /// the first `a + 1` of them instead of projecting them on the critical
    /// path. Only the projection moves: the ring, the pending rows and the
    /// carry are untouched until `takeSpeculation` confirms the outcome.
    ///
    /// With `draftsBlock`, after a round that accepted every draft, it also
    /// drafts the next block for the all-accepted outcome, whose seed is the
    /// verify's bonus target `targetTokens[depth]`: the append it implies is
    /// written into new wrappers of the ring arrays (the slots `append`
    /// would write).
    ///
    /// Only for the served one-row batched argmax round.
    public func speculateNextRound(
        targetTokens: MLXArray, targetHidden: MLXArray, depth k: Int, draftsBlock: Bool,
        draftTokens: MLXArray?, requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        speculateNextRound(
            targetTokens: targetTokens, targetHidden: targetHidden, depth: k,
            draftsBlock: draftsBlock, draftTokens: draftTokens, dynTreeMeta: nil,
            requestState: requestState)
    }

    public func speculateNextRound(
        targetTokens: MLXArray, targetHidden: MLXArray, depth k: Int, draftsBlock: Bool,
        draftTokens: MLXArray?, dynTreeMeta: MLXArray?, requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        let s = state(requestState)
        s.speculation = nil
        let c = model.configuration
        let context = s.context
        let window = context.window
        let n = k + 1
        let ring = context.arrays
        guard speculatesAfterVerify, s.refusal == nil, s.memo != nil,
            s.selection == nil, !s.prestarted, s.pending == nil, !s.awaitsCarry, k >= 1,
            k == s.stagedInputCount,
            n <= window, context.count > 0, ring.count == 2,
            targetTokens.shape == [1, n], targetHidden.ndim == 3, targetHidden.dim(0) == 1,
            targetHidden.dim(1) == n, targetHidden.dim(2) == c.contextWidth
        else { return [] }

        // The verify rows as finalizeRound and the engine slice them; the
        // all-accepted outcome's entry is all of them.
        let features = targetHidden[0 ..< 1, 0 ..< n, 0...].reshaped([n, c.contextWidth])
        let start = context.nextPosition
        let projected = speculativeProjectAndContextKV(
            features, at: start, treeMeta: dynTreeMeta)
        var speculation = Speculation(
            depth: k, anchor: start, first: context.firstPosition, keys: projected.keys,
            values: projected.values)
        if let draftTokens, draftTokens.shape == [1, k],
            context.count + n + context.blockSize <= window
        {
            return speculateOutcomeBlock(
                s, speculation: speculation, targetTokens: targetTokens,
                draftTokens: draftTokens, depth: k, start: start, projected: projected,
                dynTreeMeta: dynTreeMeta)
        }
        guard draftsBlock, s.lastRoundFullyAccepted else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }

        let next = start + n
        let first = max(context.firstPosition, next - window)
        let count = next - first

        // startRound's block and memo from the bonus target as the seed.
        let seed = maximum(targetTokens[0 ..< 1, k ..< n].asType(.int32), MLXArray(Int32(0)))
        let anchor = target.dflashTokenEmbeddings(seed)
        guard anchor.size == c.hiddenSize else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }
        let rows: MLXArray
        if context.count + n <= window {
            // Fast path: ring has not wrapped yet — concatenate existing slice
            // and projected rows instead of copying the entire 40 MB ring.
            let existingCount = context.count
            let ctxKeys = concatenated([
                ring[0][0..., 0..., 0 ..< existingCount, 0...],
                projected.keys.asType(context.dtype),
            ], axis: 2)
            let ctxValues = concatenated([
                ring[1][0..., 0..., 0 ..< existingCount, 0...],
                projected.values.asType(context.dtype),
            ], axis: 2)
            let held = Array(first ..< next)
            let mask = MuseGlimmerDFlashContextState.visibility(
                anchor: next, blockSize: context.blockSize, window: window, contextPositions: held
            ).map { MLXArray($0, [context.blockSize, count + context.blockSize]) }
            rows = fastDraftBlockRows(
                embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding), at: next,
                context: (ctxKeys, ctxValues), mask: mask,
                draftCount: settings.maximumDraftTokens)
        } else {
            // context.append on new wrappers. Slot zero holds a position
            // congruent to the ring's base position modulo the window: the
            // first held one before the ring wraps, the base lap's slot zero
            // after (`heldPositions`).
            let slotZero = context.heldPositions[0]
            func slot(_ position: Int) -> Int { ((position - slotZero) % window + window) % window }
            let keys = ring[0].sharingArray()
            let values = ring[1].sharingArray()
            var written = 0
            while written < n {
                let at = slot(start + written)
                let run = min(n - written, window - at)
                let source = written ..< (written + run)
                keys[0..., 0..., at ..< (at + run), 0...] =
                    projected.keys[0..., 0..., source, 0...].asType(context.dtype)
                values[0..., 0..., at ..< (at + run), 0...] =
                    projected.values[0..., 0..., source, 0...].asType(context.dtype)
                written += run
            }
            // visibleContext of the appended ring.
            let held: [Int]
            if count == window {
                let oldest = slot(first)
                held = (0 ..< window).map { first + ($0 - oldest + window) % window }
            } else {
                held = Array(first ..< next)
            }
            let mask = MuseGlimmerDFlashContextState.visibility(
                anchor: next, blockSize: context.blockSize, window: window, contextPositions: held
            ).map { MLXArray($0, [context.blockSize, count + context.blockSize]) }
            rows = fastDraftBlockRows(
                embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding), at: next,
                context: (keys[0..., 0..., 0 ..< count, 0...], values[0..., 0..., 0 ..< count, 0...]),
                mask: mask, draftCount: settings.maximumDraftTokens)
        }
        // startRound's head: the speculated memo must equal the normal one.
        let logits = draftHeadProduct(rows, readout: false)
        guard logits.shape == [settings.maximumDraftTokens, target.dflashVocabularySize] else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }
        let memo: (tokens: MLXArray, hidden: MLXArray)
        let confidence: MLXArray
        if let textTarget = target as? MuseGlimmerTextModel, logits.dtype == .bfloat16 {
            let (tokens, conf) = Self.fusedArgMaxAndConfidence(
                logits,
                outputMultiplier: textTarget.configuration.outputMultiplier,
                softcap: textTarget.configuration.finalLogitSoftcapping)
            memo = (tokens: tokens, hidden: rows)
            confidence = conf
        } else {
            memo = (tokens: proposals(logits, selection: nil), hidden: rows)
            // Per-row top-1 probability of the proposal, softmax over the
            // target's readout scale: exp(max - logsumexp). Advisory only: it
            // sizes the next verify window and never changes a proposal.
            let scores = target.dflashReadout(logits).asType(.float32)
            confidence = exp(scores.max(axis: -1) - scores.logSumExp(axis: -1))
        }
        speculation.memo = memo
        speculation.confidence = confidence
        s.speculation = speculation
        return [memo.tokens, memo.hidden, projected.keys, projected.values, confidence]
    }

    /// Drafts the next block on the GPU for whichever acceptance count `0...k`
    /// the current verify round decides, masking rejected context rows `> a`
    /// and advancing block RoPE on the device with zero host synchronization.
    private func speculateOutcomeBlock(
        _ s: State, speculation inProgress: Speculation, targetTokens: MLXArray,
        draftTokens: MLXArray, depth k: Int, start: Int,
        projected: (keys: MLXArray, values: MLXArray),
        dynTreeMeta: MLXArray? = nil
    ) -> [MLXArray] {
        var speculation = inProgress
        let c = model.configuration
        let context = s.context
        let n = k + 1
        let ring = context.arrays
        let existingCount = context.count
        let block = context.blockSize
        let targets = targetTokens.asType(.int32)
        let accepted: MLXArray
        let carrySlot: MLXArray
        let projectedVisible: MLXArray
        if let dynTreeMeta {
            let res = Self.fastTreeOutcome(
                drafts: draftTokens.asType(.int32), targets: targets,
                treeMeta: dynTreeMeta, k: k)
            carrySlot = res.carrySlot
            accepted = res.accepted
            projectedVisible = res.visible
        } else {
            let slotIndices = MLXArray(Array(Int32(0) ..< Int32(n)))
            let match = equal(draftTokens.asType(.int32), targets[0..., 0 ..< k]).asType(.int32)
            let chainAccepted = cumprod(match, axis: 1).sum()
            accepted = chainAccepted
            carrySlot = chainAccepted.reshaped([1, 1])
            projectedVisible = lessEqual(slotIndices, chainAccepted)
        }
        let seed = maximum(
            takeAlong(targets, carrySlot, axis: 1), MLXArray(Int32(0)))
        let anchor = target.dflashTokenEmbeddings(seed)
        guard anchor.size == c.hiddenSize else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }
        let ctxKeys: MLXArray
        let ctxValues: MLXArray
        let usedZeroCopyTail: Bool
        if let tail = context.writeSpeculativeTail(keys: projected.keys, values: projected.values) {
            ctxKeys = tail.keys
            ctxValues = tail.values
            usedZeroCopyTail = true
            if !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.strictLowVerifyC3 {
                speculation.tailTicket = tail.ticket
            }
        } else {
            ctxKeys = concatenated([
                ring[0][0..., 0..., 0 ..< existingCount, 0...],
                projected.keys.asType(context.dtype),
            ], axis: 2)
            ctxValues = concatenated([
                ring[1][0..., 0..., 0 ..< existingCount, 0...],
                projected.values.asType(context.dtype),
            ], axis: 2)
            usedZeroCopyTail = false
        }
        let next = start + n
        let count = existingCount + n
        let columns = concatenated([
            MLXArray.ones([existingCount], type: Bool.self), projectedVisible,
            MLXArray.ones([block], type: Bool.self),
        ], axis: 0)
        let mask = broadcast(columns.reshaped([1, count + block]), to: [block, count + block])
        let position = MLXArray(Int32(start + 1)) + accepted
        let prevLow = CBv2DraftRegime.lowAcceptance
        CBv2DraftRegime.lowAcceptance = true
        defer { CBv2DraftRegime.lowAcceptance = prevLow }
        let useSpecBlockTail =
            usedZeroCopyTail && Self.zeroCopyBlockTailEnabled
            && CBv2DraftRegime.strictLowVerifyC3
            && context.canWriteSpeculativeBlockTail(at: existingCount + n, rows: block)
        let rows = fastDraftBlockRows(
            embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding), at: next,
            context: (ctxKeys, ctxValues), mask: mask,
            draftCount: settings.maximumDraftTokens, devicePosition: position,
            ringContext: useSpecBlockTail ? context : nil,
            ringTailStart: useSpecBlockTail ? (existingCount + n) : 0)
        if usedZeroCopyTail {
            context.recordSpeculativeRead(rows)
        }
        if CBv2DraftRegime.xnorDraftHeadEligible,
            CBv2DraftRegime.verifyTreeMode != 0, CBv2DraftRegime.dynamicTreeEnabled,
            let comp = compressedDraftHead,
            let p = comp.dynamicTreePartials(rows, params: dynTreeParams)
        {
            speculation.dynPartials = (val: p.val, idx: p.idx, sum: p.sum, hidden: rows)
            speculation.memo = nil
            speculation.confidence = nil
            speculation.deviceOutcome = true
            s.speculation = speculation
            return [p.val, p.idx, p.sum, rows, projected.keys, projected.values]
        }
        let logits = draftHeadProduct(rows, readout: false)
        guard logits.shape == [settings.maximumDraftTokens, target.dflashVocabularySize] else {
            s.speculation = speculation
            return [projected.keys, projected.values]
        }
        if CBv2DraftRegime.verifyTreeMode != 0, CBv2DraftRegime.dynamicTreeEnabled,
            logits.ndim == 2, logits.dtype == .bfloat16, logits.dim(1) >= 256
        {
            let p = Self.fastDynamicTreeStage1(logits, params: dynTreeParams)
            speculation.dynPartials = (val: p.val, idx: p.idx, sum: p.sum, hidden: rows)
            speculation.memo = nil
            speculation.confidence = nil
            speculation.deviceOutcome = true
            s.speculation = speculation
            return [p.val, p.idx, p.sum, rows, projected.keys, projected.values]
        }
        let memo = (tokens: proposals(logits, selection: nil), hidden: rows)
        speculation.memo = memo
        speculation.confidence = nil
        speculation.deviceOutcome = true
        s.speculation = speculation
        return [memo.tokens, memo.hidden, projected.keys, projected.values]
    }

    /// The confirmed speculation's per-row draft confidence, still lazy (the
    /// caller reads it only once it is available); nil otherwise.
    public func speculatedDraftConfidence(requestState: any CBv2MTPRequestState) -> MLXArray? {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released,
            let speculation = s.speculation, speculation.carry != nil, speculation.memo != nil,
            (speculation.deviceOutcome || speculation.accepted == speculation.depth),
            (speculation.accepted == 0 || speculation.pending != nil)
        else { return nil }
        return speculation.confidence
    }

    /// Finalize stored `carryHidden` (this object) as the row's carry after
    /// `acceptedDrafts` accepted drafts, the carry being verify row
    /// `acceptedDrafts`. `carryTokenIsTarget`: its token is the verify's
    /// own target at that row.
    public func confirmSpeculation(
        carryHidden: MLXArray, acceptedDrafts: Int, carryTokenIsTarget: Bool,
        requestState: any CBv2MTPRequestState
    ) {
        confirmSpeculation(
            carryHidden: carryHidden, acceptedDrafts: acceptedDrafts,
            acceptedPath: Array(0 ... acceptedDrafts),
            carryTokenIsTarget: carryTokenIsTarget, requestState: requestState)
    }

    public func confirmSpeculation(
        carryHidden: MLXArray, acceptedDrafts: Int, acceptedPath: [Int], carryTokenIsTarget: Bool,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        guard var speculation = s.speculation, speculation.carry == nil,
            let accepted = speculation.accepted, accepted == acceptedDrafts,
            (accepted == 0) == (speculation.pending == nil),
            acceptedPath.count == accepted + 1,
            acceptedPath.allSatisfy({ $0 >= 0 && $0 <= speculation.depth })
        else {
            s.speculation = nil
            return
        }
        let isLinear = acceptedPath.elementsEqual(0 ... accepted)
        // The block assumed every draft accepted and the bonus target as the
        // seed unless `deviceOutcome` drafted for the verified accept path.
        if (!speculation.deviceOutcome && (accepted != speculation.depth || !isLinear))
            || !carryTokenIsTarget
        {
            speculation.memo = nil
            speculation.dynPartials = nil
            speculation.confidence = nil
        }
        speculation.carry = carryHidden
        speculation.acceptedPath = acceptedPath
        s.speculation = speculation
    }

    public func dropSpeculation(requestState: any CBv2MTPRequestState) {
        state(requestState).speculation = nil
    }

    /// Index 0 of the round after a speculated one: when the state is the
    /// one `speculateNextRound` assumed for the outcome finalize confirmed
    /// (that carry object, the pending rows `finalizeRound` set for it, the
    /// context unchanged since), apply commitCarry's transitions and append
    /// that outcome's rows, and hand out the block's memo when it was built
    /// for this outcome. `.normal` leaves the state as it was.
    private func takeSpeculation(_ speculation: Speculation, state s: State, hidden: MLXArray)
        -> TakenSpeculation
    {
        guard settings.batchesRequests, s.awaitsCarry,
            let carry = speculation.carry, carry === hidden,
            let accepted = speculation.accepted, accepted >= 0, accepted <= speculation.depth,
            s.context.nextPosition == speculation.anchor,
            s.context.firstPosition == speculation.first
        else { return .normal }
        if let expected = speculation.pending {
            guard let pending = s.pending, pending === expected else { return .normal }
        } else {
            guard accepted == 0, s.pending == nil else { return .normal }
        }
        let memo =
            (s.selection == nil && CBv2DraftRegime.activeVerifyTree == nil
                && CBv2DraftRegime.activeDynamicTreeK == 0) ? speculation.memo : nil
        let dynPartials =
            (s.selection == nil && CBv2DraftRegime.activeVerifyTree == nil)
            ? speculation.dynPartials : nil
        s.pending = nil
        s.awaitsCarry = false
        s.committedInputCount += 1
        let path = speculation.acceptedPath ?? Array(0 ... accepted)
        if !path.elementsEqual(0 ... accepted) {
            let idx = MLXArray(path.map { Int32($0) })
            s.context.append(
                keys: speculation.keys.take(idx, axis: 2),
                values: speculation.values.take(idx, axis: 2))
        } else if let ticket = speculation.tailTicket,
            s.context.promoteSpeculativePrefix(ticket: ticket, count: accepted + 1)
        {
            // Verified linear prefix is already resident in the ring tail;
            // `promoteSpeculativePrefix` advanced `nextPosition` in place.
        } else if accepted == speculation.depth {
            s.context.append(keys: speculation.keys, values: speculation.values)
        } else {
            let rows = 0 ..< (accepted + 1)
            s.context.append(
                keys: speculation.keys[0..., 0..., rows, 0...],
                values: speculation.values[0..., 0..., rows, 0...])
        }
        if let dp = dynPartials {
            let dynK = CBv2DraftRegime.activeDynamicTreeK
            let rows = dp.hidden.dim(0)
            if dynK >= 2, dynK <= rows, rows <= 7 {
                let dyn = Self.fastDynamicTreeStage2(
                    dp.val, dp.idx, dp.sum, k: dynK, vocab: target.dflashVocabularySize,
                    params: dynTreeParams)
                CBv2DraftRegime.activeDynamicTreeMeta = dyn.treeMeta
                s.memo = (dyn.tokens, dp.hidden)
                return .drafted
            } else {
                let tokens = Self.fastDynamicTreeArgMax(
                    dp.val, dp.idx, vocab: target.dflashVocabularySize)
                s.memo = (tokens, dp.hidden)
                return .drafted
            }
        }
        guard let memo else { return .committed }
        s.memo = memo
        return .drafted
    }

    /// The head product of draft rows `[rows, hiddenSize]`: MLX's NVFP4
    /// product of the target's head when it is NVFP4, else the target's own.
    /// Without `readout` (argmax proposals only) the NVFP4 product skips
    /// `weight_scale_2`: a positive scalar, so the row argmax is the same up
    /// to bf16 rounding ties. A coupled selection reads the target's readout
    /// of the product and needs the scale.
    private func draftHeadProduct(_ rows: MLXArray, readout: Bool) -> MLXArray {
        guard let draftHead, rows.dim(-1) == draftHead.inputDimensions else {
            return target.dflashHeadProduct(rows)
        }
        return readout ? draftHead(rows) : draftHead.unscaledProduct(rows)
    }

    private static func concatQuantized(_ layers: [QuantizedLinear]) -> QuantizedLinear? {
        guard let first = layers.first, layers.count >= 2 else { return nil }
        let gs = first.groupSize
        let bits = first.bits
        let mode = first.mode
        let hasBiases = first.biases != nil
        guard
            layers.allSatisfy({
                $0.groupSize == gs && $0.bits == bits && $0.mode == mode && $0.bias == nil
                    && ($0.biases != nil) == hasBiases
            })
        else { return nil }
        let w = concatenated(layers.map(\.weight), axis: 0)
        let sc = concatenated(layers.map(\.scales), axis: 0)
        let bi: MLXArray? = hasBiases ? concatenated(layers.compactMap(\.biases), axis: 0) : nil
        eval([w, sc] + (bi.map { [$0] } ?? []))
        return QuantizedLinear(
            weight: w, bias: nil, scales: sc, biases: bi, groupSize: gs, bits: bits, mode: mode)
    }

    private static func buildFusedAttentionProjections(_ model: MuseGlimmerDFlashDraftModel)
        -> (contextKV: QuantizedLinear?, blockQKV: [QuantizedLinear], lastKV: QuantizedLinear?)
    {
        guard fusedProjEnabled, model.layers.count == 5, model.configuration.headDimensions == 128,
            model.activationDType == .bfloat16, Device.defaultDevice().deviceType == .gpu
        else { return (nil, [], nil) }
        var kProjs: [QuantizedLinear] = []
        var vProjs: [QuantizedLinear] = []
        var blockQKV: [QuantizedLinear] = []
        for (i, layer) in model.layers.enumerated() {
            guard let q = layer.attention.qProj as? QuantizedLinear,
                let k = layer.attention.kProj as? QuantizedLinear,
                let v = layer.attention.vProj as? QuantizedLinear
            else { return (nil, [], nil) }
            kProjs.append(k)
            vProjs.append(v)
            if i + 1 < model.layers.count {
                guard let qkv = concatQuantized([q, k, v]) else { return (nil, [], nil) }
                blockQKV.append(qkv)
            }
        }
        let contextKV = concatQuantized(kProjs + vProjs)
        let lastKV = concatQuantized([kProjs[4], vProjs[4]])
        return (contextKV, blockQKV, lastKV)
    }

    /// Layer `index`'s MLP: the 3-bit copy while a low-acceptance round
    /// drafts (`CBv2DraftRegime.lowAcceptance`), the pinned one otherwise.
    private func draftMLP(_ index: Int, _ mlp: MuseGlimmerDFlashMLP, _ x: MLXArray) -> MLXArray {
        if CBv2DraftRegime.lowAcceptance, index < lowMLP.count {
            let low = lowMLP[index]
            if let guProj = low.gateUp {
                let gu = guProj(x)
                if gu.dtype == .bfloat16 && gu.dim(0) == 1 && (gu.dim(2) / 2) % 4 == 0 {
                    return low.down(Self.splitSiluMul(gu))
                }
                let mid = gu.dim(-1) / 2
                return low.down(compiledSiluProduct(gu[0..., 0..., 0 ..< mid], gu[0..., 0..., mid...]))
            }
            if let g = low.gate, let u = low.up {
                return low.down(compiledSiluProduct(g(x), u(x)))
            }
        }
        return mlp.down(compiledSiluProduct(mlp.gate(x), mlp.up(x)))
    }

    private static func lowAcceptanceMLP(_ model: MuseGlimmerDFlashDraftModel)
        -> [(
            gate: QuantizedLinear?, up: QuantizedLinear?, gateUp: QuantizedLinear?,
            down: QuantizedLinear
        )]
    {
        let bits = ProcessInfo.processInfo.environment["CBV2_DRAFTER_LOW_MLP_BITS"]
            .flatMap { Int($0) }.map { min(max($0, 2), 5) } ?? 3
        guard bits < 5 else { return [] }
        func low(_ linear: Linear) -> QuantizedLinear? {
            guard let ql = linear as? QuantizedLinear, ql.mode == .affine, ql.bits > bits else {
                return nil
            }
            let dense = MLX.dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
            let copy = QuantizedLinear(
                weight: dense, bias: ql.bias, groupSize: 32, bits: bits, mode: .affine)
            eval([copy.weight, copy.scales] + (copy.biases.map { [$0] } ?? []))
            return copy
        }
        var copies:
            [(
                gate: QuantizedLinear?, up: QuantizedLinear?, gateUp: QuantizedLinear?,
                down: QuantizedLinear
            )] = []
        for layer in model.layers {
            guard let gate = low(layer.mlp.gate), let up = low(layer.mlp.up),
                let down = low(layer.mlp.down)
            else { return [] }
            if fusedProjEnabled, let gateUp = concatQuantized([gate, up]) {
                copies.append((nil, nil, gateUp, down))
            } else {
                copies.append((gate, up, nil, down))
            }
        }
        // Compile the copies' kernels at load for the block's row counts
        // (16 rows and all active draft counts 1...7), not in a timed round.
        if let first = copies.first {
            let width = model.configuration.hiddenSize
            var warm = [MLXArray]()
            for rows in [model.configuration.blockSize, 1, 2, 3, 4, 5, 6, 7] {
                let x = MLXArray.zeros([1, rows, width], dtype: model.activationDType)
                if let guProj = first.gateUp {
                    let gu = guProj(x)
                    if gu.dtype == .bfloat16 && (gu.dim(2) / 2) % 4 == 0 {
                        warm.append(first.down(splitSiluMul(gu)))
                    } else {
                        let mid = gu.dim(-1) / 2
                        warm.append(
                            first.down(
                                compiledSiluProduct(gu[0..., 0..., 0 ..< mid], gu[0..., 0..., mid...])))
                    }
                } else if let g = first.gate, let u = first.up {
                    warm.append(first.down(compiledSiluProduct(g(x), u(x))))
                }
            }
            eval(warm)
        }
        return copies
    }

    private static let requantLock = NSLock()
    nonisolated(unsafe) private static var requantizedModels = Set<ObjectIdentifier>()

    /// Re-quantizes the drafter's 8-bit linear projections (`encoder.fc` and
    /// `self_attn.{q,k,v,o}_proj`, ~597 MB) in memory on load to 4-bit
    /// `groupSize: 32` (~335 MB), leaving the 5-bit MLP weights untouched.
    /// Permitted by the track contract (drafter only proposes tokens; the
    /// pinned target verifies every emitted token).
    private static func requantize8BitLayers(_ model: MuseGlimmerDFlashDraftModel) {
        let key = ObjectIdentifier(model)
        requantLock.lock()
        let claimed = requantizedModels.insert(key).inserted
        requantLock.unlock()
        guard claimed else { return }

        let updates = model.leafModules().compactMapValues { m -> Module? in
            guard let ql = m as? QuantizedLinear, ql.bits == 8, ql.mode == .affine else {
                return nil
            }
            let dense = MLX.dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
            let replacement = QuantizedLinear(
                weight: dense, bias: ql.bias, groupSize: 32, bits: 4, mode: .affine)
            if let biases = replacement.biases {
                eval(replacement.weight, replacement.scales, biases)
            } else {
                eval(replacement.weight, replacement.scales)
            }
            return replacement
        }
        model.update(modules: updates)
    }

    private static func buildNVFP4Drafter(_ model: MuseGlimmerDFlashDraftModel)
        -> NVFP4DrafterState?
    {
        let c = model.configuration
        guard nvfp4DrafterEnabled, fusedProjEnabled,
            Device.defaultDevice().deviceType == .gpu,
            model.activationDType == .bfloat16,
            model.layers.count == 5, c.headDimensions == 128,
            c.hiddenSize % 16 == 0, c.intermediateSize % 16 == 0, c.contextWidth % 16 == 0,
            c.hiddenSize % 4 == 0, (c.intermediateSize) % 4 == 0
        else { return nil }

        func denseWeight(_ linear: Linear) -> MLXArray? {
            guard linear.bias == nil else { return nil }
            if let ql = linear as? QuantizedLinear {
                return MLX.dequantized(
                    ql.weight, scales: ql.scales, biases: ql.biases,
                    groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode
                ).asType(.bfloat16)
            }
            return linear.weight.asType(.bfloat16)
        }

        func pickScale(_ w: MLXArray) -> Float {
            let maxAbs = MLX.abs(w).max().asType(.float32).item(Float.self)
            guard maxAbs.isFinite, maxAbs > 0 else { return 1.0 }
            if maxAbs <= 32.0 { return 64.0 }
            if maxAbs <= 128.0 { return 16.0 }
            if maxAbs <= 512.0 { return 4.0 }
            return 1.0
        }

        func quantNVFP4(_ w: MLXArray, scale: Float) -> QuantizedLinear {
            let scaled = (scale == 1.0) ? w : (w.asType(.float32) * scale).asType(.bfloat16)
            let ql = QuantizedLinear(
                weight: scaled, bias: nil, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: nil)
            eval(ql.weight, ql.scales)
            return ql
        }

        guard let fcW = denseWeight(model.encoder.fc) else { return nil }
        let fcScale = pickScale(fcW)
        let fc = quantNVFP4(fcW, scale: fcScale)
        let fcEps = model.encoder.outputNorm.eps * fcScale * fcScale

        var kDense: [MLXArray] = []
        var vDense: [MLXArray] = []
        kDense.reserveCapacity(5)
        vDense.reserveCapacity(5)
        for layer in model.layers {
            guard let kw = denseWeight(layer.attention.kProj),
                let vw = denseWeight(layer.attention.vProj)
            else { return nil }
            kDense.append(kw)
            vDense.append(vw)
        }
        let allCtxKV = concatenated(kDense + vDense, axis: 0)
        let ctxKVScale = pickScale(allCtxKV)
        let fusedContextKV = quantNVFP4(allCtxKV, scale: ctxKVScale)
        let encOutNormW =
            (model.encoder.outputNorm.weight.asType(.float32) * (1.0 / ctxKVScale)).asType(.bfloat16)

        var blockQKV: [QuantizedLinear] = []
        var oProj: [QuantizedLinear] = []
        var oInvScale: [Float] = []
        var gateUp: [QuantizedLinear] = []
        var down: [QuantizedLinear] = []
        var downInvScale: [Float] = []
        var inputNormW: [MLXArray] = []
        var postAttnNormW: [MLXArray] = []
        var lastKVOpt: QuantizedLinear? = nil
        var lastQOpt: QuantizedLinear? = nil

        for (i, layer) in model.layers.enumerated() {
            guard let qw = denseWeight(layer.attention.qProj),
                let ow = denseWeight(layer.attention.oProj),
                let gw = denseWeight(layer.mlp.gate),
                let uw = denseWeight(layer.mlp.up),
                let dw = denseWeight(layer.mlp.down)
            else { return nil }

            if i < 4 {
                let qkvW = concatenated([qw, kDense[i], vDense[i]], axis: 0)
                let sQKV = pickScale(qkvW)
                blockQKV.append(quantNVFP4(qkvW, scale: sQKV))
                inputNormW.append(
                    (layer.inputNorm.weight.asType(.float32) * (1.0 / sQKV)).asType(.bfloat16))
            } else {
                let kvW = concatenated([kDense[i], vDense[i]], axis: 0)
                let sL4 = min(pickScale(kvW), pickScale(qw))
                lastKVOpt = quantNVFP4(kvW, scale: sL4)
                lastQOpt = quantNVFP4(qw, scale: sL4)
                inputNormW.append(
                    (layer.inputNorm.weight.asType(.float32) * (1.0 / sL4)).asType(.bfloat16))
            }

            let sO = pickScale(ow)
            oProj.append(quantNVFP4(ow, scale: sO))
            oInvScale.append(1.0 / sO)

            let guW = concatenated([gw, uw], axis: 0)
            let sGU = pickScale(guW)
            gateUp.append(quantNVFP4(guW, scale: sGU))
            postAttnNormW.append(
                (layer.postAttentionNorm.weight.asType(.float32) * (1.0 / sGU)).asType(.bfloat16))

            let sDown = pickScale(dw)
            down.append(quantNVFP4(dw, scale: sDown))
            downInvScale.append(1.0 / sDown)
        }
        guard let lastKV = lastKVOpt, let lastQ = lastQOpt else { return nil }
        eval([encOutNormW] + inputNormW + postAttnNormW)
        return NVFP4DrafterState(
            fc: fc, fcEps: fcEps, encOutputNormWeightScaled: encOutNormW,
            fusedContextKV: fusedContextKV, blockQKV: blockQKV, lastKV: lastKV, lastQ: lastQ,
            oProj: oProj, oInvScale: oInvScale, gateUp: gateUp, down: down,
            downInvScale: downInvScale, inputNormWeightScaled: inputNormW,
            postAttnNormWeightScaled: postAttnNormW)
    }

    private static let drafterLayer4NAX: Bool =
        ProcessInfo.processInfo.environment["CBV2_DRAFTER_L4_NAX"] == "1"

    /// One block forward returning only the `draftCount` active draft rows
    /// (`1 ..< 1 + draftCount`). Layers `0 ..< last` run on all `blockSize`
    /// rows (with `compiledSiluProduct` in the MLP). The final layer projects
    /// K/V on all `blockSize` rows so attention sees the full block, then
    /// slices queries, attention, O-proj, MLP and final norm to only the
    /// `draftCount` active rows (7 instead of 16), cutting final-layer work
    /// in half with mathematically identical proposals.
    private func fastDraftBlockRows(
        embeddings: MLXArray, at position: Int,
        context: (keys: MLXArray, values: MLXArray)?, mask: MLXArray?,
        draftCount: Int, devicePosition: MLXArray? = nil,
        ringContext: FastInPlaceContext? = nil, ringTailStart: Int = 0
    ) -> MLXArray {
        func rope(_ x: MLXArray, _ shift: Int) -> MLXArray {
            if let devicePosition {
                return MLXFast.RoPE(
                    x, dimensions: model.configuration.headDimensions, traditional: false,
                    base: model.configuration.ropeTheta, scale: 1,
                    offset: shift == 0 ? devicePosition : devicePosition + Int32(shift))
            }
            return MLXFast.RoPE(
                x, dimensions: model.configuration.headDimensions, traditional: false,
                base: model.configuration.ropeTheta, scale: 1, offset: position + shift)
        }
        let c = model.configuration
        let block = c.blockSize
        let scale = 1 / Float(c.headDimensions).squareRoot()
        let act = embeddings.dtype == model.activationDType
            ? embeddings : embeddings.asType(model.activationDType)
        var hidden = act.reshaped([1, block, c.hiddenSize])
        let lastIndex = model.layers.count - 1
        let nv: NVFP4DrafterState? =
            (CBv2DraftRegime.lowAcceptance && CBv2DraftRegime.xnorDraftHeadEligible
                && !Self.drafterLayer4NAX) ? nvfp4Drafter : nil
        let useFusedAttn =
            Self.fusedProjEnabled && CBv2DraftRegime.lowAcceptance && contextKeyEpilogue != nil
            && (nv != nil || (fusedBlockQKV.count == lastIndex && fusedLastKV != nil))
            && c.headDimensions == 128 && model.activationDType == .bfloat16
        let useFusedAddNorm = useFusedAttn && c.hiddenSize % 4 == 0
        let useNVFP4 = (nv != nil) && useFusedAddNorm
        let posScalar: MLXArray? =
            useFusedAttn
            ? (devicePosition.map { $0.reshaped([1]) } ?? MLXArray([Int32(position)])) : nil
        let ringBuffers = (useFusedAttn && context != nil) ? ringContext?.ringTailBuffers() : nil
        var tailFence = ringBuffers?.fence
        var nextX: MLXArray? = nil
        for (index, layer) in model.layers.enumerated() {
            let attention = layer.attention
            let x: MLXArray
            if let nx = nextX {
                x = nx
            } else if useNVFP4, let nv {
                x = MLXFast.rmsNorm(
                    hidden, weight: nv.inputNormWeightScaled[index], eps: layer.inputNorm.eps)
            } else {
                x = layer.inputNorm(hidden)
            }
            nextX = nil
            if useFusedAttn, index < lastIndex, let epilogue = contextKeyEpilogue, let posScalar {
                let qkv = useNVFP4 ? nv!.blockQKV[index](x) : fusedBlockQKV[index](x)
                let queries: MLXArray
                var keys: MLXArray
                var values: MLXArray
                if let rb = ringBuffers, let prevFence = tailFence, let ringContext {
                    let launch = Self.preparedBlockQKVRing(
                        qHeads: c.attentionHeads, kvHeads: c.kvHeads, block: block,
                        window: ringContext.window)
                    let iparams = MLXArray([Int32(index), Int32(ringTailStart)])
                    let out = launch([
                        qkv, attention.qNorm.weight, attention.kNorm.weight,
                        rb.keys, rb.values, posScalar, epilogue.coefficients, iparams, prevFence,
                    ])
                    queries = out[0]
                    tailFence = out[1]
                    let kvDep = depends(inputs: [rb.keys, rb.values], dependencies: [out[1]])
                    let span = 0 ..< (ringTailStart + block)
                    keys = kvDep[0][index ..< index + 1, 0..., span, 0...]
                    values = kvDep[1][index ..< index + 1, 0..., span, 0...]
                } else {
                    let key = (UInt64(c.attentionHeads) << 24) | (UInt64(c.kvHeads) << 12) | UInt64(block)
                    let launch: MLXFast.MLXFastKernel.Launch =
                        Self.drafterLaunchLock.withLockUnchecked { box in
                            if let hit = box.blkQKV[key] { return hit }
                            let made = Self.blockQKVEpilogueKernel.prepare(
                                template: [("HQ_", c.attentionHeads), ("HKV_", c.kvHeads), ("B_", block)],
                                grid: (32, (c.attentionHeads + 2 * c.kvHeads) * block, 1),
                                threadGroup: (32, 1, 1),
                                outputShapes: [
                                    [1, c.attentionHeads, block, 128],
                                    [1, c.kvHeads, block, 128],
                                    [1, c.kvHeads, block, 128],
                                ],
                                outputDTypes: [.bfloat16, .bfloat16, .bfloat16])
                            box.blkQKV[key] = made
                            return made
                        }
                    let out = launch([
                        qkv, attention.qNorm.weight, attention.kNorm.weight, posScalar,
                        epilogue.coefficients,
                    ])
                    queries = out[0]
                    keys = out[1]
                    values = out[2]
                    if let context {
                        keys = concatenated([context.keys[index ..< index + 1], keys], axis: 2)
                        values = concatenated([context.values[index ..< index + 1], values], axis: 2)
                    }
                }
                let attended = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: mask)
                let attnFlat = attended.transposed(0, 2, 1, 3).reshaped([
                    1, block, c.attentionHeads * c.headDimensions,
                ])
                if useNVFP4, let nv {
                    let attnOut = nv.oProj[index](attnFlat)
                    let (hMid, mx) = Self.fusedAddRMSNorm(
                        residual: hidden, delta: attnOut,
                        weight: nv.postAttnNormWeightScaled[index],
                        eps: layer.postAttentionNorm.eps, deltaScale: nv.oInvScale[index])
                    let gu = nv.gateUp[index](mx)
                    let mlpOut = nv.down[index](
                        Self.splitSiluMul(gu, scale: nv.downInvScale[index]))
                    let nextLayer = model.layers[index + 1]
                    let (hNext, xNext) = Self.fusedAddRMSNorm(
                        residual: hMid, delta: mlpOut,
                        weight: nv.inputNormWeightScaled[index + 1],
                        eps: nextLayer.inputNorm.eps)
                    hidden = hNext
                    nextX = xNext
                } else {
                    let attnOut = attention.oProj(attnFlat)
                    if useFusedAddNorm {
                        let (hMid, mx) = Self.fusedAddRMSNorm(
                            residual: hidden, delta: attnOut,
                            weight: layer.postAttentionNorm.weight,
                            eps: layer.postAttentionNorm.eps)
                        let mlpOut = draftMLP(index, layer.mlp, mx)
                        let nextLayer = model.layers[index + 1]
                        let (hNext, xNext) = Self.fusedAddRMSNorm(
                            residual: hMid, delta: mlpOut, weight: nextLayer.inputNorm.weight,
                            eps: nextLayer.inputNorm.eps)
                        hidden = hNext
                        nextX = xNext
                    } else {
                        hidden = hidden + attnOut
                        let mx = layer.postAttentionNorm(hidden)
                        hidden = hidden + draftMLP(index, layer.mlp, mx)
                    }
                }
                continue
            }
            var keys: MLXArray
            var values: MLXArray
            if useFusedAttn, index == lastIndex, let epilogue = contextKeyEpilogue, let posScalar,
                (useNVFP4 || fusedLastKV != nil)
            {
                let kv = useNVFP4 ? nv!.lastKV(x) : fusedLastKV!(x)
                if let rb = ringBuffers, let prevFence = tailFence, let ringContext {
                    let launch = Self.preparedBlockKVRing(
                        kvHeads: c.kvHeads, block: block, window: ringContext.window)
                    let iparams = MLXArray([Int32(index), Int32(ringTailStart)])
                    let fence = launch([
                        kv, attention.kNorm.weight, rb.keys, rb.values,
                        posScalar, epilogue.coefficients, iparams, prevFence,
                    ])[0]
                    tailFence = fence
                    let kvDep = depends(inputs: [rb.keys, rb.values], dependencies: [fence])
                    let span = 0 ..< (ringTailStart + block)
                    keys = kvDep[0][index ..< index + 1, 0..., span, 0...]
                    values = kvDep[1][index ..< index + 1, 0..., span, 0...]
                } else {
                    let key = (UInt64(c.kvHeads) << 12) | UInt64(block)
                    let launch: MLXFast.MLXFastKernel.Launch =
                        Self.drafterLaunchLock.withLockUnchecked { box in
                            if let hit = box.blkKV[key] { return hit }
                            let made = Self.blockKVEpilogueKernel.prepare(
                                template: [("HKV_", c.kvHeads), ("B_", block)],
                                grid: (32, 2 * c.kvHeads * block, 1),
                                threadGroup: (32, 1, 1),
                                outputShapes: [
                                    [1, c.kvHeads, block, 128],
                                    [1, c.kvHeads, block, 128],
                                ],
                                outputDTypes: [.bfloat16, .bfloat16])
                            box.blkKV[key] = made
                            return made
                        }
                    let out = launch([kv, attention.kNorm.weight, posScalar, epilogue.coefficients])
                    keys = out[0]
                    values = out[1]
                    if let context {
                        keys = concatenated([context.keys[index ..< index + 1], keys], axis: 2)
                        values = concatenated([context.values[index ..< index + 1], values], axis: 2)
                    }
                }
            } else {
                keys = rope(
                    attention.kNorm(
                        attention.kProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                            .transposed(0, 2, 1, 3)), 0)
                values = attention.vProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                    .transposed(0, 2, 1, 3)
                if let context {
                    keys = concatenated([context.keys[index ..< index + 1], keys], axis: 2)
                    values = concatenated([context.values[index ..< index + 1], values], axis: 2)
                }
            }
            if index == lastIndex
                && (draftCount == 1 || !(Self.drafterLayer4NAX && CBv2DraftRegime.lowAcceptance))
            {
                let active = 1 ..< (1 + draftCount)
                let xActive = x[0..., active, 0...]
                let queries: MLXArray
                if useNVFP4, let nv, let epilogue = contextKeyEpilogue, let posScalar {
                    queries = Self.applyBlockQEpilogue(
                        nv.lastQ(xActive), qNormWeight: attention.qNorm.weight,
                        posScalar: posScalar, coefficients: epilogue.coefficients,
                        qHeads: c.attentionHeads, rows: draftCount)
                } else {
                    queries = rope(
                        attention.qNorm(
                            attention.qProj(xActive).reshaped([
                                1, draftCount, c.attentionHeads, c.headDimensions,
                            ]).transposed(0, 2, 1, 3)), 1)
                }
                let activeMask = mask.map { $0[active, 0...] }
                let attended = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: activeMask)
                let attnFlat = attended.transposed(0, 2, 1, 3).reshaped([
                    1, draftCount, c.attentionHeads * c.headDimensions,
                ])
                let result: MLXArray
                if useNVFP4, let nv {
                    let attnOut = nv.oProj[index](attnFlat)
                    let (hMid, mx) = Self.fusedAddRMSNorm(
                        residual: hidden[0..., active, 0...], delta: attnOut,
                        weight: nv.postAttnNormWeightScaled[index],
                        eps: layer.postAttentionNorm.eps, deltaScale: nv.oInvScale[index])
                    let gu = nv.gateUp[index](mx)
                    let mlpOut = nv.down[index](
                        Self.splitSiluMul(gu, scale: nv.downInvScale[index]))
                    let (_, normed) = Self.fusedAddRMSNorm(
                        residual: hMid, delta: mlpOut, weight: model.norm.weight,
                        eps: model.norm.eps)
                    result = normed.reshaped([draftCount, c.hiddenSize])
                } else if useFusedAddNorm {
                    let attnOut = attention.oProj(attnFlat)
                    let (hMid, mx) = Self.fusedAddRMSNorm(
                        residual: hidden[0..., active, 0...], delta: attnOut,
                        weight: layer.postAttentionNorm.weight, eps: layer.postAttentionNorm.eps)
                    let mlpOut = draftMLP(index, layer.mlp, mx)
                    let (_, normed) = Self.fusedAddRMSNorm(
                        residual: hMid, delta: mlpOut, weight: model.norm.weight,
                        eps: model.norm.eps)
                    result = normed.reshaped([draftCount, c.hiddenSize])
                } else {
                    let attnOut = attention.oProj(attnFlat)
                    var h = hidden[0..., active, 0...] + attnOut
                    let mx = layer.postAttentionNorm(h)
                    h = h + draftMLP(index, layer.mlp, mx)
                    result = model.norm(h).reshaped([draftCount, c.hiddenSize])
                }
                if let ringContext, let fence = tailFence {
                    ringContext.commitBlockTail(fence: fence, readDep: result)
                }
                return result
            }
            let queries = rope(
                attention.qNorm(
                    attention.qProj(x).reshaped([1, block, c.attentionHeads, c.headDimensions])
                        .transposed(0, 2, 1, 3)), 0)
            let attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
            hidden =
                hidden
                + attention.oProj(
                    attended.transposed(0, 2, 1, 3).reshaped([
                        1, block, c.attentionHeads * c.headDimensions,
                    ]))
            let mx = layer.postAttentionNorm(hidden)
            hidden =
                hidden
                + draftMLP(index, layer.mlp, mx)
        }
        let result = model.norm(hidden).reshaped([block, c.hiddenSize])[1 ..< (1 + draftCount), 0...]
        if let ringContext, let fence = tailFence {
            ringContext.commitBlockTail(fence: fence, readDep: result)
        }
        return result
    }

    /// Fused 2-stage split-V argmax + softcapped-readout top-1 probability.
    /// Replaces 9 separate dispatches (~50 MB traffic) over `[M, V]` logits
    /// with one read of `logits` (2.8 MB) and a 64-way partial reduction.
    private static let argMaxConfStage1 = MLXFast.metalKernel(
        name: "dflash_argmax_conf_s1",
        inputNames: ["logits", "params"],
        outputNames: ["p_raw", "p_idx", "p_sc", "p_sum"],
        source: """
            constexpr int V = V_;
            constexpr int CHUNKS = 64;
            constexpr int chunk_size = (V + CHUNKS - 1) / CHUNKS;
            const uint chunk = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;

            const int start = int(chunk) * chunk_size;
            const int end = min(V, start + chunk_size);
            const device bfloat16_t* row_ptr = logits + size_t(row) * size_t(V);
            const bfloat16_t b_mult = bfloat16_t(params[0]);
            const bfloat16_t b_cap = bfloat16_t(params[1]);
            const bool has_cap = params[1] > 0.0f;

            float best_raw = -INFINITY;
            int best_idx = V;
            float best_sc = -INFINITY;
            float sum_exp = 0.0f;
            for (int i = start + int(lid); i < end; i += 256) {
              const bfloat16_t raw_bf = row_ptr[i];
              const float raw_f = float(raw_bf);
              if (raw_f > best_raw || (raw_f == best_raw && i < best_idx)) {
                best_raw = raw_f;
                best_idx = i;
              }
              bfloat16_t s_bf = raw_bf * b_mult;
              if (has_cap) {
                s_bf = bfloat16_t(metal::precise::tanh(float(s_bf / b_cap))) * b_cap;
              }
              const float sc = float(s_bf);
              if (sc > best_sc) {
                sum_exp = (sum_exp > 0.0f) ? (sum_exp * metal::exp(best_sc - sc) + 1.0f) : 1.0f;
                best_sc = sc;
              } else {
                sum_exp += metal::exp(sc - best_sc);
              }
            }

            const float sg_raw = simd_max(best_raw);
            const int sg_idx = simd_min((best_raw == sg_raw) ? best_idx : V);
            const float sg_sc = simd_max(best_sc);
            const float sg_sum = simd_sum(
                (sum_exp > 0.0f) ? (sum_exp * metal::exp(best_sc - sg_sc)) : 0.0f);

            threadgroup float sh_raw[8];
            threadgroup int sh_idx[8];
            threadgroup float sh_sc[8];
            threadgroup float sh_sum[8];
            if (lane == 0) {
              sh_raw[sg] = sg_raw;
              sh_idx[sg] = sg_idx;
              sh_sc[sg] = sg_sc;
              sh_sum[sg] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float v_raw = (lane < 8) ? sh_raw[lane] : -INFINITY;
              const int v_idx = (lane < 8) ? sh_idx[lane] : V;
              const float v_sc = (lane < 8) ? sh_sc[lane] : -INFINITY;
              const float v_sum = (lane < 8) ? sh_sum[lane] : 0.0f;

              const float tg_raw = simd_max(v_raw);
              const int tg_idx = simd_min((v_raw == tg_raw) ? v_idx : V);
              const float tg_sc = simd_max(v_sc);
              const float tg_sum = simd_sum(
                  (v_sum > 0.0f) ? (v_sum * metal::exp(v_sc - tg_sc)) : 0.0f);

              if (lane == 0) {
                const uint out_i = row * uint(CHUNKS) + chunk;
                p_raw[out_i] = tg_raw;
                p_idx[out_i] = tg_idx;
                p_sc[out_i] = tg_sc;
                p_sum[out_i] = tg_sum;
              }
            }
            """)

    private static let argMaxConfStage2 = MLXFast.metalKernel(
        name: "dflash_argmax_conf_s2",
        inputNames: ["p_raw", "p_idx", "p_sc", "p_sum"],
        outputNames: ["tokens", "confidence"],
        source: """
            constexpr int V = V_;
            constexpr int CHUNKS = 64;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint idx = row * uint(CHUNKS) + lid;

            const float v_raw = p_raw[idx];
            const int v_idx = p_idx[idx];
            const float v_sc = p_sc[idx];
            const float v_sum = p_sum[idx];

            const float sg_raw = simd_max(v_raw);
            const int sg_idx = simd_min((v_raw == sg_raw) ? v_idx : V);
            const float sg_sc = simd_max(v_sc);
            const float sg_sum = simd_sum(
                (v_sum > 0.0f) ? (v_sum * metal::exp(v_sc - sg_sc)) : 0.0f);

            threadgroup float sh_raw[2];
            threadgroup int sh_idx[2];
            threadgroup float sh_sc[2];
            threadgroup float sh_sum[2];
            if (lane == 0) {
              sh_raw[sg] = sg_raw;
              sh_idx[sg] = sg_idx;
              sh_sc[sg] = sg_sc;
              sh_sum[sg] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float f_raw = (lane < 2) ? sh_raw[lane] : -INFINITY;
              const int f_idx = (lane < 2) ? sh_idx[lane] : V;
              const float f_sc = (lane < 2) ? sh_sc[lane] : -INFINITY;
              const float f_sum = (lane < 2) ? sh_sum[lane] : 0.0f;

              const float row_raw = simd_max(f_raw);
              const int row_idx = simd_min((f_raw == row_raw) ? f_idx : V);
              const float row_sc = simd_max(f_sc);
              const float row_sum = simd_sum(
                  (f_sum > 0.0f) ? (f_sum * metal::exp(f_sc - row_sc)) : 0.0f);

              if (lane == 0) {
                tokens[row] = row_idx;
                confidence[row] = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
              }
            }
            """)

    private static func fusedArgMaxAndConfidence(
        _ logits: MLXArray, outputMultiplier: Float, softcap: Float
    ) -> (tokens: MLXArray, confidence: MLXArray) {
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        let chunks = 64
        let params = MLXArray([outputMultiplier, softcap])
        let partials = argMaxConfStage1(
            [logits, params],
            template: [("V_", vocab)],
            grid: (256 * chunks, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, chunks], [rows, chunks], [rows, chunks], [rows, chunks]],
            outputDTypes: [.float32, .int32, .float32, .float32])
        let out = argMaxConfStage2(
            partials,
            template: [("V_", vocab)],
            grid: (chunks, rows, 1), threadGroup: (chunks, 1, 1),
            outputShapes: [[rows], [rows]],
            outputDTypes: [.int32, .float32])
        return (out[0], out[1])
    }

    private static let fastArgMaxStage1 = MLXFast.metalKernel(
        name: "dflash_fast_argmax_s1",
        inputNames: ["logits"],
        outputNames: ["p_val", "p_idx"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint chunk = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint vocab = uint(V_);
            const uint chunk_size = (vocab + CHUNKS - 1) / CHUNKS;
            const uint start = chunk * chunk_size;
            const uint end = min(vocab, start + chunk_size);
            const device bfloat16_t* row_ptr = logits + ulong(row) * ulong(vocab);

            float best_val = -INFINITY;
            int best_idx = int(vocab);
            const uint start4 = (start + 3u) & ~3u;
            const uint end4 = end & ~3u;
            if (lid == 0) {
              for (uint i = start; i < min(start4, end); i++) {
                const float v = float(row_ptr[i]);
                if (v > best_val || (v == best_val && int(i) < best_idx)) {
                  best_val = v;
                  best_idx = int(i);
                }
              }
            }
            if (end4 > start4) {
              const device vec<bfloat16_t, 4>* src4 =
                  (const device vec<bfloat16_t, 4>*)(row_ptr + start4);
              const uint count4 = (end4 - start4) >> 2;
              for (uint j = lid; j < count4; j += 256u) {
                const vec<bfloat16_t, 4> vv = src4[j];
                const uint base_i = start4 + (j << 2);
                #pragma unroll
                for (uint k = 0; k < 4; k++) {
                  const float v = float(vv[k]);
                  const int idx = int(base_i + k);
                  if (v > best_val || (v == best_val && idx < best_idx)) {
                    best_val = v;
                    best_idx = idx;
                  }
                }
              }
            }
            if (lid == 0) {
              const uint tail_start = max(start4, end4);
              for (uint i = tail_start; i < end; i++) {
                const float v = float(row_ptr[i]);
                if (v > best_val || (v == best_val && int(i) < best_idx)) {
                  best_val = v;
                  best_idx = int(i);
                }
              }
            }

            const float sg_val = simd_max(best_val);
            const int sg_idx = simd_min((best_val == sg_val) ? best_idx : int(vocab));
            threadgroup float sh_val[8];
            threadgroup int sh_idx[8];
            if (lane == 0) {
              sh_val[sg] = sg_val;
              sh_idx[sg] = sg_idx;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float v = (lane < 8) ? sh_val[lane] : -INFINITY;
              const int idx = (lane < 8) ? sh_idx[lane] : int(vocab);
              const float tg_val = simd_max(v);
              const int tg_idx = simd_min((v == tg_val) ? idx : int(vocab));
              if (lane == 0) {
                const uint out_i = row * CHUNKS + chunk;
                p_val[out_i] = tg_val;
                p_idx[out_i] = tg_idx;
              }
            }
            """)

    private static let fastArgMaxStage2 = MLXFast.metalKernel(
        name: "dflash_fast_argmax_s2",
        inputNames: ["p_val", "p_idx"],
        outputNames: ["tokens"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint vocab = uint(V_);
            const uint idx = row * CHUNKS + lid;
            const float v = p_val[idx];
            const int id = p_idx[idx];
            const float sg_val = simd_max(v);
            const int sg_idx = simd_min((v == sg_val) ? id : int(vocab));
            threadgroup float sh_val[2];
            threadgroup int sh_idx[2];
            if (lane == 0) {
              sh_val[sg] = sg_val;
              sh_idx[sg] = sg_idx;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float fv = (lane < 2) ? sh_val[lane] : -INFINITY;
              const int fid = (lane < 2) ? sh_idx[lane] : int(vocab);
              const float rv = simd_max(fv);
              const int rid = simd_min((fv == rv) ? fid : int(vocab));
              if (lane == 0) {
                tokens[row] = rid;
              }
            }
            """)

    private static func fastArgMax(_ logits: MLXArray) -> MLXArray {
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        let chunks = 64
        let partials = fastArgMaxStage1(
            [logits],
            template: [("V_", vocab)],
            grid: (256 * chunks, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, chunks], [rows, chunks]],
            outputDTypes: [.float32, .int32])
        return fastArgMaxStage2(
            partials,
            template: [("V_", vocab)],
            grid: (chunks, rows, 1), threadGroup: (chunks, 1, 1),
            outputShapes: [[rows]],
            outputDTypes: [.int32])[0]
    }

    private static let fastDynTreeHeader = """
        inline void dflash_insert_top5(
            thread float& v0, thread int& i0,
            thread float& v1, thread int& i1,
            thread float& v2, thread int& i2,
            thread float& v3, thread int& i3,
            thread float& v4, thread int& i4,
            float v, int idx, int ivocab
        ) {
            if (idx >= ivocab) { return; }
            if (!(v > v4 || (v == v4 && idx < i4))) { return; }
            if (v > v0 || (v == v0 && idx < i0)) {
                v4 = v3; i4 = i3;
                v3 = v2; i3 = i2;
                v2 = v1; i2 = i1;
                v1 = v0; i1 = i0;
                v0 = v;  i0 = idx;
            } else if (v > v1 || (v == v1 && idx < i1)) {
                v4 = v3; i4 = i3;
                v3 = v2; i3 = i2;
                v2 = v1; i2 = i1;
                v1 = v;  i1 = idx;
            } else if (v > v2 || (v == v2 && idx < i2)) {
                v4 = v3; i4 = i3;
                v3 = v2; i3 = i2;
                v2 = v;  i2 = idx;
            } else if (v > v3 || (v == v3 && idx < i3)) {
                v4 = v3; i4 = i3;
                v3 = v;  i3 = idx;
            } else {
                v4 = v;  i4 = idx;
            }
        }
        inline float dflash_scale_logit(float raw, float eff_scale, float cap) {
            float v = raw * eff_scale;
            if (cap > 0.0f) {
                v = cap * metal::fast::tanh(v / cap);
            }
            return v;
        }
        inline void dflash_accum_softmax(
            thread float& m, thread float& s, float v, float inv_t_log2e
        ) {
            if (v > m) {
                s = (m == -INFINITY) ? 1.0f : (s * metal::fast::exp2((m - v) * inv_t_log2e) + 1.0f);
                m = v;
            } else {
                const float diff = (m - v) * inv_t_log2e;
                if (diff < 24.0f) {
                    s += metal::fast::exp2(-diff);
                }
            }
        }
        """

    private static let fastDynTreeStage1Kernel = MLXFast.metalKernel(
        name: "dflash_fast_dyntree_s1",
        inputNames: ["logits", "tparams"],
        outputNames: ["p_val", "p_idx", "p_sum"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint chunk = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint vocab = uint(V_);
            const int ivocab = int(vocab);
            const float eff_scale = tparams[0];
            const float cap = tparams[1];
            const float row_T = max(0.05f, tparams[2] + float(row) * tparams[14]);
            const float inv_t_log2e = 1.44269504f / row_T;
            const uint chunk_size = (((vocab + CHUNKS - 1u) / CHUNKS) + 3u) & ~3u;
            const uint start = chunk * chunk_size;
            const uint end = min(vocab, start + chunk_size);
            const device bfloat16_t* row_ptr = logits + ulong(row) * ulong(vocab);

            float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY, v4 = -INFINITY;
            int i0 = ivocab, i1 = ivocab, i2 = ivocab, i3 = ivocab, i4 = ivocab;
            float sm_m = -INFINITY, sm_s = 0.0f;
            const uint start4 = (start + 3u) & ~3u;
            const uint end4 = end & ~3u;
            if (lid == 0) {
              for (uint i = start; i < min(start4, end); i++) {
                const float v = dflash_scale_logit(float(row_ptr[i]), eff_scale, cap);
                dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, v, int(i), ivocab);
                dflash_accum_softmax(sm_m, sm_s, v, inv_t_log2e);
              }
            }
            if (end4 > start4) {
              const device vec<bfloat16_t, 4>* src4 =
                  (const device vec<bfloat16_t, 4>*)(row_ptr + start4);
              const uint count4 = (end4 - start4) >> 2;
              const uint fast_limit = min(count4, 1024u);
              vec<bfloat16_t, 4> buf[4];
              uint n_it = 0u;
              float rv0 = -INFINITY, rv1 = -INFINITY, rv2 = -INFINITY, rv3 = -INFINITY, rv4 = -INFINITY;
              int ri0 = ivocab, ri1 = ivocab, ri2 = ivocab, ri3 = ivocab, ri4 = ivocab;
              for (uint j = lid; j < fast_limit; j += 256u) {
                const vec<bfloat16_t, 4> vv = src4[j];
                buf[n_it++] = vv;
                const uint base_i = start4 + (j << 2);
                #pragma unroll
                for (uint k = 0; k < 4; k++) {
                  const float raw = float(vv[k]);
                  dflash_insert_top5(rv0, ri0, rv1, ri1, rv2, ri2, rv3, ri3, rv4, ri4, raw, int(base_i + k), ivocab);
                }
              }
              if (n_it > 0u) {
                if (ri0 < ivocab) { dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, dflash_scale_logit(rv0, eff_scale, cap), ri0, ivocab); }
                if (ri1 < ivocab) { dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, dflash_scale_logit(rv1, eff_scale, cap), ri1, ivocab); }
                if (ri2 < ivocab) { dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, dflash_scale_logit(rv2, eff_scale, cap), ri2, ivocab); }
                if (ri3 < ivocab) { dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, dflash_scale_logit(rv3, eff_scale, cap), ri3, ivocab); }
                if (ri4 < ivocab) { dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, dflash_scale_logit(rv4, eff_scale, cap), ri4, ivocab); }
                const float buf_m = dflash_scale_logit(rv0, eff_scale, cap);
                float raw_cutoff = -INFINITY;
                if (eff_scale > 0.0f) {
                  const float sc_cut = max(sm_m, buf_m) - 24.0f / inv_t_log2e;
                  if (cap > 0.0f) {
                    const float u = sc_cut / cap;
                    if (u > -0.9999f && u < 0.9999f) {
                      raw_cutoff = (cap / eff_scale) * (0.5f * metal::fast::log((1.0f + u) / (1.0f - u)));
                    } else if (u >= 0.9999f) {
                      raw_cutoff = rv0;
                    }
                  } else {
                    raw_cutoff = sc_cut / eff_scale;
                  }
                }
                for (uint it = 0u; it < n_it; ++it) {
                  const vec<bfloat16_t, 4> vv = buf[it];
                  #pragma unroll
                  for (uint k = 0; k < 4; k++) {
                    const float raw = float(vv[k]);
                    if (raw >= raw_cutoff) {
                      const float v = dflash_scale_logit(raw, eff_scale, cap);
                      dflash_accum_softmax(sm_m, sm_s, v, inv_t_log2e);
                    }
                  }
                }
              }
              for (uint j = lid + 1024u; j < count4; j += 256u) {
                const vec<bfloat16_t, 4> vv = src4[j];
                const uint base_i = start4 + (j << 2);
                #pragma unroll
                for (uint k = 0; k < 4; k++) {
                  const float v = dflash_scale_logit(float(vv[k]), eff_scale, cap);
                  dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, v, int(base_i + k), ivocab);
                  dflash_accum_softmax(sm_m, sm_s, v, inv_t_log2e);
                }
              }
            }
            if (lid == 0) {
              const uint tail_start = max(start4, end4);
              for (uint i = tail_start; i < end; i++) {
                const float v = dflash_scale_logit(float(row_ptr[i]), eff_scale, cap);
                dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, v, int(i), ivocab);
                dflash_accum_softmax(sm_m, sm_s, v, inv_t_log2e);
              }
            }

            const float sg_m = simd_max(sm_m);
            const float scaled_s = (sm_m == -INFINITY)
                ? 0.0f : sm_s * metal::fast::exp2((sm_m - sg_m) * inv_t_log2e);
            const float sg_s = simd_sum(scaled_s);

            // 5-step hardware simd_max + simd_min peel-off (10 SIMD ops, zero branches).
            float out_v[5];
            int out_i[5];
            int ptr = 0;
            #pragma unroll
            for (int k = 0; k < 5; ++k) {
              const float cv = (ptr == 0) ? v0 : (ptr == 1) ? v1 : (ptr == 2) ? v2 : (ptr == 3) ? v3 : (ptr == 4) ? v4 : -INFINITY;
              const int   ci = (ptr == 0) ? i0 : (ptr == 1) ? i1 : (ptr == 2) ? i2 : (ptr == 3) ? i3 : (ptr == 4) ? i4 : ivocab;
              const float bv = simd_max(cv);
              const int   bi = simd_min((cv == bv) ? ci : ivocab);
              out_v[k] = bv;
              out_i[k] = bi;
              if (ci == bi && bi < ivocab) { ptr++; }
            }
            threadgroup float sh_v[8][5];
            threadgroup int sh_i[8][5];
            threadgroup float sh_m[8];
            threadgroup float sh_s[8];
            if (lane == 0) {
              #pragma unroll
              for (int k = 0; k < 5; ++k) {
                sh_v[sg][k] = out_v[k];
                sh_i[sg][k] = out_i[k];
              }
              sh_m[sg] = sg_m;
              sh_s[sg] = sg_s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              float lm = (lane < 8) ? sh_m[lane] : -INFINITY;
              float ls = (lane < 8) ? sh_s[lane] : 0.0f;
              const float tg_m = simd_max(lm);
              const float tg_s = simd_sum(
                  (lm == -INFINITY) ? 0.0f : ls * metal::fast::exp2((lm - tg_m) * inv_t_log2e));
              if (lane < 8) {
                v0 = sh_v[lane][0]; i0 = sh_i[lane][0];
                v1 = sh_v[lane][1]; i1 = sh_i[lane][1];
                v2 = sh_v[lane][2]; i2 = sh_i[lane][2];
                v3 = sh_v[lane][3]; i3 = sh_i[lane][3];
                v4 = sh_v[lane][4]; i4 = sh_i[lane][4];
              } else {
                v0 = -INFINITY; v1 = -INFINITY; v2 = -INFINITY; v3 = -INFINITY; v4 = -INFINITY;
                i0 = ivocab; i1 = ivocab; i2 = ivocab; i3 = ivocab; i4 = ivocab;
              }
              ptr = 0;
              #pragma unroll
              for (int k = 0; k < 5; ++k) {
                const float cv = (ptr == 0) ? v0 : (ptr == 1) ? v1 : (ptr == 2) ? v2 : (ptr == 3) ? v3 : (ptr == 4) ? v4 : -INFINITY;
                const int   ci = (ptr == 0) ? i0 : (ptr == 1) ? i1 : (ptr == 2) ? i2 : (ptr == 3) ? i3 : (ptr == 4) ? i4 : ivocab;
                const float bv = simd_max(cv);
                const int   bi = simd_min((cv == bv) ? ci : ivocab);
                out_v[k] = bv;
                out_i[k] = bi;
                if (ci == bi && bi < ivocab) { ptr++; }
              }
              if (lane == 0) {
                const uint out_base = (row * CHUNKS + chunk) * 5u;
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                  p_val[out_base + k] = out_v[k];
                  p_idx[out_base + k] = out_i[k];
                }
                p_sum[row * CHUNKS + chunk] = tg_s;
              }
            }
            """,
        header: fastDynTreeHeader)

    private static let fastDynTreeStage2Kernel = MLXFast.metalKernel(
        name: "dflash_fast_dyntree_s2",
        inputNames: ["p_val", "p_idx", "p_sum", "tparams", "hist_in"],
        outputNames: ["tokens", "tree_meta"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int ivocab = int(V_);
            const float t0 = tparams[2];
            const float chain_bonus = tparams[3];
            const float decay = tparams[4];
            const int max_leaves = (K_ >= 6) ? int(tparams[10]) : ((K_ >= 5) ? int(tparams[7]) : int(tparams[5]));
            const int max_leaf_depth = max(1, int(tparams[6]));
            const int max_branch_len = max(1, int(tparams[8]));
            const float cont_decay = tparams[9];
            const int purity_mode = int(tparams[11]);
            const int dp_mode = int(tparams[12]);
            const int top_k_use = min(5, max(3, int(tparams[13])));
            const float t_slope = tparams[14];
            const float leaf_decay = tparams[15];
            const float purity_exp = tparams[16];

            threadgroup float sh_v[2][5];
            threadgroup int sh_i[2][5];
            threadgroup float sh_chunk_v0[64];
            threadgroup float sh_chunk_s[64];
            threadgroup float sh_sg_s[2];

            int row_ids[7][5];
            float row_probs[7][5];

            for (uint r = 0; r < uint(R_); ++r) {
              const float row_T = max(0.05f, t0 + float(r) * t_slope);
              const float inv_t_log2e = 1.44269504f / row_T;
              const uint base = (r * CHUNKS + lid) * 5u;
              float v0 = p_val[base + 0], v1 = p_val[base + 1], v2 = p_val[base + 2], v3 = p_val[base + 3], v4 = p_val[base + 4];
              int   i0 = p_idx[base + 0], i1 = p_idx[base + 1], i2 = p_idx[base + 2], i3 = p_idx[base + 3], i4 = p_idx[base + 4];
              sh_chunk_v0[lid] = v0;
              sh_chunk_s[lid] = p_sum[r * CHUNKS + lid];

              for (ushort off = 16; off > 0; off >>= 1) {
                const float ov0 = simd_shuffle_down(v0, off);
                const int   oi0 = simd_shuffle_down(i0, off);
                const float ov1 = simd_shuffle_down(v1, off);
                const int   oi1 = simd_shuffle_down(i1, off);
                const float ov2 = simd_shuffle_down(v2, off);
                const int   oi2 = simd_shuffle_down(i2, off);
                const float ov3 = simd_shuffle_down(v3, off);
                const int   oi3 = simd_shuffle_down(i3, off);
                const float ov4 = simd_shuffle_down(v4, off);
                const int   oi4 = simd_shuffle_down(i4, off);
                if (lane < off) {
                  dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov0, oi0, ivocab);
                  dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov1, oi1, ivocab);
                  dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov2, oi2, ivocab);
                  dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov3, oi3, ivocab);
                  dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov4, oi4, ivocab);
                }
              }
              if (lane == 0) {
                sh_v[sg][0] = v0; sh_i[sg][0] = i0;
                sh_v[sg][1] = v1; sh_i[sg][1] = i1;
                sh_v[sg][2] = v2; sh_i[sg][2] = i2;
                sh_v[sg][3] = v3; sh_i[sg][3] = i3;
                sh_v[sg][4] = v4; sh_i[sg][4] = i4;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (lid == 0) {
                dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, sh_v[1][0], sh_i[1][0], ivocab);
                dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, sh_v[1][1], sh_i[1][1], ivocab);
                dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, sh_v[1][2], sh_i[1][2], ivocab);
                dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, sh_v[1][3], sh_i[1][3], ivocab);
                dflash_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, sh_v[1][4], sh_i[1][4], ivocab);
                sh_v[0][0] = v0; sh_v[0][1] = v1; sh_v[0][2] = v2; sh_v[0][3] = v3; sh_v[0][4] = v4;
                sh_i[0][0] = i0; sh_i[0][1] = i1; sh_i[0][2] = i2; sh_i[0][3] = i3; sh_i[0][4] = i4;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              const float row_max = sh_v[0][0];
              const float c_v0 = sh_chunk_v0[lid];
              const float c_s = (c_v0 == -INFINITY)
                  ? 0.0f : sh_chunk_s[lid] * metal::fast::exp2((c_v0 - row_max) * inv_t_log2e);
              const float sg_sum = simd_sum(c_s);
              if (lane == 0) { sh_sg_s[sg] = sg_sum; }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (lid == 0) {
                const float total_s = max(1e-20f, sh_sg_s[0] + sh_sg_s[1]);
                const float inv_s = 1.0f / total_s;
                row_ids[r][0] = sh_i[0][0];
                row_ids[r][1] = sh_i[0][1];
                row_ids[r][2] = sh_i[0][2];
                row_ids[r][3] = sh_i[0][3];
                row_ids[r][4] = sh_i[0][4];
                row_probs[r][0] = inv_s;
                for (int c = 1; c < 5; ++c) {
                  row_probs[r][c] = (sh_v[0][c] == -INFINITY)
                      ? 0.0f : metal::fast::exp2((sh_v[0][c] - row_max) * inv_t_log2e) * inv_s;
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
              for (int s = 0; s < R_; ++s) {
                tokens[s] = row_ids[s][0];
              }
              float surv[7];
              float leaf_mult[7];
              surv[0] = 1.0f;
              leaf_mult[0] = 1.0f;
              float ld_pow = 1.0f;
              for (int r = 1; r < R_; ++r) {
                surv[r] = surv[r - 1] * row_probs[r - 1][0] * decay;
                ld_pow *= leaf_decay;
                const float pm = (purity_exp > 0.0f)
                    ? metal::pow(max(surv[r], 1e-6f), purity_exp)
                    : ((purity_mode != 0) ? surv[r] : 1.0f);
                leaf_mult[r] = ld_pow * pm;
              }
              int   n_parent[8];
              int   n_depth[8];
              int   n_token[8];
              float n_ev[8];
              int   n_bstep[8];
              bool  n_cont_used[8];
              int   chain_node[8];
              n_parent[0] = -1;
              n_depth[0] = 0;
              n_token[0] = 0;
              n_ev[0] = 1.0f;
              n_bstep[0] = 0;
              n_cont_used[0] = false;
              chain_node[0] = 0;

              int chain_len = 0;
              int num_nodes = 1;

              if (dp_mode != 0) {
                const int h_len = min(7, max(0, hist_in[0]));
                const float h_p0 = float(hist_in[1]) * 1e-6f;
                const float h_pc = float(hist_in[2]) * 1e-6f;
                int h_tok[7];
                for (int r = 0; r < 7; ++r) { h_tok[r] = hist_in[3 + r]; }
                int h_agree = 0;
                while (h_agree < R_ && h_agree < h_len && h_tok[h_agree] == row_ids[h_agree][0]) {
                  h_agree += 1;
                }
                float h_chain_ev[7];
                for (int r = 0; r < 7; ++r) { h_chain_ev[r] = 0.0f; }
                if (h_agree > 0) {
                  h_chain_ev[0] = h_p0;
                  for (int r = 1; r < h_agree; ++r) {
                    h_chain_ev[r] = h_chain_ev[r - 1] * h_pc;
                  }
                }
                const int max_L = min(K_, R_);
                const int min_L = min(max_L, (h_len > 0)
                    ? max((chain_bonus > 0.0f) ? 1 : 2, K_ - max(0, max_leaves))
                    : ((chain_bonus > 0.0f) ? 1 : max(2, K_ - max(0, max_leaves))));
                float best_total = -1.0f;
                int best_L = max_L;
                int best_bcnt = 0;
                int best_par[6];
                int best_dep[6];
                int best_tok[6];
                float best_ev[6];
                int best_bs[6];
                for (int L = max_L; L >= min_L; --L) {
                  float total = 0.0f;
                  for (int r = 0; r < L; ++r) {
                    const float c_ev = (r < h_agree)
                        ? max(surv[r] * row_probs[r][0], h_chain_ev[r])
                        : (surv[r] * row_probs[r][0]);
                    total += c_ev * chain_bonus;
                  }
                  const int budget = min(K_ - L, max(0, max_leaves));
                  const int leaf_limit = min(L, max_leaf_depth);
                  bool used[7][5];
                  for (int r = 0; r < 7; ++r) {
                    for (int c = 0; c < 5; ++c) { used[r][c] = false; }
                  }
                  const int h_div = h_agree;
                  bool h_placed = false;
                  int h_node_idx = -1;
                  int h_next_pos = h_div;
                  int cur_bcnt = 0;
                  int cur_par[6];
                  int cur_dep[6];
                  int cur_tok[6];
                  float cur_ev[6];
                  int cur_bs[6];
                  bool cur_cont[6];
                  int cur_hidx[6];
                  for (int b = 0; b < budget; ++b) {
                    float bev = -1.0f;
                    int bkind = -1, ba = -1, bb = -1;
                    for (int r = 0; r < leaf_limit; ++r) {
                      const float base_w = surv[r] * leaf_mult[r];
                      for (int c = 1; c < top_k_use; ++c) {
                        if (!used[r][c] && row_ids[r][c] < ivocab) {
                          const float ev = base_w * row_probs[r][c];
                          if (ev > bev) {
                            bev = ev;
                            bkind = 1;
                            ba = r;
                            bb = c;
                          }
                        }
                      }
                    }
                    if (max_branch_len > 1 && cont_decay > 0.0f) {
                      for (int j = 0; j < cur_bcnt; ++j) {
                        if (cur_hidx[j] < 0 && cur_bs[j] >= 1 && cur_bs[j] < max_branch_len
                            && !cur_cont[j] && cur_dep[j] < R_) {
                          const int d = cur_dep[j];
                          if (row_ids[d][0] < ivocab) {
                            const float ev = cur_ev[j] * row_probs[d][0] * cont_decay;
                            if (ev > bev) {
                              bev = ev;
                              bkind = 2;
                              ba = j;
                              bb = d;
                            }
                          }
                        }
                      }
                    }
                    if (h_len > 0 && !h_placed && h_div < h_len && h_div <= L
                        && h_tok[h_div] >= 0 && h_tok[h_div] < ivocab) {
                      int dup_j = -1;
                      for (int j = 0; j < cur_bcnt; ++j) {
                        if (cur_par[j] == h_div && cur_tok[j] == h_tok[h_div]) {
                          dup_j = j;
                          break;
                        }
                      }
                      if (dup_j >= 0) {
                        h_placed = true;
                        h_node_idx = dup_j;
                        cur_hidx[dup_j] = h_div;
                        h_next_pos = h_div + 1;
                      } else {
                        const float h_base = (h_div == 0)
                            ? h_p0 : (surv[min(h_div, R_ - 1)] * h_pc);
                        float h_ev = h_base;
                        int match_c = -1;
                        if (h_div < R_) {
                          const float base_w = surv[h_div] * leaf_mult[h_div];
                          for (int c = 1; c < top_k_use; ++c) {
                            if (row_ids[h_div][c] == h_tok[h_div]) {
                              match_c = c;
                              h_ev = max(h_ev, base_w * row_probs[h_div][c]);
                              break;
                            }
                          }
                        }
                        if (h_ev > bev) {
                          bev = h_ev;
                          bkind = 3;
                          ba = h_div;
                          bb = match_c;
                        }
                      }
                    }
                    if (h_len > 0 && h_placed && h_node_idx >= 0 && h_next_pos < h_len
                        && !cur_cont[h_node_idx] && cur_dep[h_node_idx] < K_) {
                      const int ht = h_tok[h_next_pos];
                      if (ht >= 0 && ht < ivocab) {
                        const float ev = cur_ev[h_node_idx] * h_pc;
                        if (ev > bev) {
                          bev = ev;
                          bkind = 4;
                          ba = h_node_idx;
                          bb = h_next_pos;
                        }
                      }
                    }
                    if (bkind == 1 && bev > 0.0f) {
                      used[ba][bb] = true;
                      total += bev;
                      cur_par[cur_bcnt] = ba;
                      cur_dep[cur_bcnt] = ba + 1;
                      cur_tok[cur_bcnt] = row_ids[ba][bb];
                      cur_ev[cur_bcnt] = bev;
                      cur_bs[cur_bcnt] = 1;
                      cur_cont[cur_bcnt] = false;
                      cur_hidx[cur_bcnt] = -1;
                      cur_bcnt += 1;
                    } else if (bkind == 2 && bev > 0.0f) {
                      cur_cont[ba] = true;
                      total += bev;
                      cur_par[cur_bcnt] = L + 1 + ba;
                      cur_dep[cur_bcnt] = bb + 1;
                      cur_tok[cur_bcnt] = row_ids[bb][0];
                      cur_ev[cur_bcnt] = bev;
                      cur_bs[cur_bcnt] = cur_bs[ba] + 1;
                      cur_cont[cur_bcnt] = false;
                      cur_hidx[cur_bcnt] = -1;
                      cur_bcnt += 1;
                    } else if (bkind == 3 && bev > 0.0f) {
                      if (bb >= 1 && ba < 7) { used[ba][bb] = true; }
                      total += bev;
                      cur_par[cur_bcnt] = ba;
                      cur_dep[cur_bcnt] = ba + 1;
                      cur_tok[cur_bcnt] = h_tok[ba];
                      cur_ev[cur_bcnt] = bev;
                      cur_bs[cur_bcnt] = 1;
                      cur_cont[cur_bcnt] = false;
                      cur_hidx[cur_bcnt] = ba;
                      h_placed = true;
                      h_node_idx = cur_bcnt;
                      h_next_pos = ba + 1;
                      cur_bcnt += 1;
                    } else if (bkind == 4 && bev > 0.0f) {
                      cur_cont[ba] = true;
                      total += bev;
                      cur_par[cur_bcnt] = L + 1 + ba;
                      cur_dep[cur_bcnt] = cur_dep[ba] + 1;
                      cur_tok[cur_bcnt] = h_tok[bb];
                      cur_ev[cur_bcnt] = bev;
                      cur_bs[cur_bcnt] = cur_bs[ba] + 1;
                      cur_cont[cur_bcnt] = false;
                      cur_hidx[cur_bcnt] = bb;
                      h_node_idx = cur_bcnt;
                      h_next_pos = bb + 1;
                      cur_bcnt += 1;
                    } else {
                      break;
                    }
                  }
                  if (total > best_total) {
                    best_total = total;
                    best_L = L;
                    best_bcnt = cur_bcnt;
                    for (int b = 0; b < cur_bcnt; ++b) {
                      best_par[b] = cur_par[b];
                      best_dep[b] = cur_dep[b];
                      best_tok[b] = cur_tok[b];
                      best_ev[b] = cur_ev[b];
                      best_bs[b] = cur_bs[b];
                    }
                  }
                }
                chain_len = best_L;
                for (int r = 0; r < chain_len; ++r) {
                  const int id = num_nodes++;
                  n_parent[id] = chain_node[r];
                  n_depth[id] = r + 1;
                  n_token[id] = row_ids[r][0];
                  n_ev[id] = surv[r] * row_probs[r][0];
                  n_bstep[id] = 0;
                  n_cont_used[id] = false;
                  chain_node[r + 1] = id;
                }
                for (int b = 0; b < best_bcnt && num_nodes <= K_; ++b) {
                  const int id = num_nodes++;
                  n_parent[id] = (best_par[b] <= chain_len) ? chain_node[best_par[b]] : best_par[b];
                  n_depth[id] = best_dep[b];
                  n_token[id] = best_tok[b];
                  n_ev[id] = best_ev[b];
                  n_bstep[id] = best_bs[b];
                  n_cont_used[id] = false;
                }
              } else {
                bool chosen[7][5];
                for (int r = 0; r < 7; ++r) {
                  for (int c = 0; c < 5; ++c) { chosen[r][c] = false; }
                }
                const int min_chain = (chain_bonus > 0.0f)
                    ? 1 : min(K_, max(2, K_ - max(0, max_leaves)));
                chain_len = min(min_chain, R_);
                int leaves_used = 0;
                for (int r = 0; r < chain_len; ++r) {
                  const int id = num_nodes++;
                  n_parent[id] = chain_node[r];
                  n_depth[id] = r + 1;
                  n_token[id] = row_ids[r][0];
                  n_ev[id] = surv[r] * row_probs[r][0];
                  n_bstep[id] = 0;
                  n_cont_used[id] = false;
                  chain_node[r + 1] = id;
                  chosen[r][0] = true;
                }
                for (int step = chain_len; step < K_; ++step) {
                  float best_ev = -1.0f;
                  int best_kind = -1;
                  int best_a = -1;
                  int best_b = -1;
                  if (chain_len < R_ && row_ids[chain_len][0] < ivocab) {
                    const float ev_chain = surv[chain_len] * row_probs[chain_len][0] * chain_bonus;
                    best_ev = ev_chain;
                    best_kind = 0;
                    best_a = chain_len;
                  }
                  if (leaves_used < max_leaves) {
                    const int leaf_limit = min(chain_len, max_leaf_depth);
                    for (int r = 0; r < leaf_limit; ++r) {
                      const float base_w = surv[r] * leaf_mult[r];
                      for (int c = 1; c < top_k_use; ++c) {
                        if (!chosen[r][c] && row_ids[r][c] < ivocab) {
                          const float ev = base_w * row_probs[r][c];
                          if (ev > best_ev) {
                            best_ev = ev;
                            best_kind = 1;
                            best_a = r;
                            best_b = c;
                          }
                        }
                      }
                    }
                    if (max_branch_len > 1 && cont_decay > 0.0f) {
                      for (int u = 1; u < num_nodes; ++u) {
                        if (n_bstep[u] >= 1 && n_bstep[u] < max_branch_len
                            && !n_cont_used[u] && n_depth[u] < R_) {
                          const int d = n_depth[u];
                          if (row_ids[d][0] < ivocab) {
                            const float ev = n_ev[u] * row_probs[d][0] * cont_decay;
                            if (ev > best_ev) {
                              best_ev = ev;
                              best_kind = 2;
                              best_a = u;
                            }
                          }
                        }
                      }
                    }
                  }
                  if (best_kind == 0 && num_nodes <= K_) {
                    const int id = num_nodes++;
                    n_parent[id] = chain_node[chain_len];
                    n_depth[id] = chain_len + 1;
                    n_token[id] = row_ids[chain_len][0];
                    n_ev[id] = surv[chain_len] * row_probs[chain_len][0];
                    n_bstep[id] = 0;
                    n_cont_used[id] = false;
                    chosen[chain_len][0] = true;
                    chain_len += 1;
                    chain_node[chain_len] = id;
                  } else if (best_kind == 1 && num_nodes <= K_) {
                    const int id = num_nodes++;
                    n_parent[id] = chain_node[best_a];
                    n_depth[id] = best_a + 1;
                    n_token[id] = row_ids[best_a][best_b];
                    n_ev[id] = best_ev;
                    n_bstep[id] = 1;
                    n_cont_used[id] = false;
                    chosen[best_a][best_b] = true;
                    leaves_used += 1;
                  } else if (best_kind == 2 && num_nodes <= K_) {
                    const int id = num_nodes++;
                    const int d = n_depth[best_a];
                    n_parent[id] = best_a;
                    n_depth[id] = d + 1;
                    n_token[id] = row_ids[d][0];
                    n_ev[id] = best_ev;
                    n_bstep[id] = n_bstep[best_a] + 1;
                    n_cont_used[id] = false;
                    n_cont_used[best_a] = true;
                    leaves_used += 1;
                  }
                }
              }

              int old_to_new[8];
              int parent[8];
              int depth[8];
              uint full_anc[8];
              uint anc[8];
              for (int s = 0; s < 8; ++s) {
                old_to_new[s] = s;
                parent[s] = s - 1;
                depth[s] = s;
                full_anc[s] = 0u;
                anc[s] = 0u;
              }
              old_to_new[0] = 0;
              parent[0] = -1;
              depth[0] = 0;
              full_anc[0] = 0u;
              anc[0] = 0u;
              for (int s = 1; s <= chain_len; ++s) {
                const int u = chain_node[s];
                old_to_new[u] = s;
                tokens[s - 1] = n_token[u];
                parent[s] = s - 1;
                depth[s] = s;
                full_anc[s] = full_anc[s - 1] | (uint(s) << (s * 4));
                anc[s] = 0u;
              }
              int next_slot = chain_len + 1;
              for (int u = 1; u < num_nodes; ++u) {
                if (n_bstep[u] > 0 && next_slot <= K_) {
                  const int s = next_slot++;
                  const int p = old_to_new[n_parent[u]];
                  const int d = depth[p] + 1;
                  old_to_new[u] = s;
                  tokens[s - 1] = n_token[u];
                  parent[s] = p;
                  depth[s] = d;
                  full_anc[s] = full_anc[p] | (uint(s) << (d * 4));
                  anc[s] = full_anc[s];
                }
              }
              uint packed_pos = 0u;
              for (int s = 0; s < 8; ++s) {
                packed_pos |= (uint(depth[s] & 0xF) << (s * 4));
              }
              uint packed_parents = uint(chain_len & 0xF) << 28;
              for (int s = 1; s <= K_; ++s) {
                packed_parents |= (uint(parent[s] & 0xF) << ((s - 1) * 4));
              }
              tree_meta[0] = int(packed_pos);
              for (int s = 0; s < 8; ++s) {
                tree_meta[1 + s] = int(anc[s]);
                tree_meta[9 + s] = depth[s];
              }
              tree_meta[17] = int(packed_parents);
            }
            """,
        header: fastDynTreeHeader)

    private static let fastDynTreeArgMaxKernel = MLXFast.metalKernel(
        name: "dflash_fast_dyntree_argmax",
        inputNames: ["p_val", "p_idx"],
        outputNames: ["tokens"],
        source: """
            constexpr uint CHUNKS = 64;
            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int ivocab = int(V_);
            const uint base = (row * CHUNKS + lid) * 5u;
            const float v = p_val[base];
            const int id = p_idx[base];
            const float sg_val = simd_max(v);
            const int sg_idx = simd_min((v == sg_val) ? id : ivocab);
            threadgroup float sh_val[2];
            threadgroup int sh_idx[2];
            if (lane == 0) {
              sh_val[sg] = sg_val;
              sh_idx[sg] = sg_idx;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float fv = (lane < 2) ? sh_val[lane] : -INFINITY;
              const int fid = (lane < 2) ? sh_idx[lane] : ivocab;
              const float rv = simd_max(fv);
              const int rid = simd_min((fv == rv) ? fid : ivocab);
              if (lane == 0) {
                tokens[row] = rid;
              }
            }
            """)

    private static let treeOutcomeKernel = MLXFast.metalKernel(
        name: "dflash_tree_outcome",
        inputNames: ["drafts", "targets", "tree_meta"],
        outputNames: ["carry_slot", "accepted", "visible"],
        source: """
            const uint packed = uint(tree_meta[17]);
            int parent[8];
            parent[0] = -1;
            for (int s = 1; s <= K_; ++s) {
              const int p = int((packed >> ((s - 1) * 4)) & 0xFu);
              parent[s] = min(p, s - 1);
            }
            bool vis[8];
            for (int s = 0; s <= K_; ++s) { vis[s] = false; }
            int cur = 0;
            int acc = 0;
            vis[0] = true;
            for (int step = 0; step < K_; ++step) {
              const int want = targets[cur];
              int next_slot = -1;
              for (int s = cur + 1; s <= K_; ++s) {
                if (parent[s] == cur && drafts[s - 1] == want) {
                  next_slot = s;
                  break;
                }
              }
              if (next_slot < 0) { break; }
              cur = next_slot;
              vis[cur] = true;
              acc += 1;
            }
            carry_slot[0] = cur;
            accepted[0] = acc;
            for (int s = 0; s <= K_; ++s) {
              visible[s] = vis[s];
            }
            """)

    private struct DynTreeS1Key: Hashable {
        let rows: Int
        let vocab: Int
    }
    private struct DynTreeS2Key: Hashable {
        let rows: Int
        let vocab: Int
        let k: Int
    }
    private static let dynTreeLock = NSLock()
    nonisolated(unsafe) private static var dynTreeS1Cache: [DynTreeS1Key: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var dynTreeS2Cache: [DynTreeS2Key: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var dynTreeArgMaxCache: [DynTreeS1Key: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var treeOutcomeCache: [Int: MLXFast.MLXFastKernel.Launch] = [:]

    private static func fastTreeOutcome(
        drafts: MLXArray, targets: MLXArray, treeMeta: MLXArray, k: Int
    ) -> (carrySlot: MLXArray, accepted: MLXArray, visible: MLXArray) {
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = dynTreeLock.withLock({ treeOutcomeCache[k] }) {
            launch = hit
        } else {
            let made = treeOutcomeKernel.prepare(
                template: [("K_", k)],
                grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1, 1], [1], [k + 1]],
                outputDTypes: [.int32, .int32, .bool])
            dynTreeLock.withLock { treeOutcomeCache[k] = made }
            launch = made
        }
        let out = launch([drafts, targets, treeMeta])
        return (out[0], out[1].squeezed(), out[2])
    }

    private static func fastDynamicTreeStage1(
        _ logits: MLXArray, params: MLXArray
    ) -> (val: MLXArray, idx: MLXArray, sum: MLXArray) {
        let rows = logits.dim(0)
        let vocab = logits.dim(1)
        let chunks = 64
        let key = DynTreeS1Key(rows: rows, vocab: vocab)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = dynTreeLock.withLock({ dynTreeS1Cache[key] }) {
            launch = hit
        } else {
            let made = fastDynTreeStage1Kernel.prepare(
                template: [("V_", vocab)],
                grid: (256 * chunks, rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, chunks, 5], [rows, chunks, 5], [rows, chunks]],
                outputDTypes: [.float32, .int32, .float32])
            dynTreeLock.withLock { dynTreeS1Cache[key] = made }
            launch = made
        }
        let partials = launch([logits, params])
        return (partials[0], partials[1], partials[2])
    }

    nonisolated(unsafe) private static let emptyHistoryHint: MLXArray = {
        let a = MLXArray.zeros([10], type: Int32.self)
        eval(a)
        return a
    }()

    private static func fastDynamicTreeStage2(
        _ pVal: MLXArray, _ pIdx: MLXArray, _ pSum: MLXArray,
        k: Int, vocab: Int, params: MLXArray, historyHint: MLXArray? = nil
    ) -> (tokens: MLXArray, treeMeta: MLXArray) {
        let rows = pVal.dim(0)
        let chunks = 64
        let key = DynTreeS2Key(rows: rows, vocab: vocab, k: k)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = dynTreeLock.withLock({ dynTreeS2Cache[key] }) {
            launch = hit
        } else {
            let made = fastDynTreeStage2Kernel.prepare(
                template: [("V_", vocab), ("R_", rows), ("K_", k)],
                grid: (chunks, 1, 1), threadGroup: (chunks, 1, 1),
                outputShapes: [[rows], [18]],
                outputDTypes: [.int32, .int32])
            dynTreeLock.withLock { dynTreeS2Cache[key] = made }
            launch = made
        }
        let hint = historyHint ?? emptyHistoryHint
        let out = launch([pVal, pIdx, pSum, params, hint])
        return (out[0], out[1])
    }

    private static func fastDynamicTreeArgMax(
        _ pVal: MLXArray, _ pIdx: MLXArray, vocab: Int
    ) -> MLXArray {
        let rows = pVal.dim(0)
        let chunks = 64
        let key = DynTreeS1Key(rows: rows, vocab: vocab)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = dynTreeLock.withLock({ dynTreeArgMaxCache[key] }) {
            launch = hit
        } else {
            let made = fastDynTreeArgMaxKernel.prepare(
                template: [("V_", vocab)],
                grid: (chunks, rows, 1), threadGroup: (chunks, 1, 1),
                outputShapes: [[rows]],
                outputDTypes: [.int32])
            dynTreeLock.withLock { dynTreeArgMaxCache[key] = made }
            launch = made
        }
        return launch([pVal, pIdx])[0]
    }

    private static func fastDynamicTree(
        _ logits: MLXArray, k: Int, params: MLXArray
    ) -> (tokens: MLXArray, treeMeta: MLXArray) {
        let p = fastDynamicTreeStage1(logits, params: params)
        return fastDynamicTreeStage2(
            p.val, p.idx, p.sum, k: k, vocab: logits.dim(1), params: params)
    }

    /// The argmax of the head products, or a selection's proposals over the
    /// target's readout of them when it gives one per row.
    private func proposals(_ logits: MLXArray, selection: CBv2MTPProposalSelection?) -> MLXArray {
        if let selection, let coupled = selection.propose(target.dflashReadout(logits)),
            coupled.shape == [logits.dim(0)]
        {
            return coupled.asType(.int32)
        }
        if logits.ndim == 2, logits.dtype == .bfloat16, logits.dim(1) >= 256 {
            let dynK = CBv2DraftRegime.activeDynamicTreeK
            if dynK >= 2, dynK <= logits.dim(0), logits.dim(0) <= 7 {
                let dyn = Self.fastDynamicTree(logits, k: dynK, params: dynTreeParams)
                CBv2DraftRegime.activeDynamicTreeMeta = dyn.treeMeta
                return dyn.tokens
            }
            return Self.fastArgMax(logits)
        }
        return argMax(logits, axis: -1).asType(.int32)
    }
    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return [] }
        return s.arrays
    }

    /// Ends a request whose target rows or round calls did not fit.
    public func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { throw MuseGlimmerDFlashError.state("foreign or released") }
        if let refusal = s.refusal { throw MuseGlimmerDFlashError.state(refusal) }
    }

    /// The accepted drafts' trusted target rows become context. The row of
    /// the last committed position follows as the next carry.
    public func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        let s = state(requestState)
        defer { clearRound(s) }
        let speculation = s.speculation
        s.speculation = nil
        s.lastRoundFullyAccepted = false
        let pendingBefore = s.pending
        guard s.refusal == nil else { return }
        let rows = confirmedInputTokens - 1
        guard s.stagedInputCount > 0, rows >= 0, rows <= s.stagedInputCount,
            committedTargetHidden.ndim == 3, committedTargetHidden.dim(1) == rows
        else {
            s.refusal = "finalize"
            return
        }
        if rows > 0 {
            guard let accepted = features(committedTargetHidden) else {
                s.refusal = "accepted rows"
                return
            }
            if settings.batchesRequests {
                // Committed with the next round's carry, in its pass.
                s.pending = s.pending.map { concatenated([$0, accepted], axis: 0) } ?? accepted
            } else {
                commitToContext(features: accepted, at: s.context.nextPosition, to: s.context)
            }
        }
        s.awaitsCarry = true
        s.committedInputCount += rows
        s.lastRoundFullyAccepted = rows == s.stagedInputCount
        // Keep the speculation for this outcome: the pending rows exactly
        // this round's first `rows` verify rows (none when `rows` is 0).
        if var kept = speculation, settings.batchesRequests, pendingBefore == nil,
            kept.depth == s.stagedInputCount, rows <= kept.depth,
            (rows == 0) == (s.pending == nil)
        {
            kept.accepted = rows
            kept.pending = s.pending
            s.speculation = kept
        }
    }

    /// Only the memo belongs to a round: a block never writes the context.
    public func discardRound(requestState: any CBv2MTPRequestState) {
        let s = state(requestState)
        s.speculation = nil
        clearRound(s)
    }

    private func clearRound(_ s: State) {
        s.memo = nil
        s.selection = nil
        s.prestarted = false
        s.stagedInputCount = 0
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return }
        clearRound(s)
        s.speculation = nil
        s.pending = nil
        s.context.release()
        s.awaitsCarry = false
        s.committedInputCount = 0
        s.released = true
    }
}

// MARK: - Admission

extension MuseGlimmerDFlashDrafter: CBv2MTPBoundedAllocationProviding {
    /// Explicit arrays of one context commit of `rows` rows: the feature rows
    /// in the drafter dtype, the projected context before and after its norm,
    /// and each layer's keys (projected, normed, rotated) and values.
    func commitBytes(rows: Int) -> Int {
        let c = model.configuration
        let size = model.activationDType.size
        let keyWidth = c.kvHeads * c.headDimensions
        return rows * size * (c.contextWidth + 2 * c.hiddenSize + c.hiddenLayers * 4 * keyWidth)
    }

    /// Explicit arrays of one block forward and its head rows. Per layer: the
    /// keys and values handed to attention (held rows and the block), float32
    /// scores, queries, the attention output and the MLP's three products.
    /// Then float32 head rows and their argmax, and the coupled race when
    /// it is on.
    var blockBytes: Int {
        let c = model.configuration
        let size = model.activationDType.size
        let keys = c.slidingWindow + c.blockSize
        let layer =
            2 * c.kvHeads * keys * c.headDimensions * size
            + c.attentionHeads * c.blockSize * keys * 4
            + c.blockSize * size
            * (3 * c.attentionHeads * c.headDimensions + 6 * c.hiddenSize + 3 * c.intermediateSize)
        let head =
            settings.maximumDraftTokens * (target.dflashVocabularySize * 4 + c.hiddenSize * 4)
        return c.hiddenLayers * layer + head + coupledProposalBytes
            + c.blockSize * (keys + c.hiddenSize * size)
    }

    /// The coupled race over the head rows (`couplesSampledProposals`), all
    /// float32 at worst: the readout's four steps (multiplier, scale, tanh,
    /// scale back) and the race's seven arrays (cast, temperature, softmax,
    /// noise, their ratio, the above-winner mask and its masked mass).
    var coupledProposalBytes: Int {
        settings.couplesSampledProposals
            ? 11 * settings.maximumDraftTokens * target.dflashVocabularySize * 4 : 0
    }

    /// A round's accepted rows held for the next round's pass
    /// (`batchesRequests`): a view of the round's target rows, which can keep
    /// every row of that round alive, charged at float32.
    var pendingBytes: Int {
        settings.batchesRequests
            ? settings.maximumSpeculativeRows * settings.maximumDraftTokens
                * model.configuration.contextWidth * 4 : 0
    }

    /// Per request: the ring's two buffers, one copy of each for a write MLX
    /// cannot make in place, and one commit of at most `window` rows (older
    /// rows of a longer chunk are never projected), plus the pending rows
    /// when requests are batched. Once per engine: one block forward for
    /// each request a round may hold. The ring does not grow with the
    /// request, so nothing is charged per token.
    public func boundedRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        let c = model.configuration
        guard limits.maximumPrefillTokens > 0, limits.maximumDraftTokens > 0 else { return nil }
        let buffer =
            MuseGlimmerDFlashContextState.residentBytes(
                configuration: c, dtype: model.activationDType) / 2
        let commitRows = min(max(limits.maximumPrefillTokens, c.blockSize), c.slidingWindow)
        // A speculative block can still be in flight when its round's
        // normal block is built.
        var owner = [
            CBv2MTPFixedBufferSpec(
                logicalBytes: blockBytes,
                allocationCount: settings.maximumSpeculativeRows + (speculatesAfterVerify ? 1 : 0))
        ]
        if speculatesAfterVerify {
            // The speculation's float32 readout of its head products and one
            // elementwise temporary, for the draft confidence.
            owner.append(
                .init(
                    logicalBytes: settings.maximumDraftTokens * target.dflashVocabularySize * 4,
                    allocationCount: 2))
        }
        if settings.backendScratchBytes > 0 {
            owner.append(.init(logicalBytes: settings.backendScratchBytes))
        }
        var resident = [CBv2MTPFixedBufferSpec(logicalBytes: buffer, allocationCount: 2)]
        if pendingBytes > 0 { resident.append(.init(logicalBytes: pendingBytes)) }
        return CBv2MTPBoundedAllocationSpec(
            resident: resident,
            // The speculation's copy of the ring (`speculateNextRound`) is a
            // second pair of copies next to an in-flight commit's.
            working: [
                .init(logicalBytes: buffer, allocationCount: speculatesAfterVerify ? 4 : 2),
                .init(logicalBytes: commitBytes(rows: commitRows)),
            ],
            hostBytes: c.blockSize * (c.slidingWindow + c.blockSize),
            ownerWorking: owner)
    }
}

// MARK: - Parameter residency

/// The drafter's own buffers (pinned and re-quantized layers, the
/// low-acceptance MLP copies) in a residency set beside the target's; the
/// draft head is the target's and already registered. Memory placement only.
extension MuseGlimmerDFlashDrafter: CBv2ResidentParametersProviding {
    public var residentParameterKey: ObjectIdentifier { ObjectIdentifier(model) }
    public func residentParameterArrays() -> [MLXArray] {
        var arrays = model.parameters().flattened().map(\.1)
        func addQL(_ ql: QuantizedLinear?) {
            guard let ql else { return }
            arrays += [ql.weight, ql.scales]
            if let biases = ql.biases { arrays.append(biases) }
        }
        for low in lowMLP {
            addQL(low.gate)
            addQL(low.up)
            addQL(low.gateUp)
            addQL(low.down)
        }
        addQL(fusedContextKV)
        for qkv in fusedBlockQKV { addQL(qkv) }
        addQL(fusedLastKV)
        if let nv = nvfp4Drafter {
            arrays += nv.residentArrays
        }
        if let comp = compressedDraftHead {
            arrays += comp.residentArrays
        }
        return arrays + [maskEmbedding]
    }
}

// MARK: - 2-Stage Compressed Draft Head (Low-Acceptance Dynamic Tree)

extension MuseGlimmerDFlashDrafter {
    /// Read-only 2-stage compressed draft head built once at load time from the
    /// frozen target NVFP4 `lm_head` (`ModelOptWeightOnlyNVFP4Linear`):
    /// - Load-time 181 MB index: TG-contiguous, strided-vocab 1-bit weight sign
    ///   bitplanes (`w1Bits4`) + per-256-dim L1 weight scales (`g256Sc`).
    /// - Stage 1 (`dflash_xnor_quant_x2` + `dflash_xnor_stage1`): 2-bit optimal
    ///   Gaussian Lloyd-Max activation quantization + 512-TG coalesced `uint4`
    ///   popcount scan selecting 1,024 distinct candidates per draft row.
    /// - Stage 2 (`dflash_xnor_stage2_dyntree`): exact NVFP4 rescore of those
    ///   1,024 candidates per row directly into `(p_val, p_idx, p_sum)` for
    ///   `fastDynamicTreeStage2`, eliminating the `[R, V]` BF16 logit tensor and
    ///   cutting draft head traffic from ~762 MB/rd to ~208 MB/rd.
    /// Target weights are strictly read-only; used only on low-acceptance dynamic
    /// tree rounds (`CBV2_DRAFTER_XNOR_HEAD=0` disables).
    private final class CompressedDraftHead: @unchecked Sendable {
        private static let enabled: Bool =
            ProcessInfo.processInfo.environment["CBV2_DRAFTER_XNOR_HEAD"] != "0"
        private static let centeredEnabled: Bool =
            ProcessInfo.processInfo.environment["CBV2_DRAFTER_XNOR_CENTER"] == "1"

        private static let header = """
            inline half dflash_xnor_e4m3(uint bits) {
                const uint v = bits & 127u;
                const uint u = (v << 7) | (((v + 1u) >> 7) << 14) | ((bits & 128u) << 8);
                return as_type<half>(ushort(u)) * 256.0h;
            }

            inline float dflash_xnor_e2m1_abs(uint nib) {
                const uint p = (nib & 7u) << 9;
                return float(as_type<half>(ushort(p)) * 16384.0h);
            }

            inline float dflash_xnor_e2m1_signed(uint nib) {
                const uint p = ((nib & 7u) << 9) | ((nib & 8u) << 12);
                return float(as_type<half>(ushort(p)) * 16384.0h);
            }

            inline void dflash_xnor_insert_top2(
                thread float& v0, thread int& i0,
                thread float& v1, thread int& i1,
                float v, int idx
            ) {
                if (idx < 0 || !(v > v1)) { return; }
                if (v > v0) {
                    v1 = v0; i1 = i0;
                    v0 = v;  i0 = idx;
                } else {
                    v1 = v;  i1 = idx;
                }
            }

            inline void dflash_xnor_insert_top5(
                thread float& v0, thread int& i0,
                thread float& v1, thread int& i1,
                thread float& v2, thread int& i2,
                thread float& v3, thread int& i3,
                thread float& v4, thread int& i4,
                float v, int idx, int ivocab
            ) {
                if (idx >= ivocab || idx < 0) { return; }
                if (!(v > v4 || (v == v4 && idx < i4))) { return; }
                if (v > v0 || (v == v0 && idx < i0)) {
                    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = v; i0 = idx;
                } else if (v > v1 || (v == v1 && idx < i1)) {
                    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v; i1 = idx;
                } else if (v > v2 || (v == v2 && idx < i2)) {
                    v4 = v3; i4 = i3; v3 = v2; i3 = i2; v2 = v; i2 = idx;
                } else if (v > v3 || (v == v3 && idx < i3)) {
                    v4 = v3; i4 = i3; v3 = v; i3 = idx;
                } else {
                    v4 = v; i4 = idx;
                }
            }

            inline float dflash_xnor_scale_logit(float raw, float eff_scale, float cap) {
                float v = raw * eff_scale;
                if (cap > 0.0f) { v = cap * metal::fast::tanh(v / cap); }
                return v;
            }
            """

        private static let colStatsKernel = MLXFast.metalKernel(
            name: "dflash_xnor_col_stats",
            inputNames: ["w", "scales"],
            outputNames: ["col_mu", "col_rms", "col_inv_rms"],
            source: """
                const uint lid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint g = threadgroup_position_in_grid.y;
                if (g >= uint(K_ / 16)) { return; }

                float s[16];
                #pragma unroll
                for (uint j = 0u; j < 16u; ++j) { s[j] = 0.0f; }
                for (uint v = lid; v < uint(V_); v += 256u) {
                    const uint2 qw = *(const device uint2*)(w + size_t(v) * size_t(K_ / 8) + size_t(g) * 2u);
                    const float sc = float(dflash_xnor_e4m3(scales[size_t(v) * size_t(K_ / 16) + size_t(g)]));
                    #pragma unroll
                    for (uint n = 0u; n < 8u; ++n) {
                        s[n] += dflash_xnor_e2m1_signed((qw.x >> (n * 4u)) & 0xFu) * sc;
                        s[8u + n] += dflash_xnor_e2m1_signed((qw.y >> (n * 4u)) & 0xFu) * sc;
                    }
                }
                threadgroup float sh_sg[8];
                threadgroup float sh_mu[16];
                for (uint j = 0u; j < 16u; ++j) {
                    const float sg_s = simd_sum(s[j]);
                    if (lane == 0u) { sh_sg[sg] = sg_s; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg == 0u) {
                        const float tot = simd_sum((lane < 8u) ? sh_sg[lane] : 0.0f);
                        if (lane == 0u) { sh_mu[j] = tot / float(V_); }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                float sq[16];
                #pragma unroll
                for (uint j = 0u; j < 16u; ++j) { sq[j] = 0.0f; }
                for (uint v = lid; v < uint(V_); v += 256u) {
                    const uint2 qw = *(const device uint2*)(w + size_t(v) * size_t(K_ / 8) + size_t(g) * 2u);
                    const float sc = float(dflash_xnor_e4m3(scales[size_t(v) * size_t(K_ / 16) + size_t(g)]));
                    #pragma unroll
                    for (uint n = 0u; n < 8u; ++n) {
                        const float d0 = dflash_xnor_e2m1_signed((qw.x >> (n * 4u)) & 0xFu) * sc - sh_mu[n];
                        const float d1 = dflash_xnor_e2m1_signed((qw.y >> (n * 4u)) & 0xFu) * sc - sh_mu[8u + n];
                        sq[n] = fma(d0, d0, sq[n]);
                        sq[8u + n] = fma(d1, d1, sq[8u + n]);
                    }
                }
                for (uint j = 0u; j < 16u; ++j) {
                    const float sg_sq = simd_sum(sq[j]);
                    if (lane == 0u) { sh_sg[sg] = sg_sq; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg == 0u) {
                        const float tot_sq = simd_sum((lane < 8u) ? sh_sg[lane] : 0.0f);
                        if (lane == 0u) {
                            const float mu_v = metal::isfinite(sh_mu[j]) ? sh_mu[j] : 0.0f;
                            const float rms = sqrt(max(0.0f, tot_sq / float(V_)));
                            const bool ok = metal::isfinite(rms) && rms > 1e-6f;
                            const uint k_idx = g * 16u + j;
                            col_mu[k_idx] = mu_v;
                            col_rms[k_idx] = ok ? rms : 0.0f;
                            col_inv_rms[k_idx] = ok ? (1.0f / rms) : 0.0f;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                """,
            header: header)

        private static let buildKernel = MLXFast.metalKernel(
            name: "dflash_build_xnor_index",
            inputNames: ["w", "scales", "col_mu", "col_inv_rms"],
            outputNames: ["w1_bits", "g256_sc"],
            source: """
                const uint slot = thread_position_in_grid.x;
                const uint g = thread_position_in_grid.y;
                const uint tg = thread_position_in_grid.z;
                if (slot >= uint(STRIDE_) || g >= uint(K256_) || tg >= uint(NUM_TGS_)) { return; }
                const uint v = slot * uint(NUM_TGS_) + tg;
                const size_t sc_off = (size_t(tg) * size_t(K256_) + size_t(g)) * size_t(STRIDE_) + size_t(slot);
                const size_t w4_base = (size_t(tg) * size_t(K256_) + size_t(g)) * 2u * size_t(STRIDE_) + size_t(slot);
                device uint4* w1_bits4 = (device uint4*)w1_bits;
                if (v >= uint(V_)) {
                    w1_bits4[w4_base] = uint4(0u);
                    w1_bits4[w4_base + size_t(STRIDE_)] = uint4(0u);
                    g256_sc[sc_off] = half(0.0f);
                    return;
                }
                const device uint* w_row = w + size_t(v) * size_t(K_ / 8) + size_t(g) * 32u;
                const device uint8_t* sc_row = scales + size_t(v) * size_t(K_ / 16) + size_t(g) * 16u;
                float l1_sum = 0.0f;
                uint out_w[8];
                for (uint j = 0u; j < 8u; ++j) {
                    uint bits = 0u;
                    for (uint b2 = 0u; b2 < 2u; ++b2) {
                        const float sc = float(dflash_xnor_e4m3(sc_row[j * 2u + b2]));
                        for (uint w2 = 0u; w2 < 2u; ++w2) {
                            const uint q = w_row[j * 4u + b2 * 2u + w2];
                            const uint bit_base = (b2 * 2u + w2) * 8u;
                            if constexpr (CENTERED_) {
                                const uint k_base = g * 256u + j * 32u + bit_base;
                                #pragma unroll
                                for (uint n = 0u; n < 8u; ++n) {
                                    const uint nib = (q >> (n * 4u)) & 0xFu;
                                    const float wv =
                                        (dflash_xnor_e2m1_signed(nib) * sc - col_mu[k_base + n])
                                        * col_inv_rms[k_base + n];
                                    if (wv >= 0.0f) {
                                        bits |= (1u << (bit_base + n));
                                    }
                                    l1_sum += fabs(wv);
                                }
                            } else {
                                float block_abs = 0.0f;
                                #pragma unroll
                                for (uint n = 0u; n < 8u; ++n) {
                                    const uint nib = (q >> (n * 4u)) & 0xFu;
                                    if ((nib & 8u) == 0u) {
                                        bits |= (1u << (bit_base + n));
                                    }
                                    block_abs += dflash_xnor_e2m1_abs(nib);
                                }
                                l1_sum += block_abs * sc;
                            }
                        }
                    }
                    out_w[j] = bits;
                }
                w1_bits4[w4_base] = uint4(out_w[0], out_w[1], out_w[2], out_w[3]);
                w1_bits4[w4_base + size_t(STRIDE_)] = uint4(out_w[4], out_w[5], out_w[6], out_w[7]);
                g256_sc[sc_off] = half(l1_sum * (1.0f / 256.0f));
                """,
            header: header)

        private static let quantKernel = MLXFast.metalKernel(
            name: "dflash_xnor_quant_x2",
            inputNames: ["x", "col_rms"],
            outputNames: ["x_bits", "x_scale"],
            source: """
                const uint lane = thread_index_in_simdgroup;
                const uint g = threadgroup_position_in_grid.y;
                const uint r = threadgroup_position_in_grid.z;
                const device bfloat16_t* xr = x + size_t(r) * size_t(K_) + size_t(g) * 256u;
                float vals[8];
                float sq = 0.0f;
                #pragma unroll
                for (uint j = 0u; j < 8u; ++j) {
                    const uint lk = j * 32u + lane;
                    const float raw_v = float(xr[lk]);
                    const float v = CENTERED_ ? (raw_v * col_rms[g * 256u + lk]) : raw_v;
                    vals[j] = v;
                    sq += v * v;
                }
                const float rms = sqrt(simd_sum(sq) * (1.0f / 256.0f) + 1e-12f);
                const float thresh = 0.9816f * rms;
                uint bhi[8], blo[8];
                #pragma unroll
                for (uint j = 0u; j < 8u; ++j) {
                    const float v = vals[j];
                    const bool hi = (v >= 0.0f);
                    const bool big = (fabs(v) >= thresh);
                    const bool lo = hi ? big : !big;
                    bhi[j] = uint(simd_vote::vote_t(simd_ballot(hi)));
                    blo[j] = uint(simd_vote::vote_t(simd_ballot(lo)));
                }
                if (lane == 0u) {
                    device uint4* x_bits4 = (device uint4*)x_bits;
                    const uint base = (g * uint(R_) + r) * 4u;
                    x_bits4[base + 0u] = uint4(bhi[0], bhi[1], bhi[2], bhi[3]);
                    x_bits4[base + 1u] = uint4(bhi[4], bhi[5], bhi[6], bhi[7]);
                    x_bits4[base + 2u] = uint4(blo[0], blo[1], blo[2], blo[3]);
                    x_bits4[base + 3u] = uint4(blo[4], blo[5], blo[6], blo[7]);
                    x_scale[g * uint(R_) + r] = half(rms);
                }
                """,
            header: header)

        private static let stage1Kernel = MLXFast.metalKernel(
            name: "dflash_xnor_stage1",
            inputNames: ["w1_bits", "g256_sc", "x_bits", "x_scale"],
            outputNames: ["cand_ids"],
            source: """
                const uint lid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint tg = threadgroup_position_in_grid.x;
                const device uint4* w1_bits4 = (const device uint4*)w1_bits;
                const device uint4* x_bits4 = (const device uint4*)x_bits;

                // 1-iteration-per-thread (tgSize = ceil32(STRIDE_), e.g. 416t = 13 SGs for V=201088):
                // eliminates the 46.5%-idle second iteration and halves live accumulators (maxT=1024),
                // keeping full sh_x TGEM + compact sh_v[NUM_SG_][2] for 2 TGs/core (832 threads/core).
                threadgroup uint4 sh_x[K256_][R_][4];
                threadgroup half sh_xs[K256_][R_];
                constexpr uint total_x4 = uint(K256_) * uint(R_) * 4u;
                for (uint i = lid; i < total_x4; i += uint(TG_SIZE_)) {
                    ((threadgroup uint4*)sh_x)[i] = x_bits4[i];
                }
                constexpr uint total_xs = uint(K256_) * uint(R_);
                for (uint i = lid; i < total_xs; i += uint(TG_SIZE_)) {
                    ((threadgroup half*)sh_xs)[i] = x_scale[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint slot = lid;
                const int v = (slot < uint(STRIDE_)) ? int(slot * uint(NUM_TGS_) + tg) : -1;
                const bool valid = (v >= 0 && v < int(V_));
                float acc[R_];
                for (uint r = 0u; r < uint(R_); ++r) { acc[r] = valid ? 0.0f : -INFINITY; }
                if (valid) {
                    const device uint4* tg_w4 = w1_bits4 + size_t(tg) * size_t(K256_ * 2 * STRIDE_);
                    const device half* tg_sc = g256_sc + size_t(tg) * size_t(K256_ * STRIDE_);
                    for (uint g = 0u; g < uint(K256_); ++g) {
                        const float sc_w = float(tg_sc[g * uint(STRIDE_) + slot]);
                        const uint4 wa = tg_w4[(g * 2u + 0u) * uint(STRIDE_) + slot];
                        const uint4 wb = tg_w4[(g * 2u + 1u) * uint(STRIDE_) + slot];
                        for (uint r = 0u; r < uint(R_); ++r) {
                            const uint4 ph = popcount(wa ^ sh_x[g][r][0]) + popcount(wb ^ sh_x[g][r][1]);
                            const uint4 pl = popcount(wa ^ sh_x[g][r][2]) + popcount(wb ^ sh_x[g][r][3]);
                            const uint p_hi = (ph.x + ph.y) + (ph.z + ph.w);
                            const uint p_lo = (pl.x + pl.y) + (pl.z + pl.w);
                            const float dot_g = 386.56f - 1.963f * float(p_hi) - 1.057f * float(p_lo);
                            acc[r] = fma(dot_g, sc_w * float(sh_xs[g][r]), acc[r]);
                        }
                    }
                }

                threadgroup float sh_v[2][NUM_SG_][2];
                threadgroup int sh_i[2][NUM_SG_][2];
                for (uint r = 0u; r < uint(R_); ++r) {
                    const uint b = r & 1u;
                    float v0 = acc[r], v1 = -INFINITY;
                    int i0 = valid ? v : -1, i1 = -1;
                    for (ushort off = 16u; off > 0u; off >>= 1u) {
                        const float ov0 = simd_shuffle_down(v0, off); const int oi0 = simd_shuffle_down(i0, off);
                        const float ov1 = simd_shuffle_down(v1, off); const int oi1 = simd_shuffle_down(i1, off);
                        if (lane < off) {
                            dflash_xnor_insert_top2(v0, i0, v1, i1, ov0, oi0);
                            dflash_xnor_insert_top2(v0, i0, v1, i1, ov1, oi1);
                        }
                    }
                    if (lane == 0u) {
                        sh_v[b][sg][0] = v0; sh_i[b][sg][0] = i0;
                        sh_v[b][sg][1] = v1; sh_i[b][sg][1] = i1;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg == 0u) {
                        if (lane < uint(NUM_SG_)) {
                            v0 = sh_v[b][lane][0]; i0 = sh_i[b][lane][0];
                            v1 = sh_v[b][lane][1]; i1 = sh_i[b][lane][1];
                        } else {
                            v0 = v1 = -INFINITY; i0 = i1 = -1;
                        }
                        for (ushort off = 8u; off > 0u; off >>= 1u) {
                            const float ov0 = simd_shuffle_down(v0, off); const int oi0 = simd_shuffle_down(i0, off);
                            const float ov1 = simd_shuffle_down(v1, off); const int oi1 = simd_shuffle_down(i1, off);
                            if (lane < off) {
                                dflash_xnor_insert_top2(v0, i0, v1, i1, ov0, oi0);
                                dflash_xnor_insert_top2(v0, i0, v1, i1, ov1, oi1);
                            }
                        }
                        if (lane == 0u) {
                            const uint base = (r * uint(NUM_TGS_) + tg) * 2u;
                            cand_ids[base + 0u] = i0;
                            cand_ids[base + 1u] = i1;
                        }
                    }
                }
                """,
            header: header)

        private static let stage2Kernel = MLXFast.metalKernel(
            name: "dflash_xnor_stage2_dyntree",
            inputNames: ["cand_ids", "w", "scales", "x", "tparams"],
            outputNames: ["p_val", "p_idx", "p_sum"],
            source: """
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint chunk = threadgroup_position_in_grid.y;
                const uint row = threadgroup_position_in_grid.z;
                const int ivocab = int(V_);
                const float eff_scale = tparams[0];
                const float cap = tparams[1];
                const float row_T = max(0.05f, tparams[2] + float(row) * tparams[14]);
                const float inv_t_log2e = 1.44269504f / row_T;
                const device bfloat16_t* x_row = x + size_t(row) * size_t(K_);

                constexpr bool USE_SHX = (K_ <= 8192);
                threadgroup bfloat16_t sh_x[USE_SHX ? K_ : 1];
                if constexpr (USE_SHX) {
                    const uint lid = sg * 32u + lane;
                    const device ushort4* x4 = (const device ushort4*)x_row;
                    threadgroup ushort4* sx4 = (threadgroup ushort4*)sh_x;
                    for (uint i = lid; i < uint(K_ / 4); i += 256u) {
                        sx4[i] = x4[i];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                threadgroup float sh_v[16];
                threadgroup int sh_i[16];

                const uint c0 = sg * 2u;
                const uint c1 = c0 + 1u;
                const int v0 = cand_ids[row * 1024u + chunk * 16u + c0];
                const int v1 = cand_ids[row * 1024u + chunk * 16u + c1];
                const bool ok0 = (v0 >= 0 && v0 < ivocab);
                const bool ok1 = (v1 >= 0 && v1 < ivocab);
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                if (ok0 || ok1) {
                    const device uint2* w2_0 = ok0 ? (const device uint2*)(w + size_t(v0) * size_t(K_ / 8)) : nullptr;
                    const device uint8_t* sc_0 = ok0 ? (scales + size_t(v0) * size_t(K_ / 16)) : nullptr;
                    const device uint2* w2_1 = ok1 ? (const device uint2*)(w + size_t(v1) * size_t(K_ / 8)) : nullptr;
                    const device uint8_t* sc_1 = ok1 ? (scales + size_t(v1) * size_t(K_ / 16)) : nullptr;
                    #define DFLASH_S2_ACC(BLK, QW0, SC0, QW1, SC1) { \\
                        float4 x0, x1, x2, x3; \\
                        if constexpr (USE_SHX) { \\
                            const threadgroup ushort4* sx = (const threadgroup ushort4*)(sh_x + (BLK) * 16u); \\
                            x0 = float4(as_type<vec<bfloat, 4>>(sx[0])); \\
                            x1 = float4(as_type<vec<bfloat, 4>>(sx[1])); \\
                            x2 = float4(as_type<vec<bfloat, 4>>(sx[2])); \\
                            x3 = float4(as_type<vec<bfloat, 4>>(sx[3])); \\
                        } else { \\
                            const device bfloat16_t* xb = x_row + (BLK) * 16u; \\
                            x0 = float4(as_type<vec<bfloat, 4>>(ushort4(*(const device packed_ushort4*)(xb + 0u)))); \\
                            x1 = float4(as_type<vec<bfloat, 4>>(ushort4(*(const device packed_ushort4*)(xb + 4u)))); \\
                            x2 = float4(as_type<vec<bfloat, 4>>(ushort4(*(const device packed_ushort4*)(xb + 8u)))); \\
                            x3 = float4(as_type<vec<bfloat, 4>>(ushort4(*(const device packed_ushort4*)(xb + 12u)))); \\
                        } \\
                        if (ok0) { \\
                            const float fs = float(dflash_xnor_e4m3(SC0)) * 16384.0f; \\
                            const uint q0 = (QW0).x; \\
                            const float2 a0 = float2(as_type<half2>(((q0 & 0x00070007u) << 9) | ((q0 & 0x00080008u) << 12))) * fs; \\
                            const float2 a1 = float2(as_type<half2>(((q0 & 0x00700070u) << 5) | ((q0 & 0x00800080u) << 8))) * fs; \\
                            const float2 a2 = float2(as_type<half2>(((q0 & 0x07000700u) << 1) | ((q0 & 0x08000800u) << 4))) * fs; \\
                            const float2 a3 = float2(as_type<half2>(((q0 & 0x70007000u) >> 3) | (q0 & 0x80008000u))) * fs; \\
                            acc0 = fma(a0.x, x0.x, acc0); acc0 = fma(a1.x, x0.y, acc0); \\
                            acc0 = fma(a2.x, x0.z, acc0); acc0 = fma(a3.x, x0.w, acc0); \\
                            acc0 = fma(a0.y, x1.x, acc0); acc0 = fma(a1.y, x1.y, acc0); \\
                            acc0 = fma(a2.y, x1.z, acc0); acc0 = fma(a3.y, x1.w, acc0); \\
                            const uint q1 = (QW0).y; \\
                            const float2 b0 = float2(as_type<half2>(((q1 & 0x00070007u) << 9) | ((q1 & 0x00080008u) << 12))) * fs; \\
                            const float2 b1 = float2(as_type<half2>(((q1 & 0x00700070u) << 5) | ((q1 & 0x00800080u) << 8))) * fs; \\
                            const float2 b2 = float2(as_type<half2>(((q1 & 0x07000700u) << 1) | ((q1 & 0x08000800u) << 4))) * fs; \\
                            const float2 b3 = float2(as_type<half2>(((q1 & 0x70007000u) >> 3) | (q1 & 0x80008000u))) * fs; \\
                            acc0 = fma(b0.x, x2.x, acc0); acc0 = fma(b1.x, x2.y, acc0); \\
                            acc0 = fma(b2.x, x2.z, acc0); acc0 = fma(b3.x, x2.w, acc0); \\
                            acc0 = fma(b0.y, x3.x, acc0); acc0 = fma(b1.y, x3.y, acc0); \\
                            acc0 = fma(b2.y, x3.z, acc0); acc0 = fma(b3.y, x3.w, acc0); \\
                        } \\
                        if (ok1) { \\
                            const float fs = float(dflash_xnor_e4m3(SC1)) * 16384.0f; \\
                            const uint q0 = (QW1).x; \\
                            const float2 a0 = float2(as_type<half2>(((q0 & 0x00070007u) << 9) | ((q0 & 0x00080008u) << 12))) * fs; \\
                            const float2 a1 = float2(as_type<half2>(((q0 & 0x00700070u) << 5) | ((q0 & 0x00800080u) << 8))) * fs; \\
                            const float2 a2 = float2(as_type<half2>(((q0 & 0x07000700u) << 1) | ((q0 & 0x08000800u) << 4))) * fs; \\
                            const float2 a3 = float2(as_type<half2>(((q0 & 0x70007000u) >> 3) | (q0 & 0x80008000u))) * fs; \\
                            acc1 = fma(a0.x, x0.x, acc1); acc1 = fma(a1.x, x0.y, acc1); \\
                            acc1 = fma(a2.x, x0.z, acc1); acc1 = fma(a3.x, x0.w, acc1); \\
                            acc1 = fma(a0.y, x1.x, acc1); acc1 = fma(a1.y, x1.y, acc1); \\
                            acc1 = fma(a2.y, x1.z, acc1); acc1 = fma(a3.y, x1.w, acc1); \\
                            const uint q1 = (QW1).y; \\
                            const float2 b0 = float2(as_type<half2>(((q1 & 0x00070007u) << 9) | ((q1 & 0x00080008u) << 12))) * fs; \\
                            const float2 b1 = float2(as_type<half2>(((q1 & 0x00700070u) << 5) | ((q1 & 0x00800080u) << 8))) * fs; \\
                            const float2 b2 = float2(as_type<half2>(((q1 & 0x07000700u) << 1) | ((q1 & 0x08000800u) << 4))) * fs; \\
                            const float2 b3 = float2(as_type<half2>(((q1 & 0x70007000u) >> 3) | (q1 & 0x80008000u))) * fs; \\
                            acc1 = fma(b0.x, x2.x, acc1); acc1 = fma(b1.x, x2.y, acc1); \\
                            acc1 = fma(b2.x, x2.z, acc1); acc1 = fma(b3.x, x2.w, acc1); \\
                            acc1 = fma(b0.y, x3.x, acc1); acc1 = fma(b1.y, x3.y, acc1); \\
                            acc1 = fma(b2.y, x3.z, acc1); acc1 = fma(b3.y, x3.w, acc1); \\
                        } \\
                    }
                    constexpr uint NBLK = uint(K_ / 16);
                    if (lane < NBLK) {
                        uint2 n0 = ok0 ? w2_0[lane] : uint2(0u);
                        uint8_t ns0 = ok0 ? sc_0[lane] : uint8_t(0u);
                        uint2 n1 = ok1 ? w2_1[lane] : uint2(0u);
                        uint8_t ns1 = ok1 ? sc_1[lane] : uint8_t(0u);
                        uint b = lane;
                        for (; b + 32u < NBLK; b += 32u) {
                            const uint2 cur0 = n0, cur1 = n1;
                            const uint8_t cs0 = ns0, cs1 = ns1;
                            if (ok0) { n0 = w2_0[b + 32u]; ns0 = sc_0[b + 32u]; }
                            if (ok1) { n1 = w2_1[b + 32u]; ns1 = sc_1[b + 32u]; }
                            DFLASH_S2_ACC(b, cur0, cs0, cur1, cs1)
                        }
                        DFLASH_S2_ACC(b, n0, ns0, n1, ns1)
                    }
                    #undef DFLASH_S2_ACC
                    if (ok0) { acc0 = simd_sum(acc0); }
                    if (ok1) { acc1 = simd_sum(acc1); }
                }
                if (lane == 0u) {
                    sh_v[c0] = ok0 ? dflash_xnor_scale_logit(acc0, eff_scale, cap) : -INFINITY;
                    sh_i[c0] = ok0 ? v0 : ivocab;
                    sh_v[c1] = ok1 ? dflash_xnor_scale_logit(acc1, eff_scale, cap) : -INFINITY;
                    sh_i[c1] = ok1 ? v1 : ivocab;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0u) {
                    const float lv = (lane < 16u) ? sh_v[lane] : -INFINITY;
                    const int li = (lane < 16u) ? sh_i[lane] : ivocab;
                    const float m = simd_max(lv);
                    const float diff = (m - lv) * inv_t_log2e;
                    const float e = (lv > -INFINITY && diff < 24.0f) ? metal::fast::exp2(-diff) : 0.0f;
                    const float s = simd_sum(e);

                    float v0 = lv, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY, v4 = -INFINITY;
                    int i0 = li, i1 = ivocab, i2 = ivocab, i3 = ivocab, i4 = ivocab;
                    for (ushort off = 8u; off > 0u; off >>= 1u) {
                        const float ov0 = simd_shuffle_down(v0, off); const int oi0 = simd_shuffle_down(i0, off);
                        const float ov1 = simd_shuffle_down(v1, off); const int oi1 = simd_shuffle_down(i1, off);
                        const float ov2 = simd_shuffle_down(v2, off); const int oi2 = simd_shuffle_down(i2, off);
                        const float ov3 = simd_shuffle_down(v3, off); const int oi3 = simd_shuffle_down(i3, off);
                        const float ov4 = simd_shuffle_down(v4, off); const int oi4 = simd_shuffle_down(i4, off);
                        if (lane < off) {
                            dflash_xnor_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov0, oi0, ivocab);
                            dflash_xnor_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov1, oi1, ivocab);
                            dflash_xnor_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov2, oi2, ivocab);
                            dflash_xnor_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov3, oi3, ivocab);
                            dflash_xnor_insert_top5(v0, i0, v1, i1, v2, i2, v3, i3, v4, i4, ov4, oi4, ivocab);
                        }
                    }
                    if (lane == 0u) {
                        const uint base = (row * 64u + chunk) * 5u;
                        p_val[base + 0u] = v0; p_idx[base + 0u] = i0;
                        p_val[base + 1u] = v1; p_idx[base + 1u] = i1;
                        p_val[base + 2u] = v2; p_idx[base + 2u] = i2;
                        p_val[base + 3u] = v3; p_idx[base + 3u] = i3;
                        p_val[base + 4u] = v4; p_idx[base + 4u] = i4;
                        p_sum[row * 64u + chunk] = s;
                    }
                }
                """,
            header: header)

        let vocab: Int
        let hidden: Int
        let k256: Int
        let numTGs: Int
        let stride: Int
        private let useCentered: Bool
        private let headWeight: MLXArray
        private let headScales: MLXArray
        private let w1Bits: MLXArray
        private let g256Sc: MLXArray
        private let colRms: MLXArray

        private let lock = NSLock()
        private var launches: [Int: (
            quant: MLXFast.MLXFastKernel.Launch,
            stage1: MLXFast.MLXFastKernel.Launch,
            stage2: MLXFast.MLXFastKernel.Launch
        )] = [:]

        var residentArrays: [MLXArray] { [w1Bits, g256Sc, colRms] }

        init?(head: ModelOptWeightOnlyNVFP4Linear) {
            guard Self.enabled else { return nil }
            let v = head.outputDimensions
            let k = head.inputDimensions
            let tgs = 512
            let st = (((v + tgs - 1) / tgs) + 15) & ~15
            guard v >= 4096, k >= 256, k % 256 == 0, st <= 512,
                head.weight.dtype == .uint32, head.weight.shape == [v, k / 8],
                head.weightScale.dtype == .uint8, head.weightScale.shape == [v, k / 16]
            else { return nil }
            self.vocab = v
            self.hidden = k
            self.k256 = k / 256
            self.numTGs = tgs
            self.stride = st
            self.useCentered = Self.centeredEnabled
            self.headWeight = head.weight
            self.headScales = head.weightScale
            let colMu: MLXArray
            let colInvRms: MLXArray
            if self.useCentered {
                let stats = Self.colStatsKernel(
                    [head.weight, head.weightScale],
                    template: [("V_", v), ("K_", k)],
                    grid: (256, k / 16, 1),
                    threadGroup: (256, 1, 1),
                    outputShapes: [[k], [k], [k]],
                    outputDTypes: [.float32, .float32, .float32])
                colMu = stats[0]
                self.colRms = stats[1]
                colInvRms = stats[2]
            } else {
                colMu = MLXArray.zeros([1], dtype: .float32)
                self.colRms = MLXArray.ones([1], dtype: .float32)
                colInvRms = MLXArray.ones([1], dtype: .float32)
            }
            let gridX = ((st + 63) / 64) * 64
            let built = Self.buildKernel(
                [head.weight, head.weightScale, colMu, colInvRms],
                template: [
                    ("V_", v), ("K_", k), ("K256_", k / 256), ("NUM_TGS_", tgs), ("STRIDE_", st),
                    ("CENTERED_", self.useCentered),
                ],
                grid: (gridX, k / 256, tgs),
                threadGroup: (64, 1, 4),
                outputShapes: [[tgs, k / 256, 2, st, 4], [tgs, k / 256, st]],
                outputDTypes: [.uint32, .float16])
            self.w1Bits = built[0]
            self.g256Sc = built[1]
            eval(self.w1Bits, self.g256Sc, self.colRms)
        }

        func dynamicTreePartials(
            _ x: MLXArray, params: MLXArray
        ) -> (val: MLXArray, idx: MLXArray, sum: MLXArray)? {
            guard x.ndim == 2, x.dtype == .bfloat16, x.dim(1) == hidden else { return nil }
            let rows = x.dim(0)
            guard (1 ... 7).contains(rows) else { return nil }
            let pair: (
                quant: MLXFast.MLXFastKernel.Launch,
                stage1: MLXFast.MLXFastKernel.Launch,
                stage2: MLXFast.MLXFastKernel.Launch
            )
            if let hit = lock.withLock({ launches[rows] }) {
                pair = hit
            } else {
                let q = Self.quantKernel.prepare(
                    template: [
                        ("R_", rows), ("K_", hidden), ("K256_", k256),
                        ("CENTERED_", useCentered),
                    ],
                    grid: (32, k256, rows),
                    threadGroup: (32, 1, 1),
                    outputShapes: [[k256, rows, 16], [k256, rows]],
                    outputDTypes: [.uint32, .float16])
                let tgSize = ((stride + 31) / 32) * 32
                let numSG = tgSize / 32
                let s1 = Self.stage1Kernel.prepare(
                    template: [
                        ("R_", rows), ("V_", vocab), ("K256_", k256),
                        ("NUM_TGS_", numTGs), ("STRIDE_", stride),
                        ("TG_SIZE_", tgSize), ("NUM_SG_", numSG),
                    ],
                    grid: (tgSize * numTGs, 1, 1),
                    threadGroup: (tgSize, 1, 1),
                    outputShapes: [[rows, numTGs * 2]],
                    outputDTypes: [.int32])
                let s2 = Self.stage2Kernel.prepare(
                    template: [("V_", vocab), ("K_", hidden)],
                    grid: (32, 8 * 64, rows),
                    threadGroup: (32, 8, 1),
                    outputShapes: [[rows, 64, 5], [rows, 64, 5], [rows, 64]],
                    outputDTypes: [.float32, .int32, .float32])
                let made = (q, s1, s2)
                lock.withLock { launches[rows] = made }
                pair = made
            }
            let qOut = pair.quant([x, colRms])
            let candIDs = pair.stage1([w1Bits, g256Sc, qOut[0], qOut[1]])[0]
            let s2Out = pair.stage2([candIDs, headWeight, headScales, x, params])
            return (s2Out[0], s2Out[1], s2Out[2])
        }
    }
}
