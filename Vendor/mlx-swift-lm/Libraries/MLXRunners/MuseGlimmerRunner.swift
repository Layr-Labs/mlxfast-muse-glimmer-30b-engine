// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Muse Glimmer 30B (`muse_glimmer`, `muse_glimmer_text`).
//
// A DENSE attention tower: 52 layers, three sliding-window layers to one full
// layer, no recurrent state and no shared KV. Every layer owns its K/V, so the
// layer table is one entry per decoder layer with no `modelLayerIndex` mapping.
//
// The checkpoint is NVIDIA's ModelOpt export and is served AS PUBLISHED: the
// load goes through `MuseGlimmerInstallation.stage(directory:)`, which hashes
// the index and every shard against the pin and installs each projection in
// the format its verified header declares (BF16, calibrated FP8, weight-only
// NVFP4). Nothing here re-quantizes, re-formats or re-shapes a weight, and a
// directory that fails the pin is refused instead of loaded.
//
// PAGED ONLY. The model declares `requiresPagedKV` and `requiresNativePagedKV`,
// so `EngineV2` serves it from segmented paged storage with the K/V dtypes the
// tower actually writes. A contiguous build is refused by name here rather
// than refused later by the engine.
//
// NUMERICS. The tower serves with exact verify windows on, unconditionally,
// so whether a drafter loaded cannot change a served byte: decode and verify
// rows get the bytes they get alone, prompt chunks keep MLX's arithmetic on the
// model's own 1,024-token grid. The profile is part of the served identity and
// a tower reporting another one is refused.
//
// SPECULATION is the published block drafter (`dflash`), a request-stateful
// assistant checkpoint: one block forward per round proposes up to seven
// tokens, one rectangular exact verify forward scores the round, and the
// engine keeps a proposal only when it equals the token the target chose — so
// with the drafter on the greedy output is the drafter-off output. Without
// `--drafter` the runner serves plain greedy decode and advertises `serial`
// only.
//
// ROUNDS ARE NOT NAMED. The drafter and the engine publish per-window
// counters, not per-round committed widths, and this layer may not add a
// record to the library, so the runner states no audit
// (`RunnerRoundAudit.swift`). A window is audited only from records an engine
// hands over; none are inferred from a token count.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// The HF tokenizer loader, named once. A protocol extension is a generic
/// context and the loader macro declares a nested type, so the loader lives
/// at file scope instead.
private enum MuseGlimmerTokenizerLoader {
    static let shared: any TokenizerLoader = #huggingFaceTokenizerLoader()
}

public final class MuseGlimmerRunner: Runner, @unchecked Sendable {

    /// The checkpoint's `model_type`. The text configuration falls back to the
    /// `_text` spelling when a `config.json` omits it, and an export may carry
    /// the tower's own name, so both are claimed.
    public static let modelTypes = ["muse_glimmer", "muse_glimmer_text"]

    /// One block forward proposes this many tokens at most; the engine's
    /// tested verify window is 1 + 7.
    public static let maximumDrafterTokens = 7

    /// Requests one exact round may hold. The scored width is one; a wider
    /// step stays correct and is what the served tower runs by default.
    public static let speculativeRows = 4

    /// The assistant artifact bound: the quantized drafter is 1,968,222,936
    /// bytes, and anything larger than this is not the artifact this runner
    /// advertises.
    public static let maximumDrafterBytes = 2 << 30

    /// The numerics profile the served bytes are pinned to: exact verify
    /// windows on the 1,024-token prompt grid.
    public static let servedNumericsProfile = "muse-glimmer-exact-windows-v4"

    public static let manifest = RunnerManifest(
        runnerID: "layr/muse-glimmer-30b",
        modelTypes: modelTypes,
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: true,
            supportsPagedKV: true,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false,
            // The tower cuts every prompt into 1,024-token cells, and a paged
            // pool that cannot hold one cell refuses the model at engine
            // start. Read off the model's own constant rather than repeated
            // here as a number that could drift from the library.
            prefillChunkAlignment: MuseGlimmerTextModel.defaultPrefillChunkAlignment),
        // PAGED ONLY, and segmented paged at that: the model requires native
        // paged KV, so a contiguous build is not a slower arm of this family,
        // it is a different family's serving shape.
        kvBackends: [.paged],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            DecoderDeclaration(
                mode: DecoderID.dflash.rawValue, drafter: .assistantCheckpoint,
                state: .requestStateful,
                depth: 1 ... CBv2MTPConfig.testedMaxDraftTokens),
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(
                batch: .upTo(speculativeRows), timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: false,
        recurrentLayers: false,
        requiresKeepMask: false)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID]
    public let headProvenance: HeadProvenance?
    public let loadedModelType: String

    private let model: MuseGlimmerTextModel
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    /// A drafter this runner built from its pinned artifact, and the directory
    /// the pin was verified against. The pair travels together so the hello's
    /// provenance hash is always taken over the artifact the bound module came
    /// from.
    struct VerifiedDrafter {
        var drafter: any CBv2MTPDrafter
        var directory: URL
    }

    private init(
        model: MuseGlimmerTextModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.model = model
        self.servingModel = model
        self.layerKinds = model.cbv2LayerKinds
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.loadedModelType = loadedModelType
        self.drafter = drafter
        self.headProvenance = headProvenance
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.loadedDecoders = drafter == nil ? [.serial] : [.serial, .dflash]
    }

    // MARK: - Load

    /// Stage the pinned artifact, then adopt it.
    ///
    /// There is no factory for this tower, so the default `Runner.load` does
    /// not apply: the installer IS the load. It hashes the index and every
    /// shard, reads each projection's format from the verified headers and
    /// installs the tower; a directory that fails the pin throws before a
    /// tensor is placed.
    ///
    /// Exact windows are switched on HERE, before the drafter binds: the
    /// drafter asks the target whether its verify windows are exact, and binds
    /// serial verification to a target that says no.
    public static func load(
        _ directory: URL, options: RunnerLoadOptions
    ) async throws -> MuseGlimmerRunner {
        let installed = try MuseGlimmerInstallation.stage(directory: directory)
        try enableExactWindows(installed.model)
        let tokenizer = try await MuseGlimmerTokenizerLoader.shared.load(from: directory)

        // The drafter enters through the same door `adopt` refuses: it is built
        // HERE, from its own artifact, and it rides in with the directory the
        // pin was checked against, so the hello's provenance hash describes the
        // drafter that is actually in use.
        var drafter: VerifiedDrafter?
        if let built = try await loadDrafter(
            options: options, directory: directory, target: installed.model)
        {
            guard let drafterDirectory = options.drafterDirectory else {
                throw RunnerError.drafterUnavailable(
                    "a loaded drafter needs the artifact directory it was read from")
            }
            drafter = VerifiedDrafter(drafter: built, directory: drafterDirectory)
        }

        return try assemble(
            model: installed.model,
            tokenizer: tokenizer,
            directory: directory,
            options: options,
            drafter: drafter)
    }

    /// Adopt a tower the caller already holds.
    ///
    /// Reads no tensor: everything derived — the layer table, the decoders,
    /// the head provenance — comes from the module, `config.json` and the
    /// drafter directory's own files. The exact-window switch is the one
    /// change made to the module, and it is the same switch `load` makes, so
    /// an adopted tower and a loaded one serve the same bytes.
    ///
    /// A DRAFTER is refused here. One handed in as a module has no verified
    /// artifact behind it: the pin, the assistant size cap, the artifact type
    /// and the target binding are all checked while the artifact is read, and a
    /// module already in memory was read somewhere else. Binding it on trust —
    /// and hashing an unrelated directory for the hello — is what this refusal
    /// replaces. Name `--drafter <directory>` and the load path checks it.
    public static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> MuseGlimmerRunner {
        if let refusal = preloadedDrafterRefusal(options) { throw refusal }
        return try assemble(
            model: model, tokenizer: tokenizer, directory: directory, options: options,
            drafter: nil)
    }

    /// The refusal a drafter handed in as a module earns, on its own so a test
    /// can prove it without staging a checkpoint.
    static func preloadedDrafterRefusal(_ options: RunnerLoadOptions) -> RunnerError? {
        guard options.preloadedDrafter != nil else { return nil }
        return .drafterUnavailable(
            "a preloaded drafter carries no verified artifact: name its directory, so this "
                + "runner checks the pin, the assistant size cap and the target binding")
    }

    /// The one construction path: `load` and `adopt` both funnel through it, so
    /// the derived state cannot drift between them.
    private static func assemble(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        directory: URL,
        options: RunnerLoadOptions,
        drafter: VerifiedDrafter?
    ) throws -> MuseGlimmerRunner {
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let eosTokenIDs = RunnerCheckpoint.eosTokenIDs(at: directory, tokenizer: tokenizer)

        guard let model = model as? MuseGlimmerTextModel else {
            throw RunnerError.unexpectedModel(
                "\(type(of: model)) — this runner serves the native NVFP4 text tower")
        }
        try enableExactWindows(model)

        // The drafter is a SEPARATE published export, so its provenance is the
        // shards in the directory its pin was verified against, in name order.
        //
        // FAIL CLOSED: the provenance rides the hello, so a drafter that cannot
        // be hashed must stop the adoption rather than be served unattributed.
        var provenance: HeadProvenance?
        if let drafter {
            provenance = try RunnerCheckpoint.provenance(ofHeadAt: drafter.directory)
        }

        return MuseGlimmerRunner(
            model: model,
            tokenizer: tokenizer,
            eosTokenIDs: eosTokenIDs,
            loadedModelType: modelType,
            drafter: drafter?.drafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    /// Turn exact verify windows on, once, and hold the tower to the profile
    /// its serving bytes are pinned to.
    ///
    /// On, every projection takes a row-exact product and the row-exact
    /// kernels of each shape are compiled at 1 to 8 rows before the first
    /// window needs them; prompt chunks keep MLX's arithmetic on the model's
    /// grid. That warm-up is on unless the process environment turns it off by
    /// name, which is the switch's own contract and stays a process fact.
    static func enableExactWindows(_ model: MuseGlimmerTextModel) throws {
        if !model.verifiesWindowsExactly {
            do {
                try model.setRowExactWindows(true)
            } catch {
                throw RunnerError.numericsProfileRefused(
                    expected: servedNumericsProfile, observed: "\(error)")
            }
        }
        guard model.verifiesWindowsExactly, model.numericsProfile == servedNumericsProfile else {
            throw RunnerError.numericsProfileRefused(
                expected: servedNumericsProfile, observed: model.numericsProfile)
        }
    }

    // MARK: - Drafter

    /// Bind the published block drafter, or serve serial when none is named.
    ///
    /// An absent `--drafter` is the plain greedy build, not a broken one. A
    /// named directory is held to two refusals before a byte of it is read as
    /// a tensor: its own size (the assistant bound) and, at load, the pin.
    public static func loadDrafter(
        options: RunnerLoadOptions,
        directory: URL,
        target: any LanguageModel
    ) async throws -> (any CBv2MTPDrafter)? {
        // A module the caller already holds went around the pin, the size cap,
        // the artifact type and the target binding. Refused before a byte of it
        // is bound.
        if let refusal = preloadedDrafterRefusal(options) { throw refusal }
        guard let drafterDirectory = options.drafterDirectory else { return nil }
        guard let target = target as? MuseGlimmerTextModel else {
            throw RunnerError.unexpectedModel(
                "\(type(of: target)) — the block drafter binds to the native text tower")
        }
        if let refusal = try drafterRefusal(directory: drafterDirectory) { throw refusal }
        do {
            let draft = try MuseGlimmerDFlashDraftModel.load(
                directory: drafterDirectory,
                pin: MuseGlimmerDFlashArtifactPin.museGlimmer30BQuantized,
                maximumBytes: maximumDrafterBytes)
            return try MuseGlimmerDFlashDrafter(
                target: target, model: draft, settings: drafterSettings(target))
        } catch {
            throw RunnerError.drafterUnavailable("\(error)")
        }
    }

    /// The assistant bound, checked on the directory's own file sizes.
    ///
    /// Returns the refusal, or nil when the artifact is within the bound. It
    /// reads sizes and nothing else, so a directory over the bound is refused
    /// without hashing or mapping it.
    public static func drafterRefusal(directory: URL) throws -> RunnerError? {
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])
        var bytes = 0
        for entry in entries {
            let values = try entry.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true else { continue }
            bytes += values.fileSize ?? 0
        }
        guard bytes <= maximumDrafterBytes else {
            return .drafterOverByteCap(bytes: bytes, cap: maximumDrafterBytes)
        }
        return nil
    }

    /// The served round shape: one rectangular exact verify forward per round,
    /// each row committing its own accepted prefix, the requests of a step
    /// sharing the drafter's passes, and the argmax proposal. Coupled
    /// proposals stay off — a real-weight replay measured lower sampled
    /// acceptance with them, and they change nothing about the output.
    static func drafterSettings(_ target: MuseGlimmerTextModel) -> MuseGlimmerDFlashDrafter.Settings {
        var settings = MuseGlimmerDFlashDrafter.Settings.verifying(
            target,
            maximumSpeculativeRows: speculativeRows,
            backendScratchBytes: 0,
            batchesVerifyColumnsAsDecodeRows: true,
            couplesSampledProposals: false,
            batchesRequests: true)
        settings.maximumDraftTokens = min(maximumDrafterTokens, CBv2MTPConfig.testedMaxDraftTokens)
        return settings
    }

    // MARK: - Serving

    public func loadSummary(
        weights: URL, options: RunnerLoadOptions
    ) -> RunnerLoadSummary {
        var summary = genericLoadSummary(weights: weights, options: options)
        summary.add("numerics_profile", model.numericsProfile)
        summary.add("exact_windows", model.verifiesWindowsExactly)
        summary.add("prompt_chunk_alignment", model.cbv2Capabilities.prefillChunkAlignment ?? 0)
        summary.add(
            "mtp_hidden_layers", model.mtpHiddenLayerIDs.map(String.init).joined(separator: ","))
        summary.add(
            "checkpoint_kv_dtypes", model.kvStorageDTypes.map { "\($0)" }.joined(separator: ","))
        if let drafter {
            summary.add(
                "draft_max_tokens",
                drafter.maximumDraftTokens.map { String($0) } ?? "unbounded")
            summary.add(
                "draft_verification",
                drafter.requiredVerificationMode?.rawValue ?? "engine_default")
            summary.add("draft_speculative_rows", drafter.maximumSpeculativeBatch ?? 1)
        }
        return summary
    }

    private func newCaches(
        _ make: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) throws -> [any CBv2AttendingLayerCache] {
        try model.newCacheV2(makeLayerCache: make)
    }

    public func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: servingModel,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: newCaches,
            nativePaged: NativePagedPlan(layerDTypes: model.kvStorageDTypes),
            // The tower's own view of itself: prompt chunks run its prefill
            // entry point (the fixed 1,024-token grid), the engine reads its
            // capabilities — paged requirement, grid, K/V dtypes — from it,
            // and with a drafter bound it also hands over the captured
            // residual stream. The generic adapter carries none of that.
            steppable: MuseGlimmerCBv2Adapter(model),
            mtpDrafter: drafter,
            // The paged pool must hold one prompt cell, or the engine refuses
            // the model at start for ANY caller whose chunk is below the grid.
            // Raised here, off the LOADED model's own declaration, so no
            // caller carries a number that has to be kept in step by hand.
            build: Self.aligningPromptGrid(build, alignment: model.cbv2Capabilities.prefillChunkAlignment))
    }

    /// The build with its prefill chunk raised to the model's prompt grid.
    ///
    /// A grid model cuts every prompt into `alignment`-token cells and a paged
    /// pool shorter than one cell refuses the model. A build that asks for less
    /// is not wrong about what it wants — it is a caller that does not know the
    /// grid — so the runner lifts it to the grid instead of letting the engine
    /// terminate the process.
    static func aligningPromptGrid(_ build: EngineBuild, alignment: Int?) -> EngineBuild {
        guard let alignment, alignment > 0,
            build.schedulerConfig.prefillChunkSize < alignment
        else { return build }
        var build = build
        build.schedulerConfig.prefillChunkSize = alignment
        return build
    }

    /// One teacher-forced row over the same forward the engine drives. The
    /// stepper is single-row and contiguous by contract; the tower's paged
    /// requirement is an ENGINE requirement, not a row requirement.
    public func makeStepper() throws -> any TeacherForcedStepper {
        CBv2SingleRowStepper(
            model: servingModel,
            layerKinds: layerKinds,
            newCaches: newCaches,
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
