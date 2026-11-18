// MuseGlimmerRunnerTests.swift
//
// The runner's declaration and its refusals, with NO weights: every case here
// is decided from the manifest, from file sizes, or from a checkpoint folder
// that fails the pin. Nothing loads a tensor, so these run on any machine.
//
// The suite cannot run on a Mac without Xcode (no swift-testing module), so it
// is held to `swiftc -parse` cleanness rather than a run until it reaches a box
// with the toolchain.

import Foundation
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("Muse Glimmer runner")
struct MuseGlimmerRunnerTests {

    // MARK: - Registry

    @Test("The registry resolves every model_type the runner claims")
    func registryResolves() throws {
        for modelType in MuseGlimmerRunner.manifest.modelTypes {
            #expect(RunnerRegistry.shared.contains(modelType: modelType))
            let resolved = try RunnerRegistry.shared.resolve(modelType: modelType)
            #expect(resolved.manifest.runnerID == MuseGlimmerRunner.manifest.runnerID)
        }
    }

    // MARK: - Declaration

    @Test("The manifest declares paged memory and the block decoder")
    func manifestDeclaresPagedAndBlockDecoder() {
        let manifest = MuseGlimmerRunner.manifest

        // Paged only: a contiguous build is not a slower arm of this family.
        #expect(manifest.kvBackends == [.paged])
        #expect(manifest.engine.supportsPagedKV)
        #expect(manifest.engine.supportsPackedPrefill == false)

        #expect(manifest.decoders.contains { $0.mode == DecoderID.serial.rawValue })

        let block = manifest.decoders.first { $0.mode == DecoderID.dflash.rawValue }
        #expect(block?.drafter == .assistantCheckpoint)
        #expect(block?.state == .requestStateful)
        #expect(block?.depth == 1 ... CBv2MTPConfig.testedMaxDraftTokens)
        #expect(MuseGlimmerRunner.maximumDrafterTokens <= CBv2MTPConfig.testedMaxDraftTokens)

        // The scored width is one, and a wider step must still be declared so
        // a cohort never has to be refused rather than served.
        let singleFreeRun = manifest.regimes.contains { regime in
            regime.batch == .single && regime.timing == .freeRun
        }
        let wideFreeRun = manifest.regimes.contains { regime in
            regime.batch == .upTo(MuseGlimmerRunner.speculativeRows)
                && regime.timing == .freeRun
        }
        let singleTeacherForced = manifest.regimes.contains { regime in
            regime.batch == .single && regime.timing == .teacherForced
        }
        #expect(singleFreeRun)
        #expect(wideFreeRun)
        #expect(singleTeacherForced)
    }

    @Test("A manifest digest is stable across reads")
    func manifestDigestStable() {
        #expect(
            MuseGlimmerRunner.manifest.sha256Digest()
                == MuseGlimmerRunner.manifest.sha256Digest())
        #expect(MuseGlimmerRunner.manifest.sha256Digest().count == 64)
    }

    // MARK: - Refusals

    @Test("A contiguous backend is refused by name")
    func contiguousBackendRefused() {
        let build = EngineBuild(kvBackend: .contiguous, kvBytesCapacity: 1 << 30)
        let refusal = RunnerEngineAssembly.backendRefusal(
            manifest: MuseGlimmerRunner.manifest, build: build)
        #expect(
            refusal
                == .kvBackendRefused(requested: "contiguous", declared: ["paged"]))
    }

    @Test("The declared paged backend is not refused")
    func pagedBackendAccepted() {
        let build = EngineBuild(kvBackend: .paged, kvBytesCapacity: 1 << 30)
        #expect(
            RunnerEngineAssembly.backendRefusal(
                manifest: MuseGlimmerRunner.manifest, build: build) == nil)
    }

    @Test("A drafter over the assistant bound is refused")
    func drafterOverCapRefused() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // A SPARSE file: the scan reads the logical size and nothing is
        // allocated for it, so the bound is exercised without 2 GiB of disk.
        let oversized = directory.appendingPathComponent("model.safetensors")
        #expect(FileManager.default.createFile(atPath: oversized.path, contents: nil))
        let handle = try FileHandle(forWritingTo: oversized)
        try handle.truncate(atOffset: UInt64(MuseGlimmerRunner.maximumDrafterBytes) + 1)
        try handle.close()

        #expect(
            try MuseGlimmerRunner.drafterRefusal(directory: directory)
                == .drafterOverByteCap(
                    bytes: MuseGlimmerRunner.maximumDrafterBytes + 1,
                    cap: MuseGlimmerRunner.maximumDrafterBytes))
    }

    @Test("A drafter within the bound passes the size check")
    func drafterWithinCapAccepted() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = directory.appendingPathComponent("model.safetensors")
        #expect(
            FileManager.default.createFile(
                atPath: weights.path, contents: Data(repeating: 0, count: 64)))
        #expect(try MuseGlimmerRunner.drafterRefusal(directory: directory) == nil)
    }

    @Test("A checkpoint folder that fails the pin is refused")
    func unpinnedCheckpointRefused() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // The published architecture decodes from an empty `text_config`, so
        // the refusal below is the PIN talking and not a configuration one.
        try Data(#"{"text_config":{}}"#.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        // An index whose bytes are not the pinned artifact's.
        try Data(#"{"weight_map":{"model.layers.0.mlp.down_proj.weight":"model-00001-of-00003.safetensors"}}"#.utf8)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))

        #expect(throws: MuseGlimmerInstallation.Failure.self) {
            _ = try MuseGlimmerInstallation.stage(directory: directory)
        }
    }

    // MARK: - Helpers

    private static func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("muse-runner-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }
}
