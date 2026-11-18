// RunnerRegistryTests.swift
//
// The registry is Darkbloom's advertise gate and bench-worker's resolution
// path, so an unclaimed `model_type` must REFUSE rather than fall through to
// some default family.

import Foundation
import Testing

@testable import MLXRunners

@Suite("Runner registry")
struct RunnerRegistryTests {

    @Test("Every model_type in one manifest resolves to that runner")
    func everyModelTypeResolvesToOneRunner() throws {
        RunnerRegistry.shared.register(MockRunner.self)
        let manifest = MockRunner.manifest
        let runner = try RunnerRegistry.shared.resolve(modelType: manifest.modelTypes[0])
        for modelType in manifest.modelTypes {
            #expect(RunnerRegistry.shared.contains(modelType: modelType))
            #expect(
                try RunnerRegistry.shared.resolve(modelType: modelType).manifest.runnerID
                    == runner.manifest.runnerID)
        }
    }

    @Test("An unclaimed model_type is refused, not defaulted")
    func refusesUnknown() {
        #expect(!RunnerRegistry.shared.contains(modelType: "not_a_model"))
        #expect(throws: RunnerRegistry.RegistryError.unknownModelType("not_a_model")) {
            _ = try RunnerRegistry.shared.resolve(modelType: "not_a_model")
        }
    }

    @Test("manifests() lists one manifest per runner, stably ordered")
    func manifestListing() {
        RunnerRegistry.shared.register(MockRunner.self)
        RunnerRegistry.shared.register(CohortMockRunner.self)
        let ids = RunnerRegistry.shared.manifests().map(\.runnerID)
        #expect(ids == ids.sorted())
        #expect(ids.contains("layr/mock-adapter"))
        #expect(ids.contains("layr/mock-cohort"))
        #expect(ids.count == Set(ids).count)
    }

    /// A track repo registers its own runner before resolving; this proves
    /// the claim lands on every `model_type` the manifest names. The mock
    /// used here is the teacher-forced one, so nothing in the test suite ever
    /// registers a model_type a real runner will claim later.
    @Test("A registered runner claims every model_type in its manifest")
    func registrationClaimsAll() throws {
        RunnerRegistry.shared.register(TeacherForcedOnlyMockRunner.self)
        #expect(RunnerRegistry.shared.contains(modelType: "mock-teacher-forced"))
        let resolved = try RunnerRegistry.shared.resolve(modelType: "mock-teacher-forced")
        #expect(resolved.manifest.runnerID == "layr/mock-teacher-forced")
    }
}
