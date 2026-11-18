import Foundation
import MLXFastEngine
import Testing

// The view carries no model identity: it checks the pinned intake LAYOUT and
// copies the staged snapshot's own `config.json` verbatim. The shard count is
// the contract fixture's `raw_checkpoint_shard_count`, pinned in the type
// rather than passed by a caller, so no caller can ask for a looser check.

private let stagedConfig = Data(
    #"{"model_type":"muse_glimmer","text_config":{"num_hidden_layers":52}}"#.utf8)

private func stagedIntake(
    shards: Int,
    writeConfiguration: Bool = true
) throws -> URL {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    if writeConfiguration {
        try stagedConfig.write(to: root.appendingPathComponent("config.json"))
    }
    for index in 1 ... max(shards, 1) where shards > 0 {
        let name = String(format: "model-%05d-of-00003.safetensors", index)
        try Data([UInt8(index)]).write(to: root.appendingPathComponent(name))
    }
    return root
}

@Test("the intake shard count is the pinned target's, not the caller's")
func checkpointIntakeViewPinsTheTargetShardCount() throws {
    // `raw_checkpoint_shard_count` in the contract fixture.
    #expect(CheckpointIntakeView.expectedShardCount == 3)
}

@Test("the checkpoint view copies the staged config and links every shard")
func checkpointIntakeViewCopiesTheStagedConfiguration() throws {
    let fileManager = FileManager.default
    let weights = try stagedIntake(shards: 3)
    let temporary = weights.appendingPathComponent("tmp")
    try fileManager.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: weights) }
    try Data("manifest only\n".utf8).write(
        to: weights.appendingPathComponent("SHA256SUMS"))

    let view = try CheckpointIntakeView(source: weights, temporaryDirectory: temporary)
    let entries = try fileManager.contentsOfDirectory(atPath: view.directory.path)
    #expect(entries.count == CheckpointIntakeView.expectedShardCount + 1)
    #expect(entries.contains("config.json"))
    #expect(!entries.contains("SHA256SUMS"))

    // Byte-identical to the staged snapshot's own file: the view never
    // authors a configuration of its own.
    let written = try Data(contentsOf: view.directory.appendingPathComponent("config.json"))
    #expect(written == stagedConfig)
    let object = try #require(
        JSONSerialization.jsonObject(with: written) as? [String: Any])
    #expect(object["model_type"] as? String == "muse_glimmer")
}

@Test("the checkpoint view refuses an intake with no configuration")
func checkpointIntakeViewRequiresTheStagedConfiguration() throws {
    let fileManager = FileManager.default
    let weights = try stagedIntake(shards: 3, writeConfiguration: false)
    defer { try? fileManager.removeItem(at: weights) }

    #expect(
        throws: CheckpointIntakeError.missingConfiguration(
            weights.appendingPathComponent("config.json").path)
    ) {
        _ = try CheckpointIntakeView(source: weights)
    }
}

@Test("the checkpoint view refuses an incomplete shard set")
func checkpointIntakeViewRejectsMissingShard() throws {
    let fileManager = FileManager.default
    let weights = try stagedIntake(shards: 1)
    defer { try? fileManager.removeItem(at: weights) }

    #expect(throws: CheckpointIntakeError.shardCount(expected: 3, actual: 1)) {
        _ = try CheckpointIntakeView(source: weights)
    }
}
