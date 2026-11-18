import Foundation
import MLXFastEngine
import Testing

@Test("Qwen3.8 checkpoint view embeds config and ignores SHA256SUMS")
func qwen38CheckpointView() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let weights = root.appendingPathComponent("weights")
    let temporary = root.appendingPathComponent("tmp")
    try fileManager.createDirectory(at: weights, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    for index in 1 ... Qwen38CheckpointView.expectedShardCount {
        let name = String(format: "model-%05d-of-00018.safetensors", index)
        try Data([UInt8(index)]).write(to: weights.appendingPathComponent(name))
    }
    try Data("manifest only\n".utf8).write(
        to: weights.appendingPathComponent("SHA256SUMS"))

    let view = try Qwen38CheckpointView(
        source: weights, temporaryDirectory: temporary)
    let entries = try fileManager.contentsOfDirectory(atPath: view.directory.path)
    #expect(entries.count == Qwen38CheckpointView.expectedShardCount + 1)
    #expect(entries.contains("config.json"))
    #expect(!entries.contains("SHA256SUMS"))

    let object = try #require(
        JSONSerialization.jsonObject(with: Qwen38CheckpointView.configuration)
            as? [String: Any])
    #expect(object["model_type"] as? String == "qwen3_5")
    let text = try #require(object["text_config"] as? [String: Any])
    #expect(text["num_hidden_layers"] as? Int == 64)
    #expect(text["hidden_size"] as? Int == 5120)
}

@Test("Qwen3.8 checkpoint view refuses an incomplete shard set")
func qwen38CheckpointViewRejectsMissingShard() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }
    try Data().write(to: root.appendingPathComponent("model-00001-of-00018.safetensors"))

    #expect(throws: Qwen38CheckpointError.shardCount(expected: 18, actual: 1)) {
        _ = try Qwen38CheckpointView(source: root)
    }
}
