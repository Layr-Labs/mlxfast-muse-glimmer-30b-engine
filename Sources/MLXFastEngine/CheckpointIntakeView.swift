import Foundation

public enum CheckpointIntakeError: Error, CustomStringConvertible, Equatable {
    case notDirectory(String)
    case missingConfiguration(String)
    case shardCount(expected: Int, actual: Int)
    case unreadableShard(String)

    public var description: String {
        switch self {
        case .notDirectory(let path):
            return "checkpoint is not a directory: \(path)"
        case .missingConfiguration(let path):
            return "checkpoint carries no config.json: \(path)"
        case .shardCount(let expected, let actual):
            return "checkpoint needs \(expected) safetensors shards; found \(actual)"
        case .unreadableShard(let path):
            return "checkpoint shard is not readable: \(path)"
        }
    }
}

/// A temporary checkpoint view over the staged target's intake directory: the
/// staged `config.json` copied beside links to the staged safetensors shards.
///
/// WHAT IT CHECKS, AND WHY THE VALUES ARE NOT ARGUMENTS. The intake layout is
/// part of the contract, so the shard count and the configuration bytes are
/// pinned HERE, against the staged snapshot, rather than supplied by a caller:
/// an intake that stages a different number of shards, or no configuration at
/// all, is refused by name. A caller cannot ask for a looser check, and this
/// type carries no model identity of its own -- the identity lives in the
/// pinned snapshot's own `config.json`, which is copied verbatim.
public final class CheckpointIntakeView: @unchecked Sendable {
    /// The pinned target's safetensors shard count
    /// (`raw_checkpoint_shard_count` in the contract fixture).
    public static let expectedShardCount = 3

    /// The file the loader reads beside the shards.
    public static let configurationName = "config.json"

    public let source: URL
    public let directory: URL

    public init(
        source: URL,
        temporaryDirectory: URL? = nil,
        fileManager: FileManager = .default
    ) throws {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: source.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw CheckpointIntakeError.notDirectory(source.path)
        }

        let configurationURL = source.appendingPathComponent(Self.configurationName)
        guard let configuration = fileManager.contents(atPath: configurationURL.path) else {
            throw CheckpointIntakeError.missingConfiguration(configurationURL.path)
        }

        let shards = try fileManager.contentsOfDirectory(
            at: source, includingPropertiesForKeys: [.isRegularFileKey])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard shards.count == Self.expectedShardCount else {
            throw CheckpointIntakeError.shardCount(
                expected: Self.expectedShardCount, actual: shards.count)
        }
        for shard in shards {
            let values = try shard.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true, fileManager.isReadableFile(atPath: shard.path)
            else {
                throw CheckpointIntakeError.unreadableShard(shard.path)
            }
        }

        self.source = source.standardizedFileURL
        // The resident runs under sandbox-exec, where FileManager.default
        // .temporaryDirectory resolves to DARWIN_USER_TEMP_DIR — a path the
        // seatbelt profile denies. The boot contract gives the resident
        // TMPDIR (its private seatbelt-allowed directory), so honor it first.
        let tempRoot = temporaryDirectory
            ?? ProcessInfo.processInfo.environment["TMPDIR"].map {
                URL(fileURLWithPath: $0, isDirectory: true)
            }
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        self.directory = tempRoot.appendingPathComponent(
            "mlxfast-checkpoint-\(UUID().uuidString)", isDirectory: true)
        do {
            try fileManager.createDirectory(
                at: directory, withIntermediateDirectories: false)
            // Not .atomic: Foundation's atomic write path is denied by the
            // resident's seatbelt profile even when the target directory is
            // allowed via (allow file-write* (subpath TMPDIR)); a plain
            // write passes. This is sandbox-private scratch, so atomicity
            // carries no weight here.
            try configuration.write(
                to: directory.appendingPathComponent(Self.configurationName))
            for shard in shards {
                try fileManager.createSymbolicLink(
                    at: directory.appendingPathComponent(shard.lastPathComponent),
                    withDestinationURL: shard)
            }
        } catch {
            try? fileManager.removeItem(at: directory)
            throw error
        }
    }

    deinit {
        try? FileManager.default.removeItem(at: directory)
    }
}

/// The intake layout check above is all this file carries now.
///
/// REMOVED with the move to the runner registry: a loader that built a
/// family-local text model and adopted it through a family runner. Both are
/// gone (this tree carries the runner boundary and no family), and the
/// bench-worker's entry point resolves its runner from the registry and loads
/// through the one construction path in `MLXRunners`.
