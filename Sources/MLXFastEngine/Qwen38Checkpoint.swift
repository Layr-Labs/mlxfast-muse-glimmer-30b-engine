import Foundation

public enum Qwen38CheckpointError: Error, CustomStringConvertible, Equatable {
    case notDirectory(String)
    case shardCount(expected: Int, actual: Int)
    case unreadableShard(String)

    public var description: String {
        switch self {
        case .notDirectory(let path):
            return "Qwen3.8 checkpoint is not a directory: \(path)"
        case .shardCount(let expected, let actual):
            return "Qwen3.8 checkpoint needs \(expected) safetensors shards; found \(actual)"
        case .unreadableShard(let path):
            return "Qwen3.8 checkpoint shard is not readable: \(path)"
        }
    }
}

/// A temporary checkpoint view for the fixed Qwen/Qwen3.8-27B intake.
///
/// The intake stages only 18 safetensors shards and `SHA256SUMS`. The model
/// loader also needs `config.json`. This view writes the pinned config under
/// `TMPDIR` and links the shards into it. It does not alter the staged files.
public final class Qwen38CheckpointView: @unchecked Sendable {
    public static let expectedShardCount = 18
    public static let modelRevision = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"

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
            throw Qwen38CheckpointError.notDirectory(source.path)
        }

        let shards = try fileManager.contentsOfDirectory(
            at: source, includingPropertiesForKeys: [.isRegularFileKey])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard shards.count == Self.expectedShardCount else {
            throw Qwen38CheckpointError.shardCount(
                expected: Self.expectedShardCount, actual: shards.count)
        }
        for shard in shards {
            let values = try shard.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true, fileManager.isReadableFile(atPath: shard.path)
            else {
                throw Qwen38CheckpointError.unreadableShard(shard.path)
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
            "mlxfast-qwen38-\(UUID().uuidString)", isDirectory: true)
        do {
            try fileManager.createDirectory(
                at: directory, withIntermediateDirectories: false)
            // Not .atomic: Foundation's atomic write path is denied by the
            // resident's seatbelt profile even when the target directory is
            // allowed via (allow file-write* (subpath TMPDIR)); a plain
            // write passes. This is sandbox-private scratch, so atomicity
            // carries no weight here.
            try Self.configuration.write(
                to: directory.appendingPathComponent("config.json"))
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

    /// Qwen/Qwen3.8-27B `config.json` at ``modelRevision``, reduced to the
    /// text fields the MLX model reads. A root EOS id supplies the runner's
    /// token-id-only stop policy.
    public static let configuration = Data(
        #"""
        {
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "eos_token_id": 248044,
          "model_type": "qwen3_5",
          "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attn_output_gate": true,
            "bos_token_id": 248044,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 17408,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "output_gate_type": "swish",
            "pad_token_id": null,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
              "mrope_interleaved": true,
              "mrope_section": [11, 11, 10],
              "partial_rotary_factor": 0.25,
              "rope_theta": 10000000,
              "rope_type": "default"
            },
            "tie_word_embeddings": false,
            "use_cache": true,
            "vocab_size": 248320
          },
          "tie_word_embeddings": false
        }
        """#.utf8)
}

/// The intake layout check above is all this file carries now.
///
/// REMOVED with the move to the runner registry: a loader that built a
/// family-local text model and adopted it through a family runner. Both are
/// gone (this tree carries the runner boundary and no family), and the
/// bench-worker's entry point resolves its runner from the registry and loads
/// through the one construction path in `MLXRunners`.
