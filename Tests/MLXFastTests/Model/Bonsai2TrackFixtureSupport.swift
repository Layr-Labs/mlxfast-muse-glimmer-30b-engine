import CryptoKit
import Foundation
import MLXFastCore
import Testing

/// Fixture locations and pins for the track target,
/// `muse-glimmer-30b-mlx-v1`.
///
/// THE TARGET AND THE HEAD ARE PENDING. prism-ml has not published the Bonsai 2
/// 9B pack or a head for it, so every pin that names their bytes carries this
/// exact sentinel. Exact-match only, never a prefix check.
let bonsai2PendingSentinel = "BONSAI-MLX-PENDING-ORGANIZER"

/// The geometry source until the pack is published: the Qwen 3.5 9B backbone's
/// own `config.json`, checked in VERBATIM as
/// `fixtures/qwen35_9b_base_config.json`. The digest below is the fixture's own
/// digest and the contract's `target.geometry_source` pin; a test holds them
/// together.
let bonsai2BaseRepository = "Qwen/Qwen3.5-9B"
let bonsai2BaseRevision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
let bonsai2BaseConfigSHA256 =
    "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
let bonsai2BaseConfigByteCount = 3_126

/// THE PACKING REFERENCE. The 9B pack is expected to use the Ternary Bonsai 2
/// 27B packing, so the 27B pack's own published `config.json`
/// (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ 3f926b41) is checked in
/// VERBATIM as `fixtures/bonsai2_27b_config.json`. It is the source of the
/// packing fields only, never of the geometry.
let bonsai2PackingRepository = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
let bonsai2PackingRevision = "3f926b415992eaa2ae9dd7b573706494d6bbf787"
let bonsai2PackingConfigSHA256 =
    "238de7c512cc56a733421e3fd011d88f8260739e3d00e32c5d65b7943cc9f837"
let bonsai2PackingConfigByteCount = 58_145

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let bonsai2RepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let bonsai2BaseConfigFixtureURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/qwen35_9b_base_config.json")

let bonsai2PackingConfigFixtureURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_27b_config.json")

let bonsai2TrackContractURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/muse_glimmer_30b_mlx_v1_track.json")

let bonsai2ReferenceManifestURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_9b_2bit.sha256")

let bonsai2HeadManifestURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_9b_mtp_head.sha256")

let bonsai2HeadDeclarationURL = bonsai2RepositoryRoot
    .appendingPathComponent("mtp-head.manifest.json")

func bonsai2TrackContractObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2TrackContractURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Bonsai 2 track contract fixture must be a JSON object"
        )
    }
    return object
}

func bonsai2BaseConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2BaseConfigFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Qwen 3.5 9B base config fixture must be a JSON object"
        )
    }
    return object
}

func bonsai2PackingConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2PackingConfigFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Ternary Bonsai 2 27B packing config fixture must be a JSON object"
        )
    }
    return object
}

/// A SYNTHETIC Bonsai 2 9B pack `config.json`, for the transform tests only.
///
/// The 9B pack is not published, so no real config exists. This one is the
/// packing reference's top-level fields, with the Qwen 3.5 9B backbone's
/// `text_config` and `vision_config` in place of the 27B's, and a `modules`
/// list regenerated for the 32-layer schedule in the packing reference's own
/// order: the head, then each layer from the last to the first (MLP, then the
/// mixer), then the embedding. The text config declares no MTP layer, as the
/// 27B pack does. It is never written into `fixtures/` and pins nothing.
func bonsai2SyntheticPackConfigObject() throws -> [String: Any] {
    var config = try bonsai2PackingConfigObject()
    let base = try bonsai2BaseConfigObject()
    var text = try #require(base["text_config"] as? [String: Any])
    text["mtp_num_hidden_layers"] = 0
    config["text_config"] = text
    config["vision_config"] = base["vision_config"]
    let layerTypes = try #require(text["layer_types"] as? [String])
    func module(_ path: String, embedding: Bool = false) -> [String: Any] {
        ["path": path, "block": 1024, "embedding": embedding, "dtype": "float16"]
    }
    var modules = [module("lm_head")]
    for index in layerTypes.indices.reversed() {
        let prefix = "model.layers.\(index)"
        for projection in ["up_proj", "down_proj", "gate_proj"] {
            modules.append(module("\(prefix).mlp.\(projection)"))
        }
        let mixer = layerTypes[index] == "full_attention"
            ? ["self_attn.o_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.q_proj"]
            : ["linear_attn.out_proj", "linear_attn.in_proj_z", "linear_attn.in_proj_qkv"]
        for projection in mixer {
            modules.append(module("\(prefix).\(projection)"))
        }
    }
    modules.append(module("model.embed_tokens", embedding: true))
    config["modules"] = modules
    return config
}

/// The synthetic pack config written to a fresh temporary file, for the API
/// that reads a config by path.
func bonsai2SyntheticPackConfigURL() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("bonsai2-synthetic-pack-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("config.json")
    try JSONSerialization.data(
        withJSONObject: try bonsai2SyntheticPackConfigObject(),
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    ).write(to: url)
    return url
}

/// One `<sha256> <byte_count> <relative_path>` record of a pinned manifest.
/// Comment and blank lines are skipped.
struct Bonsai2ManifestRecord: Equatable {
    let sha256: String
    let byteCount: Int
    let path: String
}

func bonsai2ManifestRecords(at url: URL) throws -> [Bonsai2ManifestRecord] {
    let text = try String(contentsOf: url, encoding: .utf8)
    return try text.split(separator: "\n").compactMap { rawLine -> Bonsai2ManifestRecord? in
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 3, let byteCount = Int(fields[1]) else {
            throw MLXFastError.invalidInput(
                "reference manifest record is not <sha256> <bytes> <path>: \(line)"
            )
        }
        return Bonsai2ManifestRecord(
            sha256: String(fields[0]),
            byteCount: byteCount,
            path: String(fields[2])
        )
    }
}

func bonsai2ReferenceManifestRecords() throws -> [Bonsai2ManifestRecord] {
    try bonsai2ManifestRecords(at: bonsai2ReferenceManifestURL)
}

func bonsai2HeadManifestRecords() throws -> [Bonsai2ManifestRecord] {
    try bonsai2ManifestRecords(at: bonsai2HeadManifestURL)
}

/// The two shape pins a manifest header declares, read back out of it. They
/// are asserted against the records they must be summed from: a count that is
/// not summed from the digests it must agree with is not a pin.
func bonsai2ManifestHeaderPins(at url: URL) throws -> (records: Int, bytes: Int) {
    let text = try String(contentsOf: url, encoding: .utf8)
    func pin(_ key: String) throws -> Int {
        guard let line = text.split(separator: "\n").first(where: { $0.contains(key) }),
              let value = line.split(separator: ":").last
                  .map({ $0.trimmingCharacters(in: .whitespaces) }),
              let number = Int(value)
        else {
            throw MLXFastError.invalidInput(
                "reference manifest header is missing \(key)"
            )
        }
        return number
    }
    return (
        try pin("MLXFAST_REFERENCE_MANIFEST_RECORDS"),
        try pin("MLXFAST_REFERENCE_MANIFEST_BYTES")
    )
}

/// Lowercase hex SHA256 of some bytes, the spelling every pin in this
/// repository is written in.
func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

/// The fixture has two coherent states: STAMPED-UNARMED (scoring off, every
/// pack pin the sentinel) and ARMED (scoring on, every pin concrete). The
/// arming commit flips `official_scoring_enabled`; the contract suite reads
/// that one switch and asserts the matching state as a whole.
func bonsai2TrackIsArmed(_ object: [String: Any]) -> Bool {
    object["official_scoring_enabled"] as? Bool == true
}

/// A concrete pin: present, not empty, not the pending sentinel.
func bonsai2IsConcretePin(_ value: Any?) -> Bool {
    switch value {
    case let text as String: return !text.isEmpty && text != bonsai2PendingSentinel
    case let number as Int: return number >= 0
    case is Bool: return true
    default: return false
    }
}

func bonsai2IsHex64(_ value: Any?) -> Bool {
    guard let text = value as? String, text.count == 64 else { return false }
    return text.allSatisfy { $0.isHexDigit && !$0.isUppercase }
}
