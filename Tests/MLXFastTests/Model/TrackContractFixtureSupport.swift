import CryptoKit
import Foundation
import MLXFastCore
import Testing

/// Fixture locations and pins for the track target,
/// `muse-glimmer-30b-mlx-v1`.
///
/// THE TARGET AND THE DRAFTER ARE PINNED. The checkpoint is
/// `nvidia/Muse-Glimmer-30B-NVFP4` at 4781837, and the DFlash drafter's two
/// files are pinned by SHA-256 in the contract fixture's `mtp_head` block.
///
/// THE HIDDEN GOLDEN IS NOT. It is organizer material and does not exist yet,
/// so `hidden_correctness_golden.sha256` carries this track's own pending
/// sentinel. Exact-match only, never a prefix check. The sentinel is NOT
/// `MLXFastConstants.pendingOrganizerSentinel`: that constant is the
/// reference-download gate `setup.sh` compares against, a different contract.
let trackGoldenSentinel = "MUSE-GLIMMER-30B-MLX-V1-PENDING-ORGANIZER"

/// The target's own `config.json`, checked in VERBATIM as
/// `fixtures/muse_glimmer_30b_nvfp4_config.json`. The digest below is the
/// fixture's own digest, and the contract fixture pins the same three values
/// under `target.config_fixture`; a test holds them together.
let targetConfigSHA256 =
    "f3add3cfff62038fc7087e72f7df5a560660707aba7c8e65761bfa6bf8f08774"
let targetConfigByteCount = 88_682

/// THE PACKED-FAMILY SUBSTRATE. The transform tests exercise the packed
/// families the transform serves, and those need a packed config to read. The
/// fields come from the packing reference's own published `config.json`
/// (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ 3f926b41), checked in VERBATIM
/// as `fixtures/bonsai2_27b_config.json`, and the geometry comes from that
/// family's published backbone `config.json`, checked in VERBATIM as
/// `fixtures/qwen35_9b_base_config.json`. Both are cited by repository and
/// revision: a citation that does not match its source is false. Neither is
/// this track's target, and nothing outside `Tests/` reads them.
let packedFamilyBaseRepository = "Qwen/Qwen3.5-9B"
let packedFamilyBaseRevision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
let packedFamilyBaseConfigSHA256 =
    "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
let packedFamilyBaseConfigByteCount = 3_126

let packingReferenceRepository = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
let packingReferenceRevision = "3f926b415992eaa2ae9dd7b573706494d6bbf787"
let packingReferenceConfigSHA256 =
    "238de7c512cc56a733421e3fd011d88f8260739e3d00e32c5d65b7943cc9f837"
let packingReferenceConfigByteCount = 58_145

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let repositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let packedFamilyBaseConfigFixtureURL = repositoryRoot
    .appendingPathComponent("fixtures/qwen35_9b_base_config.json")

let packingReferenceConfigFixtureURL = repositoryRoot
    .appendingPathComponent("fixtures/bonsai2_27b_config.json")

let trackContractURL = repositoryRoot
    .appendingPathComponent("fixtures/muse_glimmer_30b_mlx_v1_track.json")

let targetConfigFixtureURL = repositoryRoot
    .appendingPathComponent("fixtures/muse_glimmer_30b_nvfp4_config.json")

let targetManifestURL = repositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_9b_2bit.sha256")

let headManifestURL = repositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_9b_mtp_head.sha256")

let headDeclarationURL = repositoryRoot
    .appendingPathComponent("mtp-head.manifest.json")

func jsonObject(at url: URL, describedAs description: String) throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: url)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput("\(description) must be a JSON object")
    }
    return object
}

func trackContractObject() throws -> [String: Any] {
    try jsonObject(at: trackContractURL, describedAs: "the track contract fixture")
}

func targetConfigObject() throws -> [String: Any] {
    try jsonObject(at: targetConfigFixtureURL, describedAs: "the target config fixture")
}

/// The target's nested text tower block.
func targetTextConfigObject() throws -> [String: Any] {
    let config = try targetConfigObject()
    return try #require(
        config["text_config"] as? [String: Any],
        "the target config fixture carries no text_config")
}

func packedFamilyBaseConfigObject() throws -> [String: Any] {
    try jsonObject(
        at: packedFamilyBaseConfigFixtureURL, describedAs: "the packed-family base config fixture")
}

func packingReferenceConfigObject() throws -> [String: Any] {
    try jsonObject(
        at: packingReferenceConfigFixtureURL, describedAs: "the packing reference config fixture")
}

/// A SYNTHETIC packed `config.json`, for the transform tests only.
///
/// No real config of that family at this size is published, so this one is the
/// packing reference's top-level fields, with the base backbone's
/// `text_config` and `vision_config` in place of the 27B's, and a `modules`
/// list regenerated for the 32-layer schedule in the packing reference's own
/// order: the head, then each layer from the last to the first (MLP, then the
/// mixer), then the embedding. The text config declares no MTP layer, as the
/// 27B pack does. It is never written into `fixtures/` and pins nothing.
func syntheticPackConfigObject() throws -> [String: Any] {
    var config = try packingReferenceConfigObject()
    let base = try packedFamilyBaseConfigObject()
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

/// The synthetic packed config written to a fresh temporary file, for the API
/// that reads a config by path.
func syntheticPackConfigURL() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("synthetic-pack-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("config.json")
    try JSONSerialization.data(
        withJSONObject: try syntheticPackConfigObject(),
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    ).write(to: url)
    return url
}

/// One `<sha256> <byte_count> <relative_path>` record of a pinned manifest.
/// Comment and blank lines are skipped.
struct PinnedManifestRecord: Equatable {
    let sha256: String
    let byteCount: Int
    let path: String
}

func pinnedManifestRecords(at url: URL) throws -> [PinnedManifestRecord] {
    let text = try String(contentsOf: url, encoding: .utf8)
    return try text.split(separator: "\n").compactMap { rawLine -> PinnedManifestRecord? in
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 3, let byteCount = Int(fields[1]) else {
            throw MLXFastError.invalidInput(
                "reference manifest record is not <sha256> <bytes> <path>: \(line)"
            )
        }
        return PinnedManifestRecord(
            sha256: String(fields[0]),
            byteCount: byteCount,
            path: String(fields[2])
        )
    }
}

func targetManifestRecords() throws -> [PinnedManifestRecord] {
    try pinnedManifestRecords(at: targetManifestURL)
}

func headManifestRecords() throws -> [PinnedManifestRecord] {
    try pinnedManifestRecords(at: headManifestURL)
}

/// The two shape pins a manifest header declares, read back out of it. They
/// are asserted against the records they must be summed from: a count that is
/// not summed from the digests it must agree with is not a pin.
func pinnedManifestHeader(at url: URL) throws -> (records: Int, bytes: Int) {
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

/// The fixture has two coherent states: NOT ARMED (scoring off, the hidden
/// golden carries the pending sentinel) and ARMED (scoring on, the golden is a
/// concrete pin). The arming commit flips `official_scoring_enabled`; the
/// contract suite reads that one switch and asserts the matching state as a
/// whole.
func trackContractIsArmed(_ object: [String: Any]) -> Bool {
    object["official_scoring_enabled"] as? Bool == true
}

/// A concrete pin: present, not empty, not the pending sentinel.
func isConcretePin(_ value: Any?) -> Bool {
    switch value {
    case let text as String: return !text.isEmpty && text != trackGoldenSentinel
    case let number as Int: return number >= 0
    case is Bool: return true
    default: return false
    }
}

func isHex64(_ value: Any?) -> Bool {
    guard let text = value as? String, text.count == 64 else { return false }
    return text.allSatisfy { $0.isHexDigit && !$0.isUppercase }
}

/// A 40-hex git commit id: the shape every commit pin in this tree is written
/// in. A 64-hex DIGEST check does not accept one, and a count check alone does
/// not reject an uppercase or non-hex character.
func isHex40(_ value: Any?) -> Bool {
    guard let text = value as? String, text.count == 40 else { return false }
    return text.allSatisfy { $0.isHexDigit && !$0.isUppercase }
}
