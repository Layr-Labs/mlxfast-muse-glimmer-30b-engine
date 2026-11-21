import Foundation
import MLXFastCore

/// Tensor namespaces of the pinned Muse Glimmer 30B NVFP4 checkpoint
/// (`model_type` "muse_glimmer", nvidia/Muse-Glimmer-30B-NVFP4 @ 47818374).
///
/// The text tower is `model.language_model.*` plus the untied `lm_head.*`.
/// The vision stack is `model.vision_tower.*`, `model.vision_adapter.*` and
/// `model.vision_projection.*`; the runtime verifies those tensors and does
/// not load them (MuseGlimmerInstallation drops the namespace at load).
///
/// WHY THE VISION TENSORS ARE CARRIED, NOT STRIPPED. The runtime loads only a
/// checkpoint whose index and three shards match the pinned SHA-256 values
/// (MuseGlimmerCheckpointPin), and shards 1 and 3 hold text and vision
/// tensors side by side. Stripping the vision tensors would rewrite those two
/// shards and the index, and the runtime would refuse the tree. So this
/// family selects the text tower, refuses any third namespace, and passes
/// every pinned shard through byte-identical (an APFS clone on one volume).
enum MuseGlimmerCheckpointLayout {
    static let modelType = "muse_glimmer"

    static let textTowerPrefixes = ["model.language_model.", "lm_head."]
    static let visionPrefixes = [
        "model.vision_tower.", "model.vision_adapter.", "model.vision_projection.",
    ]

    static func isTextTowerKey(_ key: String) -> Bool {
        textTowerPrefixes.contains { key.hasPrefix($0) }
    }

    static func isVisionKey(_ key: String) -> Bool {
        visionPrefixes.contains { key.hasPrefix($0) }
    }

    /// The pass-through set: the text tower and the vision tensors the pinned
    /// shards carry beside it.
    static func isSelectedKey(_ key: String) -> Bool {
        isTextTowerKey(key) || isVisionKey(key)
    }

    /// Refuses an index with no text tower or with a namespace that is
    /// neither text nor the known vision stack.
    static func validateIndexKeys<Keys: Sequence>(_ keys: Keys) throws
    where Keys.Element == String {
        var textTensorCount = 0
        for key in keys {
            if key.hasPrefix(textTowerPrefixes[0]) {
                textTensorCount += 1
            } else if !isTextTowerKey(key), !isVisionKey(key) {
                throw MLXFastError.invalidInput(
                    "muse_glimmer checkpoint tensor \(key) is outside the text tower and the vision stack"
                )
            }
        }
        guard textTensorCount > 0 else {
            throw MLXFastError.invalidInput(
                "muse_glimmer checkpoint index contains no model.language_model. tensors"
            )
        }
    }
}
