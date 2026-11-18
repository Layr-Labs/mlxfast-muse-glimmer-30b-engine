// Copyright © 2026 Eigen Labs.
import MLXLMCommon

/// Text-only load for the Qwen 3.8 27B intake checkpoint: the same
/// architecture as `Qwen35Model`, with the checkpoint's 333
/// `model.visual.*` vision-tower tensors excluded by name at load
/// (`WeightNameFiltering`), mirroring `PrismHadamardQwen35TextModel`.
///
/// This subclass must live in this module: `Qwen35Model` is
/// `public`, not `open`, so it cannot be subclassed from
/// `MLXFastEngine` or any other module.
public final class Qwen38TextOnlyModel: Qwen35Model, WeightNameFiltering {
    public override init(_ args: Qwen35Configuration) {
        super.init(args)
    }

    public func shouldLoadWeight(named name: String) -> Bool {
        !name.hasPrefix("vision_tower") && !name.hasPrefix("model.visual")
    }
}
