// make-dflash-drafter -- write the quantized DFlash drafter this track pins.
//
// The drafter the engine loads (`--drafter <tree>/weights/drafter`) is
// PRODUCED, not downloaded: the pinned artifact writer converts the published
// BF16 assistant (meta-models/Muse-Glimmer-30B-assistant at c676cce9) with the
// pinned quantized plan. Until this tool, only an environment-gated study test
// in Vendor/mlx-swift-lm (writesTheQuantizedArtifact) called that writer. This
// tool calls the same public Vendor API in the same order, and edits nothing
// in Vendor/:
//
//   1. load the BF16 source through the strict loader, which checks both of its
//      files against `MuseGlimmerDFlashArtifactPin.museGlimmer30BBF16`;
//   2. copy it into a fresh BF16 model, as the study test does;
//   3. quantize it with `museGlimmer30BQuantizedPlan`;
//   4. write the artifact (`writeArtifact`) into an empty directory;
//   5. refuse unless the written pin is `museGlimmer30BQuantized`, then load it
//      back through the strict loader with that pin.
//
// tools/make-drafter.sh runs this tool, checks both written files against the
// track fixture's mtp_head.artifact pins, and moves the directory into place.
//
// Usage: make-dflash-drafter --source <bf16 dir> --output <empty dir>
// Exit:  0 written and verified, 1 refused or failed, 2 bad arguments.

import Foundation
import MLX
import MLXLLM
import MLXNN

#if canImport(Darwin)
    import Darwin
#endif

struct Refusal: Error, CustomStringConvertible {
    let description: String
}

func standardError(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

func usage() -> Never {
    standardError("usage: make-dflash-drafter --source <bf16 assistant dir> --output <empty dir>")
    exit(2)
}

func parseArguments(_ arguments: [String]) -> (source: URL, output: URL) {
    var source: String?
    var output: String?
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard index + 1 < arguments.count else { usage() }
        switch flag {
        case "--source": source = arguments[index + 1]
        case "--output": output = arguments[index + 1]
        default: usage()
        }
        index += 2
    }
    guard let source, let output else { usage() }
    return (URL(fileURLWithPath: source), URL(fileURLWithPath: output))
}

/// The BF16 source's tensor payload bound, the value the study test loads with.
let sourceMaximumBytes = 5_111_970_304

/// The provenance line the study test writes into `manifest.json`. It is not
/// part of either pinned file, so the two hashes do not depend on it.
let sourceDescription =
    "meta-models/Muse-Glimmer-30B-assistant@c676cce91658f9568aaf264a416211e2782735b7 "
    + "model.safetensors sha256 \(MuseGlimmerDFlashArtifactPin.museGlimmer30BBF16.weightsSHA256); "
    + "MLX affine quantization, variant q5mlp-q8rest"

func run(source: URL, output: URL) throws {
    let expected = MuseGlimmerDFlashArtifactPin.museGlimmer30BQuantized
    let loaded = try MuseGlimmerDFlashDraftModel.load(
        directory: source, pin: .museGlimmer30BBF16, maximumBytes: sourceMaximumBytes)
    let configuration = loaded.configuration

    let model = try MuseGlimmerDFlashDraftModel(configuration, dtype: .bfloat16)
    let parameters = loaded.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
    try model.update(parameters: .unflattened(parameters), verify: .all)
    eval(model)
    try model.quantize(configuration.museGlimmer30BQuantizedPlan)

    let written = try model.writeArtifact(to: output, source: sourceDescription)
    guard written.pin.weightsSHA256 == expected.weightsSHA256,
        written.pin.configurationSHA256 == expected.configurationSHA256
    else {
        throw Refusal(
            description:
                "the written drafter does not match its pin: model.safetensors \(written.pin.weightsSHA256) "
                + "(pinned \(expected.weightsSHA256)), config.json \(written.pin.configurationSHA256) "
                + "(pinned \(expected.configurationSHA256))")
    }
    _ = try MuseGlimmerDFlashDraftModel.load(
        directory: output, pin: expected, maximumBytes: written.manifest.totalBytes)
    print(
        "make-dflash-drafter: wrote \(output.path): model.safetensors \(written.pin.weightsSHA256), "
            + "config.json \(written.pin.configurationSHA256), \(written.manifest.totalBytes) bytes")
}

let (source, output) = parseArguments(Array(CommandLine.arguments.dropFirst()))
do {
    try run(source: source, output: output)
} catch {
    standardError("make-dflash-drafter: REFUSED: \(error)")
    exit(1)
}
