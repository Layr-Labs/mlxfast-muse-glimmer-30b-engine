# mlxfast — Muse Glimmer 30B MLX

This repository is the engine for the Muse Glimmer 30B MLX speedup benchmark.
The track identifier is `muse-glimmer-30b-mlx-v1`.

> **NOTE — the target is pinned and official scoring is armed.**
> The target is `nvidia/Muse-Glimmer-30B-NVFP4` at `47818374`, served through
> the paged engine with the quantized DFlash drafter pinned in the contract
> fixture's `mtp_head` block. A default `./setup.sh` run still refuses the
> download by name; [Current status](#current-status) states why. The contract
> fixture sets
> `official_scoring_enabled` to `true` and pins three live goldens (`pool-9`,
> `pool-10` and `pool-11`) in `timed_prompt_pool`.
> [`docs/scoring-contract.md`](docs/scoring-contract.md) states how a ranked
> run becomes a score, each line with its source.

## What this repository is

This repository holds the engine. The engine is the Swift and Metal code that
runs the target model on Apple Silicon. You optimize the engine. You make the
model do the same work in less time.

The benchmarker measures the engine. The benchmarker is a separate program
called `benchd`. It arrives as a verified prebuilt binary. It owns all
timing, all scoring, and all gates. Nothing in this repository measures or
scores anything.

The ranked run runs ONE STREAM AT A TIME. The ranked run measures your engine
and a serial control engine in the same session, on the same box. The score
compares the two. The control engine is the organizer's reference tree, and its
cost is measured in that same session. No file stores it.

> **NOTE — the track goldens are not in this repository.**
> A track's timed tapes and per-depth oracles are organizer material. They are
> published in R2 at the `r2_path` keys the contract pins. The ranked box
> stages them out of band into the directory its runner service exports as
> `MLXFAST_QWEN38_GOLDEN_DIR`. That environment name keeps its Qwen spelling
> because it is the fleet's contract with the runner service, not this track's
> own name. `tools/ranked-box-preflight.sh` verifies every file there against
> the contract's `{sha256, bytes}` and refuses an extra `*.json`. They are
> never in git, so your clone does not carry them. The two public captures for
> the local modes are different material: they ship in this repository under
> `correctness_prompts/muse-glimmer-30b-mlx-v1/`.
>
> The fixture pins the three live goldens by digest. The two public captures
> ship; section [The public captures](#the-public-captures) names them.
>
> The organizer stages them with the signer this repository vendors:
>
> ```bash
> R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
>   tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
> tools/ranked-box-preflight.sh
> ```

> Section [Current status](#current-status) states the arm state and what
> still blocks a ranked run.
> [`Vendor/mlx-swift-lm/docs/muse-glimmer.md`](Vendor/mlx-swift-lm/docs/muse-glimmer.md)
> is the engineering record of the Muse Glimmer port.

### Lineage

This repository was seeded from the engines of earlier MLX tracks. Those
tracks rank different models under different rules. Only this track's rules
apply here.

The `.gemma4` transform family in `Sources/MLXFastTransform/` is not a
leftover. It is the synthetic-fixture substrate the transform pipeline tests
run on. Those tests exercise the transform, not a Gemma model.

## Requirements

- An Apple Silicon Mac.
- Enough unified memory for the target model and its working set. The target
  checkpoint is 24,669,346,036 bytes (`target.raw_checkpoint_total_size_bytes`
  in the contract fixture). A speculative run also loads the DFlash drafter,
  1,968,222,936 bytes. Add the paged KV cache and the decode buffers on top of
  that.
- Free disk space for the checkpoint, the drafter and the transformed
  `weights/`. `./setup.sh` checks for 22 GiB before the download starts. That
  default is less than the target checkpoint alone, so keep more space free.
  Change the limit with `MLXFAST_REFERENCE_MIN_FREE_GIB`.
- macOS 14 or later. `Package.swift` sets that platform floor. CI builds on a
  `macos-26` runner, and the Metal toolchain policy in `setup.sh` treats
  macOS 26 as its own case.
- Swift 6, through Xcode or through the Xcode Command Line Tools.
  `Package.swift` declares `swift-tools-version: 6.3`.
- The Xcode Metal Toolchain. `./setup.sh` tries to download it. The Command
  Line Tools alone carry no `metal` compiler, so a box that builds the metallib
  needs full Xcode. Install it, open it once, and accept the license with
  `sudo xcodebuild -license accept`.
- CMake. `./setup.sh` installs it through Homebrew when it is missing.
- Homebrew, when CMake is missing. Install it from <https://brew.sh> first.
  `./setup.sh` never downloads or runs the Homebrew installer; without `brew`
  it stops and names that link.
- Git.

You do not need Rust. The benchmarker arrives as a prebuilt binary.
`./tools/fetch-benchd.sh` fetches it from the public bench repository's dist
at a pinned commit and verifies it against that dist's `benchd.manifest.json`.
The default fetch needs no token.

Known limit: the public channel, `Layr-Labs/benchd`, holds only the prebuilt,
hash-pinned benchd dist. You cannot rebuild benchd from source from public
sources today. Verify the binary against its manifest instead.

## Quickstart

Run these commands in order. One sentence describes each command.

```bash
git clone https://github.com/Layr-Labs/mlxfast-muse-glimmer-30b-engine.git
```

This command copies the repository to your machine.

```bash
cd mlxfast-muse-glimmer-30b-engine
```

This command makes the repository your working directory.

```bash
./tools/fetch-benchd.sh
```

This command resolves the benchmarker binary from the dist channel into
`benchd-bin/` and verifies its sha256 and its byte count against the channel's
`benchd.manifest.json` (installed beside the binary).

```bash
./setup.sh
```

This command checks your toolchain, builds the two Swift binaries, builds
`mlx.metallib`, and downloads and verifies the target model. `setup.sh` reads
the target pin from the fixture's `target.upstream_model_id` and
`target.upstream_revision`. `MLXFAST_REFERENCE_MODEL_REPO` and
`MLXFAST_REFERENCE_REVISION` override it.

```bash
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
```

This command builds the two Swift binaries and `mlx.metallib` and skips the
model download. It still checks the toolchain, can ask to install the Xcode
Metal Toolchain and missing tools, and installs `macmon` unless
`MLXFAST_SKIP_MACMON_INSTALL=1` is set.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/bonsai2-9b-target \
  --output weights
```

This command converts the downloaded checkpoint into the `weights/` tree that
the engine loads. The `--reference` path is the `DEFAULT_REFERENCE_DIR` of
`setup.sh`. Its name is a legacy compatibility value from the source track.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test against the shipped local-iterate capture,
with the engine `./setup.sh` built at `.build/release/bench-worker`. See
[The public captures](#the-public-captures).

The DFlash drafter is a separate artifact. `./setup.sh` runs
`./setup-mtp-head.sh` for its slot. See [The DFlash drafter is a separate
artifact](#the-dflash-drafter-is-a-separate-artifact).

> **NOTE — the local modes need no environment variable.**
> `--local-iterate` uses the short shipped capture and `--local-submit` the
> long one. `MLXFAST_CORRECTNESS_GOLDEN_PATH` overrides that choice, and
> `MLXFAST_ENGINE_BIN` overrides the engine binary. `--official` has no
> default golden: it takes the organizer-provisioned hidden golden.

> **WARNING — the local test runs serial unless you give it a depth.**
> `./benchmark.sh` does not read `mtp-head.manifest.json`. Without
> `--mtp-depth`, the local test runs depth 0, also when your tree declares a
> depth. The ranked run uses the declared depth. To test the depth that your
> tree declares, add `--mtp-depth N`, where N is the value that
> `tools/spec-declaration.sh draft-len` prints. When that value is 0, do not
> add the flag.
>
> ```bash
> ./benchmark.sh --local-submit --mtp-depth "$(tools/spec-declaration.sh draft-len)"
> ```

> **WARNING — do not pass `--golden`, `--weights`, or `--score-path` to
> `./benchmark.sh`.**
> The script rejects these flags. Use the environment variables instead.
> `MLXFAST_WEIGHTS_PATH` defaults to `weights`.

### The public captures

The local test checks correctness against a public capture. The two public
captures ship in this repository under
`correctness_prompts/muse-glimmer-30b-mlx-v1/`. The organizer recorded them on
the track's box against the pinned target, serial, from one public 512-token
prompt. Greedy decode is deterministic, so the short capture is the long
capture truncated: its 128 expected decode tokens are the first 128 of the
long capture's 1023. Neither capture is a live or hidden golden.

| Capture | Path | Use |
|---|---|---|
| Short | `correctness_prompts/muse-glimmer-30b-mlx-v1/public-local-iterate.golden.json` | `--local-iterate`. The drift tripwire. |
| Long | `correctness_prompts/muse-glimmer-30b-mlx-v1/public-local-submit.golden.json` | `--local-submit`. The long capture. |

Nothing fetches them, and no environment variable is needed to use them. The
hidden goldens are different material: they live in R2 and on the ranked box.
See the note at the top of this file.

> **NOTE — a golden must name the checkpoint it came from.**
> Each `.json` golden must carry a `model_provenance` block. The block names
> the repository and the revision of the pinned model. The loader refuses a
> golden that carries no block. The error message names `model_provenance`.
> The loader does not skip the check when the block is absent.

## Repository structure

| Path | What it holds | Status |
|---|---|---|
| `Sources/MLXFastTransform/` | The offline transform that writes `weights/`. | Editable |
| `Sources/MLXFastCLI/` | The trusted CLI, `mlxfast-swift`. | Trusted |
| `Sources/MLXFastCore/` | Shared constants and contracts. | Trusted |
| `Sources/MLXFastTrustedHarness/` | The editable-surface budget, the head declaration reader, transform verification, and the metallib fingerprint. | Trusted |
| `Vendor/mlx-swift/` | The pinned MLX fork, a vendored tree. The listed Metal kernel sources are editable. | Mixed |
| `Vendor/mlx-swift-lm/` | The engine fork, a vendored tree. It holds the model, the runner, and `bench-worker`. The model files that `editablePaths` lists are editable. | Mixed |
| `fixtures/` | The track contract, the checkpoint manifests, the Muse Glimmer `config.json` copy, and configuration fixtures inherited from the source track. | Trusted |
| `tools/` | Setup, build, lint, and measurement scripts. | Trusted |
| `benchd-bin/` | Where `./tools/fetch-benchd.sh` installs the verified binary. Git ignores it. | Fetched |
| `mtp-head.manifest.json` | The speculative-decoder declaration. It names the decoder and the draft depth. It declares; it carries no weights. | Editable, optional |
| `correctness_prompts/` | The two public captures for the local modes. The hidden goldens live in R2 and on the ranked box, never here. | Trusted |
| `weights/` | The transformed weights the engine loads. | Generated |
| `benchmark.json` | The Yukon track manifest. It lists every editable path. | Trusted |

The launchd plist template in `tools/box-runner/` keeps the file name and
label of the source track. Those names are legacy compatibility values that
`install.sh` reads.

### Starting a new track from this repository

This repository is the template for the next track. Seed a new repository with a
copy of this tree, then run `tools/new-track.sh` in it. The script stamps the new
identity into the manifest, the contract fixture, the checkpoint file list, the
runner label, the engine pin and the docs. It touches no golden: a track's
goldens live in R2 and on its own box, never in git. It never commits.
`docs/new-track-repo-procedure.md` holds the full procedure and the usage line.

### The engine is a vendored tree

`Vendor/mlx-swift-lm` is a copy of `Layr-Labs/mlx-swift-lm`, as plain files.
That fork holds the three parts of the engine:

* the model, including the Muse Glimmer text tower and the DFlash drafter that
  binds to it;
* the runner, which loads the checkpoint and builds the engine and the one-row
  stepper;
* `bench-worker`, the Engine Protocol v1 server. The benchmarker starts this
  binary. One binary serves every model family, and it selects the runner from
  the checkpoint's own `model_type`.

A plain `git clone` carries the whole engine. This repository has no submodule.

The tree is the local merge of `Layr-Labs/mlx-swift-lm`
`feat/muse-glimmer-speed` `62cfabd5` and `feat/muse-glimmer-rowexact-m5-d5`
`44378936`, whose tree is `0391eacd84cb00ce4001d70763f2303fa2a01210`. No
upstream commit names that merge, so the contract fixture's
`mlx_swift_lm_revision` records the tree, not a commit. Engine-local changes
are commits of THIS repository. Nothing goes upstream. Edit the tree in place,
here.

The fork is a copy and not a submodule for one reason: its model files are an
editable surface. An editable path names bytes in this tree, and a gitlink
names a commit.

`Vendor/mlx-swift` is a vendored tree for the same reason: its Metal kernel
sources are editable. The fork asks for `mlx-swift` from the network. SwiftPM
uses this package's local `Vendor/mlx-swift` instead, because a local path
dependency of the root package wins. SwiftPM prints a "conflicting identity"
warning when it does this. The warning is expected.

The copy is cut from `Layr-Labs/mlx-swift`
`b1c94d29c5ddff16838fdc25b83fbe8fde48b3e8` (MLX 0.32.5). It holds the two
nested submodules of that repository as plain files: `Source/Cmlx/mlx` at
`72dffa2dbbdce95601fd84f5e8b6d7b4ec97a11b` and `Source/Cmlx/mlx-c` at
`2ac897e34495c924f9cecfa93fffaf4316bbcb91`. To move the pin, copy the whole
tree again from a fresh checkout of the new commit, with its submodules, then
rebuild the metallib.

### The DFlash drafter is a separate artifact

This track has one declarable speculative decoder, `dflash`. The target
checkpoint carries no MTP layer (`target.components.mtp` is `false`). The
drafter is produced from `meta-models/Muse-Glimmer-30B-assistant` by the
pinned DFlash artifact writer, and the contract fixture's `mtp_head` block pins
its two files by SHA-256. The block keeps the name `mtp_head` because
`mtp-head.manifest.json` is the declaration slot the trusted harness reads.

| Property | Value |
|---|---|
| Source repository | `meta-models/Muse-Glimmer-30B-assistant` @ `c676cce91658f9568aaf264a416211e2782735b7` |
| Model type | `muse_glimmer_assistant` |
| Hidden layers | 5, block size 16, target layer taps 1, 13, 25, 37, 49 |
| Quantization | MLP projections 5-bit affine at group 64; attention projections and the encoder output projection 8-bit; norms dense |
| Artifact | `config.json` and `model.safetensors`, 1,968,222,936 bytes, under the 2 GiB cap |
| Embeddings | None of its own. It shares the target's `embed_tokens`. |
| Permitted draft depths | 1 to 7 |
| Maximum speculative batch | 1. The arm is single-stream. |

No submission carries drafter weights. `./setup-mtp-head.sh` downloads the
pinned BF16 assistant into `reference_weights/muse-glimmer-30b-assistant`, and
`tools/make-drafter.sh` writes the quantized drafter from it into
`weights/drafter` and checks both files against the fixture's
`mtp_head.artifact` pins. `./setup.sh` runs it after the target. A dflash leg's
`tools/resident-up.sh --boot` names that tree's `weights/drafter` to the worker
with `--drafter`; the serial control leg loads no drafter.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 97 entries. The
rule behind the list is simple. Code that **proposes** tokens or computes the
forward pass is editable. Code that **verifies**, **measures**, or **ledgers**
stays trusted.

The editable surface has five groups.

1. The decoder declaration. `mtp-head.manifest.json`. The declaration file
   only. It names the decoder and the draft depth.
2. The offline transform. `Sources/MLXFastTransform/`.
3. The model files: the Muse Glimmer target, the DFlash drafter and the
   files they share. The exact list is below this one, generated from
   `benchmark.json`.
4. The batching engine and the cache layer: the whole
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` directory
   (the CBv2 engine, the MTP round driver, the paged KV pool and its kernels),
   twelve `Libraries/MLXLMCommon/` cache, attention, RoPE and config files, and
   `Libraries/MLXLMServer/Runtime/ToolStreamHandler.swift`. This is the shape
   the Gemma 4 track opened.
5. The 68 vendored MLX Metal kernel files the forward pass dispatches. These
   are the quantized matmul, the gather-GEMM, SDPA and steel attention,
   RoPE, RMSNorm, softmax, sort, reduce, copy, elementwise, `arg_reduce`, and
   gather indexing.

<!-- BEGIN GENERATED: editable model files (tools/editable-paths-doc.sh) -->
The 13 model files on the editable surface, of the 97 `benchmark.json`
`editablePaths` entries. This list is generated from `benchmark.json`;
run `tools/editable-paths-doc.sh --write` after the list changes.

- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/MuseGlimmer/MuseGlimmerTextModel+CBv2.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/MuseGlimmer/DFlash/MuseGlimmerDFlashDrafter.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/PrismHadamardQwen35.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35+CompleteCheckpoint.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35A3BTargetVerify.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP+PrefixCheckpoint.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTPTopTwo.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MoE.swift`
- `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/PrismHadamardCheckpoint.swift`
- `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`
<!-- END GENERATED: editable model files -->

The rest of the engine fork is trusted: the runner and its registry,
`bench-worker`, the server, and every `Libraries/MLXLMCommon/` file not named
above.

### The decoder declaration

`mtp-head.manifest.json` is the declaration file. It stays editable and
optional. `spec.decoder` names the decoder the submission arms, and this track
declares `dflash`:

```json
"spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": 3 }
```

An absent `decoder` key reads as `"dflash"` in `tools/spec-declaration.sh`.
The two trusted readers do not yet agree on the older slot name `"mtp"`:
`tools/spec-declaration.sh` accepts it, and the Swift reader in
`Sources/MLXFastTrustedHarness/MTPHeadDeclaration.swift` refuses it as an
unknown decoder. Declare `dflash`.

The shipped declaration switches the decoder off, so the stock tree runs
serial (depth 0):

```json
"spec": { "decoder": "dflash", "enabled": false, "num_speculative_tokens": 0 }
```

The declaration accepts `"source": "pinned"` only. On this track that means the
drafter the fixture's `mtp_head` block pins. `"source": "remote"` and
`"source": "in_branch"` are refused by name.

The drafter's cap is 2 GiB (`max_bytes` = 2147483648). A declaration may lower
the cap. It may not raise it.

An absent declaration selects the pinned drafter slot. That is the normal case.
A declaration that is present but broken is a refusal. The runner never falls
back silently.

The size cap is the only gate on a declaration. A declared `sha256` is
optional, and the runner does not verify it against the head bytes.
`docs/participant-contract.md` section 4.3 states that limit plainly.

`tools/resident-up.sh` validates the leg's own declaration through
`tools/spec-declaration.sh`. It decides nothing else: the worker takes the
decoder and the depth from the wire, per request.

The MTP head is the organizer's pinned weights. You may re-quantize it. You may
not replace it, and you may not upload head weights of your own. Custom head
weights are not accepted on this track. A re-quantization happens ON LOAD, in
memory. Nothing on disk changes. The head loader calls `quantize(model:)` while
it binds the head, and the file that holds that call is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.
Change the geometry that call selects. `docs/participant-contract.md` section
4.4 is the authority.

The head only **proposes** tokens. The pinned target model decides every
emitted token. Both legs load the head; only the candidate leg drafts with it.

### The draft depth

> **NOTE — the draft depth is a free lever.**
> The scored stream count is 1 and you may not tune it. The draft depth is
> yours. You declare it in `mtp-head.manifest.json`, which is editable:
>
> ```json
> "spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> The drafter permits a depth from 1 to 7
> (`mtp_head.permitted_draft_depths`). With no `spec` block,
> `enabled: false`, or `0`, the run is serial (depth 0). A depth outside the
> list is refused before the engine starts.
> `tools/spec-declaration.sh describe` prints what your tree declares.
> The ranked run sends the depth to the engine on each request and seals the
> engine's `effective_spec` echo.
>
> A declared depth is scored against the oracle recorded AT THAT DEPTH for each
> live golden (`fixtures/muse_glimmer_30b_mlx_v1_track.json`
> `live_golden_speculative`, keyed by depth and then by live golden name). The
> runner verifies a draft window in one target forward, and the multi-row
> kernels round differently from the single-row ones, so a speculative run is
> not always token-identical to the serial tape; it is token-identical to its
> own depth's tape, which the organizer records on the pinned runner.
> Correctness at a depth means matching that tape.
>
> Every run seals what actually ran: `effective_spec` for the declared depth,
> `effective_mean_draft_len` for the realized draft length.

Try a depth locally with one flag. The local flag names the depth directly, so
you do not have to edit the declaration to try a value. The local test runs
serial unless you give it `--mtp-depth`.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test with no speculation.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the local test with the MTP head at depth 3. The flag also
exports the head directory to the worker in `BENCH_WORKER_DRAFTER`.

### The byte budget

`benchmark.json` `editableSurfaceByteBudget` caps the enforced editable
surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 7949663 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is absent. The exemption existed to let head weights ride in a
submission outside the source budget. A submission carries no head weights, so
there is nothing to exempt. The two exempt caps stay declared because both
enforcers carry the same numbers as compiled-in fallbacks and this manifest is
what holds them to a reviewed value.

### The target quantization is frozen

The target model's quantization is frozen as shipped. Do not re-quantize a
target weight. Do not re-represent one. Do not change the numerical format of
one. This holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model instead of
optimizing the accepted one.

The MTP head is a narrow exception, and the exception is re-quantization only.
You may re-quantize the head. You may not replace it. The head stays within its
2 GiB declaration cap. The head only proposes tokens, and the pinned target
decides every emitted token.

### What you must not change

- Everything in `Sources/` that `editablePaths` does not list.
- `Package.swift` and `Package.resolved`. The dependency graph is frozen.
- Everything in `Vendor/` that `editablePaths` does not list.
- `fixtures/`, `benchmark.json`, the scripts, the tests, and the
  documents.
- `weights/`, the reference checkpoints, the scores, and the goldens.

Do not hardcode hidden prompts. Do not hardcode hidden token identifiers. Do
not use timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

Do not add a cache keyed on a request's input tokens whose only possible hit is
the harness repeating one identical computation. The benchmark measures
single-pass inference. Input-independent caches stay legal. These are weights,
dequantized tensors, and RoPE or mask tables keyed on shapes and offsets.
Within-request KV reuse also stays legal.

## Local testing vs the ranked run

The local test and the ranked run are different by design. Read this section
before you tune.

The local test runs a **single stream**. It uses a public golden. It prints a
single-stream estimate.

### What each local mode checks

Both local modes run one fused checked-timing pass. The pass teacher-forces the
golden's expected tokens and times the wall clock. It judges correctness from
that same pass. A mismatch is reported as a teacher-forced token mismatch.

A correctness failure does not discard the timing. The benchmarker reruns the
timing phase in a mismatch-tolerant form. It then reports the correctness
failure together with real timing numbers.

| Mode | Decode steps | Expected tokens the golden must hold | Cool gate |
|---|---|---|---|
| `--local-iterate` | 128 | 129 | On, because `./benchmark.sh` always passes `--cool-gate` |
| `--local-submit` | 1023 | 1024 | On |

The gate is on because `./benchmark.sh` arms it. Driving the Swift CLI directly
skips it and times a hot GPU. Use `./benchmark.sh`. See AGENTS.md, "The
cool-down gate".

The two public captures differ in length for this reason. `--local-iterate`
uses the 256-token capture and `--local-submit` the 1024-token capture by
default.

> **NOTE — both local modes check correctness and speed.**
> Neither local mode is a speed-only signal. Both apply the teacher-forced
> check. Neither one runs the ranked gates.

The ranked run also runs one stream at a time, so the two shapes agree. A local
number is still only a smoke signal: your machine is not the ranked box.

> **WARNING — a local score is directional, not predictive.**
> Treat a local score as a smoke signal for speed and correctness. Do not treat
> it as a prediction of the ranked composite. The ranked M5 run is the
> authority.

## Scoring and gates

[`docs/scoring-contract.md`](docs/scoring-contract.md) is the scoring contract:
the formula, the direction, the units, rounding and ties, the correctness gate,
the estimator, the trial count and the variance limits, each with its source in
the code. This section explains them.

### The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1.

The decode gain is measured over the decode window only. The seed prefill of
the prompt is not in it. The prefill gain is measured over its own prefill
phase. benchd refuses a leg whose seed step is too slow against its timed
prefill (the early-decode limit, one measured value per declared spec:
`docs/scoring-contract.md` section 2).

### The pair is measured, not stored

A ranked run measures pairs. It measures them on the SAME box, in the SAME
job. The fixture lists the live prompts in `live_goldens`. Pair k times the
prefill of live prompt (k - 1) mod N and the decode (the seed step and the
decode window) of live prompt k mod N, because the fixture's
`decode_golden_offset` is 1. N is the number of live prompts, so
`official_pairs` must be a multiple of N. A pair measures TWO legs over the
same prompts:

1. The **serial-control leg**. It runs on the organizer's reference tree. That
   tree is a build of this repository at the commit the fixture names in
   `baseline_reference_commit`. This leg uses no speculation. Its tokens are
   checked against the serial tape, never against a per-depth tape.
2. The **candidate leg**. It runs on your tree, at the draft depth you declare.

The score is the ratio of the two measurements. Both numbers come from the same
machine, minutes apart.

**NO FILE HOLDS A BASELINE PAIR.** The scoring constants hold none. The fixture
holds none. The goldens hold none. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is refused on the ranked path.
`tools/lint-benchmark-manifest.py` keeps the two fields out of the tree.

The organizer stages the reference tree on each ranked box. The ranked job
verifies that tree. It does not fetch it and it does not build it.

The reference tree is a build of the control commit named by
`baseline_reference_commit`. Re-pin it, and re-stage the reference tree and
recalibrate on every box, whenever a trusted-side code change merges that the
control leg must carry.

### Each box has its own calibration

Each ranked box records what its own serial-control leg costs. The record is a
file. `MLXFAST_BASELINE_CALIBRATION` names it.

**THE FILE IS A HEALTH BAND. IT IS NEVER A DENOMINATOR.** The benchmarker
compares the measured control leg against the band. The run stops by name when
the leg falls outside the band. A stale calibration file can stop a run. It can
never move a score.

An operator writes the file on the box:

```bash
tools/calibrate-box.sh "<runner name>" /path/to/baseline-calibration.json
```

The command takes the box GPU lock. It then runs the serial-control leg three
times for each live golden under the full official methodology: the quiescence
gate and the cool gate before each pass, one resident worker for each pass, and
the live goldens the ranked run scores over. It writes one entry for each live
golden: the mean, the coefficient of variation and the band for prefill and for
decode. Decode is the decode window. It writes no file when the coefficient of
variation is more than 1 percent on either axis: the fixture's
`calibration_max_cv_percent` is 1.0. A box that
cannot repeat itself has no band. `tools/ranked-box-preflight.sh` accepts calibration file
version 3 only.

The `box` value in the file must equal the runner name. The `reference_commit`
value must equal the fixture's `baseline_reference_commit`.
`tools/ranked-box-preflight.sh` refuses the run when either differs.

### The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 512 |
| Checked decode steps | 128 |
| Golden shape | 512 prompt tokens and 129 expected tokens |
| Streams per window | 1 |
| Timed prompts per pair | 2 (pair k: prefill of live golden (k - 1) mod N, decode of live golden k mod N) |
| Legs per pair | 2 (serial control, then candidate) |

The two legs run one after the other. Each leg loads the weights once. The
unmeasured warm-up prefill pass stays at 1 pass, and it applies to both legs in
the same way. It runs the never-scored `warmup_golden` prompt, and each pair
decodes the next live golden after the one it prefills (`decode_golden_offset`).

The serial-control leg runs entirely inside the reference tree. It uses that
tree's worker, that tree's Metal library and that tree's own transformed
weights. Nothing you change can move it.

### One resident worker per LEG, booted by the benchmarker

A ranked job has two legs on two trees. The weights load ONCE per leg.

The benchmarker starts `bench-worker runtime-worker` once for each phase: the
warmup, the timed prefill, the timed decode and the correctness pass. Each start
used to load the whole checkpoint again. It no longer does: a resident owns the
weights for the leg, and each per-phase worker attaches to it.

**THE RESIDENT BELONGS TO THE LEG, NOT TO THE WINDOW.** The two legs run
different trees with different weights, so one resident cannot serve both. The
benchmarker knows where a leg begins and ends, so the benchmarker boots it. For
each leg it calls that leg's OWN copy of `tools/resident-up.sh`:

```bash
tools/resident-up.sh --boot --spec <serial|dflash> --draft-len <N> --socket-out <file>
tools/resident-up.sh --stop --socket <path>
```

`--boot` loads that tree's own `.build/release/bench-worker` and that tree's
own `weights/`, plus, for a dflash leg, that tree's own quantized drafter
`weights/drafter` (`--drafter`), waits for a healthy hello,
writes the socket path as the first line
of the `--socket-out` file, and exits 0 with the resident still running. A
`<socket>.pid` sidecar and a `<socket>.ready` marker sit beside the socket.
`--stop` ends that resident and removes all three files. A second `--stop` is a
no-op.

`--draft-len` is 0 with `serial`. With `dflash` it must be the depth the tree's
own declaration names (`tools/spec-declaration.sh describe` prints
`dflash<N>`). A mismatch, a missing drafter file and any other `--spec` value
are refusals.

`--spec` is authoritative over what the leg DOES. It is the label the leg
records, so a reference tree that declares a draft depth still boots a serial
control leg when the benchmarker says serial. The worker takes the decoder and
the depth from the wire, per request, and never from the tree.

`--spec` does NOT choose the drafter. The boot validates the leg's OWN
`mtp-head.manifest.json` through `tools/spec-declaration.sh`, and a dflash boot
loads the tree's own `weights/drafter`. Nothing crosses between the two trees.

`tools/stage-baseline-workspace.sh` stages no drafter: the reference tree is
the serial-control leg, and a serial boot loads none. The ranked job installs
the drafter into the candidate tree from `MLXFAST_MTP_HEAD_REFERENCE_DIR`
(default `$MLXFAST_REFERENCE_DIR/drafter`), the box's verified copy, after the transform and only for a dflash declaration.

`tools/muse-glimmer-30b-measure-and-score.sh` boots nothing. It takes the box GPU
lock `/tmp/mtplx-gpu-exclusive.lock` and holds it for the whole measurement,
because a resident holds the whole checkpoint in unified memory whoever booted
it, and the box needs exactly one loader. `tools/resident-up.sh` refuses
to boot when nobody holds that lock, so every per-leg boot happens inside that
window.

| Name | What it is |
|---|---|
| `BENCH_WORKER_RESIDENT_SOCKET` | A leg's resident Unix socket. A per-phase worker attaches to it instead of loading. The measure script never sets it, and REFUSES when the environment does. |
| `RESIDENT_IDENTITY_FILE` | A JSON file that records the resident, its socket, its declared leg and its hello. |

> **WARNING — do not set `BENCH_WORKER_RESIDENT_SOCKET` on a ranked box.**
> It names one already-loaded resident. Both legs would attach to it, so the
> serial-control leg would run on the candidate's weights. The measure script
> and `tools/ranked-box-preflight.sh` both refuse it.

The wrapper form of `tools/resident-up.sh` stays for local, unscored use:

```bash
tools/resident-up.sh --weights <dir> -- <command>
```

It boots one resident, exports `BENCH_WORKER_RESIDENT_SOCKET` to the command,
and halts the resident when the command ends. The ranked path does not use it.

### The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 3 |
| `minPairsPerCohort` | 3 |
| `decodeSpeedupFloor` | 0.95 |
| `decodeSpeedupCeiling` | 5.0 (the fixture's `decode_speedup_ceiling`) |
| `kvBackend` | `paged` |

The scored width is fixed. A width the benchmarker has not certified has no
series tag, and the benchmarker refuses that width rather than run it.

The fixture's `official_pairs` sets how many pairs one ranked run measures.
Each pair scores on its own control leg. The run score is the mean of the
per-pair composites (`official_pair_combine: "mean"`). Every pair stays in the
artifact as measured.

The candidate's slowness gate is the two floors: 0.95 on the prefill gain and
0.95 on the decode gain. The plausibility gate is the fixture's
`decode_speedup_ceiling` of 5.0: a pair whose decode gain is above 5.0 refuses
the run with `SPEEDUP-ABOVE-CEILING` and seals `score: null`; a gain of exactly
5.0 passes. The prefill gain has its own ceiling, the
fixture's `prefill_speedup_ceiling` (its value: `docs/scoring-contract.md`
section 2), with the same refusal. Two runs whose scores are equal tie;
in the final ranking, ties go to the most recent submission. The control leg's
health band is the box calibration.
The decode band allows 3 percent upward.

`kvBackend` is pinned `paged` on both legs. The model declares
`requiresPagedKV`, and its engine build refuses a contiguous cache. The
benchmarker refuses when it cannot honour the pinned backend. It does not
degrade to another backend.

### Token fidelity

The benchmarker applies two token checks.

1. **The timed window is exact.** The fixture declares no
   `timed_token_tolerance_per_thousand`, so the prefill token, the seed token
   and each of the 128 committed decode tokens must equal the tape. One
   different token fails the run.
2. **The untimed correctness gate has a 10% budget.** Each teacher-forced case
   of 64 steps passes when at most 10% of its steps are not accepted. A
   top-logit tie is accepted.

A speculative declaration needs a per-depth tape in the fixture's
`live_golden_speculative`. That block pins one for every live golden at
`dflash1` to `dflash7`; each equals the serial tape of its prompt byte for byte
(greedy decode), so each names that tape's object. A depth with no tape is
refused before it is measured.

A near-tie argmax can diverge across Apple Silicon generations, even for
correct code. Before you treat a local failure as your own regression, check
whether an unmodified `main` fails at the same token position on your machine.

### Current status

Seven statements are true right now.

1. The target is pinned. `target.upstream_model_id` and
   `target.upstream_revision` name `nvidia/Muse-Glimmer-30B-NVFP4` at
   `47818374517751c48c55cde2621594926b1888b6`, and
   `Sources/MLXFastCore/Constants.swift` names the same pin. The fixture's
   `target.manifest_path` pins the checkpoint file by file. `setup.sh` and
   `setup-mtp-head.sh` read their repository and revision from the fixture.
   The drafter slot's manifest (`mtp_head.manifest_path`) pins the BF16
   source's two files, and `mtp_head.artifact` pins the produced drafter.
2. `fixtures/muse_glimmer_30b_mlx_v1_track.json` sets
   `official_scoring_enabled` to `true`. That flag is the single authority on
   the arm state, and the benchmarker enforces it. It refuses to seal an
   official scoring artifact while the flag is false or absent.
3. `timed_prompt_pool` pins three goldens by `sha256` and `bytes`, and
   `live_goldens` names them: `pool-9`, `pool-10` and `pool-11`, each at a
   512-token seed. `hidden_correctness_golden` is pinned by digest.
4. No runner advertises the ranked label set
   `[self-hosted, macOS, muse-glimmer-30b-mlx-v1]`, and no box is staged
   with the goldens, the reference tree or a calibration file.
   `tools/stage-baseline-workspace.sh` builds the reference tree, and
   `tools/calibrate-box.sh` writes the calibration file, when a box is ready.
5. Scoring is single-stream and paired. `scored_batch_size` is `1`. The
   candidate leg runs one stream at its declared depth. The serial-control leg
   runs on the organizer's reference tree at the fixture's
   `baseline_reference_commit`. The score is the mean of the per-pair live
   ratios. No golden carries a baseline pair.
6. benchd is pinned in `tools/fetch-benchd.sh`: `BASE_URL`, `PINNED_COMMIT`
   (the dist commit) and `PINNED_SOURCE_COMMIT` (the source that dist was built
   from). The script reads `dist/benchd.manifest.json` at that commit, refuses
   a manifest that names another `source_commit`, verifies the binary against
   the `sha256` and `bytes` the manifest names, and prints the identity it
   resolved. Read the pin off the script, not off this page.
7. A ranked run can score a serial or a dflash declaration. The fixture's
   `live_golden_speculative` pins a per-depth tape for every live golden at
   `dflash1` to `dflash7` (each byte-equal to the serial tape of its prompt),
   and `tools/resident-up.sh --boot --spec dflash` boots a dflash leg with the
   tree's own `weights/drafter`.

The engine serves the Muse Glimmer 30B text tower. The vision tower in the
checkpoint is not loaded.
[`Vendor/mlx-swift-lm/docs/muse-glimmer.md`](Vendor/mlx-swift-lm/docs/muse-glimmer.md)
holds the port detail.

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

<!-- YUKON-CLI-INSTALL: owner fills -->
Install the Yukon CLI: <install source to be added>

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login
```

This command authenticates you. It prints a sign-in URL and a code; approve
the request with GitHub in any browser.

```bash
yukon clone <benchmark-id-or-name>
```

This command clones the benchmark repository. The benchmark name in
`benchmark.json` is `mlxfast-muse-glimmer-30b`.

```bash
yukon setup
```

This command runs `./setup.sh`.

```bash
yukon run
```

This command runs `benchmarkCommand`, which is
`tools/muse-glimmer-30b-yukon-run.sh`. Off a ranked box it transforms the weights
when `weights/` is absent, then runs `./benchmark.sh --local-submit` at the
declared MTP depth. The result has `"score": null`. Only the ranked runner
gives a score.

```bash
yukon submit --model "<exact model name>" --note-file submission-note.md
```

This command uploads your editable-path archive.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first.
No local run blocks the upload. Run the local test yourself before you submit.

## The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `nvidia/Muse-Glimmer-30B-NVFP4` @ `47818374517751c48c55cde2621594926b1888b6` |
| Target manifest | The fixture's `target.manifest_path`: 13 records, every one the pinned checkpoint. The file name is a legacy compatibility value from the source track. |
| DFlash drafter | Produced from `meta-models/Muse-Glimmer-30B-assistant` @ `c676cce91658f9568aaf264a416211e2782735b7`; the two artifact files are pinned by SHA-256 in the fixture's `mtp_head.artifact` |
| Geometry source | The pinned snapshot's `config.json`, checked in as `fixtures/muse_glimmer_30b_nvfp4_config.json` |
| Inherited fixtures | `fixtures/qwen35_9b_base_config.json` and a packing-reference `config.json` from the source track; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) attributes both |
| Benchmarker | The bench repository's dist at the commit `tools/fetch-benchd.sh` pins (`PINNED_COMMIT`, `PINNED_SOURCE_COMMIT`); `benchd.manifest.json` is the authority for the bytes, and the script logs the resolved `source_commit`/sha256 |
| Track id | `muse-glimmer-30b-mlx-v1` — the PLATFORM name: leaderboard namespace, runner labels, R2 prefix |
| Engine fork upstream base | tree `0391eacd84cb00ce4001d70763f2303fa2a01210`, the merge of `62cfabd5c67c6042edcfd812698bd34e4b5df7d7` and `443789360ddbf804d55d063ca9cda4ea2d3013a9` (a local merge branch, not pushed upstream; engine-local changes on top) |

### The target model

The values below are the contract fixture's `target` block, read off
`fixtures/muse_glimmer_30b_nvfp4_config.json`, a byte copy of the pinned
snapshot's own `config.json`.

| Property | Value |
|---|---|
| Architecture | `MuseGlimmerForConditionalGeneration`. The model type is `muse_glimmer`; the text model type is `muse_glimmer_text`. Schema 2. |
| Hidden layers | 52. Three sliding-attention layers, then one full-attention layer, repeated. |
| Layer counts | 39 sliding attention at window 2048, 13 full attention |
| Full-attention layers | At indices 3, 7, 11, … 51 |
| Attention | 32 query heads, 2 KV heads, head dimension 128, no bias, output gate on |
| Rotary | `rope_theta` 500000 on the sliding layers. The full-attention layers have no position encoding (NoPE). |
| MLP | Dense, width 19968, `silu` activation. No experts. |
| Hidden size | 6656 |
| Vocabulary | 202048 |
| Embeddings | Untied |
| Context | 131072 |
| Tokens | bos 200000, eos 200001 |
| Output | `final_logit_softcapping` 20.0, `output_multiplier` 0.19611613513818404 |
| Quantization | Mixed precision modelopt (`nvfp4-mixed`): 4-bit NVFP4 at group 16 with float8 e4m3 scales on 213 modules, float8 on 180 modules |
| Served tensors | 1413: 1410 language-model tensors and 3 `lm_head` tensors |
| Vision | The checkpoint carries a 50-layer vision tower. This track serves text only and never loads it. |
| Base model | The same checkpoint, never re-quantized and never re-represented |
| License | Apache-2.0 (`target.license`) |

Every layer owns its own paged KV cache. No layers share KV, and no layer
uses attention sinks.

## Building after an edit

Two build forms matter, because the vendored MLX package builds in JIT mode.

Kernel families with an `mlx-generated/*.cpp` twin compile at runtime from the
C++ source strings inside those files. For these families the twin is the
runtime-effective source. Edit the twin. Keep the readable `.metal` and `.h`
pair in step with it.

RoPE, RMSNorm, the SDPA vector kernel, and `arg_reduce` load ahead of time from
`mlx.metallib`. After you edit one of those sources, run
`tools/build-mlx-metallib.sh`. `./setup.sh` runs that script for you.

`_nax` names are the M5-generation kernel variants. The ranked runner selects
them. Tune the `_nax` twin as well as the plain one.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI into `.build/release`.

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker \
  --product bench-worker
```

This command builds the scored engine into `.build-worker/release`. The engine
is the fork's generic `bench-worker`, and this package builds it as a
dependency product, so the build reads `Vendor/mlx-swift` and your kernel edits
reach it.

```bash
tools/stage-bench-worker.sh
```

This command copies the engine and its `mlx.metallib` into `.build/release`,
where the benchmarker resolves them.

> **WARNING — always pass `--force-resolved-versions`.**
> The dependency graph is frozen. A bare `swift build` or `swift test` can
> rewrite `Package.resolved`. `./setup.sh` then refuses to run. Restore the
> file with `git checkout -- Package.resolved`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. It runs repository hygiene checks on `ubuntu-latest`. It runs
`swift build --build-tests` and `swift test` on a hosted `macos-26` runner. It
treats first-party warnings as errors.

CI is advisory. No status check is required. A red run blocks neither a merge
nor a dispatch. `docs/ci-coverage.md` holds the detail.

CI never measures and never scores. CI holds no secret, downloads no weights,
and runs no GPU test. The GPU tests and the checkpoint tests are box-only.
`docs/ci-coverage.md` lists them, and CI fails when that list drifts.

`.github/workflows/benchmark.yml` is the ranked pipeline. It triggers on
`workflow_dispatch` only. It holds no secret. Its ranked job runs on
`[self-hosted, macOS, muse-glimmer-30b-mlx-v1]`.
The primary guard is the Yukon intake: an entry is built from the base plus
only the files in `editablePaths`. The in-workflow surface check
(`.github/scripts/enforce-modifiable-surface.sh`, the surface-check step of
`.github/workflows/benchmark.yml`) is a second line that a changed workflow
file could remove. Before the ranked job measures, it verifies
every box-staged asset against the contract's `{sha256, bytes}` pins. It
publishes no score until the runner is registered, the box is staged, and the
benchmarker emits a composite. Each of those gaps gives a non-zero exit and no
artifact.

## Where to get help

| Question | Authority |
|---|---|
| What the track measures, path by path | `benchmark.json` |
| Pins, the timed pool, scoring values | `fixtures/muse_glimmer_30b_mlx_v1_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering record for this port | `Vendor/mlx-swift-lm/docs/muse-glimmer.md` |
| What CI covers | `docs/ci-coverage.md` |
| Agent and contributor guidance | `AGENTS.md` |

> **NOTE — the order of authority.**
> The ranked M5 run is the authority on any score. The contract fixture
> `fixtures/muse_glimmer_30b_mlx_v1_track.json` wins over this document. This document
> only explains; it never overrides. If either disagrees with the benchmarker
> about measurement, the benchmarker wins.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The
contract fixture records the checkpoint's license as Apache-2.0. This
repository distributes no model weights.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) holds the full third-party
attribution.
