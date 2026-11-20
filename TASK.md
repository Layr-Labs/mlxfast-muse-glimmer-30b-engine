# The task — Muse Glimmer 30B MLX

Make the Muse Glimmer 30B model run faster on Apple Silicon.

The ranked track is `muse-glimmer-30b-mlx-v1`. Read [README.md](README.md) for the
setup steps and the repository structure. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind the rules.

## What you optimize

You optimize the engine. The engine is the MLX runner, the offline transform,
and the vendored MLX Metal kernels that the forward pass dispatches. You also
optimize the speculative-decode arm.

> **NOTE — the target is pinned; the default setup run still refuses.**
> The contract fixture pins `nvidia/Muse-Glimmer-30B-NVFP4` at
> `47818374517751c48c55cde2621594926b1888b6`. `setup.sh` and
> `setup-mtp-head.sh` still default their repository and revision to a legacy
> pending sentinel, so a run with no override refuses by name. README.md,
> "Current status", states each fact.

The target model is `nvidia/Muse-Glimmer-30B-NVFP4`. The text tower is 52
layers: three sliding-attention layers at window 2048, then one
full-attention layer with no position encoding, repeated, so 39 sliding and 13
full-attention layers. Attention has 32 query heads over 2 KV heads at head
dimension 128. The MLP is dense, width 19968, with no experts. The vocabulary
is 202048, and the embeddings are untied.

The weights are mixed precision modelopt: 4-bit NVFP4 at group 16 on 213
modules and float8 on 180 modules. Read
[`Vendor/mlx-swift-lm/docs/muse-glimmer.md`](Vendor/mlx-swift-lm/docs/muse-glimmer.md)
before you change a projection.

The checkpoint also carries a vision tower. This track serves text only, and
the model never loads that namespace.

The track has ONE declarable speculative decoder, `dflash`. It proposes
tokens, and the target model decides every emitted token. The drafter is
produced from a pinned assistant repository by the pinned DFlash artifact
writer.

| Decoder | Source | Pinned by | Depths |
|---|---|---|---|
| `dflash` | `meta-models/Muse-Glimmer-30B-assistant` @ `c676cce9`, quantized to 1,968,222,936 bytes | the fixture's `mtp_head.artifact` (two SHA-256 pins) | 1 to 7 |

The DFlash drafter proposes a block of 16 rows, and the target verifies a
chain at the declared depth.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 97 entries in five
groups.

| Group | Paths |
|---|---|
| The decoder declaration | `mtp-head.manifest.json` (the declaration file only) |
| The offline transform | `Sources/MLXFastTransform/` |
| The model files | `MuseGlimmer/MuseGlimmerTextModel+CBv2.swift` and `MuseGlimmer/DFlash/MuseGlimmerDFlashDrafter.swift`, plus files inherited from the source track: `PrismHadamardCheckpoint.swift`, `PrismHadamardQwen35.swift`, the 8 `Qwen35*.swift` files, and `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift` |
| The batching engine and cache layer | `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` and 12 `MLXLMCommon` cache, attention, RoPE and config files, plus `MLXLMServer/Runtime/ToolStreamHandler.swift` (the Gemma 4 shape) |
| The vendored kernels | The 68 MLX Metal files the forward pass dispatches |

The engine fork `Vendor/mlx-swift-lm` is a vendored tree of plain files. Its
listed model files, the batching engine and the cache layer are
editable. The runner, `bench-worker` and the server in the same fork stay
trusted.

The rule behind the list is simple. Code that **proposes** tokens or computes
the forward pass is editable. Code that **verifies**, **measures**, or
**ledgers** stays trusted.

The MTP head is the organizer's pinned weights. You may re-quantize it. You
may not replace it, and you may not upload head weights of your own.

No submission carries a head weight file. Nothing stages one.
`mtp-head.manifest.json` stays editable and optional, and it accepts
`"source": "pinned"` only, which on this track means the drafter the
fixture's `mtp_head` block pins. Its slot manifest holds no record yet.
`"source": "remote"` and `"source": "in_branch"` are refused by name.

The MTP head's declaration cap is 2 GiB. `max_bytes` may lower the cap and may
not raise it. The size cap is the only gate on a declaration. A declared
`sha256` is optional, and the runner does not verify it.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes, and no
artifact travels in a submission.

The head loader calls `quantize(model:)` while it binds the head. That call is
the seam, and the file that holds it is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.
Change the geometry that call selects. `docs/participant-contract.md` section
4.4 is the authority.

> **WARNING — the target quantization is frozen.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. An editable transform does not license the change. The MTP
> head is a narrow exception, and the exception is re-quantization only. You may
> re-quantize the head within its 2 GiB declaration cap. You may not replace it.

> **NOTE — the stream count is locked. The depth is not.**
> The ranked run scores one stream. You may not tune that.
>
> The draft depth is a free lever, declared in `mtp-head.manifest.json`, which
> is editable. The depth is not pinned at 1.
>
> ```json
> "spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> An absent `decoder` key reads as `"dflash"` in `tools/spec-declaration.sh`.
> Declare `dflash`: the Swift reader refuses the older slot name `"mtp"`. The
> shipped declaration switches the decoder off, so the stock tree runs serial
> (depth 0).
>
> Select a depth from 1 to 7 (`mtp_head.permitted_draft_depths`). A depth
> outside the list is refused before the engine starts.
>
> Each run seals `effective_spec` and `effective_mean_draft_len`, so the depth
> that ran and the draft length it realized are both visible afterwards.

You may not change anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing code.

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the Swift binaries and downloads the target model. It then
runs `./setup-mtp-head.sh` for the drafter slot. With no override of
`MLXFAST_REFERENCE_MODEL_REPO` and `MLXFAST_REFERENCE_REVISION`, it refuses by
name.

```bash
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
```

This command builds the Swift binaries and skips the model download. It still
checks the toolchain and can install missing tools.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/bonsai2-9b-target \
  --output weights
```

This command writes the `weights/` tree that the engine loads. The
`--reference` path is the `DEFAULT_REFERENCE_DIR` of `setup.sh`, a legacy
compatibility name from the source track.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test against the shipped local-iterate capture,
with no speculation. It needs no environment variable: the engine is the one
`./setup.sh` built, and the golden is the shipped capture.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the same local test at draft depth 3. The flag
also exports the head directory to the worker in `BENCH_WORKER_DRAFTER`.

> **WARNING — the local test runs serial unless you give it a depth.**
> `./benchmark.sh` does not read `mtp-head.manifest.json`. Without
> `--mtp-depth`, the local test runs depth 0, also when your tree declares a
> depth. The ranked run uses the declared depth. To test the declared depth,
> add `--mtp-depth N`, where N is the value that
> `tools/spec-declaration.sh draft-len` prints.

> **WARNING — no public capture ships yet.**
> The organizer records the two public captures on the track's box against the
> pinned target. They are not recorded yet, so the local test has no capture
> to check against.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = control_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1. The decode
gain is measured over the decode window only. The seed prefill is not in it.

The ranked run measures pairs on the same box, in the same job. A pair is a
serial-control leg on the organizer's reference tree, then your candidate leg
at the draft depth you declare, over one live prompt. Each leg times one
stream over a 512-token seed and a 128-step decode window. It runs 3 pairs,
and pair k measures live prompt (k - 1) mod N. The score is the mean of the
per-pair composites. Every pair must clear the 0.95 floor on prefill and on
decode. `benchmark.json` states a decode ceiling of 5.0, but the fixture
declares none, so no ceiling is applied. The KV backend is pinned `paged`.

The benchmarker applies two token checks. The timed window is exact: the
prefill token, the seed token and each of the 128 committed decode tokens must
equal the tape. The untimed correctness gate allows at most 10% of the steps
of each 64-step teacher-forced case to be not accepted.
[`docs/scoring-contract.md`](docs/scoring-contract.md) states both, each with
its source.

## The current state

> **NOTE — the scoring flag is armed.**
> `fixtures/muse_glimmer_30b_mlx_v1_track.json` sets `official_scoring_enabled`
> to `true`. The timed prompt pool pins three goldens by digest, and
> `live_goldens` names them: `pool-9`, `pool-10` and `pool-11`. The
> `live_golden_speculative` block is empty, so a ranked run can score a serial
> declaration only. README.md, "Current status", lists what else a ranked run
> needs.

> **NOTE — the engine fork is a vendored tree, edited here.**
> `Vendor/mlx-swift-lm` is the local merge of `Layr-Labs/mlx-swift-lm`
> `feat/muse-glimmer-speed` `62cfabd5` and `feat/muse-glimmer-rowexact-m5-d5`
> `44378936`; engine-local changes are this repository's commits. Nothing goes
> upstream. `Vendor/mlx-swift-lm/docs/muse-glimmer.md` holds the port detail.

## Local runs are directional

The local test runs one stream and so does the ranked run, so the two shapes
agree. Your machine is still not the ranked box: its memory budget, its thermal
behavior and its kernel selection all differ. Treat a local score as a smoke
signal, not as a prediction. The ranked M5 run is the authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/muse_glimmer_30b_mlx_v1_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`, verified against the dist `benchd.manifest.json`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
