# The task — Muse Glimmer 30B MLX

Make the Muse Glimmer 30B model run faster on Apple Silicon.

The ranked track is `muse-glimmer-30b-mlx-v1`. Read [README.md](README.md) for the
setup steps and the repository structure. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind the rules, and [`docs/scoring-contract.md`](docs/scoring-contract.md)
for the scoring rules with their sources.

## What you optimize

You optimize the engine. The engine is the MLX runner, the offline transform,
and the vendored MLX Metal kernels that the forward pass dispatches. You also
optimize the speculative-decode arm.

> **NOTE — the target is pinned.**
> The target is `nvidia/Muse-Glimmer-30B-NVFP4` at
> `47818374517751c48c55cde2621594926b1888b6`, pinned file by file in
> `fixtures/reference_bonsai2_9b_2bit.sha256`. That file name is inherited
> from the source track; every record in it is the Muse Glimmer checkpoint.
> `./setup.sh` still defaults its download repository and revision to the
> sentinel `BONSAI-MLX-PENDING-ORGANIZER`, so it refuses by name unless
> `MLXFAST_REFERENCE_MODEL_REPO` and `MLXFAST_REFERENCE_REVISION` name the pin.

The geometry is read off `fixtures/muse_glimmer_30b_nvfp4_config.json`, a byte
copy of the pinned snapshot's own `config.json`: 52 text layers, hidden size
6656, dense MLP at 19968, vocabulary 202048, 32 query heads over 2 KV heads at
head_dim 128, context 131072, 39 sliding-attention layers at window 2048 and 13
full-attention NoPE layers at every index `% 4 == 3`. The weights are mixed
precision modelopt NVFP4.

The checkpoint also carries a 50-layer vision tower. This track serves text
only, and the model never loads that namespace.

The track has ONE declarable speculative decoder, a DFlash drafter. It proposes
tokens, and the target model decides every emitted token.

| Decoder | Source | Declared in | Depths |
|---|---|---|---|
| DFlash drafter | Produced from `meta-models/Muse-Glimmer-30B-assistant` @ `c676cce91658f9568aaf264a416211e2782735b7` by the pinned writer; the artifact identity is in the fixture's `mtp_head.artifact` | `mtp-head.manifest.json` | 1 to 7 |

The fixture block keeps the name `mtp_head` and the declaration file keeps the
name `mtp-head.manifest.json`, because that is the slot the trusted harness
reads. The decoder declared there is `dflash`.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 97 entries in five
groups.

| Group | Paths |
|---|---|
| The decoder declaration | `mtp-head.manifest.json` (the declaration file only) |
| The offline transform | `Sources/MLXFastTransform/` |
| The model files | The Muse Glimmer text model and DFlash drafter files under `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/MuseGlimmer/`, the inherited Qwen 3.5 and Prism Hadamard model files, and `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift` |
| The batching engine and cache layer | `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` and 12 `MLXLMCommon` cache, attention, RoPE and config files, plus `MLXLMServer/Runtime/ToolStreamHandler.swift` |
| The vendored kernels | The 68 MLX Metal files the forward pass dispatches |

The engine fork `Vendor/mlx-swift-lm` is a vendored tree of plain files. Its
model files, the batching engine and the cache layer are editable. The runner,
`bench-worker` and the server in the same fork stay trusted.

The rule behind the list is simple. Code that **proposes** tokens or computes
the forward pass is editable. Code that **verifies**, **measures**, or
**ledgers** stays trusted.

> **WARNING — the target quantization is frozen.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. An editable transform does not license the change.

> **NOTE — the stream count is locked. The depth is not.**
> The ranked run scores one stream. You may not tune that.
>
> The draft depth is declared in `mtp-head.manifest.json`, which is editable:
>
> ```json
> "spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> The shipped declaration switches the decoder off, so the stock tree runs
> serial (depth 0). A depth outside `mtp_head.permitted_draft_depths` (1 to 7)
> is refused before the engine starts.
>
> A speculative leg needs a per-depth tape in the fixture's
> `live_golden_speculative`. That block is empty today, so the ranked measure
> script refuses a speculative declaration before it measures.

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
runs `./setup-mtp-head.sh` for the decoder slot. It refuses by name while its
reference repository or revision is the pending sentinel (see the NOTE above).

```bash
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
```

This command builds the Swift binaries and downloads nothing.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/bonsai2-9b-target \
  --output weights
```

This command writes the `weights/` tree that the engine loads. The reference
directory name is inherited from the source track.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test against the shipped local-iterate capture,
with no speculation.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the same local test with the speculative leg at depth 3. The
flag also exports the decoder directory to the worker in
`BENCH_WORKER_DRAFTER`.

> **WARNING — the local test runs serial unless you give it a depth.**
> `./benchmark.sh` does not read `mtp-head.manifest.json`. Without
> `--mtp-depth`, the local test runs depth 0, also when your tree declares a
> depth. To test the declared depth, add `--mtp-depth N`, where N is the value
> that `tools/spec-declaration.sh draft-len` prints.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = control_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1. The decode
gain is measured over the decode window only. The seed prefill is not in it.

The ranked run measures pairs on the same box, in the same job. A pair is a
serial-control leg on the organizer's reference tree, then your candidate leg,
over one live prompt. Each leg times one stream over a 512-token seed and a
128-step decode window. It runs 3 pairs, and pair k measures live prompt
(k - 1) mod 3. The score is the mean of the per-pair composites. Every pair
must clear the 0.95 floor on prefill and on decode. A pair whose decode gain is
above the fixture's `decode_speedup_ceiling` of 5.0 refuses the run and seals
`score: null`. The KV backend is pinned `paged`; the model refuses a
contiguous cache.

The benchmarker applies two token checks.

1. **The timed window is exact.** The prefill token, the seed token and each
   of the 128 committed decode tokens must equal the tape. One different token
   fails the run.
2. **The untimed correctness gate has a 10% budget.** Each teacher-forced case
   of 64 steps passes when at most 10% of its steps are not accepted. A
   top-logit tie is accepted.

> **WARNING — the timed window demands an exact match.**
> A candidate whose timed stream leaves the tape at any token fails the run.

## The current state

> **NOTE — the track is armed.**
> `fixtures/muse_glimmer_30b_mlx_v1_track.json` sets `official_scoring_enabled`
> to `true`. `timed_prompt_pool` pins three goldens by `sha256` and `bytes`,
> and `live_goldens` names them: `pool-9`, `pool-10` and `pool-11`, each at a
> 512-token seed. `hidden_correctness_golden` is pinned by digest.

> **NOTE — the engine fork is a vendored tree, edited here.**
> `Vendor/mlx-swift-lm` is a vendored copy of `Layr-Labs/mlx-swift-lm` with
> this repository's commits on top. Nothing goes upstream.

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
