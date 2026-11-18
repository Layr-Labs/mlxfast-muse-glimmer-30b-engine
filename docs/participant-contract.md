# Bonsai 2 9B — participant contract

This document states the terms that bind a submission to track
`bonsai2-9b-mlx-v1`.

`benchmark.json` is the Yukon track manifest.
`fixtures/bonsai2_9b_mlx_v1_track.json` is the track contract fixture. Both files
carry pure configuration: values, paths, commands, and pins. They carry no
prose. This document explains those files. It never overrides them.

## 1. Order of authority

Apply these in order. The higher entry wins.

1. The ranked run on the official runner. It is the authority on any score.
2. `fixtures/bonsai2_9b_mlx_v1_track.json` and `benchmark.json`.
3. This document.
4. `README.md` and `TASK.md`.

If either configuration file disagrees with this document on a plain value, the
configuration file wins. If either disagrees with the benchmarker about
measurement, the benchmarker wins.

The benchmarker is a prebuilt `benchd` binary resolved from the bench
repository's `main` dist channel. The channel publishes
`benchd.manifest.json` (`{branch, source_commit, sha256, bytes}`) beside the
binary; `./tools/fetch-benchd.sh` verifies the binary against that manifest,
installs both into `benchd-bin/`, and logs the resolved identity. The harness is
trusted-side: a submission cannot change what measures it. This repository
carries no submodule and no sha pin.

Both vendored trees, `Vendor/mlx-swift` and `Vendor/mlx-swift-lm`, are plain
files in this repository. Neither is a submodule. A fresh `git clone` carries
everything a build needs.

## 2. What the track measures

The track measures Bonsai 2 9B MLX inference speed.

You optimize the MLX runner, the offline transform, and the vendored MLX Metal
kernel families that the forward pass dispatches. You also optimize the
speculative-decode arm.

THE TARGET IS PENDING. The target model is the Bonsai 2 9B MLX pack of
`Qwen/Qwen3.5-9B`. prism-ml has not published it. The contract's
`target.upstream_model_id` and `target.upstream_revision` carry the exact
sentinel `BONSAI-MLX-PENDING-ORGANIZER`, and so does every other pin that only
the published pack can state. `fixtures/reference_bonsai2_9b_2bit.sha256` holds
no record. `./setup.sh` refuses the pending reference by name.

THE GEOMETRY IS THE BACKBONE'S. `target.geometry_source` names
`Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, whose
`config.json` is checked in as `fixtures/qwen35_9b_base_config.json`. The
backbone is gated-delta-net recurrence on three layers out of every four, and
full attention on the fourth.

THE PACKING IS ASSUMED. `target.packing_reference` names the Ternary Bonsai 2
27B pack, whose `config.json` is checked in as `fixtures/bonsai2_27b_config.json`.
The 9B pack is expected to use the same packing. The transform validator
refuses a pack that differs, by name.

| Property | Value |
|---|---|
| Architecture | The model type is `prism_hadamard_qwen35`, over a `qwen3_5` backbone. Schema 2. |
| Hidden layers | 32. Every fourth layer is full attention. |
| Layer counts | 8 full attention, 24 linear attention |
| Attention layers | At indices 3, 7, 11, ... 31 |
| Attention | 16 query heads, 4 KV heads, head dimension 256, no attention bias, output gate on |
| Rotary | Partial (`partial_rotary_factor` 0.25), `rope_theta` 10000000, mRoPE sections 11/11/10 |
| Linear attention | 16 key heads and 32 value heads of dimension 128, convolution kernel 4, FP32 state, grouped GDN layout |
| MLP | Dense: a gate and an up projection of width 12288 into a down projection. No experts. |
| Hidden size | 4096 |
| Vocabulary | 248320, embeddings untied |
| Context | `max_position_embeddings` 262144 |
| Tokens | eos 248044 |
| Quantization | Assumed: affine, group size 128, 2 bits, no mixed precision. FP16 scales and biases. |
| Transform | Assumed: a signed normalized block Walsh-Hadamard transform of width 1024, folded into all 202 packed modules. `hadamard.json` is the manifest. |
| Language-model tensors | 1033 expected `language_model.` tensors: 9 top-level and 32 per layer |
| Pending | The shard count, the raw tensor count and bytes, the vision tower tensor count, the bos token, and whether the pack carries an `mtp.*` block |

Only the 8 full-attention layers carry a key-value cache. The 24
linear-attention layers carry a constant-size recurrent state instead.

EVERY PACKED PROJECTION CARRIES THE TRANSFORM. The forward transform applies to
the input of each packed linear and the inverse transform applies to the output
of the embedding. Both run in FP32 and restore the incoming dtype. Call the
module. A path that reads a packed module's raw `.weight` skips the transform
and returns wrong numbers while it still compiles and still runs.

THE BACKBONE DECLARES A VISION TOWER. This track serves text only. The
transform drops that namespace, the runner declares `multimodal: false`, and
the model filters the same prefix at load.

The speculative-decode arm has ONE declarable decoder, an MTP head: one
`qwen3_5_mtp` full-attention layer for the Qwen 3.5 9B geometry. Its pin is
pending. The head has no embedding table or `lm_head` of its own. It rides the
target's, which is what `use_dedicated_embeddings: false` means here.

No DFlash 2 drafter exists for the Qwen 3.5 9B geometry. The contract
declares no `dflash_drafter` block, and `allowed_modes` is `serial` and `mtp`.

The arm is single-stream. The runner's MTP assistant declares
`maximumSpeculativeBatch == 1`, so the runner declares single-stream regimes
only.

`kv_backend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

## 3. What you may edit

`benchmark.json` `editablePaths` is the authority. It lists 97 entries.

The rule behind the list: anything that only **proposes** tokens or computes
the forward pass is editable. Anything that **verifies**, **measures**, or
**ledgers** stays trusted.

The editable surface has five groups.

1. **The decoder declaration.** `mtp-head.manifest.json`. The declaration
   file only. It names the decoder and the draft depth. It carries no weights,
   and no head weights directory exists. See section 4.
2. **The offline transform.** `Sources/MLXFastTransform/`.
3. **The model files.** Under `Vendor/mlx-swift-lm/`, unless the line says
   otherwise:

   - `Libraries/MLXLMCommon/PrismHadamardCheckpoint.swift`
   - `Libraries/MLXLLM/Models/PrismHadamardQwen35.swift`
   - `Libraries/MLXLLM/Models/Qwen35.swift`
   - `Libraries/MLXLLM/Models/Qwen35+CompleteCheckpoint.swift`
   - `Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift`
   - `Libraries/MLXLLM/Models/Qwen35A3BTargetVerify.swift`
   - `Libraries/MLXLLM/Models/Qwen35MTP.swift`
   - `Libraries/MLXLLM/Models/Qwen35MTP+PrefixCheckpoint.swift`
   - `Libraries/MLXLLM/Models/Qwen35MTPTopTwo.swift`
   - `Libraries/MLXLLM/Models/Qwen35MoE.swift`
   - `Libraries/MLXLLM/Models/DFlash2Draft.swift`
   - `Libraries/MLXLLM/Models/Qwen35DFlash2Assistant.swift`
   - `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift` (the OTHER vendored tree)

   `Qwen35MTP.swift` holds the MTP head and its adapter in one file. It is
   editable because it PROPOSES tokens, and the pinned target decides every
   emitted token. `DFlash2Draft.swift` and `Qwen35DFlash2Assistant.swift` are
   inherited from the 27B track. This track declares no DFlash 2 decoder, so
   no run reaches them.

4. **The batching engine and the cache layer.** The whole
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` directory
   (the CBv2 engine, the MTP round driver and depth controller, the paged KV
   pool and its Metal kernels), and these files under
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/`: `AttentionUtils.swift`,
   `BaseConfiguration.swift`, `CompilableKVCache.swift`,
   `CompilableRotatingKVCache.swift`, `DynamicSlice.swift`, `Evaluate.swift`,
   `JSONDecodingTypes.swift`, `KVCache.swift`, `LanguageModel.swift`,
   `RoPEApplication.swift`, `RoPEUtils.swift`, `SwitchLayers.swift`; plus
   `Vendor/mlx-swift-lm/Libraries/MLXLMServer/Runtime/ToolStreamHandler.swift`.
   This is the same shape the Gemma 4 track opened (David ruling 2026-09-16).
5. **The vendored MLX Metal kernels.** The 68 files the forward pass
   dispatches: the quantized matmul, the gather-GEMM, SDPA
   and steel attention, RoPE, RMSNorm, softmax, sort, reduce, copy,
   elementwise, `arg_reduce`, and gather indexing.

THE ENGINE FORK IS A VENDORED TREE. `Vendor/mlx-swift-lm` is a copy of
`Layr-Labs/mlx-swift-lm`, as plain files, with this repository's own commits on
top. It is a copy and not a submodule for one reason: an editable path names
bytes in this tree, and a gitlink names a commit. The upstream base the tree
was cut from is in this contract's `mlx_swift_lm_revision`.

THE MODEL FILES, THE BATCHING ENGINE AND THE CACHE LAYER ARE EDITABLE. The
rest of the fork is trusted: the runner and its registry, `bench-worker`, the
server, and every `Libraries/MLXLMCommon/` file not named above. A submission
that changes any of them is refused by the surface gate.

### 3.1 Optional paths

`optionalEditablePaths` lists `mtp-head.manifest.json`.

A submission archive has REPLACE semantics over `editablePaths`. An absent head
declaration means the pinned head. The overlay therefore skips a missing
optional path instead of failing closed.
`.github/scripts/overlay-editable-paths.sh` reads this list from the trusted
contract, never from the submission.

### 3.2 The byte budget

`editableSurfaceByteBudget` caps the editable surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 7949663 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is **absent** since 2026-08-26. The exemption existed for one
reason: to let head weights ride in a submission outside the source budget. A
submission carries no head weights any more, so there is nothing to exempt.

The two exempt caps stay declared. They cannot bind while `exemptPaths` is
absent. They stay because both enforcers carry the same two numbers as
compiled-in fallbacks, and this manifest is what holds those constants to a
reviewed value. `tools/lint-benchmark-manifest.py` check 3b enforces that
equality.

No head weight file is staged in a submission, so this budget never meets one.
The head is a pinned export outside the editable surface. What the runner
LOADS is bounded instead by the 2 GiB declaration cap in section 4.

### 3.3 What you may not edit

You may not edit anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing and telemetry code. `fixtures/` is outside
the editable surface. The scoring step reads the contract from the trusted
checkout for that reason.

### 3.4 The target quantization is frozen

The target model's quantization is frozen as shipped.

A submission must not re-quantize any target weight. It must not re-represent a
target weight. It must not change the numerical format of a target weight. This
holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model. It does not
optimize the accepted one.

The MTP head is a narrow exception, and the exception is re-quantization only.

You may re-quantize the MTP head. You may **not** replace it. You may **not**
upload head weights of your own. Custom head weights are not accepted on this
track.

This is the 2026-08-26 ruling. It replaces the earlier bring-your-own-head
design, under which a participant could declare and ship a head of their own
choosing. That design is retired.

The head is the organizer's pinned weights. `fixtures/bonsai2_9b_mlx_v1_track.json`
names its repository and revision, and `fixtures/reference_bonsai2_9b_mtp_head.sha256`
carries the per-file digests that `./setup-mtp-head.sh` verifies every
downloaded byte against. Both pins are pending. Both files live in
`fixtures/`, which is outside the editable surface.

Three things enforce this, and section 4 states each one:

1. No head weights directory exists and no submission path can hold head
   weights. A submission that carries a weight file is refused.
2. The head declaration accepts `"source": "pinned"` only. `"remote"` and
   `"in_branch"` are refused by name.
3. A re-quantization runs on load, in memory, on the benchmark machine.
   No re-quantized file is made, so there is no artifact to travel in a
   submission. Section 4.4 states the mechanism.

The loader reads a `quantization` block in the shape an MLX conversion writes.
That block selects which modules load quantized and at what geometry. The
accepted parameters are `group_size` (positive, at most 65536), `bits` between
2 and 8, and optional per-layer overrides (at most 8192 entries). A value
outside those bounds is refused by name.

The MTP head loader does not check a declare-versus-carry mismatch. An absent
declaration skips quantization, and packed weights then fail later inside the
weight bind with a shape error. A declaration with no packed tensor quantizes
nothing, silently. That limit is stated here rather than promised away.

The reason for the whole exception is the propose-and-decide split. The head
only proposes tokens. The pinned target model decides every emitted token.

## 4. The speculative decoder

The track carries ONE declarable speculative decoder and ONE declaration file.
The decoder is the MTP head, the organizer's weights, a separate pinned export.

`mtp-head.manifest.json` `spec.decoder` names the decoder a submission arms.
This track declares `"mtp"` only. The key is optional and reads as `"mtp"`
when it is absent. `"dflash"` is refused by name, because no DFlash 2 drafter
exists for the Qwen 3.5 9B geometry.

| Item | The MTP head |
|---|---|
| Declaration | `mtp-head.manifest.json` |
| Where the weights are | A separate pinned export, `reference_weights/bonsai2-9b-mtp-head` |
| Organizer pin | Pending (`BONSAI-MLX-PENDING-ORGANIZER`), `fixtures/reference_bonsai2_9b_mtp_head.sha256` |
| Shape | 1 full-attention `qwen3_5_mtp` hidden layer; the tensor count and the quantization are pending |
| Declaration cap | 2 GiB |
| Permitted draft depths | 1 to 7 |
| Staged by | `./setup-mtp-head.sh` |

The declaration file is editable. The pinned export is not.

The shipped declaration switches the decoder off
(`"enabled": false, "num_speculative_tokens": 0`), so the stock tree runs
serial.

No submission carries a head weight file. `./setup.sh` stages the head after
the target.

### 4.1 What you may declare

`"source": "pinned"` is the only accepted source. On this track it means the
separate head export the organizer pins. A declaration may also state
`max_bytes` (it may lower the 2 GiB track cap and may not raise it), a `bytes`
count, and an optional `sha256`. It carries no `arm` key: `spec.decoder` is the
selection, and it sits beside the depth it goes with.

`"source": "remote"` is refused by name. `"source": "in_branch"` is refused by
name. Both were accepted before the 2026-08-26 ruling and both meant "load
weights the participant chose". The refusal names the retired source and names
`pinned` as what replaced it.

An absent declaration selects the pinned head. A declaration that is present
but broken is a refusal. The runner never falls back silently.

### 4.2 What you may not do

You may not ship head weights. No path in the editable surface can hold them,
so a submission that carries a weight file is refused before any measurement.
`.github/scripts/enforce-modifiable-surface.sh` names the file and refuses.
`.github/scripts/overlay-editable-paths.sh` never copies it. The benchmarker's
own write-divergence gate refuses any content that differs from the trusted
baseline outside the editable surface.

You may not edit the checkpoint's head tensors. The checkpoint is not an
editable path, so any change to it is outside the surface.

### 4.3 What the size cap does and does not do

The 2 GiB declaration cap (`max_bytes` = 2147483648) bounds what the runner
loads.

The size cap is the only gate on the declaration. A declared `sha256` is
optional, and the runner does not verify it against the head bytes. It treats a
wrong digest and an absent digest alike. That is stated here plainly because it
is a real limit, not a detail: **nothing in this repository binds the loaded
head bytes to the organizer's pinned digests at run time.** The harness
computes a head digest that it reports and never compares.

What does bind at run time is relative, not absolute: the benchmarker compares
the candidate workspace against the trusted baseline workspace and refuses any
divergence outside the editable surface. A correctly provisioned baseline is
therefore load-bearing for the whole property.

The head bytes themselves are bound one level up. `./setup-mtp-head.sh`
verifies every downloaded byte against the per-file digests in
`fixtures/reference_bonsai2_9b_mtp_head.sha256`. Both legs load the head out of
that one verified copy. That pin is pending.

### 4.4 How a re-quantization reaches the box

A re-quantization happens ON LOAD, in memory. Nothing on disk changes.

You do not make a re-quantized checkpoint. Your code quantizes the head's
parameters in memory, in the same pass that binds them. The staged bytes are
only read.

#### What to edit

The head loader calls `quantize(model:)` while it binds the checkpoint. That
call is the seam, and the file that holds it is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.

Change the geometry that call selects. The default reads the checkpoint's own
per-layer quantization for the matching `mtp.` key. Your code may select a
different geometry instead.

Section 3.4 states the bounds the loader accepts: `group_size` positive and at
most 65536, `bits` between 2 and 8, and at most 8192 per-layer overrides. A
value outside those bounds is refused by name. The loader does not check a
declare-versus-carry mismatch; section 3.4 states that limit.

#### Why nothing is written

Two properties follow from the in-memory rule, and both are why this mechanism
is the safe one.

1. The benchmarker compares the candidate workspace against the trusted
   baseline workspace and refuses any change outside the editable surface. That
   comparison reads the disk. A re-quantization on load is not a disk
   operation, so there is nothing for the gate to see.
2. The ranked worker runs under a sandbox profile that denies file writes. Code
   that tried to rewrite a staged head would fail there.

Do not rewrite the head tensors on disk. Do not rewrite them from `setup.sh` or
from the `mlxfast-swift transform` command. Each of those runs before the
workspace comparison, and the benchmarker refuses the change.

#### What changes in the record

The worker reports the digest of the head it loaded. That digest is the digest
of the ORGANIZER's checkpoint bytes, before and after a re-quantization,
because the bytes do not change. The geometry you selected is not visible in
that digest.

#### What this does not permit

The exception is for the HEAD's OWN weights only. The target model's
quantization stays frozen, as section 3.4 states.

**The head is a separate module tree, and the rule is drawn by module path.**
David ruling 2026-08-27, relayed by orchestrator: "Exempt mtp.* from the
freeze." The head binds to the loaded target, so the loaded-target check has to
say which side of the line each module is on. It says it this way:

| Module path | Treatment |
|---|---|
| The head's own modules | EXEMPT. Re-quantize them on load. |
| A head module naming `embed_tokens` or `lm_head` | REFUSED BY NAME. |
| Everything else, `model.embed_tokens` and `lm_head` included | FROZEN. |

**The shared tensors are the middle row, and they are shared for a real
reason.** The head owns no embedding table and no output projection. It READS
the target's embedding table for its next-token vectors and the target's output
projection for its logits. On this pack both are PACKED HADAMARD modules, so
they are read through the module and the transform applies. Those two tensors
decide the TARGET's tokens. Coarsening one of them is a target re-quantization
whatever path it is spelled under, so a quantized module inside the head
subtree that names either one is refused, and the refusal says which shared
tensor it reached.

The target is verified TWICE, and both checks read the loaded model, not the
declaration:

1. At worker startup, immediately after the target is loaded.
2. Again at the top of each window that gets measured, immediately before the
   measured work starts.

The second check exists because the first one alone verifies a model that code
can still change afterwards. Both refuse by name, and a refusal stops the worker
before any measurement.

The head only **proposes** tokens. The organizer-pinned target model decides
every emitted token. Both legs load the pinned head; only the candidate leg
drafts with it.

## 5. Scoring

### 5.1 The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1.

**WHAT DECODE MEANS (David 2026-10-07).** Decode seconds per token is the
decode window: the time of `free_decode_run(N)` divided by N. The seed prefill
of the prompt is not part of it. Prefill is its own phase, timed on its own.
`decode_gain` is the serial leg's decode window per token divided by the
candidate leg's. The calibration band and the floors use the same figure. The
seed prefill is sealed report-only under `*_seed_prefill_window_seconds_per_token`.

### 5.1.0 The pair is measured on the box, and no file stores it

**DAVID RULING 2026-09-08: EACH RANKED MACHINE HAS ITS OWN BASELINE.** A ranked
run measures pairs, on the SAME box, in the SAME job. The fixture lists the live
prompts in `live_goldens`. Pair k measures live prompt (k - 1) mod N, where N is
the number of live prompts, so `official_pairs` must be a multiple of N. The
score is the mean of the per-pair composites. A pair measures TWO legs over one
prompt:

1. The **serial-control leg**. It runs on the organizer's reference tree, which
   `MLXFAST_BASELINE_WORKSPACE` names. That tree is a build of this repository
   at the commit the fixture names in `baseline_reference_commit`. The leg uses
   no speculation. The measure script passes the serial tape to benchd as
   `--control-golden`, so this leg is verified against the serial tape and the
   candidate leg against the tape recorded at its declared depth.
2. The **candidate leg**. It runs on the submission tree, at its declared draft
   depth.

```text
score = (ref_prefill_spt / cand_prefill_spt) ^ 0.25
      * (ref_decode_spt  / cand_decode_spt ) ^ 0.75
```

The floors do not change.

**NO STORED PAIR EXISTS ANYWHERE.** Not in the scoring constants. Not in the
fixture. Not in a golden. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is REFUSED on the ranked path.
`tools/lint-benchmark-manifest.py` check 5b keeps both fields out of this
repository.

The organizer stages the reference tree on each ranked box.
`tools/stage-baseline-workspace.sh` builds it there from a staged mirror or
bundle. The ranked job verifies the tree. It never fetches or builds it, because
the job holds no credential.

The reference tree carries its own engine, its own Metal library and its own
transformed weights. The candidate cannot move the control leg.

`baseline_reference_commit` names the repository's seed commit. It is
re-pinned to the final engine commit before the first scored window. A
trusted-side code change that the control leg must carry re-pins it, and every
box re-stages its reference tree at the new commit before its next ranked run.

### 5.1.0.1 The per-box calibration is a health band

Each ranked box records what its own serial-control leg costs.
`MLXFAST_BASELINE_CALIBRATION` names the file.

**THE FILE IS A HEALTH BAND. IT IS NEVER THE DENOMINATOR.** The benchmarker
compares the measured control leg against the band. It stops the run by name
when the leg falls outside the band, and it seals no score. A stale calibration
file can stop a run. It can never move a score.

An operator writes the file on the box:

```bash
tools/calibrate-box.sh "<runner name>" /path/to/baseline-calibration.json
```

The command takes the box GPU lock. It then runs the serial-control leg three
times under the full official methodology: the cool gate before each pass, one
resident worker for each pass, and the same live goldens the ranked run scores
over. The file it writes holds one entry for each live golden and carries
values only:

```json
{
  "version": 3,
  "track_id": "bonsai2-9b-mlx-v1",
  "box": "<the runner name>",
  "reference_commit": "<the fixture's baseline_reference_commit>",
  "captured_at": "<ISO 8601 UTC>",
  "benchd_source_commit": "<40 hex>",
  "prompts": [
    {
      "prompt": "<one name from the fixture's live_goldens>",
      "passes": 3,
      "prefill_seconds_per_token_mean": 0.0,
      "decode_seconds_per_token_mean": 0.0,
      "prefill_cv": 0.0,
      "decode_cv": 0.0,
      "prefill_band_low": 0.95,
      "prefill_band_high": 1.05,
      "decode_band_low": 0.95,
      "decode_band_high": 1.03
    }
  ]
}
```

`decode_seconds_per_token_mean` is the mean decode window per token: decode-run
time / N, with no seed prefill in it. The means and the coefficients of
variation above are placeholders. Only a
measured box can supply them, and this track has no calibrated box yet.

The calibrator writes no file when the coefficient of variation is more than
1 percent on either axis. A box that cannot repeat itself has no band worth
recording.

`tools/ranked-box-preflight.sh` refuses the run before any measurement when:

- either variable is absent;
- the workspace is not a git checkout at `baseline_reference_commit`;
- the workspace has no staged worker, no `mlx.metallib`, no fingerprint sidecar,
  or no transformed weights of its own;
- the calibration file does not parse, or its `version` is not 3. Versions 1
  and 2 recorded decode as the whole window (seed prefill plus decode, over N),
  and the refusal says to recalibrate;
- its `track_id` is not this track;
- its `box` is not this runner's `RUNNER_NAME`;
- its `reference_commit` is not the fixture's `baseline_reference_commit`;
- its `captured_at` is not later than the reference commit's date;
- any numeric value is not finite and positive;
- a band does not straddle 1 (`low < 1 < high`).

Each refusal names the failing thing.

**THE SCORED SHAPE IS SINGLE-STREAM.** The fixture pins `scored_batch_size` 1.
Each leg runs one stream, and the engine agrees: the runner declares
single-stream regimes only. A scored pair times one prompt from
`live_goldens`, on both legs of that pair.

Where a run times a pool of prompts rather than one, `aggregate` is the
**per-prompt sum**. Run each pool prompt in its own single-stream window and
add the elapsed times together. Do this for prefill and for decode separately.
Do it on the baseline leg and on the candidate leg, over the same prompts.

One ranked run measures the pairs the fixture's `official_pairs` declares. Each
pair has its own composite, from its own control leg. The fixture declares
`official_pair_combine: "mean"`, so the score is the arithmetic mean of the
per-pair composites. The floors and the bands gate every pair on its own, and
the first pair that fails stops the run. The two gains and the seconds per
token in the artifact are each the mean of the same per-pair figure. Every pair
stays in `metrics.paired_legs` as measured, and the artifact names the rule in
`metrics.official_pair_combine`.

The fixture carries the formula as `scoring_semantics.formula`, and
`benchmark.json` carries the same string as `scoring.formula`.

`scoring.mode` is `bonsai2-native-mtp-paired-composite`. It names the
measurement methodology, not the formula.

### 5.1.1 Where the prefill window is, and why you cannot move work out of it

**THE VERBS DO NOT CHANGE.** The single-stream pair is
`free_decode_begin` followed by `free_decode_run`. There is no new message and
no new field. The benchmarker splits its OWN parent clock at the verb boundary:

| Window | From | To |
|---|---|---|
| prefill | `free_decode_begin` sent | the validated `seed_token` comes back |
| decode | there | `free_decode_run(N)` returns |

Decode seconds per token is the decode window over N. The seed prefill window is
sealed report-only and is not part of decode. Both legs are bracketed
identically.

**WHAT THE ENGINE OWES, and it is not optional:**

1. `free_decode_begin` runs the FULL seed prefill -- the golden's
   `decode_seed_tokens`, all 512 of them, with the requested spec resolved --
   and replies only after that work has COMPLETED. The reply is the seed token
   (the greedy argmax after the whole seed) and the echoed `effective_spec`.
2. `free_decode_run` does NOT prefill and does not re-run any part of the seed.
   It decodes from the state `begin` left.
3. Nothing prefills before `free_decode_begin` arrives.
4. The hello advertises `free_run_decode` only.
5. Units are unchanged.

**MOVING WORK ACROSS THE BOUNDARY IS NOT AN OPTIMISATION. IT IS A SCORING
DEFECT.** Deferred seed work lands in the decode window and makes decode slower.
Decode work done early, inside `free_decode_begin`, leaves the decode window
and is not scored, which breaks obligations 1 and 2 above.

This is engine-side and unobservable on the wire. Nothing on the wire can catch
it, so it is a rule of the contract rather than a gate, and the static review
reads a change against it.

**NO SCORED RUN IS POSSIBLE ON THIS TRACK TODAY**, and section 11 states why.
The short form is the pending target, the arm state and the goldens: the pack
is not published, `official_scoring_enabled` is `false` and no organizer tape
exists.

### 5.2 The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 512 |
| Checked decode steps | 128 |
| Golden shape | 512 `prompt_tokens` and 129 `expected_tokens` |
| Streams per window | 1 |
| Timed prompts per leg | 1 (one name from the fixture's `live_goldens`) |
| Pairs per ranked job | the fixture's `official_pairs` |
| Legs per pair | 2 (serial control, then candidate) |

The score is the mean of the per-pair composites.

The two legs of a pair run one after the other, in one job. Each leg loads the
weights once, and each leg gets its own worker residency. The unmeasured warm-up prefill
pass stays at 1 pass on MLX, and it applies to both legs in the same way.

`MLXFastConstants.correctnessPromptTokens`, `benchmarkPrefillPromptTokens`, and
`benchmarkDecodeSeedTokens` all equal 512. `benchmarkDecodeSteps` is 128.

Every timed leg runs on a cool, quiescent box. The benchmarker holds each timed
phase of each leg behind two fixed gates, in this order: the quiescence gate (the
1-minute load average is below 2.0 and GPU utilization is below 10 %, polled
every 15 s, refused after 900 s) and the cool-down gate (GPU at or below 40 C,
refused after 900 s). Calibration passes run behind the same two gates. The job
refuses to measure at all when the box has no GPU temperature reader, or when
that reader returns a frozen or implausible value. Only pairs accepted under
both gates feed the composite.

### 5.3 The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 3 |
| `minPairsPerCohort` | 3 |
| `decodeSpeedupFloor` | 0.95 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `contiguous` |

The fixture's own floors are `decode_speedup_floor` and
`prefill_speedup_floor`, both 0.95. They gate the two gains. The manifest's
`decodeSpeedupFloor` states the same 0.95, and `pairsPerCohort` states the
fixture's `official_pairs`. The benchmarker reads the fixture; the manifest
block restates it, and `tools/lint-benchmark-manifest.py` keeps the two equal.

There is no sweep and no per-run choice of width. A width the benchmarker has
not certified has no series tag, and the benchmarker refuses that width rather
than run it.

The ranked run scores the mean of the per-pair composites over the
`official_pairs` pairs (section 5.1).

The candidate's slowness gate is the two floors. The control leg's health band
is the box calibration, and the fixture's decode band allows 3 percent upward
(`decode_band_up_tolerance` 0.03).

### 5.4 Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10%
budget**.

This track does not require token-for-token equality with the serial
trajectory. A speculative round verifies a whole draft chain in one target
forward, and a multi-row forward rounds differently from a single-row one, so
it can diverge from the serial forward at a near-tie argmax. The gate prices
that divergence against the 10% budget. The gate accepts similar output. It
does not certify lossless output.

### 5.5 Arming

`fixtures/bonsai2_9b_mlx_v1_track.json` sets `official_scoring_enabled` to
`false`. That flag is the SINGLE authority on this track's arm state, and it is
load-bearing. The pinned benchmarker reads it from the `--contract` fixture. It
refuses to seal an official scoring artifact while the flag is `false`. It also
refuses while the flag is absent, because it treats an absent flag as unarmed
rather than armed. No submission can publish an official score until that flag
flips.

The benchmarker, not the engine, produces the composite. It computes the
composite from benchd's own parent-clocked prefill and decode windows, summed
over the accepted pairs, at the certified exponent pair. No engine-reported
value feeds it, and it does not depend on per-stream instrumentation. Each
record seals exactly one of `composite` and `composite_absent_reason`. A
composite is absent only when the record accepted no pair, or when a window is
degenerate, and the reason names which.

What is missing is the arm state and the goldens, not this repository's score
path. `tools/bonsai2-9b-measure-and-score.sh` refuses with a non-zero
exit rather than emit a score. It does not substitute a diagnostic for the
ruled composite formula. Refuse, not degrade, is the standing posture for this
track. The `kv_backend` check and the byte-budget check use it too.

The timed prompt pool is EMPTY. `timed_prompt_pool` holds no entry,
`live_goldens` is an empty list, `live_golden_speculative` holds no per-depth
oracle, and `hidden_correctness_golden` carries the sentinel
`BONSAI-MLX-PENDING-ORGANIZER`. An empty pool is legal only while the track is
unarmed: `tools/lint-benchmark-manifest.py` requires at least one pinned pool
prompt, with no two entries alike, before `official_scoring_enabled` may be
`true`. `tools/ranked-box-preflight.sh` refuses a contract that still carries
any `PENDING-ORGANIZER` sentinel, the target pins included. The planned live
prompts are `pool-9`, `pool-10` and `pool-11` at a 512-token seed.

### 5.6 Which goldens you can hold

| Object | Where it lives | Can you have it? |
|---|---|---|
| The public captures | This repository, under `correctness_prompts/bonsai2-9b-mlx-v1/`. | **Yes.** They are the local goldens. See section 11.3. |
| `timed_prompt_pool[]` tapes | R2, at the `r2_path` keys the fixture pins. The ranked box stages them out of band into `MLXFAST_QWEN38_GOLDEN_DIR`. | **No.** They are organizer material and they are never in git. |
| `live_golden_speculative{}` per-depth oracles, one per live prompt for each depth | The same: R2 keys, staged on the box. | **No.** Same material, same handling. |
| `hidden_correctness_golden` | A live golden, pinned by digest only. It is one of the staged files. | **No.** It is the token-fidelity oracle and it stays on the box. |
| The reference tree (`MLXFAST_BASELINE_WORKSPACE`) | Built on the ranked box at `baseline_reference_commit`. | **No.** It is the serial-control leg's engine. Its commit is public: the fixture names it. |

`MLXFAST_QWEN38_GOLDEN_DIR` keeps its Qwen spelling because it is the fleet's
contract with the box's runner service, not this track's own name. The same
holds for `MLXFAST_QWEN38_R2_DOWNLOADER`, the override for the R2 request
signer.

`tools/fetch-goldens.sh` is the organizer-side, pin-verified fetcher for R2
objects. It reads the R2 base from the environment variable
`R2_BUCKET_ENDPOINT` only. That value is secret-tier and is absent from this
repository. The script verifies the byte count first, then the sha256, and
deletes the file on either mismatch. It refuses to fetch anything the contract
declares hidden, and that guard fails closed when it cannot read the contract.

The organizer stages the whole pinned set on a ranked box with the same tool.
`--all` reads the fixture, fetches every tape and every per-depth oracle, and
verifies each one against its `{sha256, bytes}` pin. It signs the requests with
the signer vendored at `tools/download-r2-object.sh`, so it needs R2 credentials
and refuses without them. A file that already matches its pin is left alone, so
the command is safe to re-run:

```bash
R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
  tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
tools/ranked-box-preflight.sh
```

The preflight then verifies the staged directory against the fixture again and
refuses an extra `*.json` in it.

## 6. Running the benchmark

`benchmarkCommand` runs `tools/bonsai2-9b-yukon-run.sh`. That script selects
one of two paths:

| Condition | Path |
|---|---|
| `GITHUB_ACTIONS` is `true`, or one or more of `MLXFAST_QWEN38_GOLDEN_DIR`, `MLXFAST_BASELINE_WORKSPACE` and `MLXFAST_BASELINE_CALIBRATION` is set | Ranked. The script replaces itself with `tools/bonsai2-9b-measure-and-score.sh`. |
| None of these | Local. The script transforms the weights when `weights/config.json` is absent. Then it runs `./benchmark.sh --local-submit` at the declared depth. The result has `"score": null`. |

The ranked workflow calls `tools/bonsai2-9b-measure-and-score.sh` directly.
Both scripts are trusted-side tooling. They are not editable paths, so a
submission cannot rewrite the measurement pipeline from inside its own archive.

The ranked path targets `benchd iterate --mode official`.

The wrapped invocation is:

```text
benchd iterate --mode official \
  --engine .build/release/bench-worker \
  --weights <transformed weights> \
  --golden $MLXFAST_QWEN38_GOLDEN_DIR/<live golden>.golden.json \
  --contract fixtures/bonsai2_9b_mlx_v1_track.json \
  --baseline-workspace $MLXFAST_BASELINE_WORKSPACE \
  --baseline-calibration $MLXFAST_BASELINE_CALIBRATION \
  [--box <this box>] [--mtp-depth N] \
  --score-path score.json
```

The script passes one `--golden` per live golden, in the order of
`live_goldens`, with one `--golden-sha256` and one `--golden-bytes` pin for
each. It passes the serial tape of each live golden the same way, as
`--control-golden` with its pins. The benchmarker matches the flags by
position. Pair k measures live golden (k - 1) mod N, where N is the number of
live goldens.

`--engine` and `--weights` are RELATIVE to the checkout root. The benchmarker
re-roots both under `--baseline-workspace` to find the serial-control leg's own
worker and its own transformed weights. The script refuses an engine or a
weights directory outside the checkout: such a path has no relative form, and
the control leg would then run the candidate's own binary or read the
candidate's own transform.

`--mtp-depth` carries the depth the tree declares. `tools/spec-declaration.sh
draft-len` derives it from `mtp-head.manifest.json`, and the script refuses
rather than run a serial leg against a speculative declaration when the
resolved benchd has no `--mtp-depth`.

`--box` is passed only on a hand run. On a runner the benchmarker reads
`RUNNER_NAME` itself and lets it win over the flag, so passing the flag there
would be argv that cannot matter. A hand run has no `RUNNER_NAME`, and the value
then comes from the calibration file's own `box`.

The measure script sets `RESIDENT_UP_LOG_DIR` to a new directory in the job's
temporary directory when it is unset. Both legs' residents keep their logs
there, because the ranked job account cannot write the reference workspace.

The measure script refuses by name when `MLXFAST_BASELINE_WORKSPACE` or
`MLXFAST_BASELINE_CALIBRATION` is absent on a real run. `--preflight-only`
requires neither: a participant runs it off the box.

benchd seals `score.json` itself, in the `{score, metrics}` shape the scorer
reads. This script writes no score and converts nothing.

**WHERE THE COMPOSITE LIVES ON THIS TRACK.** A single-stream run has no cohort
record, so benchd seals the composite ONE LEVEL UP: `results.json` carries
`composite` (`{composite_score, composite_speedup_floor,
composite_speedup_floor_met, decode_gain, prefill_gain}`) beside
`composite_scored_exponents`, and exactly one of `composite` /
`composite_absent_reason` is present. The published score is
`composite.composite_score`.

The decode-only median is NOT the score on this track. It is a different
formula -- decode only, no prefill component, no exponents -- and the emitter
refuses a single-stream record that seals no composite rather than publishing
the median in its place. The median is forwarded in `metrics` as a diagnostic.

Two drift tripwires run at that seam, because benchd reports a floor and an
exponent pair but wires neither to an exit code: the emitter REFUSES a run
whose `composite_speedup_floor_met` is false, and refuses a run whose sealed
`composite_scored_exponents` differ from `benchmark.json`
`scoring.scoredExponents`.

`preSubmitCommand` runs
`./tools/bonsai2-9b-measure-and-score.sh --preflight-only`.
That runs the arm gate and, for each live golden that is staged, the integrity
pin.
It exits without measuring, and it needs no reference tree.

The ranked pipeline is `.github/workflows/benchmark.yml`, which `benchmark.json`
`runner.workflow` names. It triggers on `workflow_dispatch` only. Its hosted
surface-check job gates its ranked job, which runs on the self-hosted labels
`[self-hosted, macOS, bonsai2-9b-mlx-v1]` — the third label is the track id.
The ranked job holds no credential. The organizer stages the timed-pool tapes
onto the box out of band, from R2; they are never in the checkout. Before
`./setup.sh` runs, `tools/ranked-box-preflight.sh` verifies each tape against
this track's `{sha256, bytes}` pins. One ranked run occupies the box at a time.
A second dispatch queues rather than cancelling the first.

**ONE RESIDENT WORKER PER LEG, AND THE BENCHMARKER BOOTS IT.** A ranked job has
two legs on two trees, so one resident cannot serve both: the reference leg's
weights are not the candidate's. The benchmarker knows where a leg begins and
ends, so for each leg it calls that leg's OWN copy of `tools/resident-up.sh`:

```text
tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N> --socket-out <file>
tools/resident-up.sh --stop --socket <path>
```

`--boot` loads that tree's own `.build/release/bench-worker` and that tree's own
`weights/`, waits for a healthy hello, writes the socket path as the first line
of the `--socket-out` file, and exits 0 with the resident still running. A
`<socket>.pid` sidecar and a `<socket>.ready` marker sit beside the socket, and
`--stop` ends the resident and removes all three. Every per-phase
`bench-worker runtime-worker` the benchmarker starts attaches to that leg's
socket instead of loading the checkpoint again.

`--spec` is authoritative over what the leg DOES. It is the label the leg
records, so a reference tree that declares a draft depth still boots a SERIAL
control leg when the benchmarker says serial, and the worker takes the decoder
and the depth from the wire per request.

`--spec` does NOT choose the drafter. The boot validates the leg's OWN
`mtp-head.manifest.json` through `tools/spec-declaration.sh` and loads that
tree's own MTP head. Nothing crosses between the two trees.

`tools/bonsai2-9b-measure-and-score.sh` boots no resident and exports no
socket. What it still owns is the window: it takes the box GPU lock
`/tmp/mtplx-gpu-exclusive.lock` and holds it for the whole measurement, because
a resident holds the whole checkpoint in unified memory whoever booted it and
the box needs exactly one loader. `tools/resident-up.sh` refuses to boot when nobody
holds that lock, so every per-leg boot happens inside that window.

**A `BENCH_WORKER_RESIDENT_SOCKET` IN THE ENVIRONMENT IS REFUSED.** It names one
already-loaded resident, so both legs would attach to it and the serial-control
leg would run on the candidate's weights. Both
`tools/ranked-box-preflight.sh` and the measure script refuse an inherited
socket by name.

None of this changes what is scored. It removes repeated loads, not measured
work.

The wrapper form of `tools/resident-up.sh` (`--weights <dir> -- <command>`) is
kept for LOCAL, UNSCORED use. The ranked path has no caller for it.

`setupCommand` is `./tools/fetch-benchd.sh && ./setup.sh`. It chains no head
stager of its own, because `./setup.sh` runs `./setup-mtp-head.sh`. The
checked-in `mtp-head.manifest.json` declares `"source": "pinned"`, and the
stager points the one verified downloader at the head's pin file. While the
target is pending, `./setup.sh` refuses by name.

## 7. The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | Pending (`BONSAI-MLX-PENDING-ORGANIZER`), the Bonsai 2 9B pack of `Qwen/Qwen3.5-9B` |
| Target manifest | `fixtures/reference_bonsai2_9b_2bit.sha256` (pending, no record) |
| MTP head | Pending (`BONSAI-MLX-PENDING-ORGANIZER`), `fixtures/reference_bonsai2_9b_mtp_head.sha256` (pending, no record) |
| Geometry source | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` `config.json`, as `fixtures/qwen35_9b_base_config.json` (3,126 bytes) |
| Packing reference | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `3f926b415992eaa2ae9dd7b573706494d6bbf787` `config.json`, as `fixtures/bonsai2_27b_config.json` (58,145 bytes) |
| Engine fork upstream base | tree `0391eacd84cb00ce4001d70763f2303fa2a01210`, the merge of `62cfabd5c67c6042edcfd812698bd34e4b5df7d7` and `443789360ddbf804d55d063ca9cda4ea2d3013a9` (a local merge branch, not pushed upstream; engine-local changes on top) |

The two reference manifests hold no record until the pack and the head are
published. When they are, `tools/new-track.sh` writes the target manifest from
the published tree, and the contract's target and head pins move with it.

`Sources/MLXFastCore/Constants.swift` mirrors the same repository and revision
pin, which is the sentinel today. The golden loader therefore refuses every
golden that names a checkpoint. `fixtures/qwen35_9b_base_config.json` is the
geometry source, and the contract's `target.geometry_source` pins its sha256
and bytes. `Sources/MLXFastTransform/PrismHadamardCheckpointValidation.swift`
derives the expected 1,033-tensor inventory from that geometry and the assumed
packing, and checks the pack's headers against it. The pack's own tensor
inventory fixture is pending with the pack.

The checkpoint's `.gitattributes`, `LICENSE` and image files are deliberately
not pinned. None is engine-loaded, and `setup.sh` derives what it downloads
from the manifest.

Participants never supply the target weights. Substituting or re-deriving the
target is a failure.

### 7.1 The decoder and the draft depth

You select the draft depth. The depth is not pinned at 1. This track declares
one decoder, the MTP head.

The declaration sets the depth, and `mtp-head.manifest.json` is an editable
path, so the depth is a free lever:

```json
"spec": { "decoder": "mtp", "enabled": true, "num_speculative_tokens": 3 }
```

The MTP head permits 1 to 7. Three layers agree on that number:

| Layer | Where |
|---|---|
| The contract | `mtp_head.permitted_draft_depths` is `[1 … 7]` |
| The protocol envelope | `protocol.mtp_envelope_constants_from_darkbloom.max_draft_tokens` is 7 |
| The engine | the runner declares `depth: 1 ... CBv2MTPConfig.testedMaxDraftTokens`, which is 7 |

`tools/spec-declaration.sh` is the single trusted reader of the declaration. It
refuses a decoder the track does not declare, and a depth that is not in the
DECLARED decoder's list, so a serve can never boot a decoder or a depth the
contract does not permit. With no `spec` block, `enabled: false`, or `0`, the
declaration resolves to depth 0 and the leg runs serial, whatever the
decoder.

The same envelope pins the speculative batch at 1 and the verification mode at
`rectangular_exact`. The arm is single-stream on both sides: the runner's
assistant declares `maximumSpeculativeBatch == 1`.

Every run seals the depth that operated. Read `effective_spec` for the depth the
run declared. Read `effective_mean_draft_len` for the draft length that the run
realized. The two can differ: a run can declare depth 2 and realize a mean draft
length near 1.

## 8. Prohibited techniques

A submission that uses any of these fails the static review.

- A cache or memo keyed on a request's input tokens whose only possible hit is
  the harness repeating one identical computation. Bit-identical output does
  not make it legitimate. The benchmark measures single-pass inference. An
  optimization must save work that recurs in single-pass production inference.
- Hardcoded hidden prompts, hidden token identifiers, or answers.
- Timing shortcuts, protocol injection, network access, and filesystem
  exfiltration.
- Any change outside `editablePaths`.

Input-independent caching stays legal. This covers weights, dequantized
tensors, and RoPE or mask tables keyed on shapes and offsets. Within-request KV
reuse stays legal.

Keep every change prompt-independent and model-general. The hidden prompts
differ from the public fixtures.

### 8.1 The ranked box confines your code

On the ranked box, the code from your editable paths runs in a sandbox.

- The transform (`mlxfast-swift transform`) can write only three things: its
  output directory, the hidden staging directories beside it whose names start
  with `.<output name>.`, and its own temporary directory (`TMPDIR`). It cannot
  use the network. It cannot start another program.
- The resident `bench-worker` can write only its Unix socket and its own
  temporary directory (`TMPDIR`). It cannot use the network except through its
  socket. It cannot start another program.
- The build compiles your C++, Metal and Swift files in the same sandbox
  (`tools/sandboxed-build.sh`). A compile can write only the build directories
  and temporary directories. It cannot use the network. A source file that
  names a file outside the repository (`#include`, `#embed`, `.incbin`) does
  not get the organizer material, and the build fails.
- None of these processes can read or write the organizer material on the box.

The sandbox starts before the transform process starts. A trusted script
(`tools/sandboxed-cli.sh`) writes the profile and starts `mlxfast-swift` under
`sandbox-exec`.

Make your transform obey these limits. A transform that writes to a different
location fails on the ranked box, and the run gets no score.

## 9. Submitting

Use the Yukon CLI for every account operation and every submission operation.
`README.md` holds the commands.

A submission archive packages only `editablePaths`. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first,
and no local run blocks the upload.

The ranked run on the official runner is the gate that ranks a submission.

## 10. License

The pack's license is pending with the pack. The contract's `license` block
carries the sentinel and verifies no repository yet.

The pack is built on `Qwen/Qwen3.5-9B`, which carries Apache-2.0.

This repository distributes no model weights.

## 11. What is not in place yet

Read this section before you conclude that something is broken.

### 11.1 The track is not armed

Section 5.5 states the arm state. The target and the head are pending,
`official_scoring_enabled` is `false`, the timed prompt pool and the live
goldens are empty, and the hidden correctness oracle is the sentinel. No runner advertises the ranked label set
`[self-hosted, macOS, bonsai2-9b-mlx-v1]`, and the box is not staged.

THE CHANNEL RESOLVES FROM `main`. benchd is published from the bench
repository's `main` branch; there is no per-track bench branch (ruled
2026-09-07). `./tools/fetch-benchd.sh` reads `dist/benchd.manifest.json` at the
tip of branch `BENCHD_BRANCH` (default `main`), checks the manifest's `branch`
against that channel, and installs the binary only when it matches the `sha256`
and `bytes` the manifest names. The script then prints the identity it resolved
and keeps the manifest beside the binary in `benchd-bin/`.

THAT MANIFEST IS THE SOURCE OF TRUTH for which `benchd` measures your run,
and it moves when the organizer republishes dist. This document therefore names
no `{source_commit, sha256, bytes}` triple as current: run the script and read
the identity line.

While the track is internal the channel host is the DEVELOPMENT bench
repository, which is private, so `./tools/fetch-benchd.sh` needs `GITHUB_TOKEN`
(or `BENCHD_DIST_TOKEN`), or a verified pair through `BENCHD_DIST_LOCAL`. At
launch the channel default flips to the public repository.

### 11.2 The engine serves the 9B geometry; the pack is pending

The engine constructs and gates `prism_hadamard_qwen35`. The geometry in
`Sources/MLXFastCore/Constants.swift`, the checkpoint validator in
`Sources/MLXFastTransform` and this contract's `target.*` block are all the
Qwen 3.5 9B backbone's, and they move as ONE SET: a gate holding some fields of
one model and some of another rejects every checkpoint and explains none of
them.

The runner, the Prism Hadamard loader and the model read every geometry fact
from the checkpoint's `config.json`, so the 9B geometry needed no code change.
The engine has run serial and MTP depths 1 to 7 against a stand-in checkpoint
with the same geometry and its own MTP layer. The stand-in is never a track
pin. The engine has not run against the Bonsai 2 9B pack, because the pack is
not published.

The model, the runner and the MTP assistant are in the vendored
`Vendor/mlx-swift-lm` tree. The Bonsai 2 and Qwen 3.5 model files in it are
editable paths, and so is `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`. The
runner is not; see section 3.
`bench-worker` selects the runner from the checkpoint's own `model_type`, and
this track passes it no `--resource` argument: the model loads no out-of-band
row source.

`docs/bonsai2-9b-port-notes.md` holds what the port changed, what is
verified and what is still open.

### 11.3 The public captures

The public captures are the goldens that `--local-iterate` and
`--local-submit` check against. They ship in this repository under
`correctness_prompts/bonsai2-9b-mlx-v1/`:

| File | Mode |
|---|---|
| `public-local-iterate.golden.json` | `--local-iterate` |
| `public-local-submit.golden.json` | `--local-submit` |

The organizer records both on the track's box, against the pinned target, in
one recording, so the short capture is the long capture truncated. The target
is pending, so neither is recorded yet and neither ships. Nothing fetches them
and no environment variable is needed. The hidden goldens are different
material (section 5.6): they live in R2 and on the ranked box, and
`tools/lint-benchmark-manifest.py` refuses a pinned hidden golden in the
checkout.

**A GOLDEN MUST CARRY `model_provenance`.** The block names the repository and
the revision of the pinned model. `loadQwenGoldenFixture` is the loader that
every golden consumer in `Sources/` calls, and it refuses a golden that carries
no block. The error message names `model_provenance`. While the reference pin
is the sentinel, it refuses every golden that names a checkpoint, by name. The
model-agnostic `loadGoldenFixture` keeps the reference schema, which has no
such key.

### 11.4 The batched cohort path is not part of this track

The scored shape is single-stream, and the engine declares nothing else. The
runner's MTP assistant declares `maximumSpeculativeBatch == 1`, so the runner
declares only the two single-stream regimes: free-run and teacher-forced. The
benchmarker therefore refuses a batched request at its pre-measurement
capability check rather than after it has spent box time.

That agrees with the fixture, which pins `scored_batch_size` 1 and
`scoring.mode` `bonsai2-native-mtp-paired-composite`. Section 5 describes the
single-stream paired series that follows.

### 11.5 The verify runs at the draft depth

A speculative round verifies its whole draft chain in ONE target forward. The
verify width is the resolved draft depth on the mtp leg; the serial control
still runs one token at a time. The envelope pins the verification mode
`rectangular_exact`.

**WHAT THE WIDE VERIFY RESTS ON.** Not bit-identity -- MLX dispatches a
different kernel at one row than at several, by design, so the logits differ in
their last bits. It rests on ARGMAX AGREEMENT: the wide forward picking the
same tokens. A committed token is correct when it matches what that forward
says, not when it matches what a one-token-at-a-time decode would have said.
Section 5.4 is the gate that prices any resulting difference in emitted tokens.

**WHAT THIS MEANS FOR YOU.** The arm is not strictly more work than serial for
the same output: a round pays one target forward for its whole chain instead of
one per committed token. It still pays the head forwards that proposed the
chain. Whether that becomes a speedup on the ranked box is a measurement, not a
promise, and it depends on the drafter -- an accepted draft is what buys the
forward back. Nothing on this track has been measured on a ranked box yet.

### 11.6 The KV backend is contiguous, and paged is not built

The model supports paged KV and sets `requiresNativePagedKV`, but
`RunnerEngineAssembly`'s paged branch builds neither the segment-size nor the
layer-dtype declaration that `EngineV2` demands. The runner therefore declares
the `contiguous` backend only. Declaring the paged one today would halt the
process instead of refusing the build, so the narrow declaration is the correct
state and not an oversight.
