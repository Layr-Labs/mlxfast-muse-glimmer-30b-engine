# Agent guide — Muse Glimmer 30B MLX engine

This file is the working contract for coding agents in this repository.
`CLAUDE.md` is a symbolic link to this file.

Read [README.md](README.md) first. It states what this repository is, how to
set it up, and what the structure is. This file adds the operational rules that
a person or an agent needs while iterating here.

The ranked track is `muse-glimmer-30b-mlx-v1`.

## Goal

Make the Muse Glimmer 30B model decode and prefill faster on Apple Silicon. Do not
change the observable model behavior beyond what the token-tolerance gate
allows.

## Authorities

| Question | Authority |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/muse_glimmer_30b_mlx_v1_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering record for this port | `docs/bonsai2-9b-port-notes.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`, verified against the dist `benchd.manifest.json`) |

`benchmark.json` and `fixtures/muse_glimmer_30b_mlx_v1_track.json` carry pure
configuration. They hold values, paths, commands, and pins. They carry no prose.
Where this file disagrees with the fixture, the fixture wins.

## Lineage

This repository is a seed of the Ternary Bonsai 2 27B MLX engine
(`Layr-Labs/mlxfast-bonsai2-27b-engine-dev`). That engine descends from the
Nemotron 3.5 30B A3B and Qwen 3.8 125B A6B MLX engines, and from
`Layr-Labs/mlxfast-challenge-dev`. Those repositories rank different models
under different rules. Only this track's rules apply here.

The `.gemma4` transform family in `Sources/MLXFastTransform/` is not a
leftover. It is the synthetic-fixture substrate the transform pipeline tests
run on, and the tests exercise the transform rather than a Gemma model.
Renaming it would make the name lie about what it holds.

Four environment names keep a Qwen spelling on purpose, because each is a
contract with the other side rather than this track's own name:
`MLXFAST_QWEN_MTP_TRACK_ID` is the name benchd requires for the track id,
`MLXFAST_QWEN38_GOLDEN_DIR` is the fleet's golden-directory contract, and
`MLXFAST_QWEN38_R2_DOWNLOADER` and `MLXFAST_RUN_QWEN_REFERENCE_PARITY` keep
their names for the same reason.

## Current state

THE TARGET IS PINNED. The target is `nvidia/Muse-Glimmer-30B-NVFP4` at
`47818374517751c48c55cde2621594926b1888b6`, pinned file by file in
`fixtures/reference_bonsai2_9b_2bit.sha256` (that file NAME is inherited from
the source track; every record in it is the Muse Glimmer checkpoint).
`./setup.sh` provisions exactly that checkpoint.
`MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh` builds the harness and the engine
only.

The geometry is read off `fixtures/muse_glimmer_30b_nvfp4_config.json`, a byte
copy of the pinned snapshot's own `config.json`: 52 text layers, hidden size
6656, dense MLP at 19968, vocabulary 202048, 32 query heads over 2 KV heads at
head_dim 128, context 131072, 39 sliding-attention layers at window 2048 and 13
full-attention NoPE layers at every index `% 4 == 3`. The weights are mixed
precision modelopt: 4-bit NVFP4 at group 16 on 213 modules and float8 on 180.
The checkpoint also carries a 50-layer vision tower; this text-only track never
loads that namespace.

Official scoring is NOT armed. `fixtures/muse_glimmer_30b_mlx_v1_track.json`
sets `official_scoring_enabled` to `false`, its `timed_prompt_pool` is empty,
its `live_goldens` list is empty, and `hidden_correctness_golden` carries the
sentinel `MUSE-GLIMMER-30B-MLX-V1-PENDING-ORGANIZER`. The benchmarker refuses to
seal an official artifact in that state. The planned live prompts are the three
coding prompts `pool-9`, `pool-10` and `pool-11` at a 512-token seed, recorded
on that box.

The track has no goldens. A track's goldens are organizer material: they are
recorded on the track's own box, published in R2 under
`correctness_prompts/<track id>/`, and staged out of band into the directory
the box's runner service exports as `MLXFAST_QWEN38_GOLDEN_DIR`. They are never
in git.

The two public captures for the local modes are different material. They ship
in this repository under `correctness_prompts/muse-glimmer-30b-mlx-v1/`, so a local
test needs no R2 access. The organizer records them on the track's box when the
goldens are recorded. None ship yet. Never commit a hidden golden, a pool tape, a
prompt file or an R2 key.

Scoring is paired, with a per-box baseline. A ranked run measures pairs on the
same box in the same job. The fixture lists the live prompts in `live_goldens`.
Pair k measures live prompt (k - 1) mod N, where N is the number of live
prompts, so `official_pairs` must be a multiple of N. The score is the mean of
the per-pair composites (`official_pair_combine: "mean"`). A pair measures two
legs over one prompt: a serial-control leg on the organizer's reference tree
(`MLXFAST_BASELINE_WORKSPACE`) and the candidate leg at its declared draft
depth. The decode gain is measured over the decode window only; the seed
prefill is not in it. The score of a pair is the live ratio. **NO FILE STORES
A BASELINE PAIR.** No golden carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token`, and
`tools/lint-benchmark-manifest.py` check 5b keeps both fields out of the tree.

`baseline_reference_commit` names the repository's seed commit. It is
re-pinned to the final engine commit before the first scored window. A
trusted-side code change that the control leg must carry re-pins it, and every
box re-stages its reference tree.

Each box carries its own calibration (`MLXFAST_BASELINE_CALIBRATION`). It is a
health band for the control leg, never a denominator: the run stops by name
when the control leg falls outside the band. The file holds one entry for each
live golden, and the preflight accepts calibration file version 3 only.
`tools/calibrate-box.sh` writes the file on the box in three passes, and
`tools/stage-baseline-workspace.sh` builds the reference tree there.

No ranked runner is registered yet. `.github/workflows/benchmark.yml` runs the
hosted surface check, then the self-hosted ranked job on
`[self-hosted, macOS, muse-glimmer-30b-mlx-v1]`. The job holds no credential:
the goldens, the reference tree and the calibration file are staged on the box
and verified by `tools/ranked-box-preflight.sh`, which refuses rather than
fetch, build or substitute.

`./tools/fetch-benchd.sh` resolves the bench repository's `main` dist channel
and verifies the pair against its manifest.

Scoring is single-stream. The fixture sets `scored_batch_size` to `1`. Each leg
runs one stream, and the candidate leg runs with the DFlash drafter at the
declared depth. The runner declares single-stream regimes only, because its
drafter declares `maximumSpeculativeBatch == 1`.

The track declares ONE speculative decoder: `dflash`, served by the quantized
DFlash drafter at depth 1 to 7. The drafter is produced deterministically from
`meta-models/Muse-Glimmer-30B-assistant` at `c676cce9` by the pinned artifact
writer, and its two files are pinned by SHA-256 in the contract fixture's
`mtp_head` block (block size 16, taps [1, 13, 25, 37, 49], 1,968,222,936 bytes,
under the 2 GiB cap). The shipped `mtp-head.manifest.json` declares the decoder
switched off, so the stock tree runs serial (depth 0). `tools/spec-declaration.sh`
is the single trusted reader of that file, and `tools/spec-declaration.sh
describe` prints what the tree declares.

The KV backend is PAGED. The model's own engine build refuses a contiguous
cache, so `kv_backend` in the fixture is `paged` and no mode may ask for
anything else.

> **The engine fork is a VENDORED TREE, not a submodule.**
> `Vendor/mlx-swift-lm` holds plain files cut at the local merge of
> `feat/muse-glimmer-speed` `62cfabd5` and `feat/muse-glimmer-rowexact-m5-d5`
> `44378936`, whose tree is `0391eacd84cb00ce4001d70763f2303fa2a01210`. No
> upstream commit names that merge, so the contract fixture's
> `mlx_swift_lm_revision` records the TREE, not a commit.
> This tree is the source of truth: edit it in place, here, and push nothing
> upstream. Participants may edit only the model files `benchmark.json` names;
> every other change to the tree is a trusted-side PR here.

## Notes for autonomous agents

These behaviors are expected. They are not bugs.

### The cool-down gate

The benchmarker waits for the GPU to cool before it starts a timed run. The
local modes pass `--cool-gate` to the benchmarker automatically. The gate reads
the GPU temperature through `macmon`.

**The gate lives in the benchmarker, and only `./benchmark.sh` arms it.**
`./benchmark.sh` passes `--cool-gate` to `benchd`, and `benchd` runs the gate
itself before each timed phase. Prefill and decode are gated separately.

The trusted CLI no longer runs the model, so there is no ungated Swift path
left to arm. Use `./benchmark.sh`. It is the measured path.

`./benchmark.sh --local-cool-gate-only` exits 0 without probing anything. The
bare probe is the benchmarker's own entry point.

```bash
benchd-bin/benchd --local-cool-gate-only
```

> **WARNING — a run that pauses on a cool-down message is working, not hung.**
> Do not kill it. Do not treat the wait as a failure.

The gate aborts with a non-zero exit when the GPU stays hot and is not trending
down. That abort means something else is loading the GPU. Free the GPU and
retry. The abort does not mean your change is wrong.

`./setup.sh` installs `macmon` as a pinned, hash-verified release binary. The
gate warns and skips when `macmon` is absent. Skip the install with
`MLXFAST_SKIP_MACMON_INSTALL=1`.

> **WARNING — a skipped gate still produces a number.**
> Locally, no reader means no gate, and the run times whatever temperature the
> GPU happens to be at. Treat a timing taken without `macmon` as unmeasured.

The ranked box does the opposite. A missing or frozen reader is a hard refusal
there, before any measurement (`tools/ranked-box-preflight.sh`, sections 2b and
2c). A ranked run never proceeds without thermal control.

The gate mirrors the ranked runner's fixed 40 C thermal contract. That contract
is operator-owned. The benchmarker owns the exact thresholds; this repository
does not set them. The threshold is a fixed constant inside the benchmarker and
no fixture can move it.

### Fan control for a stalled cool-down

Use the fan helper when the local gate sits hot with no cooling progress. The
helper is manual only. No gate, script, or workflow invokes it. Nothing boosts
the fans on your behalf, and a stalled cool-down will not fix itself.

```bash
tools/fan-control.sh boost
```

This command forces every fan to 70% of its maximum speed.

```bash
tools/fan-control.sh normal
```

This command returns the fans to macOS's automatic curve.

```bash
tools/fan-control.sh status
```

This command prints `manual`, `auto`, or `none`.

Fan targets are SMC keys. macOS accepts SMC writes only from root. The helper
therefore runs its writes under `sudo`. `sudo` prompts for the password itself.
The helper never reads, stores, echoes, or logs the password. It drops the
cached credential with `sudo -k` right after the writes. The helper needs an
`smc` CLI. It refuses cleanly on a fanless Mac.

### Measurement discipline

Trust a timing number only from a cool, quiescent machine. Back-to-back runs
heat the GPU and throttle it. A 2-minute to 3-minute pause between local runs is
normal.

> **WARNING — a local score is directional.**
> The local test and the ranked run both time one stream, so the shapes agree.
> Your machine is still not the ranked box: its memory budget, its thermal
> behavior and its kernel selection all differ. Do not read a local score as a
> prediction of the ranked composite.

Record a same-machine baseline before you optimize. Sync to the latest tip
first. Do not compare a change against a stale branch or an old local run. Rerun
the baseline whenever the base commit changes.

### One model-holding run at a time

The target model is RAM-resident. Two model residencies at once can exhaust a
local machine's memory.

> **WARNING — run one model-holding command at a time.**
> Do not start a second local run while the first is alive. Every model
> residency in this tree is a `bench-worker` process that the benchmarker
> started. The trusted CLI loads no model.

No run lock enforces this. The discipline is yours to keep.

`swift test` never loads the real model. It is safe to run alongside.

Check for an orphaned worker when a run aborts. A worker whose parent process
identifier is 1 is usually an orphan. Verify it, then kill it.

### The startup memory profile

The runtime selects a low-memory profile automatically below 64 GiB of physical
memory. The profile caps the MLX allocator cache at 6 GiB, shortens command
buffers, and releases free warmup buffers before the worker serves requests.

The profile is pure memory management. It disables no code path and no
output-affecting feature. It announces itself on stderr. Force it either way
with `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto`.

A machine that is too small fails loudly with an out-of-memory error. It does
not diverge silently from ranked behavior.

### The non-M5 near-tie caveat

A greedy continuation is captured on one machine. A near-tie argmax can diverge
on another Apple Silicon generation, even for correct code.

> **WARNING — a local gate failure on non-M5 hardware may not be your bug.**
> Check whether an unmodified `main` fails at the same token position on your
> machine. Do that before you treat a local failure as a regression.

Rerun with `MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` when unmodified `main` fails the
same way. The local mode then still publishes its timing estimate.

The override is local-only. It hides nothing. The score keeps
`passed_correctness: false`, records the diverging tokens, and explains itself in
`metrics.error`.

> **WARNING — never use the override to paper over a real regression.**
> The mismatch is yours when unmodified `main` passes on your machine.

### One ranked machine, one queue

Ranked runs execute serially on a single runner. Duplicate dispatches queue
behind the run in flight. They do not cancel it. Expect delays. Do not dispatch
several ranked runs in parallel and expect concurrent results.

### Know the runnable surface

Only the `benchmark.json` `editablePaths` entries ship in a submission. A change
anywhere else does not upload, even when it helps locally. Official ranking
needs hidden organizer goldens. It is not runnable locally.

## Building

Two build trees exist. Keep them straight.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI into `.build/release`.

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker \
  --product bench-worker
```

This command builds the scored engine into `.build-worker/release`. The engine
is the fork's generic `bench-worker`, built as a dependency product of this
package, so it reads `Vendor/mlx-swift` and your kernel edits reach it.

The engine builds under its own scratch root so a participant compile can never
write into the trusted tree.

```bash
tools/stage-bench-worker.sh
```

This command copies the finished engine and its `mlx.metallib` into
`.build/release`.

> **WARNING — a bare `swift build -c release` is not enough.**
> The scored binary is `.build-worker/release/bench-worker`. Metal loads
> `mlx.metallib` from the directory of the running binary. The staging step
> puts the pair where the benchmarker resolves them. `./setup.sh` runs that
> step for you.

### Kernel edits

The vendored MLX package builds in JIT mode. Two forms matter.

Families with an `mlx-generated/*.cpp` twin compile at runtime from the C++
source strings inside those files. The twin is the runtime-effective source.
Edit the twin. Keep the readable `.metal` and `.h` pair in step.

RoPE, RMSNorm, the SDPA vector kernel, and `arg_reduce` load ahead of time from
`mlx.metallib`.

```bash
tools/build-mlx-metallib.sh
```

This command rebuilds `mlx.metallib` from the vendored `.metal` sources. Run it
after you edit an ahead-of-time source. `./setup.sh` runs it for you.

`_nax` names are the M5-generation kernel variants. The ranked runner selects
them. Tune the `_nax` twin as well as the plain one.

Rebuild both binaries after any kernel edit. Then re-measure through the
benchmarker.

### The frozen dependency graph

> **WARNING — pass `--force-resolved-versions` on every direct `swift build`
> and `swift test`.**
> The dependency graph is frozen. A bare invocation can rewrite
> `Package.resolved` silently. `./setup.sh` then refuses to run. The flag makes
> SwiftPM fail closed instead.

Avoid bare `swift package resolve` and `swift package update`. They can rewrite
`Package.resolved` and there is no fail-closed flag for `resolve`. Restore the
file with `git checkout -- Package.resolved` when it shows as modified.

## Common commands

```bash
swift test --force-resolved-versions
```

This command runs the cheap contract tests.

```bash
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
```

This command also runs the MLX runtime tests. Use it when a change touches MLX
runtime behavior and the machine can run those tests.

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command provisions the target model, then stages the MTP head beside it
with `./setup-mtp-head.sh`. While the target is pending, it refuses by name.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the local test with the MTP head at depth 3.

## Swift tooling

Use the Swift toolchain that `./setup.sh` validates. `sourcekit-lsp` is the
standard Swift language server. Xcode or the Swift toolchain usually installs
it. Point your editor at the repository root. SourceKit-LSP then reads
`Package.swift` and resolves the SwiftPM targets.

Prefer SourceKit-LSP symbol navigation over string-only edits when you change
Swift model code.

## Where to spend effort

Good changes improve one or more of these.

- Kernel-level work inside the vendored Metal sources. Prioritize kernels the
  seed prefill and the timed decode window reach.
- The speculative arm. The draft depth, the block proposal and the wide verify
  forward are all competitive surface. The DFlash drafter proposes a block of
  16 rows and the target verifies a chain at depth 1 to 7.
- Layer dispatch. The tower mixes 13 full-attention NoPE layers and 39
  sliding-attention layers at window 2048 on a fixed every-fourth schedule, and
  the two types take different paths.
- The paged KV path. Every layer owns its own cache: three sliding layers and
  one full layer per repeat, no KV sharing, no sinks.
- The exact-window path. Decode and verify rows can run row-exact at any row
  count while prompt chunks keep MLX's arithmetic on a 1,024-token grid; the
  projection kernels that make that possible are the hottest arithmetic in the
  tower.
- The quantized matmul. The weights are mixed precision modelopt: 4-bit NVFP4
  at group 16 and float8, dispatched through the vendored quantized kernel
  families (the `_nax` twins are the M5-generation variants).
- Weight loading and reuse. Prepare eagerly at init. Warm kernels before the
  first scored forward. Avoid redundant conversions.
- MLX operation scheduling and synchronization.
- Transform metadata that lets the runtime skip work safely.

## Wrong strategies

Do not specialize for the public correctness prompt. Keep every change
prompt-independent and model-general. The hidden prompts differ from the public
fixtures.

Do not assume the ranked box has your local machine's memory budget. A strategy
tuned on one Apple Silicon generation can move differently on another.

Do not treat a local-only environment override as proof of a valid improvement.
Disabling the sandbox, skipping the transform without verifying `weights/`, and
pointing at a user-specific reference path are debugging aids. They do not
establish a rankable optimization.

Do not draw a conclusion from a tiny local run alone. A local run is a smoke
test. It is especially weak for sequence-length-dependent changes, because it
may not exercise the ranked sequence lengths or the ranked memory pressure.

Be conservative with numeric reassociation. A changed accumulation order can
flip a near-tie greedy argmax.

> **WARNING — the target quantization is frozen as shipped.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. `Sources/MLXFastTransform/` is editable, but that does not
> license a change of target format: a lossier target substitutes a degraded
> model instead of optimizing the accepted one. The MTP head is a narrow
> exception, and the exception is RE-QUANTIZATION ONLY (David ruling
> 2026-08-26) — re-quantize the head within its 2 GiB declaration cap, but do
> not replace it and do not upload head weights. The head is a SEPARATE pinned
> export, and `mtp-head.manifest.json` accepts `"source": "pinned"` only, which
> means that export. A head re-quantization happens ON LOAD, in memory: the
> head loader calls `quantize(model:)` while it binds the head, and the file
> that holds that call is editable (`Qwen35MTP.swift`). Nothing on disk changes
> (`docs/participant-contract.md` section 4.4).
> The head only proposes tokens; the pinned target decides every emitted token.
> The target's own quantization is verified on the LOADED model TWICE: once at
> worker startup, and again at the top of every window that gets measured,
> immediately before the measured work starts. The second check is there because
> the first alone verifies a model that later code can still change in place. An
> in-memory re-quantization of the target is refused by name, and the refusal
> stops the worker before any measurement.

> **WARNING — do not add a cache keyed on a request's input tokens whose only
> possible hit is the harness repeating one identical computation.**
> Bit-identical output does not make it legitimate. The benchmark measures
> single-pass inference. An optimization must save work that recurs in
> single-pass production inference. The harness never legitimately issues the
> same whole-prompt forward twice to one worker process. Any such repetition is
> a harness bug, never a contract to rely on. Input-independent caching stays
> fine. Within-request KV reuse stays fine. A change in this category fails the
> static review as bypass behavior.

Do not hardcode hidden prompts, hidden token identifiers, or answers. Do not use
timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

## Before submitting

Run at least these commands.

```bash
swift test --force-resolved-versions
```

This command runs the contract tests.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI.

```bash
./setup.sh
```

This command provisions the target model and the MTP head.

```bash
./tools/fetch-benchd.sh
```

This command resolves the pinned benchmarker. Run a local test afterwards.

Check the non-M5 near-tie caveat above when local correctness fails. Prefer a
more conservative optimization when performance improves but correctness turns
fragile.

Use the Yukon CLI for every account operation and every submission operation.
README.md holds the submission commands. Python is not part of the challenge
runtime.
