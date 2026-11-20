# Bonsai 2 9B MLX — port notes

> **HISTORICAL RECORD — NOT THIS TRACK.** This page is the engineering record
> of an earlier track that this repository was seeded from. It does not
> describe the Muse Glimmer 30B track. `tools/new-track.sh` names this file in
> its stamping exemption list. The record for this track is
> `Vendor/mlx-swift-lm/docs/muse-glimmer.md`.

This document is the engineering record for track `bonsai2-9b-mlx-v1`. It
states what the port changed, what is verified and what is open.
`docs/bonsai2-27b-port-notes.md` is the record of the track this repository
was stamped from.

## 1. The seed

The repository `main` starts with one signed commit, `578f41d`. Its tree is
identical to the source engine's `main` `08c75e9`.
The history stays in that repository
(`docs/new-track-repo-procedure.md` section 1).

## 2. The layers

The port is five pull requests, one for each layer.

| Pull request | Layer | What it does |
|---|---|---|
| #1 | Stamp and contract | `tools/new-track.sh` stamps the track id, the manifest name, the fixture name and the runner label. The contract carries the Qwen 3.5 9B geometry and the pending pins. The compiled constants and the transform validator move to the 9B geometry. The golden loader takes an injectable reference pin. |
| #2 | Engine fixes | The fixes of the Nemotron 3.5 30B A3B MLX engine, pull requests #11 to #28, one commit for each source commit. |
| #3 | Vendored model and runner | A test pins the 9B residency. The runner header names the pending pack. |
| #4 | Tools | This track's tool names, the pending pins in the setup scripts, and one decoder. |
| #5 | Docs | The product docs and these notes. |

### 2.1 The engine fixes taken in #2

| Nemotron pull request | Fix |
|---|---|
| #11 | `live_goldens` is a list. `live_golden_speculative` is keyed by depth, then by live golden name. Calibration, the preflight and the measure script take one golden flag for each live golden. |
| #12 | The bench-worker phase-close drain settles before it reports `cache_memory`. |
| #13 | `setup.sh` exports the log label to the shard downloaders. |
| #15 | `setup.sh` verifies a read-only reference checkpoint and changes nothing. |
| #16 | The job account boundary: Seatbelt for the build, the CLI and the resident; preflight checks 7a and 7b; `RESIDENT_UP_LOG_DIR`. |
| #17 | Calibration runs three passes. |
| #18 | The score is the mean of the per-pair composites (`official_pair_combine: "mean"`). |
| #22 | The preflight accepts calibration file version 3 only, where decode is the decode window. |
| #23 | `yukon run` off a ranked box runs the local test at the declared depth. |
| #24 | The paged-attention kernel source is cached once per process. |
| #25 | The Metal device is queried once, without the window server. |
| #27 | `tools/resident-up.sh` removes the sandboxed resident's temporary directory. |
| #28 | The decode band allows 3 percent upward. |

Not taken: #13's goldens, seed and re-pin, and #14's arming, because they are
Nemotron's own material; #19, because #1 already ships the stock declaration
serial; #26, because the reference re-pin waits for the final engine. The
score artifact upload in `.github/workflows/benchmark.yml` was already the
same as Nemotron's.

`tools/golden-arming-patch.py` is this track's own tool. It moves to the list
shape: `--live` is repeatable, and each depth carries one oracle pin for each
live golden.

## 3. The pending design

prism-ml has not published the Bonsai 2 9B MLX pack. The design is built so
that the published pack replaces sentinels and nothing else.

- **Sentinel.** Every pin that names the pack's bytes or the MTP head's bytes
  carries the exact string `BONSAI-MLX-PENDING-ORGANIZER`: the contract's
  target and head pins, the license, the hidden correctness golden, and the
  compiled reference pin. Both reference manifests hold no record.
  `./setup.sh` refuses the pending reference by name.
  `tools/ranked-box-preflight.sh` refuses any `PENDING-ORGANIZER` string.
- **Geometry.** `fixtures/qwen35_9b_base_config.json` is the published
  `config.json` of `Qwen/Qwen3.5-9B` @ `c2022362`, verbatim. The contract's
  `target.geometry_source` pins its sha256 and bytes, and the tests assert the
  contract's geometry against it.
- **Packing.** The pack is assumed to use the Ternary Bonsai 2 27B packing.
  `fixtures/bonsai2_27b_config.json` stays as `target.packing_reference`. The
  transform validator derives 1,033 language-model tensors and 202 packed
  modules from the 9B geometry and that packing. The transform tests use a
  synthetic 9B pack config built from the packing reference and the backbone.
- **Golden loader.** The default reference pin is the compiled pin, which is
  the sentinel. The loader refuses every golden that names a checkpoint until
  the pack is pinned. The tests pass a synthetic pin.
- **One decoder.** No DFlash 2 drafter exists for the Qwen 3.5 9B geometry.
  `z-lab/Qwen3.5-9B-DFlash` is a DFlash v1 model (`DFlashDraftModel`), and the
  engine's drafter is DFlash 2. The contract declares serial and MTP only.
  `tools/spec-declaration.sh`, `tools/resident-up.sh` and the trusted head
  declaration parser refuse `dflash` by name. The vendored DFlash 2 code and
  its two editable model files stay.
- **Prompts.** The planned live prompts are the Bonsai coding prompts
  `pool-9`, `pool-10` and `pool-11` at a 512-token seed. The 27B Bonsai
  tokenizer and the Qwen 3.5 9B tokenizer have the same vocabulary, the same
  added tokens and the same merges, and the non-thinking chat wrap is the same,
  so the token files of those prompts carry over. Recheck this against the
  published pack.
- **Reference commit.** `baseline_reference_commit` names the seed. It is
  re-pinned to the final engine commit before the first scored window.

## 4. The stand-in validation

A laptop run against a stand-in checkpoint proved the engine on the 9B
geometry. The stand-in is `EigenLabs/Qwen3.5-9B-MLX-4bit-mtp` @ `556dcba5`:
the same geometry, at 4 bits, with its own embedded MTP layer. The run used the
repository's local tools. It recorded a serial golden twice, and the two
recordings were byte-identical. It then ran the local test serial and at MTP
depths 1 to 7. Each run passed correctness and matched the serial free run
token for token.

The stand-in is never a track pin, and no track golden comes from it. Its
golden stayed on the laptop and is not in this repository.

## 5. Open items

1. The pack's real name, its repository and revision, its file pins and its
   tensor inventory.
2. The MTP head: its repository and revision, its pins, its quantization and
   its tensor count.
3. Whether the pack carries its own `mtp.*` block. The transform refuses one
   today; the rule and the head pin must move together if it does.
4. The tokenizer and the chat template of the published pack, checked
   against the planned prompts.
5. The ranked box: the runner, the staged reference tree and the calibration.
6. The goldens: the live goldens and the per-depth oracles, recorded on the
   box, A and B byte-identical, uploaded to R2; the public captures, committed.
7. The re-pin of `baseline_reference_commit` to the final engine commit.
8. Arming `official_scoring_enabled`, last, in its own pull request, after one
   clean scored window.
9. The MTP draft depth above 4. The contract permits depths 1 to 7, but the
   Qwen 3.5 MTP assistant drafts at most four tokens in a round
   (`maximumDraftTokens` in
   `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`, an editable
   file). Depths 5, 6 and 7 therefore run as depth 4, and the effective spec
   still echoes the declared depth. The Ternary Bonsai 2 27B track has the same
   limit.
