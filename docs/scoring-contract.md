# Muse Glimmer 30B MLX — scoring contract

> Last checked: 2026-11-20, against this repository's `main` at `4fff0ec` and
> benchd source commit `a366c69`, the source commit that the bench channel's
> `dist/benchd.manifest.json` named on that date. The channel tip can move:
> `tools/fetch-benchd.sh` prints the identity it resolved for your run.

This page states how one ranked run of `muse-glimmer-30b-mlx-v1` becomes a
score, and when it becomes no score. Every line names its source. A source in
this repository is `file:line`. A source in benchd is `benchd <path>:<line>`,
a path in the benchd source tree at `a366c69`. When this page and the code
disagree, the code wins; section 1 of the
[participant contract](participant-contract.md) gives the full order of
authority.

## 1. The contract in nine lines

| # | Item | What the code does | Source |
|---|---|---|---|
| 1 | Formula | Each pair: `composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75`, with `gain = control leg seconds per token / candidate leg seconds per token`. The score is the arithmetic mean of the 3 per-pair composites. | `fixtures/muse_glimmer_30b_mlx_v1_track.json:6-7,42-45`; `benchmark.json:142`; benchd `crates/bench-core/src/score.rs:42-51,88`; benchd `crates/benchd/src/official.rs:2022-2030,2136` |
| 2 | Direction | Higher is better: `direction` is `+`. A faster candidate leg has a gain above 1. | `benchmark.json:7`; benchd `crates/bench-core/src/score.rs:50` |
| 3 | Units | Inputs are seconds per token. Prefill: the prefill time divided by the 512 prompt tokens. Decode: the time of `free_decode_run(128)` divided by 128. A gain is a ratio of two seconds-per-token figures, so the gains and the score have no unit. | benchd `crates/bench-core/src/score.rs:25-37`; `fixtures/muse_glimmer_30b_mlx_v1_track.json:35,40` |
| 4 | Rounding, ties | benchd does not round the score. `score.json` holds the full double, so two runs tie only when the two values are equal. benchd has no tie-break, and this repository states no tie rule. | benchd `crates/benchd/src/official.rs:2136`; benchd `crates/benchd/src/score.rs:554-575` (only diagnostic fields are rounded) |
| 5 | Fail-closed correctness | Timed window: the prefill token, the seed token and each of the 128 committed decode tokens must equal the tape. One different token fails the run. Untimed correctness gate: each teacher-forced case of 64 steps passes with at most 10% of its steps not accepted (a top-logit tie within 1e-6 is accepted). Any failure seals `score: null` and `passed: false`. | benchd `crates/benchd/src/main.rs:5264-5265` and `crates/benchd/src/official.rs:1093-1096` (the fixture declares no `timed_token_tolerance_per_thousand`); benchd `crates/bench-runner/src/timing.rs:1031,1220,1284`; benchd `crates/bench-core/src/conformance.rs:448-493`; benchd `crates/bench-core/src/constants.rs:642`; benchd `crates/benchd/src/score.rs:24-27`; `fixtures/muse_glimmer_30b_mlx_v1_track.json:34` |
| 6 | Estimator | Prefill: one timed pass after 1 unmeasured warm-up pass (the fixture declares no `official_prefill_timed_runs`, so benchd times 1 pass; the median of one pass is that pass). Decode: one 128-token free-run window per leg, one sample. Score: the arithmetic mean over the 3 pairs. Health band: the arithmetic mean of the calibration passes. | `fixtures/muse_glimmer_30b_mlx_v1_track.json:37`; benchd `crates/bench-core/src/contract.rs:856-860`; benchd `crates/bench-core/src/constants.rs:649`; benchd `crates/bench-runner/src/timing.rs:1052-1068,1378-1386`; benchd `crates/benchd/src/official.rs:2136`; benchd `crates/benchd/src/baseline.rs:1034-1035` |
| 7 | Trial count | 3 pairs per job, so 6 legs: 3 serial-control legs and 3 candidate legs. Each leg: 1 warm-up and 1 timed prefill pass, then 1 decode window of 128 tokens. Calibration: 3 passes per live golden. | `fixtures/muse_glimmer_30b_mlx_v1_track.json:6,35,37`; benchd `crates/bench-core/src/constants.rs:649`; `tools/calibrate-box.sh:198` |
| 8 | Max variance | Calibration only. The fixture declares no `calibration_max_cv_percent`, so the limit is benchd's default, 2.0%. The calibration refuses when the sample coefficient of variation (CV) of its passes is above 2.0% on either axis. A ranked run checks the CV in the calibration file against the same limit when it loads the file. The scored legs have no CV limit. Each control leg must be at most the calibration mean x 1.05 (prefill) and x 1.03 (decode); the low side is not checked. | benchd `crates/bench-core/src/constants.rs:292`; benchd `crates/benchd/src/baseline.rs:157-171,436-475,514-530,961-965,1041-1051`; benchd `crates/benchd/src/main.rs:4903-4909`; `fixtures/muse_glimmer_30b_mlx_v1_track.json:26-31` |
| 9 | Over-budget variance writes no score | A calibration over the CV limit writes no file, and with no valid file the preflight and benchd refuse the ranked run: no score. A control leg above its band refuses the run with `SERIAL-CONTROL-LEG-OUTSIDE-BAND` and seals `score: null`. The spread of the candidate legs is not checked: no run is refused for candidate variance. | benchd `crates/benchd/src/baseline.rs:1047-1051,460-475`; `tools/ranked-box-preflight.sh:405-414`; benchd `crates/benchd/src/score.rs:26` |

## 2. The gates on every pair

The fixture declares `official_pair_combine: "mean"`, so every pair is scored
and every pair is gated against its own control leg. The first pair that fails
stops the run, and the run seals `score: null`
(benchd `crates/benchd/src/official.rs:1689-1700,2037-2068`).

| Gate | Value | Source |
|---|---|---|
| Composite is finite | required | benchd `crates/benchd/src/official.rs:2032-2036` |
| Prefill gain floor | 0.95 | `fixtures/muse_glimmer_30b_mlx_v1_track.json:23` |
| Decode gain floor | 0.95 | `fixtures/muse_glimmer_30b_mlx_v1_track.json:22` |
| Control leg health band | at most mean x 1.05 (prefill), mean x 1.03 (decode) | `fixtures/muse_glimmer_30b_mlx_v1_track.json:26-31`; benchd `crates/benchd/src/baseline.rs:436-475` |
| Gain ceiling | none applied (see below) | benchd `crates/bench-core/src/contract.rs:184-188`; benchd `crates/bench-core/src/score.rs:239-258` |

**The gain ceiling.** `benchmark.json:152` states `decodeSpeedupCeiling` 5.0.
benchd reads its ceilings from the fixture only
(benchd `crates/benchd/src/main.rs:5256-5257`), and the fixture declares no
`decode_speedup_ceiling` or `prefill_speedup_ceiling`. benchd skips a ceiling
that is not declared, so no gain ceiling is applied today.

**The tape of the candidate leg.** The control leg is checked against the
serial tape. A speculative declaration needs a per-depth tape in the fixture's
`live_golden_speculative`, which is empty today
(`fixtures/muse_glimmer_30b_mlx_v1_track.json:288`), so the measure script
refuses a speculative declaration before it measures
(`tools/muse-glimmer-30b-measure-and-score.sh:311-316`). `mtp-head.manifest.json`
declares the drafter disabled, so the candidate leg runs serial and is checked
against the serial tape.

## 3. What the sealed score is

benchd `iterate --mode official` is the only writer of the sealed `score.json`
(`.github/workflows/benchmark.yml:84-87`). The file holds `score`, `passed` and
`metrics`. `score` is a finite number on a passed run and `null` on any failure
(benchd `crates/benchd/src/score.rs:24-27`). Every pair stays in
`metrics.paired_legs` as measured (benchd `crates/benchd/src/official.rs:1689-1693`).
