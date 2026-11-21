#!/usr/bin/env bash
#
# test-lint-guard-keys.sh -- tools/lint-benchmark-manifest.py check 5d (guard
# keys) and the published-limit mirror refuse a bad anti-cheat key by name.
#
# WHAT IS UNDER TEST. The pinned benchd already carries the anti-cheat guards;
# the track turns them on with keys in the contract fixture. benchd refuses a
# malformed key at run time, on the box, after the job has queued. The linter
# refuses it at rest, in CI. This suite proves each refusal binds, and that the
# shipped fixture and benchmark.json pass it.
#
# HERMETIC. Each case writes a throwaway copy of benchmark.json whose
# contractPath is an ABSOLUTE path to a throwaway copy of the fixture, then
# lints it against this repository root. The shipped files are never written.
#
# Usage: tools/test-lint-guard-keys.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LINT="${REPO_ROOT}/tools/lint-benchmark-manifest.py"
CONTRACT="${REPO_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json"
MANIFEST="${REPO_ROOT}/benchmark.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
passes=0
case_n=0
pass() { passes=$((passes + 1)); echo "ok    $*"; }
fail() { failures=$((failures + 1)); echo "FAIL  $*" >&2; }

# lint_case CONTRACT_EDIT MANIFEST_EDIT -- write the two throwaway copies, apply
# each edit (a python statement over `c` = contract dict or `m` = manifest
# dict; empty = no edit), lint, and leave the output in LINT_OUT and the exit
# status in LINT_RC.
lint_case() {
  case_n=$((case_n + 1))
  local c_out="${WORK}/contract-${case_n}.json" m_out="${WORK}/manifest-${case_n}.json"
  python3 - "${CONTRACT}" "${MANIFEST}" "${c_out}" "${m_out}" "$1" "$2" <<'PY' || return 1
import json, sys
src_c, src_m, out_c, out_m, edit_c, edit_m = sys.argv[1:7]
c = json.load(open(src_c, encoding="utf-8"))
m = json.load(open(src_m, encoding="utf-8"))
if edit_c:
    exec(edit_c, {"c": c})
if edit_m:
    exec(edit_m, {"m": m})
m["contractPath"] = out_c
json.dump(c, open(out_c, "w", encoding="utf-8"), indent=2)
json.dump(m, open(out_m, "w", encoding="utf-8"), indent=2)
PY
  LINT_OUT="$(cd "${REPO_ROOT}" && python3 "${LINT}" --repo-root "${REPO_ROOT}" \
    --manifest "${m_out}" --gitlink-targets report 2>&1)"
  LINT_RC=$?
}

# expect_refusal NAME NEEDLE -- the last lint_case exited 1 and printed a FAIL
# line holding NEEDLE.
expect_refusal() {
  local name="$1" needle="$2"
  if [[ "${LINT_RC}" == "1" ]] && grep -F "FAIL  " <<<"${LINT_OUT}" | grep -qF -- "${needle}"; then
    pass "${name}"
  else
    fail "${name}: want exit 1 and a FAIL line with '${needle}'; got exit ${LINT_RC}:"
    grep -E '^(FAIL|Traceback)' <<<"${LINT_OUT}" | sed 's/^/    /' >&2
  fi
}

# expect_no_fail NAME NEEDLE -- the last lint_case printed no FAIL line holding
# NEEDLE, and the linter ran to its summary (no traceback).
expect_no_fail() {
  local name="$1" needle="$2"
  if ! grep -qE '(checks passed|FAILURE\(S\))' <<<"${LINT_OUT}"; then
    fail "${name}: the linter did not reach its summary:"
    tail -5 <<<"${LINT_OUT}" | sed 's/^/    /' >&2
  elif grep -F "FAIL  " <<<"${LINT_OUT}" | grep -qF -- "${needle}"; then
    fail "${name}: unexpected FAIL line with '${needle}':"
    grep -F "FAIL  " <<<"${LINT_OUT}" | sed 's/^/    /' >&2
  else
    pass "${name}"
  fi
}

# --- M3: the prefill speed ceiling ------------------------------------------
# The value is the fixture's, read here, so the suite holds whatever value the
# fixture and benchmark.json state; the needles below use the linter's own
# rendering of it (Python repr).
CEIL="$(python3 -c 'import json,sys; print(repr(json.load(open(sys.argv[1]))["prefill_speedup_ceiling"]))' "${CONTRACT}")"
OTHER_CEIL="$(python3 -c 'import sys; print(repr(float(sys.argv[1]) + 1.0))' "${CEIL}")"

# 1. The shipped pair passes the ceiling checks, and both files state one value.
lint_case "" ""
expect_no_fail "shipped fixture: no guard-key refusal on the gain ceilings" "_speedup_ceiling"
expect_no_fail "shipped manifest: the published ceilings equal the fixture" "SpeedupCeiling"
if [[ "$(jq -r '.prefill_speedup_ceiling' "${CONTRACT}")" == "$(jq -r '.scoring.prefillSpeedupCeiling' "${MANIFEST}")" ]]; then
  pass "shipped prefill ceiling ${CEIL} is the same in the fixture and in benchmark.json"
else
  fail "shipped prefill ceiling differs between the fixture and benchmark.json"
fi

# 2. A ceiling of exactly 1.0 is refused (benchd: must be above 1.0).
lint_case 'c["prefill_speedup_ceiling"] = 1.0' 'm["scoring"]["prefillSpeedupCeiling"] = 1.0'
expect_refusal "prefill ceiling 1.0 refused" "guard keys: prefill_speedup_ceiling = 1.0 must be a finite number above 1.0"

# 3. A string is refused, not coerced.
lint_case 'c["prefill_speedup_ceiling"] = "5.0"' ''
expect_refusal "prefill ceiling as a string refused" "guard keys: prefill_speedup_ceiling = '5.0'"

# 4. A boolean is refused (true is an int in Python).
lint_case 'c["prefill_speedup_ceiling"] = True' ''
expect_refusal "prefill ceiling as a boolean refused" "guard keys: prefill_speedup_ceiling = True"

# 5. A ceiling at or below its own floor is refused: no gain could pass.
lint_case 'c["prefill_speedup_ceiling"] = 1.2; c["prefill_speedup_floor"] = 1.5' \
          'm["scoring"]["prefillSpeedupCeiling"] = 1.2'
expect_refusal "prefill ceiling not above its floor refused" "is not above prefill_speedup_floor = 1.5"

# 6. benchmark.json publishing a different ceiling from the fixture is refused.
lint_case '' "m['scoring']['prefillSpeedupCeiling'] = ${OTHER_CEIL}"
expect_refusal "published prefill ceiling differs from the fixture" "scoring.prefillSpeedupCeiling = ${OTHER_CEIL} disagrees with contract prefill_speedup_ceiling = ${CEIL}"

# 7. A fixture ceiling benchmark.json does not publish is refused.
lint_case '' 'del m["scoring"]["prefillSpeedupCeiling"]'
expect_refusal "fixture ceiling not published in benchmark.json" "contract prefill_speedup_ceiling = ${CEIL} but benchmark.json scoring states no prefillSpeedupCeiling"

# 8. A published ceiling the fixture does not enforce is refused.
lint_case 'del c["prefill_speedup_ceiling"]' ''
expect_refusal "published ceiling the fixture lacks" "scoring.prefillSpeedupCeiling = ${CEIL} but the contract declares no prefill_speedup_ceiling"

# --- M2: the prompt-reuse guard (warm-up golden + decode offset) ------------
# SEEDS fills the per-spec early-decode limits (M1, below) with measured-looking
# values, so an M2 case sees only its own failure.
SEEDS='c["seed_window_max_ratio_by_spec"] = {k: 1.5 for k in c["seed_window_max_ratio_by_spec"]}'
RECORDED='c["warmup_golden"] = {"r2_path": "correctness_prompts/%s/warmup-01.golden.json" % c["track_id"], "sha256": "ab" * 32, "bytes": 1234}; '"${SEEDS}"

# PENDING puts back the two organizer placeholders a fresh stamp carries
# (tools/new-track.sh), so the cases below see the unmeasured form.
PENDING='c["warmup_golden"] = {"r2_path": "MUSE-GLIMMER-30B-MLX-V1-WARMUP-PENDING-ORGANIZER", "sha256": "MUSE-GLIMMER-30B-MLX-V1-WARMUP-PENDING-ORGANIZER", "bytes": 0}; c["seed_window_max_ratio_by_spec"] = {k: "MUSE-GLIMMER-30B-MLX-V1-SEED-WINDOW-PENDING-ORGANIZER" for k in c["seed_window_max_ratio_by_spec"]}'

# 9. The shipped fixture: armed, warm-up pin and limits recorded -> it lints clean.
lint_case "" ""
if [[ "${LINT_RC}" == "0" && "$(grep -c '^FAIL  ' <<<"${LINT_OUT}")" == "0" ]]; then
  pass "shipped tree: armed, recorded warm-up pin and measured limits, no lint failure"
else
  fail "shipped tree: want exit 0 and no FAIL line; got exit ${LINT_RC}: $(grep '^FAIL  ' <<<"${LINT_OUT}" | tr '\n' ' ')"
fi

# 9b. The two organizer placeholders on an armed tree -> the TWO expected
#     failures, by name, so the tree cannot ship with either empty.
lint_case "${PENDING}" ""
expect_refusal "placeholder warm-up golden fails an armed tree" "guard keys: warmup_golden not recorded"
expect_refusal "placeholder early-decode limits fail an armed tree" "guard keys: seed_window_max_ratio not measured"
if [[ "$(grep -c '^FAIL  ' <<<"${LINT_OUT}")" == "2" ]]; then
  pass "placeholder tree: the two placeholders are the only lint failures"
else
  fail "placeholder tree: want exactly 2 FAIL lines, got: $(grep '^FAIL  ' <<<"${LINT_OUT}" | tr '\n' ' ')"
fi

# 10. A recorded pin (and measured limits) pass, and with them the whole
#     tree lints clean.
lint_case "${RECORDED}" ""
if [[ "${LINT_RC}" == "0" ]] && grep -qF "guard keys: prompt-reuse guard decode_golden_offset=1, warmup_golden" <<<"${LINT_OUT}"; then
  pass "recorded warm-up pin: the tree lints clean"
else
  fail "recorded warm-up pin: want exit 0 and the guard ok line; got exit ${LINT_RC}: $(grep -E '^FAIL' <<<"${LINT_OUT}" | tr '\n' ' ')"
fi

# 11. An UNARMED track (a fresh stamp) may carry the placeholder: reported, not failed.
lint_case "${PENDING}; c['official_scoring_enabled'] = False; ${SEEDS}" ""
expect_no_fail "unarmed track: the placeholder is reported, not failed" "warmup_golden"
grep -qF "allowed only while official_scoring_enabled is false" <<<"${LINT_OUT}" \
  && pass "unarmed track: the report line says why" \
  || fail "unarmed track: no report line for the placeholder"

# 12-14. decode_golden_offset must be an integer from 1 to count - 1.
lint_case "${RECORDED}; c['decode_golden_offset'] = 0" ""
expect_refusal "decode_golden_offset 0 refused" "decode_golden_offset = 0 must be an integer of at least 1"
lint_case "${RECORDED}; c['decode_golden_offset'] = len(c['live_goldens'])" ""
expect_refusal "decode_golden_offset = live golden count refused" "must be less than the live golden count"
lint_case "${RECORDED}; c['decode_golden_offset'] = True" ""
expect_refusal "decode_golden_offset as a boolean refused" "decode_golden_offset = True"

# 15. A second timed prefill pass would reuse the first-exposure prompt.
lint_case "${RECORDED}; c['official_prefill_timed_runs'] = 5" ""
expect_refusal "official_prefill_timed_runs 5 refused with the guard on" "official_prefill_timed_runs = 5"

# 16-17. The warm-up prompt must be one no timed pass measures.
lint_case "${RECORDED}; c['warmup_golden']['r2_path'] = c['timed_prompt_pool'][0]['r2_path']" ""
expect_refusal "warm-up r2_path naming a pool file refused" "names a timed_prompt_pool file"
lint_case "${RECORDED}; c['warmup_golden']['sha256'] = c['timed_prompt_pool'][1]['sha256']" ""
expect_refusal "warm-up sha256 equal to a pool pin refused" "equals a timed_prompt_pool pin"

# 18-19. Malformed pins are refused as not recorded.
lint_case "${RECORDED}; c['warmup_golden']['bytes'] = '1234'" ""
expect_refusal "warm-up bytes as a string refused" "warmup_golden not recorded"
lint_case "c['warmup_golden'] = 'warmup-01'; ${SEEDS}" ""
expect_refusal "warm-up pin that is not an object refused" "warmup_golden = 'warmup-01' must be an object"

# --- M1: the early-decode limit, one per declared spec ------------------------

# 20. The shipped map names exactly serial + dflash<d> for every permitted depth.
want_specs="$(jq -r '["serial"] + ([.mtp_head.permitted_draft_depths[]] | map("dflash\(.)")) | join(" ")' "${CONTRACT}")"
have_specs="$(jq -r '.seed_window_max_ratio_by_spec | keys_unsorted | join(" ")' "${CONTRACT}")"
if [[ "${have_specs}" == "${want_specs}" && "$(wc -w <<<"${have_specs}" | tr -d ' ')" == "8" ]]; then
  pass "shipped map holds the 8 declarable specs: ${have_specs}"
else
  fail "shipped map specs '${have_specs}', want '${want_specs}' (8)"
fi

# 21. Measured values: the guard reports the limits and fails nothing.
lint_case "${RECORDED}" ""
grep -qF "guard keys: early-decode limit per spec serial=1.5" <<<"${LINT_OUT}" \
  && pass "measured limits: the guard ok line names them" \
  || fail "measured limits: no early-decode ok line: $(grep -E '^(FAIL|ok).*early-decode' <<<"${LINT_OUT}")"

# 22. One placeholder left is enough to fail, by the same words.
lint_case "${RECORDED}; c['seed_window_max_ratio_by_spec']['dflash4'] = 'X-PENDING-ORGANIZER'" ""
expect_refusal "one unmeasured spec fails" "seed_window_max_ratio not measured (seed_window_max_ratio_by_spec dflash4='X-PENDING-ORGANIZER')"

# 23-25. A value benchd would refuse is not measured: 1.0, a boolean, a string number.
lint_case "${RECORDED}; c['seed_window_max_ratio_by_spec']['serial'] = 1.0" ""
expect_refusal "limit 1.0 refused" "seed_window_max_ratio not measured (seed_window_max_ratio_by_spec serial=1.0)"
lint_case "${RECORDED}; c['seed_window_max_ratio_by_spec']['dflash1'] = True" ""
expect_refusal "limit as a boolean refused" "dflash1=True"
lint_case "${RECORDED}; c['seed_window_max_ratio_by_spec']['dflash7'] = '1.5'" ""
expect_refusal "limit as a string refused" "dflash7='1.5'"

# 26-27. The map must cover every declarable spec, and nothing else.
lint_case "${RECORDED}; del c['seed_window_max_ratio_by_spec']['dflash3']" ""
expect_refusal "missing spec refused" "missing ['dflash3']"
lint_case "${RECORDED}; c['seed_window_max_ratio_by_spec']['dflash8'] = 1.5" ""
expect_refusal "undeclarable spec refused" "not declarable ['dflash8']"

# 28. The scalar beside the map is two sources for one limit.
lint_case "${RECORDED}; c['seed_window_max_ratio'] = 1.5" ""
expect_refusal "scalar beside the map refused" "declares both seed_window_max_ratio and seed_window_max_ratio_by_spec"

# 29. An armed track with no early-decode limit at all fails; an unarmed one reports it.
lint_case "${RECORDED}; del c['seed_window_max_ratio_by_spec']" ""
expect_refusal "armed track without the limit refused" "no early-decode limit declared"
lint_case "${RECORDED}; del c['seed_window_max_ratio_by_spec']; c['official_scoring_enabled'] = False" ""
expect_no_fail "unarmed track without the limit: reported, not failed" "early-decode"

# 30. Not an object.
lint_case "${RECORDED}; c['seed_window_max_ratio_by_spec'] = 1.5" ""
expect_refusal "map that is not an object refused" "seed_window_max_ratio_by_spec = 1.5 must be an object"

echo
if [[ "${failures}" -gt 0 ]]; then
  echo "test-lint-guard-keys.sh: ${failures} FAILED, ${passes} passed" >&2
  exit 1
fi
echo "test-lint-guard-keys.sh: all ${passes} cases passed"
