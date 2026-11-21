#!/usr/bin/env bash
#
# seed-window-limit.sh -- resolve the early-decode limit for one declared spec.
#
# SOURCE THIS FILE; do not run it. It defines one function:
#
#   seed_window_limit_resolve <contract> <spec> <work dir>
#
# <spec> is what tools/spec-declaration.sh describe prints: "serial" or
# "dflashN".
#
# WHAT THE LIMIT IS. benchd splits each leg's clock at the seed token: the seed
# prefill window (`free_decode_begin` until the seed token comes back) and the
# decode window. Decode work done early, inside the seed step, leaves the
# decode window and is not scored. benchd's `seed_window_max_ratio` stops
# that: for every control and candidate leg it divides the seed-window time
# per token by the timed prefill time per token and refuses the run when the
# ratio is above the limit (SEED-WINDOW-ABOVE-LIMIT) or cannot be measured
# (SEED-WINDOW-NOT-OBSERVED).
#
# ONE LIMIT PER DECLARED SPEC. The honest ratio can differ by spec (a dflash
# leg boots the drafter for its seed step), so the fixture holds one measured
# limit per spec in `seed_window_max_ratio_by_spec`
# ({"serial": x, "dflash1": x, ... "dflash7": x}).
# benchd reads one number, `seed_window_max_ratio`, and holds BOTH legs of a run
# to it. So this function writes a run contract: the fixture as jq renders it,
# plus `seed_window_max_ratio` set to the larger of the serial
# limit (the control leg is serial) and the declared spec's limit. benchd's
# sealed `metrics.contract_sha256` names that run contract; anyone can rebuild
# it from the fixture and the declared spec.
#
# RESULT. On return 0 the function sets two globals:
#   SEED_WINDOW_CONTRACT    the contract to hand benchd (the run contract, or
#                           the fixture itself when it declares no per-spec map)
#   SEED_WINDOW_MAX_RATIO   the limit benchd applies ("" when no map is declared)
#
# It prints the reason and returns 1 when the map is declared and is NOT
# MEASURED for this run (an organizer placeholder, a value that is not a finite
# number above 1.0, or no entry for the declared spec or for serial), when the
# fixture also carries a top-level `seed_window_max_ratio` (two sources for one
# limit), or when the run contract cannot be written. "Measured" is the same
# test as tools/lint-benchmark-manifest.py check 5d, which fails the tree at
# rest with the same words ("seed_window_max_ratio not measured").

seed_window_limit_resolve() {
  local contract="$1" spec="$2" work_dir="$3" declared scalar serial_v spec_v limit out
  SEED_WINDOW_CONTRACT="${contract}"
  SEED_WINDOW_MAX_RATIO=""

  declared="$(jq -r 'has("seed_window_max_ratio_by_spec")' "${contract}" 2>/dev/null)" || {
    echo "seed-window-limit.sh: cannot read the contract at ${contract}" >&2
    return 1
  }
  [[ "${declared}" == "true" ]] || return 0

  scalar="$(jq -r 'has("seed_window_max_ratio")' "${contract}")"
  if [[ "${scalar}" == "true" ]]; then
    echo "seed-window-limit.sh: the fixture declares both seed_window_max_ratio and seed_window_max_ratio_by_spec; keep the per-spec map only (this script sets benchd's key per run)." >&2
    return 1
  fi

  # A measured value is a JSON number, finite, above 1.0. jq prints "" for
  # anything else (a placeholder string, a boolean, null, a missing entry).
  serial_v="$(jq -r '.seed_window_max_ratio_by_spec.serial | if type == "number" and . > 1.0 and isinfinite == false then tostring else "" end' "${contract}")"
  spec_v="$(jq -r --arg s "${spec}" '.seed_window_max_ratio_by_spec[$s] | if type == "number" and . > 1.0 and isinfinite == false then tostring else "" end' "${contract}")"
  if [[ -z "${serial_v}" || -z "${spec_v}" ]]; then
    echo "seed-window-limit.sh: seed_window_max_ratio not measured (seed_window_max_ratio_by_spec.serial = $(jq -c '.seed_window_max_ratio_by_spec.serial' "${contract}"), seed_window_max_ratio_by_spec.${spec} = $(jq -c --arg s "${spec}" '.seed_window_max_ratio_by_spec[$s]' "${contract}"))." >&2
    echo "  The fixture declares the early-decode limit per spec but the value for this run is still a placeholder. Measure it from honest runs on the track's box and fill the fixture; benchd cannot hold the seed window without it." >&2
    return 1
  fi

  limit="$(jq -rn --argjson a "${serial_v}" --argjson b "${spec_v}" '[$a, $b] | max | tostring')"
  out="${work_dir}/run-contract.${spec}.json"
  if ! jq --argjson v "${limit}" '. + {seed_window_max_ratio: $v}' "${contract}" > "${out}"; then
    echo "seed-window-limit.sh: cannot write the run contract at ${out}" >&2
    return 1
  fi
  SEED_WINDOW_CONTRACT="${out}"
  SEED_WINDOW_MAX_RATIO="${limit}"
  return 0
}
