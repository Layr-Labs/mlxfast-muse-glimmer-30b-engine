#!/usr/bin/env bash
#
# warmup-golden.sh -- resolve the track's never-scored warm-up golden.
#
# SOURCE THIS FILE; do not run it. It defines one function:
#
#   warmup_golden_resolve <contract> <golden dir>
#
# WHAT THE WARM-UP GOLDEN IS. The fixture's `warmup_golden` names one extra
# prompt that benchd runs for both warm-up forms of every paired leg and never
# scores. With it (and `decode_golden_offset`), no prompt the timed prefill
# measures was seen by the engine before in the same leg, so a prefill memo or
# a replay of the warm-up answer has nothing to hit. benchd refuses a paired
# run or a calibration that declares it and is not given it
# (WARMUP-GOLDEN-REQUIRED), so both callers -- the ranked measurement
# (tools/muse-glimmer-30b-measure-and-score.sh) and the box calibration
# (tools/calibrate-box.sh) -- resolve it here, the same way.
#
# RESULT. On return 0 the function sets three globals:
#   WARMUP_GOLDEN_PATH     the staged file in <golden dir>, by the basename of
#                          the pin's r2_path
#   WARMUP_GOLDEN_SHA256   the pin's sha256
#   WARMUP_GOLDEN_BYTES    the pin's byte count
# All three are empty when the contract declares no warmup_golden: the caller
# then passes no flag, and benchd keeps the legacy same-prompt warm-up.
#
# It prints the reason and returns 1 when the contract declares a warm-up
# golden that is NOT RECORDED (the organizer placeholder, or any pin that is
# not a 64-hex sha256, a positive byte count and an r2_path under
# correctness_prompts/), or when the recorded file is not staged. "Recorded"
# is the same test as tools/lint-benchmark-manifest.py check 5d, which fails
# the tree at rest with the same words.
#
# It does not hash the file: benchd checks the bytes against the pin itself,
# and tools/ranked-box-preflight.sh checks them before the job reaches benchd.

warmup_golden_resolve() {
  local contract="$1" golden_dir="$2" declared r2_path sha bytes base
  WARMUP_GOLDEN_PATH=""
  WARMUP_GOLDEN_SHA256=""
  WARMUP_GOLDEN_BYTES=""

  declared="$(jq -r 'has("warmup_golden")' "${contract}" 2>/dev/null)" || {
    echo "warmup-golden.sh: cannot read the contract at ${contract}" >&2
    return 1
  }
  [[ "${declared}" == "true" ]] || return 0

  r2_path="$(jq -r '.warmup_golden.r2_path // "" | tostring' "${contract}")"
  sha="$(jq -r '.warmup_golden.sha256 // "" | tostring' "${contract}")"
  bytes="$(jq -r '.warmup_golden.bytes // "" | tostring' "${contract}")"
  if ! printf '%s' "${sha}" | grep -Eq '^[0-9a-f]{64}$' \
     || ! printf '%s' "${bytes}" | grep -Eq '^[1-9][0-9]*$' \
     || ! printf '%s' "${r2_path}" | grep -Eq '^correctness_prompts/[^/]+/[^/]+\.golden\.json$'; then
    echo "warmup-golden.sh: warmup_golden not recorded (r2_path '${r2_path}', sha256 '${sha}', bytes '${bytes}')." >&2
    echo "  The fixture declares the never-scored warm-up golden but its pin is still a placeholder. Record it on the track's box, publish it at its r2_path, fill the pin, and stage it into the golden directory; benchd refuses a paired run or a calibration without it." >&2
    return 1
  fi

  base="${r2_path##*/}"
  if [[ -z "${golden_dir}" || ! -f "${golden_dir}/${base}" ]]; then
    echo "warmup-golden.sh: the warm-up golden is not staged at ${golden_dir:-<unset>}/${base} (warmup_golden.r2_path ${r2_path})." >&2
    return 1
  fi
  WARMUP_GOLDEN_PATH="${golden_dir}/${base}"
  WARMUP_GOLDEN_SHA256="${sha}"
  WARMUP_GOLDEN_BYTES="${bytes}"
  return 0
}
