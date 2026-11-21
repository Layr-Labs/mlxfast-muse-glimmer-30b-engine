#!/usr/bin/env bash
#
# track-fixture.sh -- read one scalar value from the track contract fixture.
#
# SOURCE THIS FILE; do not run it. It defines one function:
#
#   track_fixture_value <dotted.key.path>
#
# It prints the value at the key path (for example `target.upstream_model_id`)
# and returns 0. It prints nothing and returns 1 when the key is absent, when
# the value is not a string, a number or a boolean, or when the fixture cannot
# be read. A caller decides what a missing key means; this reader never
# invents a default.
#
# WHY A SHARED READER. setup.sh and setup-mtp-head.sh used to carry their own
# copies of the target and drafter pins, and those copies fell behind the
# fixture. The fixture is the authority for the pins; these scripts now read
# it here instead of restating it.
#
# WHY plutil FIRST. setup.sh runs on a fresh Mac before jq or python3 can be
# assumed, and plutil ships with macOS. A float is printed in plutil's own form
# (5.0 prints as 5.000000). python3 is the fallback on other hosts.
#
# TRACK_FIXTURE_PATH overrides the fixture path (the tests use it).

TRACK_FIXTURE_PATH="${TRACK_FIXTURE_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/fixtures/muse_glimmer_30b_mlx_v1_track.json}"

track_fixture_value() {
  local key="$1"
  local kind

  [[ -n "${key}" && -r "${TRACK_FIXTURE_PATH}" ]] || return 1

  if command -v plutil >/dev/null 2>&1; then
    kind="$(plutil -type "${key}" "${TRACK_FIXTURE_PATH}" 2>/dev/null)" || return 1
    case "${kind}" in
      string|integer|float|bool) ;;
      *) return 1 ;;
    esac
    plutil -extract "${key}" raw -o - "${TRACK_FIXTURE_PATH}" 2>/dev/null || return 1
    return 0
  fi

  command -v python3 >/dev/null 2>&1 || return 1
  python3 -I - "${TRACK_FIXTURE_PATH}" "${key}" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as handle:
        node = json.load(handle)
except (OSError, ValueError):
    sys.exit(1)
for part in key.split("."):
    if isinstance(node, dict) and part in node:
        node = node[part]
    elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
        node = node[int(part)]
    else:
        sys.exit(1)
if isinstance(node, bool):
    print("true" if node else "false")
elif isinstance(node, (int, float, str)):
    print(node)
else:
    sys.exit(1)
PY
}

# track_fixture_floor_gib <margin GiB> <bytes key>...
#
# Prints the free-space floor, in whole GiB, for staging the artifacts whose
# byte counts the named keys hold: their sum rounded up to whole GiB, plus the
# margin. Returns 1 (prints nothing) when a key is absent or not a whole
# number of bytes, so a caller never measures against a guessed size.
track_fixture_floor_gib() {
  local margin_gib="$1"
  local total=0
  local key
  local bytes
  shift
  [[ "${margin_gib}" =~ ^[0-9]+$ && "$#" -gt 0 ]] || return 1
  for key in "$@"; do
    bytes="$(track_fixture_value "${key}")" || return 1
    [[ "${bytes}" =~ ^[0-9]+$ ]] || return 1
    total=$((total + bytes))
  done
  printf '%s\n' "$(( (total + 1073741823) / 1073741824 + margin_gib ))"
}
