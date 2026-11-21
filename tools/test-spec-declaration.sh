#!/usr/bin/env bash
# Unit test for tools/spec-declaration.sh. Offline, no GPU, no serve -- it drives
# the trusted declaration->serve-spec deriver against the REAL contract fixture
# with a series of throwaway manifests, so the envelope it enforces is the one
# that actually ships.
#
# WHAT THIS PROVES, and it is the part worth having: the envelope REFUSALS. The
# decoder and the draft depth become benchd's candidate-spec request, and this
# script is the one trusted bridge that turns the participant's declaration into
# that request -- so an enabled depth outside the declared decoder's
# permitted_draft_depths must be REFUSED here, before any engine spawns. The
# accept cases pin every permitted depth; the refuse cases pin the first depth
# above the contract set and the first depth above the structural ceiling. If a
# future fixture edit ever WIDENS mtp_head.permitted_draft_depths past 7, the
# matching case below turns this test red -- that is the tripwire.
#
# THE DECODER LIST IS THE FIXTURE'S. The declarable decoders are the fixture's
# allowed_modes without serial -- today `dflash` alone. Any other name,
# including the older `mtp` slot spelling, is refused by name, which is the
# answer the Swift reader gives too.
#
# Usage: tools/test-spec-declaration.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DECL="${SCRIPT_DIR}/tools/spec-declaration.sh"
CONTRACT="${SCRIPT_DIR}/fixtures/muse_glimmer_30b_mlx_v1_track.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fails=0
mani=0

# Write a throwaway manifest and echo its path.
manifest() {
  mani=$((mani + 1))
  local path="${WORK}/manifest-${mani}.json"
  printf '%s' "$1" > "${path}"
  printf '%s' "${path}"
}

# expect_ok <label> <manifest-json> <subcommand> <expected-stdout>
expect_ok() {
  local label="$1" json="$2" sub="$3" want="$4" got rc
  got="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" SPEC_DECLARATION_CONTRACT="${CONTRACT}" \
    "${DECL}" "${sub}" 2>/dev/null)"
  rc=$?
  if [[ ${rc} -eq 0 && "${got}" == "${want}" ]]; then
    echo "PASS ${label} (${sub} => ${got})"
  else
    echo "FAIL ${label}: ${sub} rc=${rc} got='${got}' want rc=0 '${want}'"
    fails=$((fails + 1))
  fi
}

# expect_refuse <label> <manifest-json> <needle-in-stderr>
expect_refuse() {
  local label="$1" json="$2" needle="$3" err rc
  err="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" SPEC_DECLARATION_CONTRACT="${CONTRACT}" \
    "${DECL}" draft-len 2>&1 >/dev/null)"
  rc=$?
  if [[ ${rc} -ne 0 && "${err}" == *"${needle}"* ]]; then
    echo "PASS ${label} (refused: ${err##*REFUSING -- })"
  else
    echo "FAIL ${label}: expected refusal containing '${needle}', got rc=${rc} err='${err}'"
    fails=$((fails + 1))
  fi
}

# --- the contract this test pins --------------------------------------------
# The whole point is to read the SHIPPING envelope, not a copy. Assert it up
# front so a reader knows exactly what the accept/refuse split below rests on.
permitted="$(jq -c '.mtp_head.permitted_draft_depths' "${CONTRACT}")"
if [[ "${permitted}" != "[1,2,3,4,5,6,7]" ]]; then
  echo "FAIL contract envelope: mtp_head.permitted_draft_depths is ${permitted}, expected [1,2,3,4,5,6,7] -- the track's draft-depth envelope changed; update this test deliberately or revert the fixture"
  fails=$((fails + 1))
fi
if [[ "$(jq -r '.mtp_head.decoder // ""' "${CONTRACT}")" != "dflash" ]]; then
  echo "FAIL contract envelope: mtp_head.decoder is not \"dflash\"; this track declares the DFlash drafter"
  fails=$((fails + 1))
fi

# --- serial (the no-op) -----------------------------------------------------
expect_ok  "absent-spec-is-serial"        '{}' describe serial
expect_ok  "disabled-is-serial"           '{"spec":{"enabled":false,"num_speculative_tokens":0}}' describe serial
expect_ok  "enabled-zero-is-serial"       '{"spec":{"enabled":true,"num_speculative_tokens":0}}'  describe serial

# --- the declarable decoders are the fixture's allowed_modes minus serial ---
expect_ok  "absent-decoder-is-dflash"     '{}' decoder dflash
expect_ok  "explicit-dflash-decoder"     '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":7}}' decoder  dflash
expect_ok  "explicit-dflash-describe"    '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":7}}' describe dflash7
expect_ok  "dflash3-speculative"         '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":3}}' speculative 1
expect_ok  "dflash-disabled-is-serial"   '{"spec":{"decoder":"dflash","enabled":false,"num_speculative_tokens":0}}' describe serial
# The fixture's allowed_modes is [serial, dflash], so the older `mtp` slot name
# is refused by name -- the same answer the Swift reader gives.
expect_refuse "mtp-decoder-refused"      '{"spec":{"decoder":"mtp","enabled":true,"num_speculative_tokens":3}}' "spec.decoder='mtp' is not a declarable decoder; this track declares dflash"
expect_refuse "mtp-disabled-refused"     '{"spec":{"decoder":"mtp","enabled":false,"num_speculative_tokens":0}}' "is not a declarable decoder"
expect_refuse "unknown-decoder-refused"  '{"spec":{"decoder":"dspark","enabled":true,"num_speculative_tokens":7}}' "this track declares dflash (the fixture's allowed_modes)"
expect_refuse "serial-is-not-a-decoder"  '{"spec":{"decoder":"serial","enabled":true,"num_speculative_tokens":1}}' "is not a declarable decoder"

# --- the seven permitted dflash depths --------------------------------------
for depth in 1 2 3 4 5 6 7; do
  expect_ok "dflash${depth}-permitted" "{\"spec\":{\"decoder\":\"dflash\",\"enabled\":true,\"num_speculative_tokens\":${depth}}}" describe "dflash${depth}"
  expect_ok "dflash${depth}-draftlen"  "{\"spec\":{\"decoder\":\"dflash\",\"enabled\":true,\"num_speculative_tokens\":${depth}}}" draft-len "${depth}"
done

# --- the restriction: depths ABOVE 7 are refused, never clamped -------------
# 8 is inside the structural 0..8 ceiling but OUTSIDE permitted_draft_depths,
# so it must be refused by the contract-membership check. This is the tripwire
# for "cap the drafter at 7": widening permitted_draft_depths reddens it.
expect_refuse "dflash8-refused"  '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":8}}' "not a contract-permitted dflash draft depth"
# 9 is beyond the structural ceiling and refuses there, before the membership check.
expect_refuse "dflash9-refused"  '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":9}}' "outside the permitted range 0..8"

# --- the structural ceiling ------------------------------------------------
expect_refuse "dflash16-refused" '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":16}}' "outside the permitted range 0..8 for decoder dflash"

# --- the list follows the fixture, not a copy in the script -----------------
# A fixture that allows a decoder this reader has no envelope for is refused
# by name; a fixture that allows no decoder at all is refused as broken.
jq '.allowed_modes = ["serial", "dflash", "eagle"]' "${CONTRACT}" > "${WORK}/contract-eagle.json"
err="$(SPEC_DECLARATION_MANIFEST="$(manifest '{"spec":{"decoder":"eagle","enabled":true,"num_speculative_tokens":1}}')" \
  SPEC_DECLARATION_CONTRACT="${WORK}/contract-eagle.json" "${DECL}" draft-len 2>&1 >/dev/null)"
if [[ "${err}" == *"no draft-depth envelope"* ]]; then
  echo "PASS fixture-decoder-without-envelope-refused"
else
  echo "FAIL fixture-decoder-without-envelope-refused: err='${err}'"; fails=$((fails + 1))
fi
jq '.allowed_modes = ["serial"]' "${CONTRACT}" > "${WORK}/contract-serial-only.json"
err="$(SPEC_DECLARATION_MANIFEST="$(manifest '{}')" \
  SPEC_DECLARATION_CONTRACT="${WORK}/contract-serial-only.json" "${DECL}" describe 2>&1 >/dev/null)"
if [[ "${err}" == *"names no speculative decoder"* ]]; then
  echo "PASS fixture-without-decoder-refused"
else
  echo "FAIL fixture-without-decoder-refused: err='${err}'"; fails=$((fails + 1))
fi

# --- malformed declarations refuse ------------------------------------------
expect_refuse "negative-refused"    '{"spec":{"enabled":true,"num_speculative_tokens":-1}}' "outside the permitted range 0..8"
expect_refuse "non-integer-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1.5}}' "must be an integer"
expect_refuse "unknown-key-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1,"foo":1}}' "unknown key"
expect_refuse "bad-enabled-refused" '{"spec":{"enabled":"yes","num_speculative_tokens":1}}' "must be a boolean"
expect_refuse "bad-decoder-refused" '{"spec":{"decoder":"dspark","enabled":true,"num_speculative_tokens":1}}' "is not a declarable decoder"
expect_refuse "bad-decoder-type-refused" '{"spec":{"decoder":3,"enabled":true,"num_speculative_tokens":1}}' "spec.decoder must be a string"

if [[ ${fails} -eq 0 ]]; then
  echo "OK: all spec-declaration envelope cases passed"
  exit 0
fi
echo "FAILED: ${fails} case(s)"
exit 1
