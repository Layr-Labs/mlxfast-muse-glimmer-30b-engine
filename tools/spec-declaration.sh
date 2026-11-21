#!/usr/bin/env bash
#
# spec-declaration.sh -- the SINGLE trusted source that reads the participant's
# speculative-decode DECLARATION and derives the decoder and the draft depth the
# ranked run requests from it. tools/muse-glimmer-30b-measure-and-score.sh resolves
# both THROUGH this one script, so the fail-closed validation and the derivation
# can never drift between the arm gate and the benchd request.
#
# NOT AN EDITABLE PATH. The DECLARATION (mtp-head.manifest.json) is editable --
# a submission EXPRESSES a value there -- but this DERIVATION is trusted, so a
# submission cannot rewrite how the value is interpreted or relax the envelope.
#
# HOW THE DECLARATION REACHES THE ENGINE ON THIS TRACK. The runtime worker is
# in-process, so the decoder and the draft depth ride the Engine Protocol wire
# per request. benchd sends the candidate spec on free_decode_begin and the
# engine echoes `effective_spec`, which the sealed score carries:
#
#   dflash  {"mode":"dflash","dflash":{"depth":N}}    (`benchd iterate --candidate-spec <that object>`)
#
# With no declaration, or with `enabled: false`, benchd sends no spec and the
# engine runs SERIAL (depth 0) -- the baseline validation's leg.
#
# DECLARATION SHAPE (mtp-head.manifest.json, optional key):
#
#   "spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": N }
#
#   decoder absent         => "dflash", this track's decoder
#   decoder in the list    => that decoder, depth from the contract envelope
#   enabled=false or N=0   => serial (no spec on the wire)
#   anything else          => REFUSED here, before any engine spawns
#
# THE DECLARABLE DECODERS ARE THE FIXTURE'S. The list is the contract fixture's
# `allowed_modes` without `serial` -- today `dflash` alone, the quantized DFlash
# drafter pinned in the fixture's `mtp_head` block. The Swift reader
# (Sources/MLXFastTrustedHarness/MTPHeadDeclaration.swift) reads the same list,
# so the two readers cannot disagree about which decoders a submission may
# name. The block keeps the name `mtp_head` because `mtp-head.manifest.json` is
# the declaration slot the harness reads. The permitted depth set is the
# fixture's `mtp_head.permitted_draft_depths`.
#
# VERBS
#   speculative  prints 1 when a depth is requested, else 0
#   decoder      prints the declared decoder (for example "dflash")
#   draft-len    prints the requested depth (0 when serial)
#   describe     prints "serial" or "<decoder>N", for example "dflash3"
#   validate     exits 0 when the declaration is well-formed and in-envelope
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${REPO_ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${REPO_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json}"

fail() {
  echo "spec-declaration.sh: REFUSING -- $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || fail "jq is required to read the declaration"

# --- read the declaration ---------------------------------------------------
# An absent manifest OR an absent `spec` block is SERIAL: the pure no-op. It is
# resolved WITHOUT requiring the file to exist, so the stock-repo derivation
# never depends on a spec block being present.
decoder="dflash"
enabled="false"
raw_tokens="0"
if [[ -f "${MANIFEST}" ]]; then
  jq -e . >/dev/null 2>&1 < "${MANIFEST}" || fail "mtp-head.manifest.json is not valid JSON"
  if [[ "$(jq -r 'has("spec")' "${MANIFEST}")" == "true" ]]; then
    [[ "$(jq -r '.spec | type' "${MANIFEST}")" == "object" ]] \
      || fail "the \"spec\" declaration must be an object with keys {decoder, enabled, num_speculative_tokens}"
    # A typo'd key must not silently read as its default: config drift is a
    # refusal, matching the sibling's rejectUnknownSpecKeys.
    unknown="$(jq -r '.spec | keys[] | select(. != "decoder" and . != "enabled" and . != "num_speculative_tokens")' "${MANIFEST}")"
    [[ -z "${unknown}" ]] \
      || fail "the \"spec\" declaration carries unknown key(s): $(printf '%s' "${unknown}" | tr '\n' ' '); allowed keys are decoder, enabled, num_speculative_tokens"
    if [[ "$(jq -r '.spec | has("decoder")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.decoder | type' "${MANIFEST}")" == "string" ]] \
        || fail "spec.decoder must be a string (got $(jq -r '.spec.decoder | type' "${MANIFEST}"))"
      decoder="$(jq -r '.spec.decoder' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("enabled")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.enabled | type' "${MANIFEST}")" == "boolean" ]] \
        || fail "spec.enabled must be a boolean (got $(jq -r '.spec.enabled | type' "${MANIFEST}"))"
      enabled="$(jq -r '.spec.enabled' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("num_speculative_tokens")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}")" == "number" ]] \
        || fail "spec.num_speculative_tokens must be an integer (got $(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}"))"
      raw_tokens="$(jq -r '.spec.num_speculative_tokens' "${MANIFEST}")"
    fi
  fi
fi

# --- the declarable decoders come from the fixture ---------------------------
# allowed_modes without "serial". An empty list is a broken contract, not a
# serial-only track: the fixture must name the decoders it allows.
[[ -f "${CONTRACT}" ]] || fail "the track contract fixture is missing: ${CONTRACT}"
declarable="$(jq -r '(.allowed_modes // [])[] | select(type == "string" and . != "serial")' "${CONTRACT}" 2>/dev/null)" \
  || fail "the track contract fixture's allowed_modes cannot be read: ${CONTRACT}"
[[ -n "${declarable}" ]] \
  || fail "the track contract fixture's allowed_modes names no speculative decoder: ${CONTRACT}"
if ! grep -qxF -- "${decoder}" <<<"${declarable}"; then
  fail "spec.decoder='${decoder}' is not a declarable decoder; this track declares $(printf '%s' "${declarable}" | paste -sd, - | sed 's/,/, /g') (the fixture's allowed_modes)"
fi

# --- resolve the decoder's envelope -----------------------------------------
# Each decoder has its OWN structural ceiling and its OWN contract block. The
# structural ceiling is the outer type/range guard (the knob accepts a declared
# integer, not a hardcoded value); the contract's permitted set is the tighter,
# AUTHORITATIVE one enforced below when the value is enabled. A decoder the
# fixture allows but this reader has no envelope for is refused, never served
# with a guessed one.
case "${decoder}" in
  dflash)
    SPEC_MAX_TOKENS=8
    CONTRACT_DEPTHS=".mtp_head.permitted_draft_depths"
    ;;
  *)
    fail "spec.decoder='${decoder}' is in the fixture's allowed_modes, but this reader has no draft-depth envelope for it"
    ;;
esac

# integer + structural 0..SPEC_MAX_TOKENS ceiling
printf '%s' "${raw_tokens}" | grep -Eq '^-?[0-9]+$' \
  || fail "spec.num_speculative_tokens must be an integer (got '${raw_tokens}')"
if (( raw_tokens < 0 || raw_tokens > SPEC_MAX_TOKENS )); then
  fail "spec.num_speculative_tokens=${raw_tokens} is outside the permitted range 0..${SPEC_MAX_TOKENS} for decoder ${decoder}"
fi

# --- derive the effective serve spec ----------------------------------------
# enabled:false OR num_speculative_tokens 0 => serial (the no-op). enabled:true
# with N>0 must be a CONTRACT-permitted depth for the declared decoder.
spec="0"
draft="0"
if [[ "${enabled}" == "true" && "${raw_tokens}" -gt 0 ]]; then
  # The fixture's permitted set is the authority. Enforce membership so the
  # serve can never boot a depth the scored envelope would refuse. When the
  # contract declares no such set, the structural ceiling above stands alone.
  if jq -e "${CONTRACT_DEPTHS} | arrays" >/dev/null 2>&1 < "${CONTRACT}"; then
    if [[ "$(jq -r --argjson n "${raw_tokens}" "(${CONTRACT_DEPTHS} | index(\$n)) != null" "${CONTRACT}")" != "true" ]]; then
      permitted="$(jq -r "${CONTRACT_DEPTHS} | map(tostring) | join(\", \")" "${CONTRACT}")"
      fail "spec.num_speculative_tokens=${raw_tokens} is not a contract-permitted ${decoder} draft depth (permitted_draft_depths: ${permitted})"
    fi
  fi
  spec="1"
  draft="${raw_tokens}"
fi

case "${1:-}" in
  speculative) echo "${spec}" ;;
  decoder)     echo "${decoder}" ;;
  draft-len)   echo "${draft}" ;;
  describe)    if [[ "${spec}" == "1" ]]; then echo "${decoder}${draft}"; else echo "serial"; fi ;;
  validate)    : ;;  # validation already ran above; a clean exit means valid
  *)
    echo "spec-declaration.sh: usage: spec-declaration.sh {speculative|decoder|draft-len|describe|validate}" >&2
    exit 2
    ;;
esac
