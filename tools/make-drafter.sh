#!/usr/bin/env bash
#
# make-drafter.sh -- put the quantized DFlash drafter this track pins where a
# dflash leg loads it (<tree>/weights/drafter), and verify both of its files.
#
# THE DRAFTER IS PRODUCED, NOT DOWNLOADED. The track fixture's mtp_head block
# pins it twice:
#
#   mtp_head.source_artifact   the published BF16 assistant
#                              (mtp_head.upstream_model_id at
#                              mtp_head.upstream_revision): config.json and
#                              model.safetensors by sha256
#   mtp_head.artifact          the quantized drafter the pinned writer makes from
#                              it: config.json and model.safetensors by sha256,
#                              and its byte count
#
# TWO WAYS IN, ONE CHECK OUT.
#
#   --source <dir>    PRODUCE it. The BF16 assistant directory (./setup-mtp-head.sh
#                     downloads and verifies it). The make-dflash-drafter product
#                     of this package runs the pinned Vendor writer on it; that
#                     loader checks the source against its own BF16 pin.
#   --artifact <dir>  COPY an already produced drafter, for example a box
#                     operator's verified copy. No conversion runs.
#
# AN OFFICIAL RUN ONLY COPIES. --source builds make-dflash-drafter from this
# tree and runs it, outside the Seatbelt sandbox the other ranked steps use, so
# on an official run (RUNNER_ENVIRONMENT=self-hosted or
# MLXFAST_OFFICIAL_BENCHMARK_RUN=1, the same test as tools/sandboxed-cli.sh and
# tools/sandboxed-build.sh) it is refused, also in a dry run. There the only
# way in is --artifact: the operator-staged verified copy, checked against the
# pins below. Producing the drafter is the organizer's own setup step, run by
# hand on the box, never in a ranked job.
#
# Either way the result is written into a new directory beside --output, both
# files are checked against the fixture's mtp_head.artifact pins (sha256) and
# byte count, and only then is the directory moved to --output. An --output
# that already holds a drafter which passes the same check is left alone; one
# that fails it is refused, never overwritten.
#
# Usage:
#   tools/make-drafter.sh [--source <dir> | --artifact <dir>] [--output <dir>] [--dry-run]
#
#   --source <dir>    default MLXFAST_DRAFTER_SOURCE_DIR, else
#                     reference_weights/muse-glimmer-30b-assistant
#   --output <dir>    default MLXFAST_MTP_HEAD_DIR, else weights/drafter (the
#                     path tools/resident-up.sh --boot --spec dflash loads)
#   --dry-run         resolve the pins and the paths, check the inputs exist,
#                     print the plan, and stop. Nothing is built, converted,
#                     copied or written.
#
# Environment:
#   MLXFAST_MAKE_DRAFTER_BIN   the make-dflash-drafter binary. Default
#                              .build/release/make-dflash-drafter, built with
#                              `swift build -c release --product
#                              make-dflash-drafter` when absent. Metal loads
#                              mlx.metallib from the binary's own directory, so
#                              that directory must hold it (./setup.sh stages one
#                              in .build/release).
#   TRACK_FIXTURE_PATH         the track fixture (tools/track-fixture.sh).
#
# Exit: 0 the drafter is in place and verified (or --dry-run found every input);
#       1 a refusal or a failed step; 2 bad arguments.
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT_DIR}"
# shellcheck source=tools/track-fixture.sh
source "${ROOT_DIR}/tools/track-fixture.sh"

log() { printf 'make-drafter.sh: %s\n' "$*"; }
die() { printf 'make-drafter.sh: REFUSED: %s\n' "$*" >&2; exit 1; }
usage() {
  echo "usage: tools/make-drafter.sh [--source <dir> | --artifact <dir>] [--output <dir>] [--dry-run]" >&2
  exit 2
}

SOURCE_DIR=""
ARTIFACT_DIR=""
OUTPUT_DIR="${MLXFAST_MTP_HEAD_DIR:-weights/drafter}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) [[ $# -ge 2 ]] || usage; SOURCE_DIR="$2"; shift 2 ;;
    --artifact) [[ $# -ge 2 ]] || usage; ARTIFACT_DIR="$2"; shift 2 ;;
    --output) [[ $# -ge 2 ]] || usage; OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "make-drafter.sh: unknown argument '$1'" >&2; usage ;;
  esac
done
[[ -z "${SOURCE_DIR}" || -z "${ARTIFACT_DIR}" ]] || die "--source and --artifact are exclusive: produce the drafter, or copy a produced one"
OFFICIAL_RUN=0
if [[ "${RUNNER_ENVIRONMENT:-}" == "self-hosted" || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ]]; then
  OFFICIAL_RUN=1
fi
if [[ "${OFFICIAL_RUN}" == "1" && -z "${ARTIFACT_DIR}" ]]; then
  die "this is an official run (RUNNER_ENVIRONMENT=${RUNNER_ENVIRONMENT:-} MLXFAST_OFFICIAL_BENCHMARK_RUN=${MLXFAST_OFFICIAL_BENCHMARK_RUN:-}): it only copies the operator-staged verified drafter (--artifact <dir>). Producing it (--source) builds and runs code from this tree outside the sandbox; that is the organizer's own setup step, not a ranked job's"
fi
if [[ -z "${ARTIFACT_DIR}" ]]; then
  SOURCE_DIR="${SOURCE_DIR:-${MLXFAST_DRAFTER_SOURCE_DIR:-reference_weights/muse-glimmer-30b-assistant}}"
fi

# --- the pins, from the fixture only ------------------------------------------
pin() {
  track_fixture_value "$1" || die "${TRACK_FIXTURE_PATH} has no $1; the drafter is checked against the fixture's pins and nothing else"
}
WANT_WEIGHTS="$(pin mtp_head.artifact.weights_sha256)"
WANT_CONFIG="$(pin mtp_head.artifact.configuration_sha256)"
WANT_BYTES="$(pin mtp_head.artifact.bytes)"
SOURCE_REPO="$(pin mtp_head.upstream_model_id)"
SOURCE_REVISION="$(pin mtp_head.upstream_revision)"
SOURCE_WEIGHTS="$(pin mtp_head.source_artifact.weights_sha256)"
SOURCE_CONFIG="$(pin mtp_head.source_artifact.configuration_sha256)"
[[ "${WANT_WEIGHTS}" =~ ^[0-9a-f]{64}$ && "${WANT_CONFIG}" =~ ^[0-9a-f]{64}$ ]] \
  || die "the fixture's mtp_head.artifact sha256 pins are not 64 lower-case hex digits"
[[ "${WANT_BYTES}" =~ ^[0-9]+$ ]] || die "the fixture's mtp_head.artifact.bytes is not a whole number"

sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }

# verify_drafter DIR: both files are regular files with the pinned sha256, and
# their sizes add up to the pinned byte count. Prints the reason on failure.
verify_drafter() {
  local dir="$1" file want got total=0
  for file in config.json model.safetensors; do
    if [[ ! -f "${dir}/${file}" || -L "${dir}/${file}" ]]; then
      echo "${dir}/${file} is missing or is not a regular file"
      return 1
    fi
    total=$((total + $(wc -c < "${dir}/${file}" | tr -d ' ')))
  done
  if [[ "${total}" != "${WANT_BYTES}" ]]; then
    echo "${dir} holds ${total} bytes in config.json and model.safetensors; the fixture pins ${WANT_BYTES}"
    return 1
  fi
  for file in config.json model.safetensors; do
    if [[ "${file}" == "config.json" ]]; then want="${WANT_CONFIG}"; else want="${WANT_WEIGHTS}"; fi
    got="$(sha256_of "${dir}/${file}")"
    if [[ "${got}" != "${want}" ]]; then
      echo "${dir}/${file} has sha256 ${got}; the fixture pins ${want}"
      return 1
    fi
  done
}

MAKER_BIN="${MLXFAST_MAKE_DRAFTER_BIN:-${ROOT_DIR}/.build/release/make-dflash-drafter}"

log "drafter pins (fixture mtp_head.artifact): model.safetensors ${WANT_WEIGHTS}, config.json ${WANT_CONFIG}, ${WANT_BYTES} bytes"
if [[ "${OFFICIAL_RUN}" == "1" ]]; then
  log "official run: copy only (no build, no writer run)"
fi
if [[ -n "${ARTIFACT_DIR}" ]]; then
  log "plan: copy the produced drafter ${ARTIFACT_DIR} -> ${OUTPUT_DIR}, then verify both files"
else
  log "source pins (fixture mtp_head.source_artifact): ${SOURCE_REPO}@${SOURCE_REVISION}, model.safetensors ${SOURCE_WEIGHTS}, config.json ${SOURCE_CONFIG}"
  log "plan: ${MAKER_BIN} --source ${SOURCE_DIR} -> ${OUTPUT_DIR}, then verify both files"
fi

# --- an output that is already there ------------------------------------------
if [[ -e "${OUTPUT_DIR}" ]]; then
  [[ -d "${OUTPUT_DIR}" && ! -L "${OUTPUT_DIR}" ]] || die "${OUTPUT_DIR} exists and is not a directory"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry run: ${OUTPUT_DIR} exists; a real run checks it against the pins and keeps it when it passes"
    exit 0
  fi
  if reason="$(verify_drafter "${OUTPUT_DIR}")"; then
    log "${OUTPUT_DIR} already holds the pinned drafter (both files verified); nothing to do"
    exit 0
  fi
  die "${OUTPUT_DIR} exists and is not the pinned drafter (${reason}); move it aside and run again"
fi

# --- the inputs ---------------------------------------------------------------
if [[ -n "${ARTIFACT_DIR}" ]]; then
  [[ -d "${ARTIFACT_DIR}" ]] || die "--artifact ${ARTIFACT_DIR} is not a directory"
else
  for file in config.json model.safetensors; do
    [[ -f "${SOURCE_DIR}/${file}" ]] \
      || die "the BF16 source has no ${file} at ${SOURCE_DIR}; ./setup-mtp-head.sh downloads and verifies ${SOURCE_REPO}@${SOURCE_REVISION} there"
  done
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ -z "${ARTIFACT_DIR}" && ! -x "${MAKER_BIN}" ]]; then
    log "dry run: ${MAKER_BIN} is not built; a real run builds it with swift build -c release --product make-dflash-drafter"
  fi
  log "dry run: every input is present; nothing was built, converted, copied or written"
  exit 0
fi

# --- write into a new directory beside the output, verify, then move ----------
mkdir -p "$(dirname "${OUTPUT_DIR}")"
STAGING="$(mktemp -d "${OUTPUT_DIR%/}.staging.XXXXXX")"
trap 'rm -rf "${STAGING}"' EXIT

if [[ -n "${ARTIFACT_DIR}" ]]; then
  cp "${ARTIFACT_DIR}/config.json" "${ARTIFACT_DIR}/model.safetensors" "${STAGING}/"
  if [[ -f "${ARTIFACT_DIR}/manifest.json" ]]; then
    cp "${ARTIFACT_DIR}/manifest.json" "${STAGING}/"
  fi
else
  if [[ ! -x "${MAKER_BIN}" ]]; then
    [[ -z "${MLXFAST_MAKE_DRAFTER_BIN:-}" ]] || die "MLXFAST_MAKE_DRAFTER_BIN=${MAKER_BIN} is not executable"
    log "building make-dflash-drafter (swift build -c release --product make-dflash-drafter)"
    swift build -c release --force-resolved-versions --product make-dflash-drafter
  fi
  [[ -f "$(dirname "${MAKER_BIN}")/mlx.metallib" ]] \
    || die "$(dirname "${MAKER_BIN}") has no mlx.metallib, and Metal loads it from the binary's own directory; run ./setup.sh first (it stages mlx.metallib in .build/release)"
  "${MAKER_BIN}" --source "${SOURCE_DIR}" --output "${STAGING}"
fi

reason="$(verify_drafter "${STAGING}")" || die "the drafter written for ${OUTPUT_DIR} fails its pins: ${reason}"
mv "${STAGING}" "${OUTPUT_DIR}"
trap - EXIT
log "the pinned drafter is in place at ${OUTPUT_DIR} (model.safetensors ${WANT_WEIGHTS}, config.json ${WANT_CONFIG})"
log "a dflash leg loads it with: tools/resident-up.sh --boot --spec dflash --draft-len <N> (--drafter ${OUTPUT_DIR})"
