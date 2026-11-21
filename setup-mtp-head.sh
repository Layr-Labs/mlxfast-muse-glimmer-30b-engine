#!/usr/bin/env bash
# Provision the organizer-pinned drafter slot (the file and variable names keep
# the older MTP HEAD spelling) for track muse-glimmer-30b-mlx-v1.
#
# This script provisions the drafter slot and nothing else. The TARGET is
# nvidia/Muse-Glimmer-30B-NVFP4, which ./setup.sh downloads and verifies against
# the target manifest the contract fixture names (target.manifest_path), so
# there is no second target download here and no way for this script to point
# the track at a different checkpoint.
#
# THE DRAFTER IS PRODUCED FROM A PINNED SOURCE. The engine loads the quantized
# DFlash artifact the fixture pins in mtp_head.artifact. Nobody publishes that
# artifact; the pinned writer makes it from the BF16 assistant the fixture names
# in mtp_head.upstream_model_id at mtp_head.upstream_revision, whose two files
# fixtures/reference_bonsai2_9b_mtp_head.sha256 pins (the same sha256 values as
# mtp_head.source_artifact). So this script does two things:
#
#   1. download and verify the BF16 source into MLXFAST_DRAFTER_SOURCE_DIR
#      (default reference_weights/muse-glimmer-30b-assistant), through
#      ./setup.sh's downloader;
#   2. run tools/make-drafter.sh, which writes the quantized drafter into
#      MLXFAST_MTP_HEAD_DIR (default weights/drafter, the path a dflash leg
#      loads) and checks both of its files against mtp_head.artifact.
#
# When the fixture names no source, this script REFUSES by key name and
# downloads nothing. MLXFAST_MTP_HEAD_REPO and MLXFAST_MTP_HEAD_REVISION still
# name an operator-staged source explicitly.
#
# WHY THIS IS A WRAPPER AND NOT A DOWNLOADER. setup.sh already holds the
# resumable, hash-verified, stall-detecting download logic, and it is fully
# parameterised by environment (MLXFAST_REFERENCE_MODEL_REPO / _REVISION /
# _MANIFEST_PATH / _BASE_URL / _CACHE_DIR / _DIR / _COMPAT_LINK). This script
# points those at the head and delegates. One downloader, one verification
# path, one set of stall and resume semantics.
#
# The delegated run does NOT rebuild the Swift products: it passes
# MLXFAST_SKIP_SWIFT_BUILD=1, so an existing pair is reused. Nothing about the
# products can have changed between the two commands. The tool-installing legs
# and the metallib build are skipped for the same reason in the other
# direction: ./setup.sh owns those and this script must not mutate global state
# a second time.
#
# ./setup.sh runs this script for you as its last step. Run it by hand only to
# re-verify the drafter, or after MLXFAST_SKIP_MTP_HEAD=1 skipped it. Step 2
# needs .build/release/mlx.metallib, which ./setup.sh stages before it runs
# this script.
#
# Usage: ./setup-mtp-head.sh
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="muse-glimmer-30b-mlx-v1"
# shellcheck source=tools/track-fixture.sh
source "${ROOT_DIR}/tools/track-fixture.sh"
HEAD_REPO_KEY="mtp_head.upstream_model_id"
HEAD_REVISION_KEY="mtp_head.upstream_revision"
HEAD_REPO="${MLXFAST_MTP_HEAD_REPO:-$(track_fixture_value "${HEAD_REPO_KEY}" || true)}"
HEAD_REVISION="${MLXFAST_MTP_HEAD_REVISION:-$(track_fixture_value "${HEAD_REVISION_KEY}" || true)}"
HEAD_MANIFEST="${MLXFAST_MTP_HEAD_MANIFEST_PATH:-fixtures/reference_bonsai2_9b_mtp_head.sha256}"
SOURCE_DIR="${MLXFAST_DRAFTER_SOURCE_DIR:-reference_weights/muse-glimmer-30b-assistant}"
HEAD_DIR="${MLXFAST_MTP_HEAD_DIR:-weights/drafter}"

if [[ -z "${HEAD_REPO}" || -z "${HEAD_REVISION}" ]]; then
  echo "setup-mtp-head.sh: no drafter source is pinned: ${TRACK_FIXTURE_PATH} has no ${HEAD_REPO_KEY} / ${HEAD_REVISION_KEY}, and MLXFAST_MTP_HEAD_REPO / MLXFAST_MTP_HEAD_REVISION are unset. The DFlash drafter is produced from that source; nothing is downloaded. Skip this step with MLXFAST_SKIP_MTP_HEAD=1." >&2
  exit 1
fi

# The free-space floor: the BF16 source and the drafter written from it, by
# their fixture byte counts, rounded up to whole GiB, plus the same 2 GiB
# staging margin setup.sh uses for the target.
HEAD_MIN_FREE_GIB="${MLXFAST_MTP_HEAD_MIN_FREE_GIB:-$(track_fixture_floor_gib 2 mtp_head.source_artifact.bytes mtp_head.artifact.bytes || true)}"
[[ -n "${HEAD_MIN_FREE_GIB}" ]] || {
  echo "setup-mtp-head.sh: ${TRACK_FIXTURE_PATH} has no whole-byte mtp_head.source_artifact.bytes / mtp_head.artifact.bytes, so the drafter's free-space floor cannot be computed; set MLXFAST_MTP_HEAD_MIN_FREE_GIB" >&2
  exit 1
}
echo "setup-mtp-head.sh: drafter free-space floor ${HEAD_MIN_FREE_GIB} GiB"

[[ -f "${HEAD_MANIFEST}" ]] || {
  echo "setup-mtp-head.sh: pin file is missing: ${HEAD_MANIFEST}" >&2
  exit 1
}

echo "setup-mtp-head.sh: track ${TRACK_ID}; drafter source ${HEAD_REPO}@${HEAD_REVISION} -> ${SOURCE_DIR}; drafter -> ${HEAD_DIR}"

# Step 1. The source has no compatibility link of its own: nothing resolves it
# by a legacy path. Point the link at the source directory so setup.sh does not
# create or check one for the target while provisioning the source.
MLXFAST_SKIP_SWIFT_BUILD=1 \
MLXFAST_SKIP_MACMON_INSTALL=1 \
MLXFAST_SKIP_MLX_METALLIB=1 \
MLXFAST_SKIP_MTP_HEAD=1 \
MLXFAST_REFERENCE_MODEL_REPO="${HEAD_REPO}" \
MLXFAST_REFERENCE_REVISION="${HEAD_REVISION}" \
MLXFAST_REFERENCE_MANIFEST_PATH="${HEAD_MANIFEST}" \
MLXFAST_REFERENCE_DIR="${SOURCE_DIR}" \
MLXFAST_REFERENCE_COMPAT_LINK="${SOURCE_DIR}" \
MLXFAST_REFERENCE_MIN_FREE_GIB="${HEAD_MIN_FREE_GIB}" \
  ./setup.sh "$@"

echo "setup-mtp-head.sh: the drafter source is staged and verified at ${SOURCE_DIR}"

# Step 2. Write the quantized drafter and check it against mtp_head.artifact.
MLXFAST_MTP_HEAD_DIR="${HEAD_DIR}" tools/make-drafter.sh --source "${SOURCE_DIR}" --output "${HEAD_DIR}"
echo "setup-mtp-head.sh: the engine reads it with --drafter ${HEAD_DIR}"
