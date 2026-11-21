#!/usr/bin/env bash
#
# test-reference-stamp.sh -- ./setup.sh --check-reference-stamp runs the stamp
# test the read-only setup path uses, names the input that made a stamp stale,
# and prints the fix. tools/ranked-box-preflight.sh runs the same check before a
# job starts (tools/test-ranked-box-preflight-env.sh pins that wiring).
#
# Nothing here downloads, builds or hashes a checkpoint. Each case builds a
# two-file checkpoint and its pin file in a temporary directory, writes a stamp
# in the version 3 format setup.sh writes (write_reference_cache_lock), then
# changes one input and checks the result.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-reference-stamp.XXXXXX")"
trap 'chmod -R u+w "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT

failures=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; failures=$((failures + 1)); }

REPO="example/stamp-model"
REV="0123456789abcdef0123456789abcdef01234567"
CKPT="${WORK}/checkpoint"
PIN="${WORK}/pins.sha256"
STAMP="${CKPT}/.mlxfast-reference-cache.lock"

# The signature setup.sh's file_metadata_signature writes: BSD stat first,
# GNU stat when that fails. GNU stat can print to stdout before the BSD-style
# call fails, so the output of a failed first call is dropped.
signature() {
  local value
  if value="$(stat -f '%d:%i:%z:%Fm:%Fc:%v' "$1" 2>/dev/null)"; then
    printf '%s\n' "${value}"
  else
    stat -c '%d:%i:%s:%Y:%Z:0' "$1"
  fi
}

# make_checkpoint: two pinned files, the pin file, and a CURRENT stamp.
make_checkpoint() {
  rm -rf "${CKPT}"
  mkdir -p "${CKPT}"
  printf '{"model_type":"stamp-test"}\n' > "${CKPT}/config.json"
  printf 'weights' > "${CKPT}/model.safetensors"
  {
    echo "# Revision: ${REV}"
    for f in config.json model.safetensors; do
      printf '%s %s %s\n' "$(shasum -a 256 "${CKPT}/${f}" | awk '{print $1}')" "$(wc -c < "${CKPT}/${f}" | tr -d ' ')" "${f}"
    done
  } > "${PIN}"
  local bytes=0 f
  for f in config.json model.safetensors; do bytes=$((bytes + $(wc -c < "${CKPT}/${f}" | tr -d ' '))); done
  {
    echo "version=3"
    echo "model_repo=${REPO}"
    echo "revision=${REV}"
    echo "manifest_sha256=$(shasum -a 256 "${PIN}" | awk '{print $1}')"
    echo "file_count=2"
    echo "byte_count=${bytes}"
    echo "verified_at=2026-11-20T00:00:00Z"
    echo "--files--"
    for f in config.json model.safetensors; do
      printf '%s\t%s\t%s\n' "${f}" "$(wc -c < "${CKPT}/${f}" | tr -d ' ')" "$(signature "${CKPT}/${f}")"
    done
  } > "${STAMP}"
}

# check [ENV=VAL...]: run the stamp check from the repository root; sets rc, out.
check() {
  set +e
  out="$(cd "${REPO_ROOT}" && env -u MLXFAST_REFERENCE_HASH_VERIFY -u MLXFAST_REFERENCE_CACHE_LOCK_PATH \
    MLXFAST_REFERENCE_DIR="${CKPT}" \
    MLXFAST_REFERENCE_MANIFEST_PATH="${PIN}" \
    MLXFAST_REFERENCE_MODEL_REPO="${REPO}" \
    MLXFAST_REFERENCE_REVISION="${REV}" \
    "$@" ./setup.sh --check-reference-stamp 2>&1)"
  rc=$?
  set -e
}

expect_stale() {
  local name="$1" needle="$2"
  if [[ "${rc}" -eq 1 && "${out}" == *"absent or not current"* && "${out}" == *"${needle}"* \
        && "${out}" == *"Fix: as the account that owns the checkpoint, run MLXFAST_REFERENCE_DIR=${CKPT} ./setup.sh"* ]]; then
    pass "${name}"
  else
    fail "${name} (rc ${rc}): ${out}"
  fi
}

# --- 1. a current stamp passes, and the check builds nothing ----------------
make_checkpoint
check
if [[ "${rc}" -eq 0 && "${out}" == *"is current"* ]]; then
  pass "a current stamp passes"
else
  fail "a current stamp did not pass (rc ${rc}): ${out}"
fi
if [[ "${out}" != *"building"* && "${out}" != *"download"* ]]; then
  pass "the check builds and downloads nothing (no build line in its output)"
else
  fail "the check started a build: ${out}"
fi

# --- 2. a chmod after the stamp makes it stale (ctime) -----------------------
# This is what a box converge does when it makes the checkpoint read-only for
# the job account after the operator stamped it.
make_checkpoint
sleep 1
chmod 0444 "${CKPT}/model.safetensors"
check
expect_stale "a chmod after the stamp is named as a changed file signature" "the file signature of ${CKPT}/model.safetensors changed after the stamp"

# --- 3. a pin file change makes it stale ------------------------------------
make_checkpoint
printf '# a comment\n' >> "${PIN}"
check
expect_stale "a changed pin file is named by its sha256" "this checkout's ${PIN} is"

# --- 4. another model or revision makes it stale ----------------------------
make_checkpoint
check MLXFAST_REFERENCE_REVISION=fedcba9876543210fedcba9876543210fedcba98
expect_stale "another pinned revision is named" "the stamp is for revision '${REV}'"

# --- 5. no stamp, and hash verification off ---------------------------------
make_checkpoint
rm -f "${STAMP}"
check
expect_stale "an absent stamp is named" "there is no stamp at ${STAMP}"
make_checkpoint
check MLXFAST_REFERENCE_HASH_VERIFY=0
expect_stale "MLXFAST_REFERENCE_HASH_VERIFY=0 is named" "MLXFAST_REFERENCE_HASH_VERIFY is not 1"

# --- 6. a replaced file (new inode) makes it stale --------------------------
make_checkpoint
cp "${CKPT}/config.json" "${WORK}/config.copy"
mv "${WORK}/config.copy" "${CKPT}/config.json"
check
expect_stale "a file replaced by a copy is named" "the file signature of ${CKPT}/config.json changed"

if [[ "${failures}" -eq 0 ]]; then
  echo "test-reference-stamp.sh: all cases passed"
  exit 0
fi
echo "test-reference-stamp.sh: ${failures} case(s) failed" >&2
exit 1
