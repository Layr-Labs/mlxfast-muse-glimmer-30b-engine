#!/usr/bin/env bash
#
# test-setup-pins.sh -- setup.sh and setup-mtp-head.sh take their default pins
# from the track fixture, and refuse by name when the fixture names none.
#
# Nothing here downloads or builds. Every case either prints the help text
# (which resolves the pins and stops) or stops at a refusal that runs before
# the first download.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
FIXTURE="${REPO_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-setup-pins.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

command -v jq >/dev/null 2>&1 || { echo "test-setup-pins.sh: jq is required" >&2; exit 1; }

failures=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; failures=$((failures + 1)); }

# The whole environment a developer could have set is cleared, so each case
# sees only what it sets itself.
clean_env() {
  env -u MLXFAST_REFERENCE_MODEL_REPO -u MLXFAST_REFERENCE_REVISION \
      -u MLXFAST_REFERENCE_MIN_FREE_GIB -u MLXFAST_MTP_HEAD_REPO \
      -u MLXFAST_MTP_HEAD_REVISION -u TRACK_FIXTURE_PATH "$@"
}

want_repo="$(jq -r '.target.upstream_model_id' "${FIXTURE}")"
want_rev="$(jq -r '.target.upstream_revision' "${FIXTURE}")"

# --- 1. the default target pin is the fixture's ------------------------------
help_out="$(cd "${REPO_ROOT}" && clean_env ./setup.sh --help)"
if grep -qF "${want_repo}@${want_rev}" <<<"${help_out}"; then
  pass "setup.sh default target is the fixture pin (${want_repo}@${want_rev})"
else
  fail "setup.sh default target is not the fixture pin"
fi
if grep -q 'PENDING-ORGANIZER' <<<"${help_out}"; then
  fail "setup.sh help still names the pending sentinel"
else
  pass "setup.sh help names no pending sentinel"
fi

# --- 2. an override still wins ------------------------------------------------
help_out="$(cd "${REPO_ROOT}" && clean_env MLXFAST_REFERENCE_MODEL_REPO=org/other MLXFAST_REFERENCE_REVISION=abc123 ./setup.sh --help)"
if grep -qF "org/other@abc123" <<<"${help_out}"; then
  pass "MLXFAST_REFERENCE_MODEL_REPO / _REVISION override the fixture"
else
  fail "the environment override did not win over the fixture"
fi

# --- 3. a fixture with no target pin is refused by name -----------------------
jq 'del(.target.upstream_model_id, .target.upstream_revision)' "${FIXTURE}" > "${WORK}/no-target.json"
set +e
refuse_out="$(cd "${REPO_ROOT}" && clean_env TRACK_FIXTURE_PATH="${WORK}/no-target.json" \
  MLXFAST_SKIP_MACMON_INSTALL=1 ./setup.sh 2>&1)"
refuse_rc=$?
set -e
if [[ "${refuse_rc}" -ne 0 ]] && grep -q "no reference checkpoint is pinned" <<<"${refuse_out}" \
   && grep -q "target.upstream_model_id" <<<"${refuse_out}"; then
  pass "a fixture with no target pin is refused by name before any download"
else
  fail "a fixture with no target pin was not refused by name (rc ${refuse_rc}): $(tail -3 <<<"${refuse_out}")"
fi

# --- 4. the drafter source: a fixture with no source pin is refused ----------
# The drafter is produced from mtp_head.upstream_model_id at
# mtp_head.upstream_revision. A fixture without them is refused by key name
# before anything is downloaded.
jq 'del(.mtp_head.upstream_model_id) | del(.mtp_head.upstream_revision)' "${FIXTURE}" > "${WORK}/no-source.json"
set +e
head_out="$(cd "${REPO_ROOT}" && clean_env TRACK_FIXTURE_PATH="${WORK}/no-source.json" ./setup-mtp-head.sh 2>&1)"
head_rc=$?
set -e
if [[ "${head_rc}" -ne 0 ]] && grep -q "mtp_head.upstream_model_id" <<<"${head_out}" \
   && ! grep -q "downloading" <<<"${head_out}"; then
  pass "setup-mtp-head.sh refuses by fixture key when no drafter source is pinned"
else
  fail "setup-mtp-head.sh did not refuse by key (rc ${head_rc}): $(tail -3 <<<"${head_out}")"
fi
if grep -q 'PENDING-ORGANIZER' <<<"${head_out}"; then
  fail "setup-mtp-head.sh still resolves the pending sentinel"
else
  pass "setup-mtp-head.sh resolves no pending sentinel"
fi

# --- 5. the shipped fixture's source pin is used ------------------------------
# A missing pin file stops the script right after it resolves the pin, so the
# case proves the pin was read without starting a download.
want_source="$(jq -r '.mtp_head.upstream_model_id + "@" + .mtp_head.upstream_revision' "${FIXTURE}")"
set +e
head_out="$(cd "${REPO_ROOT}" && clean_env MLXFAST_MTP_HEAD_MANIFEST_PATH="${WORK}/absent.sha256" ./setup-mtp-head.sh 2>&1)"
set -e
if grep -q "pin file is missing" <<<"${head_out}" && ! grep -q "no drafter source" <<<"${head_out}"; then
  pass "setup-mtp-head.sh takes the drafter source pin from the shipped fixture (${want_source})"
else
  fail "setup-mtp-head.sh did not take the fixture source pin: $(tail -3 <<<"${head_out}")"
fi
if grep -q "pin file is missing" <<<"${head_out}" && [[ -n "$(sed -n '/^[0-9a-f]\{64\} /p' "${REPO_ROOT}/fixtures/reference_bonsai2_9b_mtp_head.sha256")" ]]; then
  pass "the drafter pin file pins records (the shipped setup-mtp-head.sh no longer refuses for want of a pin)"
else
  fail "the drafter pin file pins no record"
fi

# --- 6. the free-space floors cover what each step stages, plus a margin ------
target_bytes="$(jq -r '.target.raw_checkpoint_total_size_bytes' "${FIXTURE}")"
source_bytes="$(jq -r '.mtp_head.source_artifact.bytes' "${FIXTURE}")"
drafter_bytes="$(jq -r '.mtp_head.artifact.bytes' "${FIXTURE}")"
gib=1073741824
want_floor=$(( (target_bytes + source_bytes + drafter_bytes + gib - 1) / gib + 2 ))
help_out="$(cd "${REPO_ROOT}" && clean_env ./setup.sh --help)"
got_floor="$(awk '/MLXFAST_REFERENCE_MIN_FREE_GIB/{getline; print $2; exit}' <<<"${help_out}")"
if [[ "${got_floor}" == "${want_floor}" ]] && (( got_floor * gib >= target_bytes + source_bytes + drafter_bytes )); then
  pass "setup.sh free-space floor is ${got_floor} GiB (target ${target_bytes} + drafter source ${source_bytes} + drafter ${drafter_bytes} bytes, rounded up, + 2 GiB)"
else
  fail "setup.sh free-space floor is '${got_floor}', expected ${want_floor}"
fi
want_head_floor=$(( (source_bytes + drafter_bytes + gib - 1) / gib + 2 ))
set +e
head_out="$(cd "${REPO_ROOT}" && clean_env MLXFAST_MTP_HEAD_MANIFEST_PATH="${WORK}/absent.sha256" ./setup-mtp-head.sh 2>&1)"
set -e
if grep -q "drafter free-space floor ${want_head_floor} GiB" <<<"${head_out}"; then
  pass "setup-mtp-head.sh free-space floor is ${want_head_floor} GiB (drafter source + drafter bytes rounded up + 2 GiB)"
else
  fail "setup-mtp-head.sh free-space floor is not ${want_head_floor} GiB: $(grep 'floor' <<<"${head_out}" || true)"
fi

# --- 7. Homebrew is a prerequisite: no installer is downloaded --------------
# ensure_homebrew runs alone, with brew off PATH, no shellenv candidate and a
# curl stub that records any call. It must stop with the official link and
# must not call curl.
mkdir -p "${WORK}/nobrew-bin"
printf '#!/bin/sh\necho called >> "%s/curl.calls"\nexit 0\n' "${WORK}" > "${WORK}/nobrew-bin/curl"
chmod 755 "${WORK}/nobrew-bin/curl"
awk '/^ensure_homebrew\(\) \{/,/^}/' "${REPO_ROOT}/setup.sh" > "${WORK}/ensure_homebrew.sh"
set +e
brew_out="$(PATH="${WORK}/nobrew-bin:/usr/bin:/bin" bash -c '
  SETUP_LOG_LABEL=setup.sh
  load_homebrew_shellenv() { return 1; }
  source "$1"
  ensure_homebrew
' _ "${WORK}/ensure_homebrew.sh" 2>&1)"
brew_rc=$?
set -e
if [[ "${brew_rc}" -ne 0 ]] && grep -q "https://brew.sh" <<<"${brew_out}" && [[ ! -e "${WORK}/curl.calls" ]]; then
  pass "setup.sh stops with the Homebrew link and downloads no installer when brew is missing"
else
  fail "ensure_homebrew without brew (rc ${brew_rc}, curl called: $([[ -e "${WORK}/curl.calls" ]] && echo yes || echo no)): ${brew_out}"
fi
if grep -q 'Homebrew/install' "${REPO_ROOT}/setup.sh"; then
  fail "setup.sh still names the Homebrew installer script"
else
  pass "setup.sh names no Homebrew installer script"
fi

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "test-setup-pins.sh: all cases passed"
else
  echo "test-setup-pins.sh: ${failures} case(s) failed"
  exit 1
fi
