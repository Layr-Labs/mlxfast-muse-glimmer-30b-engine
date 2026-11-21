#!/usr/bin/env bash
#
# test-make-drafter.sh -- tools/make-drafter.sh puts a drafter in place only when
# both of its files match the track fixture's mtp_head.artifact pins, never
# overwrites a directory that fails them, and its --dry-run writes nothing.
#
# Nothing here loads a model or downloads anything. The produce path runs a
# stub make-dflash-drafter (MLXFAST_MAKE_DRAFTER_BIN); a fixture copy pins the
# stub's bytes, so the check under test is the real one.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOL="${REPO_ROOT}/tools/make-drafter.sh"
FIXTURE="${REPO_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-make-drafter.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

command -v jq >/dev/null 2>&1 || { echo "test-make-drafter.sh: jq is required" >&2; exit 1; }

failures=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; failures=$((failures + 1)); }

# run [ENV=VAL...] -- ARGS: sets rc and out.
run() {
  local envs=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" != "--" ]] || shift
  set +e
  out="$(env -u MLXFAST_MTP_HEAD_DIR -u MLXFAST_DRAFTER_SOURCE_DIR -u MLXFAST_MAKE_DRAFTER_BIN \
    -u RUNNER_ENVIRONMENT -u MLXFAST_OFFICIAL_BENCHMARK_RUN \
    ${envs[@]+"${envs[@]}"} "${TOOL}" "$@" 2>&1)"
  rc=$?
  set -e
}

# A placeholder drafter, and a fixture copy that pins exactly its bytes.
GOOD="${WORK}/good-artifact"
mkdir -p "${GOOD}"
printf '{"placeholder":"config"}\n' > "${GOOD}/config.json"
printf 'placeholder weights\n' > "${GOOD}/model.safetensors"
sha() { shasum -a 256 "$1" | awk '{print $1}'; }
good_bytes=$(( $(wc -c < "${GOOD}/config.json") + $(wc -c < "${GOOD}/model.safetensors") ))
jq --arg w "$(sha "${GOOD}/model.safetensors")" --arg c "$(sha "${GOOD}/config.json")" --argjson b "${good_bytes}" \
  '.mtp_head.artifact.weights_sha256 = $w | .mtp_head.artifact.configuration_sha256 = $c | .mtp_head.artifact.bytes = $b' \
  "${FIXTURE}" > "${WORK}/fixture.json"
PINNED="TRACK_FIXTURE_PATH=${WORK}/fixture.json"

# A BF16 source directory with the two names the writer reads.
SRC="${WORK}/source"
mkdir -p "${SRC}"
printf '{}\n' > "${SRC}/config.json"
printf 'bf16\n' > "${SRC}/model.safetensors"

# --- 1. --dry-run resolves the shipped pins and writes nothing ----------------
run -- --source "${SRC}" --output "${WORK}/dry-out" --dry-run
if [[ "${rc}" -eq 0 && "${out}" == *"$(jq -r .mtp_head.artifact.weights_sha256 "${FIXTURE}")"* \
      && "${out}" == *"$(jq -r .mtp_head.source_artifact.weights_sha256 "${FIXTURE}")"* \
      && "${out}" == *"nothing was built, converted, copied or written"* && ! -e "${WORK}/dry-out" ]]; then
  pass "--dry-run prints the fixture's drafter and source pins and writes nothing"
else
  fail "--dry-run (rc ${rc}): ${out}"
fi
if [[ -z "$(find "${WORK}" -maxdepth 1 -name 'dry-out.staging.*')" ]]; then
  pass "--dry-run leaves no staging directory"
else
  fail "--dry-run left a staging directory"
fi

# --- 2. a missing source is refused, also in a dry run -----------------------
run -- --source "${WORK}/no-source" --output "${WORK}/x" --dry-run
if [[ "${rc}" -eq 1 && "${out}" == *"the BF16 source has no config.json"* && "${out}" == *"setup-mtp-head.sh"* ]]; then
  pass "a missing BF16 source is refused by name"
else
  fail "a missing source (rc ${rc}): ${out}"
fi

# --- 3. --source and --artifact are exclusive --------------------------------
run -- --source "${SRC}" --artifact "${GOOD}" --output "${WORK}/x"
if [[ "${rc}" -eq 1 && "${out}" == *"exclusive"* ]]; then
  pass "--source with --artifact is refused"
else
  fail "--source with --artifact (rc ${rc}): ${out}"
fi

# --- 4. --artifact: a copy that fails the pins is refused, nothing is placed --
run -- --artifact "${GOOD}" --output "${WORK}/out-unpinned"
if [[ "${rc}" -eq 1 && "${out}" == *"fails its pins"* && ! -e "${WORK}/out-unpinned" ]]; then
  pass "an artifact that fails the shipped pins is refused and nothing is placed"
else
  fail "an unpinned artifact (rc ${rc}): ${out}"
fi
if [[ -z "$(find "${WORK}" -maxdepth 1 -name 'out-unpinned.staging.*')" ]]; then
  pass "a refused copy leaves no staging directory"
else
  fail "a refused copy left a staging directory"
fi

# --- 5. --artifact: a pinned copy is placed, then kept on a second run --------
run "${PINNED}" -- --artifact "${GOOD}" --output "${WORK}/weights/drafter"
if [[ "${rc}" -eq 0 && "${out}" == *"is in place at ${WORK}/weights/drafter"* \
      && "$(sha "${WORK}/weights/drafter/model.safetensors")" == "$(sha "${GOOD}/model.safetensors")" ]]; then
  pass "a pinned artifact is copied into place and verified"
else
  fail "a pinned artifact (rc ${rc}): ${out}"
fi
run "${PINNED}" -- --artifact "${GOOD}" --output "${WORK}/weights/drafter"
if [[ "${rc}" -eq 0 && "${out}" == *"already holds the pinned drafter"* ]]; then
  pass "a second run keeps a drafter that passes the pins"
else
  fail "a second run (rc ${rc}): ${out}"
fi

# --- 6. an output that fails the pins is refused, never overwritten ----------
mkdir -p "${WORK}/stale"
printf 'old\n' > "${WORK}/stale/config.json"
printf 'old\n' > "${WORK}/stale/model.safetensors"
run "${PINNED}" -- --artifact "${GOOD}" --output "${WORK}/stale"
if [[ "${rc}" -eq 1 && "${out}" == *"is not the pinned drafter"* && "$(cat "${WORK}/stale/model.safetensors")" == "old" ]]; then
  pass "an output that fails the pins is refused and left as it was"
else
  fail "a stale output (rc ${rc}): ${out}"
fi

# --- 7. --source: the writer's output is checked before it is placed ----------
STUB_DIR="${WORK}/bin"
mkdir -p "${STUB_DIR}"
printf 'metallib\n' > "${STUB_DIR}/mlx.metallib"
cat > "${STUB_DIR}/make-dflash-drafter" <<STUB
#!/usr/bin/env bash
# args: --source <dir> --output <dir>
out="\$4"
cp "${GOOD}/config.json" "\${out}/config.json"
if [[ "\${STUB_BAD:-0}" == "1" ]]; then printf 'tampered\n' > "\${out}/model.safetensors"; else cp "${GOOD}/model.safetensors" "\${out}/model.safetensors"; fi
STUB
chmod +x "${STUB_DIR}/make-dflash-drafter"
run "${PINNED}" "MLXFAST_MAKE_DRAFTER_BIN=${STUB_DIR}/make-dflash-drafter" -- --source "${SRC}" --output "${WORK}/produced"
if [[ "${rc}" -eq 0 && -f "${WORK}/produced/model.safetensors" && "${out}" == *"is in place"* ]]; then
  pass "--source runs the writer and places a drafter that passes the pins"
else
  fail "--source with a good writer (rc ${rc}): ${out}"
fi
run "${PINNED}" "MLXFAST_MAKE_DRAFTER_BIN=${STUB_DIR}/make-dflash-drafter" STUB_BAD=1 -- --source "${SRC}" --output "${WORK}/produced-bad"
if [[ "${rc}" -eq 1 && "${out}" == *"fails its pins"* && ! -e "${WORK}/produced-bad" ]]; then
  pass "--source refuses a written drafter that fails the pins, and places nothing"
else
  fail "--source with a bad writer (rc ${rc}): ${out}"
fi
rm -f "${STUB_DIR}/mlx.metallib"
run "${PINNED}" "MLXFAST_MAKE_DRAFTER_BIN=${STUB_DIR}/make-dflash-drafter" -- --source "${SRC}" --output "${WORK}/produced-nolib"
if [[ "${rc}" -eq 1 && "${out}" == *"has no mlx.metallib"* ]]; then
  pass "--source refuses a writer binary with no mlx.metallib beside it"
else
  fail "--source with no metallib (rc ${rc}): ${out}"
fi

# --- 8. an official run only copies: --source is refused, nothing runs -------
# The stub writer leaves a marker when it runs, so a refusal that still ran it
# is caught. Both official markers are tried, and a dry run is refused too.
printf 'metallib\n' > "${STUB_DIR}/mlx.metallib"
cat > "${STUB_DIR}/make-dflash-drafter-marker" <<STUB
#!/usr/bin/env bash
touch "${WORK}/writer-ran"
exit 0
STUB
chmod +x "${STUB_DIR}/make-dflash-drafter-marker"
for official in RUNNER_ENVIRONMENT=self-hosted MLXFAST_OFFICIAL_BENCHMARK_RUN=1; do
  for dry in "" --dry-run; do
    run "${PINNED}" "${official}" "MLXFAST_MAKE_DRAFTER_BIN=${STUB_DIR}/make-dflash-drafter-marker" -- \
      --source "${SRC}" --output "${WORK}/official-produced" ${dry}
    if [[ "${rc}" -eq 1 && "${out}" == *"this is an official run"* && "${out}" == *"--artifact"* \
          && ! -e "${WORK}/writer-ran" && ! -e "${WORK}/official-produced" ]]; then
      pass "official (${official}${dry:+ ${dry}}): --source is refused, no writer runs, nothing is placed"
    else
      fail "official (${official}${dry:+ ${dry}}) --source (rc ${rc}): ${out}"
    fi
  done
  run "${official}" "MLXFAST_MAKE_DRAFTER_BIN=${STUB_DIR}/make-dflash-drafter-marker" -- --output "${WORK}/official-default"
  if [[ "${rc}" -eq 1 && "${out}" == *"this is an official run"* && ! -e "${WORK}/writer-ran" ]]; then
    pass "official (${official}): the default (produce) path is refused"
  else
    fail "official (${official}) default path (rc ${rc}): ${out}"
  fi
done

# --- 9. an official run copies a pinned artifact and checks it ----------------
run "${PINNED}" MLXFAST_OFFICIAL_BENCHMARK_RUN=1 -- --artifact "${GOOD}" --output "${WORK}/official/weights/drafter"
if [[ "${rc}" -eq 0 && "${out}" == *"official run: copy only"* && "${out}" == *"is in place at ${WORK}/official/weights/drafter"* \
      && "$(sha "${WORK}/official/weights/drafter/model.safetensors")" == "$(sha "${GOOD}/model.safetensors")" ]]; then
  pass "official: a pinned artifact is copied into place and verified"
else
  fail "official pinned artifact (rc ${rc}): ${out}"
fi
run RUNNER_ENVIRONMENT=self-hosted -- --artifact "${GOOD}" --output "${WORK}/official-unpinned"
if [[ "${rc}" -eq 1 && "${out}" == *"fails its pins"* && ! -e "${WORK}/official-unpinned" ]]; then
  pass "official: an artifact that fails the shipped pins is refused and nothing is placed"
else
  fail "official unpinned artifact (rc ${rc}): ${out}"
fi

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "test-make-drafter.sh: all cases passed"
  exit 0
fi
echo "test-make-drafter.sh: ${failures} case(s) failed" >&2
exit 1
