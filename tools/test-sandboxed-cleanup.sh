#!/usr/bin/env bash
# test-sandboxed-cleanup.sh -- what confined entry code wrote does not outlive
# the wrapper that confined it.
#   1. sandboxed-cli.sh transform: a stub CLI (compiled, so the profile may exec
#      it) leaves a staging sibling .<output>.staging beside the weights and a
#      mode-000 directory in its private TMPDIR. After the wrapper exits, both
#      are gone, the wrapper's work directory is gone, and the published output
#      remains.
#   2. sandboxed-build.sh: a stub build leaves a mode-000 directory with a
#      user-immutable file in its private TMPDIR; after the wrapper exits, its
#      work directory is gone.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd -P)"
WORK="$(mktemp -d)"; WORK="$(cd "${WORK}" && pwd -P)"
trap 'find "${WORK}" -exec chflags -h nouchg {} \; -exec chmod -h u+rwx {} \; 2>/dev/null; rm -rf "${WORK}"' EXIT
FAILS=0
pass() { echo "ok    $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }
[[ -x /usr/bin/sandbox-exec ]] || { echo "test-sandboxed-cleanup.sh: SKIP (needs macOS): no /usr/bin/sandbox-exec on this host"; exit 0; }
cat > "${WORK}/stub.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
int main(int argc, char **argv) {
  const char *out = NULL;
  for (int i = 2; i + 1 < argc; i += 2) if (!strcmp(argv[i], "--output")) out = argv[i + 1];
  char path[4096], dir[4096], base[1024];
  const char *slash = strrchr(out, '/');
  snprintf(dir, sizeof dir, "%.*s", (int)(slash - out), out);
  snprintf(base, sizeof base, "%s", slash + 1);
  snprintf(path, sizeof path, "%s/.%s.staging", dir, base);
  mkdir(path, 0700);
  snprintf(path, sizeof path, "%s/.%s.staging/leftover", dir, base);
  FILE *f = fopen(path, "w"); if (f) { fputs("entry data\n", f); fclose(f); }
  mkdir(out, 0755);
  snprintf(path, sizeof path, "%s/model.safetensors", out);
  f = fopen(path, "w"); if (f) { fputs("weights\n", f); fclose(f); }
  const char *tmp = getenv("TMPDIR");
  snprintf(path, sizeof path, "%slocked", tmp ? tmp : "/tmp/");
  mkdir(path, 0700); chmod(path, 0);
  return 0;
}
C
cc -o "${WORK}/mlxfast-swift" "${WORK}/stub.c" || { echo "FAIL  cannot compile the stub"; exit 1; }
mkdir -p "${WORK}/runner-temp" "${WORK}/weights-root" "${WORK}/ref"
OUT="${WORK}/weights-root/weights"
out="$(env -i PATH="${PATH}" HOME="${WORK}" RUNNER_TEMP="${WORK}/runner-temp" MLXFAST_SWIFT_BIN="${WORK}/mlxfast-swift" \
  tools/sandboxed-cli.sh transform --reference "${WORK}/ref" --output "${OUT}" 2>&1)"; rc=$?
left="$(find "${WORK}/weights-root" -mindepth 1 -maxdepth 1 -name '.*' 2>/dev/null; find "${WORK}/runner-temp" -mindepth 1 -maxdepth 1 2>/dev/null)"
if [[ ${rc} -eq 0 && -z "${left}" && -f "${OUT}/model.safetensors" ]]; then
  pass "transform: the staging sibling and the private TMPDIR (with a mode-000 directory) are removed; the output remains"
else
  fail "transform left entry-written files (rc ${rc}): ${left//$'\n'/ } ${out}"
fi

cat > "${WORK}/build-step.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
int main(void) {
  char p[4096]; const char *t = getenv("TMPDIR");
  snprintf(p, sizeof p, "%slocked", t); mkdir(p, 0700);
  snprintf(p, sizeof p, "%slocked/note", t); FILE *f = fopen(p, "w"); if (f) { fputs("x", f); fclose(f); }
  chflags(p, UF_IMMUTABLE);
  snprintf(p, sizeof p, "%slocked", t); chmod(p, 0);
  return 0;
}
C
cc -o "${WORK}/build-step" "${WORK}/build-step.c" || { echo "FAIL  cannot compile the build stub"; exit 1; }
mkdir -p "${WORK}/build-temp" "${WORK}/build-out"
out="$(env -i PATH="${PATH}" HOME="${WORK}" RUNNER_TEMP="${WORK}/build-temp" MLXFAST_OFFICIAL_BENCHMARK_RUN=1 \
  MLXFAST_QWEN38_GOLDEN_DIR="${WORK}/g" MLXFAST_BASELINE_WORKSPACE="${WORK}/b" MLXFAST_BASELINE_CALIBRATION="${WORK}/c" \
  BENCHD_BIN_DIR="${WORK}/d" RUNNER_WORKSPACE="${WORK}/rw/a/b" \
  tools/sandboxed-build.sh --write "${WORK}/build-out" -- "${WORK}/build-step" 2>&1)"; rc=$?
left="$(find "${WORK}/build-temp" -mindepth 1 -maxdepth 1 2>/dev/null)"
if [[ ${rc} -eq 0 && -z "${left}" ]]; then
  pass "build: the private TMPDIR (mode-000 directory, immutable file) is removed"
else
  fail "build left its work directory (rc ${rc}): ${left//$'\n'/ } ${out}"
fi
if [[ ${FAILS} -eq 0 ]]; then echo "test-sandboxed-cleanup.sh: all cases passed"; exit 0; fi
echo "test-sandboxed-cleanup.sh: ${FAILS} case(s) failed"; exit 1
