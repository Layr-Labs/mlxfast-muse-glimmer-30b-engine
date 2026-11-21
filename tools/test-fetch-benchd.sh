#!/usr/bin/env bash
# test-fetch-benchd.sh -- fetch-benchd.sh resolves the renamed channel pair.
#
# Hermetic: a stub `benchd` and a manifest in the post-rename dist format
# (six top-level fields describing benchd, then the `binaries` map with a
# sha256/bytes entry per binary) are handed to fetch-benchd.sh through
# BENCHD_DIST_LOCAL. The script must install benchd + benchd.manifest.json into
# BENCHD_BIN_DIR, print the binary path, and refuse a pair whose bytes moved.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fail() { echo "test-fetch-benchd: FAIL -- $*" >&2; exit 1; }

# The channel name is an input of fetch-benchd.sh (BENCHD_BRANCH); the test
# names its own so the manifest and the resolve agree by construction.
BRANCH="test-channel-v0"
export BENCHD_BRANCH="${BRANCH}"
# The pinned source commit is an input too (BENCHD_SOURCE_COMMIT); the stub
# manifest below names this one.
TEST_SOURCE_COMMIT="0123456789abcdef0123456789abcdef01234567"
export BENCHD_SOURCE_COMMIT="${TEST_SOURCE_COMMIT}"

DIST="${WORK}/dist"; mkdir -p "${DIST}"
printf '#!/bin/sh\necho stub-benchd\n' > "${DIST}/benchd"; chmod 755 "${DIST}/benchd"
printf '#!/bin/sh\necho stub-recorder\n' > "${DIST}/record-correctness-golden"; chmod 755 "${DIST}/record-correctness-golden"
sha() { shasum -a 256 "$1" | awk '{print $1}'; }
bytes() { wc -c < "$1" | tr -d '[:space:]'; }
{
  printf '{\n'
  printf '  "version": "0.0.0",\n'
  printf '  "branch": "%s",\n' "${BRANCH}"
  printf '  "source_commit": "0123456789abcdef0123456789abcdef01234567",\n'
  printf '  "target_triple": "aarch64-apple-darwin",\n'
  printf '  "sha256": "%s",\n' "$(sha "${DIST}/benchd")"
  printf '  "bytes": %s,\n' "$(bytes "${DIST}/benchd")"
  printf '  "binaries": {\n'
  printf '    "benchd": {"sha256": "%s", "bytes": %s},\n' "$(sha "${DIST}/benchd")" "$(bytes "${DIST}/benchd")"
  printf '    "record-correctness-golden": {"sha256": "%s", "bytes": %s}\n' "$(sha "${DIST}/record-correctness-golden")" "$(bytes "${DIST}/record-correctness-golden")"
  printf '  }\n'
  printf '}\n'
} > "${DIST}/benchd.manifest.json"

BIN="${WORK}/benchd-bin"
if ! out="$(BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN}" bash "${ROOT}/tools/fetch-benchd.sh" 2>"${WORK}/err")"; then
  fail "resolve refused a good post-rename pair: $(cat "${WORK}/err")"
fi
[[ "${out}" == "${BIN}/benchd" ]] || fail "printed '${out}', expected ${BIN}/benchd"
[[ -x "${BIN}/benchd" && -f "${BIN}/benchd.manifest.json" ]] || fail "pair not installed under ${BIN}"
grep -q "source_commit=0123456789abcdef0123456789abcdef01234567" "${WORK}/err" || fail "identity line missing: $(cat "${WORK}/err")"
echo "test-fetch-benchd: PASS -- post-rename pair resolves (binaries map tolerated)"

# A good pair whose manifest names another source_commit is not the pinned
# benchd: refused on the BENCHD_DIST_LOCAL path and on the offline path.
BIN_PIN="${WORK}/benchd-bin-pin"
if BENCHD_SOURCE_COMMIT=fedcba9876543210fedcba9876543210fedcba98 BENCHD_DIST_LOCAL="${DIST}" \
     BENCHD_BIN_DIR="${BIN_PIN}" bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/err-pin"; then
  fail "a manifest with another source_commit was accepted from BENCHD_DIST_LOCAL"
fi
[[ ! -e "${BIN_PIN}/benchd" ]] || fail "a pair with another source_commit was installed"
grep -q "not the pinned benchd" "${WORK}/err-pin" || fail "the pin refusal is not named: $(cat "${WORK}/err-pin")"
if BENCHD_SOURCE_COMMIT=fedcba9876543210fedcba9876543210fedcba98 \
     BENCHD_BIN_DIR="${BIN}" bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/err-pin2"; then
  fail "an installed pair with another source_commit was run"
fi
grep -q "not the pinned benchd" "${WORK}/err-pin2" || fail "the offline pin refusal is not named: $(cat "${WORK}/err-pin2")"
echo "test-fetch-benchd: PASS -- a manifest that is not the pinned source_commit is refused (local and offline paths)"

# The verified offline pair is chmod-ed only when it is not executable yet: a
# root-owned staged pair is already executable and the job account cannot
# chmod it. A stub chmod records every call.
CHMOD_STUB="${WORK}/chmod-bin"; mkdir -p "${CHMOD_STUB}"
printf '#!/bin/sh\nprintf "%%s\\n" "$*" >> "${CHMOD_LOG}"\nexit 1\n' > "${CHMOD_STUB}/chmod"
/bin/chmod 755 "${CHMOD_STUB}/chmod"
: > "${WORK}/chmod.log"
PATH="${CHMOD_STUB}:${PATH}" CHMOD_LOG="${WORK}/chmod.log" BENCHD_BIN_DIR="${BIN}" \
  bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/err-chmod" \
  || fail "an executable verified offline pair was refused: $(cat "${WORK}/err-chmod")"
[[ ! -s "${WORK}/chmod.log" ]] || fail "an executable verified pair was chmod-ed: $(cat "${WORK}/chmod.log")"
echo "test-fetch-benchd: PASS -- an executable verified offline pair is used without chmod"

# A binary whose bytes moved after the manifest was written is refused.
printf '#!/bin/sh\necho tampered\n' > "${DIST}/benchd"
BIN2="${WORK}/benchd-bin-2"
if BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN2}" bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/err2"; then
  fail "a tampered binary was accepted"
fi
[[ ! -e "${BIN2}/benchd" ]] || fail "a tampered binary was installed"
echo "test-fetch-benchd: PASS -- a mismatching binary is refused and nothing is installed"

# --- the GitHub token goes to the approved GitHub origin only ----------------
# A stub curl records its arguments and fails, so no network is used and the
# script stops at "could not fetch" after the first request.
STUB="${WORK}/stub-bin"; mkdir -p "${STUB}"
cat > "${STUB}/curl" <<'STUBEOF'
#!/bin/sh
printf '%s\n' "$*" >> "${CURL_LOG}"
exit 22
STUBEOF
chmod 755 "${STUB}/curl"
run_fetch() {
  local log="$1"; shift
  : > "${log}"
  env -u BENCHD_DIST_TOKEN -u GITHUB_TOKEN -u BENCHD_DIST_MIRROR_TOKEN -u BENCHD_DIST_BASE_URL \
    PATH="${STUB}:${PATH}" CURL_LOG="${log}" BENCHD_BIN_DIR="${WORK}/bin-auth" "$@" \
    bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>&1 || true
  [[ -s "${log}" ]] || fail "the stub curl was not called (${log})"
}

run_fetch "${WORK}/curl-default.log" GITHUB_TOKEN=gh-test-token
grep -q "Authorization: Bearer gh-test-token" "${WORK}/curl-default.log" \
  || fail "the default GitHub origin did not receive GITHUB_TOKEN: $(cat "${WORK}/curl-default.log")"
echo "test-fetch-benchd: PASS -- the default GitHub origin receives the GitHub token"

grep -q "https://raw.githubusercontent.com/Layr-Labs/benchd/[0-9a-f]\{40\}/dist/benchd.manifest.json" "${WORK}/curl-default.log" \
  || fail "the default fetch is not <base>/<pinned commit>/dist: $(cat "${WORK}/curl-default.log")"
grep -q "refs/heads" "${WORK}/curl-default.log" && fail "the default fetch follows a branch tip: $(cat "${WORK}/curl-default.log")"
echo "test-fetch-benchd: PASS -- the default fetch reads the pinned commit's dist, not a branch tip"

run_fetch "${WORK}/curl-mirror.log" GITHUB_TOKEN=gh-test-token BENCHD_DIST_TOKEN=dist-test-token \
  BENCHD_DIST_BASE_URL=https://mirror.example.com/bench
grep -q "Authorization" "${WORK}/curl-mirror.log" \
  && fail "a mirror host received a GitHub token: $(cat "${WORK}/curl-mirror.log")"
grep -q "https://mirror.example.com/bench/" "${WORK}/curl-mirror.log" \
  || fail "the mirror URL was not fetched: $(cat "${WORK}/curl-mirror.log")"
echo "test-fetch-benchd: PASS -- a mirror host receives no GitHub token"

for lookalike in https://raw.githubusercontent.com.example.com/x https://user@raw.githubusercontent.com/x \
                 http://raw.githubusercontent.com/x https://raw.githubusercontent.com:8443/x; do
  run_fetch "${WORK}/curl-lookalike.log" GITHUB_TOKEN=gh-test-token BENCHD_DIST_BASE_URL="${lookalike}"
  grep -q "Authorization" "${WORK}/curl-lookalike.log" \
    && fail "a look-alike origin (${lookalike}) received the GitHub token"
done
echo "test-fetch-benchd: PASS -- look-alike origins (suffix, user part, http, port) receive no GitHub token"

run_fetch "${WORK}/curl-mirror-token.log" GITHUB_TOKEN=gh-test-token \
  BENCHD_DIST_MIRROR_TOKEN=mirror-test-token BENCHD_DIST_BASE_URL=https://mirror.example.com/bench
grep -q "Authorization: Bearer mirror-test-token" "${WORK}/curl-mirror-token.log" \
  || fail "the mirror did not receive BENCHD_DIST_MIRROR_TOKEN: $(cat "${WORK}/curl-mirror-token.log")"
grep -q "gh-test-token" "${WORK}/curl-mirror-token.log" \
  && fail "the mirror received the GitHub token beside its own"
echo "test-fetch-benchd: PASS -- a mirror receives only its own explicit token"

echo "test-fetch-benchd: all cases pass"
