#!/usr/bin/env bash
# Unit test for tools/ci-swift-warning-gate.sh. Offline, no build: it feeds the
# gate small build logs in the two formats it must read and checks the verdict.
#
# WHAT THIS PROVES: the gate is not weakened by the format it learned. The
# Swift 6.4 cases are written byte for byte the way Swift 6.4.2 prints them
# (ANSI colour around "warning:", OSC 8 hyperlinks, "[n / m]" progress with
# U+2009 THIN SPACE around the slash). A first-party warning in that format
# must FAIL the gate, a vendored-only log must pass, and a no-op build log
# (no progress line at all) must still FAIL the compile-activity floor. The
# older "[n/m] Compiling" format keeps its own cases so neither form regresses.
#
# Usage: tools/test-ci-swift-warning-gate.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${ROOT}/tools/ci-swift-warning-gate.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fails=0
cases=0

ESC=$'\033'
TS=$'\xe2\x80\x89'

# log <body>: write a throwaway build log and echo its path. It runs in a
# command substitution, so the file name comes from mktemp, not a counter.
log() {
  local path
  path="$(mktemp "${WORK}/build-XXXXXX")"
  printf '%s' "$1" > "${path}"
  printf '%s' "${path}"
}

# expect <label> <want-rc> <needle-in-output> <log-body>
expect() {
  local label="$1" want="$2" needle="$3" body="$4" out rc
  cases=$((cases + 1))
  out="$("${GATE}" "$(log "${body}")" 2>&1)"
  rc=$?
  if [[ ${rc} -eq ${want} && "${out}" == *"${needle}"* ]]; then
    echo "PASS ${label} (rc=${rc})"
  else
    echo "FAIL ${label}: rc=${rc} want ${want}, needle '${needle}'"
    printf '%s\n' "${out}" | sed 's/^/    | /'
    fails=$((fails + 1))
  fi
}

# Swift 6.4 pieces, as Swift 6.4.2 prints them.
progress="[1646${TS}/${TS}1652] MLXFastTests-product
[1650${TS}/${TS}1652] MLXFastTests-product
"
vendored_c="${ESC}[1m${ROOT}/Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1376:29: ${ESC}[0m${ESC}[0;1;35mwarning: ${ESC}[0m${ESC}[1m'&&' within '||' [-Wlogical-op-parentheses]${ESC}[0m
"
vendored_swift="${ROOT}/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:12:9: ${ESC}[1;33mwarning: ${ESC}[1;39mvariable 'x' was never mutated${ESC}[0m
"
macro="macro expansion #adaptHuggingFaceTokenizer:8:9: ${ESC}[1;33mwarning: ${ESC}[1;39mdeprecated${ESC}[0m
"
caret="  ${ESC}[0;36m|${ESC}[0;0m                \`- ${ESC}[1;33mwarning: ${ESC}[1;39mnever used${ESC}[0m
"
osc="${ESC}]8;;https://docs.swift.org/compiler/documentation/diagnostics/no-usage${ESC}\\NoUsage${ESC}]8;;${ESC}\\"
first_party_src="${ROOT}/Sources/MLXFastCLI/main.swift:4:16: ${ESC}[1;33mwarning: ${ESC}[1;39minitialization of immutable value 'y' was never used [#${osc}]${ESC}[0m
"
first_party_test="/private${ROOT}/Tests/MLXFastTests/GateTests.swift:7:5: ${ESC}[1;33mwarning: ${ESC}[1;39mwill never be executed${ESC}[0m
"
head="Building for debugging...
[Planning deferred tasks]
"
tail="Build complete! (33.45 sec)
"

expect "swift 6.4: vendored and macro warnings only, real progress -> clean" 0 \
  "0 first-party (Sources/, Tests/), 2 vendored/dependency, 1 inside vendored macro expansions, 2 compile steps" \
  "${head}${progress}${vendored_c}${vendored_swift}${macro}${caret}${tail}"
expect "swift 6.4: coloured first-party Sources/ warning -> FAIL" 1 \
  "FAIL  first-party warning: ${ROOT}/Sources/MLXFastCLI/main.swift:4:16: warning: initialization of immutable value 'y' was never used [#NoUsage]" \
  "${head}${progress}${vendored_c}${first_party_src}${caret}${tail}"
expect "swift 6.4: coloured first-party Tests/ warning (/private path) -> FAIL" 1 \
  "1 first-party" \
  "${head}${progress}${first_party_test}${tail}"
expect "swift 6.4: no-op build (no progress line) -> FAIL no compile activity" 1 \
  "FAIL  no compile activity" \
  "Building for debugging...
${tail}"
expect "swift 6.4: planning line only -> FAIL no compile activity" 1 \
  "FAIL  no compile activity" \
  "${head}${tail}"
expect "swift 6.4: caret line alone is not a diagnostic" 0 \
  "0 first-party" \
  "${head}${progress}${caret}${tail}"
expect "older swiftpm: [n/m] Compiling, vendored warning -> clean" 0 \
  "ci-swift-warning-gate: clean" \
  "[12/40] Compiling MLX Array.swift
${ROOT}/Vendor/mlx-swift/Source/MLX/Array.swift:3:1: warning: old style
"
expect "older swiftpm: plain first-party warning -> FAIL" 1 \
  "FAIL  first-party warning: ${ROOT}/Sources/MLXFastCLI/main.swift:1:1: warning: plain" \
  "[12/40] Compiling MLXFastCLI main.swift
${ROOT}/Sources/MLXFastCLI/main.swift:1:1: warning: plain
"
expect "older swiftpm: bare [n/m] without a verb is not compile activity" 1 \
  "FAIL  no compile activity" \
  "[3/3] Write swift-version
"

echo
if [[ ${fails} -ne 0 ]]; then
  echo "test-ci-swift-warning-gate: ${fails} case(s) FAILED"
  exit 1
fi
echo "test-ci-swift-warning-gate: all ${cases} cases passed"
