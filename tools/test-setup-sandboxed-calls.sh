#!/usr/bin/env bash
#
# test-setup-sandboxed-calls.sh -- setup.sh starts the trusted CLI only
# through tools/sandboxed-cli.sh.
#
# The CLI links the editable transform module, so code in that module can run
# before main(). setup.sh runs on the ranked runner, so every CLI start there
# must go through the Seatbelt wrapper.
#
#   1. No line of setup.sh outside a comment or the help/summary text starts
#      "${SWIFT_BIN}" or "${BENCH_WORKER_BIN}" as a command.
#   2. list_reference_shards (taken from setup.sh) starts
#      tools/sandboxed-cli.sh checkpoint-shards --index <path>, with
#      MLXFAST_SWIFT_BIN set to the CLI, and never the CLI itself. A stub
#      wrapper and a stub CLI record what ran; nothing is built or sandboxed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SETUP="${REPO_ROOT}/setup.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-setup-sandboxed-calls.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

failures=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; failures=$((failures + 1)); }

# --- 1. no direct CLI start in setup.sh --------------------------------------
direct="$(grep -nE '^[[:space:]]*"?\$\{(SWIFT_BIN|BENCH_WORKER_BIN)\}"?[[:space:]]+[a-z]' "${SETUP}" \
  | grep -vE '^[0-9]+:[[:space:]]*\$\{SWIFT_BIN\} transform --reference "\$\{REFERENCE_DIR\}" --output weights$' || true)"
if [[ -z "${direct}" ]]; then
  pass "setup.sh starts no CLI outside tools/sandboxed-cli.sh"
else
  fail "setup.sh starts the CLI directly: ${direct}"
fi

# --- 2. list_reference_shards goes through the wrapper -----------------------
fn="$(sed -n '/^list_reference_shards() {$/,/^}$/p' "${SETUP}")"
if [[ -z "${fn}" ]]; then
  fail "list_reference_shards is not in setup.sh"
else
  mkdir -p "${WORK}/tree/tools"
  LOG="${WORK}/calls.log"
  : > "${LOG}"
  cat > "${WORK}/tree/tools/sandboxed-cli.sh" <<EOF
#!/usr/bin/env bash
echo "wrapper MLXFAST_SWIFT_BIN=\${MLXFAST_SWIFT_BIN:-} \$*" >> "${LOG}"
echo "model-00001-of-00001.safetensors"
EOF
  cat > "${WORK}/cli" <<EOF
#!/usr/bin/env bash
echo "cli-direct \$*" >> "${LOG}"
echo "model-00001-of-00001.safetensors"
EOF
  chmod +x "${WORK}/tree/tools/sandboxed-cli.sh" "${WORK}/cli"
  set +e
  out="$(cd "${WORK}/tree" && SWIFT_BIN="${WORK}/cli" bash -c "
    set -euo pipefail
    ensure_swift_harness_ready() { return 0; }
    ${fn}
    list_reference_shards '${WORK}/index.json'
  " 2>&1)"
  rc=$?
  set -e
  want="wrapper MLXFAST_SWIFT_BIN=${WORK}/cli checkpoint-shards --index ${WORK}/index.json"
  if [[ "${rc}" == "0" && "${out}" == "model-00001-of-00001.safetensors" ]] \
    && grep -qxF "${want}" "${LOG}" && ! grep -q '^cli-direct' "${LOG}"; then
    pass "list_reference_shards runs checkpoint-shards under tools/sandboxed-cli.sh"
  else
    fail "list_reference_shards did not use the wrapper (rc ${rc}); calls: $(tr '\n' ';' < "${LOG}")"
  fi
fi

if [[ "${failures}" == "0" ]]; then
  echo "test-setup-sandboxed-calls.sh: all cases passed"
else
  echo "test-setup-sandboxed-calls.sh: ${failures} case(s) failed"
  exit 1
fi
