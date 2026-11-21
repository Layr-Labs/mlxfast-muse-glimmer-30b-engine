#!/usr/bin/env bash
#
# test-optional-defaults.sh -- an honest entry may leave out an optional
# editable path and still pass both hosted checks.
#
# The hosted surface-check job runs the manifest lint and the whole-surface
# static review on the candidate's work tree. Before them,
# .github/scripts/apply-optional-defaults.sh restores the protected default of
# every optionalEditablePaths entry the candidate left out, as the overlay does.
# Cases (in a temporary git repository made from this tree, no network):
#   1. a candidate without mtp-head.manifest.json passes the lint and the
#      static review after the step;
#   2. a candidate with a broken mtp-head.manifest.json is kept as written and
#      the trusted reader (tools/spec-declaration.sh) still refuses it;
#   3. benchmark.yml runs the step before the lint, with the protected base.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT
fails=0
pass() { echo "ok    $*"; }
fail() { fails=$((fails + 1)); echo "FAIL  $*" >&2; }

T="${WORK}/tree"
mkdir -p "${T}"
git -C "${REPO_ROOT}" ls-files -z --cached --others --exclude-standard \
  | (cd "${REPO_ROOT}" && tar --null -T - -cf -) | tar -xf - -C "${T}"
g() { git -C "${T}" -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
g init -q && g add -A && g commit -qm base || { echo "test-optional-defaults.sh: could not make the test repository" >&2; exit 1; }
BASE="$(g rev-parse HEAD)"

run_checks() {
  (
    cd "${T}" || exit 1
    BASE_SHA="${BASE}" .github/scripts/apply-optional-defaults.sh >/dev/null 2>"${WORK}/apply.err" || { cat "${WORK}/apply.err"; exit 10; }
    python3 tools/lint-benchmark-manifest.py --gitlink-targets report > "${WORK}/lint.out" 2>&1 || { tail -3 "${WORK}/lint.out"; exit 11; }
    CONTRACT_PATH=benchmark.json MLXFAST_SUBMISSION_REVIEW_MODE=whole-surface \
      .github/scripts/submission-static-review-checks.sh > "${WORK}/review.out" 2>&1 || { tail -3 "${WORK}/review.out"; exit 12; }
  )
}

# 1. leave the optional file out.
g rm -q mtp-head.manifest.json && g commit -qm "candidate without the optional file"
out="$(run_checks 2>&1)"; rc=$?
if [[ "${rc}" -eq 0 && -f "${T}/mtp-head.manifest.json" ]]; then
  pass "a candidate without mtp-head.manifest.json passes the lint and the static review"
else
  fail "a candidate without mtp-head.manifest.json is refused (rc ${rc}): ${out}"
fi

# 2. a broken file is kept as the candidate wrote it, and is refused. Neither
#    hosted check parses the declaration; the trusted reader that the ranked
#    job runs (tools/spec-declaration.sh) refuses it.
printf '{ "source": "pinned", \n' > "${T}/mtp-head.manifest.json"
g add mtp-head.manifest.json && g commit -qm "candidate with a broken optional file"
before_sum="$(shasum -a 256 < "${T}/mtp-head.manifest.json")"
run_checks >/dev/null 2>&1
after_sum="$(shasum -a 256 < "${T}/mtp-head.manifest.json")"
out="$(cd "${T}" && tools/spec-declaration.sh describe 2>&1)"; rc=$?
if [[ "${before_sum}" == "${after_sum}" && "${rc}" -ne 0 && "${out}" == *"not valid JSON"* ]]; then
  pass "a broken mtp-head.manifest.json is kept as written and the trusted reader refuses it"
else
  fail "a broken mtp-head.manifest.json was replaced or accepted (reader rc ${rc}): ${out}"
fi

# 3. the workflow applies the default before the lint.
wf="${REPO_ROOT}/.github/workflows/benchmark.yml"
apply_line="$(grep -n 'apply-optional-defaults.sh' "${wf}" | head -1 | cut -d: -f1)"
lint_line="$(grep -n 'python3 tools/lint-benchmark-manifest.py' "${wf}" | head -1 | cut -d: -f1)"
if [[ -n "${apply_line}" && -n "${lint_line}" && "${apply_line}" -lt "${lint_line}" ]] \
  && grep -q 'BASE_SHA: ${{ steps.surface.outputs.base_sha }}' "${wf}"; then
  pass "benchmark.yml restores the protected default before the lint, from the protected base"
else
  fail "benchmark.yml does not run apply-optional-defaults.sh with the protected base before the lint"
fi

if [[ "${fails}" -eq 0 ]]; then
  echo "test-optional-defaults.sh: all cases passed"
  exit 0
fi
echo "test-optional-defaults.sh: ${fails} case(s) failed" >&2
exit 1
