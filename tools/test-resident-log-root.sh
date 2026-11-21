#!/usr/bin/env bash
# test-resident-log-root.sh -- prove tools/resident-log-root.sh prepare:
#   1. an earlier measurement's leftovers in the root (a retained log, a
#      resident tmp.* tree with a mode-000 directory and a user-immutable
#      file) are removed, and the new log directory is the root's only entry
#   2. a root that is a symbolic link is refused
#   3. a root writable by group or other is refused
#   4. the measure script prepares its log directory through this script
set -uo pipefail
cd "$(dirname "$0")/.."
PREP="tools/resident-log-root.sh"
WORK="$(mktemp -d)"
trap 'command -v chflags >/dev/null && chflags -R nouchg "${WORK}" 2>/dev/null; chmod -R u+rwx "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT
FAILS=0
pass() { echo "ok    $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }

R1="${WORK}/root1"; mkdir -m 700 -p "${R1}/resident-up.old/tmp.1-x/locked"
printf 'leg A marker\n' > "${R1}/resident-up.old/resident.1.log"
printf 'note\n' > "${R1}/resident-up.old/tmp.1-x/locked/note"
# A user-immutable file needs BSD chflags; Linux (hosted CI) has none, so there
# the case runs with the mode-000 directory only.
if command -v chflags >/dev/null; then chflags uchg "${R1}/resident-up.old/tmp.1-x/locked/note"; fi
chmod 000 "${R1}/resident-up.old/tmp.1-x/locked"
new="$(bash "${PREP}" prepare "${R1}" 2>&1)"; rc=$?
entries="$(find "${R1}" -mindepth 1 -maxdepth 1 2>&1)"
if [[ ${rc} -eq 0 && "${new}" == "${R1}/resident-up."* && -d "${new}" && "${entries}" == "${new}" ]]; then
  pass "an earlier measurement's leftovers are removed; the new log directory is the root's only entry"
else
  fail "leftovers not removed (rc ${rc}): new '${new}' entries '${entries}'"
fi

mkdir -p "${WORK}/real"; ln -s "${WORK}/real" "${WORK}/link-root"
out="$(bash "${PREP}" prepare "${WORK}/link-root" 2>&1)"; rc=$?
[[ ${rc} -eq 2 && "${out}" == *log-root-unusable* ]] && pass "a symbolic link as the root is refused" || fail "a linked root was accepted (rc ${rc}): ${out}"

mkdir -p "${WORK}/wide"; chmod 0777 "${WORK}/wide"
out="$(bash "${PREP}" prepare "${WORK}/wide" 2>&1)"; rc=$?
[[ ${rc} -eq 2 && "${out}" == *log-root-unusable* ]] && pass "a root writable by group or other is refused" || fail "a 0777 root was accepted (rc ${rc}): ${out}"

if grep -q 'tools/resident-log-root.sh" prepare' tools/muse-glimmer-30b-measure-and-score.sh \
   && ! grep -q 'mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/resident-up.XXXXXX"' tools/muse-glimmer-30b-measure-and-score.sh; then
  pass "the measure script prepares its log directory in the fixed root"
else
  fail "the measure script still makes a fresh unchecked log directory"
fi

# 5. the ranked job empties the root before any entry code (build, transform)
wf=.github/workflows/benchmark.yml
prep="$(grep -n 'name: Empty the resident log root before any entry code runs' "${wf}" | head -1 | cut -d: -f1)"
build="$(grep -n 'name: Setup (toolchain' "${wf}" | head -1 | cut -d: -f1)"
xform="$(grep -n 'name: Transform the reference checkpoint' "${wf}" | head -1 | cut -d: -f1)"
if [[ -n "${prep}" && -n "${build}" && -n "${xform}" && "${prep}" -lt "${build}" && "${prep}" -lt "${xform}" ]] \
   && sed -n "${prep},\$p" "${wf}" | grep -q 'RESIDENT_UP_LOG_ROOT=.*GITHUB_ENV'; then
  pass "the ranked job empties the log root before the build and the transform"
else
  fail "the ranked job does not empty the log root before entry code (prepare ${prep:-none}, setup ${build:-none}, transform ${xform:-none})"
fi

if [[ ${FAILS} -eq 0 ]]; then echo "test-resident-log-root.sh: all cases passed"; exit 0; fi
echo "test-resident-log-root.sh: ${FAILS} case(s) failed"; exit 1
