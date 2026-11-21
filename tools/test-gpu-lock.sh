#!/usr/bin/env bash
# test-gpu-lock.sh -- prove tools/gpu-lock.py, the one opener of the fleet GPU
# lock, without a model, a GPU or a second real account.
#
# The fresh-box fault: a lock created under umask 022 with O_RDWR|O_CREAT 0666
# is mode 0644, so a second account cannot open it read-write. The helper opens
# O_RDONLY (flock needs no write access) and creates the lock mode 0444, so any
# account can open and lock it, and none can write data into it.
#   1. a fresh lock under umask 022 is mode 0444, empty, and an O_RDONLY open
#      of it (what the other account does) takes and sees the lock
#   2. a lock that holds data is refused; the program does not run
#   3. a lock writable by group or other is refused
#   4. an own empty 0644 lock (an old box) is set to 0444 by hold
#   5. a symbolic link as the lock is refused
#   6. held: 1 when nobody holds it, 0 when held, 2 when it holds data
#   7. calibrate-box.sh, the measure script and resident-up.sh all use the
#      helper; none opens the lock read-write any more
#   8. the Seatbelt profile denies entry reads and writes of the lock path, and
#      a process under sandbox-exec cannot write to it
set -uo pipefail
cd "$(dirname "$0")/.."
LOCK_PY="tools/gpu-lock.py"
WORK="$(mktemp -d)"
trap 'chmod -R u+w "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT
FAILS=0
pass() { echo "ok    $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }
# GNU stat first (hosted Ubuntu CI), BSD stat on macOS.
mode_of() { stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1" 2>/dev/null; }
size_of() { stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1" 2>/dev/null; }

# 1. fresh box
L1="${WORK}/fresh.lock"
out="$( (umask 022; python3 "${LOCK_PY}" hold "${L1}" 5 test run /bin/sh -c 'echo RAN') 2>&1)"
if [[ "${out}" == *RAN* && "$(mode_of "${L1}")" == "444" && "$(size_of "${L1}")" == "0" ]] \
   && python3 -c 'import fcntl,os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)' "${L1}"; then
  pass "a fresh lock is mode 0444 and empty; a read-only open by another account locks it"
else
  fail "fresh lock: mode $(mode_of "${L1}") size $(size_of "${L1}") out: ${out}"
fi

# 2. data in the lock
L2="${WORK}/data.lock"; printf 'leg A was here' > "${L2}"; chmod 0444 "${L2}"
out="$(python3 "${LOCK_PY}" hold "${L2}" 5 test run /bin/sh -c 'echo RAN' 2>&1)"; rc=$?
if [[ ${rc} -eq 2 && "${out}" != *RAN* && "${out}" == *"no data"* ]]; then
  pass "a lock that holds data is refused and nothing runs"
else
  fail "a lock with data was accepted (rc ${rc}): ${out}"
fi

# 3. group/other writable
L3="${WORK}/wide.lock"; : > "${L3}"; chmod 0666 "${L3}"
out="$(python3 "${LOCK_PY}" hold "${L3}" 5 test run /bin/sh -c 'echo RAN' 2>&1)"; rc=$?
if [[ ${rc} -eq 2 && "${out}" != *RAN* ]]; then
  pass "a lock writable by group or other is refused"
else
  fail "a 0666 lock was accepted (rc ${rc}): ${out}"
fi

# 4. own empty 0644 lock (old box)
L4="${WORK}/old.lock"; : > "${L4}"; chmod 0644 "${L4}"
out="$(python3 "${LOCK_PY}" hold "${L4}" 5 test run /bin/sh -c 'echo RAN' 2>&1)"; rc=$?
if [[ ${rc} -eq 0 && "${out}" == *RAN* && "$(mode_of "${L4}")" == "444" ]]; then
  pass "an own empty 0644 lock is set to 0444 before the program runs"
else
  fail "old 0644 lock: rc ${rc} mode $(mode_of "${L4}") out: ${out}"
fi

# 5. symlink
L5="${WORK}/link.lock"; ln -s "${L1}" "${L5}"
out="$(python3 "${LOCK_PY}" hold "${L5}" 5 test run /bin/sh -c 'echo RAN' 2>&1)"; rc=$?
if [[ ${rc} -eq 2 && "${out}" != *RAN* ]]; then
  pass "a symbolic link as the lock is refused"
else
  fail "a symlinked lock was accepted (rc ${rc}): ${out}"
fi

# 6. held
L6="${WORK}/held.lock"
python3 "${LOCK_PY}" held "${L1}"; r_free=$?
r_held="$(python3 "${LOCK_PY}" hold "${L1}" 5 test run /bin/sh -c "python3 '${LOCK_PY}' held '${L1}'; echo \$?" 2>&1)"
: > "${L6}"; chmod 0444 "${L6}"; chmod u+w "${L6}"; printf x > "${L6}"; chmod 0444 "${L6}"
python3 "${LOCK_PY}" held "${L6}" 2>/dev/null; r_data=$?
if [[ ${r_free} -eq 1 && "${r_held}" == "0" && ${r_data} -eq 2 ]]; then
  pass "held: 1 when free, 0 when held, 2 when the lock holds data"
else
  fail "held: free ${r_free} held '${r_held}' data ${r_data}"
fi

# 7. callers
bad=0
for f in tools/calibrate-box.sh tools/muse-glimmer-30b-measure-and-score.sh tools/resident-up.sh; do
  grep -q 'gpu-lock.py' "${f}" || { fail "${f} does not use tools/gpu-lock.py"; bad=1; }
  if grep -nE 'O_RDWR \| os\.O_CREAT|open\(sys\.argv\[1\], "a\+"\)' "${f}" >/dev/null; then
    fail "${f} still opens the GPU lock read-write"; bad=1
  fi
done
[[ ${bad} -eq 0 ]] && pass "calibrate-box.sh, the measure script and resident-up.sh open the lock through the helper only"

# 8. Seatbelt
L8="${WORK}/sb.lock"; : > "${L8}"; chmod 0444 "${L8}"
prof="$(RESIDENT_UP_LOCK_PATH="${L8}" python3 tools/seatbelt-profile.py --exec /bin/sh 2>&1)"
real8="$(cd "$(dirname "${L8}")" && pwd -P)/sb.lock"
if [[ "${prof}" == *"(deny file-read* file-write* (subpath \"${real8}\"))"* ]]; then
  pass "the Seatbelt profile denies entry reads and writes of the lock path"
else
  fail "the Seatbelt profile does not deny the lock path ${real8}"
fi
if [[ -x /usr/bin/sandbox-exec ]]; then
  chmod 0644 "${L8}"
  printf '%s\n' "${prof}" | sed 's/(deny process-fork)//' > "${WORK}/sb.prof"
  /usr/bin/sandbox-exec -f "${WORK}/sb.prof" /bin/sh -c "printf data >> '${L8}'" 2>/dev/null
  if [[ "$(size_of "${L8}")" == "0" ]]; then
    pass "a process under sandbox-exec cannot write to the lock"
  else
    fail "a process under sandbox-exec wrote $(size_of "${L8}") byte(s) into the lock"
  fi
else
  echo "SKIP (needs macOS)  case 8b: no /usr/bin/sandbox-exec on this host"
fi

if [[ ${FAILS} -eq 0 ]]; then echo "test-gpu-lock.sh: all cases passed"; exit 0; fi
echo "test-gpu-lock.sh: ${FAILS} case(s) failed"; exit 1
