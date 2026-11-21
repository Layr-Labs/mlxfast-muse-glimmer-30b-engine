#!/usr/bin/env bash
# resident-up.sh -- boot a resident bench-worker, either FOR ONE LEG (benchd
# drives it: --boot / --stop) or around one command (the local wrapper form).
#
# TWO FORMS, AND THE RANKED PATH USES THE FIRST.
#
#   tools/resident-up.sh --boot --spec <serial|dflash> --draft-len <N> \
#                        --socket-out <file>
#   tools/resident-up.sh --stop --socket <path>
#
#     The PER-LEG form. benchd calls it once per leg, and it calls THE COPY IN
#     THAT LEG'S OWN TREE. The boot loads that tree's own engine and that tree's
#     own weights, waits for a healthy hello, writes the socket path and exits 0
#     with the resident STILL RUNNING. benchd then measures the leg and calls
#     --stop. This is the Mac counterpart of the CUDA track's serve-up
#     convention, with the same argv.
#
#     WHY THIS EXISTS (public run 34230122059). The ranked measure script used
#     to boot ONE resident from the CANDIDATE tree and export
#     BENCH_WORKER_RESIDENT_SOCKET into benchd. Under the paired design the
#     reference leg then attached to the candidate's resident and was refused:
#     "resident holds <candidate>/weights but this phase asked for
#     <baseline-workspace>/weights". One resident per WINDOW is wrong when a
#     window has two legs on two trees. One resident per LEG is right, and only
#     benchd knows where a leg's boundaries are -- so benchd boots them.
#
#   tools/resident-up.sh --weights <dir> [...] -- <command> [args...]
#
#     The WRAPPER form, kept for the LOCAL UNSCORED path. It has no caller left
#     in this repository after the ranked path stopped wrapping: its remaining
#     users are a participant's hand run and anything that wants
#     BENCH_WORKER_RESIDENT_SOCKET exported around a command (tools/benchmark.sh
#     honours that variable). It is unchanged.
#
# WEIGHTS LOAD ONCE PER WINDOW. benchd spawns `bench-worker runtime-worker`
# once per phase (warmup, timed prefill, timed decode, correctness, and again
# for every leg). In process, each spawn loads the 18.5 GB checkpoint. So the
# weights get an OWNER: `bench-worker resident` (fork contract section 12f).
# This script boots exactly one of them, inside the caller's GPU-lock window,
# and exports BENCH_WORKER_RESIDENT_SOCKET. Every per-phase worker attaches to
# it and loads nothing. This is the Mac counterpart of the CUDA track's
# tools/serve-up.sh, with the same rules: caller owns the lock, one resident,
# no path leaves it up.
#
# GPU LOCK -- NOT TAKEN HERE, BUT REQUIRED. The caller owns the box GPU window
# (the ranked workflow holds /tmp/mtplx-gpu-exclusive.lock for the whole
# measurement) and this script boots the resident inside it. A resident holds
# ~18.5 GB of unified memory, so it must never outlive that window. The script
# checks that the lock is HELD (a non-blocking try-lock must fail) and refuses
# by name when nobody holds it. It never takes the lock itself, because a
# lock this script took would be released when this script exits, which is
# not the window.
#
# ONE RESIDENT PER BOX. A second resident would double-load the checkpoint.
# The script refuses by name when its pidfile names a live resident, or when a
# resident already answers on the socket.
#
# Usage:
#   tools/resident-up.sh --boot --spec <serial|dflash> --draft-len <N>
#                        --socket-out <file>
#   tools/resident-up.sh --stop --socket <path>
#   tools/resident-up.sh --weights <dir> [--hello-identity]
#                        [--socket <path>] -- <command> [args...]
#
# --boot:
#   --spec serial|dflash
#                      what this leg is. THE FLAG IS AUTHORITATIVE OVER WHAT THE
#                      LEG DOES. It is recorded and it is what the leg is called;
#                      a reference tree whose manifest declares a draft depth
#                      still boots a SERIAL control leg when benchd says serial,
#                      and the worker takes the decoder and the depth from the
#                      wire per request, never from the tree.
#                      `serial` boots the target alone. `dflash` also hands the
#                      engine the tree's quantized DFlash drafter
#                      (<tree>/weights/drafter, `--drafter`); the boot refuses
#                      when that directory or one of its two files is missing,
#                      and the engine checks both files against their pins when
#                      it loads them.
#   --draft-len <N>    0 with `serial`. With `dflash`, the depth the tree's own
#                      declaration names (tools/spec-declaration.sh describe
#                      must print dflash<N>), so the boot can only serve the
#                      declared depth. A mismatch refuses.
#   --socket-out <f>   the boot writes the resident's socket path as the FIRST
#                      LINE of this file. A `<socket>.pid` sidecar and a
#                      `<socket>.ready` marker are written beside the socket.
#                      Under the sandbox, a `<socket>.tmpdir` sidecar names the
#                      resident's private temporary directory, so that --stop
#                      can remove it.
#
#   The boot uses THE TREE THIS SCRIPT LIVES IN and nothing else: its own
#   .build/release/bench-worker, its own weights/. MLXFAST_ENGINE_BIN and
#   MLXFAST_WEIGHTS_PATH are IGNORED here on purpose -- the ranked job exports
#   them for the candidate, and honouring them would boot the candidate's engine
#   for the reference leg, which is the bug this verb fixes.
#
# --stop:
#   --socket <path>    the socket a --boot reported. The stop reads
#                      <path>.pid, ends that resident, and removes the socket,
#                      the pid sidecar, the ready marker, and the resident's
#                      private temporary directory that <path>.tmpdir names.
#                      It is IDEMPOTENT: a
#                      second stop, or a stop of a resident that already died,
#                      exits 0 and says so.
#
#   --weights <dir>    the transformed weights directory (`weights/` from
#                      setup.sh). Required.
#
#   The ranked --boot form ignores MLXFAST_MTP_HEAD_DIR: a dflash leg reads
#   the drafter of ITS OWN tree only. The wrapper form passes `--drafter` only
#   when that variable is set.
#   --hello-identity   export BENCH_WORKER_RESIDENT_HELLO=1 to the command, so
#                      each attached hello carries the resident's pid and
#                      load_epoch. Off by default until benchd admits the field.
#   --socket <path>    the resident's Unix socket. Default: a pid-keyed path
#                      under ${TMPDIR:-/tmp}. A Unix socket path holds 103
#                      bytes on macOS; a socket under the checkout would not
#                      bind on the ranked box.
#
# Environment (test seams and locations; none relaxes a refusal):
#   MLXFAST_ENGINE_BIN               the bench-worker binary benchd spawns
#                                    (default <repo>/.build/release/bench-worker,
#                                    the staged pair tools/stage-bench-worker.sh
#                                    writes, with mlx.metallib beside it; the
#                                    same variable and default the measure
#                                    script reads, so the resident and every
#                                    attaching phase run ONE binary)
#   RESIDENT_UP_LOG_DIR              pidfile, pgid file, identity file, resident
#                                    log (default <repo>/.build/resident)
#   RESIDENT_UP_LOCK_PATH            the GPU lock file (default
#                                    /tmp/mtplx-gpu-exclusive.lock)
#   RESIDENT_UP_HEALTH_TIMEOUT_S     ceiling on the load + first hello
#                                    (default 5400)
#   RESIDENT_UP_WIRED_LIMIT_READER   command printing iogpu.wired_limit_mb
#                                    (default `sysctl -n iogpu.wired_limit_mb`)
#   RESIDENT_UP_SANDBOX              1 (default) runs the resident under
#                                    sandbox-exec; 0 runs it bare, for the
#                                    hermetic stub tests only. 0 is refused on a
#                                    self-hosted runner (RUNNER_ENVIRONMENT).
#
# THE RESIDENT RUNS UNDER sandbox-exec. It executes the tree's own model code,
# and on the ranked box that is participant code. The profile denies every
# file write except the resident's socket and a private temporary directory
# under RESIDENT_UP_LOG_DIR (tmp.<pid>-<time>, mode 0700; every stop, failure
# and signal removes it), denies network except its own Unix socket, denies
# process-fork and every exec but the worker, and denies every read of the
# evaluator-only paths the runner environment names: MLXFAST_QWEN38_GOLDEN_DIR,
# MLXFAST_CORRECTNESS_GOLDEN_PATH, MLXFAST_PRIVATE_DIR,
# MLXFAST_BASELINE_WORKSPACE (unless this tree is that workspace),
# MLXFAST_BASELINE_CALIBRATION, BENCHD_BIN_DIR, the build cache root
# (MLXFAST_BUILD_CACHE_DIR, default ~/.cache/mlxfast-engine-build) and the
# runner registration files. Reads of the weights, the worker and its
# mlx.metallib, and the system paths Metal needs stay allowed.
#
# Exit codes: the wrapped command's exit code on a served window; 1 on a boot
# failure; 2 on a refusal before any load.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
log() { printf 'resident-up.sh: %s\n' "$*" >&2; }
die() { printf 'resident-up.sh: %s\n' "$*" >&2; exit 1; }
refuse() { printf 'resident-up.sh: REFUSED (%s): %s\n' "$1" "$2" >&2; exit 2; }

# remove_private_tmp DIR: remove the resident's private temporary directory and
# prove that it is gone. Participant code wrote this tree, so it can hold
# entries that the owner cannot list or remove as they are (mode 000, the
# user-immutable flag). Owner access is restored entry by entry, top down,
# before the tree is removed. Returns 1 when DIR or anything in it survives,
# so that the caller fails closed.
remove_private_tmp() {
  local dir="$1"
  [[ -e "${dir}" || -L "${dir}" ]] || return 0
  if [[ -L "${dir}" || ! -d "${dir}" || ! -O "${dir}" ]]; then
    log "'${dir}' is not a directory that this account owns; it is not removed"
    return 1
  fi
  find "${dir}" -exec /bin/sh -c '/usr/bin/chflags -h nouchg,nouappnd "$1"; /bin/chmod -h u+rwx "$1"' sh {} \; 2>/dev/null
  rm -rf -- "${dir}" 2>/dev/null
  if [[ -e "${dir}" || -L "${dir}" ]]; then
    log "FAILED to remove the resident's private temporary directory ${dir}"
    return 1
  fi
}

usage() {
  cat >&2 <<'USAGE'
usage:
  tools/resident-up.sh --boot --spec <serial|dflash> --draft-len <N> --socket-out <file>
  tools/resident-up.sh --stop --socket <path>
  tools/resident-up.sh --weights <dir> [--hello-identity] [--socket <path>] -- <command> [args...]
USAGE
  exit 2
}

# --- argv --------------------------------------------------------------------
# MODE is the verb. `wrap` is the historical form and stays the default, so a
# hand run and the local unscored path are unchanged. --boot and --stop are the
# per-leg form benchd drives.
MODE="wrap"
WEIGHTS_DIR=""
SOCKET_PATH=""
SOCKET_OUT=""
SPEC=""
DRAFT_LEN=""
HELLO_IDENTITY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --boot) [[ "${MODE}" == "wrap" ]] || refuse bad-argument "--boot and --stop are exclusive"; MODE="boot"; shift ;;
    --stop) [[ "${MODE}" == "wrap" ]] || refuse bad-argument "--boot and --stop are exclusive"; MODE="stop"; shift ;;
    --spec) [[ $# -ge 2 ]] || refuse bad-argument "--spec needs a value (serial or dflash)"; SPEC="$2"; shift 2 ;;
    --draft-len) [[ $# -ge 2 ]] || refuse bad-argument "--draft-len needs a value"; DRAFT_LEN="$2"; shift 2 ;;
    --socket-out) [[ $# -ge 2 ]] || refuse bad-argument "--socket-out needs a path"; SOCKET_OUT="$2"; shift 2 ;;
    --weights) [[ $# -ge 2 ]] || refuse bad-argument "--weights needs a directory"; WEIGHTS_DIR="$2"; shift 2 ;;
    --socket) [[ $# -ge 2 ]] || refuse bad-argument "--socket needs a path"; SOCKET_PATH="$2"; shift 2 ;;
    --hello-identity) HELLO_IDENTITY=1; shift ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) refuse bad-argument "unknown argument '$1'" ;;
  esac
done

# --- --stop: idempotent teardown, and nothing else ---------------------------
# It runs before every other check on purpose. A teardown must work on a box
# whose GPU lock is gone, whose wired limit was never pinned, and whose resident
# already died -- the failure modes a stop exists to clean up after.
if [[ "${MODE}" == "stop" ]]; then
  [[ -n "${SOCKET_PATH}" ]] || refuse bad-argument "--stop needs --socket <path>"
  stop_pidfile="${SOCKET_PATH}.pid"
  stopped=""
  if [[ -r "${stop_pidfile}" ]]; then
    stop_pid="$(tr -d '[:space:]' < "${stop_pidfile}")"
    if [[ "${stop_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${stop_pid}" 2>/dev/null; then
      # The resident leads its own process group (it was setsid'd at boot), so
      # the group signal reaches anything it spawned.
      kill -TERM -- "-${stop_pid}" 2>/dev/null || kill -TERM "${stop_pid}" 2>/dev/null || true
      for _ in $(seq 1 50); do
        kill -0 "${stop_pid}" 2>/dev/null || break
        sleep 0.2
      done
      if kill -0 "${stop_pid}" 2>/dev/null; then
        log "resident ${stop_pid} did not stop on SIGTERM; sending SIGKILL to its group"
        kill -KILL -- "-${stop_pid}" 2>/dev/null || kill -KILL "${stop_pid}" 2>/dev/null || true
        sleep 0.3
      fi
      if kill -0 "${stop_pid}" 2>/dev/null; then
        die "FAILED to halt the resident: pid ${stop_pid} survived SIGKILL (socket ${SOCKET_PATH})"
      fi
      stopped="${stop_pid}"
    fi
  fi
  # The resident's private temporary directory. It is mode 0700 and lives in
  # the tree's .build/resident by default, so a directory left behind there
  # is one that another account cannot list (ranked preflight 7a).
  # FAIL CLOSED. The sidecar is removed only after the directory is proven
  # gone; on any failure the stop exits 1 and the directory stays in the log
  # directory, where the next boot finds it and refuses (private-tmp-leftover).
  stop_tmpfile="${SOCKET_PATH}.tmpdir"
  if [[ -e "${stop_tmpfile}" ]]; then
    [[ -r "${stop_tmpfile}" ]] || die "cannot read ${stop_tmpfile}; the resident's temporary directory is unknown, so the leg is not clean"
    stop_tmp="$(head -n 1 "${stop_tmpfile}")"
    stop_log_dir="${RESIDENT_UP_LOG_DIR:-${SCRIPT_DIR}/.build/resident}"
    if [[ "${stop_tmp}" != "${stop_log_dir%/}/tmp."* || "${stop_tmp#"${stop_log_dir%/}/"}" == */* || "${stop_tmp}" == *..* ]]; then
      die "${stop_tmpfile} names '${stop_tmp}', which is not a tmp.* directory directly in ${stop_log_dir}; nothing removed, and the leg is not clean"
    fi
    remove_private_tmp "${stop_tmp}" \
      || die "the resident's temporary directory ${stop_tmp} survived the stop; ${stop_tmpfile} is kept, and the next boot refuses"
  fi
  rm -f "${SOCKET_PATH}" "${stop_pidfile}" "${SOCKET_PATH}.ready" "${stop_tmpfile}" 2>/dev/null
  for left in "${SOCKET_PATH}" "${stop_pidfile}" "${SOCKET_PATH}.ready" "${stop_tmpfile}"; do
    [[ ! -e "${left}" && ! -L "${left}" ]] || die "could not remove ${left}"
  done
  if [[ -n "${stopped}" ]]; then
    log "stopped the resident on ${SOCKET_PATH} (pid ${stopped}); socket, pid sidecar and ready marker removed"
  else
    log "nothing to stop on ${SOCKET_PATH} (no live resident); socket, pid sidecar and ready marker removed if present"
  fi
  exit 0
fi

# --- --boot: the leg's own tree, and only its own tree ------------------------
# THE FLAGS DECIDE WHAT THE LEG IS. A reference tree whose manifest declares a
# depth still boots the SERIAL control leg when benchd says serial, and the
# worker still takes the decoder and the depth from the wire per request. The
# tree's declaration is read further down only to validate it.
if [[ "${MODE}" == "boot" ]]; then
  [[ -n "${SPEC}" ]] || refuse bad-argument "--boot needs --spec <serial|dflash>"
  [[ -n "${DRAFT_LEN}" ]] || refuse bad-argument "--boot needs --draft-len <N>"
  [[ -n "${SOCKET_OUT}" ]] || refuse bad-argument "--boot needs --socket-out <file>"
  case "${SPEC}" in
    serial|dflash) : ;;
    *) refuse bad-argument "--spec must be 'serial' or 'dflash' (got '${SPEC}')" ;;
  esac
  [[ "${DRAFT_LEN}" =~ ^[0-9]+$ ]] || refuse bad-argument "--draft-len must be a non-negative integer (got '${DRAFT_LEN}')"
  if [[ "${SPEC}" == "serial" && "${DRAFT_LEN}" != "0" ]]; then
    refuse bad-argument "--spec serial requires --draft-len 0 (got ${DRAFT_LEN}); a serial control leg drafts nothing"
  fi
  if [[ "${SPEC}" == "dflash" && "${DRAFT_LEN}" == "0" ]]; then
    refuse bad-argument "--spec dflash requires a draft length of 1 or more (got 0); depth 0 is the serial leg"
  fi
  # THE TREE THIS SCRIPT LIVES IN, and nothing the environment says. The ranked
  # job exports MLXFAST_ENGINE_BIN and MLXFAST_WEIGHTS_PATH for the CANDIDATE;
  # honouring either here would boot the candidate's engine or read the
  # candidate's weights for the reference leg.
  cd "${SCRIPT_DIR}"
  WEIGHTS_DIR="${SCRIPT_DIR}/weights"
fi

if [[ "${MODE}" == "wrap" ]]; then
  [[ $# -ge 1 ]] || refuse bad-argument "no command after '--'"
fi
[[ -n "${WEIGHTS_DIR}" ]] || refuse bad-argument "--weights <dir> is required"
[[ -d "${WEIGHTS_DIR}" ]] || refuse weights-missing "weights directory is missing: '${WEIGHTS_DIR}'"

# THE DRAFTER. A serial leg loads the target alone. A dflash leg also loads
# the quantized DFlash drafter, and only the one in THIS tree
# (MTP_HEAD_DEFAULT): tools/make-drafter.sh writes it there and verifies both
# of its pinned files. The boot first checks that the depth benchd asks for is
# the depth this tree declares (its own trusted reader), then that the drafter
# is staged. The engine checks both files against their pins when it loads
# them, so a wrong file stops the load by name. The wrapper form can still name
# a drafter with MLXFAST_MTP_HEAD_DIR for diagnostics.
MTP_HEAD_DEFAULT="${SCRIPT_DIR}/weights/drafter"
DRAFTER_ARGS=()
if [[ "${MODE}" == "boot" ]]; then
  RESIDENT_DECODER="serial"
  DRAFTER_DIR=""
  if [[ "${SPEC}" == "dflash" ]]; then
    [[ -x "${SCRIPT_DIR}/tools/spec-declaration.sh" ]] \
      || refuse declaration-missing "the dflash leg's tree has no tools/spec-declaration.sh to read its declared depth"
    declared="$("${SCRIPT_DIR}/tools/spec-declaration.sh" describe 2>&1)" \
      || refuse declaration-invalid "the tree's declaration is refused by tools/spec-declaration.sh: ${declared}"
    [[ "${declared}" == "dflash${DRAFT_LEN}" ]] \
      || refuse spec-mismatch "benchd asked for dflash depth ${DRAFT_LEN}, but this tree declares '${declared}' (tools/spec-declaration.sh describe)"
    DRAFTER_DIR="${MTP_HEAD_DEFAULT}"
    for drafter_file in config.json model.safetensors; do
      [[ -f "${DRAFTER_DIR}/${drafter_file}" && ! -L "${DRAFTER_DIR}/${drafter_file}" ]] \
        || refuse drafter-missing "the dflash leg needs the quantized DFlash drafter at ${DRAFTER_DIR} (config.json and model.safetensors, regular files), and ${DRAFTER_DIR}/${drafter_file} is not there; produce it in this tree with tools/make-drafter.sh"
    done
    DRAFTER_ARGS=(--drafter "${DRAFTER_DIR}")
    RESIDENT_DECODER="dflash"
  fi
else
  DRAFTER_DIR="${MLXFAST_MTP_HEAD_DIR:-}"
  if [[ -n "${DRAFTER_DIR}" ]]; then
    [[ -d "${DRAFTER_DIR}" ]] || refuse drafter-missing "drafter directory (MLXFAST_MTP_HEAD_DIR) is missing: '${DRAFTER_DIR}'"
    DRAFTER_ARGS=(--drafter "${DRAFTER_DIR}")
    RESIDENT_DECODER="dflash"
  else
    RESIDENT_DECODER="serial"
  fi
fi

command -v python3 >/dev/null 2>&1 || refuse tool-missing "python3 is required (it speaks the resident's hello probe and the lock check)"

# In --boot the binary is THIS TREE'S staged worker, full stop: an inherited
# MLXFAST_ENGINE_BIN names the candidate's, and the reference leg must not run
# it. The wrapper form keeps honouring the variable, because there the caller
# and the tree are the same thing.
if [[ "${MODE}" == "boot" ]]; then
  BENCH_WORKER="${SCRIPT_DIR}/.build/release/bench-worker"
  if [[ -n "${MLXFAST_ENGINE_BIN:-}" && "${MLXFAST_ENGINE_BIN}" != "${BENCH_WORKER}" ]]; then
    log "ignoring MLXFAST_ENGINE_BIN=${MLXFAST_ENGINE_BIN}; --boot runs THIS tree's worker (${BENCH_WORKER}), because the leg is defined by its tree"
  fi
else
  BENCH_WORKER="${MLXFAST_ENGINE_BIN:-${SCRIPT_DIR}/.build/release/bench-worker}"
fi
[[ -x "${BENCH_WORKER}" ]] || refuse worker-missing "bench-worker is missing or not executable: ${BENCH_WORKER} (build and stage it: setup.sh, or swift build -c release --scratch-path .build-worker --product bench-worker && tools/stage-bench-worker.sh)"

LOG_DIR="${RESIDENT_UP_LOG_DIR:-${SCRIPT_DIR}/.build/resident}"
LOCK_PATH="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
HEALTH_TIMEOUT_S="${RESIDENT_UP_HEALTH_TIMEOUT_S:-5400}"
[[ "${HEALTH_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]] || refuse bad-argument "RESIDENT_UP_HEALTH_TIMEOUT_S must be a positive integer (got '${HEALTH_TIMEOUT_S}')"
WIRED_LIMIT_READER="${RESIDENT_UP_WIRED_LIMIT_READER:-sysctl -n iogpu.wired_limit_mb}"

mkdir -p "${LOG_DIR}"
# A tmp.* directory here is a previous resident's private temporary directory
# that its stop could not remove. Participant code wrote it, so no new leg
# boots while it exists.
leftover_tmp="$(find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -name 'tmp.*' -print)" \
  || refuse private-tmp-leftover "cannot list ${LOG_DIR}; a previous leg's temporary directory may remain. Nothing has been loaded."
[[ -z "${leftover_tmp}" ]] \
  || refuse private-tmp-leftover "a previous resident's temporary directory remains: $(tr '\n' ' ' <<< "${leftover_tmp}")(its stop failed). Nothing has been loaded."
# THE FIXED LOG ROOT. The ranked measurement puts each measurement's LOG_DIR
# in one fixed parent, RESIDENT_UP_LOG_ROOT (tools/resident-log-root.sh empties
# it first). LOG_DIR must sit directly in it, and nothing else may remain in it:
# a sibling is an earlier measurement's leftover, so no leg boots. The Seatbelt
# profile then denies the whole root, not only this LOG_DIR.
DENY_DIR="${LOG_DIR}"
if [[ -n "${RESIDENT_UP_LOG_ROOT:-}" ]]; then
  log_root_real="$(cd "${RESIDENT_UP_LOG_ROOT}" 2>/dev/null && pwd -P)" \
    || refuse measurement-leftover "the log root ${RESIDENT_UP_LOG_ROOT} cannot be entered. Nothing has been loaded."
  log_dir_real="$(cd "${LOG_DIR}" && pwd -P)"
  [[ "$(dirname "${log_dir_real}")" == "${log_root_real}" ]] \
    || refuse bad-argument "RESIDENT_UP_LOG_DIR ${LOG_DIR} is not directly in RESIDENT_UP_LOG_ROOT ${RESIDENT_UP_LOG_ROOT}"
  siblings="$(find "${log_root_real}" -mindepth 1 -maxdepth 1 ! -path "${log_dir_real}" -print)" \
    || refuse measurement-leftover "cannot list ${log_root_real}. Nothing has been loaded."
  [[ -z "${siblings}" ]] \
    || refuse measurement-leftover "an earlier measurement left $(tr '\n' ' ' <<< "${siblings}")in ${log_root_real}. Nothing has been loaded."
  DENY_DIR="${log_root_real}"
fi
RUN_TAG="$$-$(date -u +%Y%m%dT%H%M%SZ)"
SOCKET_PATH="${SOCKET_PATH:-${TMPDIR:-/tmp}/bench-worker-resident.${RUN_TAG}.sock}"
PIDFILE="${LOG_DIR}/resident.pid"
PGIDFILE="${LOG_DIR}/resident.pgid"
IDENTITY_FILE="${LOG_DIR}/resident-identity.json"
RESIDENT_LOG="${LOG_DIR}/resident.${RUN_TAG}.log"

# --- the GPU lock must be HELD by the caller ---------------------------------
# A non-blocking try-lock on a NEW open file description fails only when
# someone holds the lock. Success means nobody does: the caller is not inside
# a GPU window, and the try-lock is released at once.
lock_is_held() {
  # 0 held, 1 nobody holds it, 2 missing or unsafe (holds data, writable, a link).
  python3 "${SCRIPT_DIR}/tools/gpu-lock.py" held "$1"
}
[[ -e "${LOCK_PATH}" ]] || refuse lock-missing "the GPU lock file does not exist: ${LOCK_PATH} (see the runbook, step 2)"
lock_rc=0; lock_is_held "${LOCK_PATH}" || lock_rc=$?
[[ ${lock_rc} -ne 2 ]] || refuse lock-unsafe "the GPU lock ${LOCK_PATH} is not an empty read-only regular file; entry code may have written to it. Nothing has been loaded."
if [[ ${lock_rc} -ne 0 ]]; then
  refuse lock-not-held "nobody holds ${LOCK_PATH}; the caller must take the GPU lock for the whole window before booting a resident (flock, then run this script inside it). Nothing has been loaded."
fi

# --- the wired limit must be pinned ------------------------------------------
# The box runbook's boot daemon pins iogpu.wired_limit_mb so an 18.5 GB resident
# can be wired. This script verifies the pin and never sets it.
wired_limit_mb="$(${WIRED_LIMIT_READER} 2>/dev/null | tr -d '[:space:]' || true)"
if ! [[ "${wired_limit_mb}" =~ ^[0-9]+$ ]] || (( wired_limit_mb <= 0 )); then
  refuse wired-limit-unpinned "iogpu.wired_limit_mb is not pinned (reader '${WIRED_LIMIT_READER}' gave '${wired_limit_mb:-nothing}'); the boot daemon of the runbook sets it. Nothing has been loaded."
fi

# --- one resident per box ----------------------------------------------------
# The hello probe: connect, read the first line. The worker sends its hello
# unprompted at session start (hello is not a request kind in Engine Protocol
# v1), so a connection that receives an ok:true hello has found a live
# resident. The probe closes at once, which ends its session.
probe_resident() {
  python3 - "$1" <<'PY'
import json, socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(sys.argv[1])
    line = b""
    while not line.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        line += chunk
    s.close()
    reply = json.loads(line.decode())
except Exception as err:                       # noqa: BLE001 - any failure is unhealthy
    print(f"probe failed: {err}", file=sys.stderr)
    sys.exit(1)
if reply.get("ok") is not True:
    print(f"probe refused: {reply}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: reply.get(k) for k in
                  ("backend", "device", "protocol_version", "spec_modes", "runner", "resident")}))
PY
}

# ONE RESIDENT PER BOX, ACROSS TREES. The pidfile below lives under the tree
# that booted, so it cannot see a resident booted from the OTHER tree -- and
# under the paired design there are exactly two trees. A box-wide scan is what
# catches leg 2 booting before leg 1 was stopped, which would double-load 18.5 GB.
if [[ "${MODE}" == "boot" ]]; then
  live_residents="$(pgrep -f 'bench-worker resident' 2>/dev/null | tr '\n' ' ' | sed 's/ *$//' || true)"
  if [[ -n "${live_residents}" ]]; then
    refuse resident-already-up "a bench-worker resident is already running on this box (pid(s) ${live_residents}); the previous leg was not stopped. Stop it first (tools/resident-up.sh --stop --socket <path>). Nothing has been loaded."
  fi
fi

if [[ -r "${PIDFILE}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${PIDFILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    old_args="$(ps -o args= -p "${old_pid}" 2>/dev/null || true)"
    case "${old_args}" in
      *bench-worker*resident*)
        refuse resident-already-up "a resident is already running: pid ${old_pid} ('${old_args}'), recorded in ${PIDFILE}. One resident per box; halt it first (tools/resident-halt: SIGTERM the pgid in ${PGIDFILE}). Nothing has been loaded." ;;
      *)
        log "stale pidfile ${PIDFILE}: pid ${old_pid} is not a resident ('${old_args}'); removing it" ;;
    esac
  fi
  rm -f "${PIDFILE}" "${PGIDFILE}"
fi
if [[ -S "${SOCKET_PATH}" ]]; then
  if probe_resident "${SOCKET_PATH}" >/dev/null 2>&1; then
    refuse resident-already-up "a resident already answers on ${SOCKET_PATH}. One resident per box. Nothing has been loaded."
  fi
  log "stale socket ${SOCKET_PATH} (nothing answers); removing it"
  rm -f "${SOCKET_PATH}"
fi

# --- teardown ----------------------------------------------------------------
# ALWAYS. The resident holds the GPU and ~18.5 GB of unified memory, so it never
# outlives this script: the window that booted it tears it down, on success and
# on failure alike. SIGTERM to the process, then the process group, then
# SIGKILL to the group; the script then proves that nothing of the group is
# left. There is deliberately no path that leaves it up.
RESIDENT_PID=""
RESIDENT_PGID=""
RESIDENT_TMP_DIR=""
BOOT_DONE=0

remove_resident_tmp() {
  [[ -z "${RESIDENT_TMP_DIR}" ]] || remove_private_tmp "${RESIDENT_TMP_DIR}" || exit 1
}

group_members() { ps -o pid= -g "${RESIDENT_PGID}" 2>/dev/null | tr -d ' ' | grep -v '^$' || true; }

teardown() {
  if [[ -n "${RESIDENT_PID}" ]] && kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    kill -TERM "${RESIDENT_PID}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${RESIDENT_PID}" 2>/dev/null || break
      sleep 0.2
    done
  fi
  if [[ -n "${RESIDENT_PGID}" && "${RESIDENT_PGID}" != "$$" ]] && [[ -n "$(group_members)" ]]; then
    kill -TERM -- "-${RESIDENT_PGID}" 2>/dev/null || true
    for _ in $(seq 1 25); do
      [[ -n "$(group_members)" ]] || break
      sleep 0.2
    done
    if [[ -n "$(group_members)" ]]; then
      log "resident did not stop on SIGTERM; killing process group ${RESIDENT_PGID}"
      kill -KILL -- "-${RESIDENT_PGID}" 2>/dev/null || true
      sleep 0.3
    fi
  fi
  # Verify it is gone. A survivor keeps its pid and pgid files, so the next
  # boot's resident-already-up check refuses, and teardown returns 1.
  if [[ -n "${RESIDENT_PID}" ]] && { kill -0 "${RESIDENT_PID}" 2>/dev/null \
       || [[ -n "${RESIDENT_PGID}" && "${RESIDENT_PGID}" != "$$" && -n "$(group_members)" ]]; }; then
    log "FAILED to halt the resident: pid ${RESIDENT_PID} or a process of group ${RESIDENT_PGID} survived SIGKILL; ${PIDFILE} is kept so the next leg refuses"
    return 1
  fi
  if [[ -n "${RESIDENT_PID}" ]]; then
    log "resident torn down; its log is ${RESIDENT_LOG}"
  fi
  rm -f "${SOCKET_PATH}" "${PIDFILE}" "${PGIDFILE}" 2>/dev/null || true
  remove_resident_tmp
}

# boot_exit: the --boot EXIT trap. A complete boot leaves the resident up (the
# contract). Every other exit after the launch -- a failed health probe, a
# failed identity or sidecar write, a signal -- stops the resident, verifies it
# is gone, removes this boot's partial sidecars, and exits non-zero.
boot_exit() {
  local rc=$?
  # Skip teardown only on a complete boot that also EXITS 0: a write that fails
  # after the boot is marked complete makes benchd reject the boot, so the
  # resident must not stay up.
  [[ "${BOOT_DONE}" == "1" && "${rc}" -eq 0 ]] && return 0
  if [[ -n "${RESIDENT_PID}" ]]; then
    teardown || exit 1
    rm -f "${SOCKET_PATH}.pid" "${SOCKET_PATH}.ready" "${SOCKET_PATH}.tmpdir" \
      ${SOCKET_OUT:+"${SOCKET_OUT}"} 2>/dev/null || true
  else
    remove_resident_tmp
  fi
  [[ "${rc}" -ne 0 ]] || exit 1
}
# THE TRAP IS THE WRAPPER FORM'S. A --boot deliberately EXITS with the resident
# still running -- that is the whole contract -- so it must not tear down on
# exit. Its trap (boot_exit) does nothing on a complete boot -- --stop ends that
# resident -- and on every other exit after the launch stops the resident and
# verifies it is gone, so a boot that failed at any step leaves nothing alive.
if [[ "${MODE}" == "wrap" ]]; then
  trap 'teardown || exit 1' EXIT
else
  trap boot_exit EXIT
fi
trap 'exit 130' INT TERM

# --- the resident's sandbox ----------------------------------------------------
# See the header. The profile and the resident's private TMPDIR live under
# LOG_DIR (RESIDENT_UP_LOG_DIR, default <tree>/.build/resident).
RESIDENT_SANDBOX="${RESIDENT_UP_SANDBOX:-1}"
case "${RESIDENT_SANDBOX}" in
  1) : ;;
  0)
    if [[ "${RUNNER_ENVIRONMENT:-}" == "self-hosted" ]]; then
      refuse sandbox-required "RESIDENT_UP_SANDBOX=0 is refused on a self-hosted runner; the resident runs participant code and must run under sandbox-exec. Nothing has been loaded."
    fi
    log "RESIDENT_UP_SANDBOX=0: the resident runs WITHOUT sandbox-exec (hermetic tests only)"
    ;;
  *) refuse bad-argument "RESIDENT_UP_SANDBOX must be 0 or 1 (got '${RESIDENT_SANDBOX}')" ;;
esac
RESIDENT_LAUNCH=()
if [[ "${RESIDENT_SANDBOX}" == "1" ]]; then
  [[ -x /usr/bin/sandbox-exec ]] || refuse sandbox-missing "/usr/bin/sandbox-exec is not available; the resident must run under it. Nothing has been loaded."
  RESIDENT_TMP_DIR="${LOG_DIR}/tmp.${RUN_TAG}"
  mkdir -m 700 "${RESIDENT_TMP_DIR}"
  RESIDENT_PROFILE="${LOG_DIR}/resident.${RUN_TAG}.sb"
  # LOG_DIR holds every leg's resident log, profile and identity file, and both
  # legs of a pair share it. The resident may read and write only its own
  # tmp.* directory in it, so that nothing one leg's participant code printed
  # or wrote reaches a later leg.
  # The profile rules live in one place, tools/seatbelt-profile.py, which the
  # transform wrapper (tools/sandboxed-cli.sh) also uses.
  # On a self-hosted runner the profile is an official profile: the generator
  # refuses when the environment does not name a path that the profile denies.
  RESIDENT_PROFILE_MODE=()
  [[ "${RUNNER_ENVIRONMENT:-}" != "self-hosted" ]] || RESIDENT_PROFILE_MODE=(--official)
  python3 "${SCRIPT_DIR}/tools/seatbelt-profile.py" "${RESIDENT_PROFILE_MODE[@]+"${RESIDENT_PROFILE_MODE[@]}"}" \
    --exec "${BENCH_WORKER}" --tree "${SCRIPT_DIR}" \
    --deny-subpath "${DENY_DIR}" --write-subpath "${RESIDENT_TMP_DIR}" --unix-socket "${SOCKET_PATH}" > "${RESIDENT_PROFILE}" \
    || refuse sandbox-profile "could not write the resident sandbox profile ${RESIDENT_PROFILE}. Nothing has been loaded."
  RESIDENT_LAUNCH=(/usr/bin/sandbox-exec -f "${RESIDENT_PROFILE}")
  log "the resident runs under sandbox-exec with ${RESIDENT_PROFILE} (TMPDIR ${RESIDENT_TMP_DIR})"
fi

# --- boot the one resident ---------------------------------------------------
# The resident leads its own process group (setsid through python, which macOS
# has and util-linux setsid it does not), so the halt path can signal the
# group and reach anything the worker spawned.
if [[ "${MODE}" == "boot" ]]; then
  log "booting the ${SPEC} leg's resident from ${SCRIPT_DIR}: ${BENCH_WORKER} resident --weights ${WEIGHTS_DIR} (${RESIDENT_DECODER}, declared draft length ${DRAFT_LEN}); ONE load for THIS leg"
else
  log "booting the resident: ${BENCH_WORKER} resident --weights ${WEIGHTS_DIR} (${RESIDENT_DECODER}); ONE load for the whole window"
fi
# Under the sandbox the argv is `sandbox-exec -f <profile> <worker> resident ...`;
# sandbox-exec applies the profile and then execs the worker in the same
# process, so the pid and the process group stay the resident's.
RESIDENT_ENV=()
if [[ "${RESIDENT_SANDBOX}" == "1" ]]; then
  RESIDENT_ENV=(env "TMPDIR=${RESIDENT_TMP_DIR}/")
fi
"${RESIDENT_ENV[@]+"${RESIDENT_ENV[@]}"}" python3 - "${RESIDENT_LAUNCH[@]+"${RESIDENT_LAUNCH[@]}"}" "${BENCH_WORKER}" resident \
  --weights "${WEIGHTS_DIR}" \
  "${DRAFTER_ARGS[@]+"${DRAFTER_ARGS[@]}"}" \
  --speculative-protocol v1.1 \
  --socket "${SOCKET_PATH}" >"${RESIDENT_LOG}" 2>&1 <<'PY' &
import os, sys
os.setsid()
os.execv(sys.argv[1], sys.argv[1:])
PY
RESIDENT_PID=$!
RESIDENT_PGID="${RESIDENT_PID}"
printf '%s\n' "${RESIDENT_PID}" > "${PIDFILE}"
printf '%s\n' "${RESIDENT_PGID}" > "${PGIDFILE}"

start="$(date +%s)"
while :; do
  if ! kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    log "the resident exited before it was healthy; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    exit 1
  fi
  if [[ -S "${SOCKET_PATH}" ]] && probe_resident "${SOCKET_PATH}" >/dev/null 2>&1; then
    break
  fi
  now="$(date +%s)"
  if (( now - start >= HEALTH_TIMEOUT_S )); then
    log "the resident was not healthy within ${HEALTH_TIMEOUT_S}s; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
log "resident healthy on ${SOCKET_PATH} after $(( $(date +%s) - start ))s; every phase now attaches instead of loading"

HELLO_JSON="$(probe_resident "${SOCKET_PATH}")" || die "the resident stopped answering before the run started"

# --- the window's identity ---------------------------------------------------
python3 - "${IDENTITY_FILE}" "${HELLO_JSON}" <<PY
import json, sys
json.dump({
    "engine": "bench-worker",
    "weight_owner": "bench-worker-resident",
    "resident_socket": "${SOCKET_PATH}",
    "resident_pid": ${RESIDENT_PID},
    "resident_pgid": ${RESIDENT_PGID},
    "bench_worker": "${BENCH_WORKER}",
    "weights_dir": "${WEIGHTS_DIR}",
    "hello_identity": ${HELLO_IDENTITY},
    "mode": "${MODE}",
    "spec": "${SPEC:-unset}",
    "draft_len": "${DRAFT_LEN:-unset}",
    "drafter_decoder": "${RESIDENT_DECODER}",
    "drafter_dir": "${DRAFTER_DIR}",
    "hello": json.loads(sys.argv[2]),
    "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
}, open(sys.argv[1], "w"), indent=2)
PY

# --- --boot ends here, with the resident STILL RUNNING -----------------------
# The contract with benchd: the socket path is the FIRST LINE of --socket-out,
# a <socket>.pid sidecar names the process --stop must end, and a
# <socket>.ready marker records that the hello was healthy. Then exit 0 and
# leave it up. The resident was setsid'd at boot, so it does not die with this
# script.
if [[ "${MODE}" == "boot" ]]; then
  mkdir -p "$(dirname "${SOCKET_OUT}")"
  printf '%s\n' "${SOCKET_PATH}" > "${SOCKET_OUT}"
  printf '%s\n' "${RESIDENT_PID}" > "${SOCKET_PATH}.pid"
  printf '%s %s %s\n' "${SPEC}" "${DRAFT_LEN}" "${RESIDENT_PID}" > "${SOCKET_PATH}.ready"
  [[ -z "${RESIDENT_TMP_DIR}" ]] || printf '%s\n' "${RESIDENT_TMP_DIR}" > "${SOCKET_PATH}.tmpdir"
  log "boot complete: ${SPEC} leg resident pid ${RESIDENT_PID} on ${SOCKET_PATH}; socket written to ${SOCKET_OUT}. Stop it with: tools/resident-up.sh --stop --socket ${SOCKET_PATH}"
  BOOT_DONE=1
  exit 0
fi

export BENCH_WORKER_RESIDENT_SOCKET="${SOCKET_PATH}"
if [[ "${HELLO_IDENTITY}" == "1" ]]; then
  export BENCH_WORKER_RESIDENT_HELLO=1
else
  unset BENCH_WORKER_RESIDENT_HELLO || true
fi
export RESIDENT_IDENTITY_FILE="${IDENTITY_FILE}"
export RESIDENT_UP_PID="${RESIDENT_PID}"

log "running: $*"
set +e
"$@"
rc=$?
set -e
exit "${rc}"
