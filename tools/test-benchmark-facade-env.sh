#!/usr/bin/env bash
#
# test-benchmark-facade-env.sh -- the benchd facade exports the track id.
#
# `benchd iterate` REQUIRES MLXFAST_QWEN_MTP_TRACK_ID in every mode (its
# `-{platform}-v{N}` suffix keys the official baseline pair; the official path
# resolves it before the gates-only branch), and this repository once exported
# it nowhere, so `./benchmark.sh --local-iterate` and `--official` both refused
# with "no track_id". tools/benchmark.sh now reads benchmark.json's `trackId`
# and exports it before dispatch. This suite pins that, by driving the REAL
# facade with a STUB benchd that records the environment and argv it was
# spawned with. Hermetic: no toolchain, weights, GPU, benchd binary or network.
#
# Cases:
#   1. --local-iterate: the stub sees MLXFAST_QWEN_MTP_TRACK_ID == benchmark.json
#      trackId, non-empty, on an `iterate --mode local-iterate` argv.
#   2. --official (gates-only env): the same export on `--mode official`.
#   3. a manifest with NO trackId: the facade refuses (exit 1, names the field)
#      and benchd is never spawned.
#   4. a caller pre-sets a DIFFERENT MLXFAST_QWEN_MTP_TRACK_ID: refused, never
#      overridden, benchd never spawned.
#   5. a caller pre-sets the SAME value: accepted.
#
# It then pins the RANKED path's WINDOW wiring on
# tools/muse-glimmer-30b-measure-and-score.sh, with a stub bench-worker that
# binds a socket and answers a hello the way the real resident does:
#   6. THE RANKED PATH BOOTS NO RESIDENT AND EXPORTS NO SOCKET. It used to boot
#      one from the CANDIDATE tree and hand benchd the socket; under the paired
#      design the REFERENCE leg then attached to the candidate's resident and
#      was refused (public run 34230122059). benchd boots one resident PER LEG
#      now, from that leg's own tree, so the measure script must hand it
#      nothing -- and it must still take the GPU lock, because the box needs one
#      loader whoever boots it.
#   7. an INHERITED BENCH_WORKER_RESIDENT_SOCKET is refused by name, and benchd
#      is never dispatched: a socket arriving through the runner service
#      environment would reproduce the same failure.
#   8. --preflight-only boots no resident and takes no lock: a pre-GPU dry run
#      loads nothing, so it must not open a GPU window.
#
# Cases 9-13 pin the PAIRED plumbing (organizer ruling 2026-09-08). A ranked run
# measures a serial-control leg on the organizer's reference tree and a
# candidate leg on this checkout, and the score is the LIVE ratio:
#   9.  the dispatch carries --baseline-workspace and --baseline-calibration
#       from MLXFAST_BASELINE_WORKSPACE / MLXFAST_BASELINE_CALIBRATION, and
#       --engine is RELATIVE to the checkout root, because benchd re-roots that
#       path under the reference workspace to find the control leg's worker.
#   10. MLXFAST_BASELINE_WORKSPACE unset: refused by name, benchd never spawned.
#   11. MLXFAST_BASELINE_CALIBRATION unset: refused by name, benchd never spawned.
#   12. --preflight-only does NOT require either: it is benchmark.json's
#       preSubmitCommand, which a participant runs off the box.
#   13. an engine OUTSIDE the checkout is refused: it has no relative form, so
#       the control leg would run the candidate's own binary.
#   14. a WEIGHTS directory outside the checkout is refused for the same reason:
#       the control leg would read the candidate's own transform.
#   15. --box is passed ONLY when RUNNER_NAME is absent. benchd lets RUNNER_NAME
#       win over the flag, so on a runner the flag is dead argv; on a hand run
#       it is the only box identity benchd has, and it comes from the
#       calibration file's own `box`.
#
# Cases 16-19 pin the live_goldens list. A ranked run measures each live
# golden, and benchd matches the golden flags by position:
#   16. every live golden reaches benchd as --golden, --golden-sha256,
#       --golden-bytes, --control-golden, --control-golden-sha256 and
#       --control-golden-bytes, one of each per golden, in fixture order.
#   17. an empty live_goldens list is refused by name.
#   18. a live golden with no timed_prompt_pool entry is refused by name.
#   19. a declared depth with no oracle for one live golden is refused by name.
#   20. with RESIDENT_UP_LOG_DIR unset, benchd gets a resident log directory
#       in the job's temporary directory, outside both trees, because the
#       ranked job account cannot write the reference workspace.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACADE="${REPO_ROOT}/tools/benchmark.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

command -v jq >/dev/null 2>&1 || { echo "test-benchmark-facade-env.sh: jq is required" >&2; exit 1; }

EXPECTED="$(jq -r '.trackId // empty' "${REPO_ROOT}/benchmark.json")"
if [[ -z "${EXPECTED}" ]]; then
  echo "test-benchmark-facade-env.sh: benchmark.json carries no trackId; nothing to pin" >&2
  exit 1
fi

# The stub benchd: records the track id env (or __UNSET__) and its argv, emits
# an empty JSON payload the way benchd would on stdout, exits 0.
STUB="${WORK}/benchd-stub"
cat > "${STUB}" <<'STUBEOF'
#!/usr/bin/env bash
# A `--help` probe records nothing: it asks what this benchd accepts, it is
# not a run.
for arg in "$@"; do
  if [[ "${arg}" == "--help" ]]; then
    echo "usage: benchd iterate --engine <bin> [--control-golden <path>]"
    exit 0
  fi
done
printf '%s\n' "${MLXFAST_QWEN_MTP_TRACK_ID-__UNSET__}" > "${STUB_CAPTURE_ENV}"
printf '%s\n' "$@" > "${STUB_CAPTURE_ARGV}"
echo '{}'
exit 0
STUBEOF
chmod 755 "${STUB}"

GOLDEN="${WORK}/golden.json"
echo '{}' > "${GOLDEN}"

# run_facade CASE FACADE_PATH [facade args...] -- runs a facade with the stub
# benchd; captures land in ${WORK}/<case>.env / .argv / .out; sets rc.
run_facade() {
  local case_name="$1" facade="$2"
  shift 2
  rm -f "${WORK}/${case_name}.env" "${WORK}/${case_name}.argv"
  env -u MLXFAST_QWEN_MTP_TRACK_ID \
    STUB_CAPTURE_ENV="${WORK}/${case_name}.env" \
    STUB_CAPTURE_ARGV="${WORK}/${case_name}.argv" \
    BENCHD="${STUB}" \
    MLXFAST_ENGINE_BIN="${STUB}" \
    MLXFAST_CORRECTNESS_GOLDEN_PATH="${GOLDEN}" \
    MLXFAST_SCORE_PATH="${WORK}/${case_name}.score.json" \
    MLXFAST_INTEGRITY_PATH="${WORK}/${case_name}.integrity.json" \
    "${EXTRA_ENV[@]}" \
    "${facade}" "$@" > "${WORK}/${case_name}.out" 2>&1
  rc=$?
}

# Case 1: --local-iterate exports the manifest's trackId.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1)
run_facade case1 "${FACADE}" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 1: facade exited ${rc} with the stub benchd; output: $(cat "${WORK}/case1.out")"
fi
if [[ ! -f "${WORK}/case1.env" ]]; then
  fail "case 1: benchd was never spawned"
elif [[ "$(cat "${WORK}/case1.env")" != "${EXPECTED}" ]]; then
  fail "case 1: benchd saw MLXFAST_QWEN_MTP_TRACK_ID='$(cat "${WORK}/case1.env")', expected '${EXPECTED}'"
fi
if [[ -f "${WORK}/case1.argv" ]]; then
  if [[ "$(head -n 1 "${WORK}/case1.argv")" != "iterate" ]]; then
    fail "case 1: benchd subcommand is not iterate: $(head -n 1 "${WORK}/case1.argv")"
  fi
  if ! grep -qx -- '--mode' "${WORK}/case1.argv" || ! grep -qx -- 'local-iterate' "${WORK}/case1.argv"; then
    fail "case 1: argv does not carry --mode local-iterate: $(tr '\n' ' ' < "${WORK}/case1.argv")"
  fi
fi

# Case 2: --official, gates-only env (the seam-1 run the ranked box takes).
EXTRA_ENV=(MLXFAST_BENCHMARK_SKIP_TIMED=1 MLXFAST_BENCHMARK_CHECK_GATES=1)
run_facade case2 "${FACADE}" --official
if [[ "${rc}" -ne 0 ]]; then
  fail "case 2: facade exited ${rc} in official mode; output: $(cat "${WORK}/case2.out")"
fi
if [[ ! -f "${WORK}/case2.env" ]]; then
  fail "case 2: benchd was never spawned in official mode"
elif [[ "$(cat "${WORK}/case2.env")" != "${EXPECTED}" ]]; then
  fail "case 2: official mode: benchd saw MLXFAST_QWEN_MTP_TRACK_ID='$(cat "${WORK}/case2.env")', expected '${EXPECTED}'"
fi
if [[ -f "${WORK}/case2.argv" ]] && ! grep -qx -- 'official' "${WORK}/case2.argv"; then
  fail "case 2: argv does not carry --mode official: $(tr '\n' ' ' < "${WORK}/case2.argv")"
fi

# Case 3: a manifest with NO trackId refuses before benchd is spawned. The
# facade is copied into a throwaway tree whose benchmark.json lacks the field;
# it resolves the manifest relative to its own location, so nothing else moves.
mkdir -p "${WORK}/notrack/tools"
cp "${FACADE}" "${WORK}/notrack/tools/benchmark.sh"
jq 'del(.trackId)' "${REPO_ROOT}/benchmark.json" > "${WORK}/notrack/benchmark.json"
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1)
run_facade case3 "${WORK}/notrack/tools/benchmark.sh" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 3: facade ran with a manifest that carries no trackId"
fi
if ! grep -q 'trackId' "${WORK}/case3.out"; then
  fail "case 3: refusal does not name trackId; got: $(cat "${WORK}/case3.out")"
fi
if [[ -f "${WORK}/case3.env" ]]; then
  fail "case 3: benchd was spawned despite the missing trackId"
fi

# Case 4: a caller pre-sets a DIFFERENT track id -> refused, not overridden.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 MLXFAST_QWEN_MTP_TRACK_ID=some-other-track-mlx-v9)
run_facade case4 "${FACADE}" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 4: facade ran with a caller-set track id that differs from benchmark.json"
fi
if ! grep -q 'MLXFAST_QWEN_MTP_TRACK_ID' "${WORK}/case4.out"; then
  fail "case 4: refusal does not name the variable; got: $(cat "${WORK}/case4.out")"
fi
if [[ -f "${WORK}/case4.env" ]]; then
  fail "case 4: benchd was spawned despite the track id mismatch"
fi

# Case 5: a caller pre-sets the SAME value -> accepted, same export.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 "MLXFAST_QWEN_MTP_TRACK_ID=${EXPECTED}")
run_facade case5 "${FACADE}" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 5: facade refused a caller-set track id equal to benchmark.json's; output: $(cat "${WORK}/case5.out")"
fi
if [[ ! -f "${WORK}/case5.env" ]] || [[ "$(cat "${WORK}/case5.env")" != "${EXPECTED}" ]]; then
  fail "case 5: benchd did not see the expected track id"
fi

# ---------------------------------------------------------------------------
# Cases 6-8: the ranked measure script opens the GPU window and NOTHING ELSE.
#
# Hermetic root: the REAL measure script and the REAL resident script, copied
# beside the REAL benchmark.json and track fixture so both resolve their own
# paths inside the sandbox. Nothing loads: the bench-worker is a stub that
# binds the socket and answers the hello, and benchd is a stub too.
# ---------------------------------------------------------------------------
MEASURE_REL="tools/muse-glimmer-30b-measure-and-score.sh"
RES_ROOT="${WORK}/measure-root"
mkdir -p "${RES_ROOT}/tools" "${RES_ROOT}/fixtures" "${RES_ROOT}/weights"
cp "${REPO_ROOT}/${MEASURE_REL}" "${RES_ROOT}/tools/"
cp "${REPO_ROOT}/tools/resident-up.sh" "${REPO_ROOT}/tools/gpu-lock.py" "${REPO_ROOT}/tools/resident-log-root.sh" "${RES_ROOT}/tools/"
# The measure script resolves the declared draft depth through the trusted
# declaration reader (tools/spec-declaration.sh); stage it beside the script.
cp "${REPO_ROOT}/tools/spec-declaration.sh" "${RES_ROOT}/tools/"
# ... and the warm-up golden through tools/warmup-golden.sh.
cp "${REPO_ROOT}/tools/warmup-golden.sh" "${RES_ROOT}/tools/"
# ... and the early-decode limit through tools/seed-window-limit.sh.
cp "${REPO_ROOT}/tools/seed-window-limit.sh" "${RES_ROOT}/tools/"
chmod +x "${RES_ROOT}/tools/"*.sh
cp "${REPO_ROOT}/benchmark.json" "${RES_ROOT}/benchmark.json"
cp "${REPO_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json" "${RES_ROOT}/fixtures/"

# THE GOLDEN MATERIAL, SYNTHESIZED INTO THIS SUITE'S OWN COPY OF THE FIXTURE.
#
# A freshly stamped track ships an EMPTY live_goldens and an EMPTY
# timed_prompt_pool: the real tapes are organizer material, published in R2 and
# staged on the box, and they do not exist for this track yet. The measure
# script refuses without them, and rightly so -- but that refusal is not what
# these cases are about, so the copy in ${RES_ROOT} gets a synthetic live
# golden, a pool entry pinned to the file staged beside it, and a per-depth
# oracle for every contract-permitted draft depth. The shipped fixture is never
# written. When the fixture DOES carry a pool, nothing below changes it.
#
# The arm state is synthesized the same way and for the same reason: an unarmed
# track refuses at --preflight-only's arm gate before it reaches any of the
# wiring these cases pin.
GOLDEN_DIR="${WORK}/goldens"
mkdir -p "${GOLDEN_DIR}"
python3 - "${RES_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json" "${GOLDEN_DIR}" <<'SYNTHEOF'
import hashlib, json, os, sys

contract_path, golden_dir = sys.argv[1:3]
contract = json.load(open(contract_path, encoding="utf-8"))
if not contract.get("live_goldens"):
    track = contract["track_id"]

    def stage(name):
        path = os.path.join(golden_dir, name + ".golden.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"synthetic_golden": name}) + "\n")
        data = open(path, "rb").read()
        return {
            "r2_path": "correctness_prompts/%s/%s.golden.json" % (track, name),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    live = ["synthetic-live-1", "synthetic-live-2"]
    contract["live_goldens"] = live
    contract["timed_prompt_pool"] = [stage(name) for name in live]
    contract["live_golden_speculative"] = {
        "dflash%d" % depth: {name: stage("%s.dflash%d" % (name, depth)) for name in live}
        for depth in contract.get("mtp_head", {}).get("permitted_draft_depths", [])
    }
    contract["official_scoring_enabled"] = True
# THE WARM-UP GOLDEN. The shipped pin stays the organizer placeholder until the
# warm-up prompt is recorded on the box, and the measure script refuses a
# placeholder by name (case 22b holds that). Every other case is about other
# wiring, so the copy gets a synthetic RECORDED pin over a staged file.
if "warmup_golden" in contract:
    path = os.path.join(golden_dir, "synthetic-warmup.golden.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"synthetic_warmup_golden": contract["track_id"]}) + "\n")
    data = open(path, "rb").read()
    contract["warmup_golden"] = {
        "r2_path": "correctness_prompts/%s/synthetic-warmup.golden.json" % contract["track_id"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
# THE EARLY-DECODE LIMITS. The shipped values stay organizer placeholders until
# they are measured on the box, and the measure script refuses a placeholder
# by name (case 24b holds that). The copy gets distinct measured-looking values
# (serial 1.25, dflashN 1.3 + N/10) so case 24a can tell which one benchd got.
if isinstance(contract.get("seed_window_max_ratio_by_spec"), dict):
    contract["seed_window_max_ratio_by_spec"] = {
        k: (1.25 if k == "serial" else round(1.3 + int(k[len("dflash"):]) / 10, 2))
        for k in contract["seed_window_max_ratio_by_spec"]
    }
with open(contract_path, "w", encoding="utf-8") as fh:
    json.dump(contract, fh, indent=2)
    fh.write("\n")
SYNTHEOF

# Each live golden has to exist under the name the fixture resolves to. When the
# block above staged it, it is already there with the bytes its pin names; when
# the fixture carried its own pool, an empty object is enough, because the bytes
# are never read here (benchd re-verifies the pin on a real run).
LIVE_GOLDEN_NAMES=()
while IFS= read -r name; do
  LIVE_GOLDEN_NAMES+=("${name}")
done < <(jq -r '.live_goldens[]' "${RES_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json")
for name in "${LIVE_GOLDEN_NAMES[@]}"; do
  [[ -f "${GOLDEN_DIR}/${name}.golden.json" ]] \
    || echo '{}' > "${GOLDEN_DIR}/${name}.golden.json"
done

# The stub benchd for these cases: it records whether the resident socket
# reached its environment, plus its argv, and answers an `iterate --help` probe
# with a usage line instead of recording it as a run.
MEASURE_STUB="${WORK}/benchd-measure-stub"
cat > "${MEASURE_STUB}" <<'MEASURESTUBEOF'
#!/usr/bin/env bash
for arg in "$@"; do
  if [[ "${arg}" == "--help" ]]; then
    echo "usage: benchd iterate --engine <bin> [--control-golden <path>] [--candidate-spec <JSON>]"
    exit 0
  fi
done
printf '%s\n' "${BENCH_WORKER_RESIDENT_SOCKET-__UNSET__}" > "${STUB_CAPTURE_ENV}"
printf '%s\n' "$@" > "${STUB_CAPTURE_ARGV}"
printf '%s\n' "${RESIDENT_UP_LOG_DIR-__UNSET__}" > "${STUB_CAPTURE_ARGV}.resident-log-dir"
echo '{}'
exit 0
MEASURESTUBEOF
chmod 755 "${MEASURE_STUB}"

# The stub bench-worker resident: records its argv, binds the socket, and sends
# the unprompted ok:true hello on every connection, as Engine Protocol v1 does.
WORKER_STUB="${WORK}/stub-bench-worker.py"
cat > "${WORKER_STUB}" <<'WORKERSTUBEOF'
#!/usr/bin/env python3
import json, os, signal, socket, sys
argv = sys.argv[1:]
with open(os.environ["STUB_ARGV_FILE"], "w") as f:
    f.write("\n".join(argv) + "\n")
if not argv or argv[0] != "resident":
    print("stub: first argument must be resident", file=sys.stderr); sys.exit(2)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
path = argv[argv.index("--socket") + 1]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(4)
hello = {"id": 0, "nonce": "stubnonce", "ok": True, "protocol_version": 1,
         "backend": "mlx-resident", "device": "stub", "spec_modes": ["serial", "mtp"],
         "runner": {"id": "layr/stub"}, "resident": {"pid": os.getpid()}}
while True:
    conn, _ = srv.accept()
    try:
        conn.sendall((json.dumps(hello) + "\n").encode())
        while conn.recv(65536):
            pass
    except OSError:
        pass
    finally:
        conn.close()
WORKERSTUBEOF
chmod +x "${WORKER_STUB}"

# THE STUB ENGINE LIVES INSIDE THE HERMETIC ROOT, at the FIXED path
# tools/stage-bench-worker.sh writes and benchd resolves. That is not a
# convenience: the measure script hands benchd a checkout-RELATIVE --engine so
# benchd can re-root it under the reference workspace, and an engine outside the
# checkout has no relative form (case 13 pins that refusal).
mkdir -p "${RES_ROOT}/.build/release"
cp "${WORKER_STUB}" "${RES_ROOT}/.build/release/bench-worker"
chmod +x "${RES_ROOT}/.build/release/bench-worker"
STAGED_WORKER_STUB="${RES_ROOT}/.build/release/bench-worker"

# THE PAIRED LEG PAIR. The measure script refuses a real run without both, so
# every measure case needs them staged. Nothing here is measured: the reference
# tree only has to exist and the calibration file only has to be a file --
# tools/ranked-box-preflight.sh is what validates their contents, and
# tools/test-ranked-box-preflight-env.sh is what pins that.
BASELINE_WORKSPACE="${WORK}/reference"
mkdir -p "${BASELINE_WORKSPACE}"
BASELINE_CALIBRATION="${WORK}/baseline-calibration.json"
CALIBRATION_BOX_NAME="synthetic-ranked-box"
printf '{"version":1,"box":"%s"}\n' "${CALIBRATION_BOX_NAME}" > "${BASELINE_CALIBRATION}"

WIRED_OK="${WORK}/wired-ok.sh"
printf '#!/bin/sh\necho 112000\n' > "${WIRED_OK}"
chmod +x "${WIRED_OK}"

# run_measure CASE [ENV=VAL...] [args...] -- drives the REAL measure script in
# the hermetic root. Captures land in ${WORK}/<case>.env / .argv / .out, and the
# resident's own argv in ${WORK}/<case>.resident-argv when one was booted.
run_measure() {
  local case_name="$1"
  shift
  rm -f "${WORK}/${case_name}.env" "${WORK}/${case_name}.argv" "${WORK}/${case_name}.resident-argv"
  local logdir="${WORK}/${case_name}.residentlog"
  mkdir -p "${logdir}"
  local extra=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do extra+=("$1"); shift; done
  env -u BENCH_WORKER_RESIDENT_SOCKET -u MLXFAST_QWEN_MTP_TRACK_ID -u RUNNER_NAME \
    STUB_CAPTURE_ENV="${WORK}/${case_name}.env" \
    STUB_CAPTURE_ARGV="${WORK}/${case_name}.argv" \
    STUB_ARGV_FILE="${WORK}/${case_name}.resident-argv" \
    BENCHD="${MEASURE_STUB}" \
    MLXFAST_ENGINE_BIN="${STAGED_WORKER_STUB}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
    MLXFAST_BASELINE_WORKSPACE="${BASELINE_WORKSPACE}" \
    MLXFAST_BASELINE_CALIBRATION="${BASELINE_CALIBRATION}" \
    MLXFAST_SCORE_PATH="${WORK}/${case_name}.score.json" \
    RESIDENT_UP_LOCK_PATH="${WORK}/${case_name}.gpu.lock" \
    RESIDENT_UP_LOG_DIR="${logdir}" \
    RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_OK}" \
    RESIDENT_UP_HEALTH_TIMEOUT_S=60 \
    MLXFAST_GPU_LOCK_TIMEOUT_S=60 \
    ${extra[@]+"${extra[@]}"} \
    "${RES_ROOT}/${MEASURE_REL}" "$@" > "${WORK}/${case_name}.out" 2>&1
  rc=$?
}

# Case 6: the ranked path boots NOTHING, exports NO socket, and still locks.
run_measure case6
if [[ "${rc}" -ne 0 ]]; then
  fail "case 6: the measure script exited ${rc}; output: $(cat "${WORK}/case6.out")"
fi
if [[ ! -f "${WORK}/case6.env" ]]; then
  fail "case 6: benchd was never dispatched; output: $(cat "${WORK}/case6.out")"
elif [[ "$(cat "${WORK}/case6.env")" != "__UNSET__" ]]; then
  fail "case 6: BENCH_WORKER_RESIDENT_SOCKET reached benchd ($(cat "${WORK}/case6.env")); benchd boots ONE resident per leg from that leg's own tree, and a socket from here would attach the REFERENCE leg to the CANDIDATE's resident -- public run 34230122059"
fi
if [[ -f "${WORK}/case6.resident-argv" ]]; then
  fail "case 6: the measure script booted a resident: $(tr '\n' ' ' < "${WORK}/case6.resident-argv"); the legs' residents are benchd's to boot"
fi
# The lock is still this script's to hold: a resident holds ~113 GB whoever
# booted it, so the box must have exactly one loader for the whole measurement.
if [[ ! -e "${WORK}/case6.gpu.lock" ]]; then
  fail "case 6: the GPU lock was never taken; every per-leg boot benchd makes refuses when nobody holds it"
fi
if [[ -f "${WORK}/case6.argv" ]] && ! grep -qx -- 'official' "${WORK}/case6.argv"; then
  fail "case 6: the dispatch is not --mode official: $(tr '\n' ' ' < "${WORK}/case6.argv")"
fi

# Case 7: an inherited socket is refused by name, before benchd is dispatched.
run_measure case7 "BENCH_WORKER_RESIDENT_SOCKET=${WORK}/inherited.sock"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 7: the measure script ran with an inherited BENCH_WORKER_RESIDENT_SOCKET"
elif ! grep -q "BENCH_WORKER_RESIDENT_SOCKET is set" "${WORK}/case7.out"; then
  fail "case 7: the refusal does not name the variable; got: $(cat "${WORK}/case7.out")"
fi
if [[ -f "${WORK}/case7.argv" ]]; then
  fail "case 7: benchd was dispatched despite the inherited socket"
fi
if [[ -f "${WORK}/case7.resident-argv" ]]; then
  fail "case 7: a resident was booted despite the inherited socket"
fi

# Case 8: a pre-GPU dry run opens no window.
run_measure case8 --preflight-only
if [[ "${rc}" -ne 0 ]]; then
  fail "case 8: --preflight-only exited ${rc}; output: $(cat "${WORK}/case8.out")"
fi
if [[ -f "${WORK}/case8.resident-argv" ]]; then
  fail "case 8: --preflight-only booted a resident: $(tr '\n' ' ' < "${WORK}/case8.resident-argv")"
fi
if [[ -e "${WORK}/case8.gpu.lock" ]]; then
  fail "case 8: --preflight-only took the GPU lock; a dry run loads nothing and must not open a window"
fi
if [[ -f "${WORK}/case8.argv" ]] && ! grep -qx -- 'validate-golden' "${WORK}/case8.argv"; then
  fail "case 8: --preflight-only did not dispatch validate-golden: $(tr '\n' ' ' < "${WORK}/case8.argv")"
fi

# ---------------------------------------------------------------------------
# Cases 9-13: the paired plumbing.
# ---------------------------------------------------------------------------

# argv_pair FILE FLAG VALUE -- the argv carries `FLAG VALUE` adjacently.
argv_pair() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
path, flag, value = sys.argv[1], sys.argv[2], sys.argv[3]
args = open(path, encoding="utf-8").read().splitlines()
for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args) and args[i + 1] == value:
        raise SystemExit(0)
raise SystemExit(1)
PYEOF
}

# argv_value FILE FLAG -- the value that follows FLAG, or nothing.
argv_value() {
  python3 - "$1" "$2" <<'PYEOF'
import sys
path, flag = sys.argv[1], sys.argv[2]
args = open(path, encoding="utf-8").read().splitlines()
for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args):
        print(args[i + 1])
        break
PYEOF
}

# Case 9: the dispatch carries both names, and --engine is checkout-RELATIVE.
run_measure case9
if [[ "${rc}" -ne 0 ]]; then
  fail "case 9: the measure script exited ${rc} with both paired names set; output: $(cat "${WORK}/case9.out")"
elif [[ ! -f "${WORK}/case9.argv" ]]; then
  fail "case 9: benchd was never dispatched; output: $(cat "${WORK}/case9.out")"
else
  argv_pair "${WORK}/case9.argv" --baseline-workspace "${BASELINE_WORKSPACE}" \
    || fail "case 9: argv carries no --baseline-workspace ${BASELINE_WORKSPACE}"
  argv_pair "${WORK}/case9.argv" --baseline-calibration "${BASELINE_CALIBRATION}" \
    || fail "case 9: argv carries no --baseline-calibration ${BASELINE_CALIBRATION}"
  # RELATIVE, not absolute. benchd re-roots this path under the reference
  # workspace to find the control leg's worker; an absolute path would name
  # this checkout's binary and make the candidate its own control -- a score
  # of 1 by construction.
  if [[ -f "${WORK}/case9.env" && "$(cat "${WORK}/case9.env")" != "__UNSET__" ]]; then
    fail "case 9: the paired dispatch carried BENCH_WORKER_RESIDENT_SOCKET: $(cat "${WORK}/case9.env")"
  fi
  engine_value="$(argv_value "${WORK}/case9.argv" --engine)"
  if [[ -z "${engine_value}" ]]; then
    fail "case 9: argv carries no --engine"
  elif [[ "${engine_value}" == /* ]]; then
    fail "case 9: --engine is absolute (${engine_value}); benchd re-roots this path under the reference workspace, so an absolute path runs the candidate binary as its own control"
  fi
fi

# Case 10: the reference tree is required on a real run.
run_measure case10 "MLXFAST_BASELINE_WORKSPACE="
if [[ "${rc}" -eq 0 ]]; then
  fail "case 10: the measure script ran a real measurement with no reference tree"
elif ! grep -q "MLXFAST_BASELINE_WORKSPACE is unset" "${WORK}/case10.out"; then
  fail "case 10: the refusal does not name the variable; got: $(cat "${WORK}/case10.out")"
fi
if [[ -f "${WORK}/case10.argv" ]]; then
  fail "case 10: benchd was dispatched despite the missing reference tree"
fi

# Case 11: the calibration file is required on a real run.
run_measure case11 "MLXFAST_BASELINE_CALIBRATION="
if [[ "${rc}" -eq 0 ]]; then
  fail "case 11: the measure script ran a real measurement with no health band"
elif ! grep -q "MLXFAST_BASELINE_CALIBRATION is unset" "${WORK}/case11.out"; then
  fail "case 11: the refusal does not name the variable; got: $(cat "${WORK}/case11.out")"
fi
if [[ -f "${WORK}/case11.argv" ]]; then
  fail "case 11: benchd was dispatched despite the missing calibration file"
fi

# Case 12: --preflight-only requires neither. It is benchmark.json's
# preSubmitCommand, and a participant runs it where no reference tree exists.
run_measure case12 "MLXFAST_BASELINE_WORKSPACE=" "MLXFAST_BASELINE_CALIBRATION=" --preflight-only
if [[ "${rc}" -ne 0 ]]; then
  fail "case 12: --preflight-only refused off the box, where no reference tree can be staged; output: $(cat "${WORK}/case12.out")"
fi

# Case 13: an engine outside the checkout has no relative form.
OUTSIDE_ENGINE="${WORK}/outside-bench-worker"
cp "${WORKER_STUB}" "${OUTSIDE_ENGINE}"
chmod +x "${OUTSIDE_ENGINE}"
run_measure case13 "MLXFAST_ENGINE_BIN=${OUTSIDE_ENGINE}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 13: an engine outside the checkout was accepted; the control leg would run the candidate's own binary"
elif ! grep -q "outside this checkout" "${WORK}/case13.out"; then
  fail "case 13: the refusal does not explain the re-rooting; got: $(cat "${WORK}/case13.out")"
fi

# Case 14: weights outside the checkout have no relative form either.
OUTSIDE_WEIGHTS="${WORK}/outside-weights"
mkdir -p "${OUTSIDE_WEIGHTS}"
run_measure case14 "MLXFAST_WEIGHTS_PATH=${OUTSIDE_WEIGHTS}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 14: weights outside the checkout were accepted; the control leg would read the candidate's own transform"
elif ! grep -q "outside this checkout" "${WORK}/case14.out"; then
  fail "case 14: the refusal does not explain the re-rooting; got: $(cat "${WORK}/case14.out")"
fi

# Case 15a: on a runner, --box is NOT passed. benchd reads RUNNER_NAME itself
# and lets it win over the flag, so passing it would be argv that cannot matter.
run_measure case15a "RUNNER_NAME=${CALIBRATION_BOX_NAME}"
if [[ "${rc}" -ne 0 ]]; then
  fail "case 15a: the measure script exited ${rc} with RUNNER_NAME set; output: $(cat "${WORK}/case15a.out")"
elif [[ ! -f "${WORK}/case15a.argv" ]]; then
  fail "case 15a: benchd was never dispatched; output: $(cat "${WORK}/case15a.out")"
elif grep -qx -- '--box' "${WORK}/case15a.argv"; then
  fail "case 15a: --box was passed while RUNNER_NAME is set; benchd lets RUNNER_NAME win, so the flag is dead argv"
fi

# Case 15b: on a hand run there is no RUNNER_NAME, so the flag carries the box
# name the calibration file itself records.
run_measure case15b
if [[ "${rc}" -ne 0 ]]; then
  fail "case 15b: the measure script exited ${rc} with RUNNER_NAME unset; output: $(cat "${WORK}/case15b.out")"
elif [[ ! -f "${WORK}/case15b.argv" ]]; then
  fail "case 15b: benchd was never dispatched; output: $(cat "${WORK}/case15b.out")"
elif ! argv_pair "${WORK}/case15b.argv" --box "${CALIBRATION_BOX_NAME}"; then
  fail "case 15b: argv does not carry --box ${CALIBRATION_BOX_NAME} from the calibration file: $(tr '\n' ' ' < "${WORK}/case15b.argv")"
fi

# ---------------------------------------------------------------------------
# Cases 16-19: the live_goldens list.
# ---------------------------------------------------------------------------
MEASURE_CONTRACT="${RES_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json"

# argv_values FILE FLAG -- every value that follows FLAG, one per line, in argv
# order.
argv_values() {
  python3 - "$1" "$2" <<'PYEOF'
import sys
path, flag = sys.argv[1], sys.argv[2]
args = open(path, encoding="utf-8").read().splitlines()
for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args):
        print(args[i + 1])
PYEOF
}

# pool_pins FIELD -- the timed_prompt_pool FIELD of each live golden, in
# live_goldens order.
pool_pins() {
  jq -r --arg f "$1" '. as $c | .live_goldens[] as $n
    | first($c.timed_prompt_pool[] | select(.r2_path | endswith("/" + $n + ".golden.json")))[$f]
    | tostring' "${MEASURE_CONTRACT}"
}

# Case 16: every live golden reaches benchd as repeated flags, in fixture order.
run_measure case16
if [[ "${rc}" -ne 0 ]]; then
  fail "case 16: the measure script exited ${rc}; output: $(cat "${WORK}/case16.out")"
elif [[ ! -f "${WORK}/case16.argv" ]]; then
  fail "case 16: benchd was never dispatched; output: $(cat "${WORK}/case16.out")"
else
  want_paths="$(printf "${GOLDEN_DIR}/%s.golden.json\n" "${LIVE_GOLDEN_NAMES[@]}")"
  want_shas="$(pool_pins sha256)"
  want_bytes="$(pool_pins bytes)"
  [[ "$(argv_values "${WORK}/case16.argv" --golden)" == "${want_paths}" ]] \
    || fail "case 16: the --golden values are not the live goldens in fixture order: $(argv_values "${WORK}/case16.argv" --golden | tr '\n' ' ')"
  [[ "$(argv_values "${WORK}/case16.argv" --golden-sha256)" == "${want_shas}" ]] \
    || fail "case 16: the --golden-sha256 values are not the live goldens' pins in fixture order"
  [[ "$(argv_values "${WORK}/case16.argv" --golden-bytes)" == "${want_bytes}" ]] \
    || fail "case 16: the --golden-bytes values are not the live goldens' pins in fixture order"
  [[ "$(argv_values "${WORK}/case16.argv" --control-golden)" == "${want_paths}" ]] \
    || fail "case 16: the --control-golden values are not the live goldens in fixture order: $(argv_values "${WORK}/case16.argv" --control-golden | tr '\n' ' ')"
  [[ "$(argv_values "${WORK}/case16.argv" --control-golden-sha256)" == "${want_shas}" ]] \
    || fail "case 16: the --control-golden-sha256 values are not the live goldens' pins in fixture order"
  [[ "$(argv_values "${WORK}/case16.argv" --control-golden-bytes)" == "${want_bytes}" ]] \
    || fail "case 16: the --control-golden-bytes values are not the live goldens' pins in fixture order"
fi

# run_measure_with_contract CASE FILTER [ENV=VAL...] -- run_measure against a
# copy of the fixture changed by the jq FILTER, then restore the fixture.
run_measure_with_contract() {
  local case_name="$1" filter="$2"
  shift 2
  cp "${MEASURE_CONTRACT}" "${WORK}/measure-contract.saved"
  jq "${filter}" "${WORK}/measure-contract.saved" > "${MEASURE_CONTRACT}"
  run_measure "${case_name}" "$@"
  cp "${WORK}/measure-contract.saved" "${MEASURE_CONTRACT}"
}

# expect_measure_refusal CASE NEEDLE -- the run refused, named NEEDLE, and never
# dispatched benchd.
expect_measure_refusal() {
  local case_name="$1" needle="$2"
  if [[ "${rc}" -eq 0 ]]; then
    fail "${case_name}: the measure script ran; it must refuse"
  elif ! grep -qF -- "${needle}" "${WORK}/${case_name}.out"; then
    fail "${case_name}: the refusal does not name '${needle}'; got: $(cat "${WORK}/${case_name}.out")"
  fi
  if [[ -f "${WORK}/${case_name}.argv" ]]; then
    fail "${case_name}: benchd was dispatched despite the refusal"
  fi
}

# Case 17: an empty list is refused by name.
run_measure_with_contract case17 '.live_goldens = []'
expect_measure_refusal case17 "fixture declares no live_goldens"

# Case 18: a live golden with no pool entry is refused by name.
run_measure_with_contract case18 '.live_goldens += ["no-such-prompt"]'
expect_measure_refusal case18 "live golden 'no-such-prompt' has no timed_prompt_pool entry"

# Case 19: a declared depth with no oracle for one live golden is refused by
# name. The last live golden loses its dflash3 oracle. The declaration names
# dflash, the decoder the fixture's allowed_modes declares.
SPEC_MANIFEST="${WORK}/dflash3.manifest.json"
printf '{"version":1,"source":"pinned","spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":3}}\n' > "${SPEC_MANIFEST}"
# The refusal names the FIRST live golden with no dflash3 oracle after the
# mutation. That is the last live golden when the fixture carries an oracle
# for every live golden; a serial-only armed fixture carries none, so the
# first live golden is refused. Read the name from the fixture, never a literal.
CASE19_MUTATION='del(.live_golden_speculative.dflash3[.live_goldens[-1]])'
LAST_LIVE="$(jq -er "${CASE19_MUTATION}"' | . as $f | [.live_goldens[] as $n | select((($f.live_golden_speculative.dflash3 // {}) | has($n)) | not) | $n][0]' "${RES_ROOT}/fixtures/muse_glimmer_30b_mlx_v1_track.json")"
run_measure_with_contract case19 "${CASE19_MUTATION}" \
  "SPEC_DECLARATION_MANIFEST=${SPEC_MANIFEST}"
expect_measure_refusal case19 "declared spec 'dflash3' has no live_golden_speculative entry for live golden '${LAST_LIVE}'"

# Case 22: the never-scored warm-up golden (the prompt-reuse guard).
# 22a: a recorded, staged pin reaches benchd as --warmup-golden plus its pins.
if [[ "$(jq -r 'has("warmup_golden")' "${MEASURE_CONTRACT}")" == "true" ]]; then
  run_measure case22a
  if [[ "${rc}" -ne 0 || ! -f "${WORK}/case22a.argv" ]]; then
    fail "case 22a: the measure script did not dispatch benchd (rc ${rc}); output: $(cat "${WORK}/case22a.out")"
  else
    want_warmup="${GOLDEN_DIR}/$(jq -r '.warmup_golden.r2_path | split("/") | last' "${MEASURE_CONTRACT}")"
    [[ "$(argv_values "${WORK}/case22a.argv" --warmup-golden)" == "${want_warmup}" ]] \
      || fail "case 22a: --warmup-golden is '$(argv_values "${WORK}/case22a.argv" --warmup-golden)', want ${want_warmup}"
    [[ "$(argv_values "${WORK}/case22a.argv" --warmup-golden-sha256)" == "$(jq -r '.warmup_golden.sha256' "${MEASURE_CONTRACT}")" ]] \
      || fail "case 22a: --warmup-golden-sha256 is not the fixture pin"
    [[ "$(argv_values "${WORK}/case22a.argv" --warmup-golden-bytes)" == "$(jq -r '.warmup_golden.bytes' "${MEASURE_CONTRACT}")" ]] \
      || fail "case 22a: --warmup-golden-bytes is not the fixture pin"
  fi
else
  fail "case 22a: the shipped fixture declares no warmup_golden; the prompt-reuse guard is off"
fi

# 22b: the organizer placeholder (the shipped state until it is recorded) is
# refused by name before benchd runs.
run_measure_with_contract case22b \
  '.warmup_golden = {"r2_path": "X-PENDING-ORGANIZER", "sha256": "X-PENDING-ORGANIZER", "bytes": 0}'
expect_measure_refusal case22b "warmup_golden not recorded"

# 22c: a recorded pin whose file is not staged is refused by name.
run_measure_with_contract case22c \
  '.warmup_golden.r2_path = "correctness_prompts/x/not-staged-warmup.golden.json"'
expect_measure_refusal case22c "the warm-up golden is not staged"

# 22d: a fixture with no warmup_golden sends no warm-up flag.
run_measure_with_contract case22d 'del(.warmup_golden)'
if [[ "${rc}" -ne 0 || ! -f "${WORK}/case22d.argv" ]]; then
  fail "case 22d: the measure script did not dispatch benchd (rc ${rc}); output: $(cat "${WORK}/case22d.out")"
elif grep -qx -- '--warmup-golden' "${WORK}/case22d.argv"; then
  fail "case 22d: --warmup-golden was sent for a fixture that declares none"
fi

# Case 24: the early-decode limit, one per declared spec (the seed-window guard).
# argv_contract CASE -- the --contract path benchd got.
argv_contract() { argv_values "${WORK}/$1.argv" --contract; }
if [[ "$(jq -r 'has("seed_window_max_ratio_by_spec")' "${MEASURE_CONTRACT}")" == "true" ]]; then
  # 24a: serial -> benchd gets a run contract = fixture + seed_window_max_ratio
  #      = the serial limit, and nothing else changed.
  run_measure case24a
  if [[ "${rc}" -ne 0 || ! -f "${WORK}/case24a.argv" ]]; then
    fail "case 24a: the measure script did not dispatch benchd (rc ${rc}); output: $(cat "${WORK}/case24a.out")"
  else
    c24="$(argv_contract case24a)"
    if [[ "${c24}" == "${MEASURE_CONTRACT}" || ! -f "${c24}" ]]; then
      fail "case 24a: benchd got '${c24}', want a run contract beside the fixture"
    elif [[ "$(jq -c '.seed_window_max_ratio' "${c24}")" != "1.25" ]]; then
      fail "case 24a: serial run contract seed_window_max_ratio is $(jq -c '.seed_window_max_ratio' "${c24}"), want 1.25"
    elif [[ "$(jq -S 'del(.seed_window_max_ratio)' "${c24}")" != "$(jq -S . "${MEASURE_CONTRACT}")" ]]; then
      fail "case 24a: the run contract differs from the fixture in more than seed_window_max_ratio"
    fi
    grep -qF "early-decode limit for serial: seed_window_max_ratio 1.25 on both legs" "${WORK}/case24a.out" \
      || fail "case 24a: the run log does not state the limit"
  fi
  # 24b: dflash3 -> the larger of the serial limit and the dflash3 limit (1.6).
  run_measure case24b "SPEC_DECLARATION_MANIFEST=${SPEC_MANIFEST}"
  if [[ "${rc}" -ne 0 || ! -f "${WORK}/case24b.argv" ]]; then
    fail "case 24b: the measure script did not dispatch benchd (rc ${rc}); output: $(cat "${WORK}/case24b.out")"
  elif [[ "$(jq -c '.seed_window_max_ratio' "$(argv_contract case24b)")" != "1.6" ]]; then
    fail "case 24b: dflash3 run contract seed_window_max_ratio is $(jq -c '.seed_window_max_ratio' "$(argv_contract case24b)"), want 1.6"
  fi
  # 24c: a declared spec whose limit is below serial is held to the serial limit
  #      (the control leg is serial and benchd holds both legs to one number).
  run_measure_with_contract case24c '.seed_window_max_ratio_by_spec.dflash3 = 1.1' \
    "SPEC_DECLARATION_MANIFEST=${SPEC_MANIFEST}"
  if [[ "${rc}" -ne 0 || ! -f "${WORK}/case24c.argv" ]]; then
    fail "case 24c: the measure script did not dispatch benchd (rc ${rc}); output: $(cat "${WORK}/case24c.out")"
  elif [[ "$(jq -c '.seed_window_max_ratio' "$(argv_contract case24c)")" != "1.25" ]]; then
    fail "case 24c: want the serial limit 1.25, got $(jq -c '.seed_window_max_ratio' "$(argv_contract case24c)")"
  fi
  # 24d: the organizer placeholder for the declared spec is refused by name.
  run_measure_with_contract case24d '.seed_window_max_ratio_by_spec.dflash3 = "X-PENDING-ORGANIZER"' \
    "SPEC_DECLARATION_MANIFEST=${SPEC_MANIFEST}"
  expect_measure_refusal case24d "seed_window_max_ratio not measured"
  # 24e: the serial placeholder is refused for a serial declaration too.
  run_measure_with_contract case24e '.seed_window_max_ratio_by_spec.serial = "X-PENDING-ORGANIZER"'
  expect_measure_refusal case24e "seed_window_max_ratio not measured"
  # 24f: a top-level scalar beside the map is two sources for one limit.
  run_measure_with_contract case24f '.seed_window_max_ratio = 1.5'
  expect_measure_refusal case24f "declares both seed_window_max_ratio and seed_window_max_ratio_by_spec"
  # 24g: no map -> benchd gets the fixture itself.
  run_measure_with_contract case24g 'del(.seed_window_max_ratio_by_spec)'
  if [[ "${rc}" -ne 0 || ! -f "${WORK}/case24g.argv" ]]; then
    fail "case 24g: the measure script did not dispatch benchd (rc ${rc}); output: $(cat "${WORK}/case24g.out")"
  elif [[ "$(argv_contract case24g)" != "${MEASURE_CONTRACT}" ]]; then
    fail "case 24g: benchd got '$(argv_contract case24g)', want the fixture itself"
  fi
else
  fail "case 24: the shipped fixture declares no seed_window_max_ratio_by_spec; the early-decode guard is off"
fi

# Case 20: --local-iterate with NO golden variable defaults to the shipped short
# capture, anchored at the facade's own repository, so the default holds from
# any working directory. run_facade sets the variable; the empty value here wins
# and the facade treats empty as unset.
#
# The track's own captures are recorded on its box once the target is
# published, so this repository may not carry them yet. Cases 20 and 21 run
# against a throwaway tree that carries placeholder captures at the shipped
# path; the default under test is the path, not the capture's content.
CAPTURE_TRACK="$(jq -r '.trackId' "${REPO_ROOT}/benchmark.json")"
CAPTURE_ROOT="${WORK}/capturedefault"
mkdir -p "${CAPTURE_ROOT}/tools" "${CAPTURE_ROOT}/fixtures" "${CAPTURE_ROOT}/correctness_prompts/${CAPTURE_TRACK}"
cp "${FACADE}" "${CAPTURE_ROOT}/tools/benchmark.sh"
cp "${REPO_ROOT}/benchmark.json" "${REPO_ROOT}/mtp-head.manifest.json" "${CAPTURE_ROOT}/"
cp "${REPO_ROOT}"/fixtures/*.json "${CAPTURE_ROOT}/fixtures/"
for capture in public-local-iterate public-local-submit; do
  printf '{"placeholder_capture":"%s"}\n' "${capture}" \
    > "${CAPTURE_ROOT}/correctness_prompts/${CAPTURE_TRACK}/${capture}.golden.json"
done
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 MLXFAST_CORRECTNESS_GOLDEN_PATH=)
run_facade case20 "${CAPTURE_ROOT}/tools/benchmark.sh" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 20: --local-iterate refused with no golden variable; output: $(cat "${WORK}/case20.out")"
elif [[ ! -f "${WORK}/case20.argv" ]]; then
  fail "case 20: benchd was never spawned"
elif ! argv_pair "${WORK}/case20.argv" --golden "${CAPTURE_ROOT}/correctness_prompts/${CAPTURE_TRACK}/public-local-iterate.golden.json"; then
  fail "case 20: argv does not carry the shipped local-iterate capture: $(tr '\n' ' ' < "${WORK}/case20.argv")"
fi

# Case 21: --local-submit defaults to the shipped long capture.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 MLXFAST_CORRECTNESS_GOLDEN_PATH=)
run_facade case21 "${CAPTURE_ROOT}/tools/benchmark.sh" --local-submit
if [[ "${rc}" -ne 0 ]]; then
  fail "case 21: --local-submit refused with no golden variable; output: $(cat "${WORK}/case21.out")"
elif [[ ! -f "${WORK}/case21.argv" ]]; then
  fail "case 21: benchd was never spawned"
elif ! argv_pair "${WORK}/case21.argv" --golden "${CAPTURE_ROOT}/correctness_prompts/${CAPTURE_TRACK}/public-local-submit.golden.json"; then
  fail "case 21: argv does not carry the shipped local-submit capture: $(tr '\n' ' ' < "${WORK}/case21.argv")"
fi

# Case 22: --official has NO default golden. It refuses by name before benchd
# is spawned: an official run never scores against a shipped capture.
EXTRA_ENV=(MLXFAST_BENCHMARK_SKIP_TIMED=1 MLXFAST_BENCHMARK_CHECK_GATES=1 MLXFAST_CORRECTNESS_GOLDEN_PATH=)
run_facade case22 "${FACADE}" --official
if [[ "${rc}" -eq 0 ]]; then
  fail "case 22: --official ran with no golden variable"
elif ! grep -q -- "--official requires an explicit correctness golden" "${WORK}/case22.out"; then
  fail "case 22: the refusal does not name --official; got: $(cat "${WORK}/case22.out")"
fi
if [[ -f "${WORK}/case22.argv" ]]; then
  fail "case 22: benchd was spawned despite the missing official golden"
fi

# Case 23: with NO engine variable, the facade uses the engine ./setup.sh built
# at .build/release/bench-worker, anchored at its own repository. A throwaway
# tree carries the manifest, the fixture, the declaration and a stub at that
# path, so the default is observed where the real build would be.
mkdir -p "${WORK}/enginedefault/tools" "${WORK}/enginedefault/.build/release" "${WORK}/enginedefault/fixtures"
cp "${FACADE}" "${WORK}/enginedefault/tools/benchmark.sh"
cp "${REPO_ROOT}/benchmark.json" "${REPO_ROOT}/mtp-head.manifest.json" "${WORK}/enginedefault/"
cp "${REPO_ROOT}"/fixtures/*.json "${WORK}/enginedefault/fixtures/"
cp "${STUB}" "${WORK}/enginedefault/.build/release/bench-worker"
chmod +x "${WORK}/enginedefault/.build/release/bench-worker"
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 MLXFAST_ENGINE_BIN=)
run_facade case23 "${WORK}/enginedefault/tools/benchmark.sh" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 23: --local-iterate refused with no engine variable and a built bench-worker; output: $(cat "${WORK}/case23.out")"
elif [[ ! -f "${WORK}/case23.argv" ]]; then
  fail "case 23: benchd was never spawned"
elif ! argv_pair "${WORK}/case23.argv" --engine "${WORK}/enginedefault/.build/release/bench-worker"; then
  fail "case 23: argv does not carry the built bench-worker as --engine: $(tr '\n' ' ' < "${WORK}/case23.argv")"
fi

# Case 24: with NO engine variable and NO build, the facade refuses by name.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 MLXFAST_ENGINE_BIN=)
mkdir -p "${WORK}/nobuild/tools"
cp "${FACADE}" "${WORK}/nobuild/tools/benchmark.sh"
cp "${REPO_ROOT}/benchmark.json" "${WORK}/nobuild/"
run_facade case24 "${WORK}/nobuild/tools/benchmark.sh" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 24: the facade ran with no engine variable and no built bench-worker"
elif ! grep -q "MLXFAST_ENGINE_BIN" "${WORK}/case24.out"; then
  fail "case 24: the refusal does not name MLXFAST_ENGINE_BIN; got: $(cat "${WORK}/case24.out")"
fi
if [[ -f "${WORK}/case24.argv" ]]; then
  fail "case 24: benchd was spawned with no engine"
fi

# Case 25: with no RESIDENT_UP_LOG_DIR, benchd gets one outside both trees.
mkdir -p "${WORK}/case25-tmp"
run_measure case25 "RESIDENT_UP_LOG_DIR=" "RUNNER_TEMP=" "TMPDIR=${WORK}/case25-tmp"
case25_dir="$(cat "${WORK}/case25.argv.resident-log-dir" 2>/dev/null)"
if [[ "${rc}" -ne 0 ]]; then
  fail "case 25: the measure script exited ${rc} with RESIDENT_UP_LOG_DIR unset; output: $(cat "${WORK}/case25.out")"
elif [[ "${case25_dir}" != "${WORK}/case25-tmp/mlxfast-resident-logs/resident-up."* || ! -d "${case25_dir}" ]]; then
  fail "case 25: benchd got RESIDENT_UP_LOG_DIR '${case25_dir}', not a new directory in the fixed log root of the job's temporary directory"
fi

# Case 25b: with RESIDENT_UP_LOG_ROOT set (the ranked job), a preset
# RESIDENT_UP_LOG_DIR cannot skip the root preparation: refused, no benchd.
run_measure case25b "RESIDENT_UP_LOG_ROOT=${WORK}/case25b-root"
if [[ "${rc}" -ne 2 ]] || ! grep -q 'log-dir-preset' "${WORK}/case25b.out" || [[ -f "${WORK}/case25b.argv" ]]; then
  fail "case 25b: a preset RESIDENT_UP_LOG_DIR with RESIDENT_UP_LOG_ROOT set was not refused (rc ${rc}): $(tail -2 "${WORK}/case25b.out")"
fi

# Case 26: a dflash declaration with an oracle for every live golden sends the
# explicit dflash candidate spec, never --mtp-depth (that is mode "mtp" on the
# wire, a decoder the engine does not load). The fixture copy gives every live
# golden a dflash3 oracle; the serial tape's pin stands in, because the bytes
# are never read here.
CASE26_MUTATION='.live_golden_speculative.dflash3 = ([.live_goldens[] as $n | {key: $n, value: (first(.timed_prompt_pool[] | select(.r2_path | endswith("/" + $n + ".golden.json"))))}] | from_entries)'
run_measure_with_contract case26 "${CASE26_MUTATION}" \
  "SPEC_DECLARATION_MANIFEST=${SPEC_MANIFEST}"
if [[ "${rc}" -ne 0 ]]; then
  fail "case 26: the measure script exited ${rc} for a dflash3 declaration with every oracle pinned; output: $(cat "${WORK}/case26.out")"
elif [[ "$(argv_value "${WORK}/case26.argv" --candidate-spec)" != '{"mode":"dflash","dflash":{"depth":3}}' ]]; then
  fail "case 26: --candidate-spec is '$(argv_value "${WORK}/case26.argv" --candidate-spec)', not the dflash depth-3 spec"
elif grep -qx -- '--mtp-depth' "${WORK}/case26.argv"; then
  fail "case 26: benchd got --mtp-depth for a dflash declaration"
fi

# Case 27: a benchd with no --candidate-spec is refused for a dflash
# declaration, rather than running the leg serial against it.
OLD_BENCHD_STUB="${WORK}/benchd-no-candidate-spec"
cat > "${OLD_BENCHD_STUB}" <<'OLDSTUBEOF'
#!/usr/bin/env bash
for arg in "$@"; do
  if [[ "${arg}" == "--help" ]]; then
    echo "usage: benchd iterate --engine <bin> [--control-golden <path>] [--mtp-depth <N>]"
    exit 0
  fi
done
printf '%s\n' "$@" > "${STUB_CAPTURE_ARGV}"
echo '{}'
OLDSTUBEOF
chmod 755 "${OLD_BENCHD_STUB}"
run_measure_with_contract case27 "${CASE26_MUTATION}" \
  "SPEC_DECLARATION_MANIFEST=${SPEC_MANIFEST}" "BENCHD=${OLD_BENCHD_STUB}"
expect_measure_refusal case27 "this benchd has no --candidate-spec"

if [[ "${failures}" -eq 0 ]]; then
  echo "test-benchmark-facade-env.sh: all 27 cases passed (trackId=${EXPECTED})"
  exit 0
fi
echo "test-benchmark-facade-env.sh: ${failures} case(s) failed" >&2
exit 1
