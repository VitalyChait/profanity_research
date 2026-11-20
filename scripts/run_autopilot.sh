#!/usr/bin/env bash
set -euo pipefail

# Unattended end-to-end research pipeline runner (safe to run detached on a GPU server
# while your laptop is asleep or disconnected).
#
# Sequence:
#   1. Full automated test verification (`scripts/run_all_tests.sh`)
#   2. HF gated-repo check & model cache pre-download (`scripts/download_models.py`)
#   3. Synthetic & curated dataset build (`scripts/run_phase.sh data`) if not yet built
#   4. Isolated assumption pilot (`scripts/run_phase.sh pilot` -> archived out of `results/runs/`)
#   5. Full MSBA benchmark suite + final analysis (`scripts/run_msba.sh`)
#
# Usage:
#   bash scripts/run_autopilot.sh [profile] [--skip-tests] [--skip-download] [--skip-pilot] [--skip-quant] [--with-tomography]
#
# Set DRY_RUN=1 to print the planned sequence without executing.

PROFILE="${REMOTE_PROFILE:-gpu_lab1}"
if [[ $# -ge 1 && "$1" != --* ]]; then
    PROFILE="$1"
    shift
fi

SKIP_TESTS=0
SKIP_DOWNLOAD=0
SKIP_PILOT=0
SKIP_QUANT=0
WITH_TOMO=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-tests) SKIP_TESTS=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --skip-pilot) SKIP_PILOT=1 ;;
        --skip-quant) SKIP_QUANT=1 ;;
        --with-tomography) WITH_TOMO=1 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash scripts/run_autopilot.sh [profile] [--skip-tests] [--skip-download] [--skip-pilot] [--skip-quant] [--with-tomography]" >&2
            exit 1
            ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

MSBA_ARGS=()
if [[ "${SKIP_QUANT}" == "1" ]]; then
    MSBA_ARGS+=(--skip-quant)
fi
if [[ "${WITH_TOMO}" == "1" ]]; then
    MSBA_ARGS+=(--with-tomography)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[autopilot] DRY_RUN sequence for profile=${PROFILE}:"
    if [[ "${SKIP_TESTS}" == "0" ]]; then
        echo "  1) bash scripts/run_all_tests.sh ${PROFILE}"
    fi
    if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
        echo "  2) ${PYTHON} scripts/download_models.py --profile ${PROFILE} --check-gates && ${PYTHON} scripts/download_models.py --profile ${PROFILE}"
    else
        echo "  2) ${PYTHON} scripts/download_models.py --profile ${PROFILE} --check-gates"
    fi
    echo "  3) bash scripts/run_phase.sh data ${PROFILE} (if data/processed/examples.parquet missing)"
    if [[ "${SKIP_PILOT}" == "0" ]]; then
        echo "  4) bash scripts/run_phase.sh pilot ${PROFILE} -> archive to results/pilot_archive/"
    fi
    echo "  5) bash scripts/run_msba.sh ${PROFILE} ${MSBA_ARGS[*]:-}"
    exit 0
fi

if [[ -f "${REPO_ROOT}/scripts/env_sandbox.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/env_sandbox.sh"
fi

# Pin GPU devices from hardware profile before Python imports torch (e.g. GPU 0 on lab1).
PROF_YAML="${REPO_ROOT}/configs/hardware/${PROFILE}.yaml"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ -f "${PROF_YAML}" ]]; then
    PINNED="$(sed -n 's/^cuda_visible_devices:[[:space:]]*["'\'']*\([^"'\''# ]*\).*/\1/p' "${PROF_YAML}" | head -n 1)"
    if [[ -n "${PINNED}" ]]; then
        export CUDA_VISIBLE_DEVICES="${PINNED}"
    fi
fi

LOG_DIR="${REPO_ROOT}/results/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/autopilot_${TS}.log"
PID_FILE="${LOG_DIR}/autopilot.pid"
STATUS_FILE="${LOG_DIR}/autopilot.status"

if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${OLD_PID}" ]] && [[ "${OLD_PID}" != "$$" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[autopilot] ERROR: Another autopilot instance is already running (PID=${OLD_PID})." >&2
        exit 1
    fi
fi

echo "$$" > "${PID_FILE}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/autopilot_latest.log"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest.log"

CURRENT_STAGE="init"
T_ALL_START="$(date +%s)"

write_status() {
    printf '%s\n' "$1" > "${STATUS_FILE}"
}

on_exit() {
    local rc=$?
    local t_now
    t_now="$(date +%s)"
    local elapsed="$((t_now - T_ALL_START))"
    if [[ "${rc}" -eq 0 ]]; then
        write_status "DONE profile=${PROFILE} exit=0 elapsed=${elapsed}s finished_at=$(date -Iseconds)"
    else
        write_status "FAILED profile=${PROFILE} stage=${CURRENT_STAGE} exit=${rc} elapsed=${elapsed}s failed_at=$(date -Iseconds)"
    fi
    if [[ -f "${PID_FILE}" ]] && [[ "$(cat "${PID_FILE}" 2>/dev/null || true)" == "$$" ]]; then
        rm -f "${PID_FILE}"
    fi
}
trap on_exit EXIT INT TERM

log_msg() {
    printf '[%s] [autopilot] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

run_step() {
    local stage_name="$1"
    shift
    CURRENT_STAGE="${stage_name}"
    write_status "RUNNING profile=${PROFILE} stage=${stage_name} pid=$$ started_at=$(date -Iseconds)"
    log_msg ">>> STAGE ${stage_name}: $*"
    local t0
    t0="$(date +%s)"
    set +e
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local rc="${PIPESTATUS[0]}"
    set -e
    local t1
    t1="$(date +%s)"
    log_msg "<<< STAGE ${stage_name} finished (exit=${rc}, elapsed=$((t1 - t0))s)"
    return "${rc}"
}

log_msg "Starting unattended autopilot: profile=${PROFILE} host=$(hostname) pid=$$ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ "${SKIP_TESTS}" == "0" ]]; then
    run_step "tests" bash scripts/run_all_tests.sh "${PROFILE}"
else
    log_msg "Skipping tests (--skip-tests)."
fi

run_step "hf_gates" "${PYTHON}" scripts/download_models.py --profile "${PROFILE}" --check-gates

if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
    run_step "download_models" "${PYTHON}" scripts/download_models.py --profile "${PROFILE}"
else
    log_msg "Skipping model pre-download (--skip-download)."
fi

if [[ ! -f "${REPO_ROOT}/data/processed/examples.parquet" ]] || [[ ! -f "${REPO_ROOT}/data/processed/pairs.parquet" ]]; then
    run_step "data" bash scripts/run_phase.sh data "${PROFILE}"
else
    log_msg "Dataset already built at data/processed/; skipping data phase."
fi

# Isolate any existing or new pilot shards from final MSBA aggregation (`results/REPORT.md`).
PILOT_RUN_DIR="${REPO_ROOT}/results/runs/pilot"
PILOT_ARCH_DIR="${REPO_ROOT}/results/pilot_archive"
if [[ -d "${PILOT_RUN_DIR}" ]]; then
    mkdir -p "${PILOT_ARCH_DIR}"
    mv "${PILOT_RUN_DIR}" "${PILOT_ARCH_DIR}/pilot_prev_${TS}"
    log_msg "Moved pre-existing pilot shards out of results/runs/ to ${PILOT_ARCH_DIR}/pilot_prev_${TS}."
fi

if [[ "${SKIP_PILOT}" == "0" ]] && [[ ! -f "${PILOT_ARCH_DIR}/done.marker" ]]; then
    run_step "pilot" bash scripts/run_phase.sh pilot "${PROFILE}"
    if [[ -d "${PILOT_RUN_DIR}" ]]; then
        mkdir -p "${PILOT_ARCH_DIR}"
        mv "${PILOT_RUN_DIR}" "${PILOT_ARCH_DIR}/pilot_${TS}"
        date -Iseconds > "${PILOT_ARCH_DIR}/done.marker"
        log_msg "Archived pilot shards to ${PILOT_ARCH_DIR}/pilot_${TS} before full MSBA suite."
    fi
else
    log_msg "Skipping pilot stage (SKIP_PILOT=${SKIP_PILOT})."
fi

run_step "msba" bash scripts/run_msba.sh "${PROFILE}" ${MSBA_ARGS[@]+"${MSBA_ARGS[@]}"}

ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest.log"
T_ALL_END="$(date +%s)"
log_msg "AUTOPILOT COMPLETE in $((T_ALL_END - T_ALL_START))s. Report: results/REPORT.md | Timings: results/phase_timings.csv"
