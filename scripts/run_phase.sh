#!/usr/bin/env bash
set -euo pipefail

# Execute one MinorSafe pipeline phase, logging to results/logs/<phase>_<timestamp>.log.
# Usage: bash scripts/run_phase.sh <phase> [profile]
#   phases: smoke data rampup guards main recomposition subculture quant multiturn analyze
#   (aliases: main_grid ramp_up subculture_audit quant_grid)
#   profile: hardware profile name in configs/hardware/ or "auto" (default)
# Set DRY_RUN=1 to print the command without executing it.
PHASE="${1:-smoke}"
PROFILE="${2:-${PROFILE:-auto}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
export TOKENIZERS_PARALLELISM="false"
CLI=("${PYTHON}" -m minorsafe.cli)

run_experiment() {
    # --final is required (and only meaningful) for experiments whose config targets the test split.
    local exp="$1"
    local split
    split="$(sed -n 's/^split:[[:space:]]*\([a-z]*\).*/\1/p' "configs/experiments/${exp}.yaml")"
    local args=(run --experiment "${exp}" --hardware "${PROFILE}")
    if [[ "${split}" == "test" ]]; then
        args+=(--final)
    fi
    CMD=("${CLI[@]}" "${args[@]}")
}

case "${PHASE}" in
    smoke)                    CMD=("${CLI[@]}" smoke --hardware cpu_debug) ;;
    data)                     CMD=("${CLI[@]}" data build --hardware "${PROFILE}") ;;
    tomography)               CMD=("${CLI[@]}" tomography --hardware "${PROFILE}") ;;
    rampup|ramp_up)           run_experiment ramp_up ;;
    guards)                   run_experiment guards ;;
    main|main_grid)           run_experiment main_grid ;;
    recomposition)            run_experiment recomposition ;;
    subculture|subculture_audit) run_experiment subculture_audit ;;
    quant|quant_grid)         run_experiment quant_grid ;;
    multiturn)                run_experiment multiturn ;;
    analyze)                  CMD=("${CLI[@]}" analyze) ;;
    *)
        echo "Unknown phase: ${PHASE}" >&2
        echo "Valid phases: smoke data rampup guards main recomposition subculture quant multiturn analyze" >&2
        exit 1
        ;;
esac

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    echo
    exit 0
fi

# Sandbox Python/uv/HF/Triton/cache paths inside REPO_ROOT after DRY_RUN check.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_sandbox.sh"

LOG_DIR="${REPO_ROOT}/results/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${PHASE}_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest.log"

echo "[run_phase] START phase=${PHASE} profile=${PROFILE} host=$(hostname) user=${USER:-unknown} log=${LOG_FILE}" | tee -a "${LOG_FILE}"

# Run initial hardware, exclusivity, credential, and environment checks before GPU experiment phases.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]] && [[ "${PHASE}" != "smoke" ]] && [[ "${PHASE}" != "data" ]] && [[ "${PHASE}" != "analyze" ]]; then
    PREFLIGHT_ARGS=(preflight --hardware "${PROFILE}")
    if [[ "${MINORSAFE_ALLOW_BUSY:-0}" == "1" ]]; then
        PREFLIGHT_ARGS+=(--allow-busy)
    fi
    "${CLI[@]}" "${PREFLIGHT_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

GPU_MON_PID=""
cleanup_gpu_mon() {
    if [[ -n "${GPU_MON_PID}" ]]; then
        kill "${GPU_MON_PID}" 2>/dev/null || true
    fi
}
trap cleanup_gpu_mon EXIT INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_CSV="${LOG_FILE%.log}.gpu.csv"
    echo "timestamp,index,utilization.gpu,memory.used,temperature.gpu" > "${GPU_CSV}"
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader >> "${GPU_CSV}" 2>/dev/null || true
            sleep 15
        done
    ) &
    GPU_MON_PID=$!
fi

T_START="$(date +%s)"
set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e
T_END="$(date +%s)"
ELAPSED="$((T_END - T_START))"

echo "[run_phase] DONE phase=${PHASE} profile=${PROFILE} elapsed=${ELAPSED}s status=${STATUS} log=${LOG_FILE}" | tee -a "${LOG_FILE}"
exit "${STATUS}"
