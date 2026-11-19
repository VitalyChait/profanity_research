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
    data)                     CMD=("${CLI[@]}" data build) ;;
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

LOG_DIR="${REPO_ROOT}/results/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${PHASE}_$(date +%Y%m%d_%H%M%S).log"
echo "[run_phase] phase=${PHASE} profile=${PROFILE} log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
