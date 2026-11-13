#!/usr/bin/env bash
set -euo pipefail

# Execute a specific MinorSafe pipeline phase on a remote server.
# Usage: bash scripts/run_phase.sh <phase> [profile]
PHASE="${1:-smoke}"
PROFILE="${2:-gpu_80gb}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
export TOKENIZERS_PARALLELISM="false"

case "${PHASE}" in
    smoke)
        "${PYTHON}" -m minorsafe.cli smoke
        ;;
    data)
        "${PYTHON}" -m minorsafe.cli data
        ;;
    main_grid)
        "${PYTHON}" -m minorsafe.cli run --experiment main_grid --profile "${PROFILE}" --final
        ;;
    recomposition)
        "${PYTHON}" -m minorsafe.cli run --experiment recomposition --profile "${PROFILE}" --final
        ;;
    subculture_audit)
        "${PYTHON}" -m minorsafe.cli run --experiment subculture_audit --profile "${PROFILE}" --final
        ;;
    ramp_up)
        "${PYTHON}" -m minorsafe.cli run --experiment ramp_up --profile "${PROFILE}" --final
        ;;
    quant_grid)
        "${PYTHON}" -m minorsafe.cli run --experiment quant_grid --profile "${PROFILE}" --final
        ;;
    multiturn)
        "${PYTHON}" -m minorsafe.cli run --experiment multiturn --profile "${PROFILE}" --final
        ;;
    analyze)
        "${PYTHON}" -m minorsafe.cli analyze
        ;;
    *)
        echo "Unknown phase: ${PHASE}" >&2
        exit 1
        ;;
esac

