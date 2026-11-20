#!/usr/bin/env bash
set -euo pipefail

# Automatically run all required MinorSafe checks and tests in sequence:
#   1. Shell syntax & secret/ruff hygiene checks
#   2. Hardware & environment preflight (respecting CUDA_VISIBLE_DEVICES from profile)
#   3. Offline unit, integration, and CPU smoke test suite (`pytest -m "not gpu"`)
#   4. Standalone CLI `smoke` pipeline
#   5. Live CUDA GPU verification suite (`pytest -m gpu`) when CUDA is available
#   6. Optional `--with-pilot` end-to-end assumption pilot + analysis
#
# Usage:
#   bash scripts/run_all_tests.sh [profile] [--cpu-only] [--with-pilot]

PROFILE="auto"
if [[ $# -ge 1 && "$1" != --* ]]; then
    PROFILE="$1"
    shift
fi

CPU_ONLY=0
WITH_PILOT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only) CPU_ONLY=1 ;;
        --with-pilot) WITH_PILOT=1 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash scripts/run_all_tests.sh [profile] [--cpu-only] [--with-pilot]" >&2
            exit 1
            ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/scripts/env_sandbox.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/env_sandbox.sh"
fi

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

# Pin GPU devices from hardware profile before Python imports torch (e.g. GPU 0 on lab1).
PROF_YAML="${REPO_ROOT}/configs/hardware/${PROFILE}.yaml"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ -f "${PROF_YAML}" ]]; then
    PINNED="$(sed -n 's/^cuda_visible_devices:[[:space:]]*["'\'']*\([^"'\''# ]*\).*/\1/p' "${PROF_YAML}" | head -n 1)"
    if [[ -n "${PINNED}" ]]; then
        export CUDA_VISIBLE_DEVICES="${PINNED}"
        echo "[tests] Pinned CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (from ${PROFILE}.yaml)"
    fi
fi

echo "=== [1/5] Shell Syntax & Lint Checks ==="
bash -n setup.sh scripts/*.sh
if command -v ruff >/dev/null 2>&1 || [[ -x "${REPO_ROOT}/.venv/bin/ruff" ]]; then
    RUFF_BIN="${REPO_ROOT}/.venv/bin/ruff"
    [[ -x "${RUFF_BIN}" ]] || RUFF_BIN="ruff"
    "${RUFF_BIN}" check --no-cache src tests scripts
    "${RUFF_BIN}" format --no-cache --check src tests scripts
fi

echo "=== [2/5] Environment Preflight (profile=${PROFILE}) ==="
if [[ "${CPU_ONLY}" == "1" ]]; then
    "${PYTHON}" -m minorsafe.cli preflight --hardware cpu_debug
else
    "${PYTHON}" -m minorsafe.cli preflight --hardware "${PROFILE}"
fi

echo "=== [3/5] Unit, Integration & Smoke Tests (offline CPU suite) ==="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${PYTHON}" -m pytest -q -p no:cacheprovider -m "not gpu"

echo "=== [4/5] Standalone CLI Smoke Pipeline ==="
TMP_SMOKE="$(mktemp -d)"
trap 'rm -rf "${TMP_SMOKE}"' EXIT
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${PYTHON}" -m minorsafe.cli smoke --hardware cpu_debug --work-dir "${TMP_SMOKE}" --max-examples 4
rm -rf "${TMP_SMOKE}"

echo "=== [5/5] CUDA GPU Smoke Tests ==="
HAS_CUDA="$("${PYTHON}" -c "import torch; print('1' if torch.cuda.is_available() else '0')")"
if [[ "${CPU_ONLY}" == "1" ]] || [[ "${HAS_CUDA}" != "1" ]]; then
    echo "[tests] Skipping CUDA GPU tests (CPU_ONLY=${CPU_ONLY}, cuda_available=${HAS_CUDA})."
else
    "${PYTHON}" -m pytest -v -p no:cacheprovider -m gpu tests/gpu/
fi

if [[ "${WITH_PILOT}" == "1" ]]; then
    PILOT_PROF="${PROFILE}"
    if [[ "${CPU_ONLY}" == "1" ]] || [[ "${HAS_CUDA}" != "1" ]]; then
        PILOT_PROF="cpu_debug"
    fi
    echo "=== [Bonus] Running End-to-End Pilot (${PILOT_PROF}) + Analyze ==="
    bash scripts/run_phase.sh pilot "${PILOT_PROF}"
    bash scripts/run_phase.sh analyze "${PILOT_PROF}"
fi

echo "[tests] ALL REQUIRED CHECKS PASSED."
