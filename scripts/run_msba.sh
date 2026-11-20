#!/usr/bin/env bash
set -euo pipefail

# Execute the statistically powered MSBA study on a single GPU (~3-5 days on GB10)
# without altering any canonical experiment design or dropping any hypothesis (H1-H10).
#
# Usage:
#   bash scripts/run_msba.sh [profile] [--skip-quant] [--with-tomography]
#
# Default profile: gpu_gb10

PROFILE="gpu_gb10"
if [[ $# -ge 1 && "$1" != --* ]]; then
    PROFILE="$1"
    shift
fi

SKIP_QUANT=0
WITH_TOMO=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-quant) SKIP_QUANT=1 ;;
        --with-tomography) WITH_TOMO=1 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash scripts/run_msba.sh [profile] [--skip-quant] [--with-tomography]" >&2
            exit 1
            ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[run_msba] Starting MSBA suite on profile=${PROFILE} (skip_quant=${SKIP_QUANT}, with_tomo=${WITH_TOMO})"

if [[ "${DRY_RUN:-0}" != "1" ]] && [[ ! -f "${REPO_ROOT}/data/processed/examples.parquet" ]]; then
    echo "[run_msba] Building dataset on profile=${PROFILE}..."
    bash scripts/run_phase.sh data "${PROFILE}"
fi

if [[ "${WITH_TOMO}" == "1" ]] || [[ "${DRY_RUN:-0}" != "1" && ! -f "${REPO_ROOT}/results/tomography/summary.json" ]]; then
    bash scripts/run_phase.sh tomography "${PROFILE}"
fi

# Powered stratified sample sizes preserving H1-H10 across S/M/L and all 4 model families
bash scripts/run_phase.sh guards "${PROFILE}" -o max_examples=300
bash scripts/run_phase.sh recomposition "${PROFILE}" -o max_examples=200
bash scripts/run_phase.sh subculture "${PROFILE}" -o max_examples=200
bash scripts/run_phase.sh main "${PROFILE}" -o "seeds=[0]" -o max_examples=300
bash scripts/run_phase.sh rampup "${PROFILE}" -o max_examples=150

if [[ "${SKIP_QUANT}" == "0" ]]; then
    bash scripts/run_phase.sh quant "${PROFILE}" -o max_examples=150
fi

bash scripts/run_phase.sh analyze "${PROFILE}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    {
        echo "phase,profile,elapsed_s,status,log_file"
        grep -h "\[run_phase\] DONE " "${REPO_ROOT}"/results/logs/*.log 2>/dev/null \
            | sed -E 's/.*phase=([^ ]+) profile=([^ ]+) elapsed=([0-9]+)s status=([0-9]+) log=(.+)/\1,\2,\3,\4,\5/' \
            || true
    } > "${REPO_ROOT}/results/phase_timings.csv"
    echo "[run_msba] All phases complete. Timings saved to results/phase_timings.csv; report at results/REPORT.md"
fi
