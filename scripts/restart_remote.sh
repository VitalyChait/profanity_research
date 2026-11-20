#!/usr/bin/env bash
set -euo pipefail

# Stop any active MinorSafe remote jobs, clear stale pre-fix data/run artifacts,
# sync the updated local codebase, and relaunch the full unattended GPU benchmark.
#
# Usage:
#   bash scripts/restart_remote.sh [profile] [--keep-data] [--skip-setup] [--skip-tests] [--skip-download] [--skip-pilot] [--skip-quant]
#
# Defaults:
#   profile = gpu_lab1
#
# Set DRY_RUN=1 to print the planned commands without connecting to the server.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="gpu_lab1"
if [[ $# -ge 1 && "$1" != --* ]]; then
    PROFILE="$1"
    shift
fi

RESET_ARGS=()
AUTO_ARGS=(--with-tomography)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-data)
            RESET_ARGS+=(--keep-data)
            ;;
        --skip-setup|--skip-tests|--skip-download|--skip-pilot|--skip-quant)
            AUTO_ARGS+=("$1")
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash scripts/restart_remote.sh [profile] [--keep-data] [--skip-setup] [--skip-tests] [--skip-download] [--skip-pilot] [--skip-quant]" >&2
            exit 1
            ;;
    esac
    shift
done

echo "[restart_remote] 1/2 Stopping running remote MinorSafe jobs and clearing stale artifacts..."
bash "${REPO_ROOT}/scripts/remote.sh" reset ${RESET_ARGS[@]+"${RESET_ARGS[@]}"}

echo "[restart_remote] 2/2 Syncing updated code and launching autopilot (profile=${PROFILE})..."
bash "${REPO_ROOT}/scripts/remote.sh" autopilot "${PROFILE}" "${AUTO_ARGS[@]}"
