#!/usr/bin/env bash
set -euo pipefail

# Save a timestamped snapshot of OS, driver/CUDA, git, toolchain, and installed .venv package versions
# BEFORE running setup, stored under results/logs/ (with a versions_snapshot_latest.txt symlink).
#
# Usage: bash scripts/snapshot_versions.sh [log_dir] [label]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${1:-${REPO_ROOT}/results/logs}"
LABEL="${2:-pre_setup}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[snapshot] DRY_RUN: save version snapshot (${LABEL}) to ${LOG_DIR}"
    exit 0
fi

mkdir -p "${LOG_DIR}"
SNAP_FILE="${LOG_DIR}/versions_snapshot_${LABEL}_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "=== MinorSafe Pre-Setup Version Snapshot ==="
    echo "timestamp: $(date -Iseconds)"
    echo "label: ${LABEL}"
    echo "host: $(hostname)"
    echo "user: $(whoami)"
    echo "uname: $(uname -a)"
    if [[ -f /etc/os-release ]]; then
        echo "--- /etc/os-release ---"
        cat /etc/os-release || true
    fi

    echo "--- Git ---"
    if command -v git >/dev/null 2>&1 && [[ -d "${REPO_ROOT}/.git" ]]; then
        echo "git_head: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "git_status:"
        git -C "${REPO_ROOT}" status --short 2>/dev/null || true
    else
        echo "git: no .git directory (synced workspace)"
    fi

    echo "--- System Toolchain ---"
    echo "python3: $(python3 --version 2>&1 || echo 'not found')"
    if command -v uv >/dev/null 2>&1; then
        echo "uv: $(uv --version 2>&1 || echo 'unknown')"
    else
        echo "uv: not installed"
    fi
    if command -v ssh >/dev/null 2>&1; then
        echo "ssh: $(ssh -V 2>&1 || echo 'unknown')"
    else
        echo "ssh: not installed"
    fi
    if command -v rsync >/dev/null 2>&1; then
        echo "rsync: $(rsync --version 2>&1 | head -n 1 || echo 'unknown')"
    else
        echo "rsync: not installed"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia_smi:"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>&1 || true
    else
        echo "nvidia_smi: not available"
    fi
    if command -v nvcc >/dev/null 2>&1; then
        echo "nvcc:"
        nvcc --version 2>&1 || true
    else
        echo "nvcc: not installed"
    fi

    echo "--- Existing .venv Packages ---"
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        echo "venv_python: $("${REPO_ROOT}/.venv/bin/python" --version 2>&1 || echo 'unknown')"
        "${REPO_ROOT}/.venv/bin/python" -c 'import importlib.metadata as m; print("\n".join(sorted(f"{d.name}=={d.version}" for d in m.distributions())))' 2>&1 || true
    else
        echo "venv: not present prior to setup"
    fi
} > "${SNAP_FILE}"

ln -sfn "${SNAP_FILE}" "${LOG_DIR}/versions_snapshot_latest.txt"
echo "[snapshot] Saved pre-setup version snapshot to ${SNAP_FILE}"
