#!/usr/bin/env bash
set -euo pipefail

# MinorSafe repository entry-point bootstrap & preflight script.
# Ensures public-repo credential protection, local uv/.venv setup, SSH key installation
# (with OpenSSH askpass fallback when REMOTE_PASSWORD is set without sshpass),
# VPN & server exclusivity validation, and remote GPU environment provisioning.
#
# Usage:
#   ./setup.sh              # Local setup + automatic remote setup when .remote.env is configured
#   ./setup.sh --local      # Local setup & preflight only
#   ./setup.sh --remote     # Local setup + full remote VPN server bootstrap (SSH key, check, sync, GPU env)
#   ./setup.sh --check      # Run local & remote preflight / exclusivity checks without installing packages
#   ./setup.sh --help       # Show usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

MODE="auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)  MODE="local" ;;
        --remote) MODE="remote" ;;
        --check)  MODE="check" ;;
        -h|--help)
            sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "[setup] ERROR: Unknown option: $1 (use --local, --remote, --check, or --help)" >&2
            exit 1
            ;;
    esac
    shift
done

DRY="${DRY_RUN:-0}"
ENV_FILE="${REMOTE_ENV_FILE:-${REPO_ROOT}/.remote.env}"

echo "=== [1/4] Pre-Setup Snapshot, Git Hooks & Credential Safety ==="
DRY_RUN="${DRY}" bash "${REPO_ROOT}/scripts/snapshot_versions.sh" "${REPO_ROOT}/results/logs" "local_pre_setup"
if [[ "${DRY}" == "1" ]]; then
    echo "[setup] DRY_RUN: bash scripts/install_git_hooks.sh"
else
    bash "${REPO_ROOT}/scripts/install_git_hooks.sh"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    if [[ "${DRY}" == "1" ]]; then
        echo "[setup] DRY_RUN: cp .remote.env.example ${ENV_FILE} && chmod 600 ${ENV_FILE}"
    else
        cp "${REPO_ROOT}/.remote.env.example" "${ENV_FILE}"
        chmod 600 "${ENV_FILE}"
        echo "[setup] Created ${ENV_FILE} from .remote.env.example (chmod 600, gitignored)."
    fi
else
    if [[ "${DRY}" != "1" ]]; then
        chmod 600 "${ENV_FILE}" 2>/dev/null || true
    fi
fi

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

echo "=== [2/4] Local Python Environment & Preflight ==="
if [[ "${MODE}" != "check" ]]; then
    if [[ "${DRY}" == "1" ]]; then
        echo "[setup] DRY_RUN: ensure uv installed and .venv synced"
    else
        if ! command -v uv >/dev/null 2>&1; then
            echo "[setup] Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="${HOME}/.local/bin:${PATH}"
        fi
        if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
            if command -v nvidia-smi >/dev/null 2>&1; then
                echo "[setup] Local NVIDIA GPU detected; syncing gpu + vllm + dev extras..."
                uv sync --python 3.11 --extra gpu --extra vllm --extra dev
            else
                echo "[setup] Syncing local cpu + dev extras..."
                uv sync --python 3.11 --extra cpu --extra dev
            fi
        else
            echo "[setup] Local .venv already present."
        fi
    fi
fi

if [[ "${DRY}" == "1" ]]; then
    echo "[setup] DRY_RUN: .venv/bin/python -m minorsafe.cli preflight --hardware cpu_debug --allow-busy"
else
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        "${REPO_ROOT}/.venv/bin/python" -m minorsafe.cli preflight --hardware cpu_debug --allow-busy
    else
        echo "[setup] ERROR: ${REPO_ROOT}/.venv/bin/python missing; run ./setup.sh without --check first." >&2
        exit 1
    fi
fi

if [[ "${MODE}" == "local" ]]; then
    echo "[setup] Local setup complete (--local)."
    exit 0
fi

HOST_VAL="${REMOTE_HOST:-}"
IS_PLACEHOLDER=0
if [[ -z "${HOST_VAL}" ]] || [[ "${HOST_VAL}" == "gpu-server.university.edu" ]]; then
    IS_PLACEHOLDER=1
fi

if [[ "${IS_PLACEHOLDER}" == "1" ]]; then
    if [[ "${MODE}" == "remote" ]]; then
        echo "[setup] ERROR: REMOTE_HOST is not configured in ${ENV_FILE}." >&2
        exit 1
    fi
    echo "[setup] Remote setup skipped: edit ${ENV_FILE} with your GPU server details, connect to the University VPN, and re-run ./setup.sh"
    exit 0
fi

echo "=== [3/4] Remote SSH Key, VPN & Exclusivity Check (${REMOTE_USER:-std-ai}@${HOST_VAL}) ==="
if [[ "${MODE}" != "check" ]]; then
    DRY_RUN="${DRY}" REMOTE_ENV_FILE="${ENV_FILE}" bash "${REPO_ROOT}/scripts/remote.sh" setup-ssh-key
fi
DRY_RUN="${DRY}" REMOTE_ENV_FILE="${ENV_FILE}" bash "${REPO_ROOT}/scripts/remote.sh" check

echo "=== [4/4] Remote GPU Environment & Preflight ==="
if [[ "${MODE}" == "check" ]]; then
    DRY_RUN="${DRY}" REMOTE_ENV_FILE="${ENV_FILE}" bash "${REPO_ROOT}/scripts/remote.sh" preflight
else
    DRY_RUN="${DRY}" REMOTE_ENV_FILE="${ENV_FILE}" bash "${REPO_ROOT}/scripts/remote.sh" setup
fi

echo "[setup] All local and remote checks/setup steps completed."
