#!/usr/bin/env bash
set -euo pipefail

# Remote server bootstrap script for MinorSafe (Ubuntu 22.04/24.04 + CUDA 12.x)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[setup_server] Bootstrapping environment in ${REPO_ROOT}..."

# Include ~/.local/bin before checking so previously installed uv is found in non-login SSH shells.
export PATH="${HOME}/.local/bin:${PATH}"

# Capture a snapshot of pre-existing OS/CUDA/driver/venv versions before modifying anything.
bash "${REPO_ROOT}/scripts/snapshot_versions.sh" "${REPO_ROOT}/results/logs" "remote_pre_setup"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup_server] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# Install the locked environment with CUDA torch, bitsandbytes, and vLLM (plan §0.1 / Phase 10).
# --reinstall-package minorsafe guarantees the local wheel includes all subpackages (e.g. minorsafe.data).
uv sync --python 3.11 --reinstall-package minorsafe --extra gpu --extra vllm --extra dev

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="false"
mkdir -p "${HF_HOME}" data/raw data/processed results/runs results/figures results/tables results/logs

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[setup_server] WARNING: HF_TOKEN is not set; gated models (Llama, Gemma) will fail to download." >&2
fi

.venv/bin/python -m minorsafe.cli info --hardware auto || true

echo "[setup_server] Environment ready. Activate with: source .venv/bin/activate"
