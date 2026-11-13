#!/usr/bin/env bash
set -euo pipefail

# Remote server bootstrap script for MinorSafe (Ubuntu 22.04/24.04 + CUDA 12.x)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[setup_server] Bootstrapping environment in ${REPO_ROOT}..."

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup_server] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -d ".venv" ]; then
    uv venv --python 3.11 .venv
fi

uv pip install --python .venv/bin/python -e ".[dev]"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="false"
mkdir -p "${HF_HOME}" data/raw data/processed results/runs results/figures results/tables

echo "[setup_server] Environment ready. Activate with: source .venv/bin/activate"

