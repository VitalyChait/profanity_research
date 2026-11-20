#!/usr/bin/env bash
set -euo pipefail

# Remote server bootstrap script for MinorSafe (Ubuntu 22.04/24.04 + CUDA 12.x)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[setup_server] Bootstrapping environment in ${REPO_ROOT}..."

# Keep uv, Python, caches, and .venv strictly inside REPO_ROOT without touching system or ~/.bashrc.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_sandbox.sh"

# Capture a snapshot of pre-existing OS/CUDA/driver/venv versions before modifying anything.
bash "${REPO_ROOT}/scripts/snapshot_versions.sh" "${REPO_ROOT}/results/logs" "remote_pre_setup"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup_server] Installing uv into ${UV_INSTALL_DIR}..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install the locked environment with CUDA torch, bitsandbytes, and vLLM (plan §0.1 / Phase 10).
# Drivers older than CUDA 13.0 (e.g., lab1 driver 570.211 / CUDA 12.8) reject PyPI torch>=2.11 (cu13),
# so fall back cleanly to the native CUDA 12.8 torch==2.10.0 + vllm==0.19.0 stack on those hosts.
DRIVER_CUDA_VER=""
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n 1 || true)"
fi
DRIVER_CUDA_MAJOR="${DRIVER_CUDA_VER%%.*}"

if [ -n "${DRIVER_CUDA_MAJOR}" ] && [ "${DRIVER_CUDA_MAJOR}" -lt 13 ] 2>/dev/null; then
    echo "[setup_server] Driver reports CUDA ${DRIVER_CUDA_VER} (< 13.0); using CUDA 12.8 stack (torch==2.10.0, vllm==0.19.0)..."
    if ! [ -x ".venv/bin/python" ] || ! .venv/bin/python -c "import torch, vllm; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        uv venv --python 3.11 --clear .venv
    fi
    TORCH_IDX_ARGS=(--extra-index-url https://download.pytorch.org/whl/cu128)
    if uv pip install --help 2>/dev/null | grep -q -- '--torch-backend'; then
        TORCH_IDX_ARGS=(--torch-backend=cu128)
    fi
    uv pip install --python .venv/bin/python "${TORCH_IDX_ARGS[@]}" --reinstall-package minorsafe -e ".[gpu,vllm,dev]" "vllm==0.19.0" "torch==2.10.0"
else
    # --reinstall-package minorsafe guarantees the local wheel includes all subpackages (e.g. minorsafe.data).
    uv sync --python 3.11 --reinstall-package minorsafe --extra gpu --extra vllm --extra dev
fi
uv cache clean || true

mkdir -p "${HF_HOME}" data/raw data/processed results/runs results/figures results/tables results/logs

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[setup_server] WARNING: HF_TOKEN is not set; gated models (Llama, Gemma) will fail to download." >&2
fi

.venv/bin/python -m minorsafe.cli info --hardware auto || true

echo "[setup_server] Environment ready. Activate with: source .venv/bin/activate"
