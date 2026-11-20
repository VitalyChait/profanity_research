#!/usr/bin/env bash
# Sandbox all Python/uv/HF/Triton/Matplotlib/cache paths inside the repository directory
# so MinorSafe never modifies system defaults, ~/.bashrc, or ~/.local/lib/python*.
#
# Usage: source scripts/env_sandbox.sh

_MS_SANDBOX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_PROJECT_ENVIRONMENT="${_MS_SANDBOX_ROOT}/.venv"
export VIRTUAL_ENV="${_MS_SANDBOX_ROOT}/.venv"
export UV_INSTALL_DIR="${_MS_SANDBOX_ROOT}/.uv/bin"
export UV_PYTHON_INSTALL_DIR="${_MS_SANDBOX_ROOT}/.uv/python"
export UV_CACHE_DIR="${_MS_SANDBOX_ROOT}/.cache/uv"
export UV_NO_MODIFY_PATH="1"

export PYTHONNOUSERSITE="1"
export PIP_CONFIG_FILE="/dev/null"
export TOKENIZERS_PARALLELISM="false"

export XDG_CACHE_HOME="${_MS_SANDBOX_ROOT}/.cache"
export HF_HOME="${HF_HOME:-${_MS_SANDBOX_ROOT}/.cache/huggingface}"
export TORCH_HOME="${_MS_SANDBOX_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${_MS_SANDBOX_ROOT}/.cache/triton"
export MPLCONFIGDIR="${_MS_SANDBOX_ROOT}/.cache/matplotlib"
export TMPDIR="${_MS_SANDBOX_ROOT}/.cache/tmp"

export PATH="${_MS_SANDBOX_ROOT}/.venv/bin:${_MS_SANDBOX_ROOT}/.uv/bin:${HOME}/.local/bin:${PATH}"

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}" "${TMPDIR}"
unset _MS_SANDBOX_ROOT
