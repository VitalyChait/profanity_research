#!/usr/bin/env bash
# Activate the MinorSafe repository environment manually (on the GPU server or locally)
# and run hardware, exclusivity, credential, and environment preflight checks.
#
# Usage (recommended so .venv and env vars stay active in your shell):
#   source scripts/activate.sh [profile]
# Or execute directly to validate environment & preflight without keeping the shell active:
#   bash scripts/activate.sh [profile]

if (return 0 2>/dev/null); then
    _MS_SOURCED=1
else
    _MS_SOURCED=0
    set -euo pipefail
fi

_MS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_MS_PROFILE="${1:-}"

if [ -f "${_MS_REPO_ROOT}/.remote.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${_MS_REPO_ROOT}/.remote.env"
    set +a
fi

_MS_PROFILE="${_MS_PROFILE:-${REMOTE_PROFILE:-auto}}"
# Sandbox Python/uv/HF caches inside the repo directory (no user-site or system overrides).
# shellcheck disable=SC1091
source "${_MS_REPO_ROOT}/scripts/env_sandbox.sh"

if [ -x "${_MS_REPO_ROOT}/scripts/install_git_hooks.sh" ]; then
    bash "${_MS_REPO_ROOT}/scripts/install_git_hooks.sh" >/dev/null 2>&1 || true
fi

if [ -f "${_MS_REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${_MS_REPO_ROOT}/.venv/bin/activate"
else
    echo "[activate] WARNING: ${_MS_REPO_ROOT}/.venv not found; run 'bash scripts/setup_server.sh' (or 'make setup')." >&2
    if [ "${_MS_SOURCED}" -eq 1 ]; then
        return 1
    fi
    exit 1
fi

_MS_PREFLIGHT_ARGS=(preflight --hardware "${_MS_PROFILE}")
if [ "${MINORSAFE_ALLOW_BUSY:-0}" = "1" ] || [ "${REQUIRE_EXCLUSIVE:-1}" = "0" ]; then
    _MS_PREFLIGHT_ARGS+=(--allow-busy)
fi

"${_MS_REPO_ROOT}/.venv/bin/python" -m minorsafe.cli "${_MS_PREFLIGHT_ARGS[@]}"
_MS_RC=$?

unset _MS_PROFILE _MS_PREFLIGHT_ARGS
if [ "${_MS_SOURCED}" -eq 1 ]; then
    unset _MS_SOURCED _MS_REPO_ROOT
    return "${_MS_RC}"
fi
exit "${_MS_RC}"
