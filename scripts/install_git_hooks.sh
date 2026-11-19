#!/usr/bin/env bash
set -euo pipefail

# Install a pre-commit hook preventing accidental leakage of credentials into this public repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="${REPO_ROOT}/.git/hooks"

if [ ! -d "${REPO_ROOT}/.git" ]; then
    echo "[install_git_hooks] No .git directory found at ${REPO_ROOT}; skipping hook installation."
    exit 0
fi

mkdir -p "${HOOKS_DIR}"
HOOK_FILE="${HOOKS_DIR}/pre-commit"

cat > "${HOOK_FILE}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Prevent committing remote server credentials, SSH keys, or hardcoded tokens.
staged_files="$(git diff --cached --name-only --diff-filter=ACM)"
if [ -n "${staged_files}" ]; then
    while IFS= read -r f; do
        [ -z "${f}" ] && continue
        if [[ "${f}" =~ (^|/)\.remote\.env($|\.) ]] && [[ "${f}" != *".remote.env.example" ]]; then
            echo "[pre-commit] ERROR: Refusing to commit credential file: ${f}" >&2
            exit 1
        fi
        if [[ "${f}" =~ (^|/)\.secrets/ ]] || [[ "${f}" =~ \.(pem|key)$ ]] || [[ "${f}" =~ (^|/)id_(rsa|ed25519) ]]; then
            echo "[pre-commit] ERROR: Refusing to commit secret/key file: ${f}" >&2
            exit 1
        fi
    done <<< "${staged_files}"
fi

# Scan added lines in staged diff for non-empty REMOTE_PASSWORD or HF_TOKEN values
# (skip example templates, this hook generator, and unit test fixtures).
added_lines="$(git diff --cached -U0 -- . ':(exclude).remote.env.example' ':(exclude)scripts/install_git_hooks.sh' ':(exclude)tests/' | grep -E '^\+[^+]' || true)"
if [ -n "${added_lines}" ]; then
    if echo "${added_lines}" | grep -Eq 'REMOTE_PASSWORD[[:space:]]*=[[:space:]]*["'\'']?[^"'\''[:space:]#]+' ; then
        echo "[pre-commit] ERROR: Detected hardcoded REMOTE_PASSWORD in staged diff." >&2
        exit 1
    fi
    if echo "${added_lines}" | grep -Eq 'HF_TOKEN[[:space:]]*=[[:space:]]*["'\'']?hf_[A-Za-z0-9]+' ; then
        echo "[pre-commit] ERROR: Detected hardcoded HF_TOKEN in staged diff." >&2
        exit 1
    fi
fi
EOF

chmod +x "${HOOK_FILE}"
echo "[install_git_hooks] Installed pre-commit secret guard at ${HOOK_FILE}"
