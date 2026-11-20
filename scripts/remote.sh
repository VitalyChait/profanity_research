#!/usr/bin/env bash
set -euo pipefail

# Automated remote GPU server workflow (designed for a server behind a University VPN).
# Reads gitignored credentials from .remote.env (override path via REMOTE_ENV_FILE).
#
# Usage:
#   bash scripts/remote.sh <subcommand> [args...]
# Subcommands:
#   check | users                Verify VPN reachability, user activity (`w`/`ps`), GPU idle status, and exclusivity
#   setup-ssh-key                Generate (if needed) and install an SSH key onto the remote host
#   sync                         Rsync code/configs to REMOTE_DIR (never uploads .remote.env or secrets)
#   setup                        Sync repo, bootstrap .venv + GPU deps, and run remote preflight
#   preflight [profile]          Run `minorsafe preflight` on the remote host
#   run <phase> [profile] [--detach]
#                                Sync and execute a phase via scripts/run_phase.sh (optionally in tmux)
#   status                       Show remote live status, tmux sessions, GPU utilization, and log tail
#   logs                         Stream remote results/logs/latest.log
#   pull                         Rsync results/ back from the remote server (excluding weights)
#
# Set DRY_RUN=1 to print commands without connecting to the server.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SUBCMD="${1:-}"
if [[ -z "${SUBCMD}" ]]; then
    echo "Usage: bash scripts/remote.sh {check|setup-ssh-key|sync|setup|preflight|run|status|logs|pull} ..." >&2
    exit 1
fi
shift

ENV_FILE="${REMOTE_ENV_FILE:-${REPO_ROOT}/.remote.env}"
if [[ -n "${REMOTE_ENV_FILE:-}" ]] && [[ ! -f "${ENV_FILE}" ]]; then
    echo "[remote] ERROR: REMOTE_ENV_FILE does not exist: ${ENV_FILE}" >&2
    exit 1
fi

# Guard public repository against leaking credentials.
if command -v git >/dev/null 2>&1 && [[ -d "${REPO_ROOT}/.git" ]]; then
    if ! git -C "${REPO_ROOT}" check-ignore -q .remote.env; then
        echo "[remote] ERROR: .remote.env is NOT gitignored! Aborting to protect public repo." >&2
        exit 1
    fi
    if git -C "${REPO_ROOT}" ls-files --error-unmatch .remote.env >/dev/null 2>&1; then
        echo "[remote] ERROR: .remote.env is tracked in git! Run: git rm --cached .remote.env" >&2
        exit 1
    fi
fi

if [[ -f "${ENV_FILE}" ]]; then
    # Enforce owner-only permissions on credential file
    chmod 600 "${ENV_FILE}" 2>/dev/null || true
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
elif [[ "${DRY_RUN:-0}" != "1" ]] && [[ -z "${REMOTE_HOST:-}" ]]; then
    echo "[remote] ERROR: Missing ${ENV_FILE}. Run: cp .remote.env.example .remote.env && chmod 600 .remote.env" >&2
    exit 1
fi

REMOTE_HOST="${REMOTE_HOST:-gpu-server.university.edu}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:-std-ai}"
if [[ -z "${REMOTE_HOME:-}" ]]; then
    if [[ "${REMOTE_USER}" == "root" ]]; then
        REMOTE_HOME="/root"
    else
        REMOTE_HOME="/home/${REMOTE_USER}"
    fi
fi
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_HOME}/profanity_research}"
HF_HOME="${HF_HOME:-${REMOTE_HOME}/.cache/huggingface}"
REMOTE_PROFILE="${REMOTE_PROFILE:-auto}"
VPN_CHECK_TIMEOUT="${VPN_CHECK_TIMEOUT:-5}"
REQUIRE_EXCLUSIVE="${REQUIRE_EXCLUSIVE:-1}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-500}"
MAX_IDLE_GPU_UTIL_PCT="${MAX_IDLE_GPU_UTIL_PCT:-10}"
REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

# Build SSH & rsync transport arrays.
SSH_OPTS=(
    -p "${REMOTE_PORT}"
    -o "ConnectTimeout=${VPN_CHECK_TIMEOUT}"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=4
)
if [[ -n "${REMOTE_SSH_KEY:-}" ]] && [[ -f "${REMOTE_SSH_KEY}" ]]; then
    SSH_OPTS+=(-i "${REMOTE_SSH_KEY}")
fi

RSYNC_SSH="ssh -p ${REMOTE_PORT} -o ConnectTimeout=${VPN_CHECK_TIMEOUT} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
if [[ -n "${REMOTE_SSH_KEY:-}" ]] && [[ -f "${REMOTE_SSH_KEY}" ]]; then
    RSYNC_SSH+=" -i ${REMOTE_SSH_KEY}"
fi

_REMOTE_CLEANUP_FILES=()
cleanup_remote() {
    local f
    for f in "${_REMOTE_CLEANUP_FILES[@]:-}"; do
        [[ -n "${f}" && -f "${f}" ]] && rm -f "${f}" || true
    done
}
trap cleanup_remote EXIT INT TERM

AUTH_PREFIX=()
_HAS_PASSWORD=0
if [[ -n "${REMOTE_PASSWORD:-}" ]]; then
    export SSHPASS="${REMOTE_PASSWORD}"
    _HAS_PASSWORD=1
elif [[ -n "${REMOTE_PASSWORD_FILE:-}" ]]; then
    if [[ -f "${REMOTE_PASSWORD_FILE}" ]]; then
        export SSHPASS
        SSHPASS="$(cat "${REMOTE_PASSWORD_FILE}")"
        _HAS_PASSWORD=1
    elif [[ "${DRY_RUN:-0}" != "1" ]]; then
        echo "[remote] ERROR: REMOTE_PASSWORD_FILE does not exist: ${REMOTE_PASSWORD_FILE}" >&2
        exit 1
    fi
fi

if [[ "${_HAS_PASSWORD}" == "1" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        AUTH_PREFIX=()
    elif command -v sshpass >/dev/null 2>&1; then
        AUTH_PREFIX=(sshpass -e)
    else
        # OpenSSH >= 8.4 built-in non-interactive password prompt (no external sshpass required)
        _ASKPASS_SCRIPT="$(mktemp)"
        _REMOTE_CLEANUP_FILES+=("${_ASKPASS_SCRIPT}")
        cat > "${_ASKPASS_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${SSHPASS:-}"
EOF
        chmod 700 "${_ASKPASS_SCRIPT}"
        export SSH_ASKPASS="${_ASKPASS_SCRIPT}"
        export SSH_ASKPASS_REQUIRE="force"
        export DISPLAY="${DISPLAY:-minorsafe:0}"
        AUTH_PREFIX=()
    fi
fi

check_vpn_reachable() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[remote] DRY_RUN: tcp check ${REMOTE_HOST}:${REMOTE_PORT} (timeout=${VPN_CHECK_TIMEOUT}s)"
        return 0
    fi
    if ! python3 -c "import socket, sys; s = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=float(sys.argv[3])); s.close()" "${REMOTE_HOST}" "${REMOTE_PORT}" "${VPN_CHECK_TIMEOUT}" 2>/dev/null; then
        echo "[remote] ERROR: Cannot reach ${REMOTE_HOST}:${REMOTE_PORT} (timeout ${VPN_CHECK_TIMEOUT}s). Ensure the University VPN is connected and REMOTE_HOST/REMOTE_PORT in .remote.env are correct." >&2
        exit 1
    fi
}

run_ssh() {
    local remote_cmd="$1"
    local cmd=("${AUTH_PREFIX[@]}" ssh "${SSH_OPTS[@]}" "${REMOTE_TARGET}" "${remote_cmd}")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[remote] DRY_RUN: '
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi
    check_vpn_reachable
    "${cmd[@]}"
}

run_rsync_push() {
    local cmd=(
        "${AUTH_PREFIX[@]}" rsync -az --delete
        -e "${RSYNC_SSH}"
        --exclude=/.git/
        --exclude=/.venv/
        --exclude=/.uv/
        --exclude=/.cache/
        --exclude=.remote.env
        --exclude='.remote.env.*'
        --exclude=.secrets/
        --exclude='*.pem'
        --exclude='*.key'
        --exclude='id_ed25519*'
        --exclude='id_rsa*'
        --exclude=.minorsafe.lock
        --exclude=/data/
        --exclude=/results/
        --exclude=/models_cache/
        --exclude=__pycache__/
        --exclude=.pytest_cache/
        --exclude=.ruff_cache/
        "${REPO_ROOT}/" "${REMOTE_TARGET}:${REMOTE_DIR}/"
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[remote] DRY_RUN: '
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi
    check_vpn_reachable
    "${AUTH_PREFIX[@]}" ssh "${SSH_OPTS[@]}" "${REMOTE_TARGET}" "mkdir -p '${REMOTE_DIR}'"
    "${cmd[@]}"
}

pull_remote_logs() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    mkdir -p "${REPO_ROOT}/results/logs"
    "${AUTH_PREFIX[@]}" rsync -az -e "${RSYNC_SSH}" \
        "${REMOTE_TARGET}:${REMOTE_DIR}/results/logs/" \
        "${REPO_ROOT}/results/logs/" >/dev/null 2>&1 || true
}

# Build safe remote env exports (without printing secrets in DRY_RUN)
remote_env_prefix() {
    local parts="export PYTHONNOUSERSITE=1 UV_NO_MODIFY_PATH=1 TOKENIZERS_PARALLELISM=false"
    if [[ -n "${HF_HOME:-}" ]]; then
        parts+=" HF_HOME=$(printf '%q' "${HF_HOME}")"
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            parts+=" HF_TOKEN=***"
        else
            parts+=" HF_TOKEN=$(printf '%q' "${HF_TOKEN}")"
        fi
    fi
    if [[ "${REQUIRE_EXCLUSIVE}" == "0" ]]; then
        parts+=" MINORSAFE_ALLOW_BUSY=1"
    fi
    echo "${parts};"
}

case "${SUBCMD}" in
    check|users|who)
        # Validate VPN connectivity, active users (`w` + `ps`), GPU utilization/processes, and disk space.
        REMOTE_CHECK_SCRIPT="$(cat <<EOF
echo "Host: \$(hostname) | User: \$(whoami) | Date: \$(date -Iseconds)"
echo "--- Disk ---"
df -h "\$( [ -d '${REMOTE_DIR}' ] && echo '${REMOTE_DIR}' || echo / )" | tail -n 1
echo "--- Active sessions (w) ---"
W_OUT="\$(w 2>/dev/null || who -u 2>/dev/null || who 2>/dev/null || true)"
echo "\${W_OUT:-<none>}"
WHO_OUT="\$(who -u 2>/dev/null || who 2>/dev/null || true)"
CANDIDATE_USERS="\$(echo "\${WHO_OUT}" | awk -v me="\$(whoami)" 'NF>0 && \$1!=me {print \$1}' | sort -u)"
ACTIVE_OTHER_USERS=""
IDLE_OTHER_USERS=""
if [[ -n "\${CANDIDATE_USERS}" ]]; then
    echo "--- Other users' processes (top CPU) ---"
    for u in \${CANDIDATE_USERS}; do
        ps -u "\${u}" -o user=,pid=,pcpu=,pmem=,etime=,comm=,args= --sort=-pcpu 2>/dev/null | head -n 8 || true
        HIGH_CPU="\$(ps -u "\${u}" -o pcpu= 2>/dev/null | awk '\$1+0 >= 5.0 {c++} END {print c+0}')"
        COMP_PROC="\$(ps -u "\${u}" -o pcpu=,comm=,args= 2>/dev/null | awk 'tolower(\$0) ~ /(python|pytest|vllm|torchrun|jupyter|ollama|nvcc|minorsafe)/ && \$1+0 >= 1.0 {c++} END {print c+0}')"
        ACTIVE_PTS="\$(echo "\${WHO_OUT}" | awk -v u="\${u}" '\$1==u && \$2 ~ /^pts\\// && (\$5 == "." || \$5 ~ /^00:/) {c++} END {print c+0}')"
        if [[ "\${HIGH_CPU}" -gt 0 ]] || [[ "\${COMP_PROC}" -gt 0 ]] || [[ "\${ACTIVE_PTS}" -gt 0 ]]; then
            ACTIVE_OTHER_USERS="\${ACTIVE_OTHER_USERS:+\${ACTIVE_OTHER_USERS} }\${u}"
        else
            IDLE_OTHER_USERS="\${IDLE_OTHER_USERS:+\${IDLE_OTHER_USERS} }\${u}"
        fi
    done
fi
echo "--- GPUs ---"
BUSY=0
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO_OUT="\$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || true)"
    echo "\${GPU_INFO_OUT}"
    if [[ "\${GPU_INFO_OUT}" == *"N/A"* ]]; then
        echo "--- Unified Memory (free -h) ---"
        free -h | head -n 2
    fi
    APPS="\$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true)"
    echo "Compute apps: \${APPS:-<none>}"
    if [[ -n "\${APPS}" ]] && [[ "\${APPS}" != *"No running"* ]] && [[ "\${APPS}" != *"Not Supported"* ]]; then
        BUSY=1
    fi
    while IFS=, read -r _idx _name mem_used _mem_tot util _temp; do
        mem_used="\$(echo "\${mem_used:-0}" | tr -cd '0-9')"
        util="\$(echo "\${util:-0}" | tr -cd '0-9')"
        mem_used="\${mem_used:-0}"
        util="\${util:-0}"
        if [[ "\${mem_used}" -gt "${MAX_IDLE_GPU_MEM_MB}" ]] || [[ "\${util}" -gt "${MAX_IDLE_GPU_UTIL_PCT}" ]]; then
            BUSY=1
        fi
    done < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
else
    echo "nvidia-smi not found"
fi
if [[ -n "\${IDLE_OTHER_USERS}" ]]; then
    echo "[WARN] Idle login session(s) detected (\${IDLE_OTHER_USERS}) — 0 active CPU/GPU tasks; continuing."
fi
if [[ "${REQUIRE_EXCLUSIVE}" == "1" ]]; then
    if [[ -n "\${ACTIVE_OTHER_USERS}" ]]; then
        echo "ERROR: Active user(s) on server: \${ACTIVE_OTHER_USERS}" >&2
        exit 2
    fi
    if [[ "\${BUSY}" == "1" ]]; then
        echo "ERROR: GPU is currently in use (active compute apps or exceeds idle threshold)." >&2
        exit 2
    fi
fi
echo "Server exclusivity check PASSED."
EOF
)"
        run_ssh "${REMOTE_CHECK_SCRIPT}"
        ;;

    setup-ssh-key)
        KEY_PATH="${REMOTE_SSH_KEY:-${HOME}/.ssh/id_ed25519_minorsafe}"
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            echo "[remote] DRY_RUN: ensure key ${KEY_PATH} and install ${KEY_PATH}.pub into ${REMOTE_TARGET}:~/.ssh/authorized_keys"
            exit 0
        fi
        check_vpn_reachable
        if [[ ! -f "${KEY_PATH}" ]]; then
            mkdir -p "$(dirname "${KEY_PATH}")"
            chmod 700 "$(dirname "${KEY_PATH}")"
            ssh-keygen -t ed25519 -f "${KEY_PATH}" -N "" -C "minorsafe@$(hostname)"
        fi
        PUB_KEY="$(cat "${KEY_PATH}.pub" 2>/dev/null || ssh-keygen -y -f "${KEY_PATH}")"
        KEY_SSH_OPTS=("${SSH_OPTS[@]}" -i "${KEY_PATH}")
        "${AUTH_PREFIX[@]}" ssh "${KEY_SSH_OPTS[@]}" "${REMOTE_TARGET}" \
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && (grep -qxF $(printf '%q' "${PUB_KEY}") ~/.ssh/authorized_keys || printf '%s\n' $(printf '%q' "${PUB_KEY}") >> ~/.ssh/authorized_keys)"
        echo "[remote] Installed SSH public key ${KEY_PATH}.pub on ${REMOTE_TARGET}"
        env -u SSH_ASKPASS -u SSH_ASKPASS_REQUIRE -u SSHPASS ssh -o BatchMode=yes "${KEY_SSH_OPTS[@]}" "${REMOTE_TARGET}" \
            "echo '[remote] Passwordless SSH key auth verified on '\$(hostname)"
        ;;

    sync)
        run_rsync_push
        ;;

    setup)
        run_rsync_push
        ENV_PRE="$(remote_env_prefix)"
        set +e
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && bash scripts/setup_server.sh && .venv/bin/python -m minorsafe.cli preflight --hardware '${REMOTE_PROFILE}'$( [[ "${REQUIRE_EXCLUSIVE}" == "0" ]] && echo ' --allow-busy' || true )"
        RC=$?
        set -e
        if [[ "${RC}" -ne 0 ]]; then
            echo "[remote] Remote command exited with status ${RC}; auto-pulling remote results/logs/ to ${REPO_ROOT}/results/logs/ ..." >&2
            pull_remote_logs
            exit "${RC}"
        fi
        ;;

    preflight)
        PROF="${1:-${REMOTE_PROFILE}}"
        ENV_PRE="$(remote_env_prefix)"
        ALLOW_FLAG=""
        if [[ "${REQUIRE_EXCLUSIVE}" == "0" ]]; then
            ALLOW_FLAG=" --allow-busy"
        fi
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && .venv/bin/python -m minorsafe.cli preflight --hardware '${PROF}'${ALLOW_FLAG}"
        ;;

    run)
        PHASE="${1:-}"
        if [[ -z "${PHASE}" ]]; then
            echo "Usage: bash scripts/remote.sh run <phase> [profile] [--detach]" >&2
            exit 1
        fi
        shift
        RUN_PROFILE="${REMOTE_PROFILE}"
        DETACH_MODE="${DETACH:-0}"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --detach) DETACH_MODE="1" ;;
                *) RUN_PROFILE="$1" ;;
            esac
            shift
        done
        run_rsync_push
        ENV_PRE="$(remote_env_prefix)"
        if [[ "${PHASE}" == "msba" ]]; then
            RUN_CMD="${ENV_PRE} cd '${REMOTE_DIR}' && bash scripts/run_msba.sh '${RUN_PROFILE}'"
        else
            RUN_CMD="${ENV_PRE} cd '${REMOTE_DIR}' && bash scripts/run_phase.sh '${PHASE}' '${RUN_PROFILE}'"
        fi
        if [[ "${DETACH_MODE}" == "1" ]]; then
            SESSION="minorsafe_${PHASE}"
            DETACH_SCRIPT="$(cat <<EOF
mkdir -p '${REMOTE_DIR}/results/logs'
if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t '${SESSION}' 2>/dev/null || true
    tmux new-session -d -s '${SESSION}' bash -lc $(printf '%q' "${RUN_CMD}")
    echo "[remote] Started detached tmux session '${SESSION}'. Monitor with: bash scripts/remote.sh logs"
else
    nohup bash -lc $(printf '%q' "${RUN_CMD}") > '${REMOTE_DIR}/results/logs/${PHASE}_detached.log' 2>&1 < /dev/null &
    echo "[remote] Started detached nohup PID \$!. Monitor with: bash scripts/remote.sh logs"
fi
EOF
)"
            run_ssh "${DETACH_SCRIPT}"
        else
            set +e
            run_ssh "${RUN_CMD}"
            RC=$?
            set -e
            if [[ "${RC}" -ne 0 ]]; then
                echo "[remote] Remote command exited with status ${RC}; auto-pulling remote results/logs/ to ${REPO_ROOT}/results/logs/ ..." >&2
                pull_remote_logs
                exit "${RC}"
            fi
        fi
        ;;

    status)
        run_ssh "cd '${REMOTE_DIR}' && ( .venv/bin/python -m minorsafe.cli status || true ) && echo '--- tmux ---' && ( tmux ls 2>/dev/null || echo 'no tmux sessions' ) && echo '--- GPUs ---' && ( nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi' ) && echo '--- latest log ---' && ( [ -f results/logs/latest.log ] && tail -n 15 results/logs/latest.log || echo 'no logs yet' )"
        ;;

    logs)
        run_ssh "tail -n 50 -f '${REMOTE_DIR}/results/logs/latest.log'"
        ;;

    pull)
        mkdir -p "${REPO_ROOT}/results"
        cmd=(
            "${AUTH_PREFIX[@]}" rsync -avz
            -e "${RSYNC_SSH}"
            --exclude='*.safetensors'
            --exclude='*.pt'
            "${REMOTE_TARGET}:${REMOTE_DIR}/results/" "${REPO_ROOT}/results/"
        )
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            printf '[remote] DRY_RUN: '
            printf '%q ' "${cmd[@]}"
            echo
            exit 0
        fi
        check_vpn_reachable
        "${cmd[@]}"
        ;;

    gpu-test)
        run_rsync_push
        ENV_PRE="$(remote_env_prefix)"
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && .venv/bin/python -m pytest -v -p no:cacheprovider -m gpu tests/gpu/"
        ;;

    *)
        echo "Unknown subcommand: ${SUBCMD}" >&2
        echo "Valid subcommands: check users setup-ssh-key sync setup preflight gpu-test run status logs pull" >&2
        exit 1
        ;;
esac
