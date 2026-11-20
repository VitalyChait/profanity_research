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
#   test-all [profile] [flags]   Sync and execute full local/GPU verification suite (`run_all_tests.sh`)
#   autopilot [profile] [flags]  Sync, validate setup/preflight/HF gates, then launch full unattended sequence in background
#   run <phase> [profile] [--detach]
#                                Sync and execute a phase via scripts/run_phase.sh (optionally in tmux/nohup)
#   stop                         Stop running remote MinorSafe tmux sessions/processes and clear locks
#   reset [--keep-data]          Stop remote jobs and wipe stale runs/tomography/pilot (and data/processed)
#   status                       Show remote live status, autopilot state, GPU utilization, and log tail
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
    local prof="${1:-${REMOTE_PROFILE:-auto}}"
    local parts="export PYTHONNOUSERSITE=1 UV_NO_MODIFY_PATH=1 TOKENIZERS_PARALLELISM=false"
    local vis_gpus="${CUDA_VISIBLE_DEVICES:-${REMOTE_CUDA_VISIBLE_DEVICES:-}}"
    if [[ -z "${vis_gpus}" ]] && [[ -f "${REPO_ROOT}/configs/hardware/${prof}.yaml" ]]; then
        vis_gpus="$(sed -n 's/^cuda_visible_devices:[[:space:]]*["'\'']*\([^"'\''# ]*\).*/\1/p' "${REPO_ROOT}/configs/hardware/${prof}.yaml" | head -n 1)"
    fi
    if [[ -n "${vis_gpus}" ]]; then
        parts+=" CUDA_VISIBLE_DEVICES=$(printf '%q' "${vis_gpus}")"
    fi
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

_remote_stop_cmd() {
    cat <<EOF
set +e
if command -v tmux >/dev/null 2>&1; then
    for sess in \$(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^minorsafe_' || true); do
        tmux kill-session -t "\${sess}" 2>/dev/null || true
        echo "[remote] Killed tmux session \${sess}"
    done
fi
PATTERN='[r]un_autopilot\.sh|[m]inorsafe\.cli|[r]un_msba\.sh|[r]un_all_tests\.sh|[r]un_phase\.sh|[V]LLM::EngineCore|[v]llm\.worker'
SELF_PID="\$\$"
PIDS="\$(pgrep -u "\$(whoami)" -f "\${PATTERN}" 2>/dev/null | grep -vx "\${SELF_PID}" || true)"
if [ -n "\${PIDS}" ]; then
    echo "[remote] Stopping PIDs: \${PIDS}"
    kill -TERM \${PIDS} 2>/dev/null || true
    sleep 3
    REM="\$(pgrep -u "\$(whoami)" -f "\${PATTERN}" 2>/dev/null | grep -vx "\${SELF_PID}" || true)"
    if [ -n "\${REM}" ]; then
        kill -KILL \${REM} 2>/dev/null || true
        sleep 1
    fi
fi
rm -f '${REMOTE_DIR}/.minorsafe.lock' '${REMOTE_DIR}/results/logs/autopilot.pid'
echo "[remote] Remote MinorSafe processes stopped and locks cleared."
EOF
}

case "${SUBCMD}" in
    info)
        # Inspect full disk mounts, RAM, CPU cores, OS, CUDA/uv/tmux, and HF cache on remote host.
        REMOTE_INFO_SCRIPT="$(cat <<EOF
echo "=== Host ==="
hostname && uname -a && (cat /etc/os-release | grep PRETTY_NAME || true)
echo "=== CPU & RAM ==="
nproc && free -h
echo "=== Disks (df -h) ==="
df -h -x tmpfs -x devtmpfs -x squashfs 2>/dev/null || df -h
echo "=== Large extra dirs (/mnt /data /scratch /opt /home) ==="
ls -ld /mnt /data /scratch /local /home/* 2>/dev/null || true
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,driver_version --format=csv,noheader 2>/dev/null || true
echo "=== Toolchain ==="
which python3 uv tmux git nvcc nvidia-smi 2>/dev/null || true
(nvcc --version 2>/dev/null | tail -n 2) || true
echo "=== Repo & HF Cache ==="
ls -ld '${REMOTE_DIR}' '${HF_HOME}' 2>/dev/null || true
du -sh '${REMOTE_DIR}' '${HF_HOME}' 2>/dev/null || true
EOF
)"
        run_ssh "${REMOTE_INFO_SCRIPT}"
        ;;

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
        PROF="${1:-${REMOTE_PROFILE}}"
        run_rsync_push
        ENV_PRE="$(remote_env_prefix "${PROF}")"
        set +e
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && bash scripts/setup_server.sh && .venv/bin/python -m minorsafe.cli preflight --hardware '${PROF}'$( [[ "${REQUIRE_EXCLUSIVE}" == "0" ]] && echo ' --allow-busy' || true )"
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
        ENV_PRE="$(remote_env_prefix "${PROF}")"
        ALLOW_FLAG=""
        if [[ "${REQUIRE_EXCLUSIVE}" == "0" ]]; then
            ALLOW_FLAG=" --allow-busy"
        fi
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && .venv/bin/python -m minorsafe.cli preflight --hardware '${PROF}'${ALLOW_FLAG}"
        ;;

    check-gates)
        PROF="${1:-${REMOTE_PROFILE}}"
        run_rsync_push
        ENV_PRE="$(remote_env_prefix "${PROF}")"
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && .venv/bin/python scripts/download_models.py --profile '${PROF}' --check-gates"
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
        ENV_PRE="$(remote_env_prefix "${RUN_PROFILE}")"
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
    setsid nohup bash -lc $(printf '%q' "${RUN_CMD}") > '${REMOTE_DIR}/results/logs/${PHASE}_detached.log' 2>&1 < /dev/null &
    DETACH_PID="\$!"
    sleep 1
    kill -0 "\${DETACH_PID}" 2>/dev/null || { echo "[remote] ERROR: Detached process exited immediately." >&2; exit 1; }
    echo "[remote] Started detached PID \${DETACH_PID}. Safe to disconnect; monitor with: bash scripts/remote.sh status"
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

    stop)
        run_ssh "$(_remote_stop_cmd)"
        ;;

    reset)
        KEEP_DATA=0
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --keep-data) KEEP_DATA=1 ;;
                *)
                    echo "Unknown reset option: $1" >&2
                    echo "Usage: bash scripts/remote.sh reset [--keep-data]" >&2
                    exit 1
                    ;;
            esac
            shift
        done
        RESET_CMD="$(_remote_stop_cmd)"
        RESET_CMD+=$'\n'"rm -rf '${REMOTE_DIR}/results/runs' '${REMOTE_DIR}/results/tomography' '${REMOTE_DIR}/results/pilot_archive' '${REMOTE_DIR}/results/REPORT.md' '${REMOTE_DIR}/results/cost_table.csv' '${REMOTE_DIR}/results/summary.parquet' '${REMOTE_DIR}/results/figures' '${REMOTE_DIR}/results/tables'"
        if [[ "${KEEP_DATA}" == "0" ]]; then
            RESET_CMD+=$'\n'"rm -rf '${REMOTE_DIR}/data/processed' '${REMOTE_DIR}/results/data_card.md' '${REMOTE_DIR}/results/data_synth_setup.json'"
        fi
        RESET_CMD+=$'\n'"echo '[remote] Cleared stale remote experiment artifacts (logs and HF cache preserved).'"
        run_ssh "${RESET_CMD}"
        ;;

    status)
        run_ssh "cd '${REMOTE_DIR}' && ( .venv/bin/python -m minorsafe.cli status || true ) && echo '--- autopilot ---' && ( [ -f results/logs/autopilot.status ] && cat results/logs/autopilot.status || echo 'no autopilot status' ) && echo '--- tmux ---' && ( tmux ls 2>/dev/null || echo 'no tmux sessions' ) && echo '--- background procs ---' && ( pgrep -af '[r]un_autopilot\.sh|[m]inorsafe\.cli|[r]un_msba\.sh|[r]un_all_tests\.sh|[r]un_phase\.sh' || echo 'no detached minorsafe procs' ) && echo '--- GPUs ---' && ( nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi' ) && echo '--- latest log ---' && ( [ -f results/logs/latest.log ] && tail -n 15 results/logs/latest.log || echo 'no logs yet' )"
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
        PROF="${1:-${REMOTE_PROFILE}}"
        run_rsync_push
        ENV_PRE="$(remote_env_prefix "${PROF}")"
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && .venv/bin/python -m pytest -v -p no:cacheprovider -m gpu tests/gpu/"
        ;;

    test-all)
        PROF="${1:-${REMOTE_PROFILE}}"
        if [[ $# -ge 1 ]]; then
            shift
        fi
        EXTRA_FLAGS="$*"
        run_rsync_push
        ENV_PRE="$(remote_env_prefix "${PROF}")"
        run_ssh "${ENV_PRE} cd '${REMOTE_DIR}' && bash scripts/run_all_tests.sh '${PROF}' ${EXTRA_FLAGS}"
        ;;

    autopilot)
        PROF="${REMOTE_PROFILE}"
        if [[ $# -ge 1 && "$1" != --* ]]; then
            PROF="$1"
            shift
        fi
        SKIP_SETUP=0
        AUTO_ARGS=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --skip-setup) SKIP_SETUP=1 ;;
                --skip-tests|--skip-download|--skip-pilot|--skip-quant|--with-tomography)
                    AUTO_ARGS+=("$1")
                    ;;
                *)
                    echo "Unknown autopilot option: $1" >&2
                    echo "Usage: bash scripts/remote.sh autopilot [profile] [--skip-setup] [--skip-tests] [--skip-download] [--skip-pilot] [--skip-quant] [--with-tomography]" >&2
                    exit 1
                    ;;
            esac
            shift
        done

        run_rsync_push
        ENV_PRE="$(remote_env_prefix "${PROF}")"
        ALLOW_FLAG=""
        if [[ "${REQUIRE_EXCLUSIVE}" == "0" ]]; then
            ALLOW_FLAG=" --allow-busy"
        fi

        PRE_CMD="${ENV_PRE} cd '${REMOTE_DIR}'"
        if [[ "${SKIP_SETUP}" == "0" ]]; then
            PRE_CMD+=" && bash scripts/setup_server.sh"
        fi
        PRE_CMD+=" && .venv/bin/python -m minorsafe.cli preflight --hardware '${PROF}'${ALLOW_FLAG}"
        PRE_CMD+=" && .venv/bin/python scripts/download_models.py --profile '${PROF}' --check-gates"

        echo "[remote] Running pre-detach setup, preflight, and HF gated-repo check on ${REMOTE_HOST}..."
        set +e
        run_ssh "${PRE_CMD}"
        PRE_RC=$?
        set -e
        if [[ "${PRE_RC}" -ne 0 ]]; then
            echo "[remote] Pre-launch check failed (exit=${PRE_RC}); NOT starting unattended autopilot." >&2
            pull_remote_logs
            exit "${PRE_RC}"
        fi

        AUTO_CMD="${ENV_PRE} cd '${REMOTE_DIR}' && bash scripts/run_autopilot.sh '${PROF}'"
        if [[ "${#AUTO_ARGS[@]}" -gt 0 ]]; then
            for arg in "${AUTO_ARGS[@]}"; do
                AUTO_CMD+=" $(printf '%q' "${arg}")"
            done
        fi

        LAUNCH_SCRIPT="$(cat <<EOF
set -euo pipefail
mkdir -p '${REMOTE_DIR}/results/logs'
PID_FILE='${REMOTE_DIR}/results/logs/autopilot.pid'
if [ -f "\${PID_FILE}" ]; then
    OLD_PID="\$(cat "\${PID_FILE}" 2>/dev/null || true)"
    if [ -n "\${OLD_PID}" ] && kill -0 "\${OLD_PID}" 2>/dev/null; then
        echo "[remote] ERROR: Autopilot is already running on \$(hostname) (PID=\${OLD_PID})." >&2
        exit 1
    fi
fi
if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t 'minorsafe_autopilot' 2>/dev/null || true
    tmux new-session -d -s 'minorsafe_autopilot' bash -lc $(printf '%q' "${AUTO_CMD}")
    echo "[remote] Started detached tmux session 'minorsafe_autopilot'."
else
    setsid nohup bash -lc $(printf '%q' "${AUTO_CMD}") > '${REMOTE_DIR}/results/logs/autopilot_detached.log' 2>&1 < /dev/null &
    DETACH_PID="\$!"
    sleep 2
    kill -0 "\${DETACH_PID}" 2>/dev/null || { echo "[remote] ERROR: Autopilot failed to stay running after launch." >&2; exit 1; }
    echo "[remote] Started detached autopilot PID \${DETACH_PID} on \$(hostname)."
fi
EOF
)"
        run_ssh "${LAUNCH_SCRIPT}"
        echo "[remote] Safe to close or disconnect your laptop now."
        echo "[remote] Monitor anytime: bash scripts/remote.sh status"
        echo "[remote] Stream logs:     bash scripts/remote.sh logs"
        echo "[remote] Pull results:    bash scripts/remote.sh pull"
        ;;

    *)
        echo "Unknown subcommand: ${SUBCMD}" >&2
        echo "Valid subcommands: info check users setup-ssh-key sync setup preflight gpu-test test-all autopilot run stop reset status logs pull" >&2
        exit 1
        ;;
esac
