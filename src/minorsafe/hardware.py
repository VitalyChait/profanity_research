"""GPU detection, hardware profile selection, server exclusivity, run lock, credential & preflight checks."""

from __future__ import annotations

import contextlib
import getpass
import os
import shutil
import socket
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from minorsafe.utils import REPO_ROOT, get_logger, package_versions, read_json, write_json

log = get_logger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int


@dataclass
class GPUStatus:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int = 0
    utilization_pct: int = 0
    temperature_c: int = 0


@dataclass
class ExclusivityReport:
    exclusive: bool
    gpu_processes: list[dict[str, Any]]
    gpu_busy_devices: list[dict[str, Any]]
    other_logged_in_users: list[str]
    active_lock: dict[str, Any] | None
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val.strip()))
    except (TypeError, ValueError, AttributeError):
        return default


def parse_nvidia_smi(output: str) -> list[GPUInfo]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits``."""
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(GPUInfo(int(parts[0]), parts[1], int(float(parts[2]))))
        except ValueError:
            continue
    return gpus


def parse_nvidia_smi_status(output: str) -> list[GPUStatus]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu``."""
    statuses: list[GPUStatus] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            mem_tot = int(float(parts[2]))
        except ValueError:
            continue
        mem_used = _safe_int(parts[3], 0) if len(parts) > 3 else 0
        util = _safe_int(parts[4], 0) if len(parts) > 4 else 0
        temp = _safe_int(parts[5], 0) if len(parts) > 5 else 0
        statuses.append(
            GPUStatus(
                index=idx,
                name=name,
                memory_total_mb=mem_tot,
                memory_used_mb=mem_used,
                utilization_pct=util,
                temperature_c=temp,
            )
        )
    return statuses


def detect_gpus() -> list[GPUInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("nvidia-smi failed: %s", exc)
        return []
    return parse_nvidia_smi(out)


def detect_gpu_status() -> list[GPUStatus]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("nvidia-smi status query failed: %s", exc)
        return []
    return parse_nvidia_smi_status(out)


def choose_profile(gpus: list[GPUInfo] | list[GPUStatus]) -> str:
    """Map detected GPUs to a profile name in configs/hardware/."""
    if not gpus:
        return "cpu_debug"
    min_mem_gb = min(g.memory_total_mb for g in gpus) / 1024
    if len(gpus) >= 2 and min_mem_gb >= 70:
        return "multi_gpu_80gb"
    if min_mem_gb >= 70:
        return "gpu_80gb"
    if min_mem_gb >= 40:
        return "gpu_48gb"
    if min_mem_gb >= 20:
        return "gpu_24gb"
    return "cpu_debug"


def resolve_profile_name(name: str | None) -> str:
    if name and name != "auto":
        return name
    gpus = detect_gpus()
    prof = choose_profile(gpus)
    log.info("Auto-selected hardware profile %r for GPUs: %s", prof, [g.name for g in gpus] or "none")
    return prof


def gpu_summary() -> list[dict]:
    return [asdict(g) for g in detect_gpus()]


def gpu_is_exclusive(device_index: int = 0) -> bool:
    """True when no *other* process holds memory on the GPU (best-effort, via NVML)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        me = os.getpid()
        return all(p.pid == me for p in procs)
    except Exception:
        return True


def parse_nvidia_compute_apps(output: str, ignore_pids: set[int] | None = None) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits``."""
    ignored = {os.getpid()} if ignore_pids is None else set(ignore_pids)
    apps: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(float(parts[0]))
        except ValueError:
            continue
        if pid in ignored:
            continue
        apps.append(
            {
                "pid": pid,
                "process_name": parts[1],
                "used_memory_mb": _safe_int(parts[2], 0),
            }
        )
    return apps


def _resolve_current_user(current_user: str | None = None) -> str:
    if current_user:
        return current_user
    env_user = os.environ.get("USER")
    if env_user:
        return env_user
    return getpass.getuser()


def parse_who_users(output: str, current_user: str | None = None) -> list[str]:
    """Parse ``who`` output and return sorted unique usernames other than ``current_user``."""
    me = _resolve_current_user(current_user)
    others: set[str] = set()
    for line in output.strip().splitlines():
        toks = line.split()
        if not toks:
            continue
        user = toks[0]
        if user and user != me:
            others.add(user)
    return sorted(others)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _read_active_lock(lock_path: Path) -> dict[str, Any] | None:
    if not lock_path.exists():
        return None
    try:
        data = read_json(lock_path)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    pid = data.get("pid")
    if isinstance(pid, int) and pid != os.getpid() and _pid_alive(pid):
        return data
    return None


def check_server_exclusivity(
    current_user: str | None = None,
    max_idle_mem_mb: int = 500,
    max_idle_util_pct: int = 10,
    allow_other_sessions_same_user: bool = True,
    lock_path: Path | None = None,
    _nvidia_gpu_out: str | None = None,
    _nvidia_apps_out: str | None = None,
    _who_out: str | None = None,
) -> ExclusivityReport:
    """Validate that no other users, foreign GPU compute jobs, or active run locks occupy the server."""
    me = _resolve_current_user(current_user)

    if _nvidia_apps_out is not None:
        gpu_processes = parse_nvidia_compute_apps(_nvidia_apps_out)
    elif shutil.which("nvidia-smi") is not None:
        try:
            apps_out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=20,
            )
            gpu_processes = parse_nvidia_compute_apps(apps_out)
        except Exception as exc:  # pragma: no cover
            log.warning("nvidia-smi compute-apps query failed: %s", exc)
            gpu_processes = []
    else:
        gpu_processes = []

    gpu_statuses = (
        parse_nvidia_smi_status(_nvidia_gpu_out) if _nvidia_gpu_out is not None else detect_gpu_status()
    )
    gpu_busy_devices = [
        asdict(g)
        for g in gpu_statuses
        if g.memory_used_mb > max_idle_mem_mb or g.utilization_pct > max_idle_util_pct
    ]

    if _who_out is not None:
        who_text = _who_out
    elif shutil.which("who") is not None:
        try:
            who_text = subprocess.check_output(["who"], text=True, timeout=10)
        except Exception:  # pragma: no cover
            who_text = ""
    else:
        who_text = ""

    other_users = parse_who_users(who_text, current_user=me)
    resolved_lock = Path(lock_path) if lock_path else (REPO_ROOT / ".minorsafe.lock")
    active_lock = _read_active_lock(resolved_lock)

    reasons: list[str] = []
    if gpu_processes:
        proc_desc = ", ".join(
            f"pid={p['pid']} ({p['process_name']}, {p['used_memory_mb']}MB)" for p in gpu_processes
        )
        reasons.append(f"{len(gpu_processes)} foreign GPU compute process(es) active: {proc_desc}")
    if gpu_busy_devices:
        dev_desc = ", ".join(
            f"gpu{d['index']} ({d['memory_used_mb']}MB, {d['utilization_pct']}%)" for d in gpu_busy_devices
        )
        reasons.append(
            f"GPU(s) exceed idle thresholds (>{max_idle_mem_mb}MB or >{max_idle_util_pct}%): {dev_desc}"
        )
    if other_users:
        reasons.append(f"Other user(s) logged into server: {', '.join(other_users)}")
    if not allow_other_sessions_same_user:
        same_user_sessions = sum(
            1 for ln in who_text.strip().splitlines() if ln.split() and ln.split()[0] == me
        )
        if same_user_sessions > 1:
            reasons.append(f"Multiple active sessions ({same_user_sessions}) for user {me!r}")
    if active_lock is not None:
        reasons.append(
            f"Active MinorSafe run lock held by PID {active_lock.get('pid')} "
            f"(phase={active_lock.get('phase')})"
        )

    return ExclusivityReport(
        exclusive=(len(reasons) == 0),
        gpu_processes=gpu_processes,
        gpu_busy_devices=gpu_busy_devices,
        other_logged_in_users=other_users,
        active_lock=active_lock,
        reasons=reasons,
    )


@contextlib.contextmanager
def acquire_run_lock(lock_path: Path | None = None, phase: str = "run"):
    """Acquire an exclusive PID lockfile for the duration of a MinorSafe run."""
    resolved = Path(lock_path) if lock_path else (REPO_ROOT / ".minorsafe.lock")
    existing = _read_active_lock(resolved)
    if existing is not None:
        raise RuntimeError(f"Another MinorSafe process holds {resolved}: {existing}")
    payload = {
        "pid": os.getpid(),
        "user": _resolve_current_user(),
        "host": socket.gethostname(),
        "phase": phase,
        "started_at": time.time(),
    }
    write_json(resolved, payload)
    try:
        yield resolved
    finally:
        try:
            if resolved.exists():
                cur = read_json(resolved)
                if isinstance(cur, dict) and cur.get("pid") == os.getpid():
                    resolved.unlink()
        except Exception:
            pass


def check_credential_safety(repo_root: Path | None = None) -> dict[str, Any]:
    """Verify ``.remote.env`` is gitignored, untracked in git, and has owner-only (0600) permissions."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    env_path = root / ".remote.env"
    env_exists = env_path.exists()
    errors: list[str] = []
    warnings: list[str] = []

    use_git = shutil.which("git") is not None and (root / ".git").exists()
    if use_git:
        ign_res = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", ".remote.env"],
            check=False,
        )
        gitignored = ign_res.returncode == 0
        trk_res = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", ".remote.env"],
            capture_output=True,
            check=False,
        )
        tracked = trk_res.returncode == 0
    else:
        gi_path = root / ".gitignore"
        gi_lines = gi_path.read_text(encoding="utf-8").splitlines() if gi_path.exists() else []
        gitignored = any(ln.strip() == ".remote.env" for ln in gi_lines)
        tracked = False

    if not gitignored:
        errors.append(".remote.env is NOT ignored by .gitignore; add '.remote.env' before creating it.")
    if tracked:
        errors.append(".remote.env is tracked in git! Run `git rm --cached .remote.env` immediately.")

    mode_octal = "n/a"
    if env_exists:
        mode = stat.S_IMODE(env_path.stat().st_mode)
        if mode & 0o077 != 0:
            env_path.chmod(0o600)
            warnings.append(f"Tightened permissions on {env_path} from {oct(mode)} to 0o600.")
            mode = stat.S_IMODE(env_path.stat().st_mode)
        mode_octal = oct(mode)

    return {
        "ok": len(errors) == 0,
        "env_exists": env_exists,
        "gitignored": gitignored,
        "tracked": tracked,
        "mode_octal": mode_octal,
        "errors": errors,
        "warnings": warnings,
    }


def _existing_ancestor(path: Path) -> Path:
    cur = path.resolve()
    while not cur.exists() and cur.parent != cur:
        cur = cur.parent
    return cur


def run_preflight_checks(
    profile_name: str = "auto",
    require_exclusive: bool = True,
    min_free_disk_gb: float = 5.0,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run all initial checks before executing benchmarks (hardware, CUDA, exclusivity, secrets, disk, HF)."""
    import torch

    from minorsafe.config import load_hardware, load_models

    root = Path(repo_root) if repo_root else REPO_ROOT
    resolved = resolve_profile_name(profile_name)
    hw = load_hardware(resolved)
    models = load_models()

    gpus = detect_gpu_status()
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    bf16_supported = bool(torch.cuda.is_bf16_supported()) if cuda_available else False
    pkgs = package_versions()

    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    repo_usage = shutil.disk_usage(_existing_ancestor(root))
    hf_usage = shutil.disk_usage(_existing_ancestor(hf_home))
    repo_free_gb = round(repo_usage.free / (1024**3), 2)
    hf_free_gb = round(hf_usage.free / (1024**3), 2)

    gated_models = sorted(
        {
            alias
            for alias in (*hw.generators, *hw.decoder_guards, *hw.encoder_guards)
            if alias in models and models[alias].gated
        }
    )
    hf_token_present = bool(os.environ.get("HF_TOKEN", "").strip())

    cred = check_credential_safety(root)
    excl = check_server_exclusivity(lock_path=root / ".minorsafe.lock")

    errors: list[str] = list(cred["errors"])
    warnings: list[str] = list(cred["warnings"])

    if hw.device == "cuda":
        if not cuda_available:
            errors.append(f"Profile {resolved!r} requires CUDA, but torch.cuda.is_available() is False.")
        if len(gpus) < max(1, hw.num_gpus):
            errors.append(
                f"Profile {resolved!r} requires >= {max(1, hw.num_gpus)} GPU(s), detected {len(gpus)}."
            )
        if gpus:
            min_gpu_gb = min(g.memory_total_mb for g in gpus) / 1024.0
            if min_gpu_gb < hw.min_vram_gb:
                errors.append(
                    f"Profile {resolved!r} requires >= {hw.min_vram_gb:.1f} GB VRAM, "
                    f"detected {min_gpu_gb:.1f} GB."
                )

    if require_exclusive and not excl.exclusive:
        errors.extend(excl.reasons)
    elif not excl.exclusive:
        warnings.extend(excl.reasons)

    if repo_free_gb < min_free_disk_gb:
        errors.append(f"Low free disk space in {root}: {repo_free_gb:.2f} GB < {min_free_disk_gb:.2f} GB.")
    if hf_free_gb < min_free_disk_gb:
        warnings.append(
            f"Low free disk space in HF_HOME ({hf_home}): {hf_free_gb:.2f} GB < {min_free_disk_gb:.2f} GB."
        )

    if gated_models and not hf_token_present:
        warnings.append(
            f"HF_TOKEN is not set; gated models for profile {resolved!r} ({', '.join(gated_models)}) "
            "cannot be downloaded."
        )

    return {
        "ok": len(errors) == 0,
        "profile": resolved,
        "device": hw.device,
        "allow_placeholders": hw.allow_placeholders,
        "gpus": [asdict(g) for g in gpus],
        "cuda": {
            "available": cuda_available,
            "device_count": device_count,
            "bf16_supported": bf16_supported,
        },
        "packages": pkgs,
        "disk": {
            "repo_path": str(root),
            "repo_free_gb": repo_free_gb,
            "hf_home": str(hf_home),
            "hf_free_gb": hf_free_gb,
        },
        "hf_auth": {
            "token_present": hf_token_present,
            "gated_models": gated_models,
        },
        "credential_safety": cred,
        "exclusivity": excl.to_dict(),
        "warnings": warnings,
        "errors": errors,
    }
