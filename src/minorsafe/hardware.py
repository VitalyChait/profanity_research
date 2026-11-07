"""GPU detection and automatic hardware-profile selection."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass

from minorsafe.utils import get_logger

log = get_logger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int


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


def choose_profile(gpus: list[GPUInfo]) -> str:
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
        import os

        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        me = os.getpid()
        return all(p.pid == me for p in procs)
    except Exception:
        return True

