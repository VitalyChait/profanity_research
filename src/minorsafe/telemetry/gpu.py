"""Background NVML sampler for power (energy) and memory."""

from __future__ import annotations

import threading
import time

import torch


class GPUSampler:
    """Samples GPU power every ``interval_s`` seconds and integrates energy (J).

    Usage::

        with GPUSampler() as s:
            ...
        s.energy_j, s.peak_mem_mb
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = interval_s
        self.energy_j = 0.0
        self.peak_mem_mb = 0.0
        self.samples = 0
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:
            self._nvml = None

    def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            t = time.perf_counter()
            try:
                watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
            except Exception:
                break
            self.energy_j += watts * (t - last)
            self.peak_mem_mb = max(self.peak_mem_mb, mem)
            self.samples += 1
            last = t

    def __enter__(self) -> GPUSampler:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if torch.cuda.is_available():
            self.peak_mem_mb = max(self.peak_mem_mb, torch.cuda.max_memory_allocated() / 2**20)
