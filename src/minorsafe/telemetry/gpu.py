"""Background NVML sampler for power (energy) and memory."""

from __future__ import annotations

import threading
import time

import torch

from minorsafe.utils import get_logger

log = get_logger(__name__)
_WARNED: set[str] = set()  # failure reasons already logged (one warning, not one per batch)


def _nvml_handle(nvml, device_index: int | None):
    """NVML handle of ``device_index``, or by default of the GPU PyTorch is computing on.

    NVML numbers GPUs physically and ignores ``CUDA_VISIBLE_DEVICES``, so on a pinned or SLURM-allocated GPU
    NVML index 0 can be another job's GPU. The default therefore matches PyTorch's device by PCI bus id.
    """
    if device_index is not None:
        return nvml.nvmlDeviceGetHandleByIndex(device_index)
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device to measure")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    bus_id = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    return nvml.nvmlDeviceGetHandleByPciBusId(bus_id)


class GPUSampler:
    """Samples GPU power every ``interval_s`` seconds and integrates it into energy (J) over a ``with`` block.

    ``energy_j`` is NaN unless the whole block was measured: NVML must be present, the GPU must match an
    NVML device, and every power read must succeed. A failed measurement never reads as 0 J. A failed
    memory read only loses the NVML memory figure: GB10 reports ``memory.used [N/A]``, so power keeps
    being sampled and ``peak_mem_mb`` still comes from PyTorch.

    Usage::

        with GPUSampler() as s:
            ...
        s.energy_j, s.peak_mem_mb
    """

    def __init__(self, device_index: int | None = None, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = interval_s
        self.peak_mem_mb = 0.0
        self.samples = 0
        self.available = False
        self.failure: str | None = None
        self._energy_j = 0.0
        self._last = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception as exc:
            self.failure = f"NVML unavailable: {exc}"
            return
        self._nvml = pynvml
        try:
            self._handle = _nvml_handle(pynvml, device_index)
            self.available = True
        except Exception as exc:
            self.failure = f"no NVML device for this GPU: {exc}"

    @property
    def energy_j(self) -> float:
        """Energy over the block in joules; NaN when it was not measured end to end."""
        if not self.available or self.failure is not None or self.samples == 0:
            return float("nan")
        return self._energy_j

    def _sample(self) -> bool:
        """Add the energy since the previous sample. Returns False once power can no longer be read."""
        t = time.perf_counter()
        try:
            watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception as exc:
            self.failure = f"power read failed: {exc}"
            return False
        self._energy_j += watts * (t - self._last)
        self._last = t
        self.samples += 1
        try:
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
            self.peak_mem_mb = max(self.peak_mem_mb, mem)
        except Exception:
            pass
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self._sample():
                break

    def __enter__(self) -> GPUSampler:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.available:
            self._energy_j, self.samples, self.failure = 0.0, 0, None
            self._stop.clear()
            self._last = time.perf_counter()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                self.failure = "sampler thread did not stop"
            elif self.failure is None:
                self._sample()  # the stretch since the last sample
        if torch.cuda.is_available():
            self.peak_mem_mb = max(self.peak_mem_mb, torch.cuda.max_memory_allocated() / 2**20)
        if self.failure is not None and torch.cuda.is_available() and self.failure not in _WARNED:
            _WARNED.add(self.failure)
            log.warning("GPU energy not measured, recorded as NaN: %s", self.failure)
