"""claude_issues_found.md M10: GPU energy that could not be measured must be NaN, never 0 J."""

from __future__ import annotations

import math
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest
import torch

import minorsafe.telemetry.gpu as gpu_mod
from minorsafe.telemetry.gpu import GPUSampler

pytestmark = pytest.mark.unit


def _fake_nvml(
    monkeypatch, *, watts=100.0, power_ok_reads=None, memory_ok=True, init_ok=True, seen_bus_ids=None
):
    """Install a pynvml stand-in. ``power_ok_reads=n`` makes every power read after the n-th fail."""
    state = {"power_reads": 0}
    nvml = ModuleType("pynvml")

    def init():
        if not init_ok:
            raise RuntimeError("NVML Shared Library Not Found")

    def power(handle):
        state["power_reads"] += 1
        if power_ok_reads is not None and state["power_reads"] > power_ok_reads:
            raise RuntimeError("Not Supported")
        return int(watts * 1000)  # milliwatts

    def memory(handle):
        if not memory_ok:
            raise RuntimeError("Not Supported")  # GB10: memory.used [N/A]
        return SimpleNamespace(used=2048 * 2**20)

    def by_bus_id(bus_id):
        if seen_bus_ids is not None:
            seen_bus_ids.append(bus_id)
        return "handle"

    nvml.nvmlInit = init
    nvml.nvmlDeviceGetHandleByIndex = lambda i: "handle"
    nvml.nvmlDeviceGetHandleByPciBusId = by_bus_id
    nvml.nvmlDeviceGetPowerUsage = power
    nvml.nvmlDeviceGetMemoryInfo = memory
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    return state


def _measure(seconds: float, **kw) -> tuple[GPUSampler, float]:
    t0 = time.perf_counter()
    with GPUSampler(device_index=0, **kw) as s:
        time.sleep(seconds)
    return s, time.perf_counter() - t0


@pytest.mark.parametrize(
    "fake",
    [
        {"power_ok_reads": 0},  # the issue's case: the first read fails (was a finite 0.0 J)
        {"power_ok_reads": 3},  # fails midway: a partial integral would under-report
        {"init_ok": False},  # no NVML at all
    ],
)
def test_unmeasured_energy_is_nan(monkeypatch, fake):
    _fake_nvml(monkeypatch, **fake)
    s, _ = _measure(0.2, interval_s=0.01)
    assert math.isnan(s.energy_j)
    assert s.failure


def test_a_failed_memory_read_does_not_stop_power_sampling(monkeypatch):
    _fake_nvml(monkeypatch, watts=100.0, memory_ok=False)
    s, wall = _measure(0.2, interval_s=0.01)
    assert s.failure is None and s.samples > 3
    # Constant 100 W over the whole block, including the stretch after the last periodic sample.
    assert s.energy_j == pytest.approx(100.0 * wall, rel=0.1)


def test_a_block_shorter_than_the_interval_is_still_measured(monkeypatch):
    _fake_nvml(monkeypatch, watts=200.0)
    s, wall = _measure(0.02, interval_s=1.0)
    assert s.samples == 1 and s.energy_j == pytest.approx(200.0 * wall, rel=0.5)


def test_default_device_is_the_gpu_torch_computes_on(monkeypatch):
    # NVML ignores CUDA_VISIBLE_DEVICES: with GPU 3 pinned, NVML index 0 is someone else's GPU.
    seen: list[str] = []
    _fake_nvml(monkeypatch, seen_bus_ids=seen)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(pci_domain_id=0, pci_bus_id=0xC1, pci_device_id=0),
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    with GPUSampler(interval_s=0.01) as s:
        time.sleep(0.03)
    assert seen == ["0000:c1:00.0"]
    assert math.isfinite(s.energy_j)


def test_failure_is_logged_once_not_per_batch(monkeypatch, caplog):
    _fake_nvml(monkeypatch, power_ok_reads=0)
    monkeypatch.setattr(gpu_mod, "_WARNED", set())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    for _ in range(3):
        _measure(0.03, interval_s=0.01)
    # minorsafe loggers may not propagate to caplog; the dedup set is the contract.
    assert len(gpu_mod._WARNED) == 1
