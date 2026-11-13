"""Phase 1 unit tests: GPU detection and hardware profile auto-selection."""

from __future__ import annotations

import pytest

from minorsafe.hardware import GPUInfo, choose_profile, parse_nvidia_smi, resolve_profile_name

pytestmark = pytest.mark.unit


def test_parse_nvidia_smi_output():
    sample = "0, NVIDIA A100-SXM4-80GB, 81920\n1, NVIDIA A100-SXM4-80GB, 81920\nbad_line_ignored\n"
    gpus = parse_nvidia_smi(sample)
    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].name == "NVIDIA A100-SXM4-80GB"
    assert gpus[0].memory_total_mb == 81920


def test_choose_profile_across_vram_tiers():
    assert choose_profile([]) == "cpu_debug"
    assert choose_profile([GPUInfo(0, "GTX 1650", 4096)]) == "cpu_debug"
    assert choose_profile([GPUInfo(0, "RTX 4090", 24564)]) == "gpu_24gb"
    assert choose_profile([GPUInfo(0, "NVIDIA L40S", 46068)]) == "gpu_48gb"
    assert choose_profile([GPUInfo(0, "NVIDIA H100", 81559)]) == "gpu_80gb"
    assert (
        choose_profile([GPUInfo(0, "NVIDIA A100", 81920), GPUInfo(1, "NVIDIA A100", 81920)])
        == "multi_gpu_80gb"
    )


def test_resolve_profile_name_explicit():
    assert resolve_profile_name("cpu_debug") == "cpu_debug"
    assert resolve_profile_name("gpu_24gb") == "gpu_24gb"
