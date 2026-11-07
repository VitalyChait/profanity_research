"""Telemetry: CUDA-synchronised timers, token accounting, GPU sampling and token/time cost models."""

from minorsafe.telemetry.cost import SetupReport, break_even, efficiency
from minorsafe.telemetry.gpu import GPUSampler
from minorsafe.telemetry.timer import Stopwatch, TimingLogitsProcessor, cuda_sync
from minorsafe.telemetry.tokens import TokenUsage

__all__ = [
    "GPUSampler",
    "SetupReport",
    "Stopwatch",
    "TimingLogitsProcessor",
    "TokenUsage",
    "break_even",
    "cuda_sync",
    "efficiency",
]

