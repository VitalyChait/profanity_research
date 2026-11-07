"""Token- and time-based cost model (no currency - convert prices externally).

    Time(N)   = T_setup + N * mean(T_query)
    Tokens(N) = K_setup + N * mean(K_query)
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from minorsafe.telemetry.timer import cuda_sync


@dataclass
class SetupStage:
    name: str
    wall_s: float = 0.0
    gpu_s: float = 0.0
    tokens: int = 0
    examples: int = 0
    peak_vram_mb: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class SetupReport:
    """Ramp-up cost of a method (data generation, training, extraction ...)."""

    method: str
    setup_key: str
    stages: list[SetupStage] = field(default_factory=list)
    cached: bool = False  # True when artifacts were reused from a previous run
    num_gpus: int = 1

    @contextmanager
    def stage(self, name: str, examples: int = 0):
        import torch

        st = SetupStage(name=name, examples=examples)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cuda_sync()
        t0 = time.perf_counter()
        try:
            yield st
        finally:
            cuda_sync()
            st.wall_s = time.perf_counter() - t0
            st.gpu_s = st.wall_s * self.num_gpus if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                st.peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20
            self.stages.append(st)

    @property
    def wall_s(self) -> float:
        return sum(s.wall_s for s in self.stages)

    @property
    def gpu_s(self) -> float:
        return sum(s.gpu_s for s in self.stages)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.stages)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(total_wall_s=self.wall_s, total_gpu_s=self.gpu_s, total_tokens=self.tokens)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SetupReport:
        stages = [SetupStage(**s) for s in d.get("stages", [])]
        return cls(
            method=d["method"],
            setup_key=d["setup_key"],
            stages=stages,
            cached=d.get("cached", False),
            num_gpus=d.get("num_gpus", 1),
        )


def total_cost(setup: float, per_query: float, n: int | float) -> float:
    return setup + n * per_query


def break_even(setup_a: float, per_query_a: float, setup_b: float, per_query_b: float) -> float:
    """Smallest N >= 0 at which method A becomes no more expensive than B (time or tokens).

    Returns 0 if A is never more expensive, ``inf`` if A never catches up.
    """
    if setup_a <= setup_b and per_query_a <= per_query_b:
        return 0.0
    if per_query_a >= per_query_b:
        return math.inf
    return max(0.0, (setup_a - setup_b) / (per_query_b - per_query_a))


def efficiency(success_rate: float, tokens_per_query: float, seconds_per_query: float) -> dict:
    return {
        "success_per_1k_tokens": 1000.0 * success_rate / tokens_per_query if tokens_per_query > 0 else math.nan,
        "success_per_gpu_s": success_rate / seconds_per_query if seconds_per_query > 0 else math.nan,
        "tokens_per_success": tokens_per_query / success_rate if success_rate > 0 else math.inf,
        "seconds_per_success": seconds_per_query / success_rate if success_rate > 0 else math.inf,
    }

