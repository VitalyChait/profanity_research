"""Wall-clock instrumentation.

* ``Stopwatch`` - nested named spans, CUDA-synchronised at span boundaries.
* ``TimingLogitsProcessor`` - hooks into ``model.generate`` and timestamps every decoding step. The first call
  happens right after prefill, so ``t_first - t_start`` is the time-to-first-token (TTFT) for the whole batch,
  and the spacing of later calls gives decode throughput. Works for any batch size (HF streamers do not).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class Stopwatch:
    """Accumulates durations (seconds) of named spans. Spans may nest and repeat."""

    sync_cuda: bool = True
    spans: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str):
        if self.sync_cuda:
            cuda_sync()
        t0 = now_ns()
        try:
            yield
        finally:
            if self.sync_cuda:
                cuda_sync()
            dt = (now_ns() - t0) / 1e9
            self.spans[name] = self.spans.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def get(self, name: str, default: float = 0.0) -> float:
        return self.spans.get(name, default)

    def total(self, names: list[str] | None = None) -> float:
        if names is None:
            return sum(self.spans.values())
        return sum(self.spans.get(n, 0.0) for n in names)


class TimingLogitsProcessor:
    """A no-op ``LogitsProcessor`` that records a timestamp at every decoding step."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.t_start: int | None = None
        self.step_times: list[int] = []

    def start(self) -> None:
        if self.sync_cuda:
            cuda_sync()
        self.t_start = now_ns()
        self.step_times = []

    def __call__(self, input_ids, scores):
        if self.sync_cuda and scores.is_cuda:
            torch.cuda.synchronize(scores.device)
        self.step_times.append(now_ns())
        return scores

    @property
    def ttft_s(self) -> float:
        if self.t_start is None or not self.step_times:
            return float("nan")
        return (self.step_times[0] - self.t_start) / 1e9

    @property
    def n_steps(self) -> int:
        return len(self.step_times)

    def decode_seconds(self) -> float:
        if len(self.step_times) < 2:
            return 0.0
        return (self.step_times[-1] - self.step_times[0]) / 1e9

    def decode_tok_per_s(self, n_tokens: int | None = None) -> float:
        """Per-sequence decode throughput: (tokens after the first) / decode time."""
        n = (n_tokens if n_tokens is not None else self.n_steps) - 1
        dt = self.decode_seconds()
        return n / dt if dt > 0 and n > 0 else float("nan")
