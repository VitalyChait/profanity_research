"""Smart structured run logger: human-readable progress + ETA, live_status.json, and events.jsonl."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minorsafe.utils import RESULTS_DIR, get_logger, write_json

log = get_logger("minorsafe.progress")


def _clean_float(val: float) -> float | None:
    return float(val) if isinstance(val, (int, float)) and math.isfinite(val) else None


class SmartRunLogger:
    """Emits live ETA progress lines alongside ``events.jsonl`` and ``live_status.json``."""

    def __init__(
        self,
        experiment: str,
        profile: str,
        total_runs: int,
        log_dir: Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.profile = profile
        self.total_runs = max(1, int(total_runs))
        self.log_dir = Path(log_dir) if log_dir else (RESULTS_DIR / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self.status_path = self.log_dir / "live_status.json"

        self._t_exp_start = time.perf_counter()
        self._total_examples_per_run = 0
        self._n_seeds = 1
        self._total_expected = 0
        self._total_done = 0
        self._last_seed_done: dict[tuple[int, int], int] = {}
        self._current_run_meta: dict[str, Any] = {}

    def _now_iso(self) -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        import json

        rec = {"ts": self._now_iso(), "event": event_type, "experiment": self.experiment, **payload}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")

    def _write_status(self, state: str, **extra: Any) -> None:
        status = {
            "experiment": self.experiment,
            "profile": self.profile,
            "state": state,
            "total_runs": self.total_runs,
            "elapsed_s": round(time.perf_counter() - self._t_exp_start, 2),
            "updated_at": self._now_iso(),
            **extra,
        }
        write_json(self.status_path, status)

    def start_experiment(self, total_examples_per_run: int, seeds: list[int]) -> None:
        self._t_exp_start = time.perf_counter()
        self._total_examples_per_run = max(0, int(total_examples_per_run))
        self._n_seeds = max(1, len(seeds))
        self._total_expected = self.total_runs * self._total_examples_per_run * self._n_seeds
        self._total_done = 0
        self._last_seed_done.clear()

        log.info(
            "Starting experiment=%s profile=%s runs=%d examples/run=%d seeds=%s",
            self.experiment,
            self.profile,
            self.total_runs,
            self._total_examples_per_run,
            seeds,
        )
        self._emit_event(
            "experiment_start",
            {
                "profile": self.profile,
                "total_runs": self.total_runs,
                "examples_per_run": self._total_examples_per_run,
                "seeds": list(seeds),
            },
        )
        self._write_status("running", run_idx=0, run_id="", progress="0%", eta_s=None)

    def start_run(
        self,
        run_idx: int,
        run_id: str,
        method: str,
        model: str,
        quant: str,
        setup_wall_s: float,
        setup_tokens: int,
    ) -> None:
        self._current_run_meta = {
            "run_idx": run_idx,
            "run_id": run_id,
            "method": method,
            "model": model,
            "quant": quant,
        }
        log.info(
            "[run %d/%d %s] method=%s model=%s quant=%s setup=%.2fs (%d tok)",
            run_idx,
            self.total_runs,
            run_id,
            method,
            model,
            quant,
            setup_wall_s,
            setup_tokens,
        )
        self._emit_event(
            "run_start",
            {
                **self._current_run_meta,
                "setup_wall_s": float(setup_wall_s),
                "setup_tokens": int(setup_tokens),
            },
        )
        self._write_status("running", **self._current_run_meta, progress="0%", eta_s=None)

    def log_batch(
        self,
        run_idx: int,
        run_id: str,
        seed: int,
        batch_done: int,
        total_todo: int,
        batch_e2e_ms: float,
        decode_tok_per_s: float,
        peak_vram_mb: float,
        energy_j: float,
    ) -> None:
        key = (run_idx, seed)
        prev = self._last_seed_done.get(key, 0)
        delta = max(0, int(batch_done) - prev)
        self._last_seed_done[key] = max(prev, int(batch_done))
        self._total_done += delta

        elapsed = max(1e-6, time.perf_counter() - self._t_exp_start)
        pct = (100.0 * batch_done / max(1, total_todo)) if total_todo > 0 else 100.0
        rate = self._total_done / elapsed if self._total_done > 0 else 0.0
        if self._total_expected > 0:
            remaining = max(0, self._total_expected - self._total_done)
        else:
            remaining = max(0, total_todo - batch_done)
        eta_s = (remaining / rate) if rate > 0 else 0.0

        vram_str = f"{peak_vram_mb:.0f}MB" if math.isfinite(peak_vram_mb) else "n/a"
        log.info(
            "[run %d/%d %s] seed=%d %d/%d (%.0f%%) | e2e=%.1fms | tok/s=%.1f | vram=%s | eta=%.0fs",
            run_idx,
            self.total_runs,
            run_id,
            seed,
            batch_done,
            total_todo,
            pct,
            batch_e2e_ms,
            decode_tok_per_s,
            vram_str,
            eta_s,
        )
        self._emit_event(
            "batch_done",
            {
                "run_idx": run_idx,
                "run_id": run_id,
                "seed": seed,
                "batch_done": int(batch_done),
                "total_todo": int(total_todo),
                "pct": round(pct, 1),
                "batch_e2e_ms": round(float(batch_e2e_ms), 2),
                "decode_tok_per_s": round(float(decode_tok_per_s), 2),
                "peak_vram_mb": _clean_float(peak_vram_mb),
                "energy_j": _clean_float(energy_j),
                "eta_s": round(eta_s, 1),
            },
        )
        self._write_status(
            "running",
            run_idx=run_idx,
            run_id=run_id,
            seed=seed,
            progress=f"{batch_done}/{total_todo} ({pct:.0f}%)",
            decode_tok_per_s=round(float(decode_tok_per_s), 2),
            peak_vram_mb=_clean_float(peak_vram_mb),
            eta_s=round(eta_s, 1),
        )

    def finish_run(
        self,
        run_idx: int,
        run_id: str,
        n_rows: int,
        success_mean: float | None = None,
    ) -> None:
        succ_str = (
            f"{success_mean:.3f}" if (success_mean is not None and math.isfinite(success_mean)) else "n/a"
        )
        log.info("[run %d/%d %s] DONE rows=%d success=%s", run_idx, self.total_runs, run_id, n_rows, succ_str)
        self._emit_event(
            "run_done",
            {
                "run_idx": run_idx,
                "run_id": run_id,
                "n_rows": int(n_rows),
                "success_mean": _clean_float(success_mean) if success_mean is not None else None,
            },
        )

    def finish_experiment(self, completed_runs: int) -> None:
        elapsed = time.perf_counter() - self._t_exp_start
        log.info(
            "Experiment %s finished: %d/%d runs in %.1fs",
            self.experiment,
            completed_runs,
            self.total_runs,
            elapsed,
        )
        self._emit_event(
            "experiment_done", {"completed_runs": int(completed_runs), "elapsed_s": round(elapsed, 2)}
        )
        self._write_status(
            "completed",
            run_idx=completed_runs,
            run_id="",
            progress=f"{completed_runs}/{self.total_runs} runs",
            eta_s=0.0,
        )
