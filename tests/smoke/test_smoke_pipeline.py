"""Phase 6 end-to-end smoke test: data build -> M0/M1/M2/M3 run -> eval -> judge -> tomography -> analyze."""

from __future__ import annotations

from pathlib import Path

import pytest
from minorsafe.runner.run import run_smoke_pipeline

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def test_full_cpu_smoke_pipeline(tmp_path: Path):
    summary = run_smoke_pipeline(work_dir=tmp_path, max_examples=6)
    assert (tmp_path / "data" / "processed" / "examples.parquet").exists()
    assert (tmp_path / "data" / "processed" / "pairs.parquet").exists()
    assert (tmp_path / "results" / "data_card.md").exists()
    assert (tmp_path / "results" / "cost_table.csv").exists()
    assert (tmp_path / "results" / "REPORT.md").exists()
    assert (tmp_path / "results" / "human_eval_150.csv").exists()
    assert len(summary) >= 6
