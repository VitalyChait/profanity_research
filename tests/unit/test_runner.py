"""Phase 6 unit tests: experiment grid expansion, test-split --final guardrail, and atomic Parquet resume."""

from __future__ import annotations

from pathlib import Path

import pytest
from minorsafe.runner.grid import expand_experiment_grid, verify_split_guardrail
from minorsafe.runner.store import RunStore

from minorsafe.config import load_experiment, load_hardware, load_models

pytestmark = pytest.mark.unit


def test_grid_expansion_and_sweeps():
    models = load_models()
    hw = load_hardware("gpu_80gb")
    exp = load_experiment("ramp_up")
    specs = expand_experiment_grid(exp, hw, models)
    # ramp_up has 4 run groups x M/L generators x 5 n_calib values
    assert len(specs) >= 20
    n_calibs = {s.method_params.get("n_calib") for s in specs}
    assert n_calibs == {25, 50, 100, 500, 2000}


def test_test_split_guardrail_requires_final_flag():
    exp_test = load_experiment("main_grid")
    assert exp_test.split == "test"
    with pytest.raises(PermissionError, match="--final"):
        verify_split_guardrail(exp_test, final=False)
    # Allowed when final=True
    verify_split_guardrail(exp_test, final=True)


def test_run_store_atomic_write_and_crash_resume(tmp_path: Path):
    run_dir = tmp_path / "runs" / "smoke" / "m0a_raw_tiny"
    store = RunStore(run_dir)

    batch1 = [
        {"example_id": "ex1", "seed": 0, "text": "ans1", "tok_total": 15, "e2e_ms": 12.0},
        {"example_id": "ex2", "seed": 0, "text": "ans2", "tok_total": 18, "e2e_ms": 14.0},
    ]
    store.append_responses(batch1)
    assert store.completed_keys() == {("ex1", 0), ("ex2", 0)}

    # Simulate crash recovery: re-appending ex2 + new ex3 should deduplicate (ex2, 0)
    batch2 = [
        {"example_id": "ex2", "seed": 0, "text": "ans2_dup", "tok_total": 18, "e2e_ms": 14.0},
        {"example_id": "ex3", "seed": 0, "text": "ans3", "tok_total": 20, "e2e_ms": 15.0},
    ]
    store.append_responses(batch2)
    df = store.load_responses()
    assert len(df) == 3
    assert set(df["example_id"]) == {"ex1", "ex2", "ex3"}
