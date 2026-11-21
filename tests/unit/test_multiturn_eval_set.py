"""claude_issues_found.md M5: the ``multiturn`` experiment must evaluate multi-turn dialogues."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import minorsafe.generation as generation
from minorsafe.config import load_experiment
from minorsafe.data.build import build_dataset
from minorsafe.data.multiturn import build_multiturn_drift_dialogues
from minorsafe.runner.run import check_shard_dataset, run_experiment
from minorsafe.testing import DummyCausalLM, DummyTokenizer
from minorsafe.utils import read_json, write_json

pytestmark = pytest.mark.unit


def test_dialogues_are_distinct_multi_turn_conversations():
    dialogues = build_multiturn_drift_dialogues()
    assert len({d.id for d in dialogues}) == len(dialogues) >= 2
    for d in dialogues:
        assert len(d.turns) >= 5 and d.turns[-1].role == "user"
        assert {t.role for t in d.turns} == {"user", "assistant"}
        assert "(#" not in d.turns[-1].content  # no numbered copies padding n
    # The cleaning-products dialogue has no profanity; its first (reported) category says so.
    assert [d.categories[0] for d in dialogues] == ["profanity", "unsafe_other"]


def test_multiturn_experiment_uses_the_dialogues():
    assert load_experiment("multiturn").eval_set == "multiturn_drift"
    assert load_experiment("main_grid").eval_set == "processed"


def test_run_answers_the_final_turn_with_the_whole_conversation(tmp_path: Path, monkeypatch):
    build_dataset(data_dir=tmp_path / "data", results_dir=tmp_path / "results", offline=True)
    seen: list[list[dict]] = []
    real = generation.format_chat
    monkeypatch.setattr(
        generation, "format_chat", lambda tok, msgs, spec=None: seen.append(msgs) or real(tok, msgs, spec)
    )

    run_dirs = run_experiment(
        "smoke",
        profile_name="cpu_debug",
        overrides=[
            "eval_set=multiturn_drift",
            "runs=[{method: m0a_raw, generators: [S]}]",
            "capability_eval=false",
            "warmup_queries=0",
        ],
        runs_root=tmp_path / "runs",
        data_dir=tmp_path / "data",
        preloaded_generator=(DummyCausalLM(hidden_size=16, num_layers=3), DummyTokenizer()),
    )
    dialogues = {d.id: d for d in build_multiturn_drift_dialogues()}
    (rd,) = run_dirs
    df = pd.read_parquet(rd / "responses.parquet")
    assert set(df["example_id"]) == set(dialogues)
    for _, row in df.iterrows():
        assert row["prompt"] == dialogues[row["example_id"]].last_user
    # The generator saw every turn of each dialogue, not just the last question.
    assert sorted(len(m) for m in seen) == sorted(len(d.turns) for d in dialogues.values())
    manifest = read_json(rd / "manifest.json")
    assert manifest["eval_set"] == "multiturn_drift" and manifest["eval_digest"]


def test_shard_answered_on_other_dialogues_is_not_resumed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MINORSAFE_ACCEPT_STALE_DATA", raising=False)
    run_dir = tmp_path / "run"
    write_json(run_dir / "manifest.json", {"data_digest": "d", "eval_digest": "e1"})
    check_shard_dataset(run_dir, 4, "d", strict=True, eval_digest="e1")
    with pytest.raises(RuntimeError, match="evaluation set e1"):
        check_shard_dataset(run_dir, 4, "d", strict=True, eval_digest="e2")
    # Shards of ordinary experiments (no eval digest on either side) are unaffected.
    write_json(run_dir / "manifest.json", {"data_digest": "d"})
    check_shard_dataset(run_dir, 4, "d", strict=True)
