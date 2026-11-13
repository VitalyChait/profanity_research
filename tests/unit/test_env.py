"""Phase 0 unit tests: environment imports, deterministic seeding, JSON utilities, and CLI help."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from minorsafe.utils import (
    extract_json_object,
    git_sha,
    normalize_text,
    package_versions,
    read_json,
    seed_everything,
    stable_hash,
    write_json,
)

pytestmark = pytest.mark.unit


def test_core_dependencies_importable():
    import accelerate
    import datasets
    import pandas
    import peft
    import pyarrow
    import pydantic
    import sklearn
    import textstat
    import transformers
    import trl
    import typer
    import wordfreq

    assert all(
        m is not None
        for m in (
            accelerate,
            datasets,
            pandas,
            peft,
            pyarrow,
            pydantic,
            sklearn,
            textstat,
            transformers,
            trl,
            typer,
            wordfreq,
        )
    )


def test_seed_everything_determinism():
    seed_everything(12345)
    t1 = torch.rand(8)
    n1 = np.random.rand(8)

    seed_everything(12345)
    t2 = torch.rand(8)
    n2 = np.random.rand(8)

    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)


def test_stable_hash_and_normalize_text():
    h1 = stable_hash({"a": 1, "b": [2, 3], "emoji": "💀"})
    h2 = stable_hash({"b": [2, 3], "emoji": "💀", "a": 1})
    assert h1 == h2
    assert len(h1) == 12
    assert normalize_text("  Hello,   WORLD!! 123 💀 ") == "hello world 123"


def test_json_roundtrip_and_extractor(tmp_path: Path):
    payload = {"age_appropriate": 5, "rationale": "Safe for 12-18 💀🔥"}
    p = tmp_path / "nested" / "test.json"
    write_json(p, payload)
    loaded = read_json(p)
    assert loaded == payload

    wrapped = (
        'Here is my verdict:\n```json\n{"age_appropriate": 4, "is_refusal": false, "note": "a {b} c"}\n```'
    )
    extracted = extract_json_object(wrapped)
    assert extracted is not None
    assert extracted["age_appropriate"] == 4
    assert extracted["is_refusal"] is False
    assert extract_json_object("not json at all") is None


def test_git_sha_and_package_versions():
    sha = git_sha()
    assert isinstance(sha, str) and len(sha) >= 7
    vers = package_versions(("torch", "transformers", "nonexistent_pkg_xyz"))
    assert vers["torch"] is not None
    assert vers["transformers"] is not None
    assert vers["nonexistent_pkg_xyz"] is None


def test_cli_help_lists_subcommands():
    from minorsafe.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("info", "data", "run", "eval", "judge", "tomography", "status", "analyze", "smoke"):
        assert cmd in res.stdout
