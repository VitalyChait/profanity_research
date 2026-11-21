"""claude_issues_found.md M1: a processed dataset built by older code/config must never be silently reused."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minorsafe.data import build as build_mod
from minorsafe.data.build import build_dataset, check_dataset, ensure_dataset
from minorsafe.data.provenance import BUILD_META, data_build_modules, normalized_source
from minorsafe.runner.run import check_shard_dataset
from minorsafe.utils import REPO_ROOT, read_json, write_json

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def built_data(tmp_path_factory) -> Path:
    """One offline build, copied per test (the build itself takes a couple of seconds)."""
    root = tmp_path_factory.mktemp("prov")
    build_dataset(data_dir=root / "data", results_dir=root / "results", offline=True)
    return root / "data"


@pytest.fixture
def data_dir(built_data: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "data"
    shutil.copytree(built_data, dst)
    return dst


def _edit_meta(data_dir: Path, fn) -> None:
    path = data_dir / "processed" / BUILD_META
    meta = read_json(path)
    fn(meta)
    write_json(path, meta)


def test_code_hash_ignores_comments_formatting_and_docstrings():
    a = 'def f(x):\n    """Doc."""\n    return x + 1  # note\n'
    b = "def f(x):\n\n    return (x + 1)\n"
    assert normalized_source(a) == normalized_source(b)
    assert normalized_source(a) != normalized_source("def f(x):\n    return x + 2\n")


def test_build_code_follows_imports_from_build():
    mods = set(data_build_modules())
    assert {"minorsafe.data.sources", "minorsafe.data.synth", "minorsafe.data.splits"} <= mods
    # Not used by the build (M5) / bookkeeping only: editing them must not invalidate the data.
    assert "minorsafe.data.multiturn" not in mods
    assert "minorsafe.data.provenance" not in mods


def test_build_records_provenance_and_is_fresh(data_dir: Path):
    meta = read_json(data_dir / "processed" / BUILD_META)
    assert meta["options"]["offline"] is True
    assert meta["data_digest"] and meta["fingerprint"]
    assert "minorsafe.data.sources" in meta["code_modules"]
    status = check_dataset(data_dir)
    assert status.fresh, status.describe()
    assert status.data_digest == meta["data_digest"]


def test_code_spec_and_file_drift_make_the_dataset_stale(data_dir: Path):
    # The server case: data built before the neologism loader was added to sources.py.
    def old_sources(meta):
        meta["code_modules"]["minorsafe.data.sources"] = "0" * 16
        meta["components"]["code"] = "0" * 16

    _edit_meta(data_dir, old_sources)
    status = check_dataset(data_dir)
    assert status.state == "stale"
    assert any("minorsafe.data.sources" in r for r in status.reasons), status.reasons

    shutil.rmtree(data_dir)
    status_missing = check_dataset(data_dir)
    assert status_missing.state == "missing"


def test_spec_override_and_parquet_edits_are_detected(data_dir: Path):
    _edit_meta(data_dir, lambda m: m["options"].update(overrides=["seed=14"]))
    assert any("data spec" in r for r in check_dataset(data_dir).reasons)

    _edit_meta(data_dir, lambda m: m["options"].update(overrides=[]))
    assert check_dataset(data_dir).fresh
    with open(data_dir / "processed" / "pairs.parquet", "ab") as fh:
        fh.write(b"\0")
    assert any("changed after" in r for r in check_dataset(data_dir).reasons)


def test_dataset_without_build_record_is_unrecorded(data_dir: Path):
    (data_dir / "processed" / BUILD_META).unlink()
    status = check_dataset(data_dir)
    assert status.state == "unrecorded" and not status.fresh


def test_crashed_rebuild_cannot_pass_for_the_previous_build(data_dir: Path, tmp_path: Path, monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("teacher OOM")

    monkeypatch.setattr(build_mod, "synthesize_contrastive_pairs", boom)
    with pytest.raises(RuntimeError, match="teacher OOM"):
        build_dataset(data_dir=data_dir, results_dir=tmp_path / "results", offline=True)
    assert check_dataset(data_dir).state == "unrecorded"


def test_strict_runs_refuse_missing_offline_or_unrecorded_data(data_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MINORSAFE_ACCEPT_STALE_DATA", raising=False)
    empty = tmp_path / "empty"
    with pytest.raises(RuntimeError, match="data build"):
        ensure_dataset(empty, strict=True, results_dir=tmp_path / "r")
    assert not (empty / "processed" / "examples.parquet").exists()  # never silently built

    with pytest.raises(RuntimeError, match="built offline"):
        ensure_dataset(data_dir, strict=True)
    monkeypatch.setenv("MINORSAFE_ACCEPT_STALE_DATA", "1")
    assert ensure_dataset(data_dir, strict=True).state == "stale"

    monkeypatch.delenv("MINORSAFE_ACCEPT_STALE_DATA")
    (data_dir / "processed" / BUILD_META).unlink()
    with pytest.raises(RuntimeError, match="unrecorded"):
        ensure_dataset(data_dir, strict=True)
    # CPU/debug runs keep working on whatever is there.
    assert ensure_dataset(data_dir, strict=False).state == "unrecorded"


def test_debug_runs_still_build_a_missing_dataset_offline(tmp_path: Path):
    status = ensure_dataset(tmp_path / "data", strict=False, results_dir=tmp_path / "results")
    assert status.fresh and status.meta["options"]["offline"] is True


def test_resuming_a_shard_from_another_dataset_is_refused(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MINORSAFE_ACCEPT_STALE_DATA", raising=False)
    run_dir = tmp_path / "run"
    write_json(run_dir / "manifest.json", {"run_id": "r", "data_digest": "aaaa"})
    check_shard_dataset(run_dir, 5, "aaaa", strict=True)  # same dataset
    check_shard_dataset(run_dir, 0, "bbbb", strict=True)  # nothing stored yet
    with pytest.raises(RuntimeError, match="mix two datasets"):
        check_shard_dataset(run_dir, 5, "bbbb", strict=True)
    check_shard_dataset(run_dir, 5, "bbbb", strict=False)  # debug: warning only

    write_json(run_dir / "manifest.json", {"run_id": "r"})  # shard from before digests were recorded
    with pytest.raises(RuntimeError, match="no recorded digest"):
        check_shard_dataset(run_dir, 5, "bbbb", strict=True)


def test_run_manifest_records_the_dataset(data_dir: Path, tmp_path: Path):
    from minorsafe.runner.run import run_experiment
    from minorsafe.testing import DummyCausalLM, DummyTokenizer

    run_dirs = run_experiment(
        "smoke",
        profile_name="cpu_debug",
        overrides=["max_examples=2"],
        runs_root=tmp_path / "runs",
        data_dir=data_dir,
        preloaded_generator=(DummyCausalLM(hidden_size=16, num_layers=3), DummyTokenizer()),
    )
    digest = check_dataset(data_dir).data_digest
    assert run_dirs
    for rd in run_dirs:
        manifest = read_json(rd / "manifest.json")
        assert manifest["data_digest"] == digest
        assert manifest["data_fingerprint"] == read_json(data_dir / "processed" / BUILD_META)["fingerprint"]


def test_cli_data_status_exit_codes(data_dir: Path, tmp_path: Path, monkeypatch):
    import minorsafe.cli as cli

    runner = CliRunner()
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "nothing")
    res = runner.invoke(cli.app, ["data", "status", "--offline"])
    assert res.exit_code == cli.DATA_REBUILD_EXIT and "missing" in res.output

    monkeypatch.setattr(cli, "DATA_DIR", data_dir)
    res = runner.invoke(cli.app, ["data", "status", "--offline", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["state"] == "fresh"
    # The autopilot asks for an online build for its GPU profile: an offline build is not good enough.
    res = runner.invoke(cli.app, ["data", "status", "--hardware", "cpu_debug"])
    assert res.exit_code == cli.DATA_REBUILD_EXIT and "built offline" in res.output


# --------------------------------------------------------------------------- autopilot data stage
def _autopilot_sandbox(tmp_path: Path, status_rc: int) -> Path:
    """Copy of run_autopilot.sh with stub python / phase scripts that log their calls."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "run_autopilot.sh", root / "scripts" / "run_autopilot.sh")
    calls = root / "calls.log"
    (root / "scripts" / "run_phase.sh").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "run_phase $*" >> "{calls}"\n'
        'if [[ "$1" == data ]]; then mkdir -p data/processed && '
        "touch data/processed/examples.parquet data/processed/pairs.parquet; fi\n"
    )
    (root / "scripts" / "run_msba.sh").write_text(
        f'#!/usr/bin/env bash\necho "run_msba accept=${{MINORSAFE_ACCEPT_STALE_DATA:-unset}}" >> "{calls}"\n'
    )
    py = root / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "python $*" >> "{calls}"\n'
        'if [[ "$*" == *"data status"* ]]; then echo "stale: data-building code changed"; '
        f"exit {status_rc}; fi\nexit 0\n"
    )
    py.chmod(0o755)
    return root


def _run_autopilot(root: Path, *flags: str) -> subprocess.CompletedProcess:
    env = {
        k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "PYTHON", "MINORSAFE_ACCEPT_STALE_DATA")
    }
    return subprocess.run(
        [
            "bash",
            str(root / "scripts" / "run_autopilot.sh"),
            "cpu_debug",
            "--skip-tests",
            "--skip-download",
            "--skip-pilot",
            *flags,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _with_old_results(root: Path) -> None:
    write_json(root / "results" / "runs" / "main_grid" / "abc" / "manifest.json", {"run_id": "abc"})
    (root / "data" / "processed").mkdir(parents=True)
    for name in ("examples.parquet", "pairs.parquet"):
        (root / "data" / "processed" / name).write_bytes(b"old")


def test_autopilot_stops_instead_of_mixing_old_results_with_rebuilt_data(tmp_path: Path):
    root = _autopilot_sandbox(tmp_path, status_rc=3)
    _with_old_results(root)
    res = _run_autopilot(root)
    calls = (root / "calls.log").read_text()
    assert res.returncode == 1, res.stdout + res.stderr
    assert "--rebuild-data" in res.stdout
    assert "run_phase data" not in calls and "run_msba" not in calls
    assert (root / "results" / "runs" / "main_grid" / "abc" / "manifest.json").exists()


def test_autopilot_rebuild_data_archives_old_results_then_rebuilds(tmp_path: Path):
    root = _autopilot_sandbox(tmp_path, status_rc=3)
    _with_old_results(root)
    res = _run_autopilot(root, "--rebuild-data")
    calls = (root / "calls.log").read_text()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "run_phase data cpu_debug" in calls and "run_msba accept=unset" in calls
    assert not (root / "results" / "runs").exists()
    archived = list((root / "results" / "stale_archive").glob("data_*/runs/main_grid/abc/manifest.json"))
    assert len(archived) == 1


def test_autopilot_skips_a_current_dataset_and_keep_data_skips_the_check(tmp_path: Path):
    root = _autopilot_sandbox(tmp_path, status_rc=0)
    _with_old_results(root)
    res = _run_autopilot(root)
    calls = (root / "calls.log").read_text()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "data status --hardware cpu_debug" in calls and "run_phase data" not in calls

    (root / "calls.log").unlink()
    res = _run_autopilot(root, "--keep-data")
    calls = (root / "calls.log").read_text()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "data status" not in calls and "run_phase data" not in calls
    assert "run_msba accept=1" in calls

    res = _run_autopilot(root, "--keep-data", "--rebuild-data")
    assert res.returncode == 1 and "mutually exclusive" in res.stderr
