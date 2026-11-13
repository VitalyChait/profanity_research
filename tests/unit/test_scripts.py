"""Phase 10 unit tests: shell script syntax validation and model downloader dry-run."""

from __future__ import annotations

import subprocess
import sys

import pytest

from minorsafe.utils import REPO_ROOT

pytestmark = pytest.mark.unit


def test_shell_scripts_syntax_valid():
    scripts = [
        REPO_ROOT / "scripts" / "setup_server.sh",
        REPO_ROOT / "scripts" / "run_phase.sh",
        REPO_ROOT / "scripts" / "slurm" / "run_grid.sbatch",
    ]
    for s in scripts:
        assert s.exists(), f"Missing script {s}"
        res = subprocess.run(["bash", "-n", str(s)], capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"Syntax error in {s}: {res.stderr}"


def test_download_models_dry_run():
    script = REPO_ROOT / "scripts" / "download_models.py"
    assert script.exists()
    res = subprocess.run(
        [sys.executable, str(script), "--profile", "cpu_debug", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "tiny_llama" in res.stdout

