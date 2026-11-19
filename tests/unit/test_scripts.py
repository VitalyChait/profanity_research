"""Phase 10 unit tests: shell script syntax validation, model downloader dry-run, and remote orchestrator."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from minorsafe.utils import REPO_ROOT

pytestmark = pytest.mark.unit


def test_shell_scripts_syntax_valid():
    scripts = [
        REPO_ROOT / "scripts" / "setup_server.sh",
        REPO_ROOT / "scripts" / "run_phase.sh",
        REPO_ROOT / "scripts" / "activate.sh",
        REPO_ROOT / "scripts" / "remote.sh",
        REPO_ROOT / "scripts" / "install_git_hooks.sh",
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


def test_remote_script_dry_run_subcommands(tmp_path: Path):
    env_file = tmp_path / ".remote.env"
    env_file.write_text(
        "\n".join(
            [
                'REMOTE_HOST="10.0.0.1"',
                'REMOTE_PORT="22"',
                'REMOTE_USER="root"',
                'REMOTE_DIR="/root/profanity_research"',
                'REMOTE_PROFILE="gpu_80gb"',
                'REQUIRE_EXCLUSIVE="1"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    script = REPO_ROOT / "scripts" / "remote.sh"
    base_env = {**os.environ, "DRY_RUN": "1", "REMOTE_ENV_FILE": str(env_file)}

    cases: list[tuple[list[str], list[str]]] = [
        (["check"], ["ssh", "root@10.0.0.1"]),
        (
            ["sync"],
            [
                "rsync",
                "--exclude=.remote.env",
                "--exclude=.secrets/",
                "root@10.0.0.1:/root/profanity_research/",
            ],
        ),
        (["setup"], ["rsync", "ssh", "setup_server.sh", "preflight"]),
        (["preflight"], ["ssh", "preflight", "gpu_80gb"]),
        (["run", "main", "gpu_80gb"], ["rsync", "ssh", "run_phase.sh", "main", "gpu_80gb"]),
        (["run", "main", "gpu_80gb", "--detach"], ["rsync", "ssh", "tmux", "minorsafe_main"]),
        (["status"], ["ssh", "minorsafe.cli status", "nvidia-smi", "latest.log"]),
        (["logs"], ["ssh", "tail", "latest.log"]),
        (["pull"], ["rsync", "--exclude=*.safetensors", "root@10.0.0.1:/root/profanity_research/results/"]),
    ]

    for argv, expected_substrings in cases:
        res = subprocess.run(
            ["bash", str(script), *argv],
            env=base_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"remote.sh {argv} failed: {res.stderr}\n{res.stdout}"
        unescaped = res.stdout.replace("\\", "")
        for sub in expected_substrings:
            assert sub in unescaped, f"Expected {sub!r} in output for {argv}: {unescaped}"
