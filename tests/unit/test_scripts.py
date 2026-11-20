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
        REPO_ROOT / "setup.sh",
        REPO_ROOT / "scripts" / "setup_server.sh",
        REPO_ROOT / "scripts" / "env_sandbox.sh",
        REPO_ROOT / "scripts" / "snapshot_versions.sh",
        REPO_ROOT / "scripts" / "run_phase.sh",
        REPO_ROOT / "scripts" / "run_msba.sh",
        REPO_ROOT / "scripts" / "run_all_tests.sh",
        REPO_ROOT / "scripts" / "run_autopilot.sh",
        REPO_ROOT / "scripts" / "activate.sh",
        REPO_ROOT / "scripts" / "remote.sh",
        REPO_ROOT / "scripts" / "restart_remote.sh",
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
                'REMOTE_PASSWORD="test_secret_password"',
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
        (["setup-ssh-key"], ["authorized_keys", "root@10.0.0.1"]),
        (
            ["sync"],
            [
                "rsync",
                "--exclude=/data/",
                "--exclude=.remote.env",
                "--exclude=.secrets/",
                "root@10.0.0.1:/root/profanity_research/",
            ],
        ),
        (["setup"], ["rsync", "ssh", "setup_server.sh", "preflight"]),
        (["preflight"], ["ssh", "preflight", "gpu_80gb"]),
        (["check-gates", "gpu_lab1"], ["rsync", "ssh", "download_models.py", "--check-gates", "gpu_lab1"]),
        (["run", "main", "gpu_80gb"], ["rsync", "ssh", "run_phase.sh", "main", "gpu_80gb"]),
        (["run", "main", "gpu_80gb", "--detach"], ["rsync", "ssh", "tmux", "minorsafe_main"]),
        (["stop"], ["ssh", "tmux kill-session", ".minorsafe.lock"]),
        (["reset"], ["ssh", "tmux kill-session", "data/processed", "results/runs"]),
        (["status"], ["ssh", "minorsafe.cli status", "nvidia-smi", "latest.log"]),
        (["info"], ["ssh", "nvidia-smi", "df -h"]),
        (["test-all", "gpu_lab1"], ["rsync", "ssh", "run_all_tests.sh", "CUDA_VISIBLE_DEVICES=0"]),
        (
            ["autopilot", "gpu_lab1"],
            [
                "rsync",
                "ssh",
                "download_models.py",
                "--check-gates",
                "run_autopilot.sh",
                "CUDA_VISIBLE_DEVICES=0",
            ],
        ),
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
        assert "test_secret_password" not in res.stdout
        assert "test_secret_password" not in res.stderr
        unescaped = res.stdout.replace("\\", "")
        if argv == ["sync"]:
            assert "--exclude=data/" not in unescaped
        for sub in expected_substrings:
            assert sub in unescaped, f"Expected {sub!r} in output for {argv}: {unescaped}"

    # Non-root REMOTE_USER derives /home/<user>/profanity_research when REMOTE_DIR is omitted.
    std_env = tmp_path / ".remote_std.env"
    std_env.write_text('REMOTE_HOST="10.0.0.1"\nREMOTE_USER="std-ai"\n', encoding="utf-8")
    std_env.chmod(0o600)
    res_std = subprocess.run(
        ["bash", str(script), "sync"],
        env={**os.environ, "DRY_RUN": "1", "REMOTE_ENV_FILE": str(std_env)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_std.returncode == 0, res_std.stderr
    assert "std-ai@10.0.0.1:/home/std-ai/profanity_research/" in res_std.stdout


def test_root_setup_script_dry_run(tmp_path: Path):
    env_file = tmp_path / ".remote.env"
    pw = "test_root_pw_value"
    env_file.write_text(
        "\n".join(
            [
                'REMOTE_HOST="10.0.0.1"',
                'REMOTE_USER="root"',
                f'REMOTE_PASSWORD="{pw}"',
                'REMOTE_DIR="/root/profanity_research"',
                'REMOTE_PROFILE="gpu_80gb"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    script = REPO_ROOT / "setup.sh"
    base_env = {**os.environ, "DRY_RUN": "1", "REMOTE_ENV_FILE": str(env_file)}

    res_local = subprocess.run(
        ["bash", str(script), "--local"], env=base_env, capture_output=True, text=True, check=False
    )
    assert res_local.returncode == 0, res_local.stderr + res_local.stdout
    assert pw not in res_local.stdout and pw not in res_local.stderr

    res_remote = subprocess.run(
        ["bash", str(script), "--remote"], env=base_env, capture_output=True, text=True, check=False
    )
    assert res_remote.returncode == 0, res_remote.stderr + res_remote.stdout
    for expected in ("authorized_keys", "root@10.0.0.1", "setup_server.sh"):
        assert expected in res_remote.stdout
    assert pw not in res_remote.stdout and pw not in res_remote.stderr

    res_check = subprocess.run(
        ["bash", str(script), "--check"], env=base_env, capture_output=True, text=True, check=False
    )
    assert res_check.returncode == 0, res_check.stderr + res_check.stdout
    assert "preflight" in res_check.stdout
    assert pw not in res_check.stdout and pw not in res_check.stderr


def test_snapshot_versions_creates_snapshot_file(tmp_path: Path):
    script = REPO_ROOT / "scripts" / "snapshot_versions.sh"
    env = {**os.environ, "DRY_RUN": "0"}
    res = subprocess.run(
        ["bash", str(script), str(tmp_path), "unit_test"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    latest = tmp_path / "versions_snapshot_latest.txt"
    assert latest.exists()
    snaps = list(tmp_path.glob("versions_snapshot_unit_test_*.txt"))
    assert len(snaps) == 1
    body = snaps[0].read_text(encoding="utf-8")
    assert "python3:" in body
    assert "minorsafe==" in body


def test_run_autopilot_dry_run():
    script = REPO_ROOT / "scripts" / "run_autopilot.sh"
    res = subprocess.run(
        ["bash", str(script), "gpu_lab1", "--skip-quant"],
        env={**os.environ, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    for expected in (
        "scripts/run_all_tests.sh gpu_lab1",
        "scripts/download_models.py --profile gpu_lab1 --check-gates",
        "scripts/run_phase.sh data gpu_lab1",
        "scripts/run_phase.sh pilot gpu_lab1",
        "scripts/run_msba.sh gpu_lab1 --skip-quant",
    ):
        assert expected in res.stdout, f"Missing {expected!r} in:\n{res.stdout}"


def test_restart_remote_dry_run(tmp_path: Path):
    env_file = tmp_path / ".remote.env"
    env_file.write_text('REMOTE_HOST="10.0.0.1"\nREMOTE_USER="vitaly"\n', encoding="utf-8")
    env_file.chmod(0o600)

    script = REPO_ROOT / "scripts" / "restart_remote.sh"
    res = subprocess.run(
        ["bash", str(script), "gpu_lab1"],
        env={**os.environ, "DRY_RUN": "1", "REMOTE_ENV_FILE": str(env_file)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    unescaped = res.stdout.replace("\\", "")
    assert "tmux kill-session" in unescaped
    assert "data/processed" in unescaped
    assert "run_autopilot.sh 'gpu_lab1' --with-tomography" in unescaped
