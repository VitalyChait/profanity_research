"""Phase 1 + remote ops unit tests: GPU detection, exclusivity, locks, secret safety, preflight, and smart logger."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from minorsafe.hardware import (
    GPUInfo,
    acquire_run_lock,
    check_credential_safety,
    check_server_exclusivity,
    choose_profile,
    parse_nvidia_compute_apps,
    parse_nvidia_smi,
    parse_nvidia_smi_status,
    parse_who_users,
    resolve_profile_name,
    run_preflight_checks,
)
from minorsafe.telemetry.logger import SmartRunLogger
from minorsafe.utils import write_json

pytestmark = pytest.mark.unit


def test_parse_nvidia_smi_output():
    sample = "0, NVIDIA A100-SXM4-80GB, 81920\n1, NVIDIA A100-SXM4-80GB, 81920\nbad_line_ignored\n"
    gpus = parse_nvidia_smi(sample)
    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].name == "NVIDIA A100-SXM4-80GB"
    assert gpus[0].memory_total_mb == 81920


def test_choose_profile_across_vram_tiers():
    assert choose_profile([]) == "cpu_debug"
    assert choose_profile([GPUInfo(0, "GTX 1650", 4096)]) == "cpu_debug"
    assert choose_profile([GPUInfo(0, "RTX 4090", 24564)]) == "gpu_24gb"
    assert choose_profile([GPUInfo(0, "NVIDIA L40S", 46068)]) == "gpu_48gb"
    assert choose_profile([GPUInfo(0, "NVIDIA H100", 81559)]) == "gpu_80gb"
    assert (
        choose_profile([GPUInfo(0, "NVIDIA A100", 81920), GPUInfo(1, "NVIDIA A100", 81920)])
        == "multi_gpu_80gb"
    )


def test_resolve_profile_name_explicit():
    assert resolve_profile_name("cpu_debug") == "cpu_debug"
    assert resolve_profile_name("gpu_24gb") == "gpu_24gb"


def test_parse_nvidia_status_compute_apps_and_who():
    gpu_out = "0, NVIDIA A100-SXM4-80GB, 81920, 12, 0, 34\n1, NVIDIA A100-SXM4-80GB, 81920, 24000, 95, 68\n"
    statuses = parse_nvidia_smi_status(gpu_out)
    assert len(statuses) == 2
    assert statuses[0].memory_used_mb == 12
    assert statuses[1].utilization_pct == 95
    assert statuses[1].temperature_c == 68

    apps_out = "111, python3, 12000\n222, vllm, 24000\n"
    apps = parse_nvidia_compute_apps(apps_out, ignore_pids={111})
    assert len(apps) == 1
    assert apps[0]["pid"] == 222
    assert apps[0]["used_memory_mb"] == 24000

    who_out = "root     pts/0        2026-11-19 10:00 (10.0.0.5)\nalice    pts/1        2026-11-19 10:05 (10.0.0.8)\n"
    assert parse_who_users(who_out, current_user="root") == ["alice"]
    assert parse_who_users(who_out, current_user="alice") == ["root"]


def test_check_server_exclusivity_cases(tmp_path: Path):
    lock_file = tmp_path / ".minorsafe.lock"
    idle_gpu = "0, NVIDIA A100, 81920, 4, 0, 32\n"
    busy_gpu = "0, NVIDIA A100, 81920, 16000, 85, 65\n"
    who_clean = "root pts/0 2026-11-19 10:00\n"
    who_other = "root pts/0 2026-11-19 10:00\nbob pts/1 2026-11-19 10:02\n"

    # 1. Idle GPU, no compute apps, no other users, no lock -> exclusive
    rep_ok = check_server_exclusivity(
        current_user="root",
        lock_path=lock_file,
        _nvidia_gpu_out=idle_gpu,
        _nvidia_apps_out="",
        _who_out=who_clean,
    )
    assert rep_ok.exclusive is True
    assert rep_ok.reasons == []

    # 2. Another user logged in
    rep_user = check_server_exclusivity(
        current_user="root",
        lock_path=lock_file,
        _nvidia_gpu_out=idle_gpu,
        _nvidia_apps_out="",
        _who_out=who_other,
    )
    assert rep_user.exclusive is False
    assert rep_user.other_logged_in_users == ["bob"]

    # 3. Foreign GPU compute process
    rep_proc = check_server_exclusivity(
        current_user="root",
        lock_path=lock_file,
        _nvidia_gpu_out=idle_gpu,
        _nvidia_apps_out="999999, python, 8192\n",
        _who_out=who_clean,
    )
    assert rep_proc.exclusive is False
    assert len(rep_proc.gpu_processes) == 1

    # 4. High GPU memory / utilization
    rep_busy = check_server_exclusivity(
        current_user="root",
        lock_path=lock_file,
        _nvidia_gpu_out=busy_gpu,
        _nvidia_apps_out="",
        _who_out=who_clean,
    )
    assert rep_busy.exclusive is False
    assert len(rep_busy.gpu_busy_devices) == 1

    # 5. Active lock held by another live PID
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        write_json(lock_file, {"pid": proc.pid, "phase": "main"})
        rep_lock = check_server_exclusivity(
            current_user="root",
            lock_path=lock_file,
            _nvidia_gpu_out=idle_gpu,
            _nvidia_apps_out="",
            _who_out=who_clean,
        )
        assert rep_lock.exclusive is False
        assert rep_lock.active_lock is not None
        assert rep_lock.active_lock["pid"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_acquire_run_lock_lifecycle_and_conflict(tmp_path: Path):
    lock_file = tmp_path / ".minorsafe.lock"
    with acquire_run_lock(lock_path=lock_file, phase="unit_test") as path:
        assert path.exists()
    assert not lock_file.exists()

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        write_json(lock_file, {"pid": proc.pid, "phase": "other_run"})
        with pytest.raises(RuntimeError, match="Another MinorSafe process holds"):
            with acquire_run_lock(lock_path=lock_file, phase="blocked"):
                pass
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_check_credential_safety_and_permissions(tmp_path: Path):
    # Real repo must ignore .remote.env and never track it
    real_cred = check_credential_safety()
    assert real_cred["ok"] is True
    assert real_cred["gitignored"] is True
    assert real_cred["tracked"] is False

    # Temp dir: auto-tighten loose permissions on .remote.env to 0600
    (tmp_path / ".gitignore").write_text(".remote.env\n", encoding="utf-8")
    env_p = tmp_path / ".remote.env"
    env_p.write_text('REMOTE_HOST="10.0.0.1"\n', encoding="utf-8")
    env_p.chmod(0o644)

    res = check_credential_safety(repo_root=tmp_path)
    assert res["ok"] is True
    assert res["env_exists"] is True
    assert res["mode_octal"] == "0o600"
    assert res["warnings"], "Expected permission-tightening warning"


def test_run_preflight_checks_cpu_debug():
    rep = run_preflight_checks("cpu_debug", require_exclusive=False)
    assert rep["ok"] is True
    assert rep["profile"] == "cpu_debug"
    assert "torch" in rep["packages"]
    assert rep["disk"]["repo_free_gb"] > 0
    assert rep["credential_safety"]["ok"] is True
    assert "exclusive" in rep["exclusivity"]


def test_smart_run_logger_writes_events_and_live_status(tmp_path: Path):
    slogger = SmartRunLogger(experiment="unit_exp", profile="cpu_debug", total_runs=2, log_dir=tmp_path)
    slogger.start_experiment(total_examples_per_run=4, seeds=[0])
    slogger.start_run(
        run_idx=1,
        run_id="m0a__tiny__fp32__abc",
        method="m0a_raw",
        model="tiny_llama",
        quant="fp32",
        setup_wall_s=0.1,
        setup_tokens=0,
    )
    slogger.log_batch(
        run_idx=1,
        run_id="m0a__tiny__fp32__abc",
        seed=0,
        batch_done=2,
        total_todo=4,
        batch_e2e_ms=12.5,
        decode_tok_per_s=40.0,
        peak_vram_mb=float("nan"),
        energy_j=float("nan"),
    )
    slogger.finish_run(run_idx=1, run_id="m0a__tiny__fp32__abc", n_rows=4, success_mean=0.75)
    slogger.finish_experiment(completed_runs=1)

    events = [json.loads(ln) for ln in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == [
        "experiment_start",
        "run_start",
        "batch_done",
        "run_done",
        "experiment_done",
    ]
    assert events[2]["peak_vram_mb"] is None
    live = json.loads((tmp_path / "live_status.json").read_text(encoding="utf-8"))
    assert live["state"] == "completed"
    assert live["experiment"] == "unit_exp"
