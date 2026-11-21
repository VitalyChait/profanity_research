"""claude_issues_found.md M9: the run lock must be atomic, and SLURM array tasks must not block each other."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import minorsafe.hardware as hw
from minorsafe.hardware import acquire_run_lock, check_server_exclusivity, default_lock_path

pytestmark = pytest.mark.unit

IDLE_GPU = "0, NVIDIA A100, 81920, 4, 0, 32\n"
OTHER_USER = "root pts/0 2026-11-19 10:00\nbob pts/1 2026-11-19 10:02\n"

# A MinorSafe run: wait for a shared start time, check exclusivity and take the lock like run_experiment.
_RUN = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from minorsafe.hardware import acquire_run_lock, check_server_exclusivity, default_lock_path
    root, start = Path(sys.argv[1]), float(sys.argv[2])
    time.sleep(max(0.0, start - time.time()))
    lock = default_lock_path(root)
    rep = check_server_exclusivity(current_user="root", lock_path=lock, _nvidia_gpu_out={idle!r},
                                   _nvidia_apps_out="", _who_out={who!r})
    try:
        with acquire_run_lock(lock_path=lock, phase="p"):
            print("won", rep.exclusive, flush=True)
            time.sleep(1.5)
    except RuntimeError:
        print("lost", rep.exclusive, flush=True)
    """
).format(idle=IDLE_GPU, who=OTHER_USER)


def _run_together(root: Path, envs: list[dict[str, str]]) -> list[tuple[str, bool]]:
    start = time.time() + 1.5  # imports finish first, then all processes race for the lock
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _RUN, str(root), str(start)], env=env, stdout=subprocess.PIPE, text=True
        )
        for env in envs
    ]
    out = []
    for p in procs:
        word, exclusive = p.communicate(timeout=60)[0].split()
        out.append((word, exclusive == "True"))
    return out


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    return {**env, **extra}


def test_exactly_one_of_many_simultaneous_runs_gets_the_lock(tmp_path: Path):
    results = _run_together(tmp_path, [_env()] * 8)
    assert sorted(w for w, _ in results).count("won") == 1, results
    assert not (tmp_path / ".minorsafe.lock").exists()  # released by the winner


def test_slurm_array_tasks_on_one_node_run_side_by_side(tmp_path: Path):
    # run_grid.sbatch: six tasks with their own SLURM_JOB_ID start together on a node another user is on.
    results = _run_together(tmp_path, [_env(SLURM_JOB_ID=str(500 + i)) for i in range(6)])
    assert results == [("won", True)] * 6
    assert not any((tmp_path / ".minorsafe.locks").iterdir())


def test_default_lock_path(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert default_lock_path(tmp_path) == tmp_path / ".minorsafe.lock"
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    assert default_lock_path(tmp_path) == tmp_path / ".minorsafe.locks" / "slurm_4242.lock"


def test_slurm_job_does_not_require_an_empty_node(monkeypatch, tmp_path: Path):
    kw = dict(current_user="root", _nvidia_gpu_out=IDLE_GPU, _nvidia_apps_out="", _who_out=OTHER_USER)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert check_server_exclusivity(lock_path=default_lock_path(tmp_path), **kw).exclusive is False
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    assert check_server_exclusivity(lock_path=default_lock_path(tmp_path), **kw).exclusive is True
    busy = dict(kw, _nvidia_gpu_out="0, NVIDIA A100, 81920, 16000, 85, 65\n")
    assert check_server_exclusivity(lock_path=default_lock_path(tmp_path), **busy).exclusive is False


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_lock_left_by_a_crashed_run_on_this_host_is_replaced(tmp_path: Path):
    lock = tmp_path / ".minorsafe.lock"
    lock.write_text(json.dumps({"pid": _dead_pid(), "host": hw.socket.gethostname(), "phase": "main"}))
    with acquire_run_lock(lock_path=lock, phase="guards"):
        assert json.loads(lock.read_text())["pid"] == os.getpid()
    assert not lock.exists()


def test_lock_from_another_host_is_never_assumed_dead(tmp_path: Path):
    lock = tmp_path / ".minorsafe.lock"
    lock.write_text(json.dumps({"pid": _dead_pid(), "host": "some-other-node", "phase": "main"}))
    with pytest.raises(RuntimeError, match="some-other-node.*delete the file"):
        with acquire_run_lock(lock_path=lock):
            pass
    assert hw._read_active_lock(lock)["host"] == "some-other-node"


def test_unreadable_lock_is_respected_while_young_and_replaced_when_old(tmp_path: Path):
    lock = tmp_path / ".minorsafe.lock"
    lock.write_text("")  # another process between O_EXCL create and write
    with pytest.raises(RuntimeError, match="Another MinorSafe process"):
        with acquire_run_lock(lock_path=lock):
            pass
    old = time.time() - 120
    os.utime(lock, (old, old))
    with acquire_run_lock(lock_path=lock):
        pass
    assert not lock.exists()


def test_nested_lock_in_one_process_is_refused(tmp_path: Path):
    lock = tmp_path / ".minorsafe.lock"
    with acquire_run_lock(lock_path=lock):
        with pytest.raises(RuntimeError, match="already holds"):
            with acquire_run_lock(lock_path=lock):
                pass
        assert lock.exists()  # the failed inner attempt must not release the outer lock
    assert not lock.exists()


def test_breaking_a_stale_lock_never_deletes_a_fresh_one(tmp_path: Path):
    # B judged the lock stale, but A replaced it with a live lock before B got to remove it.
    lock = tmp_path / ".minorsafe.lock"
    stale = json.dumps({"pid": _dead_pid(), "phase": "old"})
    fresh = json.dumps({"pid": os.getppid(), "phase": "A"})
    lock.write_text(fresh)
    hw._remove_stale_lock(lock, seen=stale)
    assert lock.read_text() == fresh
    assert [p.name for p in tmp_path.iterdir()] == [".minorsafe.lock"]
    lock.write_text(stale)
    hw._remove_stale_lock(lock, seen=stale)
    assert not lock.exists()


def test_release_leaves_a_lock_that_is_no_longer_ours(tmp_path: Path):
    lock = tmp_path / ".minorsafe.lock"
    with acquire_run_lock(lock_path=lock):
        lock.write_text(json.dumps({"pid": os.getppid(), "phase": "someone else"}))
    assert lock.exists()
