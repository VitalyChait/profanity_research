"""Entry-point consistency: every command in Makefile / run_phase.sh / RUNBOOK must parse against the CLI.

Guards against the drift where the Makefile, scripts, docs, and Typer CLI disagreed on flags, subcommands,
and phase names (e.g. `make smoke --hardware`, `data build`, `run_phase.sh rampup`).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess

import pytest
from typer.main import get_command

from minorsafe.cli import app
from minorsafe.utils import REPO_ROOT

pytestmark = pytest.mark.unit

RUN_PHASE = REPO_ROOT / "scripts" / "run_phase.sh"
PHASES = [
    "smoke",
    "pilot",
    "data",
    "rampup",
    "guards",
    "main",
    "recomposition",
    "subculture",
    "tomography",
    "quant",
    "multiturn",
    "analyze",
]


def parse_cli(argv: list[str]) -> tuple[str, dict]:
    """Resolve and parse a `minorsafe ...` argv without invoking the command. Returns (path, params)."""
    cmd = get_command(app)
    name, args, parent, path = "minorsafe", list(argv), None, []
    while True:
        ctx = cmd.make_context(name, list(args), parent=parent)
        # Typer 0.28 vendors Click under typer._click, so check resolve_command instead of click.Group.
        if hasattr(cmd, "resolve_command"):
            rest = [*getattr(ctx, "_protected_args", []), *ctx.args]
            assert rest, f"missing subcommand in {argv}"
            sub_name, sub_cmd, sub_args = cmd.resolve_command(ctx, rest)
            assert sub_cmd is not None, f"unknown subcommand {rest[0]!r} in {argv}"
            path.append(sub_name)
            cmd, name, args, parent = sub_cmd, sub_name, sub_args, ctx
        else:
            assert not ctx.args, f"unexpected extra args {ctx.args} in {argv}"
            return " ".join(path), ctx.params


def _minorsafe_argv(line: str) -> list[str] | None:
    """Extract argv after `minorsafe` or `-m minorsafe.cli` from a shell line; None if absent."""
    line = line.split("#", 1)[0].strip().rstrip("\\").strip()
    if "minorsafe" not in line:
        return None
    toks = shlex.split(line)
    for i, t in enumerate(toks):
        if t == "minorsafe" or t.endswith("/minorsafe"):
            return toks[i + 1 :]
        if t == "minorsafe.cli" and i > 0 and toks[i - 1] == "-m":
            return toks[i + 1 :]
    return None


def test_makefile_minorsafe_commands_parse():
    lines = (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    cmds = [a for a in (_minorsafe_argv(ln.replace("$(UV) run ", "")) for ln in lines) if a is not None]
    assert cmds, "no minorsafe commands found in Makefile"
    for argv in cmds:
        parse_cli(argv)


def test_makefile_phases_exist_in_run_phase():
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    phases = re.findall(r"scripts/run_phase\.sh\s+(\S+)", text)
    assert phases
    for phase in phases:
        res = subprocess.run(
            ["bash", str(RUN_PHASE), phase, "gpu_80gb"],
            env={**os.environ, "DRY_RUN": "1", "PYTHON": "python"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"run_phase.sh rejects Makefile phase {phase!r}: {res.stderr}"


@pytest.mark.parametrize("phase", PHASES)
def test_run_phase_commands_parse(phase: str):
    res = subprocess.run(
        ["bash", str(RUN_PHASE), phase, "gpu_80gb"],
        env={**os.environ, "DRY_RUN": "1", "PYTHON": "python"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    argv = _minorsafe_argv(res.stdout.strip())
    assert argv is not None, res.stdout
    path, params = parse_cli(argv)
    if path == "run":
        exp = params["experiment"]
        assert (REPO_ROOT / "configs" / "experiments" / f"{exp}.yaml").exists()
        split = re.search(
            r"^split:\s*(\w+)", (REPO_ROOT / "configs" / "experiments" / f"{exp}.yaml").read_text(), re.M
        )
        assert params["final"] == (split is not None and split.group(1) == "test")


def test_run_phase_rejects_unknown_phase():
    res = subprocess.run(
        ["bash", str(RUN_PHASE), "bogus"],
        env={**os.environ, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0


def _doc_commands(path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    out = []
    for block in blocks:
        for ln in block.splitlines():
            ln = re.sub(r"^\s*DRY_RUN=1\s+", "", ln)
            argv = _minorsafe_argv(ln)
            if argv is not None:
                out.append(argv)
    return out


@pytest.mark.parametrize(
    "doc", ["docs/RUNBOOK.md", ".agents/skills/remote-gpu-run/SKILL.md"], ids=["runbook", "gpu_skill"]
)
def test_doc_commands_parse(doc: str):
    path = REPO_ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} not present")
    cmds = _doc_commands(path)
    assert cmds, f"no minorsafe commands found in {doc}"
    for argv in cmds:
        parse_cli(argv)
    # run_phase.sh invocations in docs must use valid phases
    for phase in re.findall(r"scripts/run_phase\.sh\s+(\w+)", path.read_text(encoding="utf-8")):
        assert phase in PHASES or phase in {"main_grid", "ramp_up", "subculture_audit", "quant_grid"}, phase
