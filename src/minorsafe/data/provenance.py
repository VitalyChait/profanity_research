"""Provenance of ``data/processed``: which inputs built it, and whether today's code/config would match.

``build_dataset`` writes ``processed/build_meta.json`` beside the parquet files. It records a fingerprint of
everything that decides the dataset's content:

- ``spec``: the resolved data spec (``configs/data.yaml`` plus ``-o`` overrides);
- ``code``: the data-building code. That is every ``minorsafe.data`` module reachable from ``build``,
  plus the few outside modules/functions it relies on (``_EXTERNAL_MODULES``/``_EXTERNAL_FUNCTIONS``).
  Each is hashed from its AST with docstrings stripped, so comments, formatting and docs don't count;
- ``synth_prompts``: the teacher prompt templates;
- ``teacher``: offline vs online build, teacher model, backend and device;
- ``pair_filter``: the toxicity backend the contrastive-pair filter used.

Anything that reuses an existing dataset (autopilot, ``run_experiment``, tomography) compares that record
with the fingerprint the current checkout would produce. A dataset built by older code is then rebuilt
or refused, never silently reused. ``data_digest`` hashes the parquet bytes so results can record exactly
which dataset they were computed from.

Hugging Face source rows are streamed unpinned, so upstream dataset revisions are not covered.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minorsafe.config import DataSpec, load_prompts
from minorsafe.utils import git_sha, read_json, stable_hash, write_json

BUILD_META = "build_meta.json"
PROCESSED_FILES = ("examples.parquet", "pairs.parquet")

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # src/minorsafe
_BUILD_ENTRY = "minorsafe.data.build"
# Provenance bookkeeping describes a build; it does not shape the data.
_NOT_BUILD_CODE = frozenset({"minorsafe.data.provenance"})
# Code outside ``minorsafe.data`` that changes what a build produces.
_EXTERNAL_MODULES = (
    "minorsafe.eval.safety",  # refusal regex + Detoxify scoring used by the pair filter
    "minorsafe.generation",  # HF teacher decoding
)
_EXTERNAL_FUNCTIONS = (
    ("minorsafe.utils", "normalize_text"),  # split-family and dedup keys
    ("minorsafe.utils", "extract_json_object"),  # parsing teacher output
)


# --------------------------------------------------------------------------- code hashing
def _module_path(module: str) -> Path:
    parts = module.split(".")[1:]
    path = _PACKAGE_ROOT.joinpath(*parts)
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def normalized_source(source: str) -> str:
    """Canonical form of ``source``: comments, formatting and docstrings removed."""
    return ast.unparse(_strip_docstrings(ast.parse(source)))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def data_build_modules() -> list[str]:
    """``minorsafe.data`` modules reachable by import from ``minorsafe.data.build`` (incl. lazy imports)."""
    seen: set[str] = set()
    todo = [_BUILD_ENTRY]
    while todo:
        mod = todo.pop()
        if mod in seen or mod in _NOT_BUILD_CODE:
            continue
        path = _module_path(mod)
        if not path.exists():
            continue
        seen.add(mod)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "minorsafe.data":
                    todo.extend(f"minorsafe.data.{a.name}" for a in node.names)
                elif node.module.startswith("minorsafe.data."):
                    todo.append(node.module)
            elif isinstance(node, ast.Import):
                todo.extend(a.name for a in node.names if a.name.startswith("minorsafe.data."))
    return sorted(seen)


def _function_source(module: str, name: str) -> str:
    tree = ast.parse(_module_path(module).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return ast.unparse(_strip_docstrings(node))
    raise KeyError(f"{module}.{name} not found")


def code_hashes() -> dict[str, str]:
    """Per-unit hashes of the data-building code (module name, or ``module:function``)."""
    out = {
        mod: _digest(normalized_source(_module_path(mod).read_text(encoding="utf-8")))
        for mod in [*data_build_modules(), *_EXTERNAL_MODULES]
    }
    for mod, name in _EXTERNAL_FUNCTIONS:
        out[f"{mod}:{name}"] = _digest(_function_source(mod, name))
    return out


# --------------------------------------------------------------------------- fingerprint
@dataclass(frozen=True)
class BuildFingerprint:
    components: dict[str, str]
    code_modules: dict[str, str]
    teacher: dict[str, Any]
    toxicity_backend: str

    @property
    def id(self) -> str:
        return stable_hash(self.components, 16)


def build_fingerprint(spec: DataSpec, *, teacher: dict[str, Any], toxicity_backend: str) -> BuildFingerprint:
    """Fingerprint of a build of ``spec`` by the current checkout.

    ``teacher`` describes how pairs and neologisms are synthesised: ``offline``, ``alias``, ``backend``,
    ``device`` and the teacher's model spec.
    """
    modules = code_hashes()
    components = {
        "spec": stable_hash(spec.model_dump(mode="json"), 16),
        "code": stable_hash(modules, 16),
        "synth_prompts": stable_hash(load_prompts().get("synth_pair_templates", {}), 16),
        "teacher": stable_hash(teacher, 16),
        "pair_filter": stable_hash({"toxicity_backend": toxicity_backend}, 16),
    }
    return BuildFingerprint(components, modules, teacher, toxicity_backend)


def processed_digest(proc_dir: str | Path) -> str | None:
    """SHA-256 over the processed parquet files, or None when any is missing."""
    proc_dir = Path(proc_dir)
    h = hashlib.sha256()
    for name in PROCESSED_FILES:
        path = proc_dir / name
        if not path.exists():
            return None
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- build_meta.json
def read_build_meta(proc_dir: str | Path) -> dict[str, Any] | None:
    path = Path(proc_dir) / BUILD_META
    if not path.exists():
        return None
    try:
        meta = read_json(path)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def clear_build_meta(proc_dir: str | Path) -> None:
    """Drop the record before rebuilding, so a build that dies midway is never mistaken for the old one."""
    (Path(proc_dir) / BUILD_META).unlink(missing_ok=True)


def write_build_meta(
    proc_dir: str | Path,
    fingerprint: BuildFingerprint,
    *,
    options: dict[str, Any],
    n_examples: int,
    n_pairs: int,
) -> dict[str, Any]:
    meta = {
        "fingerprint": fingerprint.id,
        "components": fingerprint.components,
        "code_modules": fingerprint.code_modules,
        "teacher": fingerprint.teacher,
        "toxicity_backend": fingerprint.toxicity_backend,
        "options": options,
        "data_digest": processed_digest(proc_dir),
        "n_examples": int(n_examples),
        "n_pairs": int(n_pairs),
        "git_sha": git_sha(),
        "built_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(Path(proc_dir) / BUILD_META, meta)
    return meta


# --------------------------------------------------------------------------- comparison
def _teacher_label(t: dict[str, Any] | None) -> str:
    if not t:
        return "unknown"
    if t.get("offline"):
        return "offline build (seed corpus, mock teacher)"
    alias = t.get("alias") or "no teacher"
    return f"{alias} via {t.get('backend', '?')} on {t.get('device', '?')}"


def diff_fingerprint(
    meta: dict[str, Any], expected: BuildFingerprint, ignore: tuple[str, ...] = ()
) -> list[str]:
    """Human-readable reasons why the recorded build differs from ``expected`` (empty when identical).

    Components named in ``ignore`` are skipped (the caller already reported them more specifically).
    """
    recorded = {k: v for k, v in (meta.get("components") or {}).items() if k not in ignore}
    want = {k: v for k, v in expected.components.items() if k not in ignore}
    reasons: list[str] = []
    if recorded.get("code") != want.get("code"):
        old = meta.get("code_modules") or {}
        changed = sorted(
            m for m in set(old) | set(expected.code_modules) if old.get(m) != expected.code_modules.get(m)
        )
        reasons.append(
            "data-building code changed since the build: " + (", ".join(changed) or "unknown modules")
        )
    if recorded.get("spec") != want.get("spec"):
        reasons.append("data spec (configs/data.yaml + overrides) changed since the build")
    if recorded.get("synth_prompts") != want.get("synth_prompts"):
        reasons.append("synth_pair_templates in configs/prompts.yaml changed since the build")
    if recorded.get("teacher") != want.get("teacher"):
        reasons.append(
            f"teacher differs: built with {_teacher_label(meta.get('teacher'))}, "
            f"expected {_teacher_label(expected.teacher)}"
        )
    if recorded.get("pair_filter") != want.get("pair_filter"):
        reasons.append(
            f"pair-filter toxicity backend differs: built with {meta.get('toxicity_backend')!r}, "
            f"expected {expected.toxicity_backend!r}"
        )
    return reasons


@dataclass
class DataStatus:
    """``state`` is ``fresh``, ``missing``, ``unrecorded`` (no build record) or ``stale``."""

    state: str
    reasons: list[str] = field(default_factory=list)
    data_digest: str | None = None
    meta: dict[str, Any] | None = None

    @property
    def fresh(self) -> bool:
        return self.state == "fresh"

    def describe(self) -> str:
        if self.fresh and self.meta:
            return (
                f"fresh (fingerprint {self.meta.get('fingerprint')}, data {self.data_digest}, "
                f"built {self.meta.get('built_at')} at {str(self.meta.get('git_sha', ''))[:8]}, "
                f"{_teacher_label(self.meta.get('teacher'))})"
            )
        return f"{self.state}: " + "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "fresh": self.fresh,
            "reasons": list(self.reasons),
            "data_digest": self.data_digest,
            "fingerprint": (self.meta or {}).get("fingerprint"),
            "built_at": (self.meta or {}).get("built_at"),
            "git_sha": (self.meta or {}).get("git_sha"),
        }
