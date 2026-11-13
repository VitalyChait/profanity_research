"""Small shared helpers: seeding, hashing, JSON I/O, logging, paths."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("MINORSAFE_CONFIG_DIR", REPO_ROOT / "configs"))
DATA_DIR = Path(os.environ.get("MINORSAFE_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("MINORSAFE_RESULTS_DIR", REPO_ROOT / "results"))


def get_logger(name: str = "minorsafe") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("minorsafe").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("minorsafe")
        root.addHandler(handler)
        root.setLevel(os.environ.get("MINORSAFE_LOGLEVEL", "INFO"))
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def stable_hash(obj: Any, length: int = 12) -> str:
    """Deterministic short hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def package_versions(
    names: tuple[str, ...] = ("torch", "transformers", "peft", "trl", "datasets", "vllm"),
) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def extract_json_object(text: str) -> dict | None:
    """Return the first parseable top-level JSON object embedded in ``text``."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None
