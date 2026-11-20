"""Deterministic hash splitting and cross-split normalized-text deduplication."""

from __future__ import annotations

import hashlib
from typing import TypeVar

from minorsafe.data.schema import ContrastivePair, Example
from minorsafe.utils import normalize_text

T = TypeVar("T", Example, ContrastivePair)


def _item_norm_key(item: Example | ContrastivePair) -> str:
    if isinstance(item, Example):
        return normalize_text(item.last_user + " " + (item.response or ""))
    return normalize_text(item.prompt + " " + item.positive + " " + item.negative)


def _bucket_score(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    val = int.from_bytes(digest[:8], "big")
    return val / float(2**64)


def assign_splits(
    items: list[T],
    ratios: dict[str, float] | None = None,
    seed: int = 13,
) -> list[T]:
    """Deduplicate by normalized text and deterministically assign each item to calib / dev / test."""
    ratios = ratios or {"calib": 0.50, "dev": 0.20, "test": 0.30}
    total = sum(ratios.values())
    c_cut = ratios.get("calib", 0.5) / total
    d_cut = c_cut + (ratios.get("dev", 0.2) / total)

    seen_norm: set[str] = set()
    out: list[T] = []
    for item in items:
        norm = _item_norm_key(item)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        u = _bucket_score(norm, seed)
        if u < c_cut:
            split_name = "calib"
        elif u < d_cut:
            split_name = "dev"
        else:
            split_name = "test"
        cloned = item.model_copy(update={"split": split_name})
        out.append(cloned)
    return out


def check_split_leakage(items: list[Example | ContrastivePair]) -> int:
    """Return count of normalized keys that appear in more than one split (must be 0)."""
    key_splits: dict[str, set[str]] = {}
    for item in items:
        if not item.split:
            continue
        k = _item_norm_key(item)
        key_splits.setdefault(k, set()).add(item.split)
    return sum(1 for splits in key_splits.values() if len(splits) > 1)
