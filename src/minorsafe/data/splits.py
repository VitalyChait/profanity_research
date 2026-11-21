"""Deterministic hash splitting and cross-split normalized-text deduplication.

Examples are split by *prompt family*, not by their own text, so that no prompt reaches two splits:
- an augmented variant (``meta["parent_id"]``, e.g. obfuscated copies) inherits its parent's family;
- padded copies that differ only by a counter suffix (``" [item 13]"``, ``" [case 4]"``) share a family;
- the family key is the prompt alone, so one prompt with several stored responses stays in one split.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import TypeVar

from minorsafe.data.schema import ContrastivePair, Example
from minorsafe.utils import normalize_text

T = TypeVar("T", Example, ContrastivePair)

# Counter suffixes appended by the seed / neologism builders to pad the corpus with copies.
_COPY_SUFFIX_RE = re.compile(r"\s*(?:\[(?:item|case)\s+\d+\]|\(#\d+\))\s*$", re.IGNORECASE)


def _item_norm_key(item: Example | ContrastivePair) -> str:
    if isinstance(item, Example):
        return normalize_text(item.last_user + " " + (item.response or ""))
    return normalize_text(item.prompt + " " + item.positive + " " + item.negative)


def _canonical_prompt_key(ex: Example) -> str:
    return normalize_text(_COPY_SUFFIX_RE.sub("", ex.last_user))


def _family_keys(items: list[Example | ContrastivePair]) -> list[str]:
    """Return the split-family key for each item (pairs keep their exact normalized key)."""
    by_id = {it.id: it for it in items if isinstance(it, Example)}

    def _root_key(ex: Example) -> str:
        cur, seen = ex, {ex.id}
        while True:
            parent_id = cur.meta.get("parent_id") if isinstance(cur.meta, dict) else None
            if not parent_id:
                return _canonical_prompt_key(cur)
            parent = by_id.get(str(parent_id))
            if parent is None:
                # Parent not in this batch: siblings still group together via the shared parent id.
                return f"parent:{parent_id}"
            if parent.id in seen:
                return _canonical_prompt_key(cur)
            seen.add(parent.id)
            cur = parent

    return [_root_key(it) if isinstance(it, Example) else _item_norm_key(it) for it in items]


def _bucket_score(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    val = int.from_bytes(digest[:8], "big")
    return val / float(2**64)


def assign_splits(
    items: list[T],
    ratios: dict[str, float] | None = None,
    seed: int = 13,
) -> list[T]:
    """Deduplicate by normalized text and deterministically assign each prompt family to calib / dev / test.

    All members of a family (see module docstring) get the same split, so augmented variants and
    suffix-padded copies of a calibration prompt can never appear in dev/test.
    """
    ratios = ratios or {"calib": 0.50, "dev": 0.20, "test": 0.30}
    total = sum(ratios.values())
    c_cut = ratios.get("calib", 0.5) / total
    d_cut = c_cut + (ratios.get("dev", 0.2) / total)

    families = _family_keys(list(items))
    seen_norm: set[str] = set()
    out: list[T] = []
    for item, family in zip(items, families, strict=True):
        norm = _item_norm_key(item)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        u = _bucket_score(family or norm, seed)
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
    """Return count of exact texts or prompt families that appear in more than one split (must be 0)."""
    items = list(items)
    key_splits: dict[tuple[str, str], set[str]] = {}
    for item, family in zip(items, _family_keys(items), strict=True):
        if not item.split:
            continue
        key_splits.setdefault(("text", _item_norm_key(item)), set()).add(item.split)
        key_splits.setdefault(("family", family), set()).add(item.split)
    return sum(1 for splits in key_splits.values() if len(splits) > 1)


def stratified_subsample(examples: list[Example], max_examples: int, seed: int = 13) -> list[Example]:
    """Deterministically round-robin across primary categories so rare subsets stay covered."""
    if max_examples <= 0 or max_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = {}
    for ex in examples:
        cat = ex.categories[0] if ex.categories else "benign"
        buckets.setdefault(cat, []).append(ex)
    for items in buckets.values():
        rng.shuffle(items)

    chosen: list[Example] = []
    keys = sorted(buckets.keys())
    idx = 0
    while len(chosen) < max_examples:
        added_any = False
        for k in keys:
            if idx < len(buckets[k]):
                chosen.append(buckets[k][idx])
                added_any = True
                if len(chosen) >= max_examples:
                    break
        if not added_any:
            break
        idx += 1
    return chosen
