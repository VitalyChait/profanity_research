"""Teacher contrastive pair synthesis and strict quality filtering (FK grade, profanity, length parity, MinHash)."""

from __future__ import annotations

import textstat
from datasketch import MinHash, MinHashLSH

from minorsafe.data.lexicons import find_profanity_spans, is_english
from minorsafe.data.minimal_pairs import build_minimal_pairs
from minorsafe.data.obfuscation import detect_obfuscated_profanity
from minorsafe.data.schema import ContrastivePair
from minorsafe.data.sources import load_seed_pairs
from minorsafe.telemetry.cost import SetupReport
from minorsafe.utils import normalize_text


def _shingle_minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    words = normalize_text(text).split()
    for i in range(max(1, len(words) - 2)):
        shingle = " ".join(words[i : i + 3])
        m.update(shingle.encode("utf-8"))
    return m


def filter_contrastive_pairs(
    pairs: list[ContrastivePair],
    english_min_ratio: float = 0.60,
    max_length_rel_diff: float = 0.45,
    jaccard_threshold: float = 0.85,
) -> tuple[list[ContrastivePair], dict[str, int]]:
    """Filter pairs by English ratio, length parity, positive cleanliness, and MinHash deduplication."""
    stats = {
        "input": len(pairs),
        "dropped_english": 0,
        "dropped_positive_unsafe": 0,
        "dropped_length_parity": 0,
        "dropped_duplicate": 0,
        "kept": 0,
    }
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=64)
    kept: list[ContrastivePair] = []

    for idx, p in enumerate(pairs):
        if not is_english(p.positive, min_ratio=english_min_ratio):
            stats["dropped_english"] += 1
            continue
        # Positives must never contain explicit or obfuscated profanity
        if find_profanity_spans(p.positive) or detect_obfuscated_profanity(p.positive):
            stats["dropped_positive_unsafe"] += 1
            continue
        if p.concept == "lexical_complexity":
            pos_fk = textstat.flesch_kincaid_grade(p.positive)
            neg_fk = textstat.flesch_kincaid_grade(p.negative)
            if pos_fk > neg_fk:
                stats["dropped_positive_unsafe"] += 1
                continue
        len_pos = max(1, len(p.positive.split()))
        len_neg = max(1, len(p.negative.split()))
        rel_diff = abs(len_pos - len_neg) / float(max(len_pos, len_neg))
        if rel_diff > max_length_rel_diff:
            stats["dropped_length_parity"] += 1
            continue

        mh = _shingle_minhash(f"{p.concept} {p.prompt} {p.positive} {p.negative}")
        if lsh.query(mh):
            stats["dropped_duplicate"] += 1
            continue
        lsh.insert(f"p_{idx}", mh)
        kept.append(p)

    stats["kept"] = len(kept)
    return kept, stats


def synthesize_contrastive_pairs(
    concepts: list[str],
    pairs_per_concept: int = 40,
    seed: int = 13,
    setup_report: SetupReport | None = None,
) -> tuple[list[ContrastivePair], SetupReport]:
    """Generate and validate contrastive pairs across requested concepts, logging SetupReport."""
    rep = setup_report or SetupReport(method="data_synth", setup_key=f"synth_{seed}")
    with rep.stage("synth_pairs", examples=len(concepts) * pairs_per_concept) as st:
        minimal = build_minimal_pairs(max_pairs=pairs_per_concept * 2, seed=seed)
        seed_p = load_seed_pairs(max_items=pairs_per_concept * 3)
        raw_pool = [p for p in (*minimal, *seed_p) if p.concept in concepts]
        filtered, _ = filter_contrastive_pairs(raw_pool)
        st.tokens = sum(len((p.prompt + p.positive + p.negative).split()) * 2 for p in filtered)
    return filtered, rep
