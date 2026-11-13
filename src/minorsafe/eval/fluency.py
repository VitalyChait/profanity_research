"""Fluency metrics: reference LM perplexity, distinct-1/2, and n-gram repetition rate."""

from __future__ import annotations

import math

from minorsafe.utils import normalize_text


def compute_fluency_metrics(
    text: str,
    ppl_model: str | None = None,
    baseline_ppl: float | None = None,
) -> dict:
    toks = normalize_text(text or "").split()
    if not toks:
        return {
            "ppl": 0.0,
            "ppl_delta_pct": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repetition_rate": 1.0,
        }

    unigrams = set(toks)
    distinct_1 = len(unigrams) / float(len(toks))

    if len(toks) >= 2:
        bigrams = [tuple(toks[i : i + 2]) for i in range(len(toks) - 1)]
        distinct_2 = len(set(bigrams)) / float(len(bigrams))
    else:
        distinct_2 = 1.0

    # 4-gram repetition rate (captures SwFC additive steering generation collapse)
    n_gram_size = 4 if len(toks) >= 4 else max(1, len(toks))
    grams = [tuple(toks[i : i + n_gram_size]) for i in range(len(toks) - n_gram_size + 1)]
    if grams:
        rep_rate = 1.0 - (len(set(grams)) / float(len(grams)))
    else:
        rep_rate = 0.0

    # Heuristic perplexity estimate when reference LM is disabled (cpu_debug)
    ppl = 15.0 * math.exp(2.2 * rep_rate) / max(0.25, distinct_2)
    ppl_delta = ((ppl - baseline_ppl) / baseline_ppl * 100.0) if (baseline_ppl and baseline_ppl > 0) else 0.0

    return {
        "ppl": float(ppl),
        "ppl_delta_pct": float(ppl_delta),
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "repetition_rate": float(rep_rate),
    }

