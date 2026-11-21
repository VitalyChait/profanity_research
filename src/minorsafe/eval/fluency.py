"""Fluency metrics: reference LM perplexity, distinct-1/2, and n-gram repetition rate."""

from __future__ import annotations

import math

import torch

from minorsafe.registry import compute_dtype, default_device, free_model
from minorsafe.utils import normalize_text

_NAN = float("nan")


def _empty_fluency() -> dict:
    """A response with no words has no perplexity (NaN, never a perfect 0.0) and counts as fully degenerate."""
    return {
        "ppl": _NAN,
        "ppl_delta_pct": _NAN,
        "distinct_1": 0.0,
        "distinct_2": 0.0,
        "repetition_rate": 1.0,
    }


def ppl_delta_pct(ppl: float, baseline_ppl: float | None) -> float:
    """Relative PPL change vs the M0 baseline in %; NaN when either side is undefined (never 0.0)."""
    if (
        baseline_ppl is None
        or not math.isfinite(baseline_ppl)
        or baseline_ppl <= 0
        or ppl is None
        or not math.isfinite(ppl)
    ):
        return _NAN
    return (ppl - baseline_ppl) / baseline_ppl * 100.0


def _ngram_stats(text: str) -> tuple[float, float, float]:
    toks = normalize_text(text or "").split()
    if not toks:
        return 0.0, 0.0, 1.0
    distinct_1 = len(set(toks)) / float(len(toks))
    if len(toks) >= 2:
        bigrams = [tuple(toks[i : i + 2]) for i in range(len(toks) - 1)]
        distinct_2 = len(set(bigrams)) / float(len(bigrams))
    else:
        distinct_2 = 1.0
    n_gram_size = 4 if len(toks) >= 4 else max(1, len(toks))
    grams = [tuple(toks[i : i + n_gram_size]) for i in range(len(toks) - n_gram_size + 1)]
    rep_rate = (1.0 - (len(set(grams)) / float(len(grams)))) if grams else 0.0
    return float(distinct_1), float(distinct_2), float(rep_rate)


@torch.no_grad()
def compute_ppl_batch(
    texts: list[str],
    ppl_model: str,
    device: str | None = None,
    max_length: int = 512,
) -> list[float]:
    """Compute per-text perplexity under a fixed external reference LM (e.g. ``gpt2-large``)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not texts:
        return []
    dev = device or default_device()
    tok = AutoTokenizer.from_pretrained(ppl_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ppl_model, dtype=compute_dtype("bf16", dev))
    model.to(dev).eval()

    ppls: list[float] = []
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    try:
        for txt in texts:
            clean = (txt or "").strip()
            if not clean:
                ppls.append(float("nan"))
                continue
            ids = tok(clean, add_special_tokens=False, truncation=True, max_length=max_length - 1)[
                "input_ids"
            ]
            if len(ids) < 2 and bos_id is not None:
                ids = [int(bos_id), *ids]
            if len(ids) < 2:
                ppls.append(float("nan"))
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=dev)
            loss = float(model(input_ids=inp, labels=inp).loss.float().item())
            ppls.append(float(math.exp(min(20.0, loss))))
    finally:
        free_model(model, tok)
    return ppls


def compute_fluency_metrics(
    text: str,
    ppl_model: str | None = None,
    baseline_ppl: float | None = None,
) -> dict:
    toks = normalize_text(text or "").split()
    if not toks:
        return _empty_fluency()

    distinct_1, distinct_2, rep_rate = _ngram_stats(text)
    if ppl_model:
        ppl = compute_ppl_batch([text], ppl_model=ppl_model)[0]
    else:
        # Heuristic perplexity estimate when reference LM is disabled (cpu_debug)
        ppl = 15.0 * math.exp(2.2 * rep_rate) / max(0.25, distinct_2)

    return {
        "ppl": float(ppl),
        "ppl_delta_pct": float(ppl_delta_pct(ppl, baseline_ppl)),
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "repetition_rate": float(rep_rate),
    }


def compute_fluency_metrics_batch(
    texts: list[str],
    ppl_model: str | None = None,
    baseline_ppls: list[float | None] | None = None,
    device: str | None = None,
) -> list[dict]:
    """Batched fluency calculation loading the reference PPL LM at most once per call."""
    if not texts:
        return []
    if ppl_model:
        ppls = compute_ppl_batch(texts, ppl_model=ppl_model, device=device)
    else:
        ppls = [None] * len(texts)  # type: ignore[list-item]

    b_list = baseline_ppls if baseline_ppls is not None else [None] * len(texts)
    out: list[dict] = []
    for txt, ppl_val, b_ppl in zip(texts, ppls, b_list, strict=True):
        toks = normalize_text(txt or "").split()
        if not toks:
            out.append(_empty_fluency())
            continue
        d1, d2, rep = _ngram_stats(txt)
        ppl = float(ppl_val) if ppl_val is not None else (15.0 * math.exp(2.2 * rep) / max(0.25, d2))
        out.append(
            {
                "ppl": float(ppl),
                "ppl_delta_pct": float(ppl_delta_pct(ppl, b_ppl)),
                "distinct_1": float(d1),
                "distinct_2": float(d2),
                "repetition_rate": float(rep),
            }
        )
    return out
