"""Semantic preservation and dynamic sentence recomposition metrics (BERTScore-F1, SARI, filter accuracy)."""

from __future__ import annotations

from collections import Counter

from minorsafe.data.lexicons import find_complex_spans, find_profanity_spans
from minorsafe.data.obfuscation import detect_obfuscated_profanity
from minorsafe.utils import normalize_text


def _ngrams(text: str, n: int) -> Counter[tuple[str, ...]]:
    toks = normalize_text(text).split()
    if len(toks) < n:
        return Counter()
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _f1(prec: float, rec: float) -> float:
    if prec + rec <= 1e-12:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def compute_sari(source: str, prediction: str, references: list[str]) -> float:
    """Compute standard SARI lexical simplification/recomposition score in [0, 100]."""
    refs = [r for r in references if r] or [source]
    add_scores, keep_scores, del_scores = [], [], []

    for n in (1, 2, 3, 4):
        src_g = set(_ngrams(source, n).keys())
        pred_g = set(_ngrams(prediction, n).keys())
        ref_g: set[tuple[str, ...]] = set()
        for r in refs:
            ref_g |= set(_ngrams(r, n).keys())

        # ADD operation: n-grams in prediction that were not in source
        pred_added = pred_g - src_g
        ref_added = ref_g - src_g
        if pred_added or ref_added:
            p_add = len(pred_added & ref_added) / max(1, len(pred_added))
            r_add = len(pred_added & ref_added) / max(1, len(ref_added))
            add_scores.append(_f1(p_add, r_add))
        else:
            add_scores.append(1.0)

        # KEEP operation: n-grams in both source and reference kept by prediction
        src_kept = src_g & pred_g
        ref_kept = src_g & ref_g
        if src_kept or ref_kept:
            p_keep = len(src_kept & ref_kept) / max(1, len(src_kept))
            r_keep = len(src_kept & ref_kept) / max(1, len(ref_kept))
            keep_scores.append(_f1(p_keep, r_keep))
        else:
            keep_scores.append(1.0)

        # DEL operation: precision of deleting n-grams from source that are not in reference
        pred_del = src_g - pred_g
        ref_del = src_g - ref_g
        if pred_del or ref_del:
            p_del = (
                len(pred_del & ref_del) / max(1, len(pred_del)) if pred_del else (1.0 if not ref_del else 0.0)
            )
            del_scores.append(p_del)
        else:
            del_scores.append(1.0)

    mean_add = sum(add_scores) / len(add_scores)
    mean_keep = sum(keep_scores) / len(keep_scores)
    mean_del = sum(del_scores) / len(del_scores)
    return 100.0 * (mean_add + mean_keep + mean_del) / 3.0


def _token_f1_similarity(a: str, b: str) -> float:
    ta = Counter(normalize_text(a).split())
    tb = Counter(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    overlap = sum((ta & tb).values())
    prec = overlap / sum(ta.values())
    rec = overlap / sum(tb.values())
    return _f1(prec, rec)


def compute_recomposition_metrics(
    source: str,
    prediction: str,
    reference: str | None = None,
    baseline_m0: str | None = None,
    bertscore_model: str | None = None,
) -> dict:
    """Compute BERTScore-F1 (or fast token F1 fallback), SARI, and target-word filtering accuracy."""
    ref_txt = reference or source
    m0_txt = baseline_m0 or source

    bs_ref: float | None = None
    bs_m0: float | None = None
    if bertscore_model:
        try:
            from bert_score import score as bert_score_fn

            _, _, f1_ref = bert_score_fn([prediction], [ref_txt], model_type=bertscore_model, verbose=False)
            bs_ref = float(f1_ref[0].item())
            _, _, f1_m0 = bert_score_fn([prediction], [m0_txt], model_type=bertscore_model, verbose=False)
            bs_m0 = float(f1_m0[0].item())
        except Exception:
            bs_ref = None
            bs_m0 = None

    if bs_ref is None:
        bs_ref = _token_f1_similarity(prediction, ref_txt)
    if bs_m0 is None:
        bs_m0 = _token_f1_similarity(prediction, m0_txt)

    sari = compute_sari(source, prediction, [ref_txt])

    # Target word filtering check: zero profane/obfuscated/complex target spans remaining in prediction
    has_bad_words = bool(
        find_profanity_spans(prediction)
        or detect_obfuscated_profanity(prediction)
        or find_complex_spans(prediction)
    )
    filter_acc = bool((not has_bad_words) and (bs_ref >= 0.70))

    return {
        "bertscore_f1_ref": float(bs_ref),
        "bertscore_f1_m0": float(bs_m0),
        "sari_score": float(sari),
        "recompose_filter_acc": filter_acc,
    }
