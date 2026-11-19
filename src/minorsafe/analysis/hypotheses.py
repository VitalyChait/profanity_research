"""Pre-registered hypothesis evaluation (plan.md §5, H1-H10).

Each hypothesis is evaluated with an explicit, data-driven decision rule. Nothing is hardcoded:

* Rules are applied per matched cell (same experiment, generator ``model`` and ``quant``).
* ``SUPPORTED``     - the rule holds in every evaluable cell
* ``NOT_SUPPORTED`` - the rule holds in no evaluable cell
* ``MIXED``         - the rule holds in some cells but not others (see ``evidence``)
* ``NOT_EVALUATED`` - required data/metrics are missing (the reason is given in ``evidence``)

Hybrid arms run with ``post_filter: true`` are excluded from all pre-registered comparisons.
Thresholds that the plan states only qualitatively are operationalized in the ``rule`` text.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from minorsafe.analysis.stats import holm_bonferroni, paired_significance_test

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
MIXED = "MIXED"
NOT_EVALUATED = "NOT_EVALUATED"

# Matched non-reasoning references for H9 (plan §5: Qwen2.5-1.5B/7B, Llama-3.2-1B/3.1-8B).
H9_REASONING_PAIRS: dict[str, tuple[str, ...]] = {
    "deepseek_r1_15b": ("qwen25_15b", "llama32_1b"),
    "deepseek_r1_8b": ("qwen25_7b", "llama31_8b"),
    "deepseek_r1_7b": ("qwen25_7b", "llama31_8b"),
}


@dataclass
class Verdict:
    hypothesis: str
    statement: str
    rule: str
    status: str
    n_cells: int
    evidence: str


STATEMENTS: dict[str, str] = {
    "H1": "StTP achieves success within 5 pts of LoRA with <1% of LoRA's setup GPU-seconds and setup tokens, "
    "and 1.0x runtime token cost.",
    "H2": "Uniform steering (SwFC) increases perplexity and 4-gram repetition significantly more than "
    "hyperplane-gated steering (StTP/StMP).",
    "H3": "Fine-tuned encoder guards match decoder guards within +/-3 Macro-F1 pts while cutting moderation "
    "latency >=10x and guard output tokens to 0.",
    "H4": "StTP and encoder MLM substitution preserve meaning (BERTScore-F1 >= 0.88) and filter target words "
    "at <=0.5x the total tokens and wall-clock time of LLM rewriting.",
    "H5": "The benign teen-slang direction has non-trivial positive cosine with the profanity direction; "
    "StTP-Ortho reduces over-refusal on benign sociolect vs StTP and M0b.",
    "H6": "Static lexicon filters fail on obfuscated/emoji profanity (recall < 50%); neural guards and "
    "latent probes/steering achieve significantly higher recall/F1.",
    "H7": "Sub-1B decoders show lower linear separability and higher fluency degradation under steering "
    "than ~1B-1.5B and ~7-8B models.",
    "H8": "LoRA degrades general reasoning (tinyMMLU/tinyGSM8K delta) more than StTP.",
    "H9": "Reasoning models (DeepSeek-R1-Distill) consume 2x-5x more tok_total and e2e_ms than matched "
    "non-reasoning models, shifting them off the cost-effectiveness Pareto frontier.",
    "H10": "INT8 retains >=95% of BF16 steerability (SRI) with transferred vectors; 4-bit degrades with "
    "transferred vectors and recovers with natively extracted vectors.",
}


# --------------------------------------------------------------------------- helpers
def _params(row: pd.Series) -> dict[str, Any]:
    raw = row.get("method_params", "{}")
    try:
        out = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (TypeError, ValueError):
        out = {}
    return out if isinstance(out, dict) else {}


def _prepare(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identity columns and drop opt-in post-filter hybrid arms."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    df = summary_df.copy()
    if "method_base" not in df.columns:
        df["method_base"] = df.get("method", pd.Series(["unknown"] * len(df)))
    if "method_params" not in df.columns:
        df["method_params"] = "{}"
    for col, default in (("experiment", "unknown"), ("model", "unknown"), ("quant", "bf16")):
        if col not in df.columns:
            df[col] = default
    keep = [not bool(_params(r).get("post_filter", False)) for _, r in df.iterrows()]
    return df[keep].reset_index(drop=True)


def _num(row: pd.Series | None, col: str) -> float:
    if row is None or col not in row.index:
        return float("nan")
    try:
        v = float(row[col])
    except (TypeError, ValueError):
        return float("nan")
    return v


def _finite(*vals: float) -> bool:
    return all(isinstance(v, float) and math.isfinite(v) for v in vals)


def _pick(
    cell: pd.DataFrame, base: str, where: Callable[[dict[str, Any]], bool] | None = None
) -> pd.Series | None:
    rows = cell[cell["method_base"] == base]
    if where is not None:
        rows = rows[[where(_params(r)) for _, r in rows.iterrows()]]
    return None if rows.empty else rows.iloc[0]


def _cells(df: pd.DataFrame, experiment: str) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    sub = df[df["experiment"] == experiment]
    return [(key, g) for key, g in sub.groupby(["model", "quant"], sort=True)]


def _combine(hid: str, rule: str, results: list[tuple[str, bool, str]], reason_if_empty: str) -> Verdict:
    """results: (cell_label, holds, detail) for each evaluable cell."""
    if not results:
        return Verdict(hid, STATEMENTS[hid], rule, NOT_EVALUATED, 0, reason_if_empty)
    n_hold = sum(1 for _, ok, _ in results if ok)
    if n_hold == len(results):
        status = SUPPORTED
    elif n_hold == 0:
        status = NOT_SUPPORTED
    else:
        status = MIXED
    details = "; ".join(f"{lbl}: {'holds' if ok else 'fails'} ({d})" for lbl, ok, d in results[:12])
    if len(results) > 12:
        details += f"; ... (+{len(results) - 12} cells)"
    return Verdict(
        hid, STATEMENTS[hid], rule, status, len(results), f"{n_hold}/{len(results)} cells hold. {details}"
    )


def _not_evaluated(hid: str, rule: str, reason: str) -> Verdict:
    return Verdict(hid, STATEMENTS[hid], rule, NOT_EVALUATED, 0, reason)


# --------------------------------------------------------------------------- hypotheses
def _h1(df: pd.DataFrame) -> Verdict:
    rule = (
        "main_grid, per (model, quant): success(StTP) >= success(LoRA) - 0.05 AND setup GPU-s(StTP) < "
        "0.01 x LoRA (wall-s if LoRA GPU-s is 0) AND setup_tokens(StTP) < 0.01 x LoRA AND "
        "tok_total(StTP) <= 1.05 x tok_total(M0a) [operationalizes '1.0x runtime token cost']."
    )
    results = []
    for (model, quant), cell in _cells(df, "main_grid"):
        st, lo, m0 = _pick(cell, "m3_steer_sttp"), _pick(cell, "m2_lora"), _pick(cell, "m0a_raw")
        if st is None or lo is None or m0 is None:
            continue
        cost_col = "setup_gpu_s" if _num(lo, "setup_gpu_s") > 0 else "setup_wall_s"
        vals = {
            "s_st": _num(st, "success_mean"),
            "s_lo": _num(lo, "success_mean"),
            "c_st": _num(st, cost_col),
            "c_lo": _num(lo, cost_col),
            "k_st": _num(st, "setup_tokens"),
            "k_lo": _num(lo, "setup_tokens"),
            "t_st": _num(st, "tok_total_mean"),
            "t_m0": _num(m0, "tok_total_mean"),
        }
        if not _finite(*vals.values()) or vals["c_lo"] <= 0 or vals["k_lo"] <= 0 or vals["t_m0"] <= 0:
            continue
        ok = (
            vals["s_st"] >= vals["s_lo"] - 0.05
            and vals["c_st"] < 0.01 * vals["c_lo"]
            and vals["k_st"] < 0.01 * vals["k_lo"]
            and vals["t_st"] <= 1.05 * vals["t_m0"]
        )
        detail = (
            f"success {vals['s_st']:.3f} vs {vals['s_lo']:.3f}; {cost_col} ratio {vals['c_st'] / vals['c_lo']:.4f}; "
            f"setup_tokens ratio {vals['k_st'] / vals['k_lo']:.4f}; tok_total/M0a {vals['t_st'] / vals['t_m0']:.3f}"
        )
        results.append((f"{model}/{quant}", ok, detail))
    return _combine(
        "H1", rule, results, "No main_grid cell with StTP, LoRA, and M0a (with nonzero LoRA setup cost)."
    )


def _h2(df: pd.DataFrame, responses: pd.DataFrame | None) -> Verdict:
    rule = (
        "main_grid, per-example paired over (model, quant, example_id) (seed-averaged): SwFC - StTP and "
        "SwFC - StMP on `ppl` and `repetition_rate`; SUPPORTED iff all 4 mean differences > 0 with "
        "Holm-adjusted two-sided Wilcoxon p < 0.05."
    )
    if responses is None or responses.empty:
        return _not_evaluated("H2", rule, "Requires per-response data (pass responses_df).")
    r = responses
    if "method_base" not in r.columns:
        r = r.assign(method_base=r.get("method"))
    if "method_params" in r.columns:  # exclude opt-in post-filter hybrid arms
        hybrid = r["method_params"].astype(str).str.contains('"post_filter": true', regex=False)
        r = r[~hybrid]
    r = r[(r.get("experiment") == "main_grid")] if "experiment" in r.columns else r.iloc[0:0]
    needed = {"ppl", "repetition_rate", "example_id", "model", "quant"}
    if r.empty or not needed.issubset(r.columns):
        return _not_evaluated("H2", rule, "No main_grid responses with ppl and repetition_rate.")
    key = ["model", "quant", "example_id"]

    def _agg(base: str) -> pd.DataFrame:
        return r[r["method_base"] == base].groupby(key)[["ppl", "repetition_rate"]].mean()

    swfc = _agg("m3_steer_swfc")
    tests: list[tuple[str, dict[str, float]]] = []
    for other in ("m3_steer_sttp", "m3_steer_stmp"):
        joined = swfc.join(_agg(other), how="inner", lsuffix="_a", rsuffix="_b")
        if joined.empty:
            return _not_evaluated("H2", rule, f"No paired SwFC/{other} responses.")
        for metric in ("ppl", "repetition_rate"):
            res = paired_significance_test(joined[f"{metric}_a"].to_numpy(), joined[f"{metric}_b"].to_numpy())
            tests.append((f"swfc-{other.split('_')[-1]}:{metric} (n={len(joined)})", res))
    adj = holm_bonferroni([t[1]["p_value"] for t in tests])
    ok_all = all(t[1]["diff_mean"] > 0 and p < 0.05 for t, p in zip(tests, adj, strict=True))
    detail = "; ".join(
        f"{lbl}: diff={t['diff_mean']:.4f}, p_holm={p:.3g}" for (lbl, t), p in zip(tests, adj, strict=True)
    )
    status = SUPPORTED if ok_all else NOT_SUPPORTED
    return Verdict("H2", STATEMENTS["H2"], rule, status, len(tests), detail)


def _h3(df: pd.DataFrame) -> Verdict:
    rule = "Guard Macro-F1 (enc vs dec, +/-3 pts), e2e_ms_p50 ratio >= 10x, tok_guard_out(enc) == 0."
    return _not_evaluated(
        "H3",
        rule,
        "Guard detection metrics (Macro-F1/AUROC) are not aggregated, and real guard models are not "
        "implemented yet (M1 is a placeholder).",
    )


def _h4(df: pd.DataFrame) -> Verdict:
    rule = (
        "recomposition, per (model, quant), reference = encoder-guard llm_rewrite (else decoder-guard): for BOTH "
        "StTP and encoder mlm_substitute: bertscore_f1_mean >= 0.88 AND tok_total_mean <= 0.5x reference AND "
        "e2e_ms_p50 <= 0.5x reference. Target-word filtering has no pre-registered threshold and is not gated."
    )

    def _action(a: str) -> Callable[[dict[str, Any]], bool]:
        return lambda p: p.get("action") == a

    results = []
    for (model, quant), cell in _cells(df, "recomposition"):
        ref = _pick(cell, "m1_encoder_guard", _action("llm_rewrite"))
        if ref is None:
            ref = _pick(cell, "m1_decoder_guard", _action("llm_rewrite"))
        cands = {
            "StTP": _pick(cell, "m3_steer_sttp"),
            "mlm_sub": _pick(cell, "m1_encoder_guard", _action("mlm_substitute")),
        }
        if ref is None or any(v is None for v in cands.values()):
            continue
        t_ref, e_ref = _num(ref, "tok_total_mean"), _num(ref, "e2e_ms_p50")
        if not _finite(t_ref, e_ref) or t_ref <= 0 or e_ref <= 0:
            continue
        parts, ok, evaluable = [], True, True
        for name, row in cands.items():
            b, t, e = _num(row, "bertscore_f1_mean"), _num(row, "tok_total_mean"), _num(row, "e2e_ms_p50")
            if not _finite(b, t, e):
                evaluable = False
                break
            ok = ok and b >= 0.88 and t <= 0.5 * t_ref and e <= 0.5 * e_ref
            parts.append(f"{name}: bert={b:.3f}, tok x{t / t_ref:.2f}, e2e x{e / e_ref:.2f}")
        if evaluable:
            results.append((f"{model}/{quant}", ok, "; ".join(parts)))
    return _combine(
        "H4", rule, results, "No recomposition cell with StTP, mlm_substitute, and llm_rewrite rows."
    )


def _h5(df: pd.DataFrame) -> Verdict:
    rule = (
        "subculture_audit: cos(v_profanity, v_slang_benign) > 0 from tomography AND, per (model, quant), "
        "slang over-refusal(StTP-Ortho) < StTP and < M0b."
    )
    parts = []
    for (model, quant), cell in _cells(df, "subculture_audit"):
        o, s, b = _pick(cell, "m3_steer_ortho"), _pick(cell, "m3_steer_sttp"), _pick(cell, "m0b_sysprompt")
        vals = [_num(x, "slang_fpr") for x in (o, s, b)]
        if _finite(*vals):
            parts.append(
                f"{model}/{quant}: slang_fpr ortho={vals[0]:.3f}, sttp={vals[1]:.3f}, m0b={vals[2]:.3f}"
            )
    reason = "Cosine clause needs tomography outputs, which are not wired into the pipeline yet."
    if parts:
        reason += " Over-refusal clause (for reference only): " + "; ".join(parts[:12])
    return _not_evaluated("H5", rule, reason)


def _h6(df: pd.DataFrame) -> Verdict:
    rule = "obfuscated subset: lexicon-filter recall < 0.5 AND neural guard / probe recall-F1 significantly higher."
    return _not_evaluated(
        "H6",
        rule,
        "Requires per-detector recall/F1 on the obfuscated subset and a lexicon-only detector baseline; "
        "neither is computed yet (and M1 guards are placeholders).",
    )


def _h7(df: pd.DataFrame) -> Verdict:
    rule = "Layer AUROC (tomography) and steering ΔPPL stratified by size_group: S < M, L (AUROC) and S > M, L (ΔPPL)."
    return _not_evaluated(
        "H7", rule, "Layer-wise separability (tomography) is not wired; ΔPPL needs a real reference LM."
    )


def _h8(df: pd.DataFrame) -> Verdict:
    rule = "main_grid, per (model, quant): mmlu_delta(LoRA) < mmlu_delta(StTP) AND gsm8k_delta(LoRA) < gsm8k_delta(StTP)."
    if not {"mmlu_delta", "gsm8k_delta"}.issubset(df.columns):
        return _not_evaluated(
            "H8", rule, "Capability retention (tinyMMLU/tinyGSM8K) is not wired into the runner."
        )
    results = []
    for (model, quant), cell in _cells(df, "main_grid"):
        lo, st = _pick(cell, "m2_lora"), _pick(cell, "m3_steer_sttp")
        v = [_num(lo, "mmlu_delta"), _num(st, "mmlu_delta"), _num(lo, "gsm8k_delta"), _num(st, "gsm8k_delta")]
        if lo is None or st is None or not _finite(*v):
            continue
        ok = v[0] < v[1] and v[2] < v[3]
        results.append(
            (f"{model}/{quant}", ok, f"mmlu {v[0]:.3f} vs {v[1]:.3f}; gsm8k {v[2]:.3f} vs {v[3]:.3f}")
        )
    return _combine("H8", rule, results, "No main_grid cell with capability deltas for both LoRA and StTP.")


def _pareto_front(points: list[tuple[str, float, float]]) -> set[str]:
    """Return ids not dominated under (maximize success, minimize tokens)."""
    front = set()
    for pid, s, t in points:
        dominated = any((s2 >= s and t2 <= t) and (s2 > s or t2 < t) for _, s2, t2 in points)
        if not dominated:
            front.add(pid)
    return front


def _h9(df: pd.DataFrame) -> Verdict:
    rule = (
        "main_grid, M0a rows, per (reasoning model, matched reference, quant): tok_total ratio >= 2 AND "
        "e2e_ms_p50 ratio >= 2 AND the reasoning model is not on the (max success, min tok_total) Pareto "
        "frontier of all main_grid rows in its size_group/quant. Ratios > 5 are reported but still count as support."
    )
    mg = df[df["experiment"] == "main_grid"]
    results = []
    for quant, qdf in mg.groupby("quant"):
        m0 = qdf[qdf["method_base"] == "m0a_raw"].set_index("model")
        for rmodel, refs in H9_REASONING_PAIRS.items():
            if rmodel not in m0.index:
                continue
            r_row = m0.loc[rmodel]
            r_row = r_row.iloc[0] if isinstance(r_row, pd.DataFrame) else r_row
            sg = str(r_row.get("size_group", ""))
            pool = qdf[qdf["size_group"].astype(str) == sg] if "size_group" in qdf.columns else qdf
            pts = [
                (f"{r['method']}|{r['model']}", _num(r, "success_mean"), _num(r, "tok_total_mean"))
                for _, r in pool.iterrows()
            ]
            pts = [p for p in pts if _finite(p[1], p[2])]
            on_front = f"{r_row['method']}|{rmodel}" in _pareto_front(pts)
            for ref in refs:
                if ref not in m0.index:
                    continue
                f_row = m0.loc[ref]
                f_row = f_row.iloc[0] if isinstance(f_row, pd.DataFrame) else f_row
                tr = _num(r_row, "tok_total_mean") / max(_num(f_row, "tok_total_mean"), 1e-9)
                er = _num(r_row, "e2e_ms_p50") / max(_num(f_row, "e2e_ms_p50"), 1e-9)
                if not _finite(tr, er):
                    continue
                ok = tr >= 2.0 and er >= 2.0 and not on_front
                results.append(
                    (f"{rmodel}~{ref}/{quant}", ok, f"tok x{tr:.2f}, e2e x{er:.2f}, on_pareto={on_front}")
                )
    return _combine(
        "H9", rule, results, "No main_grid M0a rows for a reasoning model and its matched reference."
    )


def _h10(df: pd.DataFrame) -> Verdict:
    rule = "SRI(int8, transferred) >= 0.95 AND SRI(nf4, transferred) < SRI(nf4, native)."
    return _not_evaluated(
        "H10",
        rule,
        "Transferred-vs-native steering vectors are not implemented (`vector_source_quant` is ignored "
        "by the steering method), so SRI cannot be computed.",
    )


def evaluate_hypotheses(
    summary_df: pd.DataFrame, responses_df: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    """Evaluate pre-registered H1-H10 from aggregated (and optionally per-response) results."""
    df = _prepare(summary_df)
    if df.empty:
        return [
            asdict(Verdict(h, STATEMENTS[h], "", NOT_EVALUATED, 0, "No aggregated results."))
            for h in STATEMENTS
        ]
    verdicts = [
        _h1(df),
        _h2(df, responses_df),
        _h3(df),
        _h4(df),
        _h5(df),
        _h6(df),
        _h7(df),
        _h8(df),
        _h9(df),
        _h10(df),
    ]
    return [asdict(v) for v in verdicts]


__all__ = ["MIXED", "NOT_EVALUATED", "NOT_SUPPORTED", "SUPPORTED", "evaluate_hypotheses"]
