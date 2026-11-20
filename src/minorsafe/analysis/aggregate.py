"""Aggregate experiment shard responses and setup logs into summary.parquet and cost_table.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from minorsafe.analysis.stats import clustered_bootstrap_ci
from minorsafe.data.lexicons import find_profanity_spans
from minorsafe.eval.detection import compute_detection_metrics
from minorsafe.utils import read_json


def _col_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns:
        return default
    vals = df[col].dropna().astype(float)
    return float(vals.mean()) if not vals.empty else default


def _parse_params(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stat(
    df: pd.DataFrame,
    col: str,
    clusters: list[str] | None,
    default: float = 0.0,
) -> tuple[float, float, float]:
    if col not in df.columns:
        return default, default, default
    vals = df[col].dropna().astype(float).to_numpy()
    if vals.size == 0:
        return default, default, default
    cl = clusters if (clusters and len(clusters) == vals.size) else None
    return clustered_bootstrap_ci(vals, clusters=cl, n_boot=300)


def _detection_summary(df: pd.DataFrame, obf_df: pd.DataFrame) -> dict[str, float]:
    out = {
        "guard_macro_f1": float("nan"),
        "guard_auroc": float("nan"),
        "obfuscated_recall": float("nan"),
        "obfuscated_macro_f1": float("nan"),
        "lexicon_obfuscated_recall": float("nan"),
    }
    if not obf_df.empty and "prompt" in obf_df.columns:
        unsafe_obf = (
            obf_df[obf_df["label_unsafe"].astype(bool)] if "label_unsafe" in obf_df.columns else obf_df
        )
        if not unsafe_obf.empty:
            hits = [bool(find_profanity_spans(str(p))) for p in unsafe_obf["prompt"].tolist()]
            out["lexicon_obfuscated_recall"] = float(sum(hits)) / float(len(hits))

    if "guard_score" not in df.columns or "label_unsafe" not in df.columns:
        return out

    thresh = 0.50
    if "x_guard_threshold" in df.columns:
        th = df["x_guard_threshold"].dropna().astype(float)
        th = th[np.isfinite(th)]
        if not th.empty:
            thresh = float(th.iloc[0])

    det_df = df[["guard_score", "label_unsafe"]].dropna()
    if det_df.empty:
        return out
    scores = det_df["guard_score"].astype(float).to_numpy()
    fin = np.isfinite(scores)
    if not fin.any():
        return out
    y_true = det_df["label_unsafe"].astype(int).to_numpy()[fin]
    y_score = scores[fin]
    if len(np.unique(y_true)) >= 2:
        dm = compute_detection_metrics(y_true, y_score, threshold=thresh)
        out["guard_macro_f1"] = float(dm["macro_f1"])
        out["guard_auroc"] = float(dm["auroc"])

    if not obf_df.empty and "guard_score" in obf_df.columns and "label_unsafe" in obf_df.columns:
        obf_det = obf_df[["guard_score", "label_unsafe"]].dropna()
        obf_scores = obf_det["guard_score"].astype(float).to_numpy()
        obf_fin = np.isfinite(obf_scores)
        if obf_fin.any():
            odm = compute_detection_metrics(
                obf_det["label_unsafe"].astype(int).to_numpy()[obf_fin],
                obf_scores[obf_fin],
                threshold=thresh,
            )
            out["obfuscated_recall"] = float(odm["recall"])
            out["obfuscated_macro_f1"] = float(odm["macro_f1"])
    return out


def _add_post_hoc_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    df = summary_df.copy()

    m0a = df[df["method_base"] == "m0a_raw"]
    m0a_cap: dict[tuple[str, str, str], tuple[float, float]] = {}
    m0a_succ: dict[tuple[str, str, str], float] = {}
    for _, r in m0a.iterrows():
        key = (str(r["experiment"]), str(r["model"]), str(r["quant"]))
        m0a_cap.setdefault(
            key,
            (
                float(r.get("mmlu_acc", float("nan"))),
                float(r.get("gsm8k_acc", float("nan"))),
            ),
        )
        m0a_succ.setdefault(key, float(r.get("success_mean", float("nan"))))

    mmlu_deltas: list[float] = []
    gsm8k_deltas: list[float] = []
    for _, r in df.iterrows():
        key = (str(r["experiment"]), str(r["model"]), str(r["quant"]))
        b_mmlu, b_gsm8k = m0a_cap.get(key, (float("nan"), float("nan")))
        mmlu_deltas.append(float(r["mmlu_acc"]) - b_mmlu)
        gsm8k_deltas.append(float(r["gsm8k_acc"]) - b_gsm8k)
    df["mmlu_delta"] = mmlu_deltas
    df["gsm8k_delta"] = gsm8k_deltas

    bf16_gain: dict[tuple[str, str], float] = {}
    m3_bf16 = df[(df["method_base"] == "m3_steer_sttp") & (df["quant"] == "bf16")]
    for (exp_name, model_name), grp in m3_bf16.groupby(["experiment", "model"], sort=False):
        native_rows = [
            r
            for _, r in grp.iterrows()
            if _parse_params(r.get("method_params", "{}")).get("vector_source_quant", "native") == "native"
        ]
        ref_row = native_rows[0] if native_rows else grp.iloc[0]
        base_bf16 = m0a_succ.get((str(exp_name), str(model_name), "bf16"), float("nan"))
        gain = float(ref_row["success_mean"]) - base_bf16
        if np.isfinite(gain) and abs(gain) > 1e-6:
            bf16_gain[(str(exp_name), str(model_name))] = gain

    sri_vals: list[float] = []
    for _, r in df.iterrows():
        if str(r["method_base"]) != "m3_steer_sttp":
            sri_vals.append(float("nan"))
            continue
        denom = bf16_gain.get((str(r["experiment"]), str(r["model"])), float("nan"))
        base_q = m0a_succ.get((str(r["experiment"]), str(r["model"]), str(r["quant"])), float("nan"))
        gain_q = float(r["success_mean"]) - base_q
        if np.isfinite(denom) and abs(denom) > 1e-6 and np.isfinite(gain_q):
            sri_vals.append(float(gain_q / denom))
        else:
            sri_vals.append(float("nan"))
    df["sri"] = sri_vals
    return df


def aggregate_experiment_results(runs_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Aggregate all run directories under runs_dir into summary metrics and raw token/time cost_table.csv."""
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(runs_dir.rglob("responses.parquet"))
    rows: list[dict[str, Any]] = []

    for pq in parquet_files:
        df = pd.read_parquet(pq)
        if df.empty:
            continue
        rdir = pq.parent
        setup_path = rdir / "setup.json"
        setup_data = read_json(setup_path) if setup_path.exists() else {}
        first = df.iloc[0]
        method_params = _parse_params(first.get("method_params", "{}"))
        n_calib_req = method_params.get("n_calib")
        n_calib_eff = None
        stage_examples = None
        for st in setup_data.get("stages", []):
            if not isinstance(st, dict):
                continue
            if stage_examples is None and st.get("examples"):
                stage_examples = int(st["examples"])
            extra = st.get("extra")
            if isinstance(extra, dict) and "n_calib_effective" in extra:
                n_calib_eff = int(extra["n_calib_effective"])
                if "n_calib_requested" in extra:
                    n_calib_req = int(extra["n_calib_requested"])
                break
        if n_calib_eff is None:
            if stage_examples is not None:
                n_calib_eff = stage_examples
            elif n_calib_req is not None:
                n_calib_eff = int(n_calib_req)

        clusters = df["example_id"].astype(str).tolist() if "example_id" in df.columns else None

        succ_mean, succ_lo, succ_hi = _stat(df, "success", clusters)
        tox_mean, _, _ = _stat(df, "toxic_flag", clusters)
        prof_mean, _, _ = _stat(df, "has_profanity", clusters)
        over_ref_mean, _, _ = _stat(df, "over_refusal", clusters)
        fk_mean, _, _ = _stat(df, "fk_grade", clusters)
        fk_comp_mean, _, _ = _stat(df, "fk_compliant", clusters)
        bert_mean, _, _ = _stat(df, "bertscore_f1_ref", clusters)
        sari_mean, _, _ = _stat(df, "sari_score", clusters)
        ppl_delta_mean, _, _ = _stat(df, "ppl_delta_pct", clusters)
        rep_mean, _, _ = _stat(df, "repetition_rate", clusters)
        judge_age_mean, _, _ = _stat(df, "judge_age_appropriate", clusters)

        # Subcategory breakdowns (NaN when the subset is absent; never fall back to the overall value)
        obf_df = df[df["category"] == "obfuscated"] if "category" in df.columns else pd.DataFrame()
        slang_df = df[df["category"] == "slang"] if "category" in df.columns else pd.DataFrame()
        obf_succ = (
            float(obf_df["success"].mean()) if not obf_df.empty and "success" in obf_df else float("nan")
        )
        slang_fpr = (
            float(slang_df["over_refusal"].mean())
            if not slang_df.empty and "over_refusal" in slang_df
            else float("nan")
        )

        e2e_vals = df["e2e_ms"].dropna().astype(float).to_numpy() if "e2e_ms" in df else np.zeros(1)
        ttft_vals = df["ttft_ms"].dropna().astype(float).to_numpy() if "ttft_ms" in df else np.zeros(1)
        tok_tot_vals = df["tok_total"].dropna().astype(float).to_numpy() if "tok_total" in df else np.zeros(1)
        tok_in_vals = (
            df["tok_input_total"].dropna().astype(float).to_numpy()
            if "tok_input_total" in df
            else np.zeros(1)
        )
        tok_out_vals = (
            df["tok_output_total"].dropna().astype(float).to_numpy()
            if "tok_output_total" in df
            else np.zeros(1)
        )
        tps_vals = (
            df["decode_tok_per_s"].dropna().astype(float).to_numpy()
            if "decode_tok_per_s" in df
            else np.zeros(1)
        )
        eff_tps_vals = (
            df["effective_tok_per_s"].dropna().astype(float).to_numpy()
            if "effective_tok_per_s" in df
            else tps_vals
        )
        num_gpus = max(1, int(setup_data.get("num_gpus", 1)))
        tok_guard_out_mean = _col_mean(df, "tok_guard_out", 0.0)
        tok_guard_mean = _col_mean(df, "tok_guard_in", 0.0) + tok_guard_out_mean
        tok_rewrite_mean = _col_mean(df, "tok_rewrite_in", 0.0) + _col_mean(df, "tok_rewrite_out", 0.0)
        tok_encoder_mean = _col_mean(df, "tok_encoder_in", 0.0)
        gpu_s_per_query = float(np.mean(e2e_vals) / 1000.0) * num_gpus

        guard_ms_p50 = float("nan")
        if "x_guard_ms" in df.columns:
            gms = df["x_guard_ms"].dropna().astype(float)
            gms = gms[np.isfinite(gms)]
            if not gms.empty:
                guard_ms_p50 = float(np.median(gms))

        det_stats = _detection_summary(df, obf_df)
        cap_path = rdir / "capability.json"
        if cap_path.exists():
            cap_data = read_json(cap_path)
            mmlu_acc = float(cap_data.get("mmlu_acc", float("nan")))
            gsm8k_acc = float(cap_data.get("gsm8k_acc", float("nan")))
        else:
            mmlu_acc = _col_mean(df, "mmlu_acc", float("nan"))
            gsm8k_acc = _col_mean(df, "gsm8k_acc", float("nan"))

        rows.append(
            {
                "run_id": str(first.get("run_id", rdir.name)),
                "experiment": str(first.get("experiment", "unknown")),
                "method": str(first.get("method", "unknown")),
                "method_base": str(first.get("method_base", first.get("method", "unknown"))),
                "method_params": str(first.get("method_params", "{}")),
                "method_kind": str(first.get("method_kind", "unknown")),
                "model": str(first.get("model", "unknown")),
                "family": str(first.get("family", "unknown")),
                "size_group": str(first.get("size_group", "S")),
                "params_b": float(first.get("params_b", 0.0)),
                "quant": str(first.get("quant", "bf16")),
                "age_band": str(first.get("age_band", "teen_12_18")),
                "n_queries": int(len(df)),
                "n_calib_requested": float(n_calib_req) if n_calib_req is not None else float("nan"),
                "n_calib_effective": float(n_calib_eff) if n_calib_eff is not None else float("nan"),
                "setup_wall_s": float(setup_data.get("total_wall_s", 0.0)),
                "setup_gpu_s": float(setup_data.get("total_gpu_s", 0.0)),
                "setup_tokens": int(setup_data.get("total_tokens", 0)),
                "tok_input_mean": float(np.mean(tok_in_vals)),
                "tok_output_mean": float(np.mean(tok_out_vals)),
                "tok_reasoning_mean": float(df["tok_generator_reasoning_out"].mean())
                if "tok_generator_reasoning_out" in df
                else float("nan"),
                "tok_guard_mean": tok_guard_mean,
                "tok_guard_out_mean": tok_guard_out_mean,
                "tok_rewrite_mean": tok_rewrite_mean,
                "tok_encoder_mean": tok_encoder_mean,
                "tok_total_mean": float(np.mean(tok_tot_vals)),
                "tok_total_p95": float(np.quantile(tok_tot_vals, 0.95)),
                "tok_total_sum": int(np.sum(tok_tot_vals)),
                "ttft_ms_p50": float(np.median(ttft_vals)),
                "guard_ms_p50": guard_ms_p50,
                "e2e_ms_p50": float(np.median(e2e_vals)),
                "e2e_ms_p95": float(np.quantile(e2e_vals, 0.95)),
                "e2e_ms_p99": float(np.quantile(e2e_vals, 0.99)),
                "gpu_s_per_query": gpu_s_per_query,
                "decode_tok_per_s_mean": float(np.mean(tps_vals)),
                "effective_tok_per_s_mean": float(np.mean(eff_tps_vals)),
                "peak_vram_mb": float(df["peak_vram_mb"].max()) if "peak_vram_mb" in df else 0.0,
                "energy_j_mean": float(df["energy_j"].mean()) if "energy_j" in df else 0.0,
                "success_mean": succ_mean,
                "success_ci_low": succ_lo,
                "success_ci_high": succ_hi,
                "toxic_rate": tox_mean,
                "profanity_rate": prof_mean,
                "over_refusal_rate": over_ref_mean,
                "slang_fpr": slang_fpr,
                "obfuscated_success": obf_succ,
                **det_stats,
                "mmlu_acc": mmlu_acc,
                "gsm8k_acc": gsm8k_acc,
                "fk_grade_mean": fk_mean,
                "fk_compliant_rate": fk_comp_mean,
                "bertscore_f1_mean": bert_mean,
                "sari_mean": sari_mean,
                "ppl_delta_pct_mean": ppl_delta_mean,
                "repetition_rate_mean": rep_mean,
                "judge_age_appropriate_mean": judge_age_mean,
            }
        )

    summary_df = _add_post_hoc_columns(pd.DataFrame(rows))
    if not summary_df.empty:
        summary_df.to_parquet(out_dir / "summary.parquet", index=False)
        summary_df.to_csv(out_dir / "cost_table.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "cost_table.csv", index=False)
    return summary_df
