"""Aggregate experiment shard responses and setup logs into summary.parquet and cost_table.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from minorsafe.analysis.stats import clustered_bootstrap_ci
from minorsafe.utils import read_json


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

        # Subcategory breakdowns
        obf_df = df[df["category"] == "obfuscated"] if "category" in df.columns else pd.DataFrame()
        slang_df = df[df["category"] == "slang"] if "category" in df.columns else pd.DataFrame()
        obf_succ = float(obf_df["success"].mean()) if not obf_df.empty and "success" in obf_df else succ_mean
        slang_fpr = (
            float(slang_df["over_refusal"].mean())
            if not slang_df.empty and "over_refusal" in slang_df
            else over_ref_mean
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

        rows.append(
            {
                "run_id": str(first.get("run_id", rdir.name)),
                "experiment": str(first.get("experiment", "unknown")),
                "method": str(first.get("method", "unknown")),
                "method_kind": str(first.get("method_kind", "unknown")),
                "model": str(first.get("model", "unknown")),
                "family": str(first.get("family", "unknown")),
                "size_group": str(first.get("size_group", "S")),
                "params_b": float(first.get("params_b", 0.0)),
                "quant": str(first.get("quant", "bf16")),
                "age_band": str(first.get("age_band", "teen_12_18")),
                "n_queries": int(len(df)),
                "setup_wall_s": float(setup_data.get("total_wall_s", 0.0)),
                "setup_gpu_s": float(setup_data.get("total_gpu_s", 0.0)),
                "setup_tokens": int(setup_data.get("total_tokens", 0)),
                "tok_input_mean": float(np.mean(tok_in_vals)),
                "tok_output_mean": float(np.mean(tok_out_vals)),
                "tok_total_mean": float(np.mean(tok_tot_vals)),
                "tok_total_p95": float(np.quantile(tok_tot_vals, 0.95)),
                "tok_total_sum": int(np.sum(tok_tot_vals)),
                "ttft_ms_p50": float(np.median(ttft_vals)),
                "e2e_ms_p50": float(np.median(e2e_vals)),
                "e2e_ms_p95": float(np.quantile(e2e_vals, 0.95)),
                "e2e_ms_p99": float(np.quantile(e2e_vals, 0.99)),
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
                "fk_grade_mean": fk_mean,
                "fk_compliant_rate": fk_comp_mean,
                "bertscore_f1_mean": bert_mean,
                "sari_mean": sari_mean,
                "ppl_delta_pct_mean": ppl_delta_mean,
                "repetition_rate_mean": rep_mean,
                "judge_age_appropriate_mean": judge_age_mean,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df.to_parquet(out_dir / "summary.parquet", index=False)
        summary_df.to_csv(out_dir / "cost_table.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "cost_table.csv", index=False)
    return summary_df
