"""Generate executive summary report (results/REPORT.md)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from minorsafe.analysis.tables import _df_to_markdown, evaluate_hypotheses


def generate_executive_report(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    """Create results/REPORT.md with benchmark summary, raw token/time accounting, and H1-H10 verdicts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT.md"

    verdicts_df = pd.DataFrame(evaluate_hypotheses(summary_df))
    cols = [
        c
        for c in (
            "method",
            "model",
            "setup_wall_s",
            "setup_tokens",
            "tok_total_mean",
            "e2e_ms_p50",
            "decode_tok_per_s_mean",
            "success_mean",
        )
        if c in summary_df.columns
    ]
    bench_md = _df_to_markdown(summary_df[cols] if cols else summary_df)
    hyp_md = _df_to_markdown(verdicts_df)

    content = f"""# MinorSafe Executive Benchmark Report (Age 12–18)

## 1. Cost & Accounting Policy
All cost metrics are reported strictly in **raw token counts (`tok_total_mean`, `setup_tokens`)** and **wall-clock / GPU execution time (`e2e_ms_p50`, `decode_tok_per_s_mean`, `setup_wall_s`, `setup_gpu_s`)** so you can apply your own external pricing formulas.

## 2. Main Benchmark Summary (`teen_12_18`)
{bench_md}

## 3. Pre-Registered Hypothesis Verdicts (`H1`–`H10`)
{hyp_md}
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path

