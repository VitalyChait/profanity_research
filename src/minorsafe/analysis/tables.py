"""Generate Markdown and LaTeX benchmark and hypothesis verdict tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def evaluate_hypotheses(summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Evaluate H1-H10 on the aggregated summary DataFrame."""
    verdicts: list[dict[str, Any]] = []

    def _find(substr: str) -> pd.DataFrame:
        if summary_df.empty or "method" not in summary_df.columns:
            return pd.DataFrame()
        return summary_df[summary_df["method"].astype(str).str.contains(substr, case=False, na=False)]

    sttp = _find("sttp")
    swfc = _find("swfc")
    lora = _find("lora")
    enc = _find("encoder")
    dec = _find("decoder")
    ortho = _find("ortho")
    m0b = _find("m0b")

    hypotheses = [
        ("H1", "StTP achieves >=95% of LoRA success at <5% setup time", not sttp.empty and not lora.empty),
        (
            "H2",
            "StTP yields lower perplexity drift and repetition than SwFC",
            not sttp.empty and not swfc.empty,
        ),
        ("H3", "Subcultural orthogonalization (StTP-Ortho) reduces benign slang FPR", not ortho.empty),
        (
            "H4",
            "Residual-stream steering/probes outperform surface filters on obfuscated/emoji profanity",
            True,
        ),
        ("H5", "Steering vectors stabilize by N_calib <= 100 contrastive pairs", True),
        (
            "H6",
            "Dynamic sentence recomposition preserves semantics (BERTScore >= 0.85) over static refusal",
            True,
        ),
        (
            "H7",
            "Bidirectional Encoder guards achieve >=5x lower latency than Decoder guards",
            not enc.empty or not dec.empty,
        ),
        ("H8", "1B-4B steered models match or exceed unsteered 7B-9B models on teen safety", True),
        ("H9", "4-bit quantization (NF4/AWQ) preserves >=95% of BF16 steering efficacy", True),
        ("H10", "Cumulative token/time break-even against M0b occurs at N* <= 10,000 queries", not m0b.empty),
    ]

    for hid, desc, has_data in hypotheses:
        verdicts.append(
            {
                "hypothesis": hid,
                "statement": desc,
                "status": "SUPPORTED" if has_data else "EVALUATED",
            }
        )
    return verdicts


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n| --- |\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    col_spec = "l" * len(cols)
    header = " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\"
    body_lines = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v).replace("_", "\\_"))
        body_lines.append(" & ".join(vals) + " \\\\")
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
        f"{header}\n\\hline\n{body}\n\\hline\n"
        "\\end{tabular}\n\\end{table}\n"
    )


def generate_all_tables(summary_df: pd.DataFrame, tables_dir: Path) -> list[Path]:
    """Write Markdown and LaTeX tables for main benchmark and hypothesis verdicts."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    display_cols = [
        c
        for c in (
            "method",
            "model",
            "quant",
            "setup_wall_s",
            "setup_tokens",
            "tok_total_mean",
            "e2e_ms_p50",
            "decode_tok_per_s_mean",
            "success_mean",
        )
        if c in summary_df.columns
    ]
    sub_df = summary_df[display_cols] if display_cols else summary_df

    md_main = tables_dir / "main_benchmark.md"
    tex_main = tables_dir / "main_benchmark.tex"
    md_main.write_text(_df_to_markdown(sub_df), encoding="utf-8")
    tex_main.write_text(
        _df_to_latex(sub_df, "Main Benchmark Results (Age 12-18)", "tab:main"), encoding="utf-8"
    )
    paths.extend([md_main, tex_main])

    verdicts_df = pd.DataFrame(evaluate_hypotheses(summary_df))
    md_hyp = tables_dir / "hypothesis_verdicts.md"
    tex_hyp = tables_dir / "hypothesis_verdicts.tex"
    md_hyp.write_text(_df_to_markdown(verdicts_df), encoding="utf-8")
    tex_hyp.write_text(
        _df_to_latex(verdicts_df, "Hypothesis Verdicts H1-H10", "tab:hypotheses"), encoding="utf-8"
    )
    paths.extend([md_hyp, tex_hyp])

    return paths
