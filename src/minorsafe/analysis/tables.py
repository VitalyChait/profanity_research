"""Generate Markdown and LaTeX benchmark and hypothesis verdict tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from minorsafe.analysis.hypotheses import evaluate_hypotheses

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in str(text))


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
                vals.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    col_spec = "l" * len(cols)
    header = " & ".join(_latex_escape(c) for c in cols) + " \\\\"
    body_lines = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(_latex_escape(v))
        body_lines.append(" & ".join(vals) + " \\\\")
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
        f"{header}\n\\hline\n{body}\n\\hline\n"
        "\\end{tabular}\n\\end{table}\n"
    )


def generate_all_tables(
    summary_df: pd.DataFrame, tables_dir: Path, responses_df: pd.DataFrame | None = None
) -> list[Path]:
    """Write Markdown and LaTeX tables for main benchmark and hypothesis verdicts.

    ``responses_df`` (all per-response rows) enables the paired per-example tests (e.g. H2).
    """
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
        _df_to_latex(sub_df, "Main Benchmark Results (Age 12--18)", "tab:main"), encoding="utf-8"
    )
    paths.extend([md_main, tex_main])

    verdicts_df = pd.DataFrame(evaluate_hypotheses(summary_df, responses_df=responses_df))
    md_hyp = tables_dir / "hypothesis_verdicts.md"
    tex_hyp = tables_dir / "hypothesis_verdicts.tex"
    md_hyp.write_text(_df_to_markdown(verdicts_df), encoding="utf-8")
    tex_hyp.write_text(
        _df_to_latex(verdicts_df, "Hypothesis Verdicts H1-H10", "tab:hypotheses"), encoding="utf-8"
    )
    paths.extend([md_hyp, tex_hyp])

    return paths
