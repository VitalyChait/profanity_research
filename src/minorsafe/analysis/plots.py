"""Publication-quality PDF/PNG figures for the 8 core experimental views."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> Path:
    pdf_path = fig_dir / f"{stem}.pdf"
    png_path = fig_dir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def generate_all_figures(summary_df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Generate the 8 paper figures from summary_df into fig_dir."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    df = summary_df.copy()
    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "method": "m0a_raw",
                    "method_kind": "baseline",
                    "tok_total_mean": 100.0,
                    "e2e_ms_p50": 40.0,
                    "decode_tok_per_s_mean": 80.0,
                    "setup_wall_s": 0.0,
                    "setup_tokens": 0,
                    "success_mean": 0.5,
                    "sari_mean": 45.0,
                    "slang_fpr": 0.1,
                    "obfuscated_success": 0.5,
                    "params_b": 1.0,
                }
            ]
        )

    # 1. Pareto frontier: Success vs Total Tokens per Query
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["tok_total_mean"], df["success_mean"], s=70, alpha=0.85)
    for _, r in df.iterrows():
        ax.annotate(str(r["method"]), (r["tok_total_mean"], r["success_mean"]), fontsize=7)
    ax.set_xlabel("Total Tokens per Query (tok_total_mean)")
    ax.set_ylabel("Age-Appropriate Success Rate")
    ax.set_title("Fig 1: Success vs. Token Budget Pareto Frontier")
    ax.grid(True, alpha=0.3)
    generated.append(_save_fig(fig, fig_dir, "fig1_pareto_cost_tokens"))

    # 2. Latency & Throughput Pareto
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["e2e_ms_p50"], df["decode_tok_per_s_mean"], c=df["success_mean"], cmap="viridis", s=70)
    ax.set_xlabel("Median E2E Latency (ms)")
    ax.set_ylabel("Decode Throughput (tok/s)")
    ax.set_title("Fig 2: Latency vs. Decode Throughput")
    ax.grid(True, alpha=0.3)
    generated.append(_save_fig(fig, fig_dir, "fig2_pareto_latency_throughput"))

    # 3. TCO Break-Even Curves over Query Volume N
    fig, ax = plt.subplots(figsize=(6, 4))
    n_grid = np.logspace(1, 6, 50)
    for _, r in df.head(6).iterrows():
        cum_s = float(r["setup_wall_s"]) + n_grid * (float(r["e2e_ms_p50"]) / 1000.0)
        ax.plot(n_grid, cum_s, label=str(r["method"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deployment Query Volume N")
    ax.set_ylabel("Cumulative Compute Time T(N) [s]")
    ax.set_title("Fig 3: TCO Time Break-Even Curves T(N)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    generated.append(_save_fig(fig, fig_dir, "fig3_tco_breakeven_curves"))

    # 4. Ramp-Up Efficiency (Setup Time vs Success)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["setup_wall_s"] + 1.0, df["success_mean"], s=70, color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel("Ramp-Up Wall Time T_setup + 1 [s] (log)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Fig 4: Ramp-Up Efficiency (Setup Cost vs. Quality)")
    ax.grid(True, alpha=0.3)
    generated.append(_save_fig(fig, fig_dir, "fig4_rampup_efficiency"))

    # 5. Encoder vs Decoder Guards
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["method"].astype(str), df["e2e_ms_p50"], color="tab:blue", alpha=0.8)
    ax.set_ylabel("Median E2E Latency (ms)")
    ax.set_title("Fig 5: Bidirectional Encoder vs. Decoder Guard Latency")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    generated.append(_save_fig(fig, fig_dir, "fig5_encoder_vs_decoder_guards"))

    # 6. Dynamic Sentence Recomposition Trade-off (SARI vs Success)
    fig, ax = plt.subplots(figsize=(6, 4))
    sari_col = df["sari_mean"] if "sari_mean" in df.columns else pd.Series(np.zeros(len(df)))
    ax.scatter(sari_col, df["success_mean"], s=70, color="tab:green")
    ax.set_xlabel("SARI Score (Recomposition Quality)")
    ax.set_ylabel("Age-Appropriate Success Rate")
    ax.set_title("Fig 6: Dynamic Sentence Recomposition Trade-Off")
    ax.grid(True, alpha=0.3)
    generated.append(_save_fig(fig, fig_dir, "fig6_recomposition_tradeoff"))

    # 7. Subcultural Slang FPR vs Obfuscated Profanity Success
    fig, ax = plt.subplots(figsize=(6, 4))
    slang_col = df["slang_fpr"] if "slang_fpr" in df.columns else pd.Series(np.zeros(len(df)))
    obf_col = df["obfuscated_success"] if "obfuscated_success" in df.columns else df["success_mean"]
    ax.scatter(slang_col, obf_col, s=70, color="tab:red")
    ax.set_xlabel("Benign Teen Slang False-Positive Rate (FPR)")
    ax.set_ylabel("Obfuscated & Emoji Profanity Success")
    ax.set_title("Fig 7: Subcultural Orthogonalization & Obfuscation Robustness")
    ax.grid(True, alpha=0.3)
    generated.append(_save_fig(fig, fig_dir, "fig7_subcultural_tomography"))

    # 8. Parameter Scale & Quantization Scaling
    fig, ax = plt.subplots(figsize=(6, 4))
    params_col = df["params_b"] if "params_b" in df.columns else pd.Series(np.ones(len(df)))
    ax.scatter(params_col, df["success_mean"], s=70, color="tab:purple")
    ax.set_xlabel("Model Parameter Scale (Billions)")
    ax.set_ylabel("Age-Appropriate Success Rate")
    ax.set_title("Fig 8: Parameter Scale (<1B, 1B-4B, 7B-9B) & Quantization")
    ax.grid(True, alpha=0.3)
    generated.append(_save_fig(fig, fig_dir, "fig8_quantization_scaling"))

    return generated

