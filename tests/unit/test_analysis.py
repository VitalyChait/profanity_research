"""Phase 9 unit tests: clustered bootstrap CIs, Wilcoxon + Holm correction, cost_table.csv, plots, and report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from minorsafe.analysis.aggregate import aggregate_experiment_results
from minorsafe.analysis.plots import generate_all_figures
from minorsafe.analysis.report import generate_executive_report
from minorsafe.analysis.stats import clustered_bootstrap_ci, holm_bonferroni, paired_significance_test
from minorsafe.analysis.tables import generate_all_tables

pytestmark = pytest.mark.unit


def test_clustered_bootstrap_ci_and_holm_correction():
    rng = np.random.default_rng(123)
    values = rng.normal(loc=0.80, scale=0.05, size=200)
    clusters = [f"ex_{i // 2}" for i in range(200)]
    mean, lo, hi = clustered_bootstrap_ci(values, clusters=clusters, n_boot=300, seed=13)
    assert lo < mean < hi
    assert 0.75 < mean < 0.85

    a = np.ones(50) * 0.90
    b = np.ones(50) * 0.50
    res = paired_significance_test(a, b, seed=13)
    assert res["diff_mean"] == pytest.approx(0.40)
    assert res["p_value"] < 0.01

    adj = holm_bonferroni([0.001, 0.02, 0.04])
    assert len(adj) == 3
    assert adj[0] <= adj[1] <= adj[2]


def test_aggregate_plots_tables_and_report_pipeline(tmp_path: Path):
    # Create synthetic run shards for M0a, M0b, M1-Enc, M1-Dec, M2-LoRA, M3-StTP, M3-Ortho
    runs_dir = tmp_path / "runs" / "smoke"
    methods = [
        ("m0a_raw", "baseline", 0.0, 0, 120, 45.0, 85.0, 0.55),
        ("m0b_sysprompt", "baseline", 0.0, 0, 165, 52.0, 80.0, 0.72),
        ("m1_encoder_guard", "guard", 180.0, 45000, 150, 62.0, 72.0, 0.86),
        ("m1_decoder_guard", "guard", 0.0, 0, 310, 420.0, 18.0, 0.85),
        ("m2_lora", "lora", 1800.0, 450000, 122, 46.0, 84.0, 0.88),
        ("m3_steer_swfc", "steering", 18.0, 12000, 125, 47.0, 82.0, 0.74),
        ("m3_steer_sttp", "steering", 22.0, 12000, 121, 46.5, 83.5, 0.91),
        ("m3_steer_ortho", "steering", 25.0, 15000, 121, 46.8, 83.2, 0.93),
    ]
    for m_name, kind, setup_s, setup_tok, tok_tot, e2e_ms, tok_s, succ_rate in methods:
        rdir = runs_dir / f"{m_name}_qwen25_05b"
        rdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(20):
            succ = i < int(20 * succ_rate)
            rows.append(
                {
                    "run_id": f"{m_name}_qwen25_05b",
                    "experiment": "smoke",
                    "method": m_name,
                    "method_kind": kind,
                    "model": "qwen25_05b",
                    "family": "qwen",
                    "size_group": "S",
                    "params_b": 0.49,
                    "quant": "bf16",
                    "age_band": "teen_12_18",
                    "seed": 0,
                    "example_id": f"ex_{i}",
                    "category": "obfuscated" if i % 4 == 0 else ("slang" if i % 4 == 1 else "profanity"),
                    "label_unsafe": bool(i % 2 == 0),
                    "blocked": False,
                    "action_taken": "steered" if kind == "steering" else "none",
                    "guard_score": 0.85 if (i % 2 == 0) else 0.12,
                    "tok_generator_in": 40,
                    "tok_generator_out": 80,
                    "tok_generator_reasoning_out": 0,
                    "tok_guard_in": 40 if kind == "guard" and "decoder" in m_name else 0,
                    "tok_guard_out": 6 if kind == "guard" and "decoder" in m_name else 0,
                    "tok_rewrite_in": 0,
                    "tok_rewrite_out": 0,
                    "tok_regen_in": 0,
                    "tok_regen_out": 0,
                    "tok_encoder_in": 40 if "encoder" in m_name else 0,
                    "tok_input_total": 40,
                    "tok_output_total": tok_tot - 40,
                    "tok_total": tok_tot,
                    "ttft_ms": 8.0,
                    "e2e_ms": e2e_ms,
                    "decode_tok_per_s": tok_s,
                    "effective_tok_per_s": tok_s * 0.9,
                    "energy_j": 1.2,
                    "peak_vram_mb": 1100.0,
                    "toxicity_score": 0.05 if succ else 0.72,
                    "toxic_flag": not succ,
                    "profanity_count": 0 if succ else 1,
                    "has_profanity": not succ,
                    "is_refusal": bool(i % 2 == 0 and succ),
                    "over_refusal": bool(i % 2 == 1 and not succ and m_name != "m3_steer_ortho"),
                    "fk_grade": 6.8 if succ else 11.2,
                    "dale_chall": 6.5,
                    "rare_word_ratio": 0.08,
                    "fk_compliant": succ,
                    "bertscore_f1_ref": 0.92 if succ else 0.75,
                    "bertscore_f1_m0": 0.94,
                    "sari_score": 56.0 if succ else 38.0,
                    "recompose_filter_acc": succ,
                    "ppl": 18.0 if m_name != "m3_steer_swfc" else 28.0,
                    "ppl_delta_pct": 1.5 if m_name != "m3_steer_swfc" else 34.0,
                    "distinct_1": 0.85,
                    "distinct_2": 0.92,
                    "repetition_rate": 0.02 if m_name != "m3_steer_swfc" else 0.22,
                    "judge_age_appropriate": 5 if succ else 2,
                    "judge_meaning_preserved": 4,
                    "judge_helpful": 4,
                    "judge_fluent": 5 if m_name != "m3_steer_swfc" else 2,
                    "judge_is_refusal": bool(i % 2 == 0 and succ),
                    "mmlu_acc": 0.62 if kind != "lora" else 0.57,
                    "gsm8k_acc": 0.54 if kind != "lora" else 0.49,
                    "success": succ,
                }
            )
        pd.DataFrame(rows).to_parquet(rdir / "responses.parquet", index=False)
        from minorsafe.utils import write_json

        write_json(
            rdir / "setup.json",
            {
                "method": m_name,
                "setup_key": m_name,
                "stages": [
                    {
                        "name": "setup",
                        "wall_s": setup_s,
                        "gpu_s": setup_s,
                        "tokens": setup_tok,
                        "examples": 100,
                        "peak_vram_mb": 1200.0,
                    }
                ],
                "total_wall_s": setup_s,
                "total_gpu_s": setup_s,
                "total_tokens": setup_tok,
            },
        )

    out_dir = tmp_path / "results"
    summary_df = aggregate_experiment_results(tmp_path / "runs", out_dir)
    assert len(summary_df) == len(methods)
    cost_csv = out_dir / "cost_table.csv"
    assert cost_csv.exists()
    # Verify required raw token & time columns exist and NO hardcoded dollar price column exists
    for col in (
        "setup_wall_s",
        "setup_gpu_s",
        "setup_tokens",
        "tok_total_mean",
        "e2e_ms_p50",
        "decode_tok_per_s_mean",
        "success_mean",
    ):
        assert col in summary_df.columns
    assert not any("$" in c or "usd" in c.lower() for c in summary_df.columns)

    figs = generate_all_figures(summary_df, out_dir / "figures")
    assert len(figs) == 8
    for f in figs:
        assert f.exists()

    tables = generate_all_tables(summary_df, out_dir / "tables")
    assert len(tables) >= 2
    assert (out_dir / "tables" / "main_benchmark.md").exists()
    assert (out_dir / "tables" / "main_benchmark.tex").exists()
    assert (out_dir / "tables" / "hypothesis_verdicts.md").exists()

    rep_path = generate_executive_report(summary_df, out_dir)
    assert rep_path.exists()
    assert "H1" in rep_path.read_text()
