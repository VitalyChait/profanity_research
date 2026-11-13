"""Typer CLI entry point for MinorSafe (minorsafe)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from minorsafe.analysis.aggregate import aggregate_experiment_results
from minorsafe.analysis.plots import generate_all_figures
from minorsafe.analysis.report import generate_executive_report
from minorsafe.analysis.tables import generate_all_tables
from minorsafe.config import load_age_bands, load_data_spec, load_hardware, load_models
from minorsafe.data.build import build_dataset
from minorsafe.eval.evaluate import evaluate_responses_df
from minorsafe.eval.judge import compute_human_agreement, export_human_annotation_sheet
from minorsafe.hardware import gpu_summary
from minorsafe.runner.run import run_experiment, run_smoke_pipeline
from minorsafe.utils import DATA_DIR, RESULTS_DIR, write_json

app = typer.Typer(
    name="minorsafe",
    help="Cost-Effective Age-Appropriate LLM Response Generation & Safety Steering for Minors (12-18).",
    no_args_is_help=True,
)
judge_app = typer.Typer(help="LLM Judge evaluation and 150-response human annotation tools.")
app.add_typer(judge_app, name="judge")


@app.command("info")
def cmd_info(profile: str = typer.Option("cpu_debug", "--profile", "-p")) -> None:
    """Print hardware profile, detected GPUs, and environment package versions."""
    hw = load_hardware(profile)
    info = gpu_summary()
    typer.echo(f"Hardware Profile: {hw.name} ({hw.device})")
    typer.echo(f"Detected Hardware: {info}")


@app.command("data")
def cmd_data(
    override: list[str] = typer.Option(None, "--override", "-o"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Build evaluation examples, contrastive pairs, obfuscated variants, and results/data_card.md."""
    spec = load_data_spec(overrides=override)
    examples, pairs, card_path = build_dataset(
        spec, data_dir=DATA_DIR, results_dir=RESULTS_DIR, offline=offline
    )
    typer.echo(f"Built {len(examples)} examples and {len(pairs)} pairs. Data card: {card_path}")


@app.command("run")
def cmd_run(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    profile: str = typer.Option("cpu_debug", "--profile", "-p"),
    override: list[str] = typer.Option(None, "--override", "-o"),
    final: bool = typer.Option(
        False, "--final", help="Required guardrail flag when running on 'test' split."
    ),
) -> None:
    """Execute an experiment grid with atomic Parquet checkpointing."""
    dirs = run_experiment(experiment_name=experiment, profile_name=profile, overrides=override, final=final)
    typer.echo(f"Completed {len(dirs)} run shards for experiment '{experiment}'.")


@app.command("eval")
def cmd_eval(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    profile: str = typer.Option("cpu_debug", "--profile", "-p"),
) -> None:
    """Re-evaluate saved responses.parquet shards for an experiment."""
    hw = load_hardware(profile)
    age_band = load_age_bands()["teen_12_18"]
    exp_dir = RESULTS_DIR / "runs" / experiment
    for pq in sorted(exp_dir.rglob("responses.parquet")):
        df = pd.read_parquet(pq)
        eval_df = evaluate_responses_df(df, eval_settings=hw.eval, target_fk_grade=age_band.target_fk_grade)
        eval_df.to_parquet(pq, index=False)
        typer.echo(f"Evaluated {pq} ({len(eval_df)} rows)")


@judge_app.command("export-human")
def cmd_judge_export(
    n: int = typer.Option(150, "--n", help="Fixed human annotation sample size (Decision 6: 150)."),
    out: Path = typer.Option(RESULTS_DIR / "human_eval_150.csv", "--out"),
) -> None:
    """Export a stratified 150-response CSV for blinded human annotation."""
    pqs = sorted((RESULTS_DIR / "runs").rglob("responses.parquet"))
    dfs = [pd.read_parquet(p) for p in pqs]
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    export_human_annotation_sheet(combined, n=n, out_path=out)
    typer.echo(f"Exported {n} human evaluation samples to {out}")


@judge_app.command("agreement")
def cmd_judge_agreement(
    csv_path: Path = typer.Option(RESULTS_DIR / "human_eval_150_labeled.csv", "--csv"),
    out: Path = typer.Option(RESULTS_DIR / "human_judge_agreement.json", "--out"),
) -> None:
    """Compute quadratic-weighted Cohen's kappa and Spearman's rho on labeled human_eval_150 CSV."""
    res = compute_human_agreement(csv_path)
    write_json(out, res)
    typer.echo(f"Agreement results written to {out}: {res}")


@app.command("tomography")
def cmd_tomography(
    profile: str = typer.Option("cpu_debug", "--profile", "-p"),
) -> None:
    """Run Latent Artificial Tomography (LAT) layer scans and cross-concept cosine matrix."""
    models = load_models()
    hw = load_hardware(profile)
    typer.echo(f"Tomography ready for generators {hw.generators} ({len(models)} configured models).")


@app.command("status")
def cmd_status() -> None:
    """Inspect completed and pending experiment shards under results/runs/."""
    runs_dir = RESULTS_DIR / "runs"
    pqs = sorted(runs_dir.rglob("responses.parquet")) if runs_dir.exists() else []
    typer.echo(f"Found {len(pqs)} completed run shards under {runs_dir}.")


@app.command("analyze")
def cmd_analyze(
    runs_dir: Path = typer.Option(RESULTS_DIR / "runs", "--runs-dir"),
    out_dir: Path = typer.Option(RESULTS_DIR, "--out-dir"),
) -> None:
    """Aggregate all run shards into cost_table.csv, 8 figures, LaTeX/MD tables, and REPORT.md."""
    summary_df = aggregate_experiment_results(runs_dir, out_dir)
    generate_all_figures(summary_df, out_dir / "figures")
    generate_all_tables(summary_df, out_dir / "tables")
    rep = generate_executive_report(summary_df, out_dir)
    typer.echo(f"Analysis complete. Report written to {rep}")


@app.command("smoke")
def cmd_smoke(
    work_dir: Path = typer.Option(RESULTS_DIR / "smoke_workspace", "--work-dir"),
    max_examples: int = typer.Option(6, "--max-examples"),
) -> None:
    """Run the self-contained end-to-end CPU smoke pipeline."""
    summary = run_smoke_pipeline(work_dir=work_dir, max_examples=max_examples)
    typer.echo(f"Smoke pipeline finished ({len(summary)} runs summarized in {work_dir / 'results'}).")


if __name__ == "__main__":
    app()
