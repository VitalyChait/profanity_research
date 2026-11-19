"""Typer CLI entry point for MinorSafe (minorsafe)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from minorsafe.analysis.aggregate import aggregate_experiment_results
from minorsafe.analysis.plots import generate_all_figures
from minorsafe.analysis.report import generate_executive_report
from minorsafe.analysis.tables import generate_all_tables
from minorsafe.config import load_age_bands, load_data_spec, load_hardware
from minorsafe.data.build import build_dataset
from minorsafe.eval.evaluate import evaluate_responses_df
from minorsafe.eval.judge import compute_human_agreement, export_human_annotation_sheet
from minorsafe.hardware import gpu_summary, resolve_profile_name, run_preflight_checks
from minorsafe.runner.run import run_experiment, run_smoke_pipeline
from minorsafe.utils import DATA_DIR, RESULTS_DIR, read_json, write_json

app = typer.Typer(
    name="minorsafe",
    help="Cost-Effective Age-Appropriate LLM Response Generation & Safety Steering for Minors (12-18).",
    no_args_is_help=True,
)
judge_app = typer.Typer(
    help="LLM Judge evaluation and 150-response human annotation tools.", no_args_is_help=True
)
data_app = typer.Typer(help="Dataset construction and data card.", no_args_is_help=True)
app.add_typer(judge_app, name="judge")
app.add_typer(data_app, name="data")

# One shared option definition so every command accepts the same flags (plan §0.2 uses --hardware).
_HW_HELP = "Hardware profile name in configs/hardware/, or 'auto' to detect from nvidia-smi."


def _hardware_option(default: str = "auto"):
    return typer.Option(default, "--hardware", "--profile", "-p", help=_HW_HELP)


def _load_all_responses(runs_dir: Path) -> pd.DataFrame:
    pqs = sorted(Path(runs_dir).rglob("responses.parquet")) if Path(runs_dir).exists() else []
    dfs = [pd.read_parquet(p) for p in pqs]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@app.command("info")
def cmd_info(hardware: str = _hardware_option()) -> None:
    """Print the resolved hardware profile and detected GPUs."""
    hw = load_hardware(resolve_profile_name(hardware))
    typer.echo(f"Hardware Profile: {hw.name} ({hw.device}); allow_placeholders={hw.allow_placeholders}")
    typer.echo(f"Detected Hardware: {gpu_summary()}")


@app.command("preflight")
def cmd_preflight(
    hardware: str = _hardware_option(),
    allow_busy: bool = typer.Option(
        False, "--allow-busy", help="Do not fail if another user or GPU process is active."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON preflight report."),
) -> None:
    """Run initial hardware, CUDA, server exclusivity, secret safety, disk, and HF auth checks."""
    rep = run_preflight_checks(profile_name=hardware, require_exclusive=not allow_busy)
    if as_json:
        typer.echo(json.dumps(rep, indent=2))
        raise typer.Exit(0 if rep["ok"] else 1)

    typer.echo(
        f"[PASS] Profile: {rep['profile']} (device={rep['device']}, "
        f"allow_placeholders={rep['allow_placeholders']})"
    )
    cuda = rep["cuda"]
    gpus = rep["gpus"]
    typer.echo(
        f"[{'PASS' if (rep['device'] == 'cpu' or cuda['available']) else 'FAIL'}] "
        f"CUDA: available={cuda['available']} gpus={len(gpus)} bf16={cuda['bf16_supported']}"
    )
    for g in gpus:
        typer.echo(
            f"       GPU {g['index']}: {g['name']} | mem={g['memory_used_mb']}/{g['memory_total_mb']}MB "
            f"| util={g['utilization_pct']}% | temp={g['temperature_c']}C"
        )
    excl = rep["exclusivity"]
    excl_tag = "PASS" if excl["exclusive"] else ("WARN" if allow_busy else "FAIL")
    typer.echo(
        f"[{excl_tag}] Exclusivity: exclusive={excl['exclusive']} "
        f"(gpu_procs={len(excl['gpu_processes'])}, other_users={excl['other_logged_in_users']})"
    )
    cred = rep["credential_safety"]
    typer.echo(
        f"[{'PASS' if cred['ok'] else 'FAIL'}] Secrets: .remote.env exists={cred['env_exists']} "
        f"gitignored={cred['gitignored']} tracked={cred['tracked']} mode={cred['mode_octal']}"
    )
    disk = rep["disk"]
    typer.echo(
        f"[PASS] Disk free: repo={disk['repo_free_gb']}GB, HF_HOME={disk['hf_free_gb']}GB ({disk['hf_home']})"
    )
    hf = rep["hf_auth"]
    hf_tag = "PASS" if (hf["token_present"] or not hf["gated_models"]) else "WARN"
    typer.echo(
        f"[{hf_tag}] HF Auth: token_present={hf['token_present']} gated_models={len(hf['gated_models'])}"
    )
    for w in rep["warnings"]:
        typer.echo(f"[WARN] {w}")
    for e in rep["errors"]:
        typer.echo(f"[FAIL] {e}", err=True)

    if rep["ok"]:
        typer.echo("Preflight PASSED.")
        raise typer.Exit(0)
    typer.echo(f"Preflight FAILED ({len(rep['errors'])} error(s)).", err=True)
    raise typer.Exit(1)


@data_app.command("build")
def cmd_data_build(
    override: list[str] = typer.Option(None, "--override", "-o"),
    offline: bool = typer.Option(
        False, "--offline", help="Use only the built-in seed corpus (no HF downloads)."
    ),
) -> None:
    """Build evaluation examples, contrastive pairs, obfuscated variants, and results/data_card.md."""
    spec = load_data_spec(overrides=override)
    examples, pairs, card_path = build_dataset(
        spec, data_dir=DATA_DIR, results_dir=RESULTS_DIR, offline=offline
    )
    typer.echo(f"Built {len(examples)} examples and {len(pairs)} pairs. Data card: {card_path}")


@data_app.command("report")
def cmd_data_report() -> None:
    """Print results/data_card.md."""
    card = RESULTS_DIR / "data_card.md"
    if not card.exists():
        typer.echo(f"No data card at {card}; run `minorsafe data build` first.", err=True)
        raise typer.Exit(1)
    typer.echo(card.read_text(encoding="utf-8"))


@app.command("run")
def cmd_run(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    hardware: str = _hardware_option(),
    override: list[str] = typer.Option(None, "--override", "-o"),
    final: bool = typer.Option(
        False, "--final", help="Required guardrail flag when running on 'test' split."
    ),
) -> None:
    """Execute an experiment grid with atomic Parquet checkpointing."""
    profile = resolve_profile_name(hardware)
    dirs = run_experiment(experiment_name=experiment, profile_name=profile, overrides=override, final=final)
    typer.echo(f"Completed {len(dirs)} run shards for experiment '{experiment}' on profile '{profile}'.")


@app.command("eval")
def cmd_eval(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    hardware: str = _hardware_option(),
) -> None:
    """Re-evaluate saved responses.parquet shards for an experiment."""
    hw = load_hardware(resolve_profile_name(hardware))
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
    runs_dir: Path = typer.Option(RESULTS_DIR / "runs", "--runs", "--runs-dir"),
    out: Path = typer.Option(RESULTS_DIR / "human_eval_150.csv", "--out"),
) -> None:
    """Export a stratified 150-response CSV for blinded human annotation."""
    combined = _load_all_responses(runs_dir)
    export_human_annotation_sheet(combined, n=n, out_path=out)
    typer.echo(f"Exported up to {n} human evaluation samples to {out}")


@judge_app.command("agreement")
def cmd_judge_agreement(
    csv_path: Path = typer.Option(RESULTS_DIR / "human_eval_150_labeled.csv", "--human-csv", "--csv"),
    out: Path = typer.Option(RESULTS_DIR / "human_judge_agreement.json", "--out"),
) -> None:
    """Compute quadratic-weighted Cohen's kappa and Spearman's rho on labeled human_eval_150 CSV."""
    res = compute_human_agreement(csv_path)
    write_json(out, res)
    typer.echo(f"Agreement results written to {out}: {res}")


@app.command("tomography")
def cmd_tomography(hardware: str = _hardware_option()) -> None:
    """Latent tomography (layer AUROC, cross-concept cosine, logit lens). NOT YET WIRED."""
    typer.echo(
        "`minorsafe tomography` is not implemented yet: steering/tomography.py exists but is not wired into a "
        "pipeline that loads generators and writes tomography.json (needed for H5/H7, fig7).",
        err=True,
    )
    raise typer.Exit(2)


@app.command("status")
def cmd_status(runs_dir: Path = typer.Option(RESULTS_DIR / "runs", "--runs", "--runs-dir")) -> None:
    """Show live run progress (if any) and count completed run shards under results/runs/."""
    live_path = runs_dir.parent / "logs" / "live_status.json"
    if live_path.exists():
        try:
            st = read_json(live_path)
            typer.echo(
                f"[LIVE] experiment={st.get('experiment')} state={st.get('state')} "
                f"run={st.get('run_idx')}/{st.get('total_runs')} ({st.get('run_id')}) "
                f"progress={st.get('progress')} eta_s={st.get('eta_s')} updated={st.get('updated_at')}"
            )
        except Exception:
            pass
    if not runs_dir.exists():
        typer.echo(f"No runs under {runs_dir}.")
        return
    for exp_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        n = len(list(exp_dir.rglob("responses.parquet")))
        typer.echo(f"{exp_dir.name}: {n} run shards")


@app.command("analyze")
def cmd_analyze(
    runs_dir: Path = typer.Option(RESULTS_DIR / "runs", "--runs", "--runs-dir"),
    out_dir: Path = typer.Option(RESULTS_DIR, "--out", "--out-dir"),
) -> None:
    """Aggregate all run shards into cost_table.csv, 8 figures, LaTeX/MD tables, and REPORT.md."""
    summary_df = aggregate_experiment_results(runs_dir, out_dir)
    responses_df = _load_all_responses(runs_dir)
    generate_all_figures(summary_df, out_dir / "figures")
    generate_all_tables(summary_df, out_dir / "tables", responses_df=responses_df)
    rep = generate_executive_report(summary_df, out_dir, responses_df=responses_df)
    typer.echo(f"Analysis complete. Report written to {rep}")


@app.command("smoke")
def cmd_smoke(
    hardware: str = _hardware_option("cpu_debug"),
    work_dir: Path = typer.Option(RESULTS_DIR / "smoke_workspace", "--work-dir"),
    max_examples: int = typer.Option(6, "--max-examples"),
) -> None:
    """Run the self-contained end-to-end CPU smoke pipeline (dummy models, placeholder evaluators)."""
    if hardware != "cpu_debug":
        typer.echo(
            "The CPU smoke pipeline uses dummy models and only supports --hardware cpu_debug. "
            "For a GPU smoke gate use: minorsafe run --experiment smoke --hardware <profile>",
            err=True,
        )
        raise typer.Exit(2)
    summary = run_smoke_pipeline(work_dir=work_dir, max_examples=max_examples)
    typer.echo(f"Smoke pipeline finished ({len(summary)} runs summarized in {work_dir / 'results'}).")


if __name__ == "__main__":
    app()
