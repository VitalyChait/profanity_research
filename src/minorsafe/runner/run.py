"""Experiment runner and end-to-end CPU smoke pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from minorsafe.config import (
    load_age_bands,
    load_data_spec,
    load_experiment,
    load_hardware,
    load_models,
    load_prompts,
)
from minorsafe.data.build import build_dataset
from minorsafe.data.schema import Example, load_examples, load_pairs
from minorsafe.eval.evaluate import evaluate_responses_df
from minorsafe.eval.judge import export_human_annotation_sheet
from minorsafe.methods.base import MethodOutput, RunContext, create_method
from minorsafe.runner.grid import RunSpec, expand_experiment_grid, verify_split_guardrail
from minorsafe.runner.store import RunStore
from minorsafe.utils import DATA_DIR, RESULTS_DIR, write_json


def _output_to_row(
    out: MethodOutput,
    rspec: RunSpec,
    gen_spec,
    seed: int,
    ex: Example,
) -> dict[str, Any]:
    tok = out.tokens.to_dict()
    cat = ex.categories[0] if ex.categories else "benign"
    return {
        "run_id": rspec.run_id,
        "experiment": rspec.experiment,
        "method": rspec.label or rspec.method_spec.name,
        "method_kind": rspec.method_spec.kind,
        "model": gen_spec.alias,
        "family": gen_spec.family,
        "size_group": gen_spec.size_group or "S",
        "params_b": float(gen_spec.params_b or 0.1),
        "quant": rspec.quant,
        "age_band": rspec.age_band,
        "seed": seed,
        "example_id": ex.id,
        "category": cat,
        "prompt": ex.last_user,
        "reference_safe": ex.reference_safe,
        "label_unsafe": bool(ex.label_unsafe),
        "text": out.text,
        "raw_text": out.raw_text,
        "blocked": out.blocked,
        "action_taken": out.action_taken,
        "guard_score": out.guard_score if out.guard_score is not None else 0.0,
        **tok,
        "ttft_ms": out.ttft_ms,
        "e2e_ms": out.e2e_ms,
        "decode_tok_per_s": out.decode_tok_per_s,
        "effective_tok_per_s": out.effective_tok_per_s,
        "energy_j": out.energy_j,
        "peak_vram_mb": out.peak_vram_mb,
    }


def run_experiment(
    experiment_name: str,
    profile_name: str = "cpu_debug",
    overrides: list[str] | None = None,
    final: bool = False,
    runs_root: Path | None = None,
    data_dir: Path | None = None,
    preloaded_generator: tuple[Any, Any] | None = None,
) -> list[Path]:
    """Execute an experiment grid, checkpointing shards and evaluating responses."""
    models = load_models()
    profile = load_hardware(profile_name)
    exp = load_experiment(experiment_name, overrides=overrides)
    verify_split_guardrail(exp, final=final)

    age_bands = load_age_bands()
    prompts = load_prompts()

    base_data_dir = Path(data_dir) if data_dir else DATA_DIR
    proc_dir = base_data_dir / "processed"
    if not (proc_dir / "examples.parquet").exists():
        build_dataset(load_data_spec(), data_dir=base_data_dir, results_dir=RESULTS_DIR, offline=True)

    all_examples = load_examples(proc_dir / "examples.parquet")
    all_pairs = load_pairs(proc_dir / "pairs.parquet")

    split_examples = [e for e in all_examples if e.split == exp.split]
    if exp.categories:
        cat_set = set(exp.categories)
        split_examples = [e for e in split_examples if any(c in cat_set for c in e.categories)]
    if exp.max_examples:
        split_examples = split_examples[: exp.max_examples]

    calib_ex = [e for e in all_examples if e.split == "calib"]
    dev_ex = [e for e in all_examples if e.split == "dev"]
    calib_pairs = [p for p in all_pairs if p.split == "calib"]

    specs = expand_experiment_grid(exp, profile, models)
    exp_runs_dir = (runs_root or (RESULTS_DIR / "runs")) / exp.name
    completed_dirs: list[Path] = []

    for rspec in specs:
        run_dir = exp_runs_dir / rspec.run_id
        store = RunStore(run_dir)
        done = store.completed_keys()

        age_band = age_bands[rspec.age_band]
        gen_spec = models[rspec.generator_alias]
        ctx = RunContext(
            hardware=profile,
            generator_spec=gen_spec,
            models=models,
            age_bands=age_bands,
            prompts=prompts,
            calib_pairs=calib_pairs,
            calib_examples=calib_ex,
            dev_examples=dev_ex,
            quant=rspec.quant,
            artifact_dir=exp_runs_dir / "_artifacts",
            preloaded_generator=preloaded_generator,
        )
        method = create_method(rspec.method_spec)
        setup_rec = method.setup(ctx)
        write_json(run_dir / "setup.json", setup_rec.to_dict())

        for seed in exp.seeds:
            todo = [ex for ex in split_examples if (ex.id, seed) not in done]
            if not todo:
                continue
            outputs = method.respond(todo, age_band=rspec.age_band, seed=seed)
            rows = [
                _output_to_row(out, rspec, gen_spec, seed, ex) for ex, out in zip(todo, outputs, strict=True)
            ]
            store.append_responses(rows)

        method.teardown()

        if exp.evaluate:
            df = store.load_responses()
            eval_df = evaluate_responses_df(
                df,
                eval_settings=profile.eval,
                target_fk_grade=age_band.target_fk_grade,
            )
            eval_df.to_parquet(store.responses_path, index=False)

        completed_dirs.append(run_dir)

    return completed_dirs


def run_smoke_pipeline(work_dir: Path, max_examples: int = 6) -> pd.DataFrame:
    """Fast, offline CPU smoke pipeline exercising data -> methods -> eval -> analysis."""
    from minorsafe.testing import DummyCausalLM, DummyTokenizer

    from minorsafe.analysis.aggregate import aggregate_experiment_results
    from minorsafe.analysis.plots import generate_all_figures
    from minorsafe.analysis.report import generate_executive_report
    from minorsafe.analysis.tables import generate_all_tables

    work_dir = Path(work_dir)
    data_dir = work_dir / "data"
    runs_dir = work_dir / "runs"
    results_dir = work_dir / "results"

    spec = load_data_spec(overrides=["synth_pairs_per_concept=8", "max_minimal_pairs=16"])
    for s in spec.sources:
        s.max_items = 12
    build_dataset(spec, data_dir=data_dir, results_dir=results_dir, offline=True)

    lm = DummyCausalLM(hidden_size=16, num_layers=3)
    tok = DummyTokenizer()

    run_dirs = run_experiment(
        "smoke",
        profile_name="cpu_debug",
        overrides=[f"max_examples={max_examples}"],
        runs_root=runs_dir,
        data_dir=data_dir,
        preloaded_generator=(lm, tok),
    )

    all_dfs = [
        pd.read_parquet(d / "responses.parquet") for d in run_dirs if (d / "responses.parquet").exists()
    ]
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    export_human_annotation_sheet(combined_df, n=150, out_path=results_dir / "human_eval_150.csv")

    summary_df = aggregate_experiment_results(runs_dir / "smoke", results_dir)
    generate_all_figures(summary_df, results_dir / "figures")
    generate_all_tables(summary_df, results_dir / "tables")
    generate_executive_report(summary_df, results_dir)
    return summary_df
