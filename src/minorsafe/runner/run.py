"""Experiment runner and end-to-end CPU smoke pipeline."""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
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
from minorsafe.hardware import acquire_run_lock, check_server_exclusivity
from minorsafe.methods.base import MethodOutput, RunContext, create_method
from minorsafe.placeholders import require_real_backend
from minorsafe.runner.grid import RunSpec, expand_experiment_grid, verify_split_guardrail
from minorsafe.runner.store import RunStore
from minorsafe.telemetry import GPUSampler, SmartRunLogger
from minorsafe.utils import DATA_DIR, RESULTS_DIR, read_json, write_json


def _output_to_row(
    out: MethodOutput,
    rspec: RunSpec,
    gen_spec,
    seed: int,
    ex: Example,
) -> dict[str, Any]:
    tok = out.tokens.to_dict()
    cat = ex.categories[0] if ex.categories else "benign"
    row: dict[str, Any] = {
        "run_id": rspec.run_id,
        "experiment": rspec.experiment,
        "method": rspec.label or rspec.method_spec.name,
        "method_base": rspec.method_spec.name,
        "method_params": json.dumps(rspec.method_params, sort_keys=True, default=str),
        "method_kind": rspec.method_spec.kind,
        "model": gen_spec.alias,
        "family": gen_spec.family,
        "size_group": gen_spec.size_group or "S",
        "params_b": float(gen_spec.params_b) if gen_spec.params_b is not None else float("nan"),
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
        # NaN (not 0.0) when the method is not a detector, so it cannot leak into AUROC/F1.
        "guard_score": out.guard_score if out.guard_score is not None else float("nan"),
        **tok,
        "ttft_ms": out.ttft_ms,
        "e2e_ms": out.e2e_ms,
        "decode_tok_per_s": out.decode_tok_per_s,
        "effective_tok_per_s": out.effective_tok_per_s,
        "energy_j": out.energy_j,
        "peak_vram_mb": out.peak_vram_mb,
    }
    # Persist scalar method telemetry (hook intervention rate, probe score, ...) as x_* columns.
    for k, v in (out.extra or {}).items():
        if isinstance(v, (bool, int, float, str)):
            row[f"x_{k}"] = v
    return row


def run_experiment(
    experiment_name: str,
    profile_name: str = "cpu_debug",
    overrides: list[str] | None = None,
    final: bool = False,
    runs_root: Path | None = None,
    data_dir: Path | None = None,
    preloaded_generator: tuple[Any, Any] | None = None,
    require_exclusive: bool = True,
) -> list[Path]:
    """Execute an experiment grid, checkpointing shards and evaluating responses."""
    models = load_models()
    profile = load_hardware(profile_name)
    exp = load_experiment(experiment_name, overrides=overrides)
    verify_split_guardrail(exp, final=final)

    enforce_exclusive = (
        profile.device == "cuda" and require_exclusive and os.environ.get("MINORSAFE_ALLOW_BUSY", "0") != "1"
    )
    if enforce_exclusive:
        excl = check_server_exclusivity()
        if not excl.exclusive:
            raise RuntimeError(f"Server/GPU is not exclusive: {'; '.join(excl.reasons)}")

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

    if exp.capability_eval:
        # Capability retention (tinyMMLU / tinyGSM8K) is not wired into the runner yet; do not let the
        # config flag silently do nothing on profiles used for reported results.
        require_real_backend(
            profile, "capability_eval", "tinyMMLU/tinyGSM8K retention is not wired into the runner"
        )

    batch_size = max(1, int(exp.batch_size or profile.batch_size))
    log_dir = (runs_root.parent if runs_root else RESULTS_DIR) / "logs"
    smart_log = SmartRunLogger(
        experiment=exp.name,
        profile=profile.name,
        total_runs=len(specs),
        log_dir=log_dir,
    )
    smart_log.start_experiment(len(split_examples), exp.seeds)

    lock_ctx = (
        acquire_run_lock(phase=exp.name)
        if profile.device == "cuda" and runs_root is None
        else contextlib.nullcontext()
    )

    with lock_ctx:
        for run_idx, rspec in enumerate(specs, start=1):
            run_dir = exp_runs_dir / rspec.run_id
            store = RunStore(run_dir)
            done = store.completed_keys()

            age_band = age_bands[rspec.age_band]
            gen_spec = models[rspec.generator_alias]
            todo_by_seed = {s: [ex for ex in split_examples if (ex.id, s) not in done] for s in exp.seeds}
            pending = any(todo_by_seed.values())

            if pending:
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
                    temperature=exp.temperature,
                    max_new_tokens=exp.max_new_tokens,
                )
                method = create_method(rspec.method_spec)
                setup_rec = method.setup(ctx)
                write_json(run_dir / "setup.json", setup_rec.to_dict())
                smart_log.start_run(
                    run_idx=run_idx,
                    run_id=rspec.run_id,
                    method=rspec.label or rspec.method_spec.name,
                    model=gen_spec.alias,
                    quant=rspec.quant,
                    setup_wall_s=setup_rec.wall_s,
                    setup_tokens=setup_rec.tokens,
                )

                try:
                    # Discarded warmup queries (plan §2.2) so CUDA/kernel initialisation is not timed.
                    warm = split_examples[: max(0, exp.warmup_queries)]
                    if warm:
                        method.respond(warm, age_band=rspec.age_band, seed=exp.seeds[0] if exp.seeds else 0)

                    for seed, todo in todo_by_seed.items():
                        # Generate and checkpoint batch by batch, so a crash loses at most one batch.
                        for i in range(0, len(todo), batch_size):
                            chunk = todo[i : i + batch_size]
                            if profile.device == "cuda":
                                with GPUSampler() as gpu_s:
                                    outputs = method.respond(chunk, age_band=rspec.age_band, seed=seed)
                                per_item_energy = gpu_s.energy_j / max(1, len(outputs))
                                for out in outputs:
                                    if not math.isfinite(out.energy_j):
                                        out.energy_j = per_item_energy
                                    if not math.isfinite(out.peak_vram_mb):
                                        out.peak_vram_mb = gpu_s.peak_mem_mb
                            else:
                                outputs = method.respond(chunk, age_band=rspec.age_band, seed=seed)

                            rows = [
                                _output_to_row(out, rspec, gen_spec, seed, ex)
                                for ex, out in zip(chunk, outputs, strict=True)
                            ]
                            store.append_responses(rows)

                            finite_vram = [o.peak_vram_mb for o in outputs if math.isfinite(o.peak_vram_mb)]
                            finite_energy = [o.energy_j for o in outputs if math.isfinite(o.energy_j)]
                            smart_log.log_batch(
                                run_idx=run_idx,
                                run_id=rspec.run_id,
                                seed=seed,
                                batch_done=min(len(todo), i + len(chunk)),
                                total_todo=len(todo),
                                batch_e2e_ms=float(np.mean([o.e2e_ms for o in outputs])) if outputs else 0.0,
                                decode_tok_per_s=float(np.mean([o.decode_tok_per_s for o in outputs]))
                                if outputs
                                else 0.0,
                                peak_vram_mb=max(finite_vram) if finite_vram else float("nan"),
                                energy_j=sum(finite_energy) if finite_energy else float("nan"),
                            )
                finally:
                    method.teardown()
            else:
                setup_data = read_json(run_dir / "setup.json") if (run_dir / "setup.json").exists() else {}
                smart_log.start_run(
                    run_idx=run_idx,
                    run_id=rspec.run_id,
                    method=rspec.label or rspec.method_spec.name,
                    model=gen_spec.alias,
                    quant=rspec.quant,
                    setup_wall_s=float(setup_data.get("total_wall_s", 0.0)),
                    setup_tokens=int(setup_data.get("total_tokens", 0)),
                )

            df = store.load_responses()
            if exp.evaluate and not df.empty and (pending or "success" not in df.columns):
                df = evaluate_responses_df(
                    df,
                    eval_settings=profile.eval,
                    target_fk_grade=age_band.target_fk_grade,
                )
                df.to_parquet(store.responses_path, index=False)

            succ_mean = float(df["success"].mean()) if (not df.empty and "success" in df.columns) else None
            smart_log.finish_run(run_idx=run_idx, run_id=rspec.run_id, n_rows=len(df), success_mean=succ_mean)
            completed_dirs.append(run_dir)

    smart_log.finish_experiment(len(completed_dirs))
    return completed_dirs


def run_smoke_pipeline(work_dir: Path, max_examples: int = 6) -> pd.DataFrame:
    """Fast, offline CPU smoke pipeline exercising data -> methods -> eval -> analysis."""
    from minorsafe.analysis.aggregate import aggregate_experiment_results
    from minorsafe.analysis.plots import generate_all_figures
    from minorsafe.analysis.report import generate_executive_report
    from minorsafe.analysis.tables import generate_all_tables
    from minorsafe.testing import DummyCausalLM, DummyTokenizer

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
    generate_all_tables(summary_df, results_dir / "tables", responses_df=combined_df)
    generate_executive_report(summary_df, results_dir, responses_df=combined_df)
    return summary_df
