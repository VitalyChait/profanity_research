"""Unit tests for scientific integrity guards, placeholder gating, and hypothesis decision rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from minorsafe.analysis.hypotheses import NOT_EVALUATED, NOT_SUPPORTED, SUPPORTED, evaluate_hypotheses
from minorsafe.config import (
    EvalSettings,
    MethodSpec,
    load_age_bands,
    load_hardware,
    load_method,
    load_models,
    load_prompts,
)
from minorsafe.eval.evaluate import evaluate_responses_df
from minorsafe.eval.judge import compute_human_agreement, export_human_annotation_sheet
from minorsafe.methods.base import RunContext, create_method
from minorsafe.methods.steering.extract import extract_steering_directions
from minorsafe.placeholders import PlaceholderBackendError, require_real_backend

pytestmark = pytest.mark.unit


def _make_ctx(
    hardware_name: str, dummy_lm, dummy_tokenizer, dummy_model_spec, sample_examples, sample_pairs, tmp_path
):
    return RunContext(
        hardware=load_hardware(hardware_name),
        generator_spec=dummy_model_spec,
        models=load_models(),
        age_bands=load_age_bands(),
        prompts=load_prompts(),
        calib_pairs=sample_pairs,
        calib_examples=sample_examples,
        dev_examples=sample_examples,
        artifact_dir=tmp_path / "artifacts",
        preloaded_generator=(dummy_lm, dummy_tokenizer),
    )


def test_placeholder_guard_profile_flags_and_require_real_backend():
    gpu_hw = load_hardware("gpu_80gb")
    cpu_hw = load_hardware("cpu_debug")
    assert gpu_hw.allow_placeholders is False
    assert cpu_hw.allow_placeholders is True

    # cpu_debug allows placeholders; gpu_80gb fails closed
    require_real_backend(cpu_hw, "test_component", "unit test")
    with pytest.raises(PlaceholderBackendError, match="test_component"):
        require_real_backend(gpu_hw, "test_component", "unit test")


def test_eval_backends_fail_closed_for_unimplemented_settings():
    df = pd.DataFrame(
        [
            {
                "example_id": "e1",
                "prompt": "Hello",
                "text": "Hi there, how can I help?",
                "label_unsafe": False,
            }
        ]
    )
    with pytest.raises(PlaceholderBackendError, match="judge_backend"):
        evaluate_responses_df(
            df,
            eval_settings=EvalSettings(
                judge_backend="vllm",
                ppl_model=None,
                toxicity_backend="lexicon",
                bertscore_model=None,
            ),
        )
    with pytest.raises(PlaceholderBackendError, match="ppl_model"):
        evaluate_responses_df(
            df,
            eval_settings=EvalSettings(
                judge_backend="heuristic",
                ppl_model="gpt2-large",
                toxicity_backend="lexicon",
                bertscore_model=None,
            ),
        )


@pytest.mark.parametrize("method_name", ["m1_encoder_guard", "m1_decoder_guard", "m2_lora"])
def test_placeholder_methods_fail_closed_on_gpu_profile(
    method_name: str,
    dummy_lm,
    dummy_tokenizer,
    dummy_model_spec,
    sample_examples,
    sample_pairs,
    tmp_path: Path,
):
    ctx = _make_ctx(
        "gpu_80gb", dummy_lm, dummy_tokenizer, dummy_model_spec, sample_examples, sample_pairs, tmp_path
    )
    method = create_method(load_method(method_name))
    with pytest.raises(PlaceholderBackendError):
        method.setup(ctx)


def test_no_hidden_lexicon_post_filter_on_m2_m3():
    raw_text = "what the wtf is this"
    for method_name in ("m0a_raw", "m2_lora", "m3_steer_sttp"):
        method = create_method(load_method(method_name))
        assert method.post_filter is False
        out, n_rep = method.apply_post_filter(raw_text)
        assert out == raw_text
        assert n_rep == 0

    hybrid_spec = MethodSpec(name="m3_sttp_lex", kind="steering", params={"post_filter": True})
    hybrid = create_method(hybrid_spec)
    assert hybrid.post_filter is True
    out_filt, n_rep_filt = hybrid.apply_post_filter(raw_text)
    assert "wtf" not in out_filt.lower()
    assert n_rep_filt >= 1


def test_steering_extract_rejects_missing_concept_and_missing_ortho_concept(
    dummy_lm, dummy_tokenizer, sample_pairs
):
    with pytest.raises(ValueError, match="No calibration pairs for steering concept"):
        extract_steering_directions(
            model=dummy_lm,
            tokenizer=dummy_tokenizer,
            pairs=sample_pairs,
            concepts=["nonexistent_concept"],
            layer_indices=[1],
        )

    with pytest.raises(ValueError, match="estimator='ortho' needs calibration pairs"):
        extract_steering_directions(
            model=dummy_lm,
            tokenizer=dummy_tokenizer,
            pairs=sample_pairs,
            concepts=["profanity"],
            layer_indices=[1],
            estimator="ortho",
            orthogonal_concept="missing_slang",
        )


def test_evaluate_hypotheses_empty_returns_not_evaluated():
    verdicts = evaluate_hypotheses(pd.DataFrame())
    assert len(verdicts) == 10
    assert {v["hypothesis"] for v in verdicts} == {f"H{i}" for i in range(1, 11)}
    assert all(v["status"] == NOT_EVALUATED for v in verdicts)


def _h1_summary_df(sttp_success: float, extra_rows: list[dict] | None = None) -> pd.DataFrame:
    rows = [
        {
            "experiment": "main_grid",
            "method": "m0a_raw",
            "method_base": "m0a_raw",
            "method_params": "{}",
            "model": "qwen25_15b",
            "quant": "bf16",
            "setup_wall_s": 0.0,
            "setup_gpu_s": 0.0,
            "setup_tokens": 0,
            "tok_total_mean": 100.0,
            "success_mean": 0.50,
        },
        {
            "experiment": "main_grid",
            "method": "m2_lora",
            "method_base": "m2_lora",
            "method_params": "{}",
            "model": "qwen25_15b",
            "quant": "bf16",
            "setup_wall_s": 1000.0,
            "setup_gpu_s": 1000.0,
            "setup_tokens": 500000,
            "tok_total_mean": 102.0,
            "success_mean": 0.90,
        },
        {
            "experiment": "main_grid",
            "method": "m3_steer_sttp",
            "method_base": "m3_steer_sttp",
            "method_params": "{}",
            "model": "qwen25_15b",
            "quant": "bf16",
            "setup_wall_s": 5.0,
            "setup_gpu_s": 5.0,
            "setup_tokens": 2000,
            "tok_total_mean": 101.0,
            "success_mean": sttp_success,
        },
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return pd.DataFrame(rows)


def test_evaluate_hypotheses_h1_data_driven_and_excludes_post_filter():
    verdicts_ok = {v["hypothesis"]: v for v in evaluate_hypotheses(_h1_summary_df(sttp_success=0.88))}
    assert verdicts_ok["H1"]["status"] == SUPPORTED
    assert verdicts_ok["H1"]["n_cells"] == 1

    verdicts_fail = {v["hypothesis"]: v for v in evaluate_hypotheses(_h1_summary_df(sttp_success=0.40))}
    assert verdicts_fail["H1"]["status"] == NOT_SUPPORTED
    assert verdicts_fail["H1"]["n_cells"] == 1

    # Opt-in post_filter=True hybrid rows must be excluded from pre-registered hypothesis checks:
    # if the only StTP row has post_filter=True, H1 must be NOT_EVALUATED rather than using it.
    df_only_hybrid = _h1_summary_df(sttp_success=0.95)
    df_only_hybrid.loc[df_only_hybrid["method_base"] == "m3_steer_sttp", "method_params"] = (
        '{"post_filter": true}'
    )
    verdicts_hybrid = {v["hypothesis"]: v for v in evaluate_hypotheses(df_only_hybrid)}
    assert verdicts_hybrid["H1"]["status"] == NOT_EVALUATED


def test_human_annotation_blinding_and_non_imputed_agreement(tmp_path: Path):
    rows = []
    for i in range(40):
        rows.append(
            {
                "run_id": f"run_{i % 2}",
                "experiment": "main_grid",
                "method": f"method_{i % 2}",
                "method_base": f"method_{i % 2}",
                "method_kind": "steering",
                "action_taken": "steered",
                "blocked": False,
                "guard_score": 0.1,
                "x_intervention_rate": 0.25,
                "example_id": f"ex_{i}",
                "category": "profanity" if i % 2 == 0 else "slang",
                "prompt": f"Prompt {i}",
                "text": f"Response {i}",
                "judge_age_appropriate": 5 if i % 2 == 0 else 2,
                "judge_meaning_preserved": 4,
                "judge_helpful": 4,
                "judge_fluent": 5,
            }
        )
    out_csv = tmp_path / "human_eval.csv"
    exported = export_human_annotation_sheet(pd.DataFrame(rows), n=20, out_path=out_csv, seed=13)

    assert len(exported) == 20
    for hidden_col in (
        "method",
        "method_base",
        "method_kind",
        "run_id",
        "action_taken",
        "blocked",
        "guard_score",
        "x_intervention_rate",
    ):
        assert hidden_col not in exported.columns
    key_csv = tmp_path / "human_eval_key.csv"
    assert key_csv.exists()
    key_df = pd.read_csv(key_csv)
    assert len(key_df) == 20
    assert {"sample_id", "method", "run_id", "action_taken"}.issubset(key_df.columns)

    # Leave the second half unlabeled (empty / NaN). Agreement must count only labeled rows, never fillna(3).
    labeled = exported.copy()
    for dim in ("age_appropriate", "meaning_preserved", "helpful", "fluent"):
        labeled.loc[:9, f"human_{dim}"] = labeled.loc[:9, f"judge_{dim}"].astype(str)
    # Rows 10..19 stay empty string -> NaN on read_csv
    labeled_path = tmp_path / "human_eval_labeled.csv"
    labeled.to_csv(labeled_path, index=False)

    agree = compute_human_agreement(labeled_path)
    assert agree["n_samples"] == 10
    assert agree["cohen_kappa_age_appropriate"] == pytest.approx(1.0)
