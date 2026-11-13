"""Phase 1 unit tests: YAML configuration schemas, overrides, and model/generator selection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from minorsafe.config import (
    HardwareProfile,
    ModelSpec,
    apply_overrides,
    load_age_bands,
    load_data_spec,
    load_experiment,
    load_hardware,
    load_method,
    load_models,
    load_prompts,
    select_generators,
)
from minorsafe.utils import CONFIG_DIR

pytestmark = pytest.mark.unit


def test_all_repository_yaml_configs_validate():
    models = load_models()
    assert len(models) >= 25
    # Verify all 4 target families exist
    families = {m.family for m in models.values() if m.role == "generator"}
    assert {"qwen", "deepseek", "gemma", "llama"}.issubset(families)

    # Verify <1B (S), ~1B-4B (M), ~7B-9B (L) scale coverage
    size_groups = {m.size_group for m in models.values() if m.role == "generator"}
    assert size_groups == {"S", "M", "L"}

    # Verify age band 12-18
    bands = load_age_bands()
    assert "teen_12_18" in bands
    band = bands["teen_12_18"]
    assert band.min_age == 12 and band.max_age == 18
    assert band.target_fk_grade <= 8.0

    # Verify prompts
    prompts = load_prompts()
    assert "teen_12_18" in prompts["m0b_system_prompts"]
    assert "{draft_response}" in prompts["llm_rewrite_user_template"]
    assert "{response}" in prompts["judge_rubric"]

    # Verify data spec
    data_spec = load_data_spec()
    assert data_spec.english_min_ratio == pytest.approx(0.60)
    assert "slang_benign" in data_spec.concepts
    assert "profanity" in data_spec.concepts

    # Verify every hardware profile
    for p in (CONFIG_DIR / "hardware").glob("*.yaml"):
        hw = load_hardware(p.stem)
        assert isinstance(hw, HardwareProfile)

    # Verify every method spec
    for p in (CONFIG_DIR / "methods").glob("*.yaml"):
        ms = load_method(p.stem)
        assert ms.kind in ("baseline", "guard", "lora", "steering")

    # Verify every experiment spec
    for p in (CONFIG_DIR / "experiments").glob("*.yaml"):
        es = load_experiment(p.stem)
        assert es.age_bands == ["teen_12_18"]
        assert len(es.runs) >= 1


def test_strict_model_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        ModelSpec(
            alias="bad",
            hf_id="some/model",
            arch="decoder",
            role="generator",
            family="qwen",
            unknown_field_xyz=123,
        )


def test_apply_overrides_nested():
    base = {"batch_size": 8, "eval": {"capability_items": 100, "judge_backend": "vllm"}}
    updated = apply_overrides(
        base, ["batch_size=16", "eval.capability_items=25", "eval.judge_backend=heuristic"]
    )
    assert updated["batch_size"] == 16
    assert updated["eval"]["capability_items"] == 25
    assert updated["eval"]["judge_backend"] == "heuristic"

    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(base, ["invalid_override"])


def test_select_generators_by_alias_size_and_family():
    models = load_models()
    prof = load_hardware("gpu_80gb")

    s_gens = select_generators(["S"], prof, models)
    assert set(s_gens) == {"qwen25_05b", "gemma3_270m"}

    qwen_gens = select_generators(["qwen"], prof, models)
    assert {"qwen25_05b", "qwen25_15b", "qwen25_3b", "qwen25_7b"}.issubset(set(qwen_gens))

    exact = select_generators(["llama32_1b", "deepseek_r1_15b"], prof, models)
    assert exact == ["llama32_1b", "deepseek_r1_15b"]
