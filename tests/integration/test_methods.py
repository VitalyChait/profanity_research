"""Phase 5 integration tests: end-to-end execution of M0a, M0b, M1-Enc, M1-Dec, M2-LoRA, and M3-Steering on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest
from minorsafe.methods.base import RunContext, create_method

from minorsafe.config import load_age_bands, load_hardware, load_method, load_models, load_prompts

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "method_name",
    [
        "m0a_raw",
        "m0b_sysprompt",
        "m1_encoder_guard",
        "m1_decoder_guard",
        "m2_lora",
        "m3_steer_swfc",
        "m3_steer_sttp",
        "m3_steer_stmp",
        "m3_steer_ortho",
        "m3_probe_guard",
    ],
)
def test_all_method_families_execute_and_report_telemetry(
    method_name: str,
    dummy_lm,
    dummy_tokenizer,
    dummy_model_spec,
    sample_examples,
    sample_pairs,
    tmp_path: Path,
):
    models = load_models()
    hw = load_hardware("cpu_debug")
    bands = load_age_bands()
    prompts = load_prompts()
    mspec = load_method(method_name)
    # Keep calibration/epochs tiny for fast integration test
    mspec.params["n_calib"] = 8
    mspec.params["epochs"] = 1

    ctx = RunContext(
        hardware=hw,
        generator_spec=dummy_model_spec,
        models=models,
        age_bands=bands,
        prompts=prompts,
        calib_pairs=sample_pairs,
        calib_examples=sample_examples,
        dev_examples=sample_examples,
        artifact_dir=tmp_path / "artifacts",
        preloaded_generator=(dummy_lm, dummy_tokenizer),
    )

    method = create_method(mspec)
    setup_report = method.setup(ctx)
    assert setup_report.method == mspec.name

    outputs = method.respond(sample_examples, age_band="teen_12_18", seed=0)
    assert len(outputs) == len(sample_examples)
    for out in outputs:
        assert isinstance(out.text, str) and len(out.text) > 0
        assert out.tokens.total > 0
        assert out.e2e_ms >= 0.0
    method.teardown()
