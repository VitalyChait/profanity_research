"""Setup-cost accounting, artifact-cache identity, and hardware-profile wiring (audit H1, H2, H3)."""

from __future__ import annotations

import gc
import json
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from minorsafe.analysis.hypotheses import NOT_EVALUATED, evaluate_hypotheses
from minorsafe.config import (
    MethodSpec,
    load_age_bands,
    load_experiment,
    load_hardware,
    load_models,
    load_prompts,
)
from minorsafe.methods.base import RunContext, create_method
from minorsafe.methods.steering.extract import load_steering_artifact, steering_setup_key
from minorsafe.methods.steering.sweep import sweep_setup_key
from minorsafe.runner.grid import expand_experiment_grid
from minorsafe.telemetry.cost import SetupReport, gpu_allocation
from minorsafe.testing import DummyCausalLM, DummyTokenizer

pytestmark = pytest.mark.unit


def _ctx(model_spec, examples, pairs, tmp_path: Path, *, quant="fp32", preloaded=None) -> RunContext:
    return RunContext(
        hardware=load_hardware("cpu_debug"),
        generator_spec=model_spec,
        models=load_models(),
        age_bands=load_age_bands(),
        prompts=load_prompts(),
        calib_pairs=pairs,
        calib_examples=examples,
        dev_examples=examples,
        quant=quant,
        artifact_dir=tmp_path / "artifacts",
        preloaded_generator=preloaded,
    )


# --------------------------------------------------------------------------- H1
def test_steering_setup_cost_includes_the_dev_sweep(
    dummy_lm, dummy_tokenizer, dummy_model_spec, sample_examples, sample_pairs, tmp_path
):
    params = {"mode": "sttp", "estimator": "pca", "use_sweep": True, "force_sweep": True, "n_calib": 8}
    ctx = _ctx(
        dummy_model_spec, sample_examples, sample_pairs, tmp_path, preloaded=(dummy_lm, dummy_tokenizer)
    )

    method = create_method(MethodSpec(name="m3_steer_sttp", kind="steering", params=params))
    rep = method.setup(ctx)
    method.teardown()

    names = [st.name for st in rep.stages]
    assert "sweep_hyperparams" in names and len(names) >= 2
    sweep = next(st for st in rep.stages if st.name == "sweep_hyperparams")
    assert sweep.tokens > 0 and sweep.extra["sweep_cached"] is False
    assert rep.tokens == sum(st.tokens for st in rep.stages) > sweep.tokens
    assert rep.to_dict()["total_tokens"] == rep.tokens
    # The extraction stage stays first so aggregate.py reads n_calib_effective from it.
    assert "n_calib_effective" in rep.stages[0].extra and "swept_params" in rep.stages[0].extra

    # A cached sweep still bills its original measured cost (like cached LoRA / encoder artifacts).
    method2 = create_method(MethodSpec(name="m3_steer_sttp", kind="steering", params=params))
    rep2 = method2.setup(ctx)
    method2.teardown()
    sweep2 = next(st for st in rep2.stages if st.name == "sweep_hyperparams")
    assert sweep2.extra["sweep_cached"] is True
    assert sweep2.tokens == sweep.tokens and sweep2.wall_s == pytest.approx(sweep.wall_s)


# --------------------------------------------------------------------------- H2
def test_sweep_and_steering_keys_separate_configurations():
    base = dict(
        quant="bf16",
        mode="sttp",
        estimator="pca",
        n_calib=100,
        concepts=["profanity", "lexical_complexity"],
        pooling="mean_completion",
        orthogonal_concept="slang_benign",
        grid={"rel_depths": [0.5]},
    )
    k = sweep_setup_key("qwen25_7b", **base)
    assert sweep_setup_key("qwen25_7b", **{**base, "quant": "nf4"}) != k
    assert sweep_setup_key("qwen25_7b", **{**base, "concepts": ["profanity"]}) != k
    assert sweep_setup_key("qwen25_7b", **{**base, "pooling": "last_prompt_token"}) != k
    assert sweep_setup_key("qwen25_7b", **{**base, "grid": {"rel_depths": [0.25]}}) != k
    assert sweep_setup_key("qwen25_7b", **{**base, "concepts": ["lexical_complexity", "profanity"]}) == k
    # The nuisance concept only matters to the ortho estimator.
    assert sweep_setup_key("qwen25_7b", **{**base, "orthogonal_concept": None}) == k
    ortho = {**base, "estimator": "ortho"}
    assert sweep_setup_key("m", **ortho) != sweep_setup_key("m", **{**ortho, "orthogonal_concept": "x"})

    s_args = ("m", "bf16", "ortho", "mean_completion", [5], 100, ["profanity"])
    assert steering_setup_key(*s_args, "slang_benign") != steering_setup_key(*s_args, "other")


def test_lora_and_encoder_keys_cover_training_config():
    from minorsafe.methods.guards.train_encoder import encoder_setup_key
    from minorsafe.methods.lora.train import lora_setup_key

    lora = dict(
        quant="bf16",
        n_calib=500,
        r=16,
        epochs=2,
        concepts=["profanity"],
        lora_alpha=32,
        lora_dropout=0.05,
        lr=2e-4,
        batch_size=4,
        seed=13,
    )
    k = lora_setup_key("qwen25_7b", **lora)
    for change in (
        {"lr": 1e-4},
        {"concepts": ["profanity", "mature_theme"]},
        {"lora_alpha": 16},
        {"lora_dropout": 0.1},
        {"batch_size": 8},
        {"seed": 1},
        {"quant": "nf4"},
    ):
        assert lora_setup_key("qwen25_7b", **{**lora, **change}) != k, change

    enc = dict(quant="bf16", n_calib=200, epochs=2, lr=3e-5, batch_size=16, seed=13)
    e = encoder_setup_key("modernbert_base", **enc)
    for change in ({"quant": "fp32"}, {"seed": 1}, {"batch_size": 8}):
        assert encoder_setup_key("modernbert_base", **{**enc, **change}) != e, change


def test_bf16_vector_transfer_releases_generator_and_retargets_cached_offset(
    dummy_model_spec, sample_examples, sample_pairs, tmp_path, monkeypatch
):
    """quant_grid vector_source_quant=bf16: no two models resident; cache hits honour the run's offset."""
    import minorsafe.methods.base as base_mod
    import minorsafe.methods.steering.method as steer_mod

    loads: list[tuple[str, weakref.ref]] = []
    alive_at_load: list[list[str]] = []

    def _fake_load(spec, quant="bf16", device=None):
        gc.collect()
        alive_at_load.append([q for q, ref in loads if ref() is not None])
        model = DummyCausalLM(hidden_size=16, num_layers=4)
        # Track a decoder layer, not just the wrapper: its parameters are what occupy VRAM.
        loads.append((quant, weakref.ref(model.model.layers[0])))
        return model, DummyTokenizer(), None

    monkeypatch.setattr(base_mod, "load_decoder", _fake_load)
    monkeypatch.setattr(steer_mod, "load_decoder", _fake_load)

    params = {"mode": "sttp", "estimator": "pca", "vector_source_quant": "bf16", "n_calib": 8}
    ctx = _ctx(dummy_model_spec, sample_examples, sample_pairs, tmp_path, quant="int8")

    m1 = create_method(MethodSpec(name="m3_steer_sttp", kind="steering", params=params))
    rep1 = m1.setup(ctx)
    assert [q for q, _ in loads] == ["int8", "bf16", "int8"]
    assert alive_at_load[1] == [], "quantized generator still resident while the BF16 copy loaded"
    assert rep1.cached is False
    m1.teardown()

    key = steering_setup_key(
        dummy_model_spec.alias, "bf16", "pca", "mean_completion", [2], 8, ["profanity", "lexical_complexity"]
    )
    meta_path = ctx.artifact_dir / key / "steering_meta.json"
    meta_before = meta_path.read_text()
    assert json.loads(meta_before)["setup"]["cached"] is False

    # Cache hit with a different target offset: s_l is re-targeted, the artifact is not rewritten.
    loads.clear()
    m2 = create_method(
        MethodSpec(name="m3_steer_sttp", kind="steering", params={**params, "target_offset_sigma": 1.5})
    )
    rep2 = m2.setup(ctx)
    assert [q for q, _ in loads] == ["int8"]
    assert rep2.cached is True
    for cd in m2.hook.directions:
        assert cd.s_l == pytest.approx(max(cd.mu_pos + 1.5 * cd.sigma_pos, cd.m_l + 0.1 * cd.sigma_pos))
    stored = {d.concept: d.s_l for d in load_steering_artifact(ctx.artifact_dir, key)[0]}
    assert any(abs(stored[cd.concept] - cd.s_l) > 1e-6 for cd in m2.hook.directions)
    assert meta_path.read_text() == meta_before
    m2.teardown()


# --------------------------------------------------------------------------- H3
def test_qlora_profiles_run_large_lora_in_nf4():
    models = load_models()
    exp = load_experiment("main_grid")

    specs_24 = expand_experiment_grid(exp, load_hardware("gpu_24gb"), models)
    lora_24 = [s for s in specs_24 if s.method_spec.name == "m2_lora"]
    large = [s for s in lora_24 if models[s.generator_alias].size_group == "L"]
    assert large
    for s in lora_24:
        is_large = models[s.generator_alias].size_group == "L"
        assert s.quant == ("nf4" if is_large else "bf16")
        assert (s.method_params.get("lora_mode") == "qlora") == is_large
        if is_large:
            assert s.method_params["matched_quant"] == "bf16" and "__nf4__" in s.run_id
    assert all(s.quant == "bf16" for s in specs_24 if s.method_spec.name != "m2_lora")

    specs_80 = expand_experiment_grid(exp, load_hardware("gpu_80gb"), models)
    assert all(s.quant == "bf16" for s in specs_80)

    # Explicit precision sweeps are never rewritten.
    qg = expand_experiment_grid(load_experiment("quant_grid"), load_hardware("gpu_24gb"), models)
    qg_lora = [s for s in qg if s.method_spec.name == "m2_lora"]
    assert {s.quant for s in qg_lora} == {"bf16", "nf4"}
    assert not any("lora_mode" in s.method_params for s in qg_lora)


def test_h1_compares_qlora_arm_within_the_default_precision_cell():
    base = {"experiment": "main_grid", "model": "qwen25_7b", "size_group": "L"}
    rows = [
        {**base, "method_base": "m0a_raw", "quant": "bf16", "method_params": "{}", "tok_total_mean": 100.0},
        {
            **base,
            "method_base": "m3_steer_sttp",
            "quant": "bf16",
            "method_params": "{}",
            "success_mean": 0.80,
            "setup_gpu_s": 1.0,
            "setup_wall_s": 1.0,
            "setup_tokens": 100,
            "tok_total_mean": 100.0,
        },
        {
            **base,
            "method_base": "m2_lora",
            "quant": "nf4",
            "method_params": json.dumps({"lora_mode": "qlora", "matched_quant": "bf16"}),
            "success_mean": 0.82,
            "setup_gpu_s": 1000.0,
            "setup_wall_s": 1000.0,
            "setup_tokens": 100000,
            "tok_total_mean": 100.0,
        },
    ]
    for r in rows:
        r["method"] = r["method_base"]
    v = {d["hypothesis"]: d for d in evaluate_hypotheses(pd.DataFrame(rows))}
    assert v["H1"]["status"] != NOT_EVALUATED
    assert v["H1"]["n_cells"] == 1
    assert "qwen25_7b/bf16" in v["H1"]["evidence"] and "QLoRA/nf4" in v["H1"]["evidence"]


def test_setup_reports_bill_allocated_gpus():
    assert SetupReport(method="m", setup_key="k").num_gpus == 1
    with gpu_allocation(2):
        assert SetupReport(method="m", setup_key="k").num_gpus == 2
        # Reports restored from an artifact cache keep the allocation they were measured under.
        assert SetupReport.from_dict({"method": "m", "setup_key": "k", "num_gpus": 1}).num_gpus == 1
    with gpu_allocation(0):
        assert SetupReport(method="m", setup_key="k").num_gpus == 1


def _install_fake_vllm(monkeypatch, created: list[dict], reply: str) -> None:
    class _LLM:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.prompts: list[str] = []

        def generate(self, prompts, params):
            limit = created[-1]["max_model_len"] - params.max_tokens
            outs = []
            for p in prompts:
                n_in = len(p.split())
                if n_in > limit:
                    raise ValueError(f"prompt of {n_in} tokens exceeds max_model_len")
                outs.append(
                    SimpleNamespace(
                        prompt_token_ids=list(range(n_in)),
                        outputs=[SimpleNamespace(text=reply, token_ids=[1, 2, 3])],
                    )
                )
            return outs

    fake = ModuleType("vllm")
    fake.LLM = _LLM
    fake.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", fake)


def test_judge_uses_tensor_parallelism_and_fits_long_inputs(monkeypatch):
    import minorsafe.eval.judge as judge_mod

    created: list[dict] = []
    reply = (
        '{"age_appropriate": 5, "meaning_preserved": 4, "helpful": 4, "fluent": 5, '
        '"is_refusal": false, "rationale": "ok"}'
    )
    _install_fake_vllm(monkeypatch, created, reply)
    monkeypatch.setattr(judge_mod, "load_tokenizer", lambda spec, quant: DummyTokenizer())

    judge = judge_mod.LLMJudge(
        load_models()["mistral_small_24b_awq"],
        backend="vllm",
        quant="awq",
        tensor_parallel_size=2,
        max_model_len=600,
    )
    long_reply = " ".join(f"word{i}" for i in range(3000))
    records = [
        {"prompt": "Short question?", "text": "Short answer.", "label_unsafe": False},
        {"prompt": "Explain photosynthesis.", "text": long_reply, "label_unsafe": False},
    ]
    out = judge.score_records(records)
    assert created[0]["tensor_parallel_size"] == 2 and created[0]["max_model_len"] == 600
    assert [o["judge_input_truncated"] for o in out] == [False, True]
    assert all(o["judge_parse_ok"] for o in out)
    assert out[1]["judge_tok_in"] <= 600 - 256
