"""Unit tests for batched guard/probe execution, batched Detoxify, stratified sampling, and MSBA/pilot configs."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from minorsafe.analysis.hypotheses import H9_REASONING_PAIRS
from minorsafe.config import (
    MethodSpec,
    ModelSpec,
    load_age_bands,
    load_experiment,
    load_hardware,
    load_models,
    load_prompts,
)
from minorsafe.data.schema import Example, Turn
from minorsafe.eval.safety import compute_safety_metrics, compute_safety_metrics_batch
from minorsafe.methods.base import RunContext, create_method
from minorsafe.methods.guards.decoder_guard import DecoderGuard
from minorsafe.runner.grid import expand_experiment_grid
from minorsafe.runner.run import _stratified_subsample

pytestmark = pytest.mark.unit


def _ctx(dummy_lm, dummy_tokenizer, dummy_model_spec, examples, sample_pairs, tmp_path):
    return RunContext(
        hardware=load_hardware("cpu_debug"),
        generator_spec=dummy_model_spec,
        models=load_models(),
        age_bands=load_age_bands(),
        prompts=load_prompts(),
        calib_pairs=sample_pairs,
        calib_examples=examples,
        dev_examples=examples,
        artifact_dir=tmp_path / "artifacts",
        preloaded_generator=(dummy_lm, dummy_tokenizer),
    )


def _make_example(idx: int, cat: str, unsafe: bool = False) -> Example:
    return Example(
        id=f"ex_{cat}_{idx}",
        source="unit",
        split="dev",
        turns=[Turn(role="user", content=f"Sample {cat} query number {idx}?")],
        categories=[cat],
        label_unsafe=unsafe,
        age_band="teen_12_18",
    )


def test_stratified_subsample_preserves_rare_categories():
    pool = (
        [_make_example(i, "profanity", True) for i in range(60)]
        + [_make_example(i, "obfuscated", True) for i in range(60)]
        + [_make_example(i, "benign", False) for i in range(60)]
        + [_make_example(i, "slang", False) for i in range(8)]
    )
    sampled = _stratified_subsample(pool, max_examples=32, seed=13)
    assert len(sampled) == 32
    counts = Counter(ex.categories[0] for ex in sampled)
    # Rare category 'slang' keeps all 8 of its examples rather than getting dropped by naive slicing.
    assert counts["slang"] == 8
    assert counts["profanity"] == 8
    assert counts["obfuscated"] == 8
    assert counts["benign"] == 8


def test_compute_safety_metrics_batch_lexicon_and_detoxify_mock(monkeypatch):
    texts = ["hello friendly teen", "what the fuck is happening"]
    labels = [False, True]
    batch_res = compute_safety_metrics_batch(texts, labels, backend="lexicon")
    single_res = [
        compute_safety_metrics(t, lbl, backend="lexicon") for t, lbl in zip(texts, labels, strict=True)
    ]
    assert batch_res == single_res

    calls: list[list[str]] = []

    class _FakeDetox:
        def predict(self, chunk):
            calls.append(list(chunk))
            if isinstance(chunk, str):
                return {"toxicity": 0.9 if "fuck" in chunk else 0.02}
            return {"toxicity": [0.9 if "fuck" in c else 0.02 for c in chunk]}

    import minorsafe.eval.safety as safety_mod

    monkeypatch.setattr(safety_mod, "_get_detoxify", lambda device=None: _FakeDetox())
    out = compute_safety_metrics_batch(texts, labels, backend="detoxify", batch_size=64, device="cpu")
    # Single batched call of length 2 instead of 2 serial calls
    assert len(calls) == 1
    assert calls[0] == texts
    assert out[0]["toxic_flag"] is False
    assert out[1]["toxic_flag"] is True


def test_guard_method_batches_generator_and_decoder_guard(
    dummy_lm, dummy_tokenizer, dummy_model_spec, sample_pairs, tmp_path, monkeypatch
):
    gen_batch_sizes: list[int] = []
    batch = [_make_example(i, "benign", False) for i in range(4)]

    method = create_method(
        MethodSpec(
            name="m1_encoder_guard",
            kind="guard",
            params={"guard_arch": "encoder", "train_mode": "zeroshot", "action": "fallback"},
        )
    )
    ctx = _ctx(dummy_lm, dummy_tokenizer, dummy_model_spec, batch, sample_pairs, tmp_path)
    method.setup(ctx)

    orig_gen = method.generator.generate

    def _spy_gen(batch_msgs, **kwargs):
        gen_batch_sizes.append(len(batch_msgs))
        return orig_gen(batch_msgs, **kwargs)

    monkeypatch.setattr(method.generator, "generate", _spy_gen)
    outs = method.respond(batch, seed=0)
    assert len(outs) == 4
    assert gen_batch_sizes == [4]
    method.teardown()


def test_decoder_guard_score_pairs_batch():
    class _BatchEnc(dict):
        def to(self, _dev):
            return self

    class _Tok:
        pad_token_id = 0
        eos_token_id = 0
        padding_side = "left"
        chat_template = None

        def __call__(self, texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            ids = torch.tensor([[5, 6, 7]] * n, dtype=torch.long)
            return _BatchEnc({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        def encode(self, txt, add_special_tokens=False):
            return {"safe": [1], "unsafe": [2]}.get(txt, [])

        def decode(self, ids, skip_special_tokens=True):
            mp = {1: "safe", 2: "unsafe"}
            return " ".join(mp.get(int(i), "") for i in ids)

    class _LM:
        def eval(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1)])

        def generate(self, **kwargs):
            bs, in_len = kwargs["input_ids"].shape
            logits = torch.full((bs, 8), -20.0)
            logits[0, 2] = 5.0  # row 0 unsafe
            logits[0, 1] = 1.0
            logits[1, 1] = 5.0  # row 1 safe
            logits[1, 2] = 1.0
            seq = torch.cat([kwargs["input_ids"], torch.tensor([[2], [1]])], dim=1)
            return SimpleNamespace(sequences=seq[:, : in_len + 1], scores=(logits,))

    spec = ModelSpec(alias="dec_g", hf_id="dummy", arch="decoder", role="decoder_guard", family="llama")
    guard = DecoderGuard(_LM(), _Tok(), spec)
    res = guard.score_pairs_batch([("p1", None), ("p2", "clean reply")], batch_size=8)
    assert len(res) == 2
    assert res[0][0] > 0.9
    assert res[1][0] < 0.1


def test_probe_guard_batches_prefill_and_response(
    dummy_lm, dummy_tokenizer, dummy_model_spec, sample_pairs, tmp_path, monkeypatch
):
    gen_batch_sizes: list[int] = []
    batch = [_make_example(i, "benign", False) for i in range(4)]
    method = create_method(
        MethodSpec(
            name="m3_probe_guard",
            kind="steering",
            params={
                "mode": "probe_guard",
                "action": "fallback",
                "probe_location": "prompt_end",
                "n_calib": 4,
            },
        )
    )
    ctx = _ctx(dummy_lm, dummy_tokenizer, dummy_model_spec, batch, sample_pairs, tmp_path)
    method.setup(ctx)

    orig_gen = method.generator.generate

    def _spy_gen(batch_msgs, **kwargs):
        gen_batch_sizes.append(len(batch_msgs))
        return orig_gen(batch_msgs, **kwargs)

    monkeypatch.setattr(method.generator, "generate", _spy_gen)
    outs = method.respond(batch, seed=0)
    assert len(outs) == 4
    assert gen_batch_sizes == [4]
    method.teardown()


def test_gpu_gb10_and_pilot_configs_preserve_h1_to_h10_design():
    hw = load_hardware("gpu_gb10")
    models = load_models()
    assert hw.allow_placeholders is False
    assert hw.batch_size >= 64
    families = {models[a].family for a in hw.generators}
    sizes = {models[a].size_group for a in hw.generators}
    assert families == {"qwen", "deepseek", "gemma", "llama"}
    assert sizes == {"S", "M", "L"}
    # At least one matched H9 reasoning pair present
    assert "deepseek_r1_15b" in hw.generators
    assert any(ref in hw.generators for ref in H9_REASONING_PAIRS["deepseek_r1_15b"])

    # Pilot experiment resolves cleanly on gpu_gb10
    pilot = load_experiment("pilot")
    specs = expand_experiment_grid(pilot, hw, models)
    assert len(specs) == 36
    assert {s.method_spec.name for s in specs} >= {
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
    }


def test_resolve_guard_placement_rules():
    from minorsafe.methods.guards.method import resolve_guard_placement

    assert resolve_guard_placement("auto", "fallback") == "both"
    for action in ("regenerate", "llm_rewrite", "mlm_substitute"):
        assert resolve_guard_placement("auto", action) == "output"
        assert resolve_guard_placement("output", action) == "output"
        for blocking in ("input", "both"):
            with pytest.raises(ValueError, match="never run"):
                resolve_guard_placement(blocking, action)
    assert resolve_guard_placement("input", "fallback") == "input"
    with pytest.raises(ValueError):
        resolve_guard_placement("sideways", "fallback")
    with pytest.raises(ValueError):
        resolve_guard_placement("auto", "shout")


def test_every_configured_guard_arm_runs_its_action():
    """No experiment may pair a remediation action with input-side blocking (H4 / action sweeps)."""
    import itertools

    from minorsafe.config import CONFIG_DIR, load_method
    from minorsafe.methods.guards.method import GuardResponseMethod

    checked = 0
    for path in sorted((CONFIG_DIR / "experiments").glob("*.yaml")):
        exp = load_experiment(path.stem)
        for group in exp.runs:
            if group.method not in ("m1_encoder_guard", "m1_decoder_guard"):
                continue
            base = load_method(group.method)
            keys = list(group.sweep)
            for combo in itertools.product(*(group.sweep[k] for k in keys)) if keys else [()]:
                params = {**base.params, **group.params, **dict(zip(keys, combo, strict=True))}
                method = GuardResponseMethod(MethodSpec(name=base.name, kind=base.kind, params=params))
                expected = "both" if method.action == "fallback" else "output"
                assert method.placement == expected, (path.stem, group.label, params)
                checked += 1
    assert checked >= 20


def _profane_examples() -> list[Example]:
    return [
        Example(
            id=f"prof_{i}",
            source="unit",
            split="dev",
            turns=[Turn(role="user", content=f"This fucking homework {i} is bullshit, explain fractions?")],
            categories=["profanity"],
            label_unsafe=False,
            age_band="teen_12_18",
        )
        for i in range(3)
    ]


def test_remediation_action_runs_on_flagged_profane_prompts(
    dummy_lm, dummy_tokenizer, dummy_model_spec, sample_pairs, tmp_path, monkeypatch
):
    batch = _profane_examples()
    ctx = _ctx(dummy_lm, dummy_tokenizer, dummy_model_spec, batch, sample_pairs, tmp_path)

    def _always_flag(prompts, responses=None):
        return [(0.95, 4, 0) for _ in prompts], True, 1.0

    # Remediation arm (recomposition config): drafts are generated and recomposed, never short-circuited.
    mlm = create_method(
        MethodSpec(
            name="m1_encoder_guard",
            kind="guard",
            params={"guard_arch": "encoder", "train_mode": "zeroshot", "action": "mlm_substitute"},
        )
    )
    mlm.setup(ctx)
    mlm.threshold = 0.5
    monkeypatch.setattr(mlm, "_score_batch_prompts_or_texts", _always_flag)
    gen_calls: list[int] = []
    orig_gen = mlm.generator.generate
    monkeypatch.setattr(
        mlm.generator, "generate", lambda msgs, **kw: gen_calls.append(len(msgs)) or orig_gen(msgs, **kw)
    )
    outs = mlm.respond(batch, seed=0)
    assert gen_calls == [3]
    assert [o.action_taken for o in outs] == ["mlm_substitute"] * 3
    assert not any(o.blocked for o in outs)
    assert {o.extra["guard_placement"] for o in outs} == {"output"}
    mlm.teardown()

    # Fallback arm keeps early input blocking: no draft is generated for flagged prompts.
    fb = create_method(
        MethodSpec(
            name="m1_encoder_guard",
            kind="guard",
            params={"guard_arch": "encoder", "train_mode": "zeroshot", "action": "fallback"},
        )
    )
    fb.setup(ctx)
    fb.threshold = 0.5
    monkeypatch.setattr(fb, "_score_batch_prompts_or_texts", _always_flag)
    fb_calls: list[int] = []
    monkeypatch.setattr(fb.generator, "generate", lambda msgs, **kw: fb_calls.append(len(msgs)) or [])
    outs = fb.respond(batch, seed=0)
    assert fb_calls == []
    assert all(o.blocked and o.action_taken == "fallback" for o in outs)
    assert {o.extra["guard_placement"] for o in outs} == {"both"}
    fb.teardown()
