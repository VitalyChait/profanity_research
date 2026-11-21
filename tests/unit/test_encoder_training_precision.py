"""claude_issues_found.md H5: the encoder guard must train with fp32 master weights, not pure bf16.

There is no GPU here, so the GPU's bf16 path is simulated on CPU: ``compute_dtype`` is patched to answer
bf16 the way it does on a bf16-capable GPU, and CPU bf16 autocast stands in for CUDA autocast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

import minorsafe.methods.guards.train_encoder as te
from minorsafe.config import ModelSpec
from minorsafe.data.schema import ContrastivePair, Example, Turn
from minorsafe.utils import stable_hash

pytestmark = pytest.mark.unit

_SAFE = ["please", "explain", "photosynthesis", "homework", "math", "history", "friendly", "help"]
_UNSAFE = ["damn", "hell", "crap", "stupid", "idiot", "shut", "trash", "jerk"]


def _gpu_like_dtype(quant: str, device: str) -> torch.dtype:
    return torch.float32 if quant == "fp32" else torch.bfloat16


def _data():
    pairs, calib, dev = [], [], []
    for i in range(24):
        s = " ".join(_SAFE[(i + k) % len(_SAFE)] for k in range(5))
        u = " ".join(_UNSAFE[(i + k) % len(_UNSAFE)] for k in range(5))
        pairs.append(
            ContrastivePair(concept="profanity", prompt=f"p{i}", positive=s, negative=u, split="calib")
        )
        for bucket, split in ((calib, "calib"), (dev, "dev")):
            bucket.append(
                Example(source="t", split=split, turns=[Turn(role="user", content=s)], label_unsafe=False)
            )
            bucket.append(
                Example(source="t", split=split, turns=[Turn(role="user", content=u)], label_unsafe=True)
            )
    return pairs, calib, dev


def test_encoder_guard_trains_fp32_master_weights_under_bf16_autocast(tmp_path: Path, monkeypatch):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *_SAFE, *_UNSAFE]) + "\n")
    seen = {"load_quants": [], "train_forward": []}
    snap: dict = {}

    def fake_loader(spec, num_labels=None, device=None, quant="bf16"):
        seen["load_quants"].append(quant)
        torch.manual_seed(0)
        cfg = BertConfig(
            vocab_size=32,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=num_labels or 2,
        )
        # Same contract as registry.load_encoder_classifier: weights arrive in compute_dtype(quant).
        model = BertForSequenceClassification(cfg).to(_gpu_like_dtype(quant, "cuda"))
        snap["init"] = {n: p.detach().float().clone() for n, p in model.named_parameters()}

        def record(mod, _args):
            if mod.training:
                seen["train_forward"].append(
                    (
                        next(mod.parameters()).dtype,
                        torch.is_autocast_enabled("cpu"),
                        torch.get_autocast_dtype("cpu"),
                    )
                )

        model.register_forward_pre_hook(record)
        return model, BertTokenizerFast(vocab_file=str(vocab)), None

    real_cast = te.cast_for_deployment

    def spy_cast(model, quant, device):
        snap["trained"] = {n: p.detach().clone() for n, p in model.named_parameters()}
        return real_cast(model, quant, device)

    monkeypatch.setattr(te, "compute_dtype", _gpu_like_dtype)
    monkeypatch.setattr(te, "load_encoder_classifier", fake_loader)
    monkeypatch.setattr(te, "cast_for_deployment", spy_cast)

    pairs, calib, dev = _data()
    spec = ModelSpec(
        alias="tiny_enc", hf_id="local/tiny", arch="encoder", role="encoder_guard", family="bert"
    )
    ckpt_dir, threshold, rep = te.train_encoder_guard(
        spec,
        pairs,
        calib,
        dev,
        tmp_path / "artifacts",
        n_calib=24,
        epochs=2,
        lr=3e-5,
        batch_size=8,
        device="cpu",
        quant="bf16",
    )

    # Weights are loaded in fp32 although the arm's quant is bf16.
    assert seen["load_quants"] == ["fp32"]
    # Every training forward ran fp32 weights under bf16 autocast.
    assert seen["train_forward"]
    assert all(d == (torch.float32, True, torch.bfloat16) for d in seen["train_forward"])
    # Small AdamW steps (lr=3e-5) accumulate in the fp32 master weights instead of rounding away.
    # With pure bf16 weights only ~15-30% of elements move (the issue's measurement).
    moved = sum(int((snap["trained"][n] != snap["init"][n]).sum()) for n in snap["init"])
    total = sum(t.numel() for t in snap["init"].values())
    assert moved / total > 0.95, moved / total
    # The guard is calibrated and saved in the dtype it is served in.
    weights = load_file(str(ckpt_dir / "model.safetensors"))
    assert {t.dtype for t in weights.values()} == {torch.bfloat16}
    assert 0.0 < threshold < 1.0
    stage = rep.stages[0]
    assert stage.extra["train_precision"] == te.TRAIN_PRECISION
    assert stage.extra["autocast_dtype"] == "bfloat16"


def test_cpu_training_stays_plain_fp32():
    assert te.training_autocast_dtype("bf16", "cpu") is None


def test_checkpoints_trained_in_pure_bf16_are_not_reused():
    kw = dict(quant="bf16", n_calib=200, epochs=2, lr=3e-5, batch_size=16, seed=13)
    old_key = (
        f"enc_modernbert_base_bf16_N200_ep2_lr3e-05_{stable_hash({'batch_size': 16, 'seed': 13}, length=6)}"
    )
    assert te.encoder_setup_key("modernbert_base", **kw) != old_key
