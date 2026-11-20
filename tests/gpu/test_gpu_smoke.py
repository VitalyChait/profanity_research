"""Real-CUDA smoke tests verifying GB10/H100 GPU loading, generation timing, steering, and LoRA."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minorsafe.config import load_models
from minorsafe.data.schema import ContrastivePair
from minorsafe.generation import ChatGenerator
from minorsafe.methods.guards.encoder_guard import EncoderGuard
from minorsafe.methods.lora.train import train_lora_adapter
from minorsafe.methods.steering.extract import extract_steering_directions
from minorsafe.methods.steering.hooks import ResidualSteeringHook
from minorsafe.registry import free_model, get_decoder_layers, load_decoder
from minorsafe.telemetry.gpu import GPUSampler

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
]


def _mini_pairs() -> list[ContrastivePair]:
    return [
        ContrastivePair(
            id=f"gpu_p_{i}",
            concept="profanity",
            prompt=f"Someone spilled water on laptop {i}.",
            positive="That is really frustrating, let us dry it carefully.",
            negative="What the fuck, dry this damn mess right now.",
            split="calib",
        )
        for i in range(4)
    ]


def test_gpu_qwen25_05b_generation_steering_and_lora(tmp_path: Path):
    spec = load_models()["qwen25_05b"]
    model, tok, _ = load_decoder(spec, quant="bf16", device="cuda")
    try:
        gen = ChatGenerator(model, tok, spec, max_new_tokens=20, temperature=0.0)
        with GPUSampler(device_index=0) as sampler:
            out = gen.generate([[{"role": "user", "content": "Say hello in five words."}]], seed=0)[0]
        assert out.n_out > 0
        assert 0.0 < out.ttft_s <= out.e2e_s
        assert out.decode_tok_per_s > 0.0
        assert sampler.peak_mem_mb > 0.0

        pairs = _mini_pairs()
        num_layers = int(getattr(model.config, "num_hidden_layers", 24))
        dirs, _ = extract_steering_directions(
            model=model,
            tokenizer=tok,
            pairs=pairs,
            concepts=["profanity"],
            layer_indices=[num_layers // 2],
            estimator="pca",
            n_calib=4,
        )
        assert len(dirs) == 1
        assert float(dirs[0].vector.norm().item()) > 0.0

        hook = ResidualSteeringHook(dirs, mode="sttp", alpha=2.5).attach(
            get_decoder_layers(model)[dirs[0].layer_idx]
        )
        try:
            steered_out = gen.generate([[{"role": "user", "content": "Give study tips for teens."}]], seed=0)[
                0
            ]
            assert steered_out.n_out > 0
        finally:
            hook.detach()

        # 1-epoch PEFT LoRA smoke on 4 pairs
        peft_model, rep = train_lora_adapter(
            model=model,
            tokenizer=tok,
            gen_spec=spec,
            calib_pairs=pairs,
            artifact_dir=tmp_path / "lora_art",
            concepts=["profanity"],
            n_calib=4,
            r=8,
            lora_alpha=16,
            epochs=1,
            batch_size=2,
            quant="bf16",
            method_name="m2_lora_gpu_smoke",
        )
        assert rep.tokens > 0
        merged = peft_model.merge_and_unload()
        merged.eval()
        lora_gen = ChatGenerator(merged, tok, spec, max_new_tokens=16, temperature=0.0)
        lora_out = lora_gen.generate(
            [[{"role": "user", "content": "Explain photosynthesis briefly."}]], seed=0
        )[0]
        assert lora_out.n_out > 0
    finally:
        free_model(model, tok)


def test_gpu_encoder_guard_zeroshot(tmp_path: Path):
    spec = load_models()["toxic_bert"]
    guard, _ = EncoderGuard.from_spec(
        model_spec=spec,
        train_mode="zeroshot",
        calib_pairs=[],
        calib_examples=[],
        dev_examples=[],
        artifact_dir=tmp_path / "guard_art",
        device="cuda",
        quant="bf16",
        method_name="m1_encoder_zeroshot",
    )
    try:
        scores, tok_in = guard.score_texts(
            ["What the fuck is wrong with you, stupid idiot!", "Here is a helpful summary of plant biology."]
        )
        assert len(scores) == 2
        assert all(t > 0 for t in tok_in)
        assert scores[0] > scores[1]
    finally:
        guard.close()
