"""Shared pytest fixtures: synthetic examples, contrastive pairs, and fast CPU dummy models."""

from __future__ import annotations

import pytest
import torch

from minorsafe.config import ModelSpec
from minorsafe.data.schema import ContrastivePair, Example, Turn
from minorsafe.testing import BatchDict, DummyCausalLM, DummyDecoderLayer, DummyTokenizer

__all__ = ["BatchDict", "DummyCausalLM", "DummyDecoderLayer", "DummyTokenizer"]


@pytest.fixture
def dummy_lm() -> DummyCausalLM:
    torch.manual_seed(42)
    return DummyCausalLM(vocab_size=128, hidden_size=16, num_layers=4)


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer(vocab_size=128)


@pytest.fixture
def dummy_model_spec() -> ModelSpec:
    return ModelSpec(
        alias="tiny_llama",
        hf_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        arch="decoder",
        role="generator",
        family="llama",
        size_group="S",
        params_b=0.01,
        chat_template=False,
        supports_system=True,
    )


@pytest.fixture
def sample_examples() -> list[Example]:
    return [
        Example(
            source="unit_test",
            split="dev",
            turns=[Turn(role="user", content="Why do plants need sunlight to grow?")],
            label_unsafe=False,
            categories=["benign", "complex_lexicon"],
            reference_safe="Plants use sunlight, water, and air to make food and release oxygen.",
            age_band="teen_12_18",
        ),
        Example(
            source="unit_test",
            split="dev",
            turns=[Turn(role="user", content="Bro that boss fight cooked me 💀😭 any tips?")],
            label_unsafe=False,
            categories=["slang", "benign_tricky"],
            reference_safe="Try dodging the first attack pattern and upgrading your shield!",
            age_band="teen_12_18",
        ),
        Example(
            source="unit_test",
            split="dev",
            turns=[Turn(role="user", content="This fucking level is total bullshit f*ck 🖕")],
            label_unsafe=True,
            categories=["profanity", "obfuscated"],
            reference_safe="This seriously tough level feels unfair, let's break down the strategy.",
            age_band="teen_12_18",
        ),
    ]


@pytest.fixture
def sample_pairs() -> list[ContrastivePair]:
    pairs = []
    for i in range(12):
        pairs.append(
            ContrastivePair(
                concept="profanity",
                prompt=f"How was game match {i}?",
                positive=f"Match {i} was seriously intense and a total mess at the end.",
                negative=f"Match {i} was fucking intense and total bullshit at the end.",
                minimal=True,
                split="calib",
            )
        )
        pairs.append(
            ContrastivePair(
                concept="lexical_complexity",
                prompt=f"Explain concept {i} simply.",
                positive=f"We can improve the system and reduce harmful extra steps in part {i}.",
                negative=f"We can ameliorate the system and mitigate deleterious superfluous steps in part {i}.",
                minimal=True,
                split="calib",
            )
        )
        pairs.append(
            ContrastivePair(
                concept="slang_benign",
                prompt=f"Did you see play {i}?",
                positive=f"That play {i} was fire 🔥 no cap, she totally aced it 💀!",
                negative=f"That performance {i} was impressive without exaggeration, she succeeded.",
                minimal=False,
                split="calib",
            )
        )
        pairs.append(
            ContrastivePair(
                concept="slang_toxic",
                prompt=f"Reply to player {i}.",
                positive=f"Good try player {i}, let's regroup for the next round.",
                negative=f"Shut the f*ck up you sh1t noob {i} 🖕.",
                minimal=False,
                split="calib",
            )
        )
    return pairs
