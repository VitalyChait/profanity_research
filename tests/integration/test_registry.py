"""Phase 1 integration tests: model layer path resolution and CPU registry loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn
from conftest import DummyCausalLM

from minorsafe.registry import compute_dtype, free_model, get_decoder_layers

pytestmark = pytest.mark.integration


def test_get_decoder_layers_across_architectures():
    # 1. Standard Llama / Qwen / DeepSeek layout: model.model.layers
    m1 = DummyCausalLM(num_layers=4)
    layers1 = get_decoder_layers(m1)
    assert len(layers1) == 4

    # 2. Gemma-3 multimodal layout: model.model.language_model.layers
    m2 = nn.Module()
    m2.model = nn.Module()
    m2.model.language_model = nn.Module()
    m2.model.language_model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
    layers2 = get_decoder_layers(m2)
    assert len(layers2) == 3

    # 3. PEFT-wrapped model with get_base_model()
    wrapper = SimpleNamespace(get_base_model=lambda: m1)
    layers3 = get_decoder_layers(wrapper)
    assert len(layers3) == 4


def test_compute_dtype_and_free_model():
    assert str(compute_dtype("bf16", "cpu")) == "torch.float32"
    m = DummyCausalLM(num_layers=2)
    free_model(m)
