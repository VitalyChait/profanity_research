"""Phase 5.4.2 unit tests: ResidualSteeringHook (SwFC, StTP, StMP), tuple/tensor handling, and teardown."""

from __future__ import annotations

import pytest
import torch
from minorsafe.testing import DummyCausalLM
from minorsafe.methods.steering.hooks import ConceptDirection, ResidualSteeringHook

pytestmark = pytest.mark.unit


def test_swfc_shifts_projection_by_exact_alpha():
    v = torch.zeros(16)
    v[0] = 1.0
    cd = ConceptDirection(concept="profanity", layer_idx=1, vector=v, m_l=0.0, s_l=1.0)
    hook = ResidualSteeringHook(directions=[cd], mode="swfc", alpha=3.5, prefill=True)

    h = torch.zeros(2, 4, 16)
    out = hook(None, None, (h.clone(), "cache_item"))
    assert isinstance(out, tuple) and len(out) == 2
    steered = out[0]
    proj = torch.matmul(steered, v)
    assert torch.allclose(proj, torch.full_like(proj, 3.5), atol=1e-6)


def test_sttp_leaves_compliant_tokens_bitwise_identical_and_moves_violators_to_target():
    v = torch.zeros(16)
    v[0] = 1.0
    # Decision boundary m_l = 0.0, target s_l = 2.0
    cd = ConceptDirection(concept="profanity", layer_idx=1, vector=v, m_l=0.0, s_l=2.0)
    hook = ResidualSteeringHook(directions=[cd], mode="sttp", prefill=False)

    # Batch of 2 sequences, length 3:
    # Batch 0 last token has rho = +1.5 (>= m_l, compliant -> must remain untouched!)
    # Batch 1 last token has rho = -1.2 (< m_l, non-compliant -> must move to s_l = 2.0!)
    h = torch.randn(2, 3, 16)
    h[0, -1, 0] = 1.5
    h[1, -1, 0] = -1.2
    h_orig = h.clone()

    out = hook(None, None, (h.clone(),))
    steered = out[0]

    # Compliant token is bitwise identical
    assert torch.equal(steered[0, -1], h_orig[0, -1])
    # Non-last tokens (when prefill=False) are bitwise identical
    assert torch.equal(steered[:, :-1], h_orig[:, :-1])
    # Violating token projection is now exactly s_l = 2.0, orthogonal components unchanged
    assert torch.isclose(torch.dot(steered[1, -1], v), torch.tensor(2.0), atol=1e-5)
    assert torch.allclose(steered[1, -1, 1:], h_orig[1, -1, 1:], atol=1e-6)
    assert hook.stats()["tokens_evaluated"] == 2
    assert hook.stats()["tokens_steered"] == 1
    assert hook.stats()["intervention_rate"] == pytest.approx(0.5)


def test_stmp_reflects_across_hyperplane():
    v = torch.zeros(16)
    v[0] = 1.0
    # Decision boundary m_l = 0.5; a token at rho = -0.5 is 1.0 below m_l,
    # so reflectionrho' = rho + 2*(m_l - rho) = 2*m_l - rho = 1.0 - (-0.5) = +1.5.
    cd = ConceptDirection(concept="profanity", layer_idx=1, vector=v, m_l=0.5, s_l=1.0)
    hook = ResidualSteeringHook(directions=[cd], mode="stmp", prefill=False)

    h = torch.zeros(1, 2, 16)
    h[0, -1, 0] = -0.5
    steered = hook(None, None, h.clone())  # bare Tensor output (not a tuple)
    assert isinstance(steered, torch.Tensor)
    assert torch.isclose(steered[0, -1, 0], torch.tensor(1.5), atol=1e-5)


def test_hook_registration_and_clean_teardown_on_model():
    for return_tuple in (True, False):
        model = DummyCausalLM(vocab_size=64, hidden_size=16, num_layers=4, return_tuple=return_tuple)
        inp = torch.tensor([[4, 7, 9, 12]])
        with torch.no_grad():
            base_logits = model(inp).logits.clone()

        v = torch.ones(16) / (16**0.5)
        cd = ConceptDirection(concept="profanity", layer_idx=2, vector=v, m_l=10.0, s_l=15.0)
        hook = ResidualSteeringHook(directions=[cd], mode="sttp", prefill=True)
        hook.attach(model.model.layers[2])

        with torch.no_grad():
            steered_logits = model(inp).logits.clone()
        assert not torch.allclose(base_logits, steered_logits)

        hook.detach()
        with torch.no_grad():
            restored_logits = model(inp).logits.clone()
        assert torch.equal(base_logits, restored_logits)
