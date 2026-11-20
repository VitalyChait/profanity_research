"""Phase 5.4.4 & Phase 8 unit tests: Latent Tomography, subcultural entanglement matrix, and Logit Lens."""

from __future__ import annotations

import numpy as np
import pytest

from minorsafe.data.schema import ContrastivePair
from minorsafe.methods.steering.tomography import run_latent_tomography

pytestmark = pytest.mark.unit


def test_latent_tomography_matrix_symmetry_and_logit_lens(dummy_lm, dummy_tokenizer, sample_pairs):
    report = run_latent_tomography(
        model=dummy_lm,
        tokenizer=dummy_tokenizer,
        pairs=sample_pairs,
        concepts=["profanity", "lexical_complexity", "slang_toxic", "slang_benign"],
        top_k_tokens=5,
    )
    assert "layers" in report
    assert len(report["layers"]) == len(dummy_lm.model.layers)

    for layer_entry in report["layers"]:
        cos_mat = np.array(layer_entry["cosine_matrix"])
        assert cos_mat.shape == (4, 4)
        # Symmetry: C == C^T
        assert np.allclose(cos_mat, cos_mat.T, atol=1e-5)
        # Diagonal entries == 1.0
        assert np.allclose(np.diag(cos_mat), 1.0, atol=1e-5)
        # Bounds [-1, 1]
        assert np.all((cos_mat >= -1.0001) & (cos_mat <= 1.0001))

        for c in ("profanity", "lexical_complexity", "slang_toxic", "slang_benign"):
            c_stats = layer_entry["concepts"][c]
            assert 0.0 <= c_stats["auroc"] <= 1.0
            assert len(c_stats["promoted_tokens"]) == 5
            assert len(c_stats["suppressed_tokens"]) == 5


def test_tomography_scores_auroc_on_heldout_eval_pairs(dummy_lm, dummy_tokenizer, sample_pairs):
    train_pairs = [p for p in sample_pairs if p.concept == "profanity"]
    eval_normal = [
        ContrastivePair(
            concept="profanity",
            prompt=f"Held-out game review {i}?",
            positive=f"Held-out game {i} had a rough ending and messy controls.",
            negative=f"Held-out game {i} had a fucking awful ending and total bullshit controls.",
            split="dev",
        )
        for i in range(8)
    ]
    eval_swapped = [
        p.model_copy(update={"positive": p.negative, "negative": p.positive}) for p in eval_normal
    ]

    rep_normal = run_latent_tomography(
        model=dummy_lm,
        tokenizer=dummy_tokenizer,
        pairs=train_pairs,
        concepts=["profanity"],
        top_k_tokens=3,
        eval_pairs=eval_normal,
    )
    rep_swapped = run_latent_tomography(
        model=dummy_lm,
        tokenizer=dummy_tokenizer,
        pairs=train_pairs,
        concepts=["profanity"],
        top_k_tokens=3,
        eval_pairs=eval_swapped,
    )

    assert rep_normal["auroc_eval_modes"] == {"profanity": "heldout_eval"}
    aurocs_norm = [lr["concepts"]["profanity"]["auroc"] for lr in rep_normal["layers"]]
    aurocs_swap = [lr["concepts"]["profanity"]["auroc"] for lr in rep_swapped["layers"]]
    # Because direction comes solely from train_pairs, swapping eval_pairs labels must invert AUROC -> 1 - AUROC.
    assert any(abs(a - 0.5) > 0.05 for a in aurocs_norm)
    for a_n, a_s in zip(aurocs_norm, aurocs_swap, strict=True):
        assert a_n + a_s == pytest.approx(1.0, abs=1e-4)
