"""Phase 5.4.4 & Phase 8 unit tests: Latent Tomography, subcultural entanglement matrix, and Logit Lens."""

from __future__ import annotations

import numpy as np
import pytest
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
