"""Phase 5.4.1 unit tests: CAA / PCA extraction, sociolect orthogonalization, and 1D LogReg hyperplane."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from minorsafe.methods.steering.extract import (
    calibrate_hyperplane_from_activations,
    extract_steering_directions,
    orthogonalize_vector,
)

pytestmark = pytest.mark.unit


def test_recover_planted_direction_caa_and_pca():
    rng = np.random.default_rng(42)
    d = 32
    n = 80
    true_dir = rng.normal(size=d)
    true_dir = true_dir / np.linalg.norm(true_dir)

    noise_pos = 0.15 * rng.normal(size=(n, d))
    noise_neg = 0.15 * rng.normal(size=(n, d))
    X_pos = noise_pos + 1.5 * true_dir
    X_neg = noise_neg - 1.5 * true_dir

    for estimator in ("caa", "pca"):
        cal = calibrate_hyperplane_from_activations(
            X_pos, X_neg, concept="profanity", layer_idx=2, estimator=estimator, target_offset_sigma=0.5
        )
        v_hat = cal.vector.numpy()
        cos_sim = float(np.dot(v_hat, true_dir))
        assert cos_sim > 0.97, f"{estimator} cosine similarity too low: {cos_sim}"
        assert cal.mu_pos > cal.m_l > cal.mu_neg
        assert cal.s_l > cal.m_l
        assert cal.separability_auroc > 0.99


def test_gram_schmidt_sociolect_orthogonalization():
    v_slang = np.zeros(16, dtype=np.float32)
    v_slang[0] = 1.0

    # Target vector entangled 60% with benign slang direction and 80% with true harm axis
    v_entangled = np.zeros(16, dtype=np.float32)
    v_entangled[0] = 0.6
    v_entangled[1] = 0.8

    v_ortho = orthogonalize_vector(v_entangled, v_slang)
    assert abs(float(np.dot(v_ortho, v_slang))) < 1e-6
    assert np.isclose(np.linalg.norm(v_ortho), 1.0, atol=1e-6)
    assert np.isclose(v_ortho[1], 1.0, atol=1e-5)


def test_degenerate_activations_hyperplane_fallback_never_nans():
    # Identical positive and negative activations -> slope w ~ 0 -> fallback must prevent NaN/Inf
    X_pos = np.ones((10, 8), dtype=np.float32)
    X_neg = np.ones((10, 8), dtype=np.float32)
    cal = calibrate_hyperplane_from_activations(
        X_pos, X_neg, concept="profanity", layer_idx=0, estimator="caa"
    )
    assert np.isfinite(cal.m_l)
    assert np.isfinite(cal.s_l)
    assert np.isclose(torch.norm(cal.vector).item(), 1.0, atol=1e-5)


def test_extract_steering_directions_on_dummy_model(dummy_lm, dummy_tokenizer, sample_pairs):
    directions, report = extract_steering_directions(
        model=dummy_lm,
        tokenizer=dummy_tokenizer,
        pairs=sample_pairs,
        concepts=["profanity", "lexical_complexity"],
        layer_indices=[1, 2],
        estimator="ortho",
        orthogonal_concept="slang_benign",
        pooling="mean_completion",
    )
    assert len(directions) == 2  # 2 concepts at chosen layer
    for cd in directions:
        assert cd.vector.shape == (dummy_lm.hidden_size,)
        assert np.isfinite(cd.m_l) and np.isfinite(cd.s_l)
    assert report.tokens > 0
