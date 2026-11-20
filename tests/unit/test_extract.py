"""Phase 5.4.1 unit tests: CAA / PCA extraction, sociolect orthogonalization, and 1D LogReg hyperplane."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minorsafe.methods.steering.extract import (
    calibrate_hyperplane_from_activations,
    extract_or_load_directions,
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


def test_token_level_hyperplane_calibration_and_effective_ncalib_capping(
    dummy_lm, dummy_tokenizer, sample_pairs, tmp_path
):
    rng = np.random.default_rng(7)
    d = 16
    true_dir = np.zeros(d, dtype=np.float32)
    true_dir[0] = 1.0

    # Tight pooled pair clusters vs broader single-token distribution
    X_pos = 0.05 * rng.normal(size=(20, d)) + 2.0 * true_dir
    X_neg = 0.05 * rng.normal(size=(20, d)) - 2.0 * true_dir
    tok_X_pos = 0.50 * rng.normal(size=(100, d)) + 1.0 * true_dir
    tok_X_neg = 0.50 * rng.normal(size=(100, d)) - 1.0 * true_dir

    cal = calibrate_hyperplane_from_activations(
        X_pos,
        X_neg,
        concept="profanity",
        layer_idx=1,
        estimator="pca",
        target_offset_sigma=0.5,
        token_X_pos=tok_X_pos,
        token_X_neg=tok_X_neg,
    )
    assert abs(cal.mu_pos - 1.0) < 0.25
    assert cal.sigma_pos > 0.25
    assert cal.s_l > cal.mu_pos

    # Requesting n_calib=100 when sample_pairs only has 12 per concept records capped n_calib_effective=12
    dirs, rep = extract_or_load_directions(
        model=dummy_lm,
        tokenizer=dummy_tokenizer,
        model_alias="tiny_llama",
        quant="fp32",
        pairs=sample_pairs,
        concepts=["profanity"],
        layer_indices=[1],
        artifact_dir=tmp_path / "artifacts",
        target_offset_sigma=0.5,
        n_calib=100,
    )
    assert rep.stages[0].examples == 12
    assert rep.stages[0].extra["n_calib_requested"] == 100
    assert rep.stages[0].extra["n_calib_effective"] == 12

    # Cached reload with new target_offset_sigma must recompute s_l
    dirs_reloaded, rep_cached = extract_or_load_directions(
        model=dummy_lm,
        tokenizer=dummy_tokenizer,
        model_alias="tiny_llama",
        quant="fp32",
        pairs=sample_pairs,
        concepts=["profanity"],
        layer_indices=[1],
        artifact_dir=tmp_path / "artifacts",
        target_offset_sigma=1.5,
        n_calib=100,
    )
    assert rep_cached.cached is True
    assert dirs_reloaded[0].s_l > dirs[0].s_l
