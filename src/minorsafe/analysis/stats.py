"""Clustered bootstrap confidence intervals, paired significance tests, and Holm-Bonferroni correction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import wilcoxon


def clustered_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float, float]:
    """Compute (mean, ci_low, ci_high) via clustered bootstrap over example IDs."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
        boot_means = np.mean(arr[idx], axis=1)
    else:
        cl_arr = np.asarray(clusters)
        unique_cl, inverse = np.unique(cl_arr, return_inverse=True)
        n_cl = unique_cl.size
        if n_cl <= 1:
            return mean_val, mean_val, mean_val
        cl_sums = np.bincount(inverse, weights=arr, minlength=n_cl)
        cl_counts = np.bincount(inverse, minlength=n_cl)
        sampled = rng.integers(0, n_cl, size=(n_boot, n_cl))
        tot_sums = np.sum(cl_sums[sampled], axis=1)
        tot_counts = np.sum(cl_counts[sampled], axis=1)
        boot_means = tot_sums / np.maximum(tot_counts, 1)

    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return mean_val, lo, hi


def paired_significance_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    seed: int = 13,
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank / bootstrap test between matched evaluation vectors."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diffs = arr_a - arr_b
    diff_mean, ci_lo, ci_hi = clustered_bootstrap_ci(diffs, n_boot=500, seed=seed)

    nonzero = diffs[np.abs(diffs) > 1e-12]
    if nonzero.size == 0:
        p_val = 1.0
    else:
        try:
            _, p_val = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
            p_val = float(p_val)
        except ValueError:
            p_val = 0.0 if abs(diff_mean) > 1e-6 else 1.0

    return {
        "diff_mean": diff_mean,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "p_value": p_val,
    }


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down multiple-testing correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted

