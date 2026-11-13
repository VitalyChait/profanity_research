"""Guard and probe binary detection metrics: Accuracy, Macro-F1, Precision, Recall, AUROC, AUPRC, FPR@95TPR."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_detection_metrics(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)
    if len(y_t) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "binary_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auroc": 0.5,
            "auprc": 0.5,
            "fpr_at_95_tpr": 1.0,
        }
    y_p = (y_s >= threshold).astype(int)
    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    bin_f1 = float(f1_score(y_t, y_p, average="binary", zero_division=0))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))

    if len(np.unique(y_t)) >= 2:
        auroc = float(roc_auc_score(y_t, y_s))
        auprc = float(average_precision_score(y_t, y_s))
        fpr, tpr, _ = roc_curve(y_t, y_s)
        idx = np.where(tpr >= 0.95)[0]
        fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0
    else:
        auroc = 1.0 if acc == 1.0 else 0.5
        auprc = 1.0 if acc == 1.0 else 0.5
        fpr95 = 0.0 if acc == 1.0 else 1.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "binary_f1": bin_f1,
        "precision": prec,
        "recall": rec,
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95_tpr": fpr95,
    }


def calibrate_f1_threshold(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float:
    """Pick threshold in [0.10, 0.90] on dev set maximizing Macro-F1."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)
    if len(y_t) == 0:
        return 0.50
    best_t, best_f1 = 0.50, -1.0
    for t in np.linspace(0.15, 0.85, 29):
        f1 = float(f1_score(y_t, (y_s >= t).astype(int), average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

