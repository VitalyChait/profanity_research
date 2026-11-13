"""Internal residual-stream linear probes for zero-token prompt/drift moderation (Phase 5.4.3)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from minorsafe.data.schema import ContrastivePair
from minorsafe.methods.steering.extract import capture_pair_activations
from minorsafe.registry import get_decoder_layers
from minorsafe.telemetry.cost import SetupReport


class ResidualLinearProbe:
    """Logistic regression probe trained on middle-layer residual stream activations."""

    def __init__(self, layer_idx: int = 1):
        self.layer_idx = layer_idx
        self.clf: LogisticRegression | None = None
        self.threshold: float = 0.50

    def fit_from_pairs(
        self, model, tokenizer, pairs: list[ContrastivePair], n_calib: int = 100
    ) -> SetupReport:
        layers = get_decoder_layers(model)
        self.layer_idx = max(0, min(len(layers) - 1, len(layers) // 2))
        rep = SetupReport(method="m3_probe_guard", setup_key=f"probe_L{self.layer_idx}_N{n_calib}")
        with rep.stage("fit_linear_probe", examples=min(len(pairs), n_calib)) as st:
            subset = pairs[:n_calib]
            acts, tok_cnt = capture_pair_activations(
                model, tokenizer, subset, layer_indices=[self.layer_idx], pooling="mean_completion"
            )
            st.tokens += tok_cnt
            xp, xn = acts[self.layer_idx]
            # Label 1 = unsafe (negative completion), Label 0 = safe (positive completion)
            X = np.vstack([xp, xn])
            y = np.array([0] * len(xp) + [1] * len(xn), dtype=int)
            self.clf = LogisticRegression(C=1.0, random_state=13)
            self.clf.fit(X, y)
        return rep

    @torch.no_grad()
    def score_activation(self, h_vec: np.ndarray) -> float:
        if self.clf is None:
            return 0.10
        prob = self.clf.predict_proba(h_vec.reshape(1, -1))[0, 1]
        return float(prob)

