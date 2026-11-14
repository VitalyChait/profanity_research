"""Internal residual-stream linear probes for prompt/response moderation (Phase 5.4.3).

Two training regimes, selected by ``pooling``:

* ``last_prompt_token`` (prompt-level, plan §5.4.3): trained on labeled calibration *prompts*
  (``Example.label_unsafe``) at the last prompt token; scored on the incoming prompt.
* ``mean_completion`` (response-level): trained on safety-concept contrastive pairs (positive = safe,
  negative = unsafe completion); scored on the generated response.

Pair-based training with ``last_prompt_token`` is rejected: both completions share the prompt, so the two
classes would have identical activations.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from minorsafe.data.schema import ContrastivePair, Example
from minorsafe.methods.steering.extract import capture_pair_activations, pooled_activations
from minorsafe.telemetry.cost import SetupReport


class ResidualLinearProbe:
    """Logistic regression probe trained on residual stream activations at one decoder layer."""

    def __init__(self, layer_idx: int, pooling: str = "mean_completion", threshold: float = 0.5):
        if pooling not in ("last_prompt_token", "mean_completion"):
            raise ValueError(f"Unsupported probe pooling {pooling!r}")
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.clf: LogisticRegression | None = None
        self.threshold: float = float(threshold)

    @property
    def is_prompt_level(self) -> bool:
        return self.pooling == "last_prompt_token"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            raise ValueError("Probe calibration data contains a single class; cannot fit a classifier.")
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=13)
        self.clf.fit(X, y)

    def fit_from_pairs(
        self, model, tokenizer, pairs: list[ContrastivePair], n_calib: int = 100, method_name: str = "probe"
    ) -> SetupReport:
        """Response-level probe. Label 1 = unsafe (negative completion), 0 = safe (positive completion)."""
        if self.is_prompt_level:
            raise ValueError("Pair-based probe training requires pooling='mean_completion'.")
        subset = pairs[:n_calib]
        if not subset:
            raise ValueError("No contrastive pairs available for probe calibration.")
        rep = SetupReport(method=method_name, setup_key=f"probe_resp_L{self.layer_idx}_N{n_calib}")
        with rep.stage("fit_linear_probe", examples=len(subset)) as st:
            acts, tok_cnt = capture_pair_activations(
                model, tokenizer, subset, layer_indices=[self.layer_idx], pooling="mean_completion"
            )
            st.tokens += tok_cnt
            xp, xn = acts[self.layer_idx]
            X = np.vstack([xp, xn])
            y = np.array([0] * len(xp) + [1] * len(xn), dtype=int)
            self._fit(X, y)
        return rep

    def fit_from_examples(
        self, model, tokenizer, examples: list[Example], n_calib: int = 100, method_name: str = "probe"
    ) -> SetupReport:
        """Prompt-level probe on labeled calibration prompts (label 1 = ``label_unsafe``)."""
        if not self.is_prompt_level:
            raise ValueError("Example-based probe training requires pooling='last_prompt_token'.")
        subset = [e for e in examples if e.label_unsafe is not None][:n_calib]
        if not subset:
            raise ValueError("No labeled calibration examples available for prompt-level probe.")
        rep = SetupReport(method=method_name, setup_key=f"probe_prompt_L{self.layer_idx}_N{n_calib}")
        with rep.stage("fit_linear_probe", examples=len(subset)) as st:
            feats, labels = [], []
            for ex in subset:
                acts, n_tok = pooled_activations(
                    model, tokenizer, ex.last_user, "", [self.layer_idx], pooling="last_prompt_token"
                )
                st.tokens += n_tok
                feats.append(acts[self.layer_idx])
                labels.append(int(bool(ex.label_unsafe)))
            self._fit(np.vstack(feats), np.array(labels, dtype=int))
        return rep

    def score_activation(self, h_vec: np.ndarray) -> float:
        if self.clf is None:
            raise RuntimeError("Probe has not been fitted.")
        return float(self.clf.predict_proba(h_vec.reshape(1, -1))[0, 1])

    def score_text(self, model, tokenizer, prompt: str, completion: str = "") -> tuple[float, int]:
        """Return (P(unsafe), tokens processed) for a prompt (prompt-level) or prompt+response."""
        comp = "" if self.is_prompt_level else completion
        acts, n_tok = pooled_activations(model, tokenizer, prompt, comp, [self.layer_idx], pooling=self.pooling)
        return self.score_activation(acts[self.layer_idx]), n_tok
