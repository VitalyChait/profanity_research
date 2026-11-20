"""Bidirectional Encoder safety guard (zero-shot toxicity encoders and fine-tuned ModernBERT/EuroBERT)."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from minorsafe.config import ModelSpec
from minorsafe.data.schema import ContrastivePair, Example
from minorsafe.methods.guards.train_encoder import train_encoder_guard
from minorsafe.registry import free_model, load_encoder_classifier
from minorsafe.telemetry.cost import SetupReport


class EncoderGuard:
    """Real HuggingFace sequence-classification guard with calibrated threshold and token attribution."""

    def __init__(self, model, tokenizer, threshold: float = 0.50, device: str = "cpu") -> None:
        self.model = model
        self.tok = tokenizer
        self.threshold = float(threshold)
        self.device = device
        self._unsafe_idx = self._infer_unsafe_index()

    def _infer_unsafe_index(self) -> int | None:
        """Infer the unsafe/toxic class index for binary heads, or None for multi-label toxicity heads."""
        cfg = getattr(self.model, "config", None)
        num_labels = int(getattr(cfg, "num_labels", 2))
        if num_labels == 1:
            return 0
        if num_labels > 2 or getattr(cfg, "problem_type", "") == "multi_label_classification":
            return None
        id2label = getattr(cfg, "id2label", {}) or {}
        for idx, name in id2label.items():
            low = str(name).lower()
            if any(k in low for k in ("unsafe", "toxic", "hate", "harm", "label_1")):
                return int(idx)
        for idx, name in id2label.items():
            low = str(name).lower()
            if any(k in low for k in ("safe", "neutral", "benign", "non-toxic", "nontoxic", "label_0")):
                return 1 - int(idx)
        return 1

    @classmethod
    def from_spec(
        cls,
        model_spec: ModelSpec,
        train_mode: str,
        calib_pairs: list[ContrastivePair],
        calib_examples: list[Example],
        dev_examples: list[Example],
        artifact_dir: Path,
        n_calib: int = 200,
        epochs: int = 2,
        lr: float = 3.0e-5,
        device: str = "cuda",
        quant: str = "bf16",
        method_name: str = "m1_encoder_guard",
    ) -> tuple[EncoderGuard, SetupReport]:
        if train_mode == "zeroshot":
            model, tok, _ = load_encoder_classifier(model_spec, num_labels=None, device=device, quant=quant)
            rep = SetupReport(method=method_name, setup_key=f"{method_name}_{model_spec.alias}_zeroshot")
            guard = cls(model=model, tokenizer=tok, threshold=0.50, device=device)
            # Zero-shot calibration: optionally tune threshold on dev prompts (inference-only, no weight updates).
            dev_labeled = [e for e in dev_examples if e.label_unsafe is not None]
            if len({int(bool(e.label_unsafe)) for e in dev_labeled}) >= 2:
                from minorsafe.eval.detection import calibrate_f1_threshold

                scores, _ = guard.score_texts([e.last_user for e in dev_labeled])
                guard.threshold = calibrate_f1_threshold(
                    [int(bool(e.label_unsafe)) for e in dev_labeled], scores
                )
            return guard, rep

        ckpt_dir, threshold, rep = train_encoder_guard(
            model_spec=model_spec,
            calib_pairs=calib_pairs,
            calib_examples=calib_examples,
            dev_examples=dev_examples,
            artifact_dir=artifact_dir,
            n_calib=n_calib,
            epochs=epochs,
            lr=lr,
            device=device,
            quant=quant,
        )
        rep.method = method_name
        model, tok, _ = load_encoder_classifier(str(ckpt_dir), num_labels=None, device=device, quant=quant)
        return cls(model=model, tokenizer=tok, threshold=threshold, device=device), rep

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tok(text, add_special_tokens=True, truncation=True, max_length=512)["input_ids"])

    @torch.no_grad()
    def score_texts(self, texts: list[str], batch_size: int = 32) -> tuple[list[float], list[int]]:
        """Return (P(unsafe) per text, exact encoder token count per text)."""
        if not texts:
            return [], []
        self.model.eval()
        scores: list[float] = []
        tok_counts: list[int] = []
        for i in range(0, len(texts), batch_size):
            chunk = [t or "" for t in texts[i : i + batch_size]]
            enc = self.tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            counts = enc["attention_mask"].sum(dim=1).tolist()
            tok_counts.extend(int(c) for c in counts)
            enc = enc.to(self.device)
            logits = self.model(**enc).logits.float()
            if self._unsafe_idx is None:
                # Multi-label toxicity heads (e.g. unitary/toxic-bert)
                p_unsafe = torch.sigmoid(logits).max(dim=-1).values
            elif logits.shape[-1] == 1:
                p_unsafe = torch.sigmoid(logits[:, 0])
            else:
                p_unsafe = torch.softmax(logits, dim=-1)[:, self._unsafe_idx]
            scores.extend(float(x) for x in p_unsafe.cpu().tolist())
        return scores, tok_counts

    def attribute_spans(self, text: str, top_k: int = 4) -> list[str]:
        """Leave-one-word-out log-odds drop to identify offending spans without relying on a lexicon."""
        words = (text or "").split()
        if len(words) < 2:
            return []
        base_prob = self.score_texts([text])[0][0]
        if base_prob < self.threshold:
            return []
        base_logit = math.log(max(1e-6, base_prob) / max(1e-6, 1.0 - base_prob))
        candidates = [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
        loo_probs, _ = self.score_texts(candidates, batch_size=32)
        drops: list[tuple[float, str]] = []
        for w, p in zip(words, loo_probs, strict=True):
            loo_logit = math.log(max(1e-6, p) / max(1e-6, 1.0 - p))
            drop = base_logit - loo_logit
            if drop > 0.25:
                drops.append((drop, w.strip(".,!?;:\"'()[]{}")))
        drops.sort(key=lambda x: x[0], reverse=True)
        out: list[str] = []
        for _, w in drops:
            if w and w not in out:
                out.append(w)
            if len(out) >= top_k:
                break
        return out

    def close(self) -> None:
        if self.model is not None or self.tok is not None:
            free_model(self.model, self.tok)
        self.model = None
        self.tok = None
