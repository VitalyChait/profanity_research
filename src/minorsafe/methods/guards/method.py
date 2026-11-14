"""External Guardrails (M1-Enc Bidirectional Encoder & M1-Dec Autoregressive Decoder).

PLACEHOLDER: no encoder or decoder guard model is loaded yet. Scoring is a lexicon heuristic blended with a
hashing-vectorizer logistic regression, and encoder vs decoder differ only in token bookkeeping. Gated by
``require_real_backend`` (only ``allow_placeholders`` profiles may run it). See plan §5.2 for the real
ModernBERT / toxic-bert / Llama-Guard-3 / Qwen3Guard implementations.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from minorsafe.data.lexicons import find_complex_spans, find_profanity_spans
from minorsafe.data.obfuscation import detect_obfuscated_profanity
from minorsafe.data.schema import Example
from minorsafe.eval.detection import calibrate_f1_threshold
from minorsafe.methods.actions import execute_moderation_action
from minorsafe.methods.base import MethodOutput, ResponseMethod, RunContext
from minorsafe.placeholders import require_real_backend
from minorsafe.telemetry.cost import SetupReport
from minorsafe.telemetry.tokens import TokenUsage


class GuardResponseMethod(ResponseMethod):
    def __init__(self, spec):
        super().__init__(spec)
        self.guard_arch: str = str(spec.params.get("guard_arch", "encoder"))
        self.train_mode: str = str(spec.params.get("train_mode", "finetuned"))
        self.placement: str = str(spec.params.get("placement", "both"))
        self.action: str = str(spec.params.get("action", "mlm_substitute"))
        self.threshold: float = 0.50
        self._vec: HashingVectorizer | None = None
        self._clf: LogisticRegression | None = None

    def setup(self, ctx: RunContext) -> SetupReport:
        require_real_backend(
            ctx.hardware,
            f"M1 {self.guard_arch} guard",
            "lexicon + hashing-vectorizer scorer; no guard model is loaded",
        )
        self._init_generator(ctx)
        n_calib = int(self.spec.params.get("n_calib", 100))
        rep = SetupReport(
            method=self.name,
            setup_key=f"{self.name}_{self.guard_arch}_{self.train_mode}_{n_calib}",
        )
        if self.guard_arch == "encoder" and self.train_mode == "finetuned":
            with rep.stage("train_encoder_head", examples=n_calib) as st:
                texts: list[str] = []
                labels: list[int] = []
                for p in ctx.calib_pairs[:n_calib]:
                    texts.extend([p.positive, p.negative])
                    labels.extend([0, 1])
                for ex in ctx.calib_examples[:n_calib]:
                    texts.append(ex.last_user)
                    labels.append(1 if ex.label_unsafe else 0)
                if len(set(labels)) >= 2:
                    self._vec = HashingVectorizer(n_features=512, ngram_range=(1, 2), norm="l2")
                    X = self._vec.transform(texts)
                    self._clf = LogisticRegression(C=1.0, random_state=13)
                    self._clf.fit(X, labels)
                st.tokens = sum(len(t.split()) for t in texts)

        # Threshold is calibrated on held-out dev prompts (never on the training data).
        dev = [e for e in ctx.dev_examples if e.label_unsafe is not None]
        dev_labels = [int(bool(e.label_unsafe)) for e in dev]
        if len(set(dev_labels)) >= 2:
            dev_scores = [self._score_text(e.last_user)[0] for e in dev]
            self.threshold = calibrate_f1_threshold(dev_labels, dev_scores)
        return rep

    def _score_text(self, text: str) -> tuple[float, int]:
        n_tok = max(1, len((text or "").split()))
        has_prof = bool(find_profanity_spans(text) or detect_obfuscated_profanity(text))
        has_comp = bool(find_complex_spans(text))
        base_score = 0.88 if has_prof else (0.65 if has_comp else 0.08)

        if self._clf is not None and self._vec is not None:
            X = self._vec.transform([text or ""])
            model_prob = float(self._clf.predict_proba(X)[0, 1])
            score = max(base_score, 0.5 * base_score + 0.5 * model_prob)
        else:
            score = base_score
        return float(np.clip(score, 0.0, 1.0)), n_tok

    def respond(
        self, batch: list[Example], age_band: str = "teen_12_18", seed: int = 0
    ) -> list[MethodOutput]:
        assert self.generator is not None and self.ctx is not None
        outputs: list[MethodOutput] = []

        for ex in batch:
            t0 = time.perf_counter()
            tok = TokenUsage()
            prompt_txt = ex.last_user

            in_score, in_toks = self._score_text(prompt_txt)
            if self.guard_arch == "encoder":
                tok.encoder_in += in_toks
            else:
                tok.guard_in += in_toks
                tok.guard_out += 4

            # Input guard: decided by the guard score alone (ground-truth labels are never consulted).
            if self.placement in ("input", "both") and in_score >= self.threshold:
                fallback_msg = (self.ctx.prompts.get("fallback_responses") or {}).get(
                    age_band,
                    "I can't help with that request, let's explore a safe topic instead!",
                )
                e2e_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)
                outputs.append(
                    MethodOutput(
                        example_id=ex.id,
                        text=fallback_msg,
                        raw_text=fallback_msg,
                        blocked=True,
                        action_taken="fallback",
                        guard_score=in_score,
                        tokens=tok,
                        ttft_ms=e2e_ms * 0.5,
                        e2e_ms=e2e_ms,
                        decode_tok_per_s=0.0,
                        effective_tok_per_s=len(fallback_msg.split()) / (e2e_ms / 1000.0),
                    )
                )
                continue

            # Primary decoder generation (NO system prompt per Decision 4)
            gr = self.generator.generate([ex.messages()], seed=seed)[0]
            tok.generator_in += gr.n_in
            tok.generator_out += gr.n_out
            tok.generator_reasoning_out += gr.n_reasoning

            out_score, out_toks = self._score_text(gr.text)
            if self.guard_arch == "encoder":
                tok.encoder_in += out_toks
            else:
                tok.guard_in += out_toks
                tok.guard_out += 4

            combined_score = max(in_score, out_score)
            final_text = gr.text
            action_taken = "none"

            if self.placement in ("output", "both") and combined_score >= self.threshold:
                final_text, action_taken, extra_tok, _ = execute_moderation_action(
                    action=self.action,
                    example=ex,
                    draft_text=gr.text,
                    generator=self.generator,
                    prompts=self.ctx.prompts,
                    age_band=age_band,
                    seed=seed,
                    is_unsafe=lambda t: self._score_text(t)[0] >= self.threshold,
                )
                tok = tok + extra_tok

            final_text, n_filtered = self.apply_post_filter(final_text)
            e2e_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)
            eff_tok_s = max(1, len(final_text.split())) / (e2e_ms / 1000.0)
            outputs.append(
                MethodOutput(
                    example_id=ex.id,
                    text=final_text,
                    raw_text=gr.raw_text,
                    blocked=(action_taken == "fallback"),
                    action_taken=action_taken,
                    guard_score=combined_score,
                    tokens=tok,
                    ttft_ms=gr.ttft_s * 1000.0,
                    e2e_ms=e2e_ms,
                    decode_tok_per_s=gr.decode_tok_per_s
                    if gr.decode_tok_per_s == gr.decode_tok_per_s
                    else eff_tok_s,
                    effective_tok_per_s=eff_tok_s,
                    extra={"post_filter_replacements": n_filtered} if self.post_filter else {},
                )
            )
        return outputs
