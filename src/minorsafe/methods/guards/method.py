"""External Guardrails (M1-Enc Bidirectional Encoder & M1-Dec Autoregressive Decoder)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from minorsafe.config import ModelSpec
from minorsafe.data.lexicons import find_complex_spans, find_profanity_spans
from minorsafe.data.obfuscation import detect_obfuscated_profanity
from minorsafe.data.schema import Example
from minorsafe.eval.detection import calibrate_f1_threshold
from minorsafe.methods.actions import MLMRecomposer, execute_moderation_action
from minorsafe.methods.base import MethodOutput, ResponseMethod, RunContext
from minorsafe.methods.guards.decoder_guard import DecoderGuard
from minorsafe.methods.guards.encoder_guard import EncoderGuard
from minorsafe.telemetry.cost import SetupReport
from minorsafe.telemetry.tokens import TokenUsage

_ZEROSHOT_ENCODER_ALIASES = {"toxic_bert", "roberta_tox"}


def _resolve_guard_spec(ctx: RunContext, guard_arch: str, train_mode: str, requested: str) -> ModelSpec:
    models = ctx.models
    if requested != "auto":
        if requested not in models:
            raise KeyError(f"Requested guard_model {requested!r} not found in configs/models.yaml")
        return models[requested]

    if guard_arch == "decoder":
        for alias in ctx.hardware.decoder_guards:
            if alias in models:
                return models[alias]
        raise ValueError(f"No decoder_guards configured on hardware profile {ctx.hardware.name!r}")

    pool = [a for a in ctx.hardware.encoder_guards if a in models]
    if train_mode == "zeroshot":
        for alias in pool:
            if alias in _ZEROSHOT_ENCODER_ALIASES:
                return models[alias]
    else:
        # Fine-tuned mode should start from a base encoder (ModernBERT/EuroBERT), not a pretrained toxic head.
        for alias in pool:
            if alias not in _ZEROSHOT_ENCODER_ALIASES:
                return models[alias]
    if pool:
        return models[pool[0]]
    raise ValueError(f"No encoder_guards configured on hardware profile {ctx.hardware.name!r}")


def _resolve_mlm_spec(ctx: RunContext, requested: str) -> ModelSpec:
    models = ctx.models
    if requested != "auto":
        if requested not in models:
            raise KeyError(f"Requested mlm_model {requested!r} not found in configs/models.yaml")
        return models[requested]
    for alias in ctx.hardware.encoder_mlms:
        if alias in models:
            return models[alias]
    raise ValueError(f"No encoder_mlms configured on hardware profile {ctx.hardware.name!r}")


class GuardResponseMethod(ResponseMethod):
    def __init__(self, spec):
        super().__init__(spec)
        self.guard_arch: str = str(spec.params.get("guard_arch", "encoder"))
        self.train_mode: str = str(spec.params.get("train_mode", "finetuned"))
        self.guard_model_alias: str = str(spec.params.get("guard_model", "auto"))
        self.mlm_model_alias: str = str(spec.params.get("mlm_model", "auto"))
        self.placement: str = str(spec.params.get("placement", "both"))
        self.action: str = str(spec.params.get("action", "mlm_substitute"))
        self.max_regenerations: int = int(spec.params.get("max_regenerations", 2))
        self.threshold: float = 0.50

        # Real HF backends
        self._enc_guard: EncoderGuard | None = None
        self._dec_guard: DecoderGuard | None = None
        self._mlm: MLMRecomposer | None = None
        # Offline CPU test-fixture mode only (DummyCausalLM + allow_placeholders=True)
        self._cpu_fixture_mode = False
        self._vec: HashingVectorizer | None = None
        self._clf: LogisticRegression | None = None

    def _setup_cpu_fixture(self, ctx: RunContext, n_calib: int) -> SetupReport:
        self._cpu_fixture_mode = True
        rep = SetupReport(
            method=self.name,
            setup_key=f"{self.name}_{self.guard_arch}_{self.train_mode}_{n_calib}_cpu_fixture",
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
        dev = [e for e in ctx.dev_examples if e.label_unsafe is not None]
        dev_labels = [int(bool(e.label_unsafe)) for e in dev]
        if len(set(dev_labels)) >= 2:
            dev_scores = [self._score_prompt_or_text(e.last_user)[0][0] for e in dev]
            self.threshold = calibrate_f1_threshold(dev_labels, dev_scores)
        return rep

    def setup(self, ctx: RunContext) -> SetupReport:
        self._init_generator(ctx)
        n_calib = int(self.spec.params.get("n_calib", 100))

        # Only use lightweight in-memory scoring when running offline dummy-model unit/integration tests.
        if ctx.hardware.allow_placeholders and ctx.preloaded_generator is not None:
            return self._setup_cpu_fixture(ctx, n_calib)

        self._cpu_fixture_mode = False
        guard_spec = _resolve_guard_spec(ctx, self.guard_arch, self.train_mode, self.guard_model_alias)

        if self.guard_arch == "encoder":
            self._enc_guard, rep = EncoderGuard.from_spec(
                model_spec=guard_spec,
                train_mode=self.train_mode,
                calib_pairs=ctx.calib_pairs,
                calib_examples=ctx.calib_examples,
                dev_examples=ctx.dev_examples,
                artifact_dir=ctx.artifact_dir,
                n_calib=n_calib,
                epochs=int(self.spec.params.get("epochs", 2)),
                lr=float(self.spec.params.get("lr", 3.0e-5)),
                device=ctx.hardware.device,
                quant=ctx.quant,
                method_name=self.name,
            )
            self.threshold = self._enc_guard.threshold
        elif self.guard_arch == "decoder":
            self._dec_guard = DecoderGuard.from_spec(
                model_spec=guard_spec,
                quant=ctx.quant,
                device=ctx.hardware.device,
                threshold=float(self.spec.params.get("threshold", 0.50)),
            )
            self.threshold = self._dec_guard.threshold
            rep = SetupReport(
                method=self.name,
                setup_key=f"{self.name}_{guard_spec.alias}_{ctx.quant}",
            )
        else:
            raise ValueError(f"Unsupported guard_arch: {self.guard_arch!r}")

        if self.action == "mlm_substitute":
            mlm_spec = _resolve_mlm_spec(ctx, self.mlm_model_alias)
            self._mlm = MLMRecomposer.from_spec(mlm_spec, device=ctx.hardware.device)

        return rep

    def _score_prompt_or_text(
        self, prompt: str, response: str | None = None
    ) -> tuple[tuple[float, int, int], bool]:
        """Return ((p_unsafe, in_tokens, out_tokens), is_encoder)."""
        if self._enc_guard is not None:
            txt = f"{prompt}\n{response}" if response is not None else prompt
            probs, toks = self._enc_guard.score_texts([txt])
            return (probs[0], toks[0], 0), True
        if self._dec_guard is not None:
            p, n_in, n_out = self._dec_guard.score_pair(prompt, response=response)
            return (p, n_in, n_out), False

        # Offline CPU dummy fixture fallback only
        txt = response if response is not None else prompt
        n_tok = max(1, len((txt or "").split()))
        has_prof = bool(find_profanity_spans(txt) or detect_obfuscated_profanity(txt))
        has_comp = bool(find_complex_spans(txt))
        base_score = 0.88 if has_prof else (0.65 if has_comp else 0.08)
        if self._clf is not None and self._vec is not None:
            model_prob = float(self._clf.predict_proba(self._vec.transform([txt or ""]))[0, 1])
            score = max(base_score, 0.5 * base_score + 0.5 * model_prob)
        else:
            score = base_score
        is_enc = self.guard_arch == "encoder"
        return (float(np.clip(score, 0.0, 1.0)), n_tok, 0 if is_enc else 4), is_enc

    def respond(
        self, batch: list[Example], age_band: str = "teen_12_18", seed: int = 0
    ) -> list[MethodOutput]:
        assert self.generator is not None and self.ctx is not None
        outputs: list[MethodOutput] = []

        for ex in batch:
            t0 = time.perf_counter()
            tok = TokenUsage()
            prompt_txt = ex.last_user

            in_score = 0.0
            guard_ms = 0.0
            if self.placement in ("input", "both"):
                tg0 = time.perf_counter()
                (in_score, in_toks, in_out_toks), is_enc = self._score_prompt_or_text(prompt_txt)
                guard_ms += (time.perf_counter() - tg0) * 1000.0
                if is_enc:
                    tok.encoder_in += in_toks
                else:
                    tok.guard_in += in_toks
                    tok.guard_out += in_out_toks

                if in_score >= self.threshold:
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
                            extra={
                                "input_guard_score": in_score,
                                "output_guard_score": float("nan"),
                                "guard_threshold": self.threshold,
                                "guard_ms": guard_ms,
                            },
                        )
                    )
                    continue

            # Primary decoder generation (NO system prompt per Decision 4)
            gr = self.generator.generate([ex.messages()], seed=seed)[0]
            tok.generator_in += gr.n_in
            tok.generator_out += gr.n_out
            tok.generator_reasoning_out += gr.n_reasoning

            out_score = 0.0
            if self.placement in ("output", "both"):
                tg1 = time.perf_counter()
                (out_score, out_toks, out_gen_toks), is_enc = self._score_prompt_or_text(
                    prompt_txt, response=gr.text
                )
                guard_ms += (time.perf_counter() - tg1) * 1000.0
                if is_enc:
                    tok.encoder_in += out_toks
                else:
                    tok.guard_in += out_toks
                    tok.guard_out += out_gen_toks

            combined_score = (
                max(in_score, out_score)
                if self.placement == "both"
                else (out_score if self.placement == "output" else in_score)
            )
            final_text = gr.text
            action_taken = "none"

            if self.placement in ("output", "both") and out_score >= self.threshold:
                suspect_words = (
                    self._enc_guard.attribute_spans(gr.text) if self._enc_guard is not None else None
                )
                final_text, action_taken, extra_tok, _ = execute_moderation_action(
                    action=self.action,
                    example=ex,
                    draft_text=gr.text,
                    generator=self.generator,
                    prompts=self.ctx.prompts,
                    age_band=age_band,
                    seed=seed,
                    max_regenerations=self.max_regenerations,
                    is_unsafe=lambda t, _p=prompt_txt: (
                        self._score_prompt_or_text(_p, response=t)[0][0] >= self.threshold
                    ),
                    mlm_recomposer=self._mlm,
                    extra_suspect_words=suspect_words,
                    allow_placeholder_mlm=self._cpu_fixture_mode,
                )
                tok = tok + extra_tok

            final_text, n_filtered = self.apply_post_filter(final_text)
            e2e_ms = max(0.05, (time.perf_counter() - t0) * 1000.0)
            eff_tok_s = max(1, self.generator.count_tokens(final_text)) / (e2e_ms / 1000.0)
            extra = {
                "input_guard_score": in_score,
                "output_guard_score": out_score,
                "guard_threshold": self.threshold,
                "guard_ms": guard_ms,
            }
            if self.post_filter:
                extra["post_filter_replacements"] = n_filtered
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
                    extra=extra,
                )
            )
        return outputs

    def teardown(self) -> None:
        if self._enc_guard is not None:
            self._enc_guard.close()
            self._enc_guard = None
        if self._dec_guard is not None:
            self._dec_guard.close()
            self._dec_guard = None
        if self._mlm is not None:
            self._mlm.close()
            self._mlm = None
        super().teardown()
