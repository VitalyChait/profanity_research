"""M3 Activation Steering & Probe Guard response method (SwFC, StTP, StMP, StTP-Ortho, Probe).

No hidden lexicon post-processing: the returned text is exactly what the steered model generated
(unless the explicit opt-in ``post_filter`` param is set for a separately named hybrid arm).
"""

from __future__ import annotations

import time

from minorsafe.data.schema import Example
from minorsafe.methods.actions import MLM_SUBSTITUTE_REASON, execute_moderation_action
from minorsafe.methods.base import MethodOutput, ResponseMethod, RunContext
from minorsafe.methods.steering.extract import extract_steering_directions
from minorsafe.methods.steering.hooks import ResidualSteeringHook
from minorsafe.methods.steering.probes import ResidualLinearProbe
from minorsafe.placeholders import require_real_backend
from minorsafe.registry import get_decoder_layers
from minorsafe.telemetry.cost import SetupReport
from minorsafe.telemetry.tokens import TokenUsage


class SteeringResponseMethod(ResponseMethod):
    def __init__(self, spec):
        super().__init__(spec)
        self.mode: str = str(spec.params.get("mode", "sttp"))
        self.estimator: str = str(spec.params.get("estimator", "pca"))
        self.orthogonal_concept: str | None = spec.params.get("orthogonal_concept", "slang_benign")
        self.pooling: str = str(spec.params.get("pooling", "mean_completion"))
        self.alpha: float = float(spec.params.get("alpha", 3.0))
        self.target_offset_sigma: float = float(spec.params.get("target_offset_sigma", 0.5))
        self.concepts: list[str] = list(spec.params.get("concepts", ["profanity", "lexical_complexity"]))
        self.action: str = str(spec.params.get("action", "fallback"))
        self.hook: ResidualSteeringHook | None = None
        self.probe: ResidualLinearProbe | None = None

    def setup(self, ctx: RunContext) -> SetupReport:
        self._init_generator(ctx)
        assert self.model is not None and self.tokenizer is not None
        n_calib = int(self.spec.params.get("n_calib", 50))
        layers = get_decoder_layers(self.model)
        rel_depth = float(self.spec.params.get("layer_rel_depth", 0.50))
        target_layer = max(0, min(len(layers) - 1, int(round(rel_depth * (len(layers) - 1)))))

        if self.mode == "probe_guard":
            if self.action == "mlm_substitute":
                require_real_backend(ctx.hardware, "mlm_substitute action", MLM_SUBSTITUTE_REASON)
            self.probe = ResidualLinearProbe(
                layer_idx=target_layer,
                pooling=self.pooling,
                threshold=float(self.spec.params.get("threshold", 0.5)),
            )
            if self.probe.is_prompt_level:
                rep = self.probe.fit_from_examples(
                    self.model, self.tokenizer, ctx.calib_examples, n_calib=n_calib, method_name=self.name
                )
            else:
                safety_pairs = [p for p in ctx.calib_pairs if p.concept in set(self.concepts)]
                rep = self.probe.fit_from_pairs(
                    self.model, self.tokenizer, safety_pairs, n_calib=n_calib, method_name=self.name
                )
            return rep

        directions, rep = extract_steering_directions(
            model=self.model,
            tokenizer=self.tokenizer,
            pairs=ctx.calib_pairs,
            concepts=self.concepts,
            layer_indices=[target_layer],
            estimator=self.estimator,
            orthogonal_concept=self.orthogonal_concept,
            pooling=self.pooling,
            target_offset_sigma=self.target_offset_sigma,
            n_calib=n_calib,
        )
        rep.method = self.name
        self.hook = ResidualSteeringHook(
            directions=directions,
            mode=self.mode,
            alpha=self.alpha,
            prefill=False,
        )
        self.hook.attach(layers[target_layer])
        return rep

    def _respond_probe_guard(self, batch: list[Example], age_band: str, seed: int) -> list[MethodOutput]:
        """Probe-gated moderation: the action runs only when the probe flags the prompt/response."""
        assert self.generator is not None and self.ctx is not None and self.probe is not None
        probe = self.probe
        outputs: list[MethodOutput] = []
        for ex in batch:
            t0 = time.perf_counter()
            tok = TokenUsage()
            gr = self.generator.generate([ex.messages()], seed=seed)[0]
            tok.generator_in += gr.n_in
            tok.generator_out += gr.n_out
            tok.generator_reasoning_out += gr.n_reasoning

            # NOTE: scoring is a separate forward pass (counted as guard_in tokens); fusing it into the
            # generation prefill (zero extra tokens, plan §5.4.3) is future work.
            score, n_probe_tok = probe.score_text(self.model, self.tokenizer, ex.last_user, gr.text)
            tok.guard_in += n_probe_tok
            final_text, action_taken = gr.text, "none"

            if score >= probe.threshold:

                def _still_unsafe(text: str, _ex: Example = ex) -> bool:
                    s, _ = probe.score_text(self.model, self.tokenizer, _ex.last_user, text)
                    return s >= probe.threshold

                final_text, action_taken, extra_tok, _ = execute_moderation_action(
                    action=self.action,
                    example=ex,
                    draft_text=gr.text,
                    generator=self.generator,
                    prompts=self.ctx.prompts,
                    age_band=age_band,
                    seed=seed,
                    is_unsafe=_still_unsafe,
                )
                tok = tok + extra_tok

            final_text, n_filtered = self.apply_post_filter(final_text)
            e2e_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)
            eff_tok_s = max(1, gr.n_out) / (e2e_ms / 1000.0)
            extra = {"probe_score": score, "probe_flagged": bool(score >= probe.threshold)}
            if self.post_filter:
                extra["post_filter_replacements"] = n_filtered
            outputs.append(
                MethodOutput(
                    example_id=ex.id,
                    text=final_text,
                    raw_text=gr.raw_text,
                    blocked=(action_taken == "fallback"),
                    action_taken=action_taken,
                    guard_score=score,
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

    def respond(
        self, batch: list[Example], age_band: str = "teen_12_18", seed: int = 0
    ) -> list[MethodOutput]:
        assert self.generator is not None and self.ctx is not None
        if self.mode == "probe_guard":
            return self._respond_probe_guard(batch, age_band, seed)

        if self.hook is not None:
            self.hook.reset_stats()

        # Decision 4: NO system prompt for M3
        batch_msgs = [ex.messages() for ex in batch]
        gen_results = self.generator.generate(batch_msgs, seed=seed)

        # Hook counters are batch-level; they are recorded identically on each row of the batch.
        hook_stats = self.hook.stats() if self.hook is not None else {}
        outputs: list[MethodOutput] = []

        for ex, gr in zip(batch, gen_results, strict=True):
            tok = TokenUsage(
                generator_in=gr.n_in,
                generator_out=gr.n_out,
                generator_reasoning_out=gr.n_reasoning,
            )
            final_text, n_filtered = self.apply_post_filter(gr.text)
            extra = dict(hook_stats)
            if self.post_filter:
                extra["post_filter_replacements"] = n_filtered

            e2e_ms = max(0.01, gr.e2e_s * 1000.0 / max(1, len(batch)))
            eff_tok_s = max(1, gr.n_out) / (e2e_ms / 1000.0)
            outputs.append(
                MethodOutput(
                    example_id=ex.id,
                    text=final_text,
                    raw_text=gr.raw_text,
                    blocked=False,
                    action_taken="steered",
                    guard_score=None,  # steering is not a detector
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
        if self.hook is not None:
            self.hook.detach()
            self.hook = None
        self.probe = None
        super().teardown()
