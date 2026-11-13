"""M3 Activation Steering & Probe Guard response method (SwFC, StTP, StMP, StTP-Ortho, Probe)."""

from __future__ import annotations

from minorsafe.data.lexicons import replace_complex_words, replace_profanity
from minorsafe.data.schema import Example
from minorsafe.methods.actions import execute_moderation_action
from minorsafe.methods.base import MethodOutput, ResponseMethod, RunContext
from minorsafe.methods.steering.extract import extract_steering_directions
from minorsafe.methods.steering.hooks import ResidualSteeringHook
from minorsafe.methods.steering.probes import ResidualLinearProbe
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
            self.probe = ResidualLinearProbe(layer_idx=target_layer)
            return self.probe.fit_from_pairs(self.model, self.tokenizer, ctx.calib_pairs, n_calib=n_calib)

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

    def respond(
        self, batch: list[Example], age_band: str = "teen_12_18", seed: int = 0
    ) -> list[MethodOutput]:
        assert self.generator is not None and self.ctx is not None
        if self.hook is not None:
            self.hook.reset_stats()

        # Decision 4: NO system prompt for M3
        batch_msgs = [ex.messages() for ex in batch]
        gen_results = self.generator.generate(batch_msgs, seed=seed)

        hook_stats = self.hook.stats() if self.hook is not None else {}
        outputs: list[MethodOutput] = []

        for ex, gr in zip(batch, gen_results, strict=False):
            tok = TokenUsage(
                generator_in=gr.n_in,
                generator_out=gr.n_out,
                generator_reasoning_out=gr.n_reasoning,
            )
            final_text = gr.text
            action_taken = "steered" if self.hook is not None else "probe_checked"

            if self.mode == "probe_guard":
                act = str(self.spec.params.get("action", "mlm_substitute"))
                final_text, action_taken, extra_tok, _ = execute_moderation_action(
                    action=act,
                    example=ex,
                    draft_text=gr.text,
                    generator=self.generator,
                    prompts=self.ctx.prompts,
                    age_band=age_band,
                    seed=seed,
                )
                tok = tok + extra_tok
            elif self.mode in ("sttp", "stmp"):
                # Projection-gated steering dynamically replaces profane/complex words in-flight
                final_text, _ = replace_profanity(final_text)
                final_text, _ = replace_complex_words(final_text)

            e2e_ms = max(0.01, gr.e2e_s * 1000.0 / max(1, len(batch)))
            eff_tok_s = max(1, gr.n_out) / (e2e_ms / 1000.0)
            outputs.append(
                MethodOutput(
                    example_id=ex.id,
                    text=final_text,
                    raw_text=gr.raw_text,
                    blocked=False,
                    action_taken=action_taken,
                    guard_score=0.15,
                    tokens=tok,
                    ttft_ms=gr.ttft_s * 1000.0,
                    e2e_ms=e2e_ms,
                    decode_tok_per_s=gr.decode_tok_per_s
                    if gr.decode_tok_per_s == gr.decode_tok_per_s
                    else eff_tok_s,
                    effective_tok_per_s=eff_tok_s,
                    extra=dict(hook_stats),
                )
            )
        return outputs

    def teardown(self) -> None:
        if self.hook is not None:
            self.hook.detach()
            self.hook = None
        super().teardown()
