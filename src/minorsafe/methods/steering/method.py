"""M3 Activation Steering & Probe Guard response method (SwFC, StTP, StMP, StTP-Ortho, Probe).

No hidden lexicon post-processing: the returned text is exactly what the steered model generated
(unless the explicit opt-in ``post_filter`` param is set for a separately named hybrid arm).
"""

from __future__ import annotations

import time
from dataclasses import asdict

from minorsafe.data.schema import Example
from minorsafe.methods.actions import MLMRecomposer, execute_moderation_action
from minorsafe.methods.base import MethodOutput, ResponseMethod, RunContext
from minorsafe.methods.guards.method import _resolve_mlm_spec
from minorsafe.methods.steering.extract import (
    apply_target_offset,
    extract_or_load_directions,
    load_steering_artifact,
    steering_setup_key,
)
from minorsafe.methods.steering.hooks import ResidualSteeringHook
from minorsafe.methods.steering.probes import ResidualLinearProbe
from minorsafe.methods.steering.sweep import sweep_steering_hyperparams
from minorsafe.registry import free_model, get_decoder_layers, load_decoder
from minorsafe.telemetry.cost import SetupReport, SetupStage
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
        self.vector_source_quant: str = str(spec.params.get("vector_source_quant", "native"))
        self.use_sweep: bool = bool(spec.params.get("use_sweep", False))
        self.hook: ResidualSteeringHook | None = None
        self.probe: ResidualLinearProbe | None = None
        self._mlm: MLMRecomposer | None = None
        self._cpu_fixture_mode: bool = False

    def setup(self, ctx: RunContext) -> SetupReport:
        self._init_generator(ctx)
        assert self.model is not None and self.tokenizer is not None
        self._cpu_fixture_mode = bool(ctx.hardware.allow_placeholders and ctx.preloaded_generator is not None)
        n_calib = int(self.spec.params.get("n_calib", 50))
        # Keep only the layer count: a live reference to the ModuleList would pin the generator's weights
        # in VRAM when the BF16-transfer path below releases the generator.
        n_layers = len(get_decoder_layers(self.model, layers_path=ctx.generator_spec.layers_path))
        rel_depth = float(self.spec.params.get("layer_rel_depth", 0.50))
        swept_best: dict | None = None
        should_sweep = (
            self.use_sweep
            and self.mode != "probe_guard"
            and bool(ctx.dev_examples)
            and (not ctx.hardware.allow_placeholders or bool(self.spec.params.get("force_sweep", False)))
        )

        sweep_rep: SetupReport | None = None
        if should_sweep:
            swept_best, sweep_rep = sweep_steering_hyperparams(
                model=self.model,
                tokenizer=self.tokenizer,
                gen_spec=ctx.generator_spec,
                calib_pairs=ctx.calib_pairs,
                dev_examples=ctx.dev_examples,
                mode=self.mode,
                concepts=self.concepts,
                estimator=self.estimator,
                orthogonal_concept=self.orthogonal_concept,
                pooling=self.pooling,
                n_calib=n_calib,
                quant=ctx.quant,
                artifact_dir=ctx.artifact_dir,
            )
            rel_depth = float(swept_best.get("layer_rel_depth", rel_depth))
            self.alpha = float(swept_best.get("alpha", self.alpha))
            self.target_offset_sigma = float(swept_best.get("target_offset_sigma", self.target_offset_sigma))

        target_layer = max(0, min(n_layers - 1, int(round(rel_depth * (n_layers - 1)))))

        if self.mode == "probe_guard":
            if self.action == "mlm_substitute" and not self._cpu_fixture_mode:
                mlm_alias = str(self.spec.params.get("mlm_model", "auto"))
                mlm_spec = _resolve_mlm_spec(ctx, mlm_alias)
                self._mlm = MLMRecomposer.from_spec(mlm_spec, device=ctx.hardware.device)

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

        use_artifact_dir = None if ctx.preloaded_generator is not None else ctx.artifact_dir

        if self.vector_source_quant == "bf16" and ctx.quant != "bf16" and ctx.preloaded_generator is None:
            # Transferred BF16 steering vectors onto a quantized model (H10 / quant_grid.yaml)
            bf16_key = steering_setup_key(
                ctx.generator_spec.alias,
                "bf16",
                self.estimator,
                self.pooling,
                [target_layer],
                n_calib,
                self.concepts,
                self.orthogonal_concept,
            )
            cached = load_steering_artifact(ctx.artifact_dir, bf16_key) if ctx.artifact_dir else None
            if cached is not None:
                directions, rep = cached
                # The artifact stores s_l for its extraction-time offset; re-target to this run's offset.
                apply_target_offset(directions, self.target_offset_sigma)
            else:
                # Never hold the quantized generator and a full BF16 copy at once: release the generator,
                # extract on BF16 (which caches the artifact), then reload the quantized generator.
                self._release_generator()
                directions, rep = self._extract_bf16_directions(ctx, target_layer, n_calib)
                self._init_generator(ctx)
        else:
            directions, rep = extract_or_load_directions(
                model=self.model,
                tokenizer=self.tokenizer,
                model_alias=ctx.generator_spec.alias,
                quant=ctx.quant,
                pairs=ctx.calib_pairs,
                concepts=self.concepts,
                layer_indices=[target_layer],
                artifact_dir=use_artifact_dir,
                estimator=self.estimator,
                orthogonal_concept=self.orthogonal_concept,
                pooling=self.pooling,
                target_offset_sigma=self.target_offset_sigma,
                n_calib=n_calib,
            )

        rep.method = self.name
        if swept_best is not None and rep.stages:
            rep.stages[0].extra["swept_params"] = swept_best
        if sweep_rep is not None:
            rep = _with_sweep_cost(rep, sweep_rep)
        self.hook = ResidualSteeringHook(
            directions=directions,
            mode=self.mode,
            alpha=self.alpha,
            prefill=False,
        )
        layers = get_decoder_layers(self.model, layers_path=ctx.generator_spec.layers_path)
        self.hook.attach(layers[target_layer])
        return rep

    def _release_generator(self) -> None:
        """Drop this method's references to its own generator so ``free_model`` can reclaim the VRAM."""
        if not self._owns_model:
            return
        self.model = None
        self.tokenizer = None
        self.generator = None
        free_model()

    def _extract_bf16_directions(self, ctx: RunContext, target_layer: int, n_calib: int):
        bf16_model, bf16_tok, _ = load_decoder(ctx.generator_spec, quant="bf16", device=ctx.hardware.device)
        try:
            return extract_or_load_directions(
                model=bf16_model,
                tokenizer=bf16_tok,
                model_alias=ctx.generator_spec.alias,
                quant="bf16",
                pairs=ctx.calib_pairs,
                concepts=self.concepts,
                layer_indices=[target_layer],
                artifact_dir=ctx.artifact_dir,
                estimator=self.estimator,
                orthogonal_concept=self.orthogonal_concept,
                pooling=self.pooling,
                target_offset_sigma=self.target_offset_sigma,
                n_calib=n_calib,
            )
        finally:
            del bf16_model, bf16_tok
            free_model()

    def _generate_batch_with_prefill_probe(
        self, batch_msgs: list[list[dict]], seed: int
    ) -> tuple[list, list[float]]:
        """Run batched generation while capturing prompt-end residual vectors during prefill (0 extra tokens)."""
        assert self.generator is not None and self.probe is not None and self.model is not None
        layers = get_decoder_layers(self.model, layers_path=self.ctx.generator_spec.layers_path)
        captured: list = []

        def _prefill_hook(module, inp, out):
            if not captured:
                h = out[0] if isinstance(out, tuple) else out
                captured.append(h[:, -1, :].detach().cpu().float().numpy())

        handle = layers[self.probe.layer_idx].register_forward_hook(_prefill_hook)
        try:
            gen_results = self.generator.generate(batch_msgs, seed=seed)
        finally:
            handle.remove()

        acts = captured[0]
        scores = [float(self.probe.score_activation(acts[i])) for i in range(len(batch_msgs))]
        return gen_results, scores

    def _respond_probe_guard(self, batch: list[Example], age_band: str, seed: int) -> list[MethodOutput]:
        """Probe-gated moderation: batched generation followed by action only when flagged."""
        assert self.generator is not None and self.ctx is not None and self.probe is not None
        if not batch:
            return []

        probe = self.probe
        batch_msgs = [ex.messages() for ex in batch]
        if probe.is_prompt_level:
            # Zero-extra-token prefill evaluation (plan §5.4.3)
            gen_results, prefill_scores = self._generate_batch_with_prefill_probe(batch_msgs, seed=seed)
        else:
            gen_results = self.generator.generate(batch_msgs, seed=seed)
            prefill_scores = []

        outputs: list[MethodOutput] = []
        for idx, (ex, gr) in enumerate(zip(batch, gen_results, strict=True)):
            tok = TokenUsage(
                generator_in=gr.n_in,
                generator_out=gr.n_out,
                generator_reasoning_out=gr.n_reasoning,
            )
            gen_ms_each = max(0.01, gr.e2e_s * 1000.0 / max(1, len(batch)))
            t_post0 = time.perf_counter()

            if probe.is_prompt_level:
                score = prefill_scores[idx]
            else:
                score, n_probe_tok = probe.score_text(self.model, self.tokenizer, ex.last_user, gr.text)
                tok.guard_in += n_probe_tok

            final_text, action_taken = gr.text, "none"
            if score >= probe.threshold:

                def _still_unsafe(text: str, _ex: Example = ex) -> bool:
                    if probe.is_prompt_level:
                        return True
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
                    mlm_recomposer=self._mlm,
                    allow_placeholder_mlm=self._cpu_fixture_mode,
                )
                tok = tok + extra_tok

            final_text, n_filtered = self.apply_post_filter(final_text)
            post_ms = (time.perf_counter() - t_post0) * 1000.0
            e2e_ms = max(0.01, gen_ms_each + post_ms)
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
        if self._mlm is not None:
            self._mlm.close()
            self._mlm = None
        self.probe = None
        super().teardown()


def _with_sweep_cost(rep: SetupReport, sweep_rep: SetupReport) -> SetupReport:
    """Return ``rep`` plus the dev-sweep stages, so setup cost includes the hyperparameter search (H1).

    A cached sweep contributes its original measured cost (as cached LoRA/encoder artifacts do); each
    sweep stage is tagged with ``extra["sweep_cached"]``. ``cached`` stays True only if both parts were.
    Returns a new report so a report loaded from an artifact cache is never mutated in place.
    """
    sweep_stages = [
        SetupStage(**{**asdict(st), "extra": {**st.extra, "sweep_cached": bool(sweep_rep.cached)}})
        for st in sweep_rep.stages
    ]
    return SetupReport(
        method=rep.method,
        setup_key=rep.setup_key,
        stages=[*rep.stages, *sweep_stages],
        cached=bool(rep.cached and sweep_rep.cached),
        num_gpus=rep.num_gpus,
    )
