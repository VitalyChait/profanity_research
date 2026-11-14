"""M0 Baselines: M0a (raw unmodified model, no system prompt) and M0b (12-18 system prompt)."""

from __future__ import annotations

from minorsafe.data.schema import Example
from minorsafe.methods.base import MethodOutput, ResponseMethod, RunContext
from minorsafe.telemetry.cost import SetupReport
from minorsafe.telemetry.tokens import TokenUsage


class BaselineMethod(ResponseMethod):
    def setup(self, ctx: RunContext) -> SetupReport:
        self._init_generator(ctx)
        return SetupReport(method=self.name, setup_key=f"{self.name}_{ctx.generator_spec.alias}")

    def respond(
        self, batch: list[Example], age_band: str = "teen_12_18", seed: int = 0
    ) -> list[MethodOutput]:
        assert self.generator is not None and self.ctx is not None
        use_sys = bool(self.spec.params.get("use_system_prompt", False))
        sys_prompt = None
        if use_sys:
            sys_prompt = (self.ctx.prompts.get("m0b_system_prompts") or {}).get(age_band)

        batch_msgs: list[list[dict]] = []
        for ex in batch:
            msgs = ex.messages()
            if sys_prompt:
                msgs = [{"role": "system", "content": sys_prompt}, *msgs]
            batch_msgs.append(msgs)

        gen_results = self.generator.generate(batch_msgs, seed=seed)
        outputs: list[MethodOutput] = []
        for ex, gr in zip(batch, gen_results, strict=False):
            tok = TokenUsage(
                generator_in=gr.n_in,
                generator_out=gr.n_out,
                generator_reasoning_out=gr.n_reasoning,
            )
            e2e_ms = max(0.01, gr.e2e_s * 1000.0 / max(1, len(batch)))
            ttft_ms = max(0.0, gr.ttft_s * 1000.0)
            eff_tok_s = (gr.n_out / (e2e_ms / 1000.0)) if e2e_ms > 0 else 0.0
            final_text, n_filtered = self.apply_post_filter(gr.text)
            outputs.append(
                MethodOutput(
                    example_id=ex.id,
                    text=final_text,
                    raw_text=gr.raw_text,
                    blocked=False,
                    action_taken="sysprompt" if use_sys else "none",
                    tokens=tok,
                    ttft_ms=ttft_ms,
                    e2e_ms=e2e_ms,
                    decode_tok_per_s=gr.decode_tok_per_s
                    if gr.decode_tok_per_s == gr.decode_tok_per_s
                    else eff_tok_s,
                    effective_tok_per_s=eff_tok_s,
                    extra={"post_filter_replacements": n_filtered} if self.post_filter else {},
                )
            )
        return outputs
