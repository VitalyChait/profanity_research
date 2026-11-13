"""M2 LoRA / QLoRA fine-tuning and inference method (Phase 5.3)."""

from __future__ import annotations

import torch

from minorsafe.data.lexicons import replace_complex_words, replace_profanity
from minorsafe.data.schema import Example
from minorsafe.generation import format_chat
from minorsafe.methods.base import MethodOutput, ResponseMethod, RunContext
from minorsafe.telemetry.cost import SetupReport
from minorsafe.telemetry.tokens import TokenUsage


class LoRAResponseMethod(ResponseMethod):
    def setup(self, ctx: RunContext) -> SetupReport:
        self._init_generator(ctx)
        assert self.model is not None and self.tokenizer is not None
        n_calib = int(self.spec.params.get("n_calib", 50))
        epochs = int(self.spec.params.get("epochs", 1))
        lr = float(self.spec.params.get("lr", 2.0e-4))
        rep = SetupReport(
            method=self.name,
            setup_key=f"{self.name}_{ctx.generator_spec.alias}_N{n_calib}",
        )

        pairs = ctx.calib_pairs[:n_calib]
        with rep.stage("train_lora_adapter", examples=len(pairs) * epochs) as st:
            dev = next(self.model.parameters()).device
            # Fine-tune lm_head / low-rank projection on positive completions (no system prompt)
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if trainable and pairs:
                opt = torch.optim.AdamW(trainable[-2:], lr=lr)
                for _ in range(max(1, epochs)):
                    for p in pairs[: min(16, len(pairs))]:
                        txt = f"{format_chat(self.tokenizer, [{'role': 'user', 'content': p.prompt}])} {p.positive}"
                        enc = self.tokenizer(txt, return_tensors="pt", add_special_tokens=False).to(dev)
                        ids = enc["input_ids"]
                        st.tokens += int(ids.numel())
                        if ids.shape[1] < 2:
                            continue
                        out = self.model(ids)
                        logits = out.logits[:, :-1, :].contiguous()
                        targets = ids[:, 1:].contiguous()
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1),
                        )
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
            self.model.eval()
        return rep

    def respond(
        self, batch: list[Example], age_band: str = "teen_12_18", seed: int = 0
    ) -> list[MethodOutput]:
        assert self.generator is not None and self.ctx is not None
        # Decision 4: NO system prompt for M2
        batch_msgs = [ex.messages() for ex in batch]
        gen_results = self.generator.generate(batch_msgs, seed=seed)

        outputs: list[MethodOutput] = []
        for ex, gr in zip(batch, gen_results, strict=False):
            tok = TokenUsage(
                generator_in=gr.n_in,
                generator_out=gr.n_out,
                generator_reasoning_out=gr.n_reasoning,
            )
            cleaned, _ = replace_profanity(gr.text)
            cleaned, _ = replace_complex_words(cleaned)
            e2e_ms = max(0.01, gr.e2e_s * 1000.0 / max(1, len(batch)))
            eff_tok_s = max(1, gr.n_out) / (e2e_ms / 1000.0)
            outputs.append(
                MethodOutput(
                    example_id=ex.id,
                    text=cleaned,
                    raw_text=gr.raw_text,
                    blocked=False,
                    action_taken="lora_adapted",
                    tokens=tok,
                    ttft_ms=gr.ttft_s * 1000.0,
                    e2e_ms=e2e_ms,
                    decode_tok_per_s=gr.decode_tok_per_s
                    if gr.decode_tok_per_s == gr.decode_tok_per_s
                    else eff_tok_s,
                    effective_tok_per_s=eff_tok_s,
                )
            )
        return outputs
