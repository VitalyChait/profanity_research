"""Shared chat-generation engine used by every method (same decoding + same instrumentation).

Handles: chat templates (with fallbacks when a template is missing or rejects the system role), left-padded
batching, seeded sampling, reasoning-model <think> stripping, and per-call timing (TTFT, decode tok/s, e2e).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from minorsafe.config import ModelSpec
from minorsafe.telemetry.timer import Stopwatch, TimingLogitsProcessor

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class GenResult:
    text: str  # visible answer (reasoning stripped)
    raw_text: str
    n_in: int
    n_out: int  # visible answer tokens
    n_reasoning: int  # reasoning (<think>) tokens
    ttft_s: float
    e2e_s: float  # wall time of the generate() call that produced this row
    decode_tok_per_s: float
    batch_size: int
    truncated: bool = False


def split_reasoning(raw: str) -> tuple[str, str, bool]:
    """Return (reasoning, answer, reasoning_unterminated)."""
    if "</think>" in raw:
        head, _, tail = raw.partition("</think>")
        return head.replace("<think>", "").strip(), tail.strip(), False
    if "<think>" in raw:
        return raw.split("<think>", 1)[1].strip(), "", True
    return "", raw.strip(), False


def _merge_system(messages: list[dict]) -> list[dict]:
    """Fold a system message into the first user message (for templates without a system role)."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys = messages[0]["content"]
    rest = [dict(m) for m in messages[1:]]
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{sys}\n\n{m['content']}"
            break
    return rest


def format_chat(tokenizer, messages: list[dict], spec: ModelSpec | None = None) -> str:
    """Render messages into a prompt string, robust to missing templates / system-role rejection."""
    if getattr(tokenizer, "chat_template", None) and (spec is None or spec.chat_template):
        msgs = messages
        if spec is not None and not spec.supports_system:
            msgs = _merge_system(messages)
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return tokenizer.apply_chat_template(
                _merge_system(messages), tokenize=False, add_generation_prompt=True
            )
    parts = []
    for m in messages:
        parts.append(f"{m['role'].capitalize()}: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


class ChatGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        spec: ModelSpec | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        min_new_tokens: int | None = None,
    ):
        self.model = model
        self.tok = tokenizer
        self.spec = spec
        self.max_new_tokens = (spec.max_new_tokens if spec and spec.max_new_tokens else None) or max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_new_tokens = min_new_tokens
        self.reasoning = bool(spec and spec.reasoning)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _eos_ids(self) -> set[int]:
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None) if gc is not None else None
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(eos)
        if self.tok.eos_token_id is not None:
            ids.add(self.tok.eos_token_id)
        return ids

    @torch.no_grad()
    def generate(
        self,
        batch_messages: list[list[dict]],
        seed: int | None = None,
        max_new_tokens: int | None = None,
        stopwatch: Stopwatch | None = None,
        span: str = "generate",
    ) -> list[GenResult]:
        prompts = [format_chat(self.tok, msgs, self.spec) for msgs in batch_messages]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        attn = enc["attention_mask"]
        n_in = attn.sum(dim=1).tolist()
        timer = TimingLogitsProcessor(sync_cuda=self.device.type == "cuda")
        do_sample = self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tok.pad_token_id,
            logits_processor=[timer],
        )
        if do_sample:
            gen_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        if self.min_new_tokens:
            gen_kwargs["min_new_tokens"] = self.min_new_tokens
        if seed is not None:
            torch.manual_seed(seed)
        sw = stopwatch or Stopwatch()
        before = sw.get(span)
        timer.start()
        with sw.span(span):
            out = self.model.generate(**enc, **gen_kwargs)
        e2e = sw.get(span) - before
        gen = out[:, enc["input_ids"].shape[1] :]
        eos = self._eos_ids()
        results = []
        for i in range(gen.shape[0]):
            ids = gen[i].tolist()
            n = len(ids)
            hit_eos = False
            for j, t in enumerate(ids):
                if t in eos:
                    n, hit_eos = j + 1, True
                    break
            raw = self.tok.decode(ids[:n], skip_special_tokens=True)
            if self.reasoning:
                reasoning, answer, unterminated = split_reasoning(raw)
                n_reason = self.count_tokens(reasoning) if reasoning else 0
                n_reason = min(n_reason, n)
                n_vis = n - n_reason
            else:
                answer, n_reason, n_vis, unterminated = raw.strip(), 0, n, False
            results.append(
                GenResult(
                    text=answer,
                    raw_text=raw,
                    n_in=int(n_in[i]),
                    n_out=int(n_vis),
                    n_reasoning=int(n_reason),
                    ttft_s=timer.ttft_s,
                    e2e_s=e2e,
                    decode_tok_per_s=timer.decode_tok_per_s(n),
                    batch_size=gen.shape[0],
                    truncated=(not hit_eos) or unterminated,
                )
            )
        return results

