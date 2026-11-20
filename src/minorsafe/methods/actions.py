"""Post-detection hardening and dynamic sentence recomposition actions (Phase 5.2.4).

Implements:
- ``fallback``: friendly age-appropriate refusal
- ``regenerate``: re-sample with incremented seed (no system prompt), re-checked by the caller's guard
- ``llm_rewrite``: autoregressive decoder rewrite via user-turn instruction (no hidden post-processing)
- ``mlm_substitute``: bidirectional encoder MLM span infilling (ModernBERT / XLM-R) preserving
  meaning while filtering inappropriate or overly complex spans
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import torch
from wordfreq import zipf_frequency

from minorsafe.config import ModelSpec
from minorsafe.data.lexicons import (
    COMPLEX_TO_SIMPLE,
    PROFANITY_TO_SAFE,
    find_complex_spans,
    find_profanity_spans,
    replace_complex_words,
    replace_profanity,
)
from minorsafe.data.obfuscation import detect_obfuscated_profanity
from minorsafe.data.schema import Example
from minorsafe.generation import ChatGenerator
from minorsafe.placeholders import PlaceholderBackendError
from minorsafe.registry import free_model, load_mlm
from minorsafe.telemetry.tokens import TokenUsage

_CLEAN_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{0,24}$")


def _default_is_unsafe(text: str) -> bool:
    return bool(find_profanity_spans(text or "") or detect_obfuscated_profanity(text or ""))


def mlm_recompose_text(text: str) -> tuple[str, int]:
    """CPU-only placeholder recomposition used in offline dummy-model tests."""
    cleaned, _ = replace_profanity(text)
    cleaned, _ = replace_complex_words(cleaned)
    obf_hits = detect_obfuscated_profanity(cleaned)
    for hit in obf_hits:
        cleaned = cleaned.replace(hit, "that")
    n_enc_tokens = max(1, len(text.split()))
    return cleaned, n_enc_tokens


class MLMRecomposer:
    """Real Bidirectional Masked LM span recomposer (ModernBERT / XLM-R)."""

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        self.model = model
        self.tok = tokenizer
        self.device = device

    @classmethod
    def from_spec(cls, spec: ModelSpec, device: str = "cuda") -> MLMRecomposer:
        model, tok = load_mlm(spec, device=device)
        return cls(model=model, tokenizer=tok, device=device)

    @torch.no_grad()
    def _predict_replacement(self, masked_text: str, fallback: str) -> tuple[str, int]:
        enc = self.tok(masked_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        n_tok = int(enc["input_ids"].shape[1])
        mask_id = self.tok.mask_token_id
        if mask_id is None:
            return fallback, n_tok
        mask_positions = (enc["input_ids"][0] == mask_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            return fallback, n_tok
        pos = int(mask_positions[0].item())

        self.model.eval()
        logits = self.model(**enc).logits[0, pos].float()
        top_ids = torch.topk(logits, k=min(60, logits.numel())).indices.tolist()
        for tid in top_ids:
            raw_piece = (
                self.tok.convert_ids_to_tokens(tid) if hasattr(self.tok, "convert_ids_to_tokens") else ""
            )
            if isinstance(raw_piece, str) and raw_piece.startswith("##"):
                continue
            cand = self.tok.decode([tid], skip_special_tokens=True).strip()
            if not _CLEAN_WORD_RE.match(cand):
                continue
            if find_profanity_spans(cand) or detect_obfuscated_profanity(cand):
                continue
            if cand.lower() in PROFANITY_TO_SAFE or cand.lower() in COMPLEX_TO_SIMPLE:
                continue
            if zipf_frequency(cand.lower(), "en") < 3.5:
                continue
            return cand, n_tok
        return fallback, n_tok

    def recompose(self, text: str, extra_suspect_words: list[str] | None = None) -> tuple[str, int]:
        """Replace profane, obfuscated, complex, and guard-attributed spans using masked LM infilling."""
        cur = text or ""
        if not cur.strip():
            return cur, 0
        mask_tok = getattr(self.tok, "mask_token", None) or "<mask>"
        total_tokens = 0

        targets: list[tuple[str, str]] = []
        for sp in find_profanity_spans(cur):
            targets.append((sp.word, sp.replacement))
        for sp in find_complex_spans(cur):
            targets.append((sp.word, sp.replacement))
        for hit in detect_obfuscated_profanity(cur):
            targets.append((hit, "that"))
        for word in extra_suspect_words or []:
            if word and all(word.lower() != t[0].lower() for t in targets):
                targets.append((word, "that"))

        for orig, fallback in targets:
            idx = cur.lower().find(orig.lower())
            if idx == -1:
                continue
            orig_slice = cur[idx : idx + len(orig)]
            masked = cur[:idx] + mask_tok + cur[idx + len(orig) :]
            rep, n_tok = self._predict_replacement(masked, fallback=fallback)
            total_tokens += n_tok
            if orig_slice and orig_slice[0].isupper() and rep:
                rep = rep[0].upper() + rep[1:]
            cur = cur[:idx] + rep + cur[idx + len(orig) :]

        return cur, total_tokens

    def close(self) -> None:
        if self.model is not None or self.tok is not None:
            free_model(self.model, self.tok)
        self.model = None
        self.tok = None


def execute_moderation_action(
    action: str,
    example: Example,
    draft_text: str,
    generator: ChatGenerator,
    prompts: dict,
    age_band: str = "teen_12_18",
    seed: int = 0,
    max_regenerations: int = 2,
    is_unsafe: Callable[[str], bool] | None = None,
    mlm_recomposer: MLMRecomposer | None = None,
    extra_suspect_words: list[str] | None = None,
    allow_placeholder_mlm: bool = False,
) -> tuple[str, str, TokenUsage, float]:
    """Return (final_text, action_taken, extra_tokens, extra_seconds)."""
    t0 = time.perf_counter()
    extra_tok = TokenUsage()
    check = is_unsafe or _default_is_unsafe
    fallback_msg = (prompts.get("fallback_responses") or {}).get(
        age_band,
        "I can't help with that specific request, let's explore a safe related topic!",
    )

    if action == "fallback":
        return fallback_msg, "fallback", extra_tok, time.perf_counter() - t0

    if action == "mlm_substitute":
        if mlm_recomposer is not None:
            recomposed, enc_toks = mlm_recomposer.recompose(
                draft_text, extra_suspect_words=extra_suspect_words
            )
        elif allow_placeholder_mlm:
            recomposed, enc_toks = mlm_recompose_text(draft_text)
        else:
            raise PlaceholderBackendError(
                "mlm_substitute requires an MLMRecomposer (ModernBERT/XLM-R) on non-placeholder profiles."
            )
        extra_tok.encoder_in += enc_toks
        return recomposed, "mlm_substitute", extra_tok, time.perf_counter() - t0

    if action == "regenerate":
        for k in range(max_regenerations):
            res = generator.generate([example.messages()], seed=seed + k + 1)[0]
            extra_tok.regen_in += res.n_in
            extra_tok.regen_out += res.n_out + res.n_reasoning
            if not check(res.text):
                return res.text, "regenerate", extra_tok, time.perf_counter() - t0
        return fallback_msg, "fallback", extra_tok, time.perf_counter() - t0

    if action == "llm_rewrite":
        tpl = prompts.get(
            "llm_rewrite_user_template",
            "Rewrite the following draft for ages 12-18 without profanity:\n{draft_response}",
        )
        user_msg = tpl.format(draft_response=draft_text)
        res = generator.generate([[{"role": "user", "content": user_msg}]], seed=seed)[0]
        extra_tok.rewrite_in += res.n_in
        extra_tok.rewrite_out += res.n_out + res.n_reasoning
        return res.text, "llm_rewrite", extra_tok, time.perf_counter() - t0

    raise ValueError(f"Unknown moderation action: {action!r}")
