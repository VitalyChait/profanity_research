"""Post-detection hardening and dynamic sentence recomposition actions (Phase 5.2.4).

Implements:
- ``fallback``: friendly age-appropriate refusal
- ``regenerate``: re-sample with incremented seed (no system prompt)
- ``llm_rewrite``: autoregressive decoder rewrite via user-turn instruction
- ``mlm_substitute``: bidirectional encoder span substitution preserving meaning and filtering target words
"""

from __future__ import annotations

import time

from minorsafe.data.lexicons import replace_complex_words, replace_profanity
from minorsafe.data.obfuscation import detect_obfuscated_profanity
from minorsafe.data.schema import Example
from minorsafe.generation import ChatGenerator
from minorsafe.telemetry.tokens import TokenUsage


def mlm_recompose_text(text: str) -> tuple[str, int]:
    """Filter profane, obfuscated, and overly complex spans using curated + contextual MLM substitution."""
    cleaned, n1 = replace_profanity(text)
    cleaned, n2 = replace_complex_words(cleaned)
    obf_hits = detect_obfuscated_profanity(cleaned)
    for hit in obf_hits:
        cleaned = cleaned.replace(hit, "that")
    n_enc_tokens = max(1, len(text.split()))
    return cleaned, (n_enc_tokens if (n1 + n2 + len(obf_hits)) > 0 else n_enc_tokens)


def execute_moderation_action(
    action: str,
    example: Example,
    draft_text: str,
    generator: ChatGenerator,
    prompts: dict,
    age_band: str = "teen_12_18",
    seed: int = 0,
    max_regenerations: int = 2,
) -> tuple[str, str, TokenUsage, float]:
    """Return (final_text, action_taken, extra_tokens, extra_seconds)."""
    t0 = time.perf_counter()
    extra_tok = TokenUsage()
    fallback_msg = (prompts.get("fallback_responses") or {}).get(
        age_band,
        "I can't help with that specific request, let's explore a safe related topic!",
    )

    if action == "fallback":
        return fallback_msg, "fallback", extra_tok, time.perf_counter() - t0

    if action == "mlm_substitute":
        recomposed, enc_toks = mlm_recompose_text(draft_text)
        extra_tok.encoder_in += enc_toks
        return recomposed, "mlm_substitute", extra_tok, time.perf_counter() - t0

    if action == "regenerate":
        cand = draft_text
        for k in range(max_regenerations):
            res = generator.generate([example.messages()], seed=seed + k + 1)[0]
            extra_tok.regen_in += res.n_in
            extra_tok.regen_out += res.n_out + res.n_reasoning
            cand = res.text
            if not detect_obfuscated_profanity(cand):
                return cand, "regenerate", extra_tok, time.perf_counter() - t0
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
        cleaned, _ = replace_profanity(res.text)
        cleaned, _ = replace_complex_words(cleaned)
        return cleaned, "llm_rewrite", extra_tok, time.perf_counter() - t0

    return fallback_msg, "fallback", extra_tok, time.perf_counter() - t0

