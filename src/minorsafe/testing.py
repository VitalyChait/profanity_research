"""Deterministic, dependency-free dummy causal LM + tokenizer for CPU smoke runs and tests.

These live in the package (not ``tests/conftest.py``) so ``minorsafe smoke`` works outside pytest.
They are *not* language models: outputs are meaningless and must never feed reported results.
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import torch
import torch.nn as nn


class DummyDecoderLayer(nn.Module):
    """Minimal transformer decoder layer returning (hidden_states,) tuple by default."""

    def __init__(self, hidden_size: int = 16, return_tuple: bool = True):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.return_tuple = return_tuple

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = x + 0.05 * torch.tanh(self.proj(x))
        return (out,) if self.return_tuple else out


class DummyCausalLM(nn.Module):
    """Lightweight causal LM compatible with ChatGenerator, hooks, and get_decoder_layers."""

    def __init__(
        self, vocab_size: int = 128, hidden_size: int = 16, num_layers: int = 4, return_tuple: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        layers = nn.ModuleList(
            [DummyDecoderLayer(hidden_size, return_tuple=return_tuple) for _ in range(num_layers)]
        )
        self.model = nn.Module()
        self.model.layers = layers
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            pad_token_id=0,
            eos_token_id=2,
        )
        self.generation_config = SimpleNamespace(pad_token_id=0, eos_token_id=2)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            layer_out = layer(h)
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        h = self.model.norm(h)
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits, hidden_states=h)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 8,
        min_new_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        pad_token_id: int | None = 0,
        logits_processor: list | None = None,
        **kwargs,
    ) -> torch.Tensor:
        cur = input_ids
        n_steps = max(min_new_tokens or 0, min(max_new_tokens, 6))
        for step in range(n_steps):
            out = self.forward(cur)
            scores = out.logits[:, -1, :]
            if logits_processor:
                for proc in logits_processor:
                    scores = proc(cur, scores)
            if do_sample and temperature and temperature > 0:
                probs = torch.softmax(scores / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = scores.argmax(dim=-1, keepdim=True)
            # Ensure we don't emit EOS before min_new_tokens
            if min_new_tokens and step < min_new_tokens:
                next_tok = torch.clamp(next_tok, min=3)
            cur = torch.cat([cur, next_tok], dim=1)
        return cur


class BatchDict(dict):
    def to(self, device):
        return BatchDict({k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()})


class DummyTokenizer:
    """Whitespace/byte-hash tokenizer compatible with ChatGenerator and offset alignment."""

    def __init__(self, vocab_size: int = 128, emit_reasoning: bool = False):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        self.padding_side = "left"
        self.chat_template = None
        self.emit_reasoning = emit_reasoning
        self._id_to_word: dict[int, str] = {0: "", 1: "unk", 2: "", 3: "<mask>"}

    def _encode_text(self, text: str) -> list[int]:
        tokens = []
        for w in text.split():
            tid = 4 + (zlib.crc32(w.encode("utf-8")) % (self.vocab_size - 4))
            self._id_to_word[tid] = w
            tokens.append(tid)
        return tokens or [4]

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        encoded = [self._encode_text(t) for t in batch]
        if max_length:
            encoded = [seq[:max_length] for seq in encoded]
        if single and return_tensors is None:
            return {"input_ids": encoded[0], "attention_mask": [1] * len(encoded[0])}
        max_len = max(len(seq) for seq in encoded) if padding else None
        ids_batch, attn_batch = [], []
        for seq in encoded:
            pad_len = (max_len - len(seq)) if max_len is not None else 0
            if self.padding_side == "left":
                ids_batch.append([self.pad_token_id] * pad_len + seq)
                attn_batch.append([0] * pad_len + [1] * len(seq))
            else:
                ids_batch.append(seq + [self.pad_token_id] * pad_len)
                attn_batch.append([1] * len(seq) + [0] * pad_len)
        if return_tensors == "pt":
            return BatchDict(
                {
                    "input_ids": torch.tensor(ids_batch, dtype=torch.long),
                    "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
                }
            )
        return BatchDict({"input_ids": ids_batch, "attention_mask": attn_batch})

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in (0, 2):
                continue
            words.append(self._id_to_word.get(int(tid), f"tok{tid}"))
        body = " ".join(w for w in words if w) or "Plants use sunlight and water to make food."
        if self.emit_reasoning:
            return f"<think> step one step two </think> {body}"
        return body
