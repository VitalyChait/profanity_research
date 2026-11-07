"""Unified data schema for evaluation examples and contrastive pairs, with parquet I/O.

Text is stored exactly as given (NFC-normalised only): emojis, symbols and special characters are preserved.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from minorsafe.utils import stable_hash

Split = Literal["calib", "dev", "test"]

# Canonical category vocabulary (free-form strings are allowed, these are the ones the pipeline uses).
CATEGORIES = (
    "benign",
    "benign_tricky",  # looks risky but is safe (over-refusal probes, e.g. XSTest)
    "profanity",
    "obfuscated",  # leetspeak / symbol-masked / emoji profanity
    "sexual",
    "violence",
    "self_harm",
    "drugs",
    "hate",
    "harassment",
    "slang",
    "complex_lexicon",
    "unsafe_other",
)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class Turn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _nfc(cls, v: str) -> str:
        return nfc(v)


class Example(BaseModel):
    id: str = ""
    source: str
    split: Split | None = None
    turns: list[Turn]
    label_unsafe: bool | None = None  # prompt-level label: unsuitable request for a 12-18 user
    response: str | None = None  # an existing (model) response, for response-level guard eval
    response_unsafe: bool | None = None
    categories: list[str] = Field(default_factory=list)
    reference_safe: str | None = None  # gold appropriate response / simplification
    age_band: str | None = None
    meta: dict = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = stable_hash([self.source, [t.model_dump() for t in self.turns], self.response])

    @property
    def last_user(self) -> str:
        for t in reversed(self.turns):
            if t.role == "user":
                return t.content
        return ""

    def messages(self) -> list[dict]:
        return [t.model_dump() for t in self.turns]


class ContrastivePair(BaseModel):
    id: str = ""
    concept: str  # profanity | lexical_complexity | mature_theme | slang_toxic | obfuscated
    prompt: str
    positive: str  # age-appropriate
    negative: str  # inappropriate / too complex
    minimal: bool = False  # differ only by substituted tokens
    source: str = "synthetic"
    split: Split | None = None
    meta: dict = Field(default_factory=dict)

    @field_validator("prompt", "positive", "negative")
    @classmethod
    def _nfc(cls, v: str) -> str:
        return nfc(v)

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = stable_hash([self.concept, self.prompt, self.positive, self.negative])


# --------------------------------------------------------------------------- parquet I/O
_JSON_COLS_EX = ("turns", "categories", "meta")
_JSON_COLS_PAIR = ("meta",)


def examples_to_df(examples: list[Example]) -> pd.DataFrame:
    rows = []
    for e in examples:
        d = e.model_dump()
        for c in _JSON_COLS_EX:
            d[c] = json.dumps(d[c], ensure_ascii=False)
        rows.append(d)
    return pd.DataFrame(rows)


def df_to_examples(df: pd.DataFrame) -> list[Example]:
    out = []
    for rec in df.to_dict(orient="records"):
        for c in _JSON_COLS_EX:
            if isinstance(rec.get(c), str):
                rec[c] = json.loads(rec[c])
        rec = {k: (None if _isnan(v) else v) for k, v in rec.items()}
        out.append(Example(**rec))
    return out


def pairs_to_df(pairs: list[ContrastivePair]) -> pd.DataFrame:
    rows = []
    for p in pairs:
        d = p.model_dump()
        d["meta"] = json.dumps(d["meta"], ensure_ascii=False)
        rows.append(d)
    return pd.DataFrame(rows)


def df_to_pairs(df: pd.DataFrame) -> list[ContrastivePair]:
    out = []
    for rec in df.to_dict(orient="records"):
        if isinstance(rec.get("meta"), str):
            rec["meta"] = json.loads(rec["meta"])
        rec = {k: (None if _isnan(v) else v) for k, v in rec.items()}
        out.append(ContrastivePair(**rec))
    return out


def _isnan(v) -> bool:
    return isinstance(v, float) and v != v


def save_examples(examples: list[Example], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    examples_to_df(examples).to_parquet(path, index=False)


def load_examples(path: str | Path) -> list[Example]:
    return df_to_examples(pd.read_parquet(path))


def save_pairs(pairs: list[ContrastivePair], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pairs_to_df(pairs).to_parquet(path, index=False)


def load_pairs(path: str | Path) -> list[ContrastivePair]:
    return df_to_pairs(pd.read_parquet(path))

