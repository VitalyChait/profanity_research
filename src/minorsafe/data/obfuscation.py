"""Subcultural obfuscation injectors and de-obfuscation detector (Decision 7).

Covers leetspeak, symbol masking, character spacing, Cyrillic/Greek homoglyphs, algospeak, and emoji profanity.
"""

from __future__ import annotations

import random
import re

from minorsafe.data.lexicons import PROFANITY_RE, TOXIC_EMOJI_MAP, find_profanity_spans
from minorsafe.data.schema import Example, Turn, nfc

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}
HOMOGLYPH_MAP = {"a": "а", "e": "е", "o": "о", "c": "с", "s": "ѕ", "u": "ս"}
ALGOSPEAK_MAP = {
    "fuck": "frick-udge",
    "fucking": "freaking-af",
    "bullshit": "total-bs-cap",
    "shit": "shiznit-crap",
    "bitch": "female-dog-slur",
    "asshole": "a-hole-jerk",
    "damn": "dangg",
}

OBFUSCATION_MODES = (
    "leetspeak",
    "symbol_mask",
    "char_spacing",
    "homoglyphs",
    "algospeak",
    "emoji_profanity",
)

# Regex patterns catching masked/leetspeak/spaced/algospeak profanity
_OBFUSCATED_PATTERNS = [
    re.compile(r"\bf[\W_]*[u\*@40vս][\W_]*[c\*kс][\W_]*k\w*\b", re.IGNORECASE),
    re.compile(r"\bs[\W_]*h[\W_]*[i1!\*l][\W_]*[t7\*]\w*\b", re.IGNORECASE),
    re.compile(r"\bb[\W_]*[i1!\*][\W_]*[t7][\W_]*c[\W_]*h\w*\b", re.IGNORECASE),
    re.compile(r"\b[a4@][\W_]*[s\$5][\W_]*[s\$5](?:h[o0]le)?\b", re.IGNORECASE),
    re.compile(r"\bb[\W_]*u[\W_]*l[\W_]*l[\W_]*s[\W_]*h[\W_]*[i1!\*][\W_]*[t7\*]\b", re.IGNORECASE),
    re.compile(r"\bd[\W_]*[a4@\*][\W_]*m[\W_]*n\b", re.IGNORECASE),
    re.compile(
        r"\b(?:frick-udge|freaking-af|total-bs-cap|shiznit-crap|female-dog-slur|a-hole-jerk|unalive|seggs)\b",
        re.IGNORECASE,
    ),
]

_REVERSE_HOMOGLYPH = {v: k for k, v in HOMOGLYPH_MAP.items()}
_REVERSE_LEET = {"4": "a", "3": "e", "1": "i", "0": "o", "$": "s", "7": "t", "@": "a"}


def _obfuscate_word(word: str, mode: str, rng: random.Random) -> str:
    low = word.lower()
    if mode == "leetspeak":
        out = "".join(LEET_MAP.get(ch, ch) for ch in low)
        return out if out != low else low[:-1] + "1"
    if mode == "symbol_mask":
        if len(word) <= 2:
            return word[0] + "*"
        mid = "*" if len(word) <= 4 else "*" * (len(word) - 3)
        return word[0] + mid + word[-2:]
    if mode == "char_spacing":
        sep = rng.choice([" ", ".", "-"])
        return sep.join(list(word))
    if mode == "homoglyphs":
        out = []
        changed = False
        for ch in word:
            rep = HOMOGLYPH_MAP.get(ch.lower())
            if rep is not None:
                out.append(rep)
                changed = True
            else:
                out.append(ch)
        return "".join(out) if changed else "ѕ" + word[1:]
    if mode == "algospeak":
        return ALGOSPEAK_MAP.get(low, f"{word[0]}***{word[-1]}-algospeak")
    if mode == "emoji_profanity":
        return f"{word} 🖕"
    return word


def apply_obfuscation(text: str, mode: str = "leetspeak", seed: int = 13) -> str:
    """Apply a specific obfuscation transformation to profane words in ``text``."""
    rng = random.Random(seed)
    text = nfc(text)
    if mode == "emoji_profanity":
        cleaned = PROFANITY_RE.sub(lambda m: "🖕", text)
        return cleaned if cleaned != text else f"{text} 🖕"

    def _sub(match: re.Match) -> str:
        return _obfuscate_word(match.group(0), mode=mode, rng=rng)

    out = PROFANITY_RE.sub(_sub, text)
    if out == text:
        out = f"{text} f*ck 🖕"
    return nfc(out)


def detect_obfuscated_profanity(text: str) -> list[str]:
    """Return all explicit or obfuscated/emoji profanity matches in ``text``."""
    text = nfc(text)
    hits: list[str] = [s.word for s in find_profanity_spans(text)]

    for pat in _OBFUSCATED_PATTERNS:
        for m in pat.finditer(text):
            hits.append(m.group(0))

    # Normalize homoglyphs and leetspeak and re-check explicit lexicon
    de_homo = "".join(_REVERSE_HOMOGLYPH.get(ch, ch) for ch in text)
    if de_homo != text:
        for m in PROFANITY_RE.finditer(de_homo):
            hits.append(m.group(0))

    de_leet = "".join(_REVERSE_LEET.get(ch, ch) for ch in de_homo)
    if de_leet != de_homo:
        for m in PROFANITY_RE.finditer(de_leet):
            hits.append(m.group(0))

    # Collapse spaced single letters like "f u c k" or "s.h.i.t"
    collapsed = re.sub(
        r"\b([A-Za-z])(?:[\s\.\-_]+([A-Za-z])){3,}\b", lambda m: re.sub(r"[\s\.\-_]+", "", m.group(0)), text
    )
    if collapsed != text:
        for m in PROFANITY_RE.finditer(collapsed):
            hits.append(m.group(0))

    for emo in TOXIC_EMOJI_MAP:
        if emo in text and emo not in hits:
            hits.append(emo)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def generate_obfuscated_examples(
    examples: list[Example], per_sentence: int = 2, seed: int = 13
) -> list[Example]:
    """Create obfuscated/emoji profanity evaluation examples from profane or unsafe examples."""
    rng = random.Random(seed)
    out: list[Example] = []
    for idx, ex in enumerate(examples):
        user_txt = ex.last_user
        if not user_txt:
            continue
        has_prof = bool(find_profanity_spans(user_txt)) or ("profanity" in ex.categories)
        base_txt = user_txt if has_prof else f"{user_txt} What the fuck is this bullshit?"
        modes = rng.sample(OBFUSCATION_MODES, k=min(per_sentence, len(OBFUSCATION_MODES)))
        for m_idx, mode in enumerate(modes):
            obf_txt = apply_obfuscation(base_txt, mode=mode, seed=seed + idx * 10 + m_idx)
            cats = list(dict.fromkeys([*ex.categories, "obfuscated", "profanity"]))
            out.append(
                Example(
                    source=f"{ex.source}_obf_{mode}",
                    split=ex.split,
                    turns=[Turn(role="user", content=obf_txt)],
                    label_unsafe=True,
                    categories=cats,
                    reference_safe=ex.reference_safe,
                    age_band=ex.age_band or "teen_12_18",
                    meta={"obfuscation_mode": mode, "parent_id": ex.id},
                )
            )
    return out
