"""Curated profanity -> age-appropriate (12-18) and complex -> simple lexicons, plus English detection.

Preserves all Unicode characters, emojis, and punctuation (NFC-only policy, Decision 7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wordfreq import zipf_frequency

from minorsafe.data.schema import nfc

PROFANITY_TO_SAFE: dict[str, str] = {
    "motherfucker": "jerk",
    "motherfucking": "ridiculous",
    "clusterfuck": "total mess",
    "bullshit": "nonsense",
    "horseshit": "rubbish",
    "dipshit": "fool",
    "dumbass": "foolish person",
    "jackass": "clown",
    "asshole": "jerk",
    "assholes": "jerks",
    "bastard": "scoundrel",
    "bastards": "scoundrels",
    "bitching": "complaining",
    "bitches": "mean people",
    "bitchy": "mean",
    "bitch": "mean person",
    "fucking": "seriously",
    "fucked": "ruined",
    "fucker": "jerk",
    "fucks": "cares",
    "fuck": "mess",
    "shitty": "terrible",
    "shitting": "kidding",
    "shits": "problems",
    "shit": "stuff",
    "pissed": "really annoyed",
    "piss": "annoy",
    "goddamn": "darn",
    "dammit": "darn it",
    "damn": "darn",
    "crap": "junk",
    "crappy": "lousy",
    "dickhead": "jerk",
    "dick": "jerk",
    "prick": "jerk",
    "slut": "person",
    "whore": "person",
    "cunt": "jerk",
    "twat": "fool",
    "wanker": "fool",
    "retard": "foolish person",
    "retarded": "ridiculous",
    "faggot": "person",
    "nigga": "friend",
    "nigger": "person",
    "wtf": "what on earth",
    "stfu": "be quiet",
    "gtfo": "get out",
    "lmfao": "laughing so hard",
    "badass": "awesome",
}

COMPLEX_TO_SIMPLE: dict[str, str] = {
    "ameliorate": "improve",
    "ubiquitous": "everywhere",
    "deleterious": "harmful",
    "obfuscate": "confuse",
    "perspicacious": "sharp",
    "recalcitrant": "stubborn",
    "ephemeral": "short-lived",
    "ostensible": "apparent",
    "superfluous": "extra",
    "exacerbate": "worsen",
    "mitigate": "reduce",
    "proliferate": "spread",
    "idiosyncratic": "unique",
    "concomitant": "accompanying",
    "perfunctory": "careless",
    "precipitate": "trigger",
    "parsimonious": "stingy",
    "esoteric": "obscure",
    "magnanimous": "generous",
    "equivocate": "stall",
    "corroborate": "confirm",
    "disseminate": "share",
    "juxtapose": "compare",
    "cognizant": "aware",
    "quintessential": "classic",
    "pulchritudinous": "beautiful",
    "sesquipedalian": "long-winded",
    "intransigent": "unbending",
    "mendacious": "dishonest",
    "lugubrious": "gloomy",
    "sycophant": "flatterer",
    "anachronistic": "outdated",
    "antithetical": "opposite",
    "circumlocution": "wordiness",
    "pellucid": "clear",
}

TOXIC_EMOJI_MAP: dict[str, str] = {
    "🖕": "[rude gesture]",
    "🖕🏻": "[rude gesture]",
    "🖕🏼": "[rude gesture]",
    "🖕🏽": "[rude gesture]",
    "🖕🏾": "[rude gesture]",
    "🖕🏿": "[rude gesture]",
    "🍆💦": "[inappropriate]",
    "🍑🍆": "[inappropriate]",
}

BENIGN_TEEN_EMOJIS: tuple[str, ...] = ("💀", "😭", "🔥", "✨", "🙏", "😂", "💯", "🫡", "🐐", "🤝")

_PROFANITY_KEYS = sorted(PROFANITY_TO_SAFE.keys(), key=len, reverse=True)
PROFANITY_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _PROFANITY_KEYS) + r")\b", re.IGNORECASE)

_COMPLEX_KEYS = sorted(COMPLEX_TO_SIMPLE.keys(), key=len, reverse=True)
COMPLEX_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _COMPLEX_KEYS) + r")\b", re.IGNORECASE)

ALPHA_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class SpanMatch:
    start: int
    end: int
    word: str
    replacement: str
    kind: str  # profanity | complex | emoji


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original and original[0].isupper():
        return replacement.capitalize()
    return replacement


def find_profanity_spans(text: str) -> list[SpanMatch]:
    text = nfc(text)
    spans: list[SpanMatch] = []
    for m in PROFANITY_RE.finditer(text):
        raw = m.group(0)
        rep = _preserve_case(raw, PROFANITY_TO_SAFE[raw.lower()])
        spans.append(SpanMatch(m.start(), m.end(), raw, rep, "profanity"))
    for emo, rep in TOXIC_EMOJI_MAP.items():
        idx = text.find(emo)
        while idx != -1:
            spans.append(SpanMatch(idx, idx + len(emo), emo, rep, "emoji"))
            idx = text.find(emo, idx + len(emo))
    spans.sort(key=lambda s: s.start)
    return spans


def find_complex_spans(text: str) -> list[SpanMatch]:
    text = nfc(text)
    spans: list[SpanMatch] = []
    for m in COMPLEX_RE.finditer(text):
        raw = m.group(0)
        rep = _preserve_case(raw, COMPLEX_TO_SIMPLE[raw.lower()])
        spans.append(SpanMatch(m.start(), m.end(), raw, rep, "complex"))
    return spans


def replace_profanity(text: str) -> tuple[str, int]:
    """Replace explicit profanity and toxic emojis with 12-18 safe equivalents."""
    text = nfc(text)
    count = 0

    def _sub(match: re.Match) -> str:
        nonlocal count
        count += 1
        raw = match.group(0)
        return _preserve_case(raw, PROFANITY_TO_SAFE[raw.lower()])

    out = PROFANITY_RE.sub(_sub, text)
    for emo, rep in TOXIC_EMOJI_MAP.items():
        if emo in out:
            count += out.count(emo)
            out = out.replace(emo, rep)
    return out, count


def replace_complex_words(text: str) -> tuple[str, int]:
    """Replace complex/collegiate vocabulary with accessible grade <= 8 words."""
    text = nfc(text)
    count = 0

    def _sub(match: re.Match) -> str:
        nonlocal count
        count += 1
        raw = match.group(0)
        return _preserve_case(raw, COMPLEX_TO_SIMPLE[raw.lower()])

    out = COMPLEX_RE.sub(_sub, text)
    return out, count


def english_word_ratio(text: str) -> float:
    """Fraction of alphabetic tokens recognized as English words by wordfreq (ignores emojis & symbols)."""
    words = [w.lower().strip("'") for w in ALPHA_WORD_RE.findall(text)]
    words = [w for w in words if len(w) >= 2]
    if not words:
        return 1.0
    known = sum(
        1 for w in words if zipf_frequency(w, "en") >= 3.7 or w in PROFANITY_TO_SAFE or w in COMPLEX_TO_SIMPLE
    )
    return known / len(words)


def is_english(text: str, min_ratio: float = 0.60) -> bool:
    """Decision 7: check English alphabetic ratio while preserving emojis and special characters."""
    return english_word_ratio(text) >= min_ratio
