"""Readability metrics for the 12-18 age band: Flesch-Kincaid grade, Dale-Chall, rare word ratio, and compliance."""

from __future__ import annotations

import textstat
from wordfreq import zipf_frequency

from minorsafe.data.lexicons import ALPHA_WORD_RE, COMPLEX_TO_SIMPLE


def compute_readability_metrics(text: str, target_fk_grade: float = 8.0) -> dict:
    clean = (text or "").strip()
    if not clean:
        return {
            "fk_grade": 0.0,
            "dale_chall": 0.0,
            "rare_word_ratio": 0.0,
            "fk_compliant": True,
        }

    fk = float(textstat.flesch_kincaid_grade(clean))
    dc = float(textstat.dale_chall_readability_score(clean))

    words = [w.lower().strip("'") for w in ALPHA_WORD_RE.findall(clean)]
    words = [w for w in words if len(w) >= 3]
    if words:
        rare = sum(1 for w in words if zipf_frequency(w, "en") < 3.6 or w in COMPLEX_TO_SIMPLE)
        rare_ratio = rare / float(len(words))
    else:
        rare_ratio = 0.0

    return {
        "fk_grade": fk,
        "dale_chall": dc,
        "rare_word_ratio": float(rare_ratio),
        "fk_compliant": bool(fk <= target_fk_grade),
    }

