"""Readability metrics for the 12-18 age band: Flesch-Kincaid grade, Dale-Chall, rare word ratio, and compliance."""

from __future__ import annotations

import textstat
from wordfreq import zipf_frequency

from minorsafe.data.lexicons import ALPHA_WORD_RE, COMPLEX_TO_SIMPLE
from minorsafe.utils import is_empty_response


def compute_readability_metrics(text: str, target_fk_grade: float = 8.0) -> dict:
    """Readability of ``text``. A response with no words has no grade (NaN) and is not compliant."""
    clean = (text or "").strip()
    if is_empty_response(clean):
        return {
            "fk_grade": float("nan"),
            "dale_chall": float("nan"),
            "rare_word_ratio": float("nan"),
            "fk_compliant": False,
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
