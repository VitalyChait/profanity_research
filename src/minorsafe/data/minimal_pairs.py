"""Minimal contrastive pair builder for clean activation steering and sentence recomposition."""

from __future__ import annotations

import random

from minorsafe.data.lexicons import (
    COMPLEX_TO_SIMPLE,
    PROFANITY_TO_SAFE,
    replace_complex_words,
    replace_profanity,
)
from minorsafe.data.schema import ContrastivePair

_PROFANITY_FRAMES: tuple[tuple[str, str], ...] = (
    (
        "My teammate just disconnected in the final round, what should I do?",
        "Having a teammate disconnect in the final round is {adj} frustrating and feels like total {noun}, but you can focus on defending the objective until they reconnect.",
    ),
    (
        "Why did my code crash right before the deadline?",
        "When your project crashes right before a deadline it is a {adj} stressful {noun}, so check the stack trace first and revert the last commit.",
    ),
    (
        "Someone in the group chat keeps insulting my artwork.",
        "Anyone acting like a {person} in the group chat is just trying to {verb} with your confidence, so mute them and keep creating.",
    ),
    (
        "I studied all week and still got a C on the chemistry quiz.",
        "Getting a lower grade after studying hard makes anyone {mood}, but reviewing the missed questions with your teacher will fix that {noun} quickly.",
    ),
    (
        "How do I deal with lag spikes in online games?",
        "Sudden lag spikes can {state} your match and feel like {noun2}, so switch to a wired connection and close background downloads.",
    ),
)

_COMPLEX_FRAMES: tuple[tuple[str, str], ...] = (
    (
        "How can our school club make its recycling program better?",
        "To {v_improve} the club's program and {v_reduce} {adj_harmful} waste, you can place {adj_clear} bins in every hallway and {v_share} weekly progress updates.",
    ),
    (
        "Why do scientists repeat experiments multiple times?",
        "Scientists repeat tests to {v_confirm} their findings, remove {adj_extra} errors, and stay {adj_aware} of any {adj_unique} variables.",
    ),
    (
        "How does misinformation spread on social media?",
        "Rumors {v_spread} online when people post {adj_dishonest} claims that {v_confuse} readers and {v_worsen} confusion.",
    ),
    (
        "What makes a good team captain in sports?",
        "A great captain stays {adj_generous} with praise, helps {v_improve} team morale, and avoids {adj_careless} mistakes during close games.",
    ),
)


def build_minimal_pairs(max_pairs: int = 200, seed: int = 13) -> list[ContrastivePair]:
    """Generate minimal contrastive pairs that differ ONLY in the target profane or complex spans."""
    rng = random.Random(seed)
    pairs: list[ContrastivePair] = []

    prof_tuples = [
        ("fucking", "bullshit", "asshole", "fuck", "pissed", "fucked", "horseshit"),
        ("shitty", "crap", "jackass", "piss", "goddamn", "clusterfuck", "bullshit"),
        ("motherfucking", "bullshit", "dickhead", "fuck", "pissed", "fucked", "crap"),
        ("fucking", "horseshit", "bastard", "fuck", "pissed", "shitty", "bullshit"),
    ]

    idx = 0
    while len(pairs) < max_pairs:
        # 1. Profanity minimal pair
        prompt_base, frame = _PROFANITY_FRAMES[idx % len(_PROFANITY_FRAMES)]
        adj, noun, person, verb, mood, state, noun2 = prof_tuples[idx % len(prof_tuples)]
        neg = frame.format(
            adj=adj,
            noun=noun,
            person=person,
            verb=verb,
            mood=mood,
            state=state,
            noun2=noun2,
        )
        pos, n_rep = replace_profanity(neg)
        if n_rep > 0 and pos != neg:
            suffix = f" (Case {idx + 1})" if idx >= len(_PROFANITY_FRAMES) else ""
            pairs.append(
                ContrastivePair(
                    concept="profanity",
                    prompt=f"{prompt_base}{suffix}",
                    positive=pos,
                    negative=neg,
                    minimal=True,
                    source="minimal_lexicon",
                )
            )
        if len(pairs) >= max_pairs:
            break

        # 2. Lexical complexity minimal pair
        c_prompt, c_frame = _COMPLEX_FRAMES[idx % len(_COMPLEX_FRAMES)]
        c_neg = c_frame.format(
            v_improve="ameliorate",
            v_reduce="mitigate",
            adj_harmful="deleterious",
            adj_clear="pellucid",
            v_share="disseminate",
            v_confirm="corroborate",
            adj_extra="superfluous",
            adj_aware="cognizant",
            adj_unique="idiosyncratic",
            v_spread="proliferate",
            adj_dishonest="mendacious",
            v_confuse="obfuscate",
            v_worsen="exacerbate",
            adj_generous="magnanimous",
            adj_careless="perfunctory",
        )
        c_pos, c_rep = replace_complex_words(c_neg)
        if c_rep > 0 and c_pos != c_neg:
            suffix = f" (Topic {idx + 1})" if idx >= len(_COMPLEX_FRAMES) else ""
            pairs.append(
                ContrastivePair(
                    concept="lexical_complexity",
                    prompt=f"{c_prompt}{suffix}",
                    positive=c_pos,
                    negative=c_neg,
                    minimal=True,
                    source="minimal_lexicon",
                )
            )
        idx += 1

    rng.shuffle(pairs)
    return pairs[:max_pairs]


__all__ = ["PROFANITY_TO_SAFE", "COMPLEX_TO_SIMPLE", "build_minimal_pairs"]
