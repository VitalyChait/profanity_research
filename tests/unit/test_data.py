"""Phase 3 unit tests: schema NFC & emoji preservation, lexicons, obfuscation, minimal pairs, splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from minorsafe.data.lexicons import (
    find_complex_spans,
    find_profanity_spans,
    is_english,
    replace_complex_words,
    replace_profanity,
)
from minorsafe.data.minimal_pairs import build_minimal_pairs
from minorsafe.data.neologisms import (
    ALL_NEOLOGISM_TERMS,
    BENIGN_NEOLOGISMS,
    TOXIC_NEOLOGISMS,
    build_procedural_neologism_examples,
    build_procedural_neologism_pairs,
    synthesize_neologism_data_with_llm,
)
from minorsafe.data.obfuscation import (
    apply_obfuscation,
    detect_obfuscated_profanity,
    generate_obfuscated_examples,
)
from minorsafe.data.schema import (
    Example,
    Turn,
    load_examples,
    load_pairs,
    save_examples,
    save_pairs,
)
from minorsafe.data.splits import assign_splits, check_split_leakage

pytestmark = pytest.mark.unit


def test_schema_preserves_emojis_and_special_chars(tmp_path: Path):
    raw_text = "Bro that boss fight cooked me 💀😭🔥! Wait—what's 2+2? @user #gg 🖕"
    ex = Example(
        source="test",
        turns=[Turn(role="user", content=raw_text)],
        label_unsafe=True,
        categories=["obfuscated", "slang"],
    )
    assert "💀😭🔥" in ex.last_user
    assert "🖕" in ex.last_user

    out_path = tmp_path / "examples.parquet"
    save_examples([ex], out_path)
    loaded = load_examples(out_path)
    assert len(loaded) == 1
    assert loaded[0].last_user == raw_text
    assert loaded[0].id == ex.id


def test_english_filter_ignores_emojis_and_symbols():
    assert is_english("Bro that play was insane 💀😭🔥💯!!", min_ratio=0.6) is True
    assert is_english("Plants need sunlight and water 🌱☀️💧", min_ratio=0.6) is True
    assert is_english("Ceci n'est pas une phrase anglaise du tout", min_ratio=0.6) is False


def test_profanity_and_complex_lexicon_replacement():
    dirty = "This fucking level is total bullshit and a🖕!"
    spans = find_profanity_spans(dirty)
    assert len(spans) >= 3
    cleaned, count = replace_profanity(dirty)
    assert count >= 3
    assert "fucking" not in cleaned.lower()
    assert "bullshit" not in cleaned.lower()
    assert "🖕" not in cleaned

    complex_sent = "We must ameliorate the deleterious and superfluous rules."
    c_spans = find_complex_spans(complex_sent)
    assert len(c_spans) == 3
    simplified, c_count = replace_complex_words(complex_sent)
    assert c_count == 3
    assert "improve" in simplified
    assert "harmful" in simplified
    assert "extra" in simplified


def test_obfuscation_transforms_and_detector():
    base = "That was bullshit and fuck you"
    for mode in ("leetspeak", "symbol_mask", "char_spacing", "homoglyphs", "algospeak", "emoji_profanity"):
        obf = apply_obfuscation(base, mode=mode, seed=7)
        assert obf != base
        hits = detect_obfuscated_profanity(obf)
        assert len(hits) >= 1, f"Detector missed obfuscation mode={mode}: {obf!r}"

    # Verify benign teen emojis do NOT trigger obfuscated profanity detector
    benign = "I got cooked in that math test 💀😭🔥✨"
    assert detect_obfuscated_profanity(benign) == []


def test_minimal_pairs_differ_only_in_target_spans(tmp_path: Path):
    pairs = build_minimal_pairs(max_pairs=20, seed=13)
    assert len(pairs) >= 10
    for p in pairs:
        assert p.minimal is True
        assert p.positive != p.negative
        assert p.concept in ("profanity", "lexical_complexity")
        if p.concept == "profanity":
            assert len(find_profanity_spans(p.positive)) == 0
            assert len(find_profanity_spans(p.negative)) >= 1
        elif p.concept == "lexical_complexity":
            assert len(find_complex_spans(p.positive)) == 0
            assert len(find_complex_spans(p.negative)) >= 1

    pair_path = tmp_path / "pairs.parquet"
    save_pairs(pairs, pair_path)
    loaded = load_pairs(pair_path)
    assert len(loaded) == len(pairs)


def test_deterministic_splits_and_zero_leakage():
    examples = []
    for i in range(60):
        examples.append(
            Example(
                source="synth",
                turns=[
                    Turn(role="user", content=f"Unique teen question number {i} about science and games?")
                ],
                label_unsafe=(i % 3 == 0),
                categories=["benign"] if i % 3 else ["profanity"],
            )
        )
    # Add a duplicate with different casing/punctuation to verify deduplication prevents leakage
    examples.append(
        Example(
            source="synth",
            turns=[Turn(role="user", content="UNIQUE teen question number 0 about science and games!!!")],
            label_unsafe=True,
            categories=["profanity"],
        )
    )

    s1 = assign_splits(examples, ratios={"calib": 0.5, "dev": 0.2, "test": 0.3}, seed=13)
    s2 = assign_splits(examples, ratios={"calib": 0.5, "dev": 0.2, "test": 0.3}, seed=13)
    assert [e.id for e in s1] == [e.id for e in s2]
    assert check_split_leakage(s1) == 0

    obf_examples = generate_obfuscated_examples(s1[:10], per_sentence=2, seed=13)
    assert len(obf_examples) >= 1
    assert all("obfuscated" in e.categories for e in obf_examples)


def test_neologism_corpus_and_llm_synthesis(monkeypatch):
    from minorsafe.config import load_models

    assert len(BENIGN_NEOLOGISMS) >= 20
    assert len(TOXIC_NEOLOGISMS) >= 20
    assert {"zorbmaxxing", "kriffled", "chokemaxx"}.issubset(ALL_NEOLOGISM_TERMS)
    assert is_english("Bro I got kriffled while zorbmaxxing 💀😭!!", min_ratio=0.6) is True

    examples = build_procedural_neologism_examples(max_items=60, seed=13)
    assert len(examples) == 60
    cats = {c for ex in examples for c in ex.categories}
    assert {
        "slang",
        "benign_tricky",
        "sexual",
        "self_harm",
        "drugs",
        "violence",
        "harassment",
        "profanity",
    }.issubset(cats)

    pairs = build_procedural_neologism_pairs(max_pairs=40, seed=13)
    assert len(pairs) == 40
    assert {p.concept for p in pairs} == {"slang_benign", "slang_toxic"}

    from minorsafe.data.synth import filter_contrastive_pairs

    kept, _ = filter_contrastive_pairs(pairs, toxicity_backend="lexicon")
    assert len(kept) >= 30

    # Mock cheaper teacher LLM responses for synthesize_neologism_data_with_llm
    def _fake_teacher(prompts, **_kwargs):
        all_entries = (*BENIGN_NEOLOGISMS, *TOXIC_NEOLOGISMS)
        out = []
        for idx, p in enumerate(prompts):
            term = next(e["term"] for e in all_entries if f"'{e['term']}'" in p)
            if '"positive"' in p:
                out.append(
                    (
                        f'{{"prompt": "Question {idx} about {term}?", '
                        f'"positive": "Clear standard English helpful answer number {idx} for teenagers.", '
                        f'"negative": "Bro {term} answer number {idx} with emojis 💀🔥 for teens."}}',
                        20,
                        30,
                    )
                )
            else:
                out.append(
                    (
                        f'{{"prompt": "Chat {idx} using {term} 💀?", "reference_safe": "Helpful safe response {idx}."}}',
                        15,
                        20,
                    )
                )
        return out

    import minorsafe.data.synth as synth_mod

    monkeypatch.setattr(synth_mod, "generate_teacher_texts", _fake_teacher)
    teacher_spec = load_models()["qwen25_3b_teacher"]
    llm_ex, llm_pairs, toks = synthesize_neologism_data_with_llm(
        teacher_spec=teacher_spec,
        teacher_backend="vllm",
        n_examples=6,
        n_pairs=6,
    )
    assert len(llm_ex) == 6
    assert len(llm_pairs) == 6
    assert toks > 0
    assert all(ex.meta.get("neologism") is True for ex in llm_ex)
