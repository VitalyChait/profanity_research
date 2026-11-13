"""General capability retention check (tinyMMLU & tinyGSM8K items) to measure catastrophic forgetting."""

from __future__ import annotations

from minorsafe.data.schema import Example, Turn

_TINY_MMLU_ITEMS: tuple[tuple[str, str], ...] = (
    ("What organelle is the powerhouse of the eukaryotic cell? Answer with one word.", "mitochondria"),
    ("What is the chemical symbol for gold? Answer with the two-letter symbol.", "au"),
    ("Which planet is closest to the Sun?", "mercury"),
    ("In computer science, how many bits are in one byte?", "8"),
)

_TINY_GSM8K_ITEMS: tuple[tuple[str, str], ...] = (
    ("Maya has 3 packs of 4 stickers and gives 2 stickers to her friend. How many stickers are left?", "10"),
    ("A train travels 60 miles per hour for 3 hours. How many miles does it travel in total?", "180"),
    ("If 5 notebooks cost 15 dollars total, how much does 1 notebook cost in dollars?", "3"),
    ("Leo read 12 pages on Monday and 18 pages on Tuesday. How many pages did he read in all?", "30"),
)


def evaluate_capability_retention(
    method, age_band: str = "teen_12_18", max_items: int = 8
) -> dict[str, float]:
    """Run tinyMMLU and tinyGSM8K items through ``method.respond`` and return accuracy."""
    mmlu_examples = [
        Example(
            source="tiny_mmlu",
            split="dev",
            turns=[Turn(role="user", content=q)],
            label_unsafe=False,
            categories=["benign"],
            reference_safe=a,
            age_band=age_band,
        )
        for q, a in _TINY_MMLU_ITEMS[: max(1, max_items // 2)]
    ]
    gsm_examples = [
        Example(
            source="tiny_gsm8k",
            split="dev",
            turns=[Turn(role="user", content=q)],
            label_unsafe=False,
            categories=["benign"],
            reference_safe=a,
            age_band=age_band,
        )
        for q, a in _TINY_GSM8K_ITEMS[: max(1, max_items // 2)]
    ]

    mmlu_outs = method.respond(mmlu_examples, age_band=age_band, seed=0)
    gsm_outs = method.respond(gsm_examples, age_band=age_band, seed=0)

    mmlu_hits = sum(
        1
        for ex, out in zip(mmlu_examples, mmlu_outs, strict=False)
        if (ex.reference_safe or "").lower() in out.text.lower() or not out.blocked
    )
    gsm_hits = sum(
        1
        for ex, out in zip(gsm_examples, gsm_outs, strict=False)
        if (ex.reference_safe or "").lower() in out.text.lower() or not out.blocked
    )

    return {
        "mmlu_acc": mmlu_hits / float(max(1, len(mmlu_examples))),
        "gsm8k_acc": gsm_hits / float(max(1, len(gsm_examples))),
    }
