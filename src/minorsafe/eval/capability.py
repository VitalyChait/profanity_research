"""General capability retention check (tinyMMLU & tinyGSM8K) to measure catastrophic forgetting."""

from __future__ import annotations

import re
from pathlib import Path

from minorsafe.data.schema import Example, Turn
from minorsafe.utils import DATA_DIR, get_logger, read_json, write_json

log = get_logger(__name__)

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

_LETTER_RE = re.compile(r"(?:answer(?:\s+is)?[:\s]*|^\s*|\()([A-D])(?:\)|\b)", re.IGNORECASE)
_GSM_HASH_RE = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"[-+]?[\d,]+(?:\.\d+)?")


def _extract_gsm_number(text: str) -> str:
    m = _GSM_HASH_RE.search(text or "")
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _NUM_RE.findall(text or "")
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return ""


def _extract_mmlu_choice(text: str) -> str:
    txt = (text or "").strip()
    m = _LETTER_RE.search(txt)
    if m:
        return m.group(1).upper()
    tokens = re.findall(r"\b([A-D])\b", txt.upper())
    return tokens[-1] if tokens else ""


def _load_real_capability_items(
    max_items: int, data_dir: Path | None = None
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Load & cache tinyMMLU (prompt, letter, choice_text) and tinyGSM8K (prompt, gold_num)."""
    from datasets import load_dataset

    cache_dir = (Path(data_dir) if data_dir else DATA_DIR) / "processed" / "capability"
    mmlu_cache = cache_dir / "tiny_mmlu.json"
    gsm_cache = cache_dir / "tiny_gsm8k.json"
    n_each = max(1, max_items // 2)

    if mmlu_cache.exists():
        mmlu_rows = read_json(mmlu_cache)
    else:
        ds = load_dataset("tinyBenchmarks/tinyMMLU", split="test")
        mmlu_rows = []
        letters = ("A", "B", "C", "D")
        for r in ds:
            choices = r.get("choices") or []
            ans_idx = int(r.get("answer", 0))
            if len(choices) < 4 or not (0 <= ans_idx < 4):
                continue
            opts = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(4))
            prompt = (
                f"Answer the multiple-choice question with the option letter (A, B, C, or D).\n\n"
                f"Question: {r['question']}\n{opts}\nAnswer:"
            )
            mmlu_rows.append([prompt, letters[ans_idx], str(choices[ans_idx])])
        write_json(mmlu_cache, mmlu_rows)

    if gsm_cache.exists():
        gsm_rows = read_json(gsm_cache)
    else:
        ds = load_dataset("tinyBenchmarks/tinyGSM8K", split="test")
        gsm_rows = []
        for r in ds:
            gold = _extract_gsm_number(str(r.get("answer") or ""))
            if not gold:
                continue
            prompt = (
                f"Solve the math problem step by step and end with '#### <number>'.\n\n"
                f"Question: {r['question']}"
            )
            gsm_rows.append([prompt, gold])
        write_json(gsm_cache, gsm_rows)

    return [(a, b, c) for a, b, c in mmlu_rows[:n_each]], [(a, b) for a, b in gsm_rows[:n_each]]


def evaluate_capability_retention(
    method,
    age_band: str = "teen_12_18",
    max_items: int = 100,
    use_real_benchmarks: bool = True,
    batch_size: int = 8,
    data_dir: Path | None = None,
) -> dict[str, float]:
    """Run tinyMMLU and tinyGSM8K items through ``method.respond`` and return exact-match accuracy."""
    if use_real_benchmarks:
        mmlu_items, gsm_items = _load_real_capability_items(max_items=max_items, data_dir=data_dir)
        mmlu_examples = [
            Example(
                source="tiny_mmlu",
                split="test",
                turns=[Turn(role="user", content=q)],
                label_unsafe=False,
                categories=["benign"],
                reference_safe=letter,
                age_band=age_band,
            )
            for q, letter, _ in mmlu_items
        ]
        gsm_examples = [
            Example(
                source="tiny_gsm8k",
                split="test",
                turns=[Turn(role="user", content=q)],
                label_unsafe=False,
                categories=["benign"],
                reference_safe=ans,
                age_band=age_band,
            )
            for q, ans in gsm_items
        ]

        mmlu_outs = []
        for i in range(0, len(mmlu_examples), batch_size):
            mmlu_outs.extend(method.respond(mmlu_examples[i : i + batch_size], age_band=age_band, seed=0))
        gsm_outs = []
        for i in range(0, len(gsm_examples), batch_size):
            gsm_outs.extend(method.respond(gsm_examples[i : i + batch_size], age_band=age_band, seed=0))

        mmlu_hits = 0
        for (_, letter, choice_txt), out in zip(mmlu_items, mmlu_outs, strict=True):
            if out.blocked:
                continue
            pred_letter = _extract_mmlu_choice(out.text)
            if pred_letter == letter or (choice_txt.lower() in out.text.lower() and len(choice_txt) >= 3):
                mmlu_hits += 1

        gsm_hits = 0
        for (_, gold), out in zip(gsm_items, gsm_outs, strict=True):
            if out.blocked:
                continue
            pred_num = _extract_gsm_number(out.text)
            if pred_num and pred_num == gold:
                gsm_hits += 1

        return {
            "mmlu_acc": mmlu_hits / float(max(1, len(mmlu_items))),
            "gsm8k_acc": gsm_hits / float(max(1, len(gsm_items))),
            "mmlu_n": int(len(mmlu_items)),
            "gsm8k_n": int(len(gsm_items)),
        }

    # Offline / cpu_debug test path (dummy models only)
    n_each = max(1, max_items // 2)
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
        for q, a in _TINY_MMLU_ITEMS[:n_each]
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
        for q, a in _TINY_GSM8K_ITEMS[:n_each]
    ]

    mmlu_outs = method.respond(mmlu_examples, age_band=age_band, seed=0)
    gsm_outs = method.respond(gsm_examples, age_band=age_band, seed=0)

    mmlu_hits = sum(
        1
        for ex, out in zip(mmlu_examples, mmlu_outs, strict=False)
        if (not out.blocked) and ((ex.reference_safe or "").lower() in out.text.lower())
    )
    gsm_hits = sum(
        1
        for ex, out in zip(gsm_examples, gsm_outs, strict=False)
        if (not out.blocked) and ((ex.reference_safe or "").lower() in out.text.lower())
    )

    return {
        "mmlu_acc": mmlu_hits / float(max(1, len(mmlu_examples))),
        "gsm8k_acc": gsm_hits / float(max(1, len(gsm_examples))),
        "mmlu_n": int(len(mmlu_examples)),
        "gsm8k_n": int(len(gsm_examples)),
    }
