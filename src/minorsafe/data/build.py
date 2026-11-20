"""End-to-end dataset builder and data card generator (Phase 3)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from minorsafe.config import DataSpec, load_data_spec, load_hardware, load_models
from minorsafe.data.lexicons import is_english
from minorsafe.data.neologisms import synthesize_neologism_data_with_llm
from minorsafe.data.obfuscation import generate_obfuscated_examples
from minorsafe.data.schema import ContrastivePair, Example, save_examples, save_pairs
from minorsafe.data.sources import load_all_sources
from minorsafe.data.splits import assign_splits, check_split_leakage
from minorsafe.data.synth import synthesize_contrastive_pairs
from minorsafe.hardware import resolve_profile_name
from minorsafe.telemetry.cost import gpu_allocation
from minorsafe.utils import DATA_DIR, RESULTS_DIR, get_logger, write_json

log = get_logger(__name__)


def write_data_card(
    examples: list[Example],
    pairs: list[ContrastivePair],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_counts = Counter(e.split for e in examples)
    source_counts = Counter(e.source for e in examples)
    cat_counts: Counter[str] = Counter()
    for e in examples:
        for c in e.categories:
            cat_counts[c] += 1

    pair_concepts = Counter(p.concept for p in pairs)

    lines = [
        "# MinorSafe Dataset Card (`teen_12_18`)",
        "",
        "## 1. Summary",
        f"- **Total Evaluation Examples**: {len(examples)}",
        f"- **Total Contrastive Pairs**: {len(pairs)}",
        f"- **Cross-Split Leakage (exact text or prompt family incl. obfuscated variants)**: {check_split_leakage(examples)} (Examples), {check_split_leakage(pairs)} (Pairs)",
        "- **Unicode & Emoji Policy**: Strict `NFC` normalization; all emojis (`💀`, `😭`, `🔥`, `🖕`) and special characters preserved.",
        "",
        "## 2. Examples by Split",
        "| Split | Count |",
        "| :--- | ---: |",
    ]
    for s in ("calib", "dev", "test"):
        lines.append(f"| `{s}` | {split_counts.get(s, 0)} |")

    lines.extend(["", "## 3. Examples by Category", "| Category | Count |", "| :--- | ---: |"])
    for cat, cnt in sorted(cat_counts.items()):
        lines.append(f"| `{cat}` | {cnt} |")

    lines.extend(
        [
            "",
            "## 4. Contrastive Pairs by Concept & Split",
            "| Concept | Total | Calib | Dev | Test |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for c in sorted(pair_concepts):
        c_items = [p for p in pairs if p.concept == c]
        c_sp = Counter(p.split for p in c_items)
        lines.append(
            f"| `{c}` | {len(c_items)} | {c_sp.get('calib', 0)} | {c_sp.get('dev', 0)} | {c_sp.get('test', 0)} |"
        )

    lines.extend(["", "## 5. Examples by Source", "| Source | Count |", "| :--- | ---: |"])
    for src, cnt in sorted(source_counts.items()):
        lines.append(f"| `{src}` | {cnt} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def build_dataset(
    spec: DataSpec | None = None,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    offline: bool = True,
    hardware_profile: str | None = None,
    teacher_override: str | None = None,
) -> tuple[list[Example], list[ContrastivePair], Path]:
    """Build processed examples.parquet, pairs.parquet, and results/data_card.md."""
    spec = spec or load_data_spec()
    data_dir = Path(data_dir or DATA_DIR)
    results_dir = Path(results_dir or RESULTS_DIR)

    raw_examples = load_all_sources(spec.sources, offline=offline)
    english_examples = [e for e in raw_examples if is_english(e.last_user, min_ratio=spec.english_min_ratio)]
    obf_examples = generate_obfuscated_examples(
        english_examples[: max(4, len(english_examples) // 3)],
        per_sentence=spec.obfuscation_per_sentence,
        seed=spec.seed,
    )

    teacher_spec = None
    teacher_backend = "mock"
    device = "cpu"
    vllm_util = 0.85
    tp_size = 1
    tox_backend = "lexicon"
    if not offline and hardware_profile:
        hw = load_hardware(resolve_profile_name(hardware_profile))
        teacher_backend = hw.teacher_backend
        device = hw.device
        vllm_util = hw.vllm_gpu_memory_utilization
        tp_size = max(1, hw.num_gpus) if hw.device == "cuda" else 1
        tox_backend = hw.eval.toxicity_backend
        teacher_alias = teacher_override or spec.synth_teacher or hw.teacher
        if teacher_alias:
            models = load_models()
            if teacher_override and teacher_alias not in models:
                raise KeyError(f"Unknown teacher alias {teacher_alias!r} in models.yaml")
            teacher_spec = models.get(teacher_alias)

    neo_examples: list[Example] = []
    neo_pairs: list[ContrastivePair] = []
    neo_tokens = 0
    if not offline and teacher_spec is not None and teacher_backend in ("vllm", "hf"):
        neo_examples, neo_pairs, neo_tokens = synthesize_neologism_data_with_llm(
            teacher_spec=teacher_spec,
            teacher_backend=teacher_backend,
            n_examples=spec.synth_neologism_examples,
            n_pairs=spec.synth_neologism_pairs,
            device=device,
            vllm_gpu_memory_utilization=vllm_util,
            seed=spec.seed,
            vllm_tensor_parallel_size=tp_size,
        )
        neo_examples = [e for e in neo_examples if is_english(e.last_user, min_ratio=spec.english_min_ratio)]

    all_examples = assign_splits(
        [*english_examples, *obf_examples, *neo_examples], ratios=spec.split_ratios, seed=spec.seed
    )

    with gpu_allocation(tp_size):
        raw_pairs, synth_rep = synthesize_contrastive_pairs(
            concepts=spec.concepts,
            pairs_per_concept=min(
                spec.synth_pairs_per_concept, 60 if offline else spec.synth_pairs_per_concept
            ),
            seed=spec.seed,
            teacher_spec=teacher_spec,
            teacher_backend=teacher_backend,
            oversample=spec.synth_oversample,
            device=device,
            vllm_gpu_memory_utilization=vllm_util,
            toxicity_backend=tox_backend,
            extra_pairs=neo_pairs,
            extra_tokens=neo_tokens,
            vllm_tensor_parallel_size=tp_size,
        )
    all_pairs = assign_splits(raw_pairs, ratios=spec.pair_split_ratios, seed=spec.seed)

    proc_dir = data_dir / "processed"
    save_examples(all_examples, proc_dir / "examples.parquet")
    save_pairs(all_pairs, proc_dir / "pairs.parquet")
    write_json(results_dir / "data_synth_setup.json", synth_rep.to_dict())
    card_path = write_data_card(all_examples, all_pairs, results_dir / "data_card.md")
    log.info(
        "Dataset built: %d examples, %d pairs -> %s",
        len(all_examples),
        len(all_pairs),
        proc_dir,
    )
    return all_examples, all_pairs, card_path
