"""End-to-end dataset builder and data card generator (Phase 3)."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minorsafe.config import DataSpec, ModelSpec, load_data_spec, load_hardware, load_models
from minorsafe.data.lexicons import is_english
from minorsafe.data.neologisms import synthesize_neologism_data_with_llm
from minorsafe.data.obfuscation import generate_obfuscated_examples
from minorsafe.data.provenance import (
    PROCESSED_FILES,
    BuildFingerprint,
    DataStatus,
    build_fingerprint,
    clear_build_meta,
    diff_fingerprint,
    processed_digest,
    read_build_meta,
    write_build_meta,
)
from minorsafe.data.schema import ContrastivePair, Example, save_examples, save_pairs
from minorsafe.data.sources import load_all_sources
from minorsafe.data.splits import assign_splits, check_split_leakage
from minorsafe.data.synth import synthesize_contrastive_pairs
from minorsafe.hardware import resolve_profile_name
from minorsafe.telemetry.cost import gpu_allocation
from minorsafe.utils import DATA_DIR, RESULTS_DIR, get_logger, write_json

log = get_logger(__name__)

# Set to 1 to let production runs use a dataset that ``check_dataset`` reports as not fresh.
ACCEPT_STALE_DATA_ENV = "MINORSAFE_ACCEPT_STALE_DATA"


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


@dataclass(frozen=True)
class BuildPlan:
    """How a build synthesises data: resolved from offline/profile/teacher options."""

    offline: bool
    hardware_profile: str | None  # resolved profile name; None for offline builds
    teacher_alias: str | None
    teacher_spec: ModelSpec | None
    teacher_backend: str
    device: str
    vllm_util: float
    tp_size: int
    toxicity_backend: str

    def teacher_descriptor(self) -> dict[str, Any]:
        return {
            "offline": self.offline,
            "alias": self.teacher_alias if self.teacher_spec is not None else None,
            "backend": self.teacher_backend,
            "device": self.device,
            "spec": self.teacher_spec.model_dump(mode="json") if self.teacher_spec is not None else None,
        }


def resolve_build_plan(
    spec: DataSpec,
    *,
    offline: bool,
    hardware_profile: str | None,
    teacher_override: str | None = None,
) -> BuildPlan:
    if offline or not hardware_profile:
        return BuildPlan(
            offline=offline,
            hardware_profile=None,
            teacher_alias=None,
            teacher_spec=None,
            teacher_backend="mock",
            device="cpu",
            vllm_util=0.85,
            tp_size=1,
            toxicity_backend="lexicon",
        )
    resolved = resolve_profile_name(hardware_profile)
    hw = load_hardware(resolved)
    teacher_alias = teacher_override or spec.synth_teacher or hw.teacher
    teacher_spec = None
    if teacher_alias:
        models = load_models()
        if teacher_override and teacher_alias not in models:
            raise KeyError(f"Unknown teacher alias {teacher_alias!r} in models.yaml")
        teacher_spec = models.get(teacher_alias)
    return BuildPlan(
        offline=False,
        hardware_profile=resolved,
        teacher_alias=teacher_alias,
        teacher_spec=teacher_spec,
        teacher_backend=hw.teacher_backend,
        device=hw.device,
        vllm_util=hw.vllm_gpu_memory_utilization,
        tp_size=max(1, hw.num_gpus) if hw.device == "cuda" else 1,
        toxicity_backend=hw.eval.toxicity_backend,
    )


def _plan_fingerprint(spec: DataSpec, plan: BuildPlan) -> BuildFingerprint:
    return build_fingerprint(spec, teacher=plan.teacher_descriptor(), toxicity_backend=plan.toxicity_backend)


def build_dataset(
    spec: DataSpec | None = None,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    offline: bool = True,
    hardware_profile: str | None = None,
    teacher_override: str | None = None,
    overrides: list[str] | None = None,
) -> tuple[list[Example], list[ContrastivePair], Path]:
    """Build processed examples.parquet, pairs.parquet, build_meta.json, and results/data_card.md.

    ``overrides`` are the ``-o`` overrides ``spec`` was loaded with (``spec`` is loaded with them when
    omitted). They are recorded so ``check_dataset`` can rebuild the same spec for comparison.
    """
    spec = spec or load_data_spec(overrides=overrides)
    data_dir = Path(data_dir or DATA_DIR)
    results_dir = Path(results_dir or RESULTS_DIR)
    proc_dir = data_dir / "processed"

    plan = resolve_build_plan(
        spec, offline=offline, hardware_profile=hardware_profile, teacher_override=teacher_override
    )
    # Fingerprint the code before it runs; drop the old record so a crashed build cannot look current.
    fingerprint = _plan_fingerprint(spec, plan)
    clear_build_meta(proc_dir)
    teacher_spec = plan.teacher_spec
    teacher_backend = plan.teacher_backend
    device = plan.device
    vllm_util = plan.vllm_util
    tp_size = plan.tp_size
    tox_backend = plan.toxicity_backend

    raw_examples = load_all_sources(spec.sources, offline=offline)
    english_examples = [e for e in raw_examples if is_english(e.last_user, min_ratio=spec.english_min_ratio)]
    obf_examples = generate_obfuscated_examples(
        english_examples[: max(4, len(english_examples) // 3)],
        per_sentence=spec.obfuscation_per_sentence,
        seed=spec.seed,
    )

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

    save_examples(all_examples, proc_dir / "examples.parquet")
    save_pairs(all_pairs, proc_dir / "pairs.parquet")
    write_json(results_dir / "data_synth_setup.json", synth_rep.to_dict())
    card_path = write_data_card(all_examples, all_pairs, results_dir / "data_card.md")
    meta = write_build_meta(
        proc_dir,
        fingerprint,
        options={
            "offline": plan.offline,
            "hardware_profile": plan.hardware_profile,
            "teacher_override": teacher_override,
            "overrides": list(overrides or []),
        },
        n_examples=len(all_examples),
        n_pairs=len(all_pairs),
    )
    log.info(
        "Dataset built: %d examples, %d pairs -> %s (fingerprint %s, data %s)",
        len(all_examples),
        len(all_pairs),
        proc_dir,
        meta["fingerprint"],
        meta["data_digest"],
    )
    return all_examples, all_pairs, card_path


def check_dataset(
    data_dir: str | Path | None = None,
    *,
    offline: bool | None = None,
    hardware_profile: str | None = None,
) -> DataStatus:
    """Compare ``data/processed`` with what ``build_dataset`` would produce from the current checkout.

    ``offline``/``hardware_profile`` describe the build that is expected; ``None`` reuses what the existing
    build recorded, so only code/config drift is checked. The recorded teacher override and spec overrides
    are always reused, since they are deliberate choices rather than drift.
    """
    proc_dir = Path(data_dir or DATA_DIR) / "processed"
    missing = [name for name in PROCESSED_FILES if not (proc_dir / name).exists()]
    if missing:
        return DataStatus("missing", [f"{', '.join(missing)} not found in {proc_dir}"])
    digest = processed_digest(proc_dir)
    meta = read_build_meta(proc_dir)
    if meta is None:
        return DataStatus(
            "unrecorded",
            [
                f"{proc_dir} has no build_meta.json, so it was built by code that predates build "
                "provenance and its inputs are unknown"
            ],
            digest,
        )

    reasons: list[str] = []
    if meta.get("data_digest") != digest:
        reasons.append("examples/pairs parquet changed after the recorded build")
    opts = meta.get("options") or {}
    want_offline = bool(opts.get("offline")) if offline is None else offline
    want_profile = opts.get("hardware_profile") if hardware_profile is None else hardware_profile
    try:
        spec = load_data_spec(overrides=list(opts.get("overrides") or []) or None)
        plan = resolve_build_plan(
            spec,
            offline=want_offline,
            hardware_profile=want_profile,
            teacher_override=opts.get("teacher_override"),
        )
        ignore: tuple[str, ...] = ()
        if bool(opts.get("offline")) and not want_offline:
            reasons.append("built offline (seed corpus, mock teacher), but an online build is expected")
            ignore = ("teacher", "pair_filter")
        reasons.extend(diff_fingerprint(meta, _plan_fingerprint(spec, plan), ignore=ignore))
    except Exception as exc:  # unknown profile/teacher, unreadable config, ...
        reasons.append(f"cannot recompute the expected build fingerprint: {exc}")
    return DataStatus("stale" if reasons else "fresh", reasons, digest, meta)


def stale_data_accepted() -> bool:
    return os.environ.get(ACCEPT_STALE_DATA_ENV, "0") == "1"


def ensure_dataset(
    data_dir: str | Path | None = None,
    *,
    strict: bool,
    results_dir: str | Path | None = None,
) -> DataStatus:
    """Check ``data/processed`` before an experiment or tomography reads it.

    ``strict`` is for production GPU runs. A missing dataset is an error there: it is never silently
    replaced by the offline seed corpus. A dataset that is unrecorded, stale or built offline is refused
    unless ``MINORSAFE_ACCEPT_STALE_DATA=1``. Without ``strict`` (CPU/debug), a missing dataset is built
    offline and anything else is only logged.
    """
    base = Path(data_dir or DATA_DIR)
    proc_dir = base / "processed"
    status = check_dataset(base, offline=False if strict else None)
    if status.state == "missing":
        if strict:
            raise RuntimeError(
                f"No dataset in {proc_dir} ({status.describe()}). Build it first: "
                "`minorsafe data build --hardware <profile>` (or `scripts/run_phase.sh data <profile>`)."
            )
        log.warning("No dataset in %s; building the offline seed-corpus dataset.", proc_dir)
        build_dataset(load_data_spec(), data_dir=base, results_dir=results_dir or RESULTS_DIR, offline=True)
        return check_dataset(base)
    if status.fresh:
        log.info("Dataset %s", status.describe())
        return status
    msg = (
        f"Dataset in {proc_dir} is {status.describe()}. Rebuild it with `minorsafe data build --hardware "
        "<profile>` and move results computed from the old data aside, or set "
        f"{ACCEPT_STALE_DATA_ENV}=1 to use it anyway."
    )
    if strict and not stale_data_accepted():
        raise RuntimeError(msg)
    if strict:
        log.warning("%s (accepted via %s=1)", msg, ACCEPT_STALE_DATA_ENV)
    else:
        log.info(msg)
    return status
