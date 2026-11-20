"""Teacher contrastive pair synthesis and strict quality filtering (FK grade, toxicity, length parity, MinHash)."""

from __future__ import annotations

import gc

import textstat
import torch
from datasketch import MinHash, MinHashLSH

from minorsafe.config import ModelSpec, load_prompts
from minorsafe.data.lexicons import find_profanity_spans, is_english
from minorsafe.data.minimal_pairs import build_minimal_pairs
from minorsafe.data.neologisms import build_procedural_neologism_pairs
from minorsafe.data.obfuscation import detect_obfuscated_profanity
from minorsafe.data.schema import ContrastivePair
from minorsafe.data.sources import load_seed_pairs
from minorsafe.eval.safety import compute_safety_metrics
from minorsafe.generation import ChatGenerator
from minorsafe.registry import checkpoint_id, free_model, load_decoder, load_tokenizer
from minorsafe.telemetry.cost import SetupReport
from minorsafe.utils import extract_json_object, get_logger, normalize_text

log = get_logger(__name__)


def _shingle_minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    words = normalize_text(text).split()
    for i in range(max(1, len(words) - 2)):
        shingle = " ".join(words[i : i + 3])
        m.update(shingle.encode("utf-8"))
    return m


def filter_contrastive_pairs(
    pairs: list[ContrastivePair],
    english_min_ratio: float = 0.60,
    max_length_rel_diff: float = 0.35,
    jaccard_threshold: float = 0.80,
    toxicity_backend: str = "lexicon",
    enforce_fk: bool = True,
) -> tuple[list[ContrastivePair], dict[str, int]]:
    """Filter pairs by English ratio, length parity, positive/negative quality, and MinHash deduplication."""
    stats = {
        "input": len(pairs),
        "dropped_english": 0,
        "dropped_positive_unsafe": 0,
        "dropped_negative_weak": 0,
        "dropped_length_parity": 0,
        "dropped_duplicate": 0,
        "kept": 0,
    }
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=64)
    kept: list[ContrastivePair] = []

    for idx, p in enumerate(pairs):
        if not (p.prompt or "").strip() or not (p.positive or "").strip() or not (p.negative or "").strip():
            stats["dropped_positive_unsafe"] += 1
            continue
        if p.positive.strip() == p.negative.strip():
            stats["dropped_negative_weak"] += 1
            continue
        if not is_english(p.positive, min_ratio=english_min_ratio):
            stats["dropped_english"] += 1
            continue
        if find_profanity_spans(p.positive) or detect_obfuscated_profanity(p.positive):
            stats["dropped_positive_unsafe"] += 1
            continue

        if toxicity_backend == "detoxify":
            pos_tox = compute_safety_metrics(p.positive, backend="detoxify")["toxicity_score"]
            if pos_tox >= 0.10:
                stats["dropped_positive_unsafe"] += 1
                continue
            if p.concept in ("profanity", "mature_theme", "slang_toxic"):
                neg_has_lex = bool(
                    find_profanity_spans(p.negative) or detect_obfuscated_profanity(p.negative)
                )
                neo_term = (
                    str(p.meta.get("neologism_term") or "").strip().lower()
                    if isinstance(p.meta, dict) and p.meta.get("neologism")
                    else ""
                )
                neg_has_neo = bool(
                    neo_term and neo_term in p.negative.lower() and neo_term not in p.positive.lower()
                )
                neg_tox = compute_safety_metrics(p.negative, backend="detoxify")["toxicity_score"]
                if not neg_has_lex and not neg_has_neo and neg_tox <= 0.45:
                    stats["dropped_negative_weak"] += 1
                    continue

        if enforce_fk and p.concept == "lexical_complexity":
            pos_fk = textstat.flesch_kincaid_grade(p.positive)
            neg_fk = textstat.flesch_kincaid_grade(p.negative)
            # Minimal lexical pairs use a single long sentence (inflating base FK); teacher pairs must stay <= 9.0.
            if (neg_fk - pos_fk) < 1.5 or (not p.minimal and pos_fk > 9.0):
                stats["dropped_positive_unsafe"] += 1
                continue

        len_pos = max(1, len(p.positive.split()))
        len_neg = max(1, len(p.negative.split()))
        rel_diff = abs(len_pos - len_neg) / float(max(len_pos, len_neg))
        if rel_diff > max_length_rel_diff:
            stats["dropped_length_parity"] += 1
            continue

        mh = _shingle_minhash(f"{p.concept} {p.prompt} {p.positive} {p.negative}")
        if lsh.query(mh):
            stats["dropped_duplicate"] += 1
            continue
        lsh.insert(f"p_{idx}", mh)
        kept.append(p)

    stats["kept"] = len(kept)
    return kept, stats


def generate_teacher_texts(
    prompts: list[str],
    teacher_spec: ModelSpec,
    teacher_backend: str,
    device: str = "cuda",
    vllm_gpu_memory_utilization: float = 0.85,
    seed: int = 13,
    max_tokens: int = 512,
) -> list[tuple[str, int, int]]:
    """Run batched generation over ``prompts`` using a teacher LLM via vLLM or HF."""
    if not prompts:
        return []

    quant = "awq" if "awq" in teacher_spec.hf_id.lower() else ("bf16" if device == "cuda" else "fp32")
    raw_outputs: list[tuple[str, int, int]] = []
    if teacher_backend == "vllm":
        from vllm import LLM, SamplingParams

        tok = load_tokenizer(teacher_spec, quant)
        llm = LLM(
            model=checkpoint_id(teacher_spec, quant),
            dtype="auto",
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=2048,
            seed=seed,
            trust_remote_code=False,
        )
        try:
            rendered = []
            for p in prompts:
                if getattr(tok, "chat_template", None):
                    try:
                        rendered.append(
                            tok.apply_chat_template(
                                [{"role": "user", "content": p}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                        continue
                    except Exception:
                        pass
                rendered.append(p)
            params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
            outs = llm.generate(rendered, params)
            for req_out in outs:
                txt = req_out.outputs[0].text if req_out.outputs else ""
                n_in = len(req_out.prompt_token_ids or ())
                n_out = len(req_out.outputs[0].token_ids) if req_out.outputs else 0
                raw_outputs.append((txt, n_in, n_out))
        finally:
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel

                destroy_model_parallel()
            except Exception:
                pass
            del llm
            free_model(tok)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif teacher_backend == "hf":
        model, tok, _ = load_decoder(teacher_spec, quant=quant, device=device)
        try:
            gen = ChatGenerator(model, tok, spec=teacher_spec, max_new_tokens=max_tokens, temperature=0.8)
            for i in range(0, len(prompts), 8):
                chunk_msgs = [[{"role": "user", "content": p}] for p in prompts[i : i + 8]]
                gens = gen.generate(chunk_msgs, seed=seed + i)
                for g in gens:
                    raw_outputs.append((g.text, g.n_in, g.n_out + g.n_reasoning))
        finally:
            free_model(model, tok)
    else:
        raise ValueError(f"Unsupported teacher_backend {teacher_backend!r}")
    return raw_outputs


def _generate_teacher_pairs(
    concepts: list[str],
    pairs_per_concept: int,
    oversample: float,
    teacher_spec: ModelSpec,
    teacher_backend: str,
    device: str = "cuda",
    vllm_gpu_memory_utilization: float = 0.85,
    seed: int = 13,
) -> tuple[list[ContrastivePair], int]:
    """Synthesize contrastive pairs using the configured teacher LLM via vLLM or HF."""
    templates = load_prompts().get("synth_pair_templates", {})
    n_req_per_concept = max(1, int(round(pairs_per_concept * oversample)))

    prompts_meta: list[tuple[str, str]] = []
    for concept in concepts:
        tpl = templates.get(concept)
        if not tpl:
            continue
        for k in range(n_req_per_concept):
            user_prompt = (
                f"{tpl}\n\n"
                f"Generate a fresh, realistic, unique example #{k + 1} (avoid repeating common cliches). "
                "Output ONLY the JSON object."
            )
            prompts_meta.append((concept, user_prompt))

    if not prompts_meta:
        return [], 0

    raw_outputs = generate_teacher_texts(
        prompts=[p for _, p in prompts_meta],
        teacher_spec=teacher_spec,
        teacher_backend=teacher_backend,
        device=device,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        seed=seed,
        max_tokens=512,
    )

    pairs: list[ContrastivePair] = []
    total_tokens = 0
    for (concept, _), (txt, n_in, n_out) in zip(prompts_meta, raw_outputs, strict=True):
        total_tokens += n_in + n_out
        obj = extract_json_object(txt)
        if not isinstance(obj, dict):
            continue
        pr = str(obj.get("prompt") or "").strip()
        pos = str(obj.get("positive") or "").strip()
        neg = str(obj.get("negative") or "").strip()
        if pr and pos and neg:
            pairs.append(
                ContrastivePair(
                    concept=concept,
                    prompt=pr,
                    positive=pos,
                    negative=neg,
                    source=f"teacher_{teacher_spec.alias}",
                    minimal=False,
                    age_band="teen_12_18",
                )
            )
    log.info(
        "Teacher %s generated %d raw parseable pairs across %s", teacher_spec.alias, len(pairs), concepts
    )
    return pairs, total_tokens


def synthesize_contrastive_pairs(
    concepts: list[str],
    pairs_per_concept: int = 40,
    seed: int = 13,
    setup_report: SetupReport | None = None,
    teacher_spec: ModelSpec | None = None,
    teacher_backend: str = "mock",
    oversample: float = 2.0,
    device: str = "cpu",
    vllm_gpu_memory_utilization: float = 0.85,
    toxicity_backend: str = "lexicon",
    extra_pairs: list[ContrastivePair] | None = None,
    extra_tokens: int = 0,
) -> tuple[list[ContrastivePair], SetupReport]:
    """Generate and validate contrastive pairs across requested concepts, logging SetupReport."""
    rep = setup_report or SetupReport(method="data_synth", setup_key=f"synth_{teacher_backend}_{seed}")
    with rep.stage("synth_pairs", examples=len(concepts) * pairs_per_concept) as st:
        minimal = build_minimal_pairs(max_pairs=pairs_per_concept * 2, seed=seed)
        seed_p = load_seed_pairs(max_items=pairs_per_concept * 3)
        neo_p = build_procedural_neologism_pairs(max_pairs=pairs_per_concept * 2, seed=seed)
        teacher_p: list[ContrastivePair] = []
        if teacher_backend in ("vllm", "hf") and teacher_spec is not None:
            teacher_p, teacher_toks = _generate_teacher_pairs(
                concepts=concepts,
                pairs_per_concept=pairs_per_concept,
                oversample=oversample,
                teacher_spec=teacher_spec,
                teacher_backend=teacher_backend,
                device=device,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                seed=seed,
            )
            st.tokens += teacher_toks
        st.tokens += int(extra_tokens)

        raw_pool = [
            p for p in (*minimal, *seed_p, *neo_p, *teacher_p, *(extra_pairs or ())) if p.concept in concepts
        ]
        filtered, f_stats = filter_contrastive_pairs(raw_pool, toxicity_backend=toxicity_backend)
        log.info("Filtered contrastive pairs: %s", f_stats)
        if teacher_backend == "mock" or teacher_spec is None:
            st.tokens = sum(len((p.prompt + p.positive + p.negative).split()) * 2 for p in filtered)
    return filtered, rep
