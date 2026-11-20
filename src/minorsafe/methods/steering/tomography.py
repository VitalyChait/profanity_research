"""Latent Artificial Tomography (LAT): subcultural sociolect entanglement matrix and Logit Lens (Phase 5.4.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from minorsafe.config import load_data_spec, load_hardware, load_models
from minorsafe.data.build import build_dataset
from minorsafe.data.schema import ContrastivePair, load_pairs
from minorsafe.hardware import resolve_profile_name
from minorsafe.methods.steering.extract import calibrate_hyperplane_from_activations, capture_pair_activations
from minorsafe.registry import free_model, get_decoder_layers, load_decoder
from minorsafe.utils import DATA_DIR, RESULTS_DIR, get_logger, write_json

log = get_logger(__name__)


@torch.no_grad()
def _logit_lens_tokens(
    model, tokenizer, vector: torch.Tensor, top_k: int = 10
) -> tuple[list[str], list[str]]:
    """Project steering direction through final LayerNorm + lm_head to inspect promoted/suppressed tokens."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return [f"tok+{i}" for i in range(top_k)], [f"tok-{i}" for i in range(top_k)]

    v = vector.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    norm_mod = getattr(getattr(model, "model", None), "norm", None)
    if norm_mod is not None:
        v = norm_mod(v.unsqueeze(0)).squeeze(0)
    logits = lm_head(v).float()
    k = min(top_k, logits.numel())
    top_pos = torch.topk(logits, k=k, largest=True).indices.tolist()
    top_neg = torch.topk(logits, k=k, largest=False).indices.tolist()

    def _tok_str(tid: int) -> str:
        try:
            s = tokenizer.decode([tid], skip_special_tokens=False)
            return s.strip() or f"id:{tid}"
        except Exception:
            return f"id:{tid}"

    return [_tok_str(i) for i in top_pos], [_tok_str(i) for i in top_neg]


def run_latent_tomography(
    model,
    tokenizer,
    pairs: list[ContrastivePair],
    concepts: list[str] | None = None,
    top_k_tokens: int = 10,
    n_calib: int = 40,
) -> dict:
    """Compute layer-wise AUROC, cross-concept cosine similarity matrices, and Logit Lens vocabulary projections."""
    concepts = concepts or ["profanity", "lexical_complexity", "slang_toxic", "slang_benign"]
    layers = get_decoder_layers(model)
    layer_indices = list(range(len(layers)))

    dirs_by_concept_layer: dict[str, dict[int, object]] = {c: {} for c in concepts}
    for c in concepts:
        c_pairs = [p for p in pairs if p.concept == c][:n_calib]
        if not c_pairs:
            available = sorted({p.concept for p in pairs})
            raise ValueError(
                f"No pairs for tomography concept {c!r} (available: {available}). "
                "Refusing to fall back to other concepts' pairs."
            )
        acts, _ = capture_pair_activations(model, tokenizer, c_pairs, layer_indices=layer_indices)
        for l_idx in layer_indices:
            xp, xn = acts[l_idx]
            dirs_by_concept_layer[c][l_idx] = calibrate_hyperplane_from_activations(
                xp, xn, concept=c, layer_idx=l_idx, estimator="pca"
            )

    layer_reports = []
    for l_idx in layer_indices:
        vecs = []
        concept_entries = {}
        for c in concepts:
            cd = dirs_by_concept_layer[c][l_idx]
            v_np = cd.vector.numpy().astype(np.float64)
            v_np = v_np / max(1e-8, np.linalg.norm(v_np))
            vecs.append(v_np)
            promoted, suppressed = _logit_lens_tokens(model, tokenizer, cd.vector, top_k=top_k_tokens)
            concept_entries[c] = {
                "auroc": float(cd.separability_auroc),
                "m_l": float(cd.m_l),
                "s_l": float(cd.s_l),
                "mu_pos": float(cd.mu_pos),
                "mu_neg": float(cd.mu_neg),
                "promoted_tokens": promoted,
                "suppressed_tokens": suppressed,
            }
        V = np.vstack(vecs)  # [C, D]
        cos_mat = np.clip(V @ V.T, -1.0, 1.0)
        np.fill_diagonal(cos_mat, 1.0)
        layer_reports.append(
            {
                "layer_idx": l_idx,
                "rel_depth": l_idx / float(max(1, len(layers) - 1)),
                "concepts": concept_entries,
                "concept_names": list(concepts),
                "cosine_matrix": cos_mat.tolist(),
            }
        )

    # Compute summary entanglement (profanity vs slang_benign) and mid-layer separability for H5 & H7.
    mid_layers = [lr for lr in layer_reports if 0.25 <= lr["rel_depth"] <= 0.75] or layer_reports
    prof_slang_cos: list[float] = []
    if "profanity" in concepts and "slang_benign" in concepts:
        i_p, i_s = concepts.index("profanity"), concepts.index("slang_benign")
        prof_slang_cos = [float(lr["cosine_matrix"][i_p][i_s]) for lr in mid_layers]

    mean_mid_auroc = {c: float(np.mean([lr["concepts"][c]["auroc"] for lr in mid_layers])) for c in concepts}

    return {
        "num_layers": len(layers),
        "concepts": list(concepts),
        "layers": layer_reports,
        "summary": {
            "mean_mid_profanity_slang_benign_cos": float(np.mean(prof_slang_cos)) if prof_slang_cos else 0.0,
            "max_mid_profanity_slang_benign_cos": float(np.max(prof_slang_cos)) if prof_slang_cos else 0.0,
            "mean_mid_auroc": mean_mid_auroc,
        },
    }


def run_tomography_for_profile(
    profile_name: str = "auto",
    model_aliases: list[str] | None = None,
    n_calib: int = 60,
    concepts: list[str] | None = None,
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> list[Path]:
    """Load each generator in ``profile_name``, run LAT tomography, and persist JSON artifacts."""
    resolved = resolve_profile_name(profile_name)
    hw = load_hardware(resolved)
    models = load_models()
    aliases = model_aliases or list(hw.generators)
    out_root = Path(out_dir) if out_dir else (RESULTS_DIR / "tomography")
    out_root.mkdir(parents=True, exist_ok=True)

    base_data = Path(data_dir) if data_dir else DATA_DIR
    pairs_path = base_data / "processed" / "pairs.parquet"
    if not pairs_path.exists():
        build_dataset(load_data_spec(), data_dir=base_data, results_dir=RESULTS_DIR, offline=True)
    calib_pairs = [p for p in load_pairs(pairs_path) if p.split == "calib"]

    written: list[Path] = []
    summaries: list[dict] = []
    for alias in aliases:
        if alias not in models:
            raise KeyError(f"Unknown generator alias {alias!r} in tomography")
        spec = models[alias]
        model, tok, _ = load_decoder(spec, quant=hw.default_quant, device=hw.device)
        try:
            report = run_latent_tomography(
                model=model,
                tokenizer=tok,
                pairs=calib_pairs,
                concepts=concepts,
                n_calib=n_calib,
            )
            report["model"] = alias
            report["family"] = spec.family
            report["size_group"] = spec.size_group or "S"
            report["params_b"] = float(spec.params_b or 0.0)
            report["quant"] = hw.default_quant
            out_path = out_root / f"{alias}.json"
            write_json(out_path, report)
            written.append(out_path)
            summaries.append(
                {
                    "model": alias,
                    "family": spec.family,
                    "size_group": spec.size_group or "S",
                    "params_b": float(spec.params_b or 0.0),
                    **report["summary"],
                }
            )
            log.info("Wrote tomography for %s -> %s", alias, out_path)
        finally:
            free_model(model, tok)

    write_json(out_root / "summary.json", {"profile": resolved, "models": summaries})
    return written
