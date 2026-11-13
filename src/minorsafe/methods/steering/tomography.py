"""Latent Artificial Tomography (LAT): subcultural sociolect entanglement matrix and Logit Lens (Phase 5.4.4)."""

from __future__ import annotations

import numpy as np
import torch

from minorsafe.data.schema import ContrastivePair
from minorsafe.methods.steering.extract import calibrate_hyperplane_from_activations, capture_pair_activations
from minorsafe.registry import get_decoder_layers


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

    # Extract per-layer unit vectors for every concept
    dirs_by_concept_layer: dict[str, dict[int, object]] = {c: {} for c in concepts}
    for c in concepts:
        c_pairs = [p for p in pairs if p.concept == c][:n_calib] or pairs[: max(2, min(8, len(pairs)))]
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

    return {
        "num_layers": len(layers),
        "concepts": list(concepts),
        "layers": layer_reports,
    }
