"""Contrastive vector extraction (CAA, PCA, Sociolect-Orthogonalized) and 1D LogReg hyperplane calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from minorsafe.data.schema import ContrastivePair
from minorsafe.generation import format_chat
from minorsafe.methods.steering.hooks import ConceptDirection
from minorsafe.registry import get_decoder_layers
from minorsafe.telemetry.cost import SetupReport
from minorsafe.utils import get_logger, read_json, stable_hash, write_json

log = get_logger(__name__)


def orthogonalize_vector(v_target: np.ndarray, v_nuisance: np.ndarray) -> np.ndarray:
    """Gram-Schmidt projection removal: v_ortho = v_target - <v_target, v_nuisance> v_nuisance."""
    u_t = v_target.astype(np.float64)
    u_n = v_nuisance.astype(np.float64)
    n_norm = np.linalg.norm(u_n)
    if n_norm > 1e-8:
        u_n = u_n / n_norm
        u_t = u_t - np.dot(u_t, u_n) * u_n
    t_norm = np.linalg.norm(u_t)
    if t_norm < 1e-8:
        out = np.zeros_like(v_target, dtype=np.float32)
        out[0] = 1.0
        return out
    return (u_t / t_norm).astype(np.float32)


def _compute_raw_direction(X_pos: np.ndarray, X_neg: np.ndarray, estimator: str = "pca") -> np.ndarray:
    diff_mean = np.mean(X_pos, axis=0) - np.mean(X_neg, axis=0)
    if estimator == "pca" and len(X_pos) >= 2 and X_pos.shape[1] >= 2:
        paired_diffs = (X_pos - X_neg).astype(np.float64)
        # RepE / LAT (Zou et al. 2023): symmetric pair-centered differences have zero mean,
        # so the first principal component (top singular vector of paired_diffs) isolates the concept axis.
        sym_diffs = np.vstack([paired_diffs, -paired_diffs])
        if np.var(sym_diffs) > 1e-10:
            pca = PCA(n_components=1, random_state=13)
            pca.fit(sym_diffs)
            v = pca.components_[0].astype(np.float64)
            if np.dot(v, diff_mean) < 0:
                v = -v
        else:
            v = diff_mean.astype(np.float64)
    else:
        v = diff_mean.astype(np.float64)

    norm = np.linalg.norm(v)
    if norm < 1e-8:
        v = np.zeros(X_pos.shape[1], dtype=np.float64)
        v[0] = 1.0
    else:
        v = v / norm
    return v.astype(np.float32)


def calibrate_hyperplane_from_activations(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    concept: str,
    layer_idx: int,
    estimator: str = "pca",
    target_offset_sigma: float = 0.5,
    nuisance_vector: np.ndarray | None = None,
    token_X_pos: np.ndarray | None = None,
    token_X_neg: np.ndarray | None = None,
    eval_X_pos: np.ndarray | None = None,
    eval_X_neg: np.ndarray | None = None,
) -> ConceptDirection:
    """Compute unit steering vector and calibrate 1D logistic regression threshold m_l and target s_l."""
    v_norm = _compute_raw_direction(X_pos, X_neg, estimator="pca" if estimator in ("pca", "ortho") else "caa")
    if nuisance_vector is not None:
        v_norm = orthogonalize_vector(v_norm, nuisance_vector)

    pair_pos_proj = np.dot(X_pos, v_norm).astype(np.float64)
    pair_neg_proj = np.dot(X_neg, v_norm).astype(np.float64)
    if np.mean(pair_pos_proj) < np.mean(pair_neg_proj):
        v_norm = -v_norm
        pair_pos_proj = -pair_pos_proj
        pair_neg_proj = -pair_neg_proj

    use_tokens = (
        token_X_pos is not None and token_X_neg is not None and len(token_X_pos) > 0 and len(token_X_neg) > 0
    )
    if use_tokens:
        tok_pos_proj = np.dot(token_X_pos, v_norm).astype(np.float64)
        tok_neg_proj = np.dot(token_X_neg, v_norm).astype(np.float64)
        if np.mean(tok_pos_proj) >= np.mean(tok_neg_proj):
            pos_proj, neg_proj = tok_pos_proj, tok_neg_proj
        else:
            pos_proj, neg_proj = pair_pos_proj, pair_neg_proj
    else:
        pos_proj, neg_proj = pair_pos_proj, pair_neg_proj

    mu_pos = float(np.mean(pos_proj))
    sigma_pos = max(1e-4, float(np.std(pos_proj)))
    mu_neg = float(np.mean(neg_proj))
    sigma_neg = max(1e-4, float(np.std(neg_proj)))

    X_all = np.concatenate([pos_proj, neg_proj]).reshape(-1, 1)
    y_all = np.array([1] * len(pos_proj) + [0] * len(neg_proj), dtype=int)

    midpoint = 0.5 * (mu_pos + mu_neg)
    if np.std(X_all) < 1e-6:
        m_l = float(midpoint)
    else:
        clf = LogisticRegression(fit_intercept=True, random_state=13)
        clf.fit(X_all, y_all)
        w = float(clf.coef_[0][0])
        b = float(clf.intercept_[0])
        if abs(w) < 1e-6:
            m_l = float(midpoint)
        else:
            cand = -b / w
            lo = min(mu_neg, mu_pos) - 3.0 * max(sigma_neg, sigma_pos)
            hi = max(mu_neg, mu_pos) + 3.0 * max(sigma_neg, sigma_pos)
            m_l = float(cand if lo <= cand <= hi else midpoint)

    # Evaluate linear separability on held-out pairs when provided (avoids in-sample AUROC inflation).
    use_eval = (
        eval_X_pos is not None and eval_X_neg is not None and len(eval_X_pos) > 0 and len(eval_X_neg) > 0
    )
    if use_eval:
        ev_pos = np.dot(eval_X_pos, v_norm).astype(np.float64)
        ev_neg = np.dot(eval_X_neg, v_norm).astype(np.float64)
        auroc_scores = np.concatenate([ev_pos, ev_neg])
        auroc_y = np.array([1] * len(ev_pos) + [0] * len(ev_neg), dtype=int)
    else:
        auroc_scores = np.concatenate([pair_pos_proj, pair_neg_proj])
        auroc_y = np.array([1] * len(pair_pos_proj) + [0] * len(pair_neg_proj), dtype=int)

    if np.std(auroc_scores) < 1e-6 or len(np.unique(auroc_y)) < 2:
        auroc = 0.5
    else:
        try:
            auroc = float(roc_auc_score(auroc_y, auroc_scores))
        except Exception:
            auroc = 0.5

    s_l = float(max(mu_pos + target_offset_sigma * sigma_pos, m_l + 0.1 * sigma_pos))

    return ConceptDirection(
        concept=concept,
        layer_idx=layer_idx,
        vector=torch.tensor(v_norm, dtype=torch.float32),
        m_l=m_l,
        s_l=s_l,
        mu_pos=mu_pos,
        sigma_pos=sigma_pos,
        mu_neg=mu_neg,
        sigma_neg=sigma_neg,
        separability_auroc=auroc,
    )


def _encode_prompt_completion(tokenizer, prompt: str, completion: str) -> tuple[list[int], int]:
    """Return (input_ids, n_prompt_tokens) with exact prompt/completion boundary alignment.

    The prompt (chat-formatted, incl. generation prefix) and the completion are tokenized separately and
    concatenated at the token-ID level, so the completion slice starts exactly at ``n_prompt_tokens`` and no
    artificial separator is inserted between the generation prefix and the completion.
    """
    prefix_str = format_chat(tokenizer, [{"role": "user", "content": prompt}])
    prefix_ids = list(tokenizer(prefix_str, add_special_tokens=False)["input_ids"])
    comp_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
    return prefix_ids + comp_ids, len(prefix_ids)


def _pooled_and_tokens_single_pass(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    layer_indices: list[int],
    pooling: str = "mean_completion",
    collect_tokens: bool = False,
    max_tokens_per_item: int = 32,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Run one forward pass over (prompt, completion) and return pooled and optional token activations."""
    layers = get_decoder_layers(model)
    cache: dict[int, torch.Tensor] = {}
    handles = []
    for l_idx in layer_indices:

        def _make_hook(idx: int):
            def _hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cache[idx] = h.detach()

            return _hook

        handles.append(layers[l_idx].register_forward_hook(_make_hook(l_idx)))

    try:
        ids, n_prompt = _encode_prompt_completion(tokenizer, prompt, completion)
        dev = next(model.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    n_tok = len(ids)
    start = min(n_prompt, max(0, n_tok - 1))
    pooled_out: dict[int, np.ndarray] = {}
    tokens_out: dict[int, np.ndarray] = {}
    for l_idx in layer_indices:
        h = cache[l_idx][0]  # [seq_len, hidden_dim]
        if pooling == "last_prompt_token":
            vec = h[min(max(0, n_prompt - 1), h.shape[0] - 1)]
        elif pooling == "last_completion_token":
            vec = h[-1]
        elif pooling == "mean_completion":
            vec = h[start:].mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling mode: {pooling!r}")
        pooled_out[l_idx] = vec.cpu().float().numpy()

        if collect_tokens:
            comp_h = h[start:] if start < h.shape[0] else h[-1:]
            if comp_h.shape[0] > max_tokens_per_item:
                sel = torch.linspace(0, comp_h.shape[0] - 1, steps=max_tokens_per_item, dtype=torch.long)
                comp_h = comp_h[sel]
            tokens_out[l_idx] = comp_h.cpu().float().numpy()

    return pooled_out, tokens_out, n_tok


@torch.no_grad()
def pooled_activations(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    layer_indices: list[int],
    pooling: str = "mean_completion",
) -> tuple[dict[int, np.ndarray], int]:
    """Run one forward pass over (prompt, completion) and return pooled activations per layer + token count."""
    pooled_out, _, n_tok = _pooled_and_tokens_single_pass(
        model, tokenizer, prompt, completion, layer_indices, pooling=pooling, collect_tokens=False
    )
    return pooled_out, n_tok


def _capture_pair_activations_impl(
    model,
    tokenizer,
    pairs: list[ContrastivePair],
    layer_indices: list[int],
    pooling: str = "mean_completion",
    collect_tokens: bool = False,
    max_tokens_per_item: int = 32,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]], int]:
    """Capture pooled (X_pos, X_neg) and optional completion token-level activations per layer."""
    if not pairs:
        raise ValueError("capture_pair_activations requires at least one contrastive pair")
    pos_by_layer: dict[int, list[np.ndarray]] = {l_idx: [] for l_idx in layer_indices}
    neg_by_layer: dict[int, list[np.ndarray]] = {l_idx: [] for l_idx in layer_indices}
    tok_pos_by_layer: dict[int, list[np.ndarray]] = {l_idx: [] for l_idx in layer_indices}
    tok_neg_by_layer: dict[int, list[np.ndarray]] = {l_idx: [] for l_idx in layer_indices}
    total_tokens = 0

    for p in pairs:
        for completion, p_dest, t_dest in (
            (p.positive, pos_by_layer, tok_pos_by_layer),
            (p.negative, neg_by_layer, tok_neg_by_layer),
        ):
            acts, tok_acts, n_tok = _pooled_and_tokens_single_pass(
                model,
                tokenizer,
                p.prompt,
                completion,
                layer_indices,
                pooling=pooling,
                collect_tokens=collect_tokens,
                max_tokens_per_item=max_tokens_per_item,
            )
            total_tokens += n_tok
            for l_idx, vec in acts.items():
                p_dest[l_idx].append(vec)
                if collect_tokens:
                    t_dest[l_idx].append(tok_acts[l_idx])

    pooled_out = {
        l_idx: (np.vstack(pos_by_layer[l_idx]), np.vstack(neg_by_layer[l_idx])) for l_idx in layer_indices
    }
    tok_out = (
        {
            l_idx: (np.vstack(tok_pos_by_layer[l_idx]), np.vstack(tok_neg_by_layer[l_idx]))
            for l_idx in layer_indices
        }
        if collect_tokens
        else {}
    )
    return pooled_out, tok_out, total_tokens


@torch.no_grad()
def capture_pair_and_token_activations(
    model,
    tokenizer,
    pairs: list[ContrastivePair],
    layer_indices: list[int],
    pooling: str = "mean_completion",
    max_tokens_per_item: int = 32,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]], int]:
    """Capture pooled (X_pos, X_neg) and completion token-level (T_pos, T_neg) activations per layer."""
    return _capture_pair_activations_impl(
        model,
        tokenizer,
        pairs,
        layer_indices,
        pooling=pooling,
        collect_tokens=True,
        max_tokens_per_item=max_tokens_per_item,
    )


@torch.no_grad()
def capture_pair_activations(
    model,
    tokenizer,
    pairs: list[ContrastivePair],
    layer_indices: list[int],
    pooling: str = "mean_completion",
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], int]:
    """Capture (X_pos, X_neg) activations for all requested layers in one pass per completion."""
    pooled_out, _, total_tokens = _capture_pair_activations_impl(
        model, tokenizer, pairs, layer_indices, pooling=pooling, collect_tokens=False
    )
    return pooled_out, total_tokens


def extract_steering_directions(
    model,
    tokenizer,
    pairs: list[ContrastivePair],
    concepts: list[str],
    layer_indices: list[int] | None = None,
    estimator: str = "pca",
    orthogonal_concept: str | None = "slang_benign",
    pooling: str = "mean_completion",
    target_offset_sigma: float = 0.5,
    n_calib: int = 100,
) -> tuple[list[ConceptDirection], SetupReport]:
    """Extract calibrated steering directions per concept at the best candidate layer."""
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    if not layer_indices:
        mid = max(0, min(n_layers - 1, n_layers // 2))
        layer_indices = [mid]

    report = SetupReport(method=f"steering_{estimator}", setup_key=f"steer_{estimator}_{n_calib}")
    with report.stage("extract_vectors", examples=0) as st:
        nuisance_by_layer: dict[int, np.ndarray] = {}
        if estimator == "ortho" and orthogonal_concept:
            slang_pairs = [p for p in pairs if p.concept == orthogonal_concept][:n_calib]
            if not slang_pairs:
                raise ValueError(
                    f"estimator='ortho' needs calibration pairs for {orthogonal_concept!r}; none found. "
                    "Refusing to silently degrade to non-orthogonalized steering."
                )
            if len(slang_pairs) < n_calib:
                log.warning(
                    "Requested n_calib=%d for orthogonal concept %r, but only %d pairs available.",
                    n_calib,
                    orthogonal_concept,
                    len(slang_pairs),
                )
            slang_acts, tok_s = capture_pair_activations(
                model, tokenizer, slang_pairs, layer_indices=layer_indices, pooling=pooling
            )
            st.tokens += tok_s
            for l_idx, (xp, xn) in slang_acts.items():
                nuisance_by_layer[l_idx] = _compute_raw_direction(xp, xn, estimator="pca")

        chosen_directions: list[ConceptDirection] = []
        per_concept_counts: dict[str, int] = {}
        for concept in concepts:
            c_pairs = [p for p in pairs if p.concept == concept][:n_calib]
            if not c_pairs:
                available = sorted({p.concept for p in pairs})
                raise ValueError(
                    f"No calibration pairs for steering concept {concept!r} (available: {available}). "
                    "Refusing to fall back to other concepts' pairs, which would mislabel the direction."
                )
            per_concept_counts[concept] = len(c_pairs)
            if len(c_pairs) < n_calib:
                log.warning(
                    "Requested n_calib=%d for concept %r, but only %d pairs available.",
                    n_calib,
                    concept,
                    len(c_pairs),
                )
            if pooling == "mean_completion":
                acts, token_acts, tok_c = capture_pair_and_token_activations(
                    model, tokenizer, c_pairs, layer_indices=layer_indices, pooling=pooling
                )
            else:
                acts, tok_c = capture_pair_activations(
                    model, tokenizer, c_pairs, layer_indices=layer_indices, pooling=pooling
                )
                token_acts = {}
            st.tokens += tok_c

            best_cd: ConceptDirection | None = None
            for l_idx in layer_indices:
                xp, xn = acts[l_idx]
                tok_xp, tok_xn = token_acts.get(l_idx, (None, None))
                cd = calibrate_hyperplane_from_activations(
                    xp,
                    xn,
                    concept=concept,
                    layer_idx=l_idx,
                    estimator=estimator,
                    target_offset_sigma=target_offset_sigma,
                    nuisance_vector=nuisance_by_layer.get(l_idx),
                    token_X_pos=tok_xp,
                    token_X_neg=tok_xn,
                )
                if best_cd is None or cd.separability_auroc >= best_cd.separability_auroc:
                    best_cd = cd
            if best_cd is not None:
                chosen_directions.append(best_cd)

        n_calib_effective = min(per_concept_counts.values()) if per_concept_counts else 0
        st.examples = int(n_calib_effective)
        st.extra.update(
            {
                "n_calib_requested": int(n_calib),
                "n_calib_effective": int(n_calib_effective),
                "per_concept_counts": per_concept_counts,
            }
        )

    return chosen_directions, report


def steering_setup_key(
    model_alias: str,
    quant: str,
    estimator: str,
    pooling: str,
    layer_indices: list[int],
    n_calib: int,
    concepts: list[str],
    orthogonal_concept: str | None = None,
) -> str:
    """Cache key covering every input that changes the extracted directions.

    ``target_offset_sigma`` is deliberately excluded: ``s_l`` is recomputed on load via
    :func:`apply_target_offset`. The nuisance concept only matters for the ``ortho`` estimator.
    """
    c_hash = stable_hash(sorted(concepts), length=6)
    l_tag = "-".join(str(x) for x in layer_indices)
    key = f"steer_{model_alias}_{quant}_{estimator}_{pooling}_L{l_tag}_N{n_calib}_{c_hash}"
    if estimator == "ortho" and orthogonal_concept:
        key += f"_O{orthogonal_concept}"
    return key


def apply_target_offset(directions: list[ConceptDirection], target_offset_sigma: float) -> None:
    """Recompute each direction's StTP target ``s_l`` for ``target_offset_sigma`` (in place).

    Cached artifacts store ``s_l`` for the offset used at extraction time; callers that reuse them with a
    different (e.g. swept) offset must call this so the hook does not steer toward a stale target.
    """
    for cd in directions:
        cd.s_l = float(max(cd.mu_pos + target_offset_sigma * cd.sigma_pos, cd.m_l + 0.1 * cd.sigma_pos))


def save_steering_artifact(
    artifact_dir: Path,
    setup_key: str,
    directions: list[ConceptDirection],
    report: SetupReport,
) -> Path:
    out_dir = Path(artifact_dir) / setup_key
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_items = []
    for cd in directions:
        vec_file = f"{cd.concept}_layer_{cd.layer_idx}.pt"
        torch.save(cd.vector.detach().cpu(), out_dir / vec_file)
        meta_items.append(
            {
                "concept": cd.concept,
                "layer_idx": int(cd.layer_idx),
                "vector_file": vec_file,
                "m_l": float(cd.m_l),
                "s_l": float(cd.s_l),
                "mu_pos": float(cd.mu_pos),
                "sigma_pos": float(cd.sigma_pos),
                "mu_neg": float(cd.mu_neg),
                "sigma_neg": float(cd.sigma_neg),
                "separability_auroc": float(cd.separability_auroc),
                "extra": dict(cd.extra),
            }
        )
    write_json(out_dir / "steering_meta.json", {"directions": meta_items, "setup": report.to_dict()})
    return out_dir


def load_steering_artifact(
    artifact_dir: Path, setup_key: str
) -> tuple[list[ConceptDirection], SetupReport] | None:
    meta_path = Path(artifact_dir) / setup_key / "steering_meta.json"
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    out_dir = meta_path.parent
    dirs: list[ConceptDirection] = []
    for item in meta.get("directions", []):
        vec = torch.load(out_dir / item["vector_file"], map_location="cpu", weights_only=True)
        dirs.append(
            ConceptDirection(
                concept=str(item["concept"]),
                layer_idx=int(item["layer_idx"]),
                vector=vec,
                m_l=float(item["m_l"]),
                s_l=float(item["s_l"]),
                mu_pos=float(item.get("mu_pos", 0.0)),
                sigma_pos=float(item.get("sigma_pos", 1.0)),
                mu_neg=float(item.get("mu_neg", 0.0)),
                sigma_neg=float(item.get("sigma_neg", 1.0)),
                separability_auroc=float(item.get("separability_auroc", 0.5)),
                extra=dict(item.get("extra", {})),
            )
        )
    rep = SetupReport.from_dict(meta["setup"])
    rep.cached = True
    return dirs, rep


def extract_or_load_directions(
    *,
    model,
    tokenizer,
    model_alias: str,
    quant: str,
    pairs: list[ContrastivePair],
    concepts: list[str],
    layer_indices: list[int],
    artifact_dir: Path | None = None,
    estimator: str = "pca",
    orthogonal_concept: str | None = "slang_benign",
    pooling: str = "mean_completion",
    target_offset_sigma: float = 0.5,
    n_calib: int = 100,
) -> tuple[list[ConceptDirection], SetupReport]:
    """Load cached steering directions from ``artifact_dir`` when available, else extract and cache."""
    key = steering_setup_key(
        model_alias, quant, estimator, pooling, layer_indices, n_calib, concepts, orthogonal_concept
    )
    if artifact_dir is not None:
        cached = load_steering_artifact(artifact_dir, key)
        if cached is not None:
            directions, rep = cached
            apply_target_offset(directions, target_offset_sigma)
            return directions, rep

    directions, rep = extract_steering_directions(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        concepts=concepts,
        layer_indices=layer_indices,
        estimator=estimator,
        orthogonal_concept=orthogonal_concept,
        pooling=pooling,
        target_offset_sigma=target_offset_sigma,
        n_calib=n_calib,
    )
    rep.setup_key = key
    if artifact_dir is not None:
        save_steering_artifact(artifact_dir, key, directions, rep)
    return directions, rep
