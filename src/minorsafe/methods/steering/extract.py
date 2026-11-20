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
from minorsafe.utils import read_json, stable_hash, write_json


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
) -> ConceptDirection:
    """Compute unit steering vector and calibrate 1D logistic regression threshold m_l and target s_l."""
    v_norm = _compute_raw_direction(X_pos, X_neg, estimator="pca" if estimator in ("pca", "ortho") else "caa")
    if nuisance_vector is not None:
        v_norm = orthogonalize_vector(v_norm, nuisance_vector)

    pos_proj = np.dot(X_pos, v_norm).astype(np.float64)
    neg_proj = np.dot(X_neg, v_norm).astype(np.float64)
    if np.mean(pos_proj) < np.mean(neg_proj):
        v_norm = -v_norm
        pos_proj = -pos_proj
        neg_proj = -neg_proj

    mu_pos = float(np.mean(pos_proj))
    sigma_pos = max(1e-4, float(np.std(pos_proj)))
    mu_neg = float(np.mean(neg_proj))
    sigma_neg = max(1e-4, float(np.std(neg_proj)))

    X_all = np.concatenate([pos_proj, neg_proj]).reshape(-1, 1)
    y_all = np.array([1] * len(pos_proj) + [0] * len(neg_proj), dtype=int)

    midpoint = 0.5 * (mu_pos + mu_neg)
    if np.std(X_all) < 1e-6:
        m_l = float(midpoint)
        auroc = 0.5
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
        try:
            auroc = float(roc_auc_score(y_all, X_all.ravel()))
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
    out: dict[int, np.ndarray] = {}
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
        out[l_idx] = vec.cpu().float().numpy()
    return out, n_tok


@torch.no_grad()
def capture_pair_activations(
    model,
    tokenizer,
    pairs: list[ContrastivePair],
    layer_indices: list[int],
    pooling: str = "mean_completion",
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], int]:
    """Capture (X_pos, X_neg) activations for all requested layers in one pass per completion."""
    if not pairs:
        raise ValueError("capture_pair_activations requires at least one contrastive pair")
    pos_by_layer: dict[int, list[np.ndarray]] = {l_idx: [] for l_idx in layer_indices}
    neg_by_layer: dict[int, list[np.ndarray]] = {l_idx: [] for l_idx in layer_indices}
    total_tokens = 0

    for p in pairs:
        for completion, dest in ((p.positive, pos_by_layer), (p.negative, neg_by_layer)):
            acts, n_tok = pooled_activations(model, tokenizer, p.prompt, completion, layer_indices, pooling)
            total_tokens += n_tok
            for l_idx, vec in acts.items():
                dest[l_idx].append(vec)

    out = {l_idx: (np.vstack(pos_by_layer[l_idx]), np.vstack(neg_by_layer[l_idx])) for l_idx in layer_indices}
    return out, total_tokens


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
    with report.stage("extract_vectors", examples=len(pairs)) as st:
        nuisance_by_layer: dict[int, np.ndarray] = {}
        if estimator == "ortho" and orthogonal_concept:
            slang_pairs = [p for p in pairs if p.concept == orthogonal_concept][:n_calib]
            if not slang_pairs:
                raise ValueError(
                    f"estimator='ortho' needs calibration pairs for {orthogonal_concept!r}; none found. "
                    "Refusing to silently degrade to non-orthogonalized steering."
                )
            slang_acts, tok_s = capture_pair_activations(
                model, tokenizer, slang_pairs, layer_indices=layer_indices, pooling=pooling
            )
            st.tokens += tok_s
            for l_idx, (xp, xn) in slang_acts.items():
                nuisance_by_layer[l_idx] = _compute_raw_direction(xp, xn, estimator="pca")

        chosen_directions: list[ConceptDirection] = []
        for concept in concepts:
            c_pairs = [p for p in pairs if p.concept == concept][:n_calib]
            if not c_pairs:
                available = sorted({p.concept for p in pairs})
                raise ValueError(
                    f"No calibration pairs for steering concept {concept!r} (available: {available}). "
                    "Refusing to fall back to other concepts' pairs, which would mislabel the direction."
                )
            acts, tok_c = capture_pair_activations(
                model, tokenizer, c_pairs, layer_indices=layer_indices, pooling=pooling
            )
            st.tokens += tok_c

            best_cd: ConceptDirection | None = None
            for l_idx in layer_indices:
                xp, xn = acts[l_idx]
                cd = calibrate_hyperplane_from_activations(
                    xp,
                    xn,
                    concept=concept,
                    layer_idx=l_idx,
                    estimator=estimator,
                    target_offset_sigma=target_offset_sigma,
                    nuisance_vector=nuisance_by_layer.get(l_idx),
                )
                if best_cd is None or cd.separability_auroc >= best_cd.separability_auroc:
                    best_cd = cd
            if best_cd is not None:
                chosen_directions.append(best_cd)

    return chosen_directions, report


def steering_setup_key(
    model_alias: str,
    quant: str,
    estimator: str,
    pooling: str,
    layer_indices: list[int],
    n_calib: int,
    concepts: list[str],
) -> str:
    c_hash = stable_hash(sorted(concepts), length=6)
    l_tag = "-".join(str(x) for x in layer_indices)
    return f"steer_{model_alias}_{quant}_{estimator}_{pooling}_L{l_tag}_N{n_calib}_{c_hash}"


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
    key = steering_setup_key(model_alias, quant, estimator, pooling, layer_indices, n_calib, concepts)
    if artifact_dir is not None:
        cached = load_steering_artifact(artifact_dir, key)
        if cached is not None:
            return cached

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
