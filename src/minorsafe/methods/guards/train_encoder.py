"""Supervised fine-tuning for Bidirectional Encoder safety guards + dev F1 threshold calibration (Phase 5.2)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from minorsafe.config import ModelSpec
from minorsafe.data.schema import ContrastivePair, Example
from minorsafe.eval.detection import calibrate_f1_threshold
from minorsafe.registry import free_model, load_encoder_classifier
from minorsafe.telemetry.cost import SetupReport
from minorsafe.utils import get_logger, read_json, seed_everything, stable_hash, write_json

log = get_logger(__name__)

_SAFETY_PAIR_CONCEPTS = {"profanity", "mature_theme", "slang_toxic"}


def _build_encoder_training_data(
    calib_pairs: list[ContrastivePair],
    calib_examples: list[Example],
    n_calib: int,
) -> tuple[list[str], list[int]]:
    """Build balanced (text, is_unsafe) examples from safety pairs and calibration prompts."""
    texts: list[str] = []
    labels: list[int] = []

    safety_pairs = [p for p in calib_pairs if p.concept in _SAFETY_PAIR_CONCEPTS]
    for p in safety_pairs[:n_calib]:
        texts.append(p.positive)
        labels.append(0)
        texts.append(p.negative)
        labels.append(1)

    for ex in calib_examples[:n_calib]:
        if ex.label_unsafe is None:
            continue
        texts.append(ex.last_user)
        labels.append(int(bool(ex.label_unsafe)))

    return texts, labels


@torch.no_grad()
def _score_classifier_batch(
    model, tokenizer, texts: list[str], device: str, batch_size: int = 32
) -> list[float]:
    probs: list[float] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        chunk = [t or "" for t in texts[i : i + batch_size]]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        logits = model(**enc).logits.float()
        if logits.shape[-1] == 1:
            p_unsafe = torch.sigmoid(logits[:, 0])
        elif logits.shape[-1] == 2:
            p_unsafe = torch.softmax(logits, dim=-1)[:, 1]
        else:
            p_unsafe = torch.sigmoid(logits).max(dim=-1).values
        probs.extend(p_unsafe.cpu().tolist())
    return [float(x) for x in probs]


def encoder_setup_key(
    model_alias: str, *, quant: str, n_calib: int, epochs: int, lr: float, batch_size: int, seed: int
) -> str:
    """Guard checkpoint cache key; ``quant`` sets the training dtype, batch size and seed change the head."""
    cfg_hash = stable_hash({"batch_size": batch_size, "seed": seed}, length=6)
    return f"enc_{model_alias}_{quant}_N{n_calib}_ep{epochs}_lr{lr:g}_{cfg_hash}"


def train_encoder_guard(
    model_spec: ModelSpec,
    calib_pairs: list[ContrastivePair],
    calib_examples: list[Example],
    dev_examples: list[Example],
    artifact_dir: Path,
    n_calib: int = 200,
    epochs: int = 2,
    lr: float = 3.0e-5,
    batch_size: int = 16,
    device: str = "cuda",
    quant: str = "bf16",
    seed: int = 13,
) -> tuple[Path, float, SetupReport]:
    """Fine-tune an encoder sequence classifier and calibrate its dev Macro-F1 threshold."""
    seed_everything(seed)
    setup_key = encoder_setup_key(
        model_spec.alias, quant=quant, n_calib=n_calib, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed
    )
    ckpt_dir = Path(artifact_dir) / setup_key
    meta_path = ckpt_dir / "guard_meta.json"

    if meta_path.exists():
        meta = read_json(meta_path)
        rep = SetupReport.from_dict(meta["setup"])
        rep.cached = True
        log.info("Reusing cached fine-tuned encoder guard from %s", ckpt_dir)
        return ckpt_dir, float(meta["threshold"]), rep

    texts, labels = _build_encoder_training_data(calib_pairs, calib_examples, n_calib=n_calib)
    if len(set(labels)) < 2:
        raise ValueError(
            f"Encoder guard training for {model_spec.alias} requires both safe and unsafe examples; "
            f"got labels={set(labels)}."
        )
    safety_pairs = [p for p in calib_pairs if p.concept in _SAFETY_PAIR_CONCEPTS]
    labeled_calib = [ex for ex in calib_examples if ex.label_unsafe is not None]
    n_pairs_used = min(len(safety_pairs), n_calib)
    n_ex_used = min(len(labeled_calib), n_calib)
    n_calib_effective = max(n_pairs_used, n_ex_used)
    if n_calib_effective < n_calib:
        log.warning(
            "Encoder guard %s requested n_calib=%d, effective available=%d (pairs=%d, examples=%d).",
            model_spec.alias,
            n_calib,
            n_calib_effective,
            n_pairs_used,
            n_ex_used,
        )

    rep = SetupReport(method=f"m1_enc_{model_spec.alias}", setup_key=setup_key)
    model, tok, _ = load_encoder_classifier(model_spec, num_labels=2, device=device, quant=quant)
    model.config.id2label = {0: "safe", 1: "unsafe"}
    model.config.label2id = {"safe": 0, "unsafe": 1}

    threshold = 0.50
    try:
        with rep.stage("train_encoder_head", examples=len(texts) * max(1, epochs)) as st:
            st.extra.update(
                {
                    "n_calib_requested": int(n_calib),
                    "n_calib_effective": int(n_calib_effective),
                    "n_pairs_used": int(n_pairs_used),
                    "n_examples_used": int(n_ex_used),
                }
            )
            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            dataset = TensorDataset(
                enc["input_ids"],
                enc["attention_mask"],
                torch.tensor(labels, dtype=torch.long),
            )
            loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=True)
            tokens_per_epoch = int(enc["attention_mask"].sum().item())

            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
            total_steps = max(1, len(loader) * max(1, epochs))
            scheduler = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1.0, end_factor=0.1, total_iters=total_steps
            )

            model.train()
            for _ in range(max(1, epochs)):
                st.tokens += tokens_per_epoch
                for input_ids, attention_mask, y_batch in loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    y_batch = y_batch.to(device)

                    opt.zero_grad(set_to_none=True)
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=y_batch)
                    loss = out.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    scheduler.step()

            model.eval()
            dev_labeled = [e for e in dev_examples if e.label_unsafe is not None]
            if len({int(bool(e.label_unsafe)) for e in dev_labeled}) >= 2:
                dev_texts = [e.last_user for e in dev_labeled]
                dev_labels = [int(bool(e.label_unsafe)) for e in dev_labeled]
                dev_scores = _score_classifier_batch(model, tok, dev_texts, device=device)
                threshold = calibrate_f1_threshold(dev_labels, dev_scores)
                st.tokens += sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in dev_texts)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
        write_json(
            meta_path,
            {
                "model_alias": model_spec.alias,
                "threshold": float(threshold),
                "setup": rep.to_dict(),
            },
        )
    finally:
        free_model(model, tok)

    return ckpt_dir, float(threshold), rep
