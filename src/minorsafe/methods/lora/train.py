"""PEFT LoRA / QLoRA SFT trainer on safe calibration completions with prompt-loss masking (Phase 5.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset

from minorsafe.config import ModelSpec
from minorsafe.data.schema import ContrastivePair
from minorsafe.generation import format_chat
from minorsafe.telemetry.cost import SetupReport
from minorsafe.utils import get_logger, read_json, seed_everything, write_json

log = get_logger(__name__)

_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class _MaskedSFTDataset(Dataset):
    def __init__(self, items: list[tuple[list[int], list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.items[idx]


def _encode_sft_pairs(
    tokenizer,
    gen_spec: ModelSpec,
    pairs: list[ContrastivePair],
    max_seq_len: int = 1024,
) -> list[tuple[list[int], list[int]]]:
    """Encode (prompt -> positive) with prompt tokens masked (-100) and NO system prompt (Decision 4)."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    items: list[tuple[list[int], list[int]]] = []
    for p in pairs:
        prefix_str = format_chat(tokenizer, [{"role": "user", "content": p.prompt}], spec=gen_spec)
        prefix_ids = list(tokenizer(prefix_str, add_special_tokens=False)["input_ids"])
        comp_ids = list(tokenizer(p.positive, add_special_tokens=False)["input_ids"])
        if eos_id is not None and (not comp_ids or comp_ids[-1] != eos_id):
            comp_ids = [*comp_ids, int(eos_id)]
        if not comp_ids:
            continue

        full_ids = prefix_ids + comp_ids
        labels = [-100] * len(prefix_ids) + comp_ids
        if len(full_ids) > max_seq_len:
            overflow = len(full_ids) - max_seq_len
            full_ids = full_ids[overflow:]
            labels = labels[overflow:]
        if all(lbl == -100 for lbl in labels) or len(full_ids) < 2:
            continue
        items.append((full_ids, labels))
    return items


def _collate_sft(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x[0]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for ids, lbls in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(lbls + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _resolve_target_modules(model) -> list[str] | str:
    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    found = [m for m in _DEFAULT_TARGET_MODULES if m in leaf_names]
    return found if found else "all-linear"


def train_lora_adapter(
    model,
    tokenizer,
    gen_spec: ModelSpec,
    calib_pairs: list[ContrastivePair],
    artifact_dir: Path,
    *,
    concepts: list[str] | None = None,
    n_calib: int = 500,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lr: float = 2.0e-4,
    epochs: int = 2,
    batch_size: int = 4,
    quant: str = "bf16",
    seed: int = 13,
    method_name: str = "m2_lora",
) -> tuple[Any, SetupReport]:
    """Train (or load cached) PEFT LoRA/QLoRA adapter on positive safe contrastive completions."""
    seed_everything(seed)
    concept_set = set(concepts) if concepts else None
    filtered = [p for p in calib_pairs if concept_set is None or p.concept in concept_set][:n_calib]
    if not filtered:
        raise ValueError(f"No calibration pairs available for LoRA training (concepts={concepts}).")
    if len(filtered) < n_calib:
        log.warning(
            "LoRA %s requested n_calib=%d pairs, but only %d available.",
            gen_spec.alias,
            n_calib,
            len(filtered),
        )

    setup_key = f"lora_{gen_spec.alias}_{quant}_N{n_calib}_r{r}_ep{epochs}"
    adapter_dir = Path(artifact_dir) / setup_key
    meta_path = adapter_dir / "lora_meta.json"

    if meta_path.exists() and (adapter_dir / "adapter_config.json").exists():
        meta = read_json(meta_path)
        rep = SetupReport.from_dict(meta["setup"])
        rep.method = method_name
        rep.cached = True
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
        peft_model.eval()
        log.info("Reusing cached PEFT LoRA adapter from %s", adapter_dir)
        return peft_model, rep

    encoded = _encode_sft_pairs(tokenizer, gen_spec, filtered)
    if not encoded:
        raise ValueError("All LoRA training pairs were empty after encoding.")

    rep = SetupReport(method=method_name, setup_key=setup_key)
    with rep.stage("train_lora_adapter", examples=len(encoded) * max(1, epochs)) as st:
        st.extra.update({"n_calib_requested": int(n_calib), "n_calib_effective": len(filtered)})
        if quant in ("int8", "nf4"):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_resolve_target_modules(model),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_cfg)
        dev = next(peft_model.parameters()).device

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ds = _MaskedSFTDataset(encoded)
        loader = DataLoader(
            ds,
            batch_size=max(1, batch_size),
            shuffle=True,
            collate_fn=lambda b: _collate_sft(b, pad_id=pad_id),
        )

        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        total_steps = max(1, len(loader) * max(1, epochs))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )

        peft_model.train()
        for _ in range(max(1, epochs)):
            for batch in loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                labels = batch["labels"].to(dev)
                st.tokens += int(attention_mask.sum().item())

                opt.zero_grad(set_to_none=True)
                out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                scheduler.step()

        peft_model.eval()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    write_json(
        meta_path,
        {
            "model_alias": gen_spec.alias,
            "quant": quant,
            "r": r,
            "lora_alpha": lora_alpha,
            "setup": rep.to_dict(),
        },
    )
    return peft_model, rep
