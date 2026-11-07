"""Model loading with uniform dtype / quantization / device handling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from minorsafe.config import ModelSpec
from minorsafe.utils import get_logger

log = get_logger(__name__)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class LoadReport:
    alias: str
    hf_id: str
    quant: str
    load_seconds: float
    vram_mb: float
    n_params: int
    device: str
    extra: dict = field(default_factory=dict)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_dtype(quant: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if quant in _DTYPES:
        if quant == "bf16" and not torch.cuda.is_bf16_supported():
            return torch.float16
        return _DTYPES[quant]
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _quant_kwargs(quant: str, device: str) -> dict:
    """Extra ``from_pretrained`` kwargs for a quantization mode."""
    if device == "cpu" or quant in _DTYPES:
        return {}
    from transformers import BitsAndBytesConfig

    if quant == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    if quant == "nf4":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype("bf16", device),
            )
        }
    if quant == "fp8":
        try:
            from transformers import FineGrainedFP8Config

            return {"quantization_config": FineGrainedFP8Config()}
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("fp8 requires a transformers build with FineGrainedFP8Config") from exc
    # awq / gptq: the checkpoint itself carries the quantization config
    return {}


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 2**20
    return 0.0


def checkpoint_id(spec: ModelSpec, quant: str) -> str:
    if quant in ("awq", "gptq"):
        if quant not in spec.quant_ids:
            raise ValueError(f"{spec.alias}: no pre-quantized {quant} checkpoint configured (quant_ids)")
        return spec.quant_ids[quant]
    return spec.hf_id


def load_tokenizer(spec: ModelSpec, quant: str = "bf16", padding_side: str = "left"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint_id(spec, quant))
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    return tok


def load_decoder(spec: ModelSpec, quant: str = "bf16", device: str | None = None):
    """Load a causal LM. Returns (model, tokenizer, LoadReport)."""
    from transformers import AutoModelForCausalLM

    device = device or default_device()
    t0 = time.perf_counter()
    base_vram = _vram_mb()
    kwargs = dict(dtype=compute_dtype(quant, device), **_quant_kwargs(quant, device))
    if device == "cuda":
        kwargs["device_map"] = "auto" if quant in ("int8", "nf4") else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(checkpoint_id(spec, quant), **kwargs)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    tok = load_tokenizer(spec, quant)
    if model.generation_config is not None and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tok.pad_token_id
    rep = LoadReport(
        alias=spec.alias,
        hf_id=checkpoint_id(spec, quant),
        quant=quant,
        load_seconds=time.perf_counter() - t0,
        vram_mb=_vram_mb() - base_vram,
        n_params=sum(p.numel() for p in model.parameters()),
        device=device,
    )
    log.info("Loaded %s (%s) in %.1fs, %.0f MB", spec.alias, quant, rep.load_seconds, rep.vram_mb)
    return model, tok, rep


def load_encoder_classifier(
    spec_or_path: ModelSpec | str, num_labels: int = 2, device: str | None = None, quant: str = "bf16"
):
    """Load an encoder (or decoder) with a sequence-classification head."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or default_device()
    path = spec_or_path.hf_id if isinstance(spec_or_path, ModelSpec) else spec_or_path
    t0 = time.perf_counter()
    kwargs = dict(num_labels=num_labels, dtype=compute_dtype(quant, device))
    kwargs.update(_quant_kwargs(quant, device))
    try:
        model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=False, **kwargs)
    except ValueError:
        # e.g. EuroBERT ships custom modeling code
        model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True, **kwargs)
    if "quantization_config" not in kwargs:
        model.to(device)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
        model.config.pad_token_id = tok.pad_token_id
    model.eval()
    rep = LoadReport(
        alias=getattr(spec_or_path, "alias", str(path)),
        hf_id=str(path),
        quant=quant,
        load_seconds=time.perf_counter() - t0,
        vram_mb=_vram_mb(),
        n_params=sum(p.numel() for p in model.parameters()),
        device=device,
    )
    return model, tok, rep


def load_mlm(spec: ModelSpec, device: str | None = None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = device or default_device()
    model = AutoModelForMaskedLM.from_pretrained(spec.hf_id, dtype=compute_dtype("bf16", device))
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    return model, tok


# --------------------------------------------------------------------------- decoder layer access
_LAYER_PATHS = (
    "model.layers",  # Llama, Qwen, Gemma, Mistral, DeepSeek-distill
    "model.language_model.layers",  # Gemma-3 multimodal (transformers>=4.52)
    "language_model.model.layers",  # Gemma-3 multimodal (older)
    "model.decoder.layers",  # OPT
    "transformer.h",  # GPT-2 style
    "gpt_neox.layers",
)


def _getattr_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def get_decoder_layers(model, layers_path: str | None = None) -> torch.nn.ModuleList:
    """Return the ModuleList of transformer blocks, unwrapping PEFT wrappers."""
    root = model
    if hasattr(root, "get_base_model"):
        root = root.get_base_model()
    paths = (layers_path,) if layers_path else _LAYER_PATHS
    for p in paths:
        try:
            layers = _getattr_path(root, p)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise AttributeError(f"Could not locate decoder layers on {type(model).__name__}")


def free_model(*objs) -> None:
    import gc

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

