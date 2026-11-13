"""Pydantic configuration schemas and YAML loading.

Config layout (see ``configs/``):
    hardware/<profile>.yaml   -> HardwareProfile
    models.yaml               -> dict[alias, ModelSpec]
    age_bands.yaml            -> dict[name, AgeBand]
    prompts.yaml              -> free-form prompt templates
    methods/<name>.yaml       -> MethodSpec defaults
    experiments/<name>.yaml   -> ExperimentSpec
    data.yaml                 -> DataSpec
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from minorsafe.utils import CONFIG_DIR

Quant = Literal["bf16", "fp16", "fp32", "int8", "nf4", "awq", "gptq", "fp8"]
SizeGroup = Literal["S", "M", "L"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- models
class ModelSpec(StrictModel):
    alias: str = ""
    hf_id: str
    arch: Literal["decoder", "encoder"]
    role: Literal["generator", "decoder_guard", "encoder_guard", "encoder_mlm", "judge", "teacher", "aux"]
    family: str
    size_group: SizeGroup | None = None
    params_b: float | None = None
    gated: bool = False
    reasoning: bool = False  # emits <think>...</think> (DeepSeek-R1 distills)
    max_new_tokens: int | None = None  # per-model override
    chat_template: bool = True  # tokenizer ships a chat template
    supports_system: bool = True  # chat template accepts a system role
    guard_format: Literal["llama_guard", "qwen3guard", "shieldgemma", "generic"] | None = None
    layers_path: str | None = None  # dotted path to decoder layers; auto-detected when None
    quant_ids: dict[str, str] = {}  # pre-quantized checkpoints, e.g. {"awq": "Qwen/Qwen2.5-7B-Instruct-AWQ"}
    notes: str = ""


# --------------------------------------------------------------------------- hardware
class EvalSettings(StrictModel):
    toxicity_backend: str = "detoxify"  # detoxify | hf:<model_id> | lexicon
    bertscore_model: str | None = "roberta-large"  # None disables BERTScore
    bertscore_num_layers: int | None = None
    ppl_model: str | None = "gpt2-large"  # None disables perplexity
    judge_backend: Literal["vllm", "hf", "heuristic"] = "vllm"
    judge_model: str | None = None  # alias in models.yaml
    capability_items: int = 100  # tinyMMLU / tinyGSM8K items


class HardwareProfile(StrictModel):
    name: str
    description: str = ""
    device: Literal["cuda", "cpu"] = "cuda"
    min_vram_gb: float = 0.0
    num_gpus: int = 1
    allowed_quants: list[Quant] = ["bf16"]
    default_quant: Quant = "bf16"
    batch_size: int = 8
    max_new_tokens: int = 256
    generators: list[str] = []  # aliases
    decoder_guards: list[str] = []
    encoder_guards: list[str] = []
    encoder_mlms: list[str] = []
    teacher: str | None = None
    teacher_backend: Literal["vllm", "hf", "mock"] = "vllm"
    lora_large_mode: Literal["lora", "qlora"] = "lora"
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.85
    eval: EvalSettings = EvalSettings()


# --------------------------------------------------------------------------- age bands
class AgeBand(StrictModel):
    name: str = ""
    min_age: int
    max_age: int
    target_fk_grade: float
    description: str
    disallowed: list[str] = []
    allowed: list[str] = []


# --------------------------------------------------------------------------- data
class SourceSpec(StrictModel):
    name: str
    loader: str  # function name in minorsafe.data.sources
    hf_id: str | None = None
    config: str | None = None
    split: str | None = None
    max_items: int = 2000
    enabled: bool = True
    purpose: list[str] = []
    kwargs: dict[str, Any] = {}


class DataSpec(StrictModel):
    sources: list[SourceSpec] = []
    synth_pairs_per_concept: int = 600
    synth_oversample: float = 3.0
    concepts: list[str] = ["profanity", "lexical_complexity", "mature_theme", "slang_toxic", "slang_benign"]
    split_ratios: dict[str, float] = {"calib": 0.5, "dev": 0.2, "test": 0.3}
    pair_split_ratios: dict[str, float] = {"calib": 0.7, "dev": 0.15, "test": 0.15}
    obfuscation_per_sentence: int = 2
    max_minimal_pairs: int = 3000
    english_min_ratio: float = 0.6
    seed: int = 13


# --------------------------------------------------------------------------- methods / experiments
class MethodSpec(StrictModel):
    name: str
    kind: Literal["baseline", "guard", "lora", "steering"]
    params: dict[str, Any] = {}


class RunGroup(StrictModel):
    method: str  # file stem in configs/methods
    generators: list[str] = ["*"]  # aliases, size groups (S/M/L), families, or "*"
    quants: list[Quant] | None = None  # None -> profile default
    params: dict[str, Any] = {}  # overrides of method params
    sweep: dict[str, list[Any]] = {}  # cartesian product over param values
    label: str | None = None


class ExperimentSpec(StrictModel):
    name: str
    description: str = ""
    split: Literal["calib", "dev", "test"] = "dev"
    categories: list[str] | None = None  # filter Examples by category (None = all)
    max_examples: int | None = None
    max_examples_per_category: int | None = None
    seeds: list[int] = [0]
    age_bands: list[str] = ["teen_12_18"]
    batch_size: int | None = None  # None -> profile default
    max_new_tokens: int | None = None
    temperature: float = 0.0
    warmup_queries: int = 3
    runs: list[RunGroup] = Field(default_factory=list)
    evaluate: bool = True
    capability_eval: bool = False


# --------------------------------------------------------------------------- loading helpers
def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML) to a nested dict."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override into non-dict at {p!r}")
        node[parts[-1]] = value
    return cfg


def _resolve(path_or_name: str | Path, subdir: str) -> Path:
    p = Path(path_or_name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = CONFIG_DIR / subdir / f"{path_or_name}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Config {path_or_name!r} not found (looked in {cand})")


def load_models(path: str | Path | None = None) -> dict[str, ModelSpec]:
    raw = load_yaml(path or CONFIG_DIR / "models.yaml")
    out: dict[str, ModelSpec] = {}
    for alias, spec in raw.items():
        spec = dict(spec)
        spec.setdefault("alias", alias)
        out[alias] = ModelSpec(**spec)
    return out


def load_hardware(name: str, overrides: list[str] | None = None) -> HardwareProfile:
    raw = load_yaml(_resolve(name, "hardware"))
    return HardwareProfile(**apply_overrides(raw, overrides))


def load_age_bands(path: str | Path | None = None) -> dict[str, AgeBand]:
    raw = load_yaml(path or CONFIG_DIR / "age_bands.yaml")
    return {k: AgeBand(name=k, **v) for k, v in raw.items()}


def load_prompts(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "prompts.yaml")


def load_method(name: str) -> MethodSpec:
    raw = load_yaml(_resolve(name, "methods"))
    raw.setdefault("name", Path(name).stem)
    return MethodSpec(**raw)


def load_experiment(name: str, overrides: list[str] | None = None) -> ExperimentSpec:
    raw = load_yaml(_resolve(name, "experiments"))
    return ExperimentSpec(**apply_overrides(raw, overrides))


def load_data_spec(path: str | Path | None = None, overrides: list[str] | None = None) -> DataSpec:
    raw = load_yaml(path or CONFIG_DIR / "data.yaml")
    return DataSpec(**apply_overrides(raw, overrides))


def select_generators(
    selectors: list[str], profile: HardwareProfile, models: dict[str, ModelSpec]
) -> list[str]:
    """Resolve generator selectors (alias | size group | family | '*') against the profile's generators."""
    pool = [a for a in profile.generators if a in models]
    chosen: list[str] = []
    for sel in selectors:
        for alias in pool:
            m = models[alias]
            if sel == "*" or sel == alias or sel == m.size_group or sel == m.family:
                if alias not in chosen:
                    chosen.append(alias)
    return chosen
