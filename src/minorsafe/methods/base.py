"""Common method interface (ResponseMethod, RunContext, MethodOutput) and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minorsafe.config import AgeBand, HardwareProfile, MethodSpec, ModelSpec
from minorsafe.data.schema import ContrastivePair, Example
from minorsafe.generation import ChatGenerator
from minorsafe.registry import free_model, load_decoder
from minorsafe.telemetry.cost import SetupReport
from minorsafe.telemetry.tokens import TokenUsage


@dataclass
class RunContext:
    hardware: HardwareProfile
    generator_spec: ModelSpec
    models: dict[str, ModelSpec]
    age_bands: dict[str, AgeBand]
    prompts: dict[str, Any]
    calib_pairs: list[ContrastivePair] = field(default_factory=list)
    calib_examples: list[Example] = field(default_factory=list)
    dev_examples: list[Example] = field(default_factory=list)
    quant: str = "bf16"
    artifact_dir: Path = Path("results/artifacts")
    preloaded_generator: tuple[Any, Any] | None = None


@dataclass
class MethodOutput:
    example_id: str
    text: str
    raw_text: str = ""
    blocked: bool = False
    action_taken: str = "none"
    guard_score: float | None = None
    tokens: TokenUsage = field(default_factory=TokenUsage)
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    decode_tok_per_s: float = 0.0
    effective_tok_per_s: float = 0.0
    # NaN = not measured (GPUSampler not yet wired). Never report fake zeros.
    energy_j: float = float("nan")
    peak_vram_mb: float = float("nan")
    extra: dict = field(default_factory=dict)


class ResponseMethod(ABC):
    def __init__(self, spec: MethodSpec):
        self.spec = spec
        self.name = spec.name
        self.ctx: RunContext | None = None
        self.model = None
        self.tokenizer = None
        self.generator: ChatGenerator | None = None
        self._owns_model = False
        # Opt-in lexicon post-filter. OFF by default so the measured effect is the method's own.
        # Enable only for an explicitly named hybrid arm (and pair it with an M0a + post_filter control).
        self.post_filter: bool = bool(spec.params.get("post_filter", False))

    def apply_post_filter(self, text: str) -> tuple[str, int]:
        """Apply the lexicon profanity/complex-word replacement iff ``post_filter`` is enabled."""
        if not self.post_filter:
            return text, 0
        from minorsafe.data.lexicons import replace_complex_words, replace_profanity

        cleaned, n1 = replace_profanity(text)
        cleaned, n2 = replace_complex_words(cleaned)
        return cleaned, int(n1) + int(n2)

    def _init_generator(self, ctx: RunContext) -> None:
        self.ctx = ctx
        if ctx.preloaded_generator is not None:
            self.model, self.tokenizer = ctx.preloaded_generator
            self._owns_model = False
        else:
            self.model, self.tokenizer, _ = load_decoder(
                ctx.generator_spec, quant=ctx.quant, device=ctx.hardware.device
            )
            self._owns_model = True
        self.generator = ChatGenerator(
            self.model,
            self.tokenizer,
            spec=ctx.generator_spec,
            max_new_tokens=ctx.hardware.max_new_tokens,
        )

    @abstractmethod
    def setup(self, ctx: RunContext) -> SetupReport: ...

    @abstractmethod
    def respond(
        self, batch: list[Example], age_band: str = "teen_12_18", seed: int = 0
    ) -> list[MethodOutput]: ...

    def teardown(self) -> None:
        if self._owns_model and self.model is not None:
            free_model(self.model, self.tokenizer)
        self.model = None
        self.tokenizer = None
        self.generator = None


def create_method(spec: MethodSpec) -> ResponseMethod:
    if spec.kind == "baseline":
        from minorsafe.methods.baseline import BaselineMethod

        return BaselineMethod(spec)
    if spec.kind == "guard":
        from minorsafe.methods.guards.method import GuardResponseMethod

        return GuardResponseMethod(spec)
    if spec.kind == "lora":
        from minorsafe.methods.lora.method import LoRAResponseMethod

        return LoRAResponseMethod(spec)
    if spec.kind == "steering":
        from minorsafe.methods.steering.method import SteeringResponseMethod

        return SteeringResponseMethod(spec)
    raise ValueError(f"Unknown method kind: {spec.kind!r}")
