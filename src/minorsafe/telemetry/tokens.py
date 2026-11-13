"""Per-query token accounting across all pipeline components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TokenUsage:
    generator_in: int = 0
    generator_out: int = 0  # visible answer tokens (includes reasoning when reasoning is not separated)
    generator_reasoning_out: int = 0  # <think> tokens of reasoning models (subset of the raw output)
    guard_in: int = 0
    guard_out: int = 0
    rewrite_in: int = 0
    rewrite_out: int = 0
    regen_in: int = 0
    regen_out: int = 0
    encoder_in: int = 0  # encoder guard / MLM forward tokens

    @property
    def input_total(self) -> int:
        return self.generator_in + self.guard_in + self.rewrite_in + self.regen_in + self.encoder_in

    @property
    def output_total(self) -> int:
        return (
            self.generator_out
            + self.generator_reasoning_out
            + self.guard_out
            + self.rewrite_out
            + self.regen_out
        )

    @property
    def total(self) -> int:
        return self.input_total + self.output_total

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def to_dict(self, prefix: str = "tok_") -> dict:
        d = {f"{prefix}{k}": v for k, v in asdict(self).items()}
        d[f"{prefix}input_total"] = self.input_total
        d[f"{prefix}output_total"] = self.output_total
        d[f"{prefix}total"] = self.total
        return d
