"""Experiment grid expansion and test-split guardrail."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from minorsafe.config import (
    ExperimentSpec,
    HardwareProfile,
    MethodSpec,
    ModelSpec,
    Quant,
    load_method,
    select_generators,
)
from minorsafe.utils import stable_hash


@dataclass
class RunSpec:
    run_id: str
    experiment: str
    method_spec: MethodSpec
    generator_alias: str
    quant: Quant
    age_band: str
    seed: int
    split: str
    method_params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


def verify_split_guardrail(exp: ExperimentSpec, final: bool = False) -> None:
    """Enforce that 'test' split experiments require explicit --final flag."""
    if exp.split == "test" and not final:
        raise PermissionError(
            f"Experiment '{exp.name}' targets the 'test' split. "
            "Pass --final explicitly to unlock held-out test evaluation."
        )


def expand_experiment_grid(
    exp: ExperimentSpec,
    profile: HardwareProfile,
    models: dict[str, ModelSpec],
) -> list[RunSpec]:
    """Expand an ExperimentSpec into concrete (method, generator, quant, sweep, age_band) RunSpecs."""
    specs: list[RunSpec] = []
    for group in exp.runs:
        base_method = load_method(group.method)
        gen_aliases = select_generators(group.generators, profile, models)
        quants: list[Quant] = list(group.quants) if group.quants else [profile.default_quant]

        sweep_keys = list(group.sweep.keys())
        sweep_vals = [group.sweep[k] for k in sweep_keys]
        sweep_combos = list(itertools.product(*sweep_vals)) if sweep_keys else [()]

        for gen_alias in gen_aliases:
            for quant in quants:
                if quant not in profile.allowed_quants:
                    continue
                for age_band in exp.age_bands:
                    for combo in sweep_combos:
                        merged_params = dict(base_method.params)
                        merged_params.update(group.params)
                        for k, v in zip(sweep_keys, combo, strict=True):
                            merged_params[k] = v

                        method_copy = MethodSpec(
                            name=base_method.name,
                            kind=base_method.kind,
                            params=merged_params,
                        )
                        suffix_hash = stable_hash(
                            {
                                "m": base_method.name,
                                "g": gen_alias,
                                "q": quant,
                                "a": age_band,
                                "p": merged_params,
                                "l": group.label,
                            },
                            length=8,
                        )
                        label_part = group.label or base_method.name
                        run_id = f"{label_part}__{gen_alias}__{quant}__{suffix_hash}"
                        specs.append(
                            RunSpec(
                                run_id=run_id,
                                experiment=exp.name,
                                method_spec=method_copy,
                                generator_alias=gen_alias,
                                quant=quant,
                                age_band=age_band,
                                seed=exp.seeds[0] if exp.seeds else 0,
                                split=exp.split,
                                method_params=merged_params,
                                label=group.label,
                            )
                        )
    return specs

