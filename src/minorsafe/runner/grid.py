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


QLORA_QUANT: Quant = "nf4"


def _uses_qlora(base_method: MethodSpec, gen_spec: ModelSpec, profile: HardwareProfile) -> bool:
    """True when this profile trains L-tier LoRA as QLoRA (``lora_large_mode: qlora``)."""
    return (
        base_method.kind == "lora"
        and profile.lora_large_mode == "qlora"
        and (gen_spec.size_group or "S") == "L"
    )


def expand_experiment_grid(
    exp: ExperimentSpec,
    profile: HardwareProfile,
    models: dict[str, ModelSpec],
) -> list[RunSpec]:
    """Expand an ExperimentSpec into concrete (method, generator, quant, sweep, age_band) RunSpecs.

    On profiles with ``lora_large_mode: qlora`` the default-precision LoRA arm of an L-tier generator
    runs as QLoRA (``nf4`` base); the run keeps ``quant='nf4'`` and records
    ``lora_mode='qlora'`` / ``matched_quant=<profile default>`` so analysis compares it against the
    other methods' default-precision cell. Explicit ``quants`` in a run group are never overridden.
    """
    specs: list[RunSpec] = []
    for group in exp.runs:
        base_method = load_method(group.method)
        gen_aliases = select_generators(group.generators, profile, models)

        sweep_keys = list(group.sweep.keys())
        sweep_vals = [group.sweep[k] for k in sweep_keys]
        sweep_combos = list(itertools.product(*sweep_vals)) if sweep_keys else [()]

        for gen_alias in gen_aliases:
            qlora = not group.quants and _uses_qlora(base_method, models[gen_alias], profile)
            if qlora and QLORA_QUANT not in profile.allowed_quants:
                raise ValueError(
                    f"Profile {profile.name!r} sets lora_large_mode=qlora but does not allow "
                    f"{QLORA_QUANT!r} quantization (needed for {gen_alias})."
                )
            quants: list[Quant] = (
                list(group.quants) if group.quants else [QLORA_QUANT if qlora else profile.default_quant]
            )
            for quant in quants:
                if quant not in profile.allowed_quants:
                    continue
                for age_band in exp.age_bands:
                    for combo in sweep_combos:
                        merged_params = dict(base_method.params)
                        merged_params.update(group.params)
                        for k, v in zip(sweep_keys, combo, strict=True):
                            merged_params[k] = v
                        if qlora:
                            merged_params["lora_mode"] = "qlora"
                            merged_params["matched_quant"] = profile.default_quant

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
                        # Sweep variants must be distinguishable in results (e.g. the four M1 actions).
                        sweep_suffix = ".".join(f"{k}-{v}" for k, v in zip(sweep_keys, combo, strict=True))
                        auto_label = f"{base_method.name}.{sweep_suffix}" if sweep_suffix else None
                        label = group.label or auto_label
                        label_part = label or base_method.name
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
                                label=label,
                            )
                        )
    return specs
