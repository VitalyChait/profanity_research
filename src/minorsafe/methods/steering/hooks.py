"""Runtime residual interception hooks: SwFC, StTP, and StMP (Phase 5.4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ConceptDirection:
    concept: str
    layer_idx: int
    vector: torch.Tensor  # unit steering vector [hidden_dim]
    m_l: float  # 1D logistic regression decision boundary
    s_l: float  # target projection coordinate in aligned half-space
    mu_pos: float = 0.0
    sigma_pos: float = 1.0
    mu_neg: float = 0.0
    sigma_neg: float = 1.0
    separability_auroc: float = 0.5
    extra: dict = field(default_factory=dict)


class ResidualSteeringHook:
    """PyTorch forward hook supporting SwFC, StTP, and StMP across tuple or bare-Tensor decoder layers."""

    def __init__(
        self,
        directions: list[ConceptDirection],
        mode: str = "sttp",  # swfc | sttp | stmp
        alpha: float = 3.0,
        prefill: bool = False,
    ):
        self.directions = directions
        self.mode = mode.lower()
        self.alpha = float(alpha)
        self.prefill = prefill
        self.enabled = True
        self._handles: list = []
        self.reset_stats()

    def reset_stats(self) -> None:
        self.tokens_evaluated = 0
        self.tokens_steered = 0
        self.rho_sum = 0.0

    def attach(self, layer_module: torch.nn.Module) -> ResidualSteeringHook:
        handle = layer_module.register_forward_hook(self)
        self._handles.append(handle)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _apply_direction(
        self, positions: torch.Tensor, cd: ConceptDirection
    ) -> tuple[torch.Tensor, int, int, float]:
        """Apply a single concept direction to ``positions`` of shape [N, D]."""
        v = cd.vector.to(device=positions.device, dtype=positions.dtype)
        v_norm = v / torch.clamp(torch.norm(v), min=1e-8)
        rho = torch.matmul(positions, v_norm)  # [N]
        n_eval = int(rho.numel())
        rho_total = float(rho.float().sum().item())

        if self.mode == "swfc":
            delta = self.alpha * v_norm.unsqueeze(0)
            return (positions + delta).to(positions.dtype), n_eval, n_eval, rho_total

        violation_mask = rho < cd.m_l  # [N]
        n_steered = int(violation_mask.sum().item())
        if n_steered == 0:
            return positions, n_eval, 0, rho_total

        if self.mode == "sttp":
            shift = (cd.s_l - rho).unsqueeze(-1) * v_norm.unsqueeze(0)
        elif self.mode == "stmp":
            shift = 2.0 * (cd.m_l - rho).unsqueeze(-1) * v_norm.unsqueeze(0)
        else:
            raise ValueError(f"Unknown steering mode: {self.mode!r}")

        updated = torch.where(
            violation_mask.unsqueeze(-1),
            (positions + shift).to(positions.dtype),
            positions,
        )
        return updated, n_eval, n_steered, rho_total

    def __call__(self, module, inputs, output):
        if not self.enabled or not self.directions:
            return output

        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
            return output

        h = hidden.clone()
        if h.ndim == 2:
            flat = h
            for cd in self.directions:
                flat, n_ev, n_st, r_sum = self._apply_direction(flat, cd)
                self.tokens_evaluated += n_ev
                self.tokens_steered += n_st
                self.rho_sum += r_sum
            h = flat
        else:
            # Shape [B, T, D]
            if self.prefill or h.shape[1] == 1:
                flat = h.reshape(-1, h.shape[-1])
                for cd in self.directions:
                    flat, n_ev, n_st, r_sum = self._apply_direction(flat, cd)
                    self.tokens_evaluated += n_ev
                    self.tokens_steered += n_st
                    self.rho_sum += r_sum
                h = flat.reshape_as(h)
            else:
                last_tok = h[:, -1, :]
                for cd in self.directions:
                    last_tok, n_ev, n_st, r_sum = self._apply_direction(last_tok, cd)
                    self.tokens_evaluated += n_ev
                    self.tokens_steered += n_st
                    self.rho_sum += r_sum
                h[:, -1, :] = last_tok

        if is_tuple:
            return (h, *output[1:])
        return h

    def stats(self) -> dict[str, float | int]:
        rate = self.tokens_steered / float(self.tokens_evaluated) if self.tokens_evaluated > 0 else 0.0
        mean_rho = self.rho_sum / float(self.tokens_evaluated) if self.tokens_evaluated > 0 else 0.0
        return {
            "tokens_evaluated": self.tokens_evaluated,
            "tokens_steered": self.tokens_steered,
            "intervention_rate": float(rate),
            "mean_projection_rho": float(mean_rho),
        }
