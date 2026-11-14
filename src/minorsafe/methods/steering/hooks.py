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
        # Counters are accumulated on-device (no .item() inside the hook) so the hook never forces a
        # GPU->CPU synchronisation per decoded token, which would inflate the measured M3 latency.
        self.tokens_evaluated = 0  # Python int from tensor shapes: no sync required
        self._steered_t: torch.Tensor | None = None
        self._rho_sum_t: torch.Tensor | None = None

    @property
    def tokens_steered(self) -> int:
        return int(self._steered_t.item()) if self._steered_t is not None else 0

    @property
    def rho_sum(self) -> float:
        return float(self._rho_sum_t.item()) if self._rho_sum_t is not None else 0.0

    def _accumulate(self, n_eval: int, n_steered: torch.Tensor, rho_sum: torch.Tensor) -> None:
        self.tokens_evaluated += n_eval
        n_steered = n_steered.detach().to(torch.int64)
        rho_sum = rho_sum.detach().to(torch.float64)
        self._steered_t = n_steered if self._steered_t is None else self._steered_t + n_steered
        self._rho_sum_t = rho_sum if self._rho_sum_t is None else self._rho_sum_t + rho_sum

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
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Apply a single concept direction to ``positions`` of shape [N, D].

        Returns (updated, n_evaluated, n_steered_tensor, rho_sum_tensor); tensors stay on-device.
        """
        v = cd.vector.to(device=positions.device, dtype=positions.dtype)
        v_norm = v / torch.clamp(torch.norm(v), min=1e-8)
        rho = torch.matmul(positions, v_norm)  # [N]
        n_eval = int(rho.numel())
        rho_total = rho.float().sum()

        if self.mode == "swfc":
            delta = self.alpha * v_norm.unsqueeze(0)
            n_all = torch.tensor(n_eval, device=positions.device)
            return (positions + delta).to(positions.dtype), n_eval, n_all, rho_total

        violation_mask = rho < cd.m_l  # [N]
        if self.mode == "sttp":
            shift = (cd.s_l - rho).unsqueeze(-1) * v_norm.unsqueeze(0)
        elif self.mode == "stmp":
            shift = 2.0 * (cd.m_l - rho).unsqueeze(-1) * v_norm.unsqueeze(0)
        else:
            raise ValueError(f"Unknown steering mode: {self.mode!r}")

        # torch.where keeps compliant rows bitwise identical without a data-dependent (syncing) branch.
        updated = torch.where(
            violation_mask.unsqueeze(-1),
            (positions + shift).to(positions.dtype),
            positions,
        )
        return updated, n_eval, violation_mask.sum(), rho_total

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
                self._accumulate(n_ev, n_st, r_sum)
            h = flat
        else:
            # Shape [B, T, D]
            if self.prefill or h.shape[1] == 1:
                flat = h.reshape(-1, h.shape[-1])
                for cd in self.directions:
                    flat, n_ev, n_st, r_sum = self._apply_direction(flat, cd)
                    self._accumulate(n_ev, n_st, r_sum)
                h = flat.reshape_as(h)
            else:
                last_tok = h[:, -1, :]
                for cd in self.directions:
                    last_tok, n_ev, n_st, r_sum = self._apply_direction(last_tok, cd)
                    self._accumulate(n_ev, n_st, r_sum)
                h[:, -1, :] = last_tok

        if is_tuple:
            return (h, *output[1:])
        return h

    def stats(self) -> dict[str, float | int]:
        steered = self.tokens_steered  # single sync here, outside the decode loop
        rate = steered / float(self.tokens_evaluated) if self.tokens_evaluated > 0 else 0.0
        mean_rho = self.rho_sum / float(self.tokens_evaluated) if self.tokens_evaluated > 0 else 0.0
        return {
            "tokens_evaluated": self.tokens_evaluated,
            "tokens_steered": steered,
            "intervention_rate": float(rate),
            "mean_projection_rho": float(mean_rho),
        }
