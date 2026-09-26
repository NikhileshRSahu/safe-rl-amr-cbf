from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TorchForecastOutput:
    mean_xy: torch.Tensor
    sigma_xy: torch.Tensor
    mask: torch.Tensor


def motion_nonlinearity_gate(
    history: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    onset_mps: float = 0.01,
    full_mps: float = 0.07,
) -> torch.Tensor:
    """Return a causal [0,1] gate from observed velocity changes only.

    Constant velocity is a very strong warehouse pedestrian baseline. Learned
    residuals should therefore be trusted only after the observed track shows
    evidence of a stop, restart, reversal, or turn. Thresholds are matched to
    physically realistic pedestrian deceleration: a ~0.05-0.06 m/s velocity
    change per simulator tick is already strong evidence of changing intent.
    """
    if history.ndim != 3 or history.shape[-1] < 4:
        raise ValueError("history must have shape [N,T,D>=4]")
    if history_mask.shape != history.shape[:2]:
        raise ValueError("history_mask shape mismatch")
    n, t, _ = history.shape
    if n == 0:
        return history.new_zeros((0,))
    if t < 2:
        return history.new_zeros((n,))
    vel = history[..., 2:4]
    dv = torch.linalg.norm(vel[:, 1:] - vel[:, :-1], dim=-1)
    valid_pair = history_mask[:, 1:] & history_mask[:, :-1]
    dv = torch.where(valid_pair, dv, torch.zeros_like(dv))
    max_dv = dv.max(dim=1).values
    width = max(float(full_mps) - float(onset_mps), 1e-6)
    return ((max_dv - float(onset_mps)) / width).clamp(0.0, 1.0)


class GRUHumanForecaster(nn.Module):
    """Shared per-track causal GRU residual forecaster over constant velocity."""

    def __init__(self, history_dim: int = 5, hidden_dim: int = 64, steps: int = 8, horizon_seconds: float = 2.0):
        super().__init__()
        self.history_dim = int(history_dim)
        self.hidden_dim = int(hidden_dim)
        self.steps = int(steps)
        self.horizon_seconds = float(horizon_seconds)
        self.encoder = nn.GRU(self.history_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.steps * 4),
        )
        # Start exactly at the analytical constant-velocity baseline.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor) -> TorchForecastOutput:
        if history.ndim != 3 or history.shape[-1] != self.history_dim:
            raise ValueError("history must have shape [N,T,history_dim]")
        if history_mask.shape != history.shape[:2]:
            raise ValueError("history_mask shape mismatch")
        n = history.shape[0]
        if n == 0:
            z = history.new_zeros((0, self.steps, 2))
            return TorchForecastOutput(z, z.clone(), history_mask.new_zeros((0, self.steps)))

        masked = history * history_mask.unsqueeze(-1).to(history.dtype)
        _, h = self.encoder(masked)
        raw = self.head(h[-1]).reshape(n, self.steps, 4)
        residual = raw[..., :2]
        sigma = torch.clamp(F.softplus(raw[..., 2:]) + 1e-4, 1e-4, 3.0)

        positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0).expand_as(history_mask)
        invalid = torch.full_like(positions, -1)
        last_idx = torch.where(history_mask, positions, invalid).max(dim=1).values.clamp(min=0)
        batch_idx = torch.arange(n, device=history.device)
        last_xy = history[batch_idx, last_idx, :2]
        last_vel = history[batch_idx, last_idx, 2:4]

        forecast_dt = self.horizon_seconds / max(1, self.steps)
        step_times = (
            torch.arange(1, self.steps + 1, device=history.device, dtype=history.dtype)
            * float(forecast_dt)
        )
        cv_mean = last_xy[:, None, :] + last_vel[:, None, :] * step_times[None, :, None]
        residual_gate = motion_nonlinearity_gate(history, history_mask)
        mean_xy = cv_mean + residual * residual_gate[:, None, None]

        valid_track = history_mask.any(dim=1)
        mask = valid_track[:, None].expand(n, self.steps)
        mean_xy = torch.where(mask.unsqueeze(-1), mean_xy, torch.zeros_like(mean_xy))
        sigma = torch.where(mask.unsqueeze(-1), sigma, torch.zeros_like(sigma))
        return TorchForecastOutput(mean_xy, sigma, mask)
