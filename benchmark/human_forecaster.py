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


class GRUHumanForecaster(nn.Module):
    """Shared per-track causal GRU forecaster."""

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
        delta = raw[..., :2]
        sigma = torch.clamp(F.softplus(raw[..., 2:]) + 1e-4, 1e-4, 3.0)
        positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0).expand_as(history_mask)
        invalid = torch.full_like(positions, -1)
        last_idx = torch.where(history_mask, positions, invalid).max(dim=1).values.clamp(min=0)
        last_xy = history[torch.arange(n, device=history.device), last_idx, :2]
        mean_xy = last_xy[:, None, :] + torch.cumsum(delta, dim=1)
        valid_track = history_mask.any(dim=1)
        mask = valid_track[:, None].expand(n, self.steps)
        mean_xy = torch.where(mask.unsqueeze(-1), mean_xy, torch.zeros_like(mean_xy))
        sigma = torch.where(mask.unsqueeze(-1), sigma, torch.zeros_like(sigma))
        return TorchForecastOutput(mean_xy, sigma, mask)
