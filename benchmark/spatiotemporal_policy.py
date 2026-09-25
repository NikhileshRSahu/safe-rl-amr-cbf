from __future__ import annotations

import math

import torch
import torch.nn as nn

from benchmark.warehouse_interaction_features import FEATURE_DIM


def reset_hidden(hidden: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """Reset recurrent state only for agents whose episodes terminated."""
    if hidden.ndim != 2:
        raise ValueError("hidden must be [B,H]")
    done = done.to(device=hidden.device, dtype=torch.bool).reshape(-1)
    if done.shape[0] != hidden.shape[0]:
        raise ValueError("done batch does not match hidden batch")
    keep = (~done).to(dtype=hidden.dtype).unsqueeze(-1)
    return hidden * keep


class RiskAttentionSceneEncoder(nn.Module):
    """Permutation-invariant entity attention followed by temporal memory."""

    def __init__(
        self,
        ego_dim: int,
        entity_dim: int = FEATURE_DIM,
        entity_embed_dim: int = 64,
        ego_embed_dim: int = 96,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.ego_dim = int(ego_dim)
        self.entity_dim = int(entity_dim)
        self.hidden_dim = int(hidden_dim)
        self.entity_encoder = nn.Sequential(
            nn.Linear(self.entity_dim, entity_embed_dim),
            nn.ReLU(),
            nn.Linear(entity_embed_dim, entity_embed_dim),
            nn.ReLU(),
        )
        # Risk-aware score consumes the learned embedding plus explicit
        # finite-horizon t_CPA, d_CPA and bounded risk channels.
        self.attention = nn.Sequential(
            nn.Linear(entity_embed_dim + 3, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_dim, ego_embed_dim),
            nn.ReLU(),
            nn.Linear(ego_embed_dim, ego_embed_dim),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(ego_embed_dim + entity_embed_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def _pool_entities(self, entities: torch.Tensor, mask: torch.Tensor):
        if entities.ndim != 3:
            raise ValueError("entities must be [B,N,F]")
        if mask.ndim != 2 or mask.shape[:2] != entities.shape[:2]:
            raise ValueError("mask must be [B,N]")
        b, n, f = entities.shape
        if f != self.entity_dim:
            raise ValueError(f"expected entity dim {self.entity_dim}, got {f}")
        if n == 0:
            pooled = entities.new_zeros((b, self.entity_encoder[-2].out_features))
            weights = entities.new_zeros((b, 0))
            return pooled, weights

        embedded = self.entity_encoder(entities)
        risk_channels = entities[..., 6:9]
        logits = self.attention(torch.cat([embedded, risk_channels], dim=-1)).squeeze(-1)
        valid = mask.to(dtype=torch.bool)
        very_negative = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~valid, very_negative)

        any_valid = valid.any(dim=1, keepdim=True)
        safe_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(safe_logits, dim=1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = torch.where(any_valid, weights / denom, torch.zeros_like(weights))
        pooled = torch.sum(weights.unsqueeze(-1) * embedded, dim=1)
        return pooled, weights

    def forward(
        self,
        ego_static: torch.Tensor,
        entities: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ):
        if ego_static.ndim != 2 or ego_static.shape[-1] != self.ego_dim:
            raise ValueError(f"ego_static must be [B,{self.ego_dim}]")
        b = ego_static.shape[0]
        if hidden is None:
            hidden = ego_static.new_zeros((b, self.hidden_dim))
        if hidden.shape != (b, self.hidden_dim):
            raise ValueError("hidden has wrong shape")
        pooled, weights = self._pool_entities(entities, mask)
        ego_emb = self.ego_encoder(ego_static)
        fused = self.fuse(torch.cat([ego_emb, pooled], dim=-1))
        next_hidden = self.gru(fused, hidden)
        return next_hidden, next_hidden, weights


class STASACActor(nn.Module):
    """SAC actor whose recurrent scene state directly controls (v, omega)."""

    def __init__(self, ego_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = RiskAttentionSceneEncoder(ego_dim=ego_dim, hidden_dim=hidden_dim)
        self.hidden_dim = self.encoder.hidden_dim
        self.mu = nn.Linear(self.hidden_dim, 2)
        self.log_std = nn.Linear(self.hidden_dim, 2)

    def sample(
        self,
        ego_static: torch.Tensor,
        entities: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        scene, next_hidden, weights = self.encoder(ego_static, entities, mask, hidden)
        mu = self.mu(scene)
        log_std = torch.clamp(self.log_std(scene), -5.0, 1.0)
        if deterministic:
            return torch.tanh(mu), None, next_hidden, weights
        std = torch.exp(log_std)
        noise = torch.randn_like(mu)
        pre_tanh = mu + std * noise
        action = torch.tanh(pre_tanh)
        gaussian_logp = (
            -0.5 * ((pre_tanh - mu) / std).pow(2)
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)
        squash = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        logp = gaussian_logp - squash
        return action, logp, next_hidden, weights


class _QNetwork(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 1),
        )

    def forward(self, scene: torch.Tensor, action: torch.Tensor):
        if scene.ndim != 2 or action.ndim != 2 or action.shape[-1] != 2:
            raise ValueError("scene/action shapes must be [B,H] and [B,2]")
        return self.net(torch.cat([scene, action], dim=-1))


class TwinRecurrentQ(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.q1 = _QNetwork(hidden_dim)
        self.q2 = _QNetwork(hidden_dim)

    def forward(self, scene: torch.Tensor, action: torch.Tensor):
        return self.q1(scene, action), self.q2(scene, action)
