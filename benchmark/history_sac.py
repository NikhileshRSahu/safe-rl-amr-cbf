from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from benchmark.train_multi_agent_research import OBS_DIM


HISTORY_OBS_DIM = 2 * OBS_DIM


def augment_observation(current, previous):
    """Concatenate the current observable state and its one-step visible delta.

    No latent pedestrian intent or simulator-only field is used.  At episode
    reset, callers should pass previous=current so the temporal channel is zero.
    """
    cur = np.asarray(current, dtype=np.float32)
    prev = np.asarray(previous, dtype=np.float32)
    return np.concatenate([cur, cur - prev], axis=-1).astype(np.float32)


class HistoryActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(HISTORY_OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, 2)
        self.ls = nn.Linear(128, 2)

    def sample(self, s, det=False):
        h = self.h(s)
        mu = self.mu(h)
        ls = torch.clamp(self.ls(h), -5, 1)
        z = mu if det else mu + torch.exp(ls) * torch.randn_like(mu)
        a = torch.tanh(z)
        if det:
            return a, None
        lp = (
            -.5 * ((z - mu) / torch.exp(ls)) ** 2
            - ls
            - .5 * math.log(2 * math.pi)
        ).sum(-1, keepdim=True) - torch.log(1 - a * a + 1e-6).sum(-1, keepdim=True)
        return a, lp


class HistoryQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(
            nn.Linear(HISTORY_OBS_DIM + 2, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
        )

    def forward(self, s, a):
        return self.n(torch.cat([s, a], -1))


def transfer_run11_actor(base_actor, history_actor):
    """Embed a Run-11 actor exactly into the augmented actor at zero delta.

    Current-observation weights copy into the first OBS_DIM input columns and
    all temporal-delta columns start at zero. Remaining layers are copied
    exactly, so the upgraded actor begins with identical deterministic actions
    whenever no visible one-step change is supplied.
    """
    with torch.no_grad():
        base_first = base_actor.h[0]
        hist_first = history_actor.h[0]
        hist_first.weight.zero_()
        hist_first.weight[:, :OBS_DIM].copy_(base_first.weight)
        hist_first.bias.copy_(base_first.bias)

        history_actor.h[2].weight.copy_(base_actor.h[2].weight)
        history_actor.h[2].bias.copy_(base_actor.h[2].bias)
        history_actor.mu.weight.copy_(base_actor.mu.weight)
        history_actor.mu.bias.copy_(base_actor.mu.bias)
        history_actor.ls.weight.copy_(base_actor.ls.weight)
        history_actor.ls.bias.copy_(base_actor.ls.bias)
    return history_actor
