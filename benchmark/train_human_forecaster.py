from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.human_forecast_dataset import ForecastSample
from benchmark.human_forecaster import GRUHumanForecaster


def _stack(samples: list[ForecastSample], device="cpu"):
    history = torch.as_tensor(np.stack([s.history for s in samples]), dtype=torch.float32, device=device)
    history_mask = torch.as_tensor(np.stack([s.history_mask for s in samples]), dtype=torch.bool, device=device)
    future = torch.as_tensor(np.stack([s.future_xy for s in samples]), dtype=torch.float32, device=device)
    future_mask = torch.as_tensor(np.stack([s.future_mask for s in samples]), dtype=torch.bool, device=device)
    return history, history_mask, future, future_mask


def forecast_nll(output, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1).expand_as(target)
    if not valid.any():
        return target.sum() * 0.0
    sigma = output.sigma_xy.clamp_min(1e-4)
    nll = 0.5 * ((target - output.mean_xy) / sigma).pow(2) + torch.log(sigma)
    return nll[valid].mean()


def train_forecaster(
    samples: list[ForecastSample],
    *,
    seed: int = 71,
    epochs: int = 20,
    hidden_dim: int = 64,
    lr: float = 1e-3,
) -> tuple[GRUHumanForecaster, dict]:
    if not samples:
        raise ValueError("samples must not be empty")
    seeds = sorted({int(s.seed) for s in samples})
    if any(seed_value < 10100 or seed_value > 10115 for seed_value in seeds):
        raise ValueError("forecaster training samples must use only seeds 10100-10115")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    steps = int(samples[0].future_xy.shape[0])
    model = GRUHumanForecaster(history_dim=5, hidden_dim=hidden_dim, steps=steps)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    history, history_mask, future, future_mask = _stack(samples)
    losses = []
    for _ in range(max(1, int(epochs))):
        out = model(history, history_mask)
        loss = forecast_nll(out, future, future_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
    meta = {
        "architecture": "gru_human_forecaster_v1",
        "training_seeds": seeds,
        "history_len": int(history.shape[1]),
        "forecast_steps": steps,
        "hidden_dim": int(hidden_dim),
        "final_loss": losses[-1],
    }
    return model, meta


def save_forecaster_checkpoint(path, model: GRUHumanForecaster, metadata: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": dict(metadata)}
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_forecaster_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload["metadata"])
    model = GRUHumanForecaster(hidden_dim=int(meta["hidden_dim"]), steps=int(meta["forecast_steps"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta
