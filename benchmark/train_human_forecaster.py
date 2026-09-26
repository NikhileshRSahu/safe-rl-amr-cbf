from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.human_forecast_dataset import ForecastSample
from benchmark.human_forecaster import GRUHumanForecaster


def _stack(samples: list[ForecastSample], device="cpu"):
    history = torch.as_tensor(np.stack([s.history for s in samples]), dtype=torch.float32, device=device)
    history_mask = torch.as_tensor(np.stack([s.history_mask for s in samples]), dtype=torch.bool, device=device)
    future = torch.as_tensor(np.stack([s.future_xy for s in samples]), dtype=torch.float32, device=device)
    future_mask = torch.as_tensor(np.stack([s.future_mask for s in samples]), dtype=torch.bool, device=device)
    return history, history_mask, future, future_mask


def constant_velocity_difficulty_weights(
    history: torch.Tensor,
    history_mask: torch.Tensor,
    future: torch.Tensor,
    future_mask: torch.Tensor,
    *,
    forecast_dt: float,
    strength: float = 2.0,
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Weight training examples by how badly CV predicts their training labels.

    This uses only gradient-training trajectories. It does not inspect forecast
    evaluation, validation, or holdout labels. Straight motion stays near the
    baseline weight while stop/restart/reversal samples receive more attention.
    The returned weights are normalized to mean one so optimizer scale remains
    stable when the hard-example strength changes.
    """
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history/future must be rank-3")
    if history_mask.shape != history.shape[:2] or future_mask.shape != future.shape[:2]:
        raise ValueError("mask shape mismatch")
    n = history.shape[0]
    if n == 0:
        return history.new_zeros((0,))
    positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0).expand_as(history_mask)
    invalid = torch.full_like(positions, -1)
    last_idx = torch.where(history_mask, positions, invalid).max(dim=1).values.clamp(min=0)
    batch_idx = torch.arange(n, device=history.device)
    last_xy = history[batch_idx, last_idx, :2]
    last_vel = history[batch_idx, last_idx, 2:4]
    step_times = torch.arange(1, future.shape[1] + 1, device=future.device, dtype=future.dtype) * float(forecast_dt)
    cv = last_xy[:, None, :] + last_vel[:, None, :] * step_times[None, :, None]
    displacement = torch.linalg.norm(future - cv, dim=-1)
    valid = future_mask.to(future.dtype)
    difficulty = (displacement * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    positive = difficulty[difficulty > 1e-6]
    scale = positive.median() if positive.numel() else difficulty.new_tensor(1.0)
    raw = 1.0 + float(strength) * difficulty / scale.clamp_min(1e-3)
    raw = raw.clamp(max=float(max_weight))
    return raw / raw.mean().clamp_min(1e-6)


def forecast_nll(output, target: torch.Tensor, mask: torch.Tensor, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    sigma = output.sigma_xy.clamp_min(1e-4)
    nll = 0.5 * ((target - output.mean_xy) / sigma).pow(2) + torch.log(sigma)
    point = nll.mean(dim=-1)
    valid = mask.to(target.dtype)
    per_sample = (point * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    if sample_weights is None:
        return per_sample.mean()
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


def horizon_weighted_mean_loss(
    output,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct mean-trajectory loss with extra emphasis on later predictions."""
    if target.ndim != 3 or mask.shape != target.shape[:2]:
        raise ValueError("target/mask shape mismatch")
    if target.shape[1] == 0:
        return target.sum() * 0.0
    point_loss = F.smooth_l1_loss(output.mean_xy, target, reduction="none").mean(dim=-1)
    weights = torch.linspace(1.0, 2.0, target.shape[1], device=target.device, dtype=target.dtype)
    valid_weights = mask.to(target.dtype) * weights.unsqueeze(0)
    per_sample = (point_loss * valid_weights).sum(dim=1) / valid_weights.sum(dim=1).clamp_min(1.0)
    if sample_weights is None:
        return per_sample.mean()
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


def forecast_training_loss(
    output,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    mean_loss_weight: float = 2.0,
    sample_weights: torch.Tensor | None = None,
):
    nll = forecast_nll(output, target, mask, sample_weights=sample_weights)
    mean_loss = horizon_weighted_mean_loss(output, target, mask, sample_weights=sample_weights)
    total = nll + float(mean_loss_weight) * mean_loss
    return total, {"nll": nll.detach(), "mean_loss": mean_loss.detach()}


def train_forecaster(
    samples: list[ForecastSample],
    *,
    seed: int = 71,
    epochs: int = 20,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    mean_loss_weight: float = 2.0,
    hard_example_weight: float = 2.0,
) -> tuple[GRUHumanForecaster, dict]:
    if not samples:
        raise ValueError("samples must not be empty")
    seeds = sorted({int(s.seed) for s in samples})
    if any(seed_value < 10100 or seed_value > 10115 for seed_value in seeds):
        raise ValueError("forecaster training samples must use only seeds 10100-10115")
    forecast_dts = np.asarray([float(s.forecast_dt) for s in samples], dtype=np.float64)
    forecast_dt = float(np.median(forecast_dts))
    if not np.allclose(forecast_dts, forecast_dt, rtol=0.0, atol=2e-5):
        raise ValueError("all forecaster samples must share one forecast_dt")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    steps = int(samples[0].future_xy.shape[0])
    horizon_seconds = float(steps * forecast_dt)
    model = GRUHumanForecaster(history_dim=5, hidden_dim=hidden_dim, steps=steps, horizon_seconds=horizon_seconds)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    history, history_mask, future, future_mask = _stack(samples)
    sample_weights = constant_velocity_difficulty_weights(
        history,
        history_mask,
        future,
        future_mask,
        forecast_dt=forecast_dt,
        strength=float(hard_example_weight),
    ).detach()
    losses = []
    last_parts = None
    for _ in range(max(1, int(epochs))):
        out = model(history, history_mask)
        loss, parts = forecast_training_loss(
            out,
            future,
            future_mask,
            mean_loss_weight=float(mean_loss_weight),
            sample_weights=sample_weights,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
        last_parts = parts
    meta = {
        "architecture": "gru_human_forecaster_v1",
        "training_seeds": seeds,
        "history_len": int(history.shape[1]),
        "forecast_steps": steps,
        "forecast_dt": forecast_dt,
        "horizon_seconds": horizon_seconds,
        "hidden_dim": int(hidden_dim),
        "mean_loss_weight": float(mean_loss_weight),
        "hard_example_weight": float(hard_example_weight),
        "final_loss": losses[-1],
        "final_nll": float(last_parts["nll"]) if last_parts is not None else None,
        "final_mean_loss": float(last_parts["mean_loss"]) if last_parts is not None else None,
    }
    return model, meta


def save_forecaster_checkpoint(path, model: GRUHumanForecaster, metadata: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": dict(metadata)}
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_forecaster_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload["metadata"])
    horizon_seconds = float(meta.get("horizon_seconds", float(meta.get("forecast_steps", 8)) * float(meta.get("forecast_dt", 0.25))))
    model = GRUHumanForecaster(hidden_dim=int(meta["hidden_dim"]), steps=int(meta["forecast_steps"]), horizon_seconds=horizon_seconds)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta