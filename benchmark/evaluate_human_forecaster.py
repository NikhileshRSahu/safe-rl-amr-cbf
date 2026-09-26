from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from benchmark.constant_velocity_forecaster import ConstantVelocityForecaster
from benchmark.human_history import HistorySequence


def displacement_metrics(pred_xy, target_xy, mask):
    pred = np.asarray(pred_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or mask.shape != pred.shape[:-1]:
        raise ValueError("metric shapes do not match")
    dist = np.linalg.norm(pred - target, axis=-1)
    valid = mask
    ade = float(dist[valid].mean()) if valid.any() else 0.0
    fde_values = []
    for i in range(len(dist)):
        idx = np.flatnonzero(valid[i])
        if len(idx):
            fde_values.append(float(dist[i, idx[-1]]))
    fde = float(np.mean(fde_values)) if fde_values else 0.0
    return {"ade": ade, "fde": fde}


def uncertainty_coverage(mean_xy, sigma_xy, target_xy, mask, k: float = 2.0):
    mean = np.asarray(mean_xy, np.float32)
    sigma = np.asarray(sigma_xy, np.float32)
    target = np.asarray(target_xy, np.float32)
    mask = np.asarray(mask, bool)
    inside = (np.abs(target - mean) <= float(k) * np.maximum(sigma, 1e-6)).all(axis=-1)
    return float(inside[mask].mean()) if mask.any() else 0.0


def evaluate_forecaster(model, samples):
    history = torch.as_tensor(np.stack([s.history for s in samples]), dtype=torch.float32)
    history_mask = torch.as_tensor(np.stack([s.history_mask for s in samples]), dtype=torch.bool)
    target = np.stack([s.future_xy for s in samples]).astype(np.float32)
    target_mask = np.stack([s.future_mask for s in samples]).astype(bool)
    with torch.no_grad():
        out = model(history, history_mask)
    mean = out.mean_xy.cpu().numpy()
    sigma = out.sigma_xy.cpu().numpy()
    result = displacement_metrics(mean, target, target_mask)
    result["coverage_2sigma"] = uncertainty_coverage(mean, sigma, target, target_mask)
    by_scenario = defaultdict(list)
    for i, sample in enumerate(samples):
        by_scenario[sample.scenario].append(i)
    result["per_scenario"] = {}
    for scenario, indices in by_scenario.items():
        idx = np.asarray(indices, dtype=int)
        result["per_scenario"][scenario] = displacement_metrics(mean[idx], target[idx], target_mask[idx])
    return result


def evaluate_constant_velocity(samples, *, horizon_seconds: float = 2.0):
    sequences = []
    for sample in samples:
        valid = np.flatnonzero(sample.history_mask)
        age = 0.0
        sequences.append(HistorySequence(sample.history.copy(), sample.history_mask.copy(), age))
    steps = int(samples[0].future_xy.shape[0])
    out = ConstantVelocityForecaster(horizon_seconds=horizon_seconds, steps=steps).predict(sequences)
    target = np.stack([s.future_xy for s in samples]).astype(np.float32)
    target_mask = np.stack([s.future_mask for s in samples]).astype(bool)
    return displacement_metrics(out.mean_xy, target, target_mask)
