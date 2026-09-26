from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from benchmark.human_forecast_dataset import ForecastOutput


FORECAST_RISK_DIM = 6


@dataclass(frozen=True)
class ForecastRiskBatch:
    features: np.ndarray
    mask: np.ndarray


def _vec2(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (2,):
        raise ValueError("expected 2-vector")
    return arr


def build_forecast_risk_batch(
    robot_pos,
    robot_vel,
    route_direction,
    forecast: ForecastOutput,
    *,
    forecast_dt: float,
    observation_ages,
    safe_distance: float = 0.8,
    max_distance: float = 6.0,
    max_age: float = 2.0,
) -> ForecastRiskBatch:
    """Convert causal trajectory forecasts into bounded per-entity risk summaries."""
    robot_pos = _vec2(robot_pos)
    robot_vel = _vec2(robot_vel)
    route = _vec2(route_direction)
    route_norm = float(np.linalg.norm(route))
    if route_norm <= 1e-8:
        route = np.array([1.0, 0.0], dtype=np.float32)
    else:
        route = route / route_norm
    mean = np.asarray(forecast.mean_xy, dtype=np.float32)
    sigma = np.asarray(forecast.sigma_xy, dtype=np.float32)
    mask = np.asarray(forecast.mask, dtype=np.bool_)
    if mean.ndim != 3 or mean.shape[-1] != 2 or sigma.shape != mean.shape or mask.shape != mean.shape[:2]:
        raise ValueError("forecast tensor contract broken")
    n, h, _ = mean.shape
    ages = np.asarray(observation_ages, dtype=np.float32)
    if ages.shape != (n,):
        raise ValueError("observation_ages shape mismatch")
    features = np.zeros((n, FORECAST_RISK_DIM), dtype=np.float32)
    valid_tracks = mask.any(axis=1)
    lateral_axis = np.array([-route[1], route[0]], dtype=np.float32)
    dt = max(float(forecast_dt), 1e-6)

    for i in range(n):
        valid = np.flatnonzero(mask[i])
        if len(valid) == 0:
            continue
        pts = mean[i, valid]
        sig = sigma[i, valid]
        times = (valid.astype(np.float32) + 1.0) * dt
        robot_future = robot_pos[None, :] + robot_vel[None, :] * times[:, None]
        rel = pts - robot_future
        dist = np.linalg.norm(rel, axis=1)
        k = int(np.argmin(dist))
        min_sep = float(dist[k])
        t_min = float(times[k])
        unc = float(np.mean(np.linalg.norm(sig, axis=1)))
        spatial = 1.0 / (1.0 + math.exp((min_sep - float(safe_distance)) / 0.20))
        temporal = math.exp(-t_min / 1.5)
        uncertainty_gain = min(2.0, 1.0 + unc)
        risk = float(np.clip(spatial * temporal * uncertainty_gain, 0.0, 1.0))
        longitudinal = np.dot(pts - robot_pos[None, :], route)
        lateral = np.abs(np.dot(pts - robot_pos[None, :], lateral_axis))
        corridor = float(np.max((longitudinal > 0.0) & (longitudinal < 4.0) & (lateral < 0.9)))
        features[i] = np.array([
            np.clip(min_sep / max_distance, 0.0, 1.0),
            np.clip(t_min / max(h * dt, dt), 0.0, 1.0),
            risk,
            np.clip(unc / 2.0, 0.0, 1.0),
            corridor,
            np.clip(max(0.0, float(ages[i])) / max_age, 0.0, 1.0),
        ], dtype=np.float32)
    return ForecastRiskBatch(features, valid_tracks.astype(np.bool_))
