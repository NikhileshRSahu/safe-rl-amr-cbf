from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TRAINING_SEED_MIN = 10100
TRAINING_SEED_MAX = 10115


@dataclass(frozen=True)
class ForecastSample:
    history: np.ndarray
    history_mask: np.ndarray
    future_xy: np.ndarray
    future_mask: np.ndarray
    scenario: str
    seed: int


@dataclass(frozen=True)
class ForecastOutput:
    mean_xy: np.ndarray
    sigma_xy: np.ndarray
    mask: np.ndarray


def assert_training_seed_allowed(seed: int) -> None:
    seed = int(seed)
    if not (TRAINING_SEED_MIN <= seed <= TRAINING_SEED_MAX):
        raise ValueError(f"seed {seed} is not in forecast gradient-training split 10100-10115")


def make_forecast_sample(
    timestamps,
    positions,
    *,
    t0_index: int,
    history_len: int,
    horizon_steps: int,
    scenario: str,
    seed: int,
) -> ForecastSample:
    assert_training_seed_allowed(seed)
    ts = np.asarray(timestamps, dtype=np.float32)
    xy = np.asarray(positions, dtype=np.float32)
    if ts.ndim != 1 or xy.shape != (len(ts), 2):
        raise ValueError("timestamps/positions shape mismatch")
    if not np.all(np.diff(ts) > 0):
        raise ValueError("timestamps must be strictly increasing")
    t0_index = int(t0_index)
    history_len = int(history_len)
    horizon_steps = int(horizon_steps)
    if history_len < 2 or horizon_steps < 1:
        raise ValueError("invalid history/horizon")
    if t0_index < 0 or t0_index >= len(ts):
        raise IndexError("t0_index out of range")
    if t0_index + horizon_steps >= len(ts):
        raise ValueError("not enough future samples")

    hist = np.zeros((history_len, 5), dtype=np.float32)
    hist_mask = np.zeros((history_len,), dtype=np.bool_)
    start_src = max(0, t0_index - history_len + 1)
    src_idx = np.arange(start_src, t0_index + 1)
    dst_start = history_len - len(src_idx)
    for k, src in enumerate(src_idx):
        dst = dst_start + k
        hist[dst, :2] = xy[src]
        if src > 0:
            dt = float(ts[src] - ts[src - 1])
            vel = (xy[src] - xy[src - 1]) / max(dt, 1e-6)
        else:
            vel = np.zeros(2, dtype=np.float32)
        hist[dst, 2:4] = vel
        hist[dst, 4] = ts[src]
        hist_mask[dst] = True

    future_xy = xy[t0_index + 1 : t0_index + 1 + horizon_steps].copy()
    future_mask = np.ones((horizon_steps,), dtype=np.bool_)
    if np.any(hist[:, 4][hist_mask] > ts[t0_index] + 1e-7):
        raise RuntimeError("future leakage in forecast history")
    return ForecastSample(hist, hist_mask, future_xy, future_mask, str(scenario), int(seed))
