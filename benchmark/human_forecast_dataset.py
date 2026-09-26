from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TRAINING_SEED_MIN = 10100
TRAINING_SEED_MAX = 10115
FORECAST_EVAL_SEED_MIN = 10116
FORECAST_EVAL_SEED_MAX = 10119


@dataclass(frozen=True)
class ForecastSample:
    history: np.ndarray
    history_mask: np.ndarray
    future_xy: np.ndarray
    future_mask: np.ndarray
    scenario: str
    seed: int
    forecast_dt: float = 0.1


@dataclass(frozen=True)
class ForecastOutput:
    mean_xy: np.ndarray
    sigma_xy: np.ndarray
    mask: np.ndarray


def assert_training_seed_allowed(seed: int) -> None:
    seed = int(seed)
    if not (TRAINING_SEED_MIN <= seed <= TRAINING_SEED_MAX):
        raise ValueError(f"seed {seed} is not in forecast gradient-training split 10100-10115")


def assert_evaluation_seed_allowed(seed: int) -> None:
    seed = int(seed)
    if not (FORECAST_EVAL_SEED_MIN <= seed <= FORECAST_EVAL_SEED_MAX):
        raise ValueError(f"seed {seed} is not in forecast evaluation split 10116-10119")


def make_forecast_sample(
    timestamps,
    positions,
    *,
    t0_index: int,
    history_len: int,
    horizon_steps: int,
    scenario: str,
    seed: int,
    future_stride: int = 1,
    enforce_training_seed: bool = True,
) -> ForecastSample:
    if enforce_training_seed:
        assert_training_seed_allowed(seed)
    ts = np.asarray(timestamps, dtype=np.float32)
    xy = np.asarray(positions, dtype=np.float32)
    if ts.ndim != 1 or xy.shape != (len(ts), 2):
        raise ValueError("timestamps/positions shape mismatch")
    if not np.all(np.diff(ts) > 0):
        raise ValueError("timestamps must be strictly increasing")
    t0_index, history_len, horizon_steps, future_stride = int(t0_index), int(history_len), int(horizon_steps), int(future_stride)
    if history_len < 2 or horizon_steps < 1 or future_stride < 1:
        raise ValueError("invalid history/horizon/stride")
    if t0_index < 0 or t0_index >= len(ts):
        raise IndexError("t0_index out of range")
    future_indices = t0_index + future_stride * np.arange(1, horizon_steps + 1)
    if future_indices[-1] >= len(ts):
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

    future_xy = xy[future_indices].copy()
    future_mask = np.ones((horizon_steps,), dtype=np.bool_)
    forecast_dt = float(ts[future_indices[0]] - ts[t0_index])
    if np.any(hist[:, 4][hist_mask] > ts[t0_index] + 1e-7):
        raise RuntimeError("future leakage in forecast history")
    if forecast_dt <= 0.0:
        raise RuntimeError("forecast_dt must be positive")
    return ForecastSample(hist, hist_mask, future_xy, future_mask, str(scenario), int(seed), forecast_dt)


def collect_world_forecast_samples(
    spec,
    seed: int,
    *,
    world_steps: int = 120,
    history_len: int = 8,
    horizon_steps: int = 6,
    stride: int = 2,
    forecast_step_stride: int = 3,
    max_humans: int | None = None,
    split: str = "train",
) -> list[ForecastSample]:
    """Collect causal warehouse forecast samples; future truth is labels only.

    `split='train'` accepts only 10100-10115. `split='eval'` accepts only
    10116-10119. Neither validation nor final holdout seeds are accepted.
    With simulator DT=0.1, six default targets spaced by three simulator
    ticks cover a 1.8 s horizon.
    """
    if split == "train":
        assert_training_seed_allowed(seed)
        enforce_training_seed = True
    elif split == "eval":
        assert_evaluation_seed_allowed(seed)
        enforce_training_seed = False
    else:
        raise ValueError("split must be 'train' or 'eval'")

    from benchmark.warehouse_scenario_world import make_scenario_world
    from benchmark.train_multi_agent_research import DT

    world = make_scenario_world(spec, int(seed))
    world.reset()
    times = []
    tracks = [[] for _ in range(world.nppl)]
    zero_actions = np.tile(np.array([-1.0, 0.0], dtype=np.float32), (world.n, 1))
    for _ in range(int(world_steps)):
        times.append(float(world.steps) * DT)
        for j in range(world.nppl):
            tracks[j].append(np.asarray(world.hp[j], dtype=np.float32).copy())
        world.step(zero_actions, use_cbf=False)
    ts = np.asarray(times, dtype=np.float32)
    samples: list[ForecastSample] = []
    n_people = world.nppl if max_humans is None else min(world.nppl, int(max_humans))
    first_t0 = max(1, history_len - 1)
    last_t0 = len(ts) - horizon_steps * int(forecast_step_stride) - 1
    for j in range(n_people):
        xy = np.asarray(tracks[j], dtype=np.float32)
        for t0 in range(first_t0, last_t0 + 1, max(1, int(stride))):
            samples.append(
                make_forecast_sample(
                    ts,
                    xy,
                    t0_index=t0,
                    history_len=history_len,
                    horizon_steps=horizon_steps,
                    scenario=spec.name,
                    seed=int(seed),
                    future_stride=int(forecast_step_stride),
                    enforce_training_seed=enforce_training_seed,
                )
            )
    return samples
