from __future__ import annotations

import numpy as np

from benchmark.human_forecast_dataset import ForecastOutput
from benchmark.human_history import HistorySequence


class ConstantVelocityForecaster:
    def __init__(self, horizon_seconds: float = 2.0, steps: int = 8, sigma_base: float = 0.08):
        if int(steps) < 1:
            raise ValueError("steps must be >= 1")
        self.horizon_seconds = float(horizon_seconds)
        self.steps = int(steps)
        self.sigma_base = float(sigma_base)

    def predict(self, sequences: list[HistorySequence]) -> ForecastOutput:
        n = len(sequences)
        mean_xy = np.zeros((n, self.steps, 2), dtype=np.float32)
        sigma_xy = np.zeros((n, self.steps, 2), dtype=np.float32)
        mask = np.zeros((n, self.steps), dtype=np.bool_)
        if n == 0:
            return ForecastOutput(mean_xy, sigma_xy, mask)
        dts = np.linspace(
            self.horizon_seconds / self.steps,
            self.horizon_seconds,
            self.steps,
            dtype=np.float32,
        )
        for i, seq in enumerate(sequences):
            valid = np.flatnonzero(seq.mask)
            if len(valid) == 0:
                continue
            last = seq.features[valid[-1]]
            p0 = last[:2]
            v = last[2:4]
            mean_xy[i] = p0[None, :] + dts[:, None] * v[None, :]
            growth = self.sigma_base * (1.0 + dts / max(self.horizon_seconds, 1e-6))
            sigma_xy[i] = growth[:, None]
            mask[i] = True
        return ForecastOutput(mean_xy, sigma_xy, mask)
