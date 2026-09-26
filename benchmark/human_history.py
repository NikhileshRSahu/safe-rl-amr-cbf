from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistorySequence:
    features: np.ndarray
    mask: np.ndarray
    observation_age: float


@dataclass(frozen=True)
class _Sample:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray


def _vec2(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (2,):
        raise ValueError(f"expected 2-vector, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite vector")
    return arr


class HumanTrackHistory:
    """Fixed-length causal human-track histories for motion forecasting."""

    def __init__(self, history_len: int = 8):
        if int(history_len) < 2:
            raise ValueError("history_len must be >= 2")
        self.history_len = int(history_len)
        self._samples = defaultdict(lambda: deque(maxlen=self.history_len))

    def update(self, entity_id: str, timestamp: float, position, velocity, visible: bool = True) -> None:
        if not visible:
            return
        timestamp = float(timestamp)
        q = self._samples[str(entity_id)]
        if q and timestamp <= q[-1].timestamp:
            raise ValueError("timestamps must be strictly increasing")
        q.append(_Sample(timestamp, _vec2(position).copy(), _vec2(velocity).copy()))

    def update_if_new(self, entity_id: str, timestamp: float, position, velocity, visible: bool = True) -> bool:
        if not visible:
            return False
        q = self._samples.get(str(entity_id))
        timestamp = float(timestamp)
        if q and timestamp <= q[-1].timestamp + 1e-12:
            return False
        self.update(entity_id, timestamp, position, velocity, visible=True)
        return True

    def sequence(self, entity_id: str, now: float) -> HistorySequence:
        now = float(now)
        q = self._samples.get(str(entity_id))
        features = np.zeros((self.history_len, 5), dtype=np.float32)
        mask = np.zeros((self.history_len,), dtype=np.bool_)
        if not q:
            return HistorySequence(features, mask, 0.0)
        if now < q[-1].timestamp - 1e-12:
            raise ValueError("now precedes latest observation")
        samples = list(q)[-self.history_len :]
        start = self.history_len - len(samples)
        for offset, sample in enumerate(samples):
            idx = start + offset
            features[idx, :2] = sample.position
            features[idx, 2:4] = sample.velocity
            features[idx, 4] = float(sample.timestamp)
            mask[idx] = True
        return HistorySequence(features, mask, max(0.0, now - q[-1].timestamp))
