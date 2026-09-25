from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


FEATURE_DIM = 14
ENTITY_TYPE = {"human": 0.0, "amr": 1.0}


def _vec2(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"expected 2-vector, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite vector")
    return arr


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _sigmoid(x: float) -> float:
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def compute_ttc_cpa(rel_pos, rel_vel, horizon: float = 4.0, eps: float = 1e-9):
    """Return finite-horizon time and distance at closest point of approach.

    Inputs are relative state in the ego frame: p_other - p_ego and
    v_other - v_ego.  The result is causal and uses only the supplied state.
    """
    p = _vec2(rel_pos)
    v = _vec2(rel_vel)
    horizon = max(0.0, float(horizon))
    vv = float(np.dot(v, v))
    if vv <= eps or horizon <= 0.0:
        t_cpa = 0.0
    else:
        t_cpa = float(np.clip(-float(np.dot(p, v)) / (vv + eps), 0.0, horizon))
    d_cpa = float(np.linalg.norm(p + v * t_cpa))
    return t_cpa, d_cpa


def compute_risk_features(
    rel_pos,
    rel_vel,
    *,
    horizon: float = 4.0,
    tau: float = 1.5,
    d_safe: float = 0.80,
    distance_scale: float = 0.20,
):
    """Analytic proximity/closing risk used as an attention feature.

    Risk is intentionally not a reward and does not itself command motion.
    It combines finite-horizon CPA timing and distance using bounded terms.
    """
    t_cpa, d_cpa = compute_ttc_cpa(rel_pos, rel_vel, horizon=horizon)
    tau = max(float(tau), 1e-6)
    scale = max(float(distance_scale), 1e-6)
    temporal = math.exp(-t_cpa / tau)
    spatial = _sigmoid((float(d_safe) - d_cpa) / scale)
    risk = float(np.clip(temporal * spatial, 0.0, 1.0))
    return {"t_cpa": t_cpa, "d_cpa": d_cpa, "ttc_risk": risk}


@dataclass(frozen=True)
class HistoryKinematics:
    acceleration: np.ndarray
    heading_rate: float
    observation_age: float


@dataclass(frozen=True)
class _HistorySample:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray


class ObservationHistory:
    """Per-entity causal state history with strict monotonic timestamps."""

    def __init__(self, maxlen: int = 8):
        if maxlen < 2:
            raise ValueError("maxlen must be >= 2")
        self.maxlen = int(maxlen)
        self._samples = defaultdict(lambda: deque(maxlen=self.maxlen))

    def update(self, entity_id: str, timestamp: float, position, velocity) -> None:
        timestamp = float(timestamp)
        q = self._samples[str(entity_id)]
        if q and timestamp <= q[-1].timestamp:
            raise ValueError("future observation or non-monotonic timestamp would break causality")
        q.append(_HistorySample(timestamp, _vec2(position).copy(), _vec2(velocity).copy()))

    def kinematics(self, entity_id: str, now: float) -> HistoryKinematics:
        q = self._samples.get(str(entity_id))
        now = float(now)
        if not q:
            return HistoryKinematics(np.zeros(2, dtype=np.float64), 0.0, 0.0)
        if now < q[-1].timestamp:
            raise ValueError("now precedes latest observation")
        age = now - q[-1].timestamp
        if len(q) < 2:
            return HistoryKinematics(np.zeros(2, dtype=np.float64), 0.0, age)
        prev, curr = q[-2], q[-1]
        dt = curr.timestamp - prev.timestamp
        if dt <= 0.0:
            return HistoryKinematics(np.zeros(2, dtype=np.float64), 0.0, age)
        acceleration = (curr.velocity - prev.velocity) / dt
        prev_speed = float(np.linalg.norm(prev.velocity))
        curr_speed = float(np.linalg.norm(curr.velocity))
        if prev_speed <= 1e-8 or curr_speed <= 1e-8:
            heading_rate = 0.0
        else:
            prev_heading = math.atan2(float(prev.velocity[1]), float(prev.velocity[0]))
            curr_heading = math.atan2(float(curr.velocity[1]), float(curr.velocity[0]))
            heading_rate = _wrap(curr_heading - prev_heading) / dt
        return HistoryKinematics(acceleration.astype(np.float64), float(heading_rate), float(age))


@dataclass(frozen=True)
class EntityObservation:
    entity_id: str
    entity_type: str
    position: np.ndarray
    velocity: np.ndarray
    timestamp: float
    visible: bool = True

    def __post_init__(self):
        if self.entity_type not in ENTITY_TYPE:
            raise ValueError(f"unknown entity type: {self.entity_type}")
        object.__setattr__(self, "position", _vec2(self.position).copy())
        object.__setattr__(self, "velocity", _vec2(self.velocity).copy())
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True)
class EntityBatch:
    features: np.ndarray
    mask: np.ndarray
    entity_ids: tuple[str, ...]
    feature_dim: int = FEATURE_DIM


def _history_kinematics(history, entity_id: str, now: float, timestamp: float):
    if history is None:
        return HistoryKinematics(np.zeros(2, dtype=np.float64), 0.0, max(0.0, now - timestamp))
    if isinstance(history, ObservationHistory):
        return history.kinematics(entity_id, now)
    if isinstance(history, Mapping):
        item = history.get(entity_id)
        if item is None:
            return HistoryKinematics(np.zeros(2, dtype=np.float64), 0.0, max(0.0, now - timestamp))
        return item
    raise TypeError("history must be ObservationHistory, mapping, or None")


def build_entity_batch(
    ego_pos,
    ego_vel,
    entities: Iterable[EntityObservation],
    *,
    now: float,
    history: ObservationHistory | Mapping[str, HistoryKinematics] | None = None,
    horizon: float = 4.0,
) -> EntityBatch:
    """Build deterministic variable-length entity features for attention models."""
    ego_pos = _vec2(ego_pos)
    ego_vel = _vec2(ego_vel)
    now = float(now)
    rows = []
    for obs in entities:
        if obs.timestamp > now + 1e-9:
            raise ValueError("future-state observation is not allowed")
        rel_pos = obs.position - ego_pos
        rel_vel = obs.velocity - ego_vel
        risk = compute_risk_features(rel_pos, rel_vel, horizon=horizon)
        kin = _history_kinematics(history, obs.entity_id, now, obs.timestamp)
        distance = float(np.linalg.norm(rel_pos))
        feature = np.array(
            [
                rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1],
                ENTITY_TYPE[obs.entity_type], distance,
                risk["t_cpa"], risk["d_cpa"], risk["ttc_risk"],
                kin.acceleration[0], kin.acceleration[1], kin.heading_rate,
                max(0.0, float(kin.observation_age)), float(obs.visible),
            ],
            dtype=np.float32,
        )
        rows.append((obs.entity_id, feature, bool(obs.visible), risk["ttc_risk"], distance))

    rows.sort(key=lambda item: (-item[3], item[4], item[0]))
    if rows:
        features = np.stack([x[1] for x in rows]).astype(np.float32, copy=False)
        mask = np.asarray([x[2] for x in rows], dtype=np.bool_)
        ids = tuple(x[0] for x in rows)
    else:
        features = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((0,), dtype=np.bool_)
        ids = tuple()
    return EntityBatch(features=features, mask=mask, entity_ids=ids)
