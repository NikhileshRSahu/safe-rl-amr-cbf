from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def segment_intersects_rect(a, b, rect, *, eps: float = 1e-9) -> bool:
    """Return True when closed segment a->b intersects an axis-aligned rectangle.

    Liang-Barsky clipping is used so the same deterministic geometry contract can
    be shared by RL perception and the analytical baseline.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x0, y0, x1, y1 = map(float, rect)
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    p = (-dx, dx, -dy, dy)
    q = (float(a[0] - x0), float(x1 - a[0]), float(a[1] - y0), float(y1 - a[1]))
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) <= eps:
            if qi < 0.0:
                return False
            continue
        r = qi / pi
        if pi < 0.0:
            if r > u1:
                return False
            u0 = max(u0, r)
        else:
            if r < u0:
                return False
            u1 = min(u1, r)
    return u0 <= u1 + eps


def visible_with_shelves(
    observer,
    target,
    shelves: Iterable[Sequence[float]],
    *,
    max_range: float,
    shelf_padding: float = 0.0,
) -> bool:
    """Causal warehouse visibility: range-limited and blocked by shelf bodies."""
    observer = np.asarray(observer, dtype=float)
    target = np.asarray(target, dtype=float)
    if not np.isfinite(observer).all() or not np.isfinite(target).all():
        return False
    if float(np.linalg.norm(target - observer)) > float(max_range) + 1e-9:
        return False

    pad = max(0.0, float(shelf_padding))
    for rect in shelves:
        x0, y0, x1, y1 = map(float, rect)
        expanded = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        if segment_intersects_rect(observer, target, expanded):
            return False
    return True


class LastSeenTracker:
    """Small causal tracker used by non-learning baselines during brief occlusion."""

    def __init__(self, max_age_seconds: float = 1.5):
        self.max_age_seconds = float(max_age_seconds)
        self._tracks: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    def observe(self, entity_id: int, position, velocity, now: float) -> None:
        self._tracks[int(entity_id)] = (
            np.asarray(position, dtype=np.float32).copy(),
            np.asarray(velocity, dtype=np.float32).copy(),
            float(now),
        )

    def predict(self, entity_id: int, now: float):
        track = self._tracks.get(int(entity_id))
        if track is None:
            return None
        position, velocity, seen_at = track
        age = max(0.0, float(now) - float(seen_at))
        if age > self.max_age_seconds:
            return None
        return (
            position + velocity * age,
            velocity.copy(),
            age,
        )
