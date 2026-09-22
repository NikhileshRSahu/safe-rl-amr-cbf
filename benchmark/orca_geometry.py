from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class OrcaLine:
    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.point, dtype=float)
        n = np.asarray(self.normal, dtype=float)
        norm = float(np.linalg.norm(n))
        if norm <= _EPS:
            raise ValueError("ORCA line normal must be non-zero")
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "normal", n / norm)


def _det(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def build_orca_line(
    rel_position: np.ndarray,
    rel_velocity: np.ndarray,
    combined_radius: float,
    time_horizon: float,
    responsibility: float,
) -> OrcaLine:
    """Build an ORCA half-plane in other-minus-self relative velocity space."""
    p = np.asarray(rel_position, dtype=float)
    rv_other_minus_self = np.asarray(rel_velocity, dtype=float)
    if p.shape != (2,) or rv_other_minus_self.shape != (2,):
        raise ValueError("rel_position and rel_velocity must be 2-D vectors")
    if combined_radius <= 0 or time_horizon <= 0:
        raise ValueError("combined_radius and time_horizon must be positive")
    responsibility = float(np.clip(responsibility, 0.0, 1.0))

    v = -rv_other_minus_self
    dist_sq = float(np.dot(p, p))
    r_sq = float(combined_radius * combined_radius)
    inv_tau = 1.0 / float(time_horizon)

    if dist_sq > r_sq:
        w = v - inv_tau * p
        w_len_sq = float(np.dot(w, w))
        dot_wp = float(np.dot(w, p))
        if dot_wp < 0.0 and dot_wp * dot_wp > r_sq * w_len_sq:
            w_len = math.sqrt(max(w_len_sq, _EPS))
            unit_w = w / w_len
            direction = np.array([unit_w[1], -unit_w[0]], dtype=float)
            u = (combined_radius * inv_tau - w_len) * unit_w
        else:
            leg = math.sqrt(max(0.0, dist_sq - r_sq))
            if _det(p, w) > 0.0:
                direction = np.array(
                    [
                        p[0] * leg - p[1] * combined_radius,
                        p[0] * combined_radius + p[1] * leg,
                    ],
                    dtype=float,
                ) / dist_sq
            else:
                direction = -np.array(
                    [
                        p[0] * leg + p[1] * combined_radius,
                        -p[0] * combined_radius + p[1] * leg,
                    ],
                    dtype=float,
                ) / dist_sq
            u = float(np.dot(v, direction)) * direction - v
    else:
        inv_dt = 1.0 / min(float(time_horizon), 0.1)
        w = v - inv_dt * p
        w_len = float(np.linalg.norm(w))
        unit_w = w / w_len if w_len > _EPS else -p / max(float(np.linalg.norm(p)), _EPS)
        direction = np.array([unit_w[1], -unit_w[0]], dtype=float)
        u = (combined_radius * inv_dt - w_len) * unit_w

    std_point = v + responsibility * u
    std_normal = np.array([-direction[1], direction[0]], dtype=float)
    return OrcaLine(point=-std_point, normal=-std_normal)


def satisfies_orca_line(velocity: np.ndarray, line: OrcaLine, tol: float = 1e-9) -> bool:
    v = np.asarray(velocity, dtype=float)
    return float(np.dot(v - line.point, line.normal)) >= -float(tol)


def _feasible(v: np.ndarray, lines: Iterable[OrcaLine], max_speed: float, tol: float = 1e-8) -> bool:
    if float(np.dot(v, v)) > max_speed * max_speed + tol:
        return False
    return all(satisfies_orca_line(v, line, tol) for line in lines)


def _line_circle_intersections(line: OrcaLine, radius: float):
    n = line.normal
    tangent = np.array([-n[1], n[0]], dtype=float)
    closest = line.point - float(np.dot(line.point, n)) * n
    d2 = float(np.dot(closest, closest))
    r2 = radius * radius
    if d2 > r2 + 1e-10:
        return []
    span = math.sqrt(max(0.0, r2 - d2))
    return [closest + span * tangent, closest - span * tangent]


def _line_line_intersection(a: OrcaLine, b: OrcaLine):
    matrix = np.vstack([a.normal, b.normal])
    if abs(float(np.linalg.det(matrix))) <= 1e-10:
        return None
    rhs = np.array(
        [np.dot(a.point, a.normal), np.dot(b.point, b.normal)],
        dtype=float,
    )
    return np.linalg.solve(matrix, rhs)


def project_velocity_to_orca_halfplanes(
    preferred_velocity: np.ndarray,
    lines: Iterable[OrcaLine],
    max_speed: float,
) -> np.ndarray | None:
    """Return the feasible velocity nearest preferred_velocity."""
    if max_speed < 0:
        raise ValueError("max_speed must be non-negative")
    lines = list(lines)
    pref = np.asarray(preferred_velocity, dtype=float)
    norm = float(np.linalg.norm(pref))
    clipped = pref if norm <= max_speed or norm <= _EPS else pref * (max_speed / norm)

    candidates = [clipped, np.zeros(2, dtype=float)]
    for line in lines:
        signed = float(np.dot(pref - line.point, line.normal))
        candidates.append(pref - signed * line.normal)
        candidates.extend(_line_circle_intersections(line, max_speed))
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            point = _line_line_intersection(lines[i], lines[j])
            if point is not None:
                candidates.append(point)

    feasible = [
        np.asarray(v, dtype=float)
        for v in candidates
        if _feasible(np.asarray(v, dtype=float), lines, max_speed)
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda v: (
            float(np.dot(v - pref, v - pref)),
            float(v[0]),
            float(v[1]),
        ),
    ).copy()
