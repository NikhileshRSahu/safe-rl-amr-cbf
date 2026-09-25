from __future__ import annotations

import math
import numpy as np

from benchmark.beast_classical import AStarORCADD, BeastORCAConfig
from benchmark.train_multi_agent_research import ROBOT_R, VMAX


def finite_horizon_collision(
    ego_pos,
    ego_vel,
    other_pos,
    other_vel,
    combined_radius: float,
    horizon: float,
) -> bool:
    """Return whether constant-velocity discs collide within a finite horizon.

    This is the time-domain equivalent membership test for a finite-horizon
    velocity obstacle.  The closest-approach time is explicitly clamped to
    [0, horizon], which avoids treating a collision far beyond the planning
    horizon as immediately forbidden.
    """
    p = np.asarray(other_pos, dtype=float) - np.asarray(ego_pos, dtype=float)
    rel = np.asarray(other_vel, dtype=float) - np.asarray(ego_vel, dtype=float)
    radius = max(0.0, float(combined_radius))
    tau = max(0.0, float(horizon))

    if float(np.dot(p, p)) <= radius * radius:
        return True

    rel_sq = float(np.dot(rel, rel))
    if rel_sq <= 1e-12 or tau <= 0.0:
        return False

    t_star = -float(np.dot(p, rel)) / rel_sq
    t_star = min(tau, max(0.0, t_star))
    closest = p + rel * t_star
    return float(np.dot(closest, closest)) <= radius * radius


def rvo_peer_effective_velocity(candidate_velocity, current_ego_velocity):
    """Map an RVO candidate to the equivalent full-responsibility VO velocity.

    For the classic reciprocal velocity obstacle,
        v_candidate is forbidden iff (2*v_candidate - v_ego_current)
    lies inside the ordinary velocity obstacle induced by the peer.
    """
    candidate = np.asarray(candidate_velocity, dtype=float)
    current = np.asarray(current_ego_velocity, dtype=float)
    return 2.0 * candidate - current


def _clip_velocity(v, max_speed: float = VMAX):
    v = np.asarray(v, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed <= max_speed + 1e-12:
        return v.copy()
    if speed <= 1e-12:
        return np.zeros(2, dtype=float)
    return v * (max_speed / speed)


def _deterministic_velocity_candidates(preferred, current):
    """Return a reproducible velocity set ordered later by target cost."""
    values = [
        _clip_velocity(preferred),
        _clip_velocity(current),
        np.zeros(2, dtype=float),
    ]

    # Dense enough to expose VO/RVO behavior without turning the baseline into
    # an optimizer with materially more sophistication than the original
    # velocity-obstacle methods.
    for speed in np.linspace(0.15, VMAX, 7):
        for angle in np.linspace(-math.pi, math.pi, 36, endpoint=False):
            values.append(
                np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=float)
            )

    # Deduplicate deterministically at millimetre-per-second scale.
    unique = {}
    for v in values:
        key = tuple(np.round(v, 6))
        unique.setdefault(key, v)
    return list(unique.values())


class _AStarVelocityObstacleDD(AStarORCADD):
    """Common A* + velocity-obstacle target + DD realization baseline.

    The parent class supplies the exact same global planner, replanning,
    recovery logic, unicycle command realization, static arc checks, and
    dynamic arc verification used by Peak ORCA-DD.  Only the holonomic local
    target selector is replaced, keeping the progression comparison focused
    on VO/RVO/ORCA rather than on unrelated planner differences.
    """

    reciprocal_peers = False

    def __init__(self, world, i: int, config: BeastORCAConfig | None = None):
        super().__init__(world, i, config)
        self._world_ref = world
        self._diag.update(
            velocity_candidates_evaluated=0,
            velocity_target_infeasible_events=0,
        )

    def action(self, w, i=None):
        self._world_ref = w
        return super().action(w, i)

    def _absolute_orca_lines(self, w, current_vel):
        # Critical isolation rule: VO/RVO must never inherit ORCA half-plane
        # constraints from the parent implementation.
        return []

    def _peer_candidate_velocity(self, candidate, current_velocity):
        if self.reciprocal_peers:
            return rvo_peer_effective_velocity(candidate, current_velocity)
        return np.asarray(candidate, dtype=float)

    def _candidate_is_velocity_safe(self, w, candidate, current_velocity):
        p = np.asarray(w.p[self.i], dtype=float)

        for j in range(w.n):
            if j == self.i or w.done[j]:
                continue
            q = np.asarray(w.p[j], dtype=float)
            if float(np.linalg.norm(q - p)) > self.cfg.neighbor_distance:
                continue
            peer_velocity = self._velocity(w, j)
            tested_velocity = self._peer_candidate_velocity(candidate, current_velocity)
            if finite_horizon_collision(
                p,
                tested_velocity,
                q,
                peer_velocity,
                2.0 * ROBOT_R + self.cfg.peer_margin,
                self.cfg.time_horizon,
            ):
                return False

        # Humans are deliberately non-reciprocal for both VO and RVO.  This
        # avoids giving RVO the unrealistic assumption that a person will take
        # exactly half of the collision-avoidance responsibility.
        for j in range(w.nppl):
            q = np.asarray(w.hp[j], dtype=float)
            if float(np.linalg.norm(q - p)) > self.cfg.neighbor_distance:
                continue
            if finite_horizon_collision(
                p,
                candidate,
                q,
                np.asarray(w.hv[j], dtype=float),
                2.0 * ROBOT_R + self.cfg.human_margin,
                self.cfg.time_horizon,
            ):
                return False

        return True

    def _continuous_orca_target(self, preferred_velocity, lines):
        # The parent calls this hook after computing the preferred velocity.
        # For these classes it is a VO/RVO target selector, not ORCA.
        w = self._world_ref
        current = self._velocity(w, self.i)
        preferred = _clip_velocity(preferred_velocity)
        candidates = _deterministic_velocity_candidates(preferred, current)

        # Choose the feasible velocity nearest the preferred target, with a
        # small smoothness tie-break toward current velocity.
        candidates.sort(
            key=lambda v: (
                float(np.dot(v - preferred, v - preferred))
                + 0.04 * float(np.dot(v - current, v - current)),
                -float(np.linalg.norm(v)),
                float(math.atan2(v[1], v[0])) if np.linalg.norm(v) > 1e-12 else 0.0,
            )
        )

        for candidate in candidates:
            self._diag["velocity_candidates_evaluated"] += 1
            if self._candidate_is_velocity_safe(w, candidate, current):
                return np.asarray(candidate, dtype=float)

        self._diag["velocity_target_infeasible_events"] += 1
        return np.zeros(2, dtype=float)


class AStarVODD(_AStarVelocityObstacleDD):
    """A* + classic Velocity Obstacle + common differential-drive realization."""

    reciprocal_peers = False


class AStarRVODD(_AStarVelocityObstacleDD):
    """A* + classic Reciprocal Velocity Obstacle + common DD realization."""

    reciprocal_peers = True
