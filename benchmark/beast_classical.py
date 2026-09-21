from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from classical_baseline import AStarPlanner, physical_to_normalized_action
from benchmark.train_multi_agent_research import (
    DT, WORLD, ROBOT_R, VMAX, WMAX, SHELVES, wrap,
)

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)

def _point_rect_distance(x: float, y: float, rect) -> float:
    xmin, ymin, xmax, ymax = rect
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)

@dataclass
class BeastRVOConfig:
    horizon: float = 2.0
    peer_margin: float = 0.12
    human_margin: float = 0.15
    static_margin: float = 0.10
    lookahead_distance: float = 1.8
    waypoint_tolerance: float = 0.55
    replan_interval: int = 18
    speed_samples: int = 7
    angle_samples: int = 13
    max_heading_offset: float = math.pi * 0.95
    stuck_progress_eps: float = 0.015
    stuck_ticks: int = 24
    escape_ticks: int = 28
    preferred_speed: float = 1.0

class AStarReciprocalVO:
    """
    Strong classical stack for the thesis benchmark.

    Global: inflated-grid A*.
    Local: reciprocal velocity-obstacle style velocity search in global velocity
    space with analytical time-to-closest-approach checks against controlled AMRs
    and pedestrians.
    Liveness: priority-aware yielding, periodic replanning, and deterministic
    deadlock escape.

    This is intentionally stronger than the earlier sampled unicycle "VO-style"
    baseline. It still remains a decentralized classical controller: there is no
    learned policy and no access to future ground-truth trajectories.
    """

    def __init__(self, world, i: int, config: BeastRVOConfig | None = None):
        self.i = int(i)
        self.cfg = config or BeastRVOConfig()
        self.planner = AStarPlanner(
            map_bounds=(-WORLD, WORLD, -WORLD, WORLD),
            shelves=SHELVES,
            robot_radius=ROBOT_R,
            margin=self.cfg.static_margin,
            resolution=0.25,
        )
        self.goal = np.asarray(world.g[self.i], dtype=float)
        self.path = self.planner.plan(tuple(world.p[self.i]), tuple(self.goal))
        self.idx = 0
        self.ticks = 0
        self.stuck = 0
        self.escape_left = 0
        self.last_goal_dist = float(np.linalg.norm(self.goal - world.p[self.i]))
        self.prev_cmd = np.array([0.0, 0.0], dtype=float)

    def _replan(self, w):
        new_path = self.planner.plan(tuple(w.p[self.i]), tuple(self.goal))
        if new_path:
            self.path = new_path
            self.idx = 0

    def _target(self, p: np.ndarray) -> np.ndarray:
        if not self.path:
            return self.goal.copy()
        while self.idx < len(self.path) - 1:
            q = np.asarray(self.path[self.idx], dtype=float)
            if np.linalg.norm(q - p) > self.cfg.waypoint_tolerance:
                break
            self.idx += 1
        j = self.idx
        acc = 0.0
        while j + 1 < len(self.path) and acc < self.cfg.lookahead_distance:
            a = np.asarray(self.path[j], dtype=float)
            b = np.asarray(self.path[j + 1], dtype=float)
            acc += float(np.linalg.norm(b - a))
            j += 1
        return np.asarray(self.path[j], dtype=float)

    def _static_clearance_linear(self, p: np.ndarray, vel: np.ndarray) -> float:
        minc = float("inf")
        steps = 10
        for k in range(1, steps + 1):
            t = self.cfg.horizon * k / steps
            q = p + vel * t
            wall = min(
                q[0] - (-WORLD) - ROBOT_R,
                WORLD - q[0] - ROBOT_R,
                q[1] - (-WORLD) - ROBOT_R,
                WORLD - q[1] - ROBOT_R,
            )
            minc = min(minc, float(wall))
            if wall < self.cfg.static_margin:
                return minc
            for r in SHELVES:
                c = _point_rect_distance(float(q[0]), float(q[1]), r) - ROBOT_R
                minc = min(minc, c)
                if c < self.cfg.static_margin:
                    return minc
        return minc

    @staticmethod
    def _closest_approach(rel_p: np.ndarray, rel_v: np.ndarray, horizon: float) -> tuple[float, float]:
        vv = float(np.dot(rel_v, rel_v))
        if vv < 1e-9:
            t = 0.0
        else:
            t = float(np.clip(-np.dot(rel_p, rel_v) / vv, 0.0, horizon))
        d = float(np.linalg.norm(rel_p + rel_v * t))
        return t, d

    def _dynamic_clearance(self, w, p: np.ndarray, vel: np.ndarray) -> tuple[float, bool]:
        minc = float("inf")

        # Reciprocal AMRs: each controller owns half of the avoidance response.
        for j in range(w.n):
            if j == self.i or w.done[j]:
                continue
            q = np.asarray(w.p[j], dtype=float)
            vj = np.array([
                math.cos(float(w.th[j])) * float(w.v[j]),
                math.sin(float(w.th[j])) * float(w.v[j]),
            ], dtype=float)
            _, center_d = self._closest_approach(q - p, vj - vel, self.cfg.horizon)
            clearance = center_d - 2.0 * ROBOT_R
            minc = min(minc, clearance)

            # Priority only changes how conservative the yielding robot is; it
            # never disables the hard collision check.
            my_pr = float(w.priority[self.i])
            other_pr = float(w.priority[j])
            reciprocal_margin = self.cfg.peer_margin + (0.05 if my_pr < other_pr else 0.0)
            if clearance < reciprocal_margin:
                return minc, False

        # Pedestrians are non-reciprocal: the AMR takes full responsibility.
        for j in range(w.nppl):
            q = np.asarray(w.hp[j], dtype=float)
            vj = np.asarray(w.hv[j], dtype=float)
            _, center_d = self._closest_approach(q - p, vj - vel, self.cfg.horizon)
            clearance = center_d - 2.0 * ROBOT_R
            minc = min(minc, clearance)
            if clearance < self.cfg.human_margin:
                return minc, False

        return minc, True

    def _candidate_velocities(self, preferred: np.ndarray, current_heading: float):
        speed_pref = min(self.cfg.preferred_speed, float(np.linalg.norm(preferred)))
        pref_angle = math.atan2(float(preferred[1]), float(preferred[0])) if speed_pref > 1e-6 else current_heading

        # Dense samples around preferred velocity, with explicit stop candidate.
        speeds = np.linspace(0.0, VMAX, self.cfg.speed_samples)
        offsets = np.linspace(-self.cfg.max_heading_offset, self.cfg.max_heading_offset, self.cfg.angle_samples)
        yield np.zeros(2, dtype=float)
        for speed in speeds[1:]:
            for off in offsets:
                ang = pref_angle + float(off)
                yield np.array([math.cos(ang) * speed, math.sin(ang) * speed], dtype=float)

    def _escape_velocity(self, w, p: np.ndarray, target: np.ndarray) -> np.ndarray:
        desired = _unit(target - p)
        # Deterministic right/left passing convention derived from right-of-way
        # priority prevents two peers from choosing the same evasive side.
        side = 1.0 if float(w.priority[self.i]) >= 0.5 else -1.0
        lateral = np.array([-desired[1], desired[0]]) * side
        v = 0.45 * desired + 0.55 * lateral
        return _unit(v) * 0.55

    def action(self, w, i=None):
        i = self.i
        if w.done[i]:
            return np.array([-1.0, 0.0], dtype=np.float32)

        self.ticks += 1
        p = np.asarray(w.p[i], dtype=float)
        target = self._target(p)

        goal_dist = float(np.linalg.norm(self.goal - p))
        progress = self.last_goal_dist - goal_dist
        self.last_goal_dist = goal_dist
        if progress < self.cfg.stuck_progress_eps and goal_dist > 0.7:
            self.stuck += 1
        else:
            self.stuck = max(0, self.stuck - 2)

        if self.ticks % self.cfg.replan_interval == 0:
            self._replan(w)

        if self.stuck >= self.cfg.stuck_ticks and self.escape_left <= 0:
            self.escape_left = self.cfg.escape_ticks
            self.stuck = 0
            self._replan(w)

        desired_dir = _unit(target - p)
        preferred = desired_dir * min(VMAX, max(0.30, goal_dist))

        if self.escape_left > 0:
            preferred = self._escape_velocity(w, p, target)
            self.escape_left -= 1

        current_heading = float(w.th[i])
        current_vel = np.array([
            math.cos(current_heading) * float(w.v[i]),
            math.sin(current_heading) * float(w.v[i]),
        ], dtype=float)

        best_score = -float("inf")
        best_vel = np.zeros(2, dtype=float)
        best_clear = -float("inf")

        for vel in self._candidate_velocities(preferred, current_heading):
            static_c = self._static_clearance_linear(p, vel)
            if static_c < self.cfg.static_margin:
                continue
            dyn_c, dyn_safe = self._dynamic_clearance(w, p, vel)
            if not dyn_safe:
                continue

            speed = float(np.linalg.norm(vel))
            if speed > 1e-6:
                cand_dir = vel / speed
            else:
                cand_dir = np.zeros(2)

            pref_err = float(np.linalg.norm(vel - preferred))
            progress_score = float(np.dot(cand_dir, desired_dir)) * speed
            clearance_score = min(static_c, dyn_c, 1.5)
            smooth = float(np.linalg.norm(vel - current_vel))

            # Lower-priority robot yields slightly more when another AMR is
            # close; higher-priority robot is encouraged to clear the conflict.
            close_peers = [
                j for j in range(w.n)
                if j != i and not w.done[j] and np.linalg.norm(w.p[j] - w.p[i]) < 1.8
            ]
            priority_drive = 0.0
            if close_peers:
                priority_drive = (float(w.priority[i]) - 0.5) * 0.35 * speed

            score = (
                3.6 * progress_score
                - 1.8 * pref_err
                + 0.75 * clearance_score
                + 0.28 * speed
                - 0.18 * smooth
                + priority_drive
            )
            if score > best_score:
                best_score = score
                best_vel = vel
                best_clear = clearance_score

        speed = float(np.linalg.norm(best_vel))
        if speed < 1e-5:
            v_cmd = 0.0
            # Rotate toward path even when translation is blocked. This avoids
            # repeatedly selecting a zero command in narrow conflict states.
            heading_error = wrap(math.atan2(target[1] - p[1], target[0] - p[0]) - current_heading)
            omega_cmd = float(np.clip(1.8 * heading_error, -WMAX, WMAX))
        else:
            desired_heading = math.atan2(float(best_vel[1]), float(best_vel[0]))
            heading_error = wrap(desired_heading - current_heading)
            omega_cmd = float(np.clip(2.4 * heading_error, -WMAX, WMAX))
            # Unicycle cannot instantly realize a lateral velocity. Reduce
            # forward speed while turning rather than violating the candidate.
            v_cmd = speed * max(0.08, math.cos(min(abs(heading_error), math.pi / 2)))
            v_cmd = float(np.clip(v_cmd, 0.0, VMAX))

        self.prev_cmd[:] = (v_cmd, omega_cmd)
        return physical_to_normalized_action(
            v_cmd, omega_cmd, v_min=0.0, v_max=VMAX, omega_max=WMAX
        )
