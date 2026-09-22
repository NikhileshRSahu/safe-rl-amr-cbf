from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from classical_baseline import AStarPlanner, physical_to_normalized_action
from benchmark.train_multi_agent_research import (
    DT, WORLD, ROBOT_R, VMAX, WMAX, SHELVES, wrap,
)
from benchmark.orca_geometry import OrcaLine, build_orca_line, satisfies_orca_line
from benchmark.dd_motion import (
    simulate_unicycle_arc,
    command_terminal_velocity,
    reachable_commands,
    arc_static_clearance,
)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


@dataclass
class BeastORCAConfig:
    time_horizon: float = 2.5
    neighbor_distance: float = 4.0
    peer_margin: float = 0.08
    human_margin: float = 0.12
    static_margin: float = 0.08
    planning_margin: float = 0.10
    lookahead_distance: float = 1.6
    waypoint_tolerance: float = 0.45
    replan_interval: int = 15
    preferred_speed: float = 1.0
    command_speed_samples: int = 7
    command_omega_samples: int = 15
    smoothness_weight: float = 0.12
    progress_weight: float = 4.0
    stuck_progress_eps: float = 0.01
    stuck_ticks: int = 22
    recovery_ticks: int = 20
    arc_horizon: float = 0.8


class AStarORCADD:
    def __init__(self, world, i: int, config: BeastORCAConfig | None = None):
        self.i = int(i)
        self.cfg = config or BeastORCAConfig()
        self.planner = AStarPlanner(
            map_bounds=(-WORLD, WORLD, -WORLD, WORLD),
            shelves=SHELVES,
            robot_radius=ROBOT_R,
            margin=self.cfg.planning_margin,
            resolution=0.25,
        )
        self.goal = np.asarray(world.g[self.i], dtype=float)
        self.path = self.planner.plan(tuple(world.p[self.i]), tuple(self.goal)) or [
            tuple(world.p[self.i]), tuple(self.goal)
        ]
        self.idx = 0
        self.ticks = 0
        self.stuck = 0
        self.recovery_left = 0
        self.last_goal_dist = float(np.linalg.norm(self.goal - world.p[self.i]))
        self.prev_cmd = np.array([0.0, 0.0], dtype=float)
        self._diag = dict(
            orca_constraints_total=0,
            infeasible_command_events=0,
            stop_yield_ticks=0,
            replan_count=0,
            recovery_count=0,
            candidate_commands_evaluated=0,
        )

    def diagnostics(self):
        return dict(self._diag)

    def _replan(self, w):
        path = self.planner.plan(tuple(w.p[self.i]), tuple(self.goal))
        if path:
            self.path = path
            self.idx = 0
        self._diag["replan_count"] += 1

    def _target(self, p):
        if not self.path:
            return self.goal.copy()
        while (
            self.idx < len(self.path) - 1
            and np.linalg.norm(np.asarray(self.path[self.idx]) - p)
            <= self.cfg.waypoint_tolerance
        ):
            self.idx += 1
        j = self.idx
        acc = 0.0
        while j + 1 < len(self.path) and acc < self.cfg.lookahead_distance:
            a = np.asarray(self.path[j], dtype=float)
            b = np.asarray(self.path[j + 1], dtype=float)
            acc += float(np.linalg.norm(b - a))
            j += 1
        return np.asarray(self.path[j], dtype=float)

    def _velocity(self, w, j):
        return np.array(
            [
                math.cos(float(w.th[j])) * float(w.v[j]),
                math.sin(float(w.th[j])) * float(w.v[j]),
            ],
            dtype=float,
        )

    def _absolute_orca_lines(self, w, current_vel):
        i = self.i
        p = np.asarray(w.p[i], dtype=float)
        lines = []

        for j in range(w.n):
            if j == i or w.done[j]:
                continue
            q = np.asarray(w.p[j], dtype=float)
            delta = q - p
            if float(np.linalg.norm(delta)) > self.cfg.neighbor_distance:
                continue
            other = self._velocity(w, j)
            rel = other - current_vel
            responsibility = float(
                np.clip(0.65 - 0.30 * float(w.priority[i]), 0.35, 0.65)
            )
            relative_line = build_orca_line(
                delta,
                rel,
                2 * ROBOT_R + self.cfg.peer_margin,
                self.cfg.time_horizon,
                responsibility,
            )
            lines.append(
                OrcaLine(
                    point=other - relative_line.point,
                    normal=-relative_line.normal,
                )
            )

        for j in range(w.nppl):
            q = np.asarray(w.hp[j], dtype=float)
            delta = q - p
            if float(np.linalg.norm(delta)) > self.cfg.neighbor_distance:
                continue
            other = np.asarray(w.hv[j], dtype=float)
            rel = other - current_vel
            relative_line = build_orca_line(
                delta,
                rel,
                2 * ROBOT_R + self.cfg.human_margin,
                self.cfg.time_horizon,
                1.0,
            )
            lines.append(
                OrcaLine(
                    point=other - relative_line.point,
                    normal=-relative_line.normal,
                )
            )

        self._diag["orca_constraints_total"] += len(lines)
        return lines

    def _arc_dynamic_safe(self, w, arc):
        times = np.arange(len(arc), dtype=float) * DT
        for k, qpose in enumerate(arc):
            q = qpose[:2]
            t = times[k]
            for j in range(w.n):
                if j == self.i or w.done[j]:
                    continue
                other = np.asarray(w.p[j], dtype=float) + self._velocity(w, j) * t
                if (
                    np.linalg.norm(q - other)
                    < 2 * ROBOT_R + self.cfg.peer_margin
                ):
                    return False
            for j in range(w.nppl):
                other = (
                    np.asarray(w.hp[j], dtype=float)
                    + np.asarray(w.hv[j], dtype=float) * t
                )
                if (
                    np.linalg.norm(q - other)
                    < 2 * ROBOT_R + self.cfg.human_margin
                ):
                    return False
        return True

    def _preferred(self, w, p, target):
        desired = target - p
        if self.recovery_left > 0:
            direction = _unit(desired)
            side = 1.0 if float(w.priority[self.i]) >= 0.5 else -1.0
            direction = _unit(
                0.65 * direction
                + 0.75 * np.array([-direction[1], direction[0]]) * side
            )
            desired = direction
            self.recovery_left -= 1

        heading = (
            math.atan2(float(desired[1]), float(desired[0]))
            if np.linalg.norm(desired) > 1e-9
            else float(w.th[self.i])
        )
        heading_error = wrap(heading - float(w.th[self.i]))
        omega = float(np.clip(2.2 * heading_error, -WMAX, WMAX))
        speed = min(
            self.cfg.preferred_speed,
            VMAX,
            max(0.2, float(np.linalg.norm(self.goal - p))),
        )
        speed *= max(
            0.12,
            math.cos(min(abs(heading_error), math.pi / 2)),
        )
        preferred_velocity = np.array(
            [math.cos(heading) * speed, math.sin(heading) * speed],
            dtype=float,
        )
        return speed, omega, preferred_velocity

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

        if self.stuck >= self.cfg.stuck_ticks and self.recovery_left <= 0:
            self._replan(w)
            self.recovery_left = self.cfg.recovery_ticks
            self.stuck = 0
            self._diag["recovery_count"] += 1

        current_vel = self._velocity(w, i)
        lines = self._absolute_orca_lines(w, current_vel)
        preferred_speed, preferred_omega, preferred_velocity = self._preferred(
            w, p, target
        )

        commands = reachable_commands(
            float(w.v[i]),
            float(w.w[i]),
            preferred_speed=preferred_speed,
            preferred_omega=preferred_omega,
            v_max=VMAX,
            w_max=WMAX,
            speed_samples=self.cfg.command_speed_samples,
            omega_samples=self.cfg.command_omega_samples,
        )

        best = None
        best_score = -float("inf")
        for v_cmd, omega_cmd in commands:
            self._diag["candidate_commands_evaluated"] += 1

            arc = simulate_unicycle_arc(
                np.array([p[0], p[1], float(w.th[i])]),
                v_cmd,
                omega_cmd,
                DT,
                self.cfg.arc_horizon,
            )
            if (
                arc_static_clearance(
                    arc,
                    SHELVES,
                    WORLD,
                    ROBOT_R,
                )
                < self.cfg.static_margin
            ):
                continue

            realized_velocity = command_terminal_velocity(
                float(w.th[i]),
                v_cmd,
                omega_cmd,
                DT,
            )
            if any(
                not satisfies_orca_line(realized_velocity, line, 1e-7)
                for line in lines
            ):
                continue
            if not self._arc_dynamic_safe(w, arc):
                continue

            end = arc[-1, :2]
            progress_score = float(
                np.linalg.norm(self.goal - p) - np.linalg.norm(self.goal - end)
            )
            preferred_error = float(
                np.linalg.norm(realized_velocity - preferred_velocity)
            )
            smoothness = (
                abs(v_cmd - float(self.prev_cmd[0]))
                + 0.25 * abs(omega_cmd - float(self.prev_cmd[1]))
            )
            score = (
                self.cfg.progress_weight * progress_score
                - 1.2 * preferred_error
                - self.cfg.smoothness_weight * smoothness
                + 0.08 * v_cmd
            )
            if best is None or score > best_score + 1e-12:
                best = (v_cmd, omega_cmd)
                best_score = score

        if best is None:
            self._diag["infeasible_command_events"] += 1
            best = (0.0, 0.0)

        if best[0] < 0.05:
            self._diag["stop_yield_ticks"] += 1

        self.prev_cmd[:] = best
        return physical_to_normalized_action(
            best[0],
            best[1],
            v_min=0.0,
            v_max=VMAX,
            omega_max=WMAX,
        )


AStarReciprocalVO = AStarORCADD
BeastRVOConfig = BeastORCAConfig
