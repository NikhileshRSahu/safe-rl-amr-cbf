from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from benchmark.beast_classical import AStarORCADD, BeastORCAConfig
from benchmark.orca_geometry import OrcaLine, build_orca_line
from benchmark.render_shared_success_case import (
    HUMAN_AMR_CLEARANCE,
    HUMAN_MAX_SPEED,
    HUMAN_MIN_SEPARATION,
    HUMAN_MIN_SPEED,
    RealisticHumanWorld,
    _rotate,
)
from benchmark.train_multi_agent_research import DT, ROBOT_R


TRAIN_SEEDS = tuple(range(12000, 12080))
VALIDATION_SEEDS = tuple(range(13000, 13020))
FINAL_SEEDS = tuple(range(14000, 14030))


@dataclass(frozen=True)
class AdaptiveORCAExtras:
    horizon_gain: float = 0.75
    margin_gain: float = 0.10
    neighbor_gain: float = 1.0
    max_human_time_horizon: float = 5.0
    max_human_margin_add: float = 0.12


class IntentShiftWorld(RealisticHumanWorld):
    """Physically valid pedestrians with latent, discontinuous intent changes.

    Human bodies still obey the realistic-human collision constraints. The
    difference is behavioral: pedestrians periodically stop, turn, reverse,
    or become non-yielding, making instantaneous velocity a less reliable
    predictor of near-future motion.
    """

    NORMAL = 0
    STOP = 1
    TURN = 2
    REVERSE = 3
    NONYIELD = 4

    def __init__(self, n_agents=4, n_people=12, seed=0):
        self.intent_mode = np.zeros(n_people, dtype=np.int8)
        self.human_uncertainty = np.zeros(n_people, dtype=np.float32)
        self._intent_ticks = np.zeros(n_people, dtype=np.int32)
        self._next_intent_tick = np.zeros(n_people, dtype=np.int32)
        self._intent_turn_angle = np.zeros(n_people, dtype=np.float32)
        self.intent_change_count = 0
        super().__init__(n_agents, n_people, seed)

    def reset(self):
        self.intent_mode = np.zeros(self.nppl, dtype=np.int8)
        self.human_uncertainty = np.zeros(self.nppl, dtype=np.float32)
        self._intent_ticks = np.zeros(self.nppl, dtype=np.int32)
        self._next_intent_tick = np.zeros(self.nppl, dtype=np.int32)
        self._intent_turn_angle = np.zeros(self.nppl, dtype=np.float32)
        self.intent_change_count = 0
        obs = super().reset()
        self._next_intent_tick = self.rng.integers(18, 46, size=self.nppl, dtype=np.int32)
        return obs

    def _activate_intent(self, i):
        mode = int(self.rng.choice(
            [self.STOP, self.TURN, self.REVERSE, self.NONYIELD],
            p=[0.34, 0.36, 0.14, 0.16],
        ))
        self.intent_mode[i] = mode
        self.intent_change_count += 1
        if mode == self.STOP:
            self._intent_ticks[i] = int(self.rng.integers(6, 21))
            self.human_uncertainty[i] = 0.65
        elif mode == self.TURN:
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            self._intent_turn_angle[i] = sign * float(self.rng.uniform(math.pi / 3, 2 * math.pi / 3))
            self._intent_ticks[i] = int(self.rng.integers(12, 31))
            self.human_uncertainty[i] = 1.0
        elif mode == self.REVERSE:
            self._intent_turn_angle[i] = math.pi
            self._intent_ticks[i] = int(self.rng.integers(10, 26))
            self.human_uncertainty[i] = 1.0
        else:
            self._intent_ticks[i] = int(self.rng.integers(18, 41))
            self.human_uncertainty[i] = 0.85

    def _update_intents(self):
        for i in range(self.nppl):
            if self._intent_ticks[i] > 0:
                self._intent_ticks[i] -= 1
                if self._intent_ticks[i] == 0:
                    self.intent_mode[i] = self.NORMAL
                    self.human_uncertainty[i] *= 0.45
                    self._next_intent_tick[i] = self.steps + int(self.rng.integers(18, 51))
            elif self.steps >= self._next_intent_tick[i]:
                self._activate_intent(i)
            else:
                self.human_uncertainty[i] *= 0.94

    def _choose_human_velocities(self):
        planned_points = [None] * self.nppl
        new_velocities = np.zeros_like(self.hv)
        angle_offsets = (
            0.0, 0.28, -0.28, 0.55, -0.55, 0.85, -0.85,
            1.15, -1.15, 1.57, -1.57, math.pi,
        )
        speed_scales = (1.0, 0.82, 0.62, 0.42, 0.0)

        for i in range(self.nppl):
            mode = int(self.intent_mode[i])
            if mode == self.STOP:
                planned_points[i] = self.hp[i].copy()
                continue

            current = np.asarray(self.hv[i], dtype=np.float32)
            speed = float(np.linalg.norm(current))
            if speed < 1e-6:
                a = float(self.rng.uniform(-math.pi, math.pi))
                direction = np.array([math.cos(a), math.sin(a)], dtype=np.float32)
            else:
                direction = current / speed

            if mode in (self.TURN, self.REVERSE):
                direction = _rotate(direction, float(self._intent_turn_angle[i]))
                self._intent_turn_angle[i] = 0.0
                if mode == self.REVERSE:
                    self.intent_mode[i] = self.NONYIELD
            else:
                direction = _rotate(direction, float(self.rng.normal(0.0, 0.035)))

            repulse = np.zeros(2, dtype=np.float32)
            for j in range(self.nppl):
                if i == j:
                    continue
                delta = self.hp[i] - self.hp[j]
                dist = float(np.linalg.norm(delta))
                if 1e-6 < dist < 1.35:
                    repulse += (delta / dist) * ((1.35 - dist) / 1.35)

            if mode != self.NONYIELD:
                for robot_pos in self.p:
                    delta = self.hp[i] - robot_pos
                    dist = float(np.linalg.norm(delta))
                    if 1e-6 < dist < 1.75:
                        repulse += 1.25 * (delta / dist) * ((1.75 - dist) / 1.75)

            desired = direction + 0.72 * repulse
            norm = float(np.linalg.norm(desired))
            desired = desired / norm if norm > 1e-6 else direction
            target_speed = float(np.clip(self._human_preferred_speed[i], HUMAN_MIN_SPEED, HUMAN_MAX_SPEED))

            best_v = np.zeros(2, dtype=np.float32)
            best_p = self.hp[i].copy()
            best_score = -float('inf')
            for off in angle_offsets:
                d = _rotate(desired, off)
                for scale in speed_scales:
                    v = d * (target_speed * scale)
                    nxt = self.hp[i] + v * DT
                    if not self._candidate_is_safe(i, nxt, planned_points):
                        continue
                    align = float(np.dot(d, desired))
                    speed_weight = 1.05 if mode == self.NONYIELD else 0.75
                    score = 2.0 * align + speed_weight * scale
                    if score > best_score:
                        best_score = score
                        best_v = v.astype(np.float32)
                        best_p = nxt.astype(np.float32)
            new_velocities[i] = best_v
            planned_points[i] = best_p
        self.hv = new_velocities

    def step(self, actions, use_cbf=True):
        obs, rewards, done = super(RealisticHumanWorld, self).step(actions, use_cbf)
        self._update_intents()
        self._choose_human_velocities()
        obs = [self.obs(i) for i in range(self.n)]
        return obs, rewards, done


class AdaptiveORCADD(AStarORCADD):
    """Peak ORCA-DD with observable-motion uncertainty adaptation.

    No latent intent label or environment-owned uncertainty variable is used.
    Uncertainty is estimated independently by each controller from consecutive
    observed pedestrian velocity vectors, which is information available to a
    real robot tracker.
    """

    def __init__(self, world, i: int, config: BeastORCAConfig | None = None, extras: AdaptiveORCAExtras | None = None):
        self.extras = extras or AdaptiveORCAExtras()
        self.max_human_time_horizon = self.extras.max_human_time_horizon
        self._last_human_velocity = np.asarray(world.hv, dtype=float).copy()
        self._observed_human_uncertainty = np.zeros(world.nppl, dtype=float)
        super().__init__(world, i, config)

    def _update_observed_human_uncertainty(self, w):
        current = np.asarray(w.hv, dtype=float)
        if self._last_human_velocity.shape != current.shape:
            self._last_human_velocity = current.copy()
            self._observed_human_uncertainty = np.zeros(w.nppl, dtype=float)
            return

        # Visible acceleration/heading change is the uncertainty cue. A roughly
        # 0.45 m/s one-tick velocity change saturates the detector; the state
        # then decays so one abrupt turn remains influential for several ticks.
        delta_v = np.linalg.norm(current - self._last_human_velocity, axis=1)
        innovation = np.clip(delta_v / 0.45, 0.0, 1.0)
        self._observed_human_uncertainty = np.maximum(
            0.82 * self._observed_human_uncertainty,
            innovation,
        )
        self._last_human_velocity = current.copy()

    def _human_time_horizon(self, w, j):
        u = float(self._observed_human_uncertainty[j])
        return float(min(
            self.max_human_time_horizon,
            self.cfg.time_horizon * (1.0 + self.extras.horizon_gain * u),
        ))

    def _human_margin(self, w, j):
        u = float(self._observed_human_uncertainty[j])
        return float(self.cfg.human_margin + min(
            self.extras.max_human_margin_add,
            self.extras.margin_gain * u,
        ))

    def _absolute_orca_lines(self, w, current_vel):
        self._update_observed_human_uncertainty(w)
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
            relative_line = build_orca_line(
                delta, rel, 2 * ROBOT_R + self.cfg.peer_margin,
                self.cfg.time_horizon, self._peer_responsibility(w, j),
            )
            lines.append(OrcaLine(point=other - relative_line.point, normal=-relative_line.normal))

        for j in range(w.nppl):
            q = np.asarray(w.hp[j], dtype=float)
            delta = q - p
            u = float(self._observed_human_uncertainty[j])
            neighbor_distance = self.cfg.neighbor_distance + self.extras.neighbor_gain * u
            if float(np.linalg.norm(delta)) > neighbor_distance:
                continue
            other = np.asarray(w.hv[j], dtype=float)
            rel = other - current_vel
            relative_line = build_orca_line(
                delta,
                rel,
                2 * ROBOT_R + self._human_margin(w, j),
                self._human_time_horizon(w, j),
                1.0,
            )
            lines.append(OrcaLine(point=other - relative_line.point, normal=-relative_line.normal))

        self._diag['orca_constraints_total'] += len(lines)
        return lines
