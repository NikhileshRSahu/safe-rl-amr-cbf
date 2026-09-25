from __future__ import annotations

import numpy as np

from benchmark.human_sweep_experiment import SweepHumanWorld
from benchmark.render_shared_success_case import HUMAN_MIN_SEPARATION


CROSS_CENTER = np.array([-4.5, 0.0], dtype=np.float32)


class WarehouseScenarioWorld(SweepHumanWorld):
    """Physical warehouse interaction scenarios built on the same world physics.

    The scenarios change pedestrian initial conditions and bounded intent events;
    AMR starts/goals, robot dynamics, shelves, CBF and collision definitions stay
    unchanged. Both RL and classical controllers receive the same world realization.
    """

    def __init__(
        self,
        n_agents: int,
        n_people: int,
        seed: int,
        *,
        scenario_name: str,
        speed_scale: float = 1.0,
        randomness_level: str = "baseline",
    ):
        self.scenario_name = str(scenario_name)
        super().__init__(
            n_agents,
            n_people,
            seed,
            speed_scale=speed_scale,
            randomness_level=randomness_level,
        )

    def reset(self):
        obs = super().reset()
        self._configure_scenario()
        return [self.obs(i) for i in range(self.n)]

    def _repair_remaining_humans(self, fixed_count: int):
        fixed = [self.hp[i].copy() for i in range(min(fixed_count, self.nppl))]
        for idx in range(fixed_count, self.nppl):
            q = self.hp[idx]
            bad = (
                not self._human_point_is_free(q, extra=0.08)
                or any(np.linalg.norm(q - p) < HUMAN_MIN_SEPARATION + 0.10 for p in fixed)
                or any(np.linalg.norm(q - p) < 1.0 for p in self.p)
            )
            if bad:
                for _ in range(2000):
                    candidate = self.rng.uniform(-8.35, 8.35, size=2).astype(np.float32)
                    if not self._human_point_is_free(candidate, extra=0.08):
                        continue
                    if any(
                        np.linalg.norm(candidate - p) < HUMAN_MIN_SEPARATION + 0.10
                        for p in fixed
                    ):
                        continue
                    if any(np.linalg.norm(candidate - p) < 1.0 for p in self.p):
                        continue
                    q = candidate
                    break
            self.hp[idx] = q
            fixed.append(q.copy())

    def _set_human(self, idx: int, position, velocity):
        if idx >= self.nppl:
            return
        self.hp[idx] = np.asarray(position, dtype=np.float32)
        vel = np.asarray(velocity, dtype=np.float32) * float(self.speed_scale)
        self.hv[idx] = vel
        speed = float(np.linalg.norm(vel))
        self._human_preferred_speed[idx] = max(self.human_min_speed, min(self.human_max_speed, speed))

    def _configure_cross_intersection(self):
        # Real intersection in the vertical aisle between x=-6 and x=-3 and
        # the warehouse-wide horizontal corridor y in [-2,2].
        staged = (
            ((-5.75, -0.45), (0.48, 0.00)),
            ((-3.25, 0.45), (-0.46, 0.00)),
            ((-4.95, -1.65), (0.00, 0.50)),
            ((-4.05, 1.65), (0.00, -0.44)),
            ((-5.65, 0.70), (0.42, 0.00)),
            ((-3.35, -0.70), (-0.40, 0.00)),
        )
        count = min(len(staged), self.nppl)
        for i in range(count):
            self._set_human(i, *staged[i])
        self._repair_remaining_humans(count)

    def _configure_shelf_corner(self):
        # Human 0 begins behind the lower-right corner of shelf [-8,2]-[-6,8]
        # from AMR 1's left-corridor view and walks down into y~0 traffic.
        staged = (
            ((-5.55, 2.65), (0.00, -0.46)),
            ((-4.55, 1.65), (0.00, -0.38)),
            ((-5.45, -1.45), (0.00, 0.34)),
        )
        count = min(len(staged), self.nppl)
        for i in range(count):
            self._set_human(i, *staged[i])
        self._repair_remaining_humans(count)

    def _configure_hesitation(self):
        staged = (
            ((-5.70, -0.35), (0.46, 0.00)),
            ((-4.25, -1.55), (0.00, 0.44)),
        )
        count = min(len(staged), self.nppl)
        for i in range(count):
            self._set_human(i, *staged[i])
        self._repair_remaining_humans(count)

    def _configure_mixed(self):
        self._configure_cross_intersection()

    def _configure_scenario(self):
        if self.scenario_name == "cross_intersection":
            self._configure_cross_intersection()
        elif self.scenario_name == "shelf_corner":
            self._configure_shelf_corner()
        elif self.scenario_name == "hesitation":
            self._configure_hesitation()
        elif self.scenario_name == "mixed_behavior":
            self._configure_mixed()

    def _choose_human_velocities(self):
        super()._choose_human_velocities()
        # Deterministic, physically bounded intent changes supplement the normal
        # stochastic pedestrian model. They depend only on current episode time.
        if self.nppl == 0:
            return
        tick = int(self.steps)
        if self.scenario_name == "hesitation":
            if 28 <= tick < 42:
                self.hv[0] = 0.0
            elif 42 <= tick < 58:
                direction = np.array([1.0, 0.0], dtype=np.float32)
                self.hv[0] = direction * min(0.42 * self.speed_scale, self.human_max_speed)
            elif 58 <= tick < 68:
                self.hv[0] = 0.0
        elif self.scenario_name == "mixed_behavior":
            if 24 <= tick < 35:
                self.hv[0] = 0.0
            if self.nppl > 1 and 45 <= tick < 62:
                speed = float(np.linalg.norm(self.hv[1]))
                if speed > 1e-6:
                    self.hv[1] = -self.hv[1] / speed * min(speed, self.human_max_speed)


def make_scenario_world(spec, seed: int):
    return WarehouseScenarioWorld(
        spec.n_amr,
        spec.humans,
        int(seed),
        scenario_name=spec.name,
        speed_scale=spec.speed_scale,
        randomness_level=spec.randomness_level,
    )
