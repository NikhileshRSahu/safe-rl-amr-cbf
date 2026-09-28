from __future__ import annotations

import numpy as np

from benchmark.human_motion_profiles import hesitation_speed_factor
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
        self._scenario_seed = int(seed)
        event_rng = np.random.default_rng(self._scenario_seed + 91573)
        self._event_tick = int(event_rng.integers(28, 41))
        self._event_duration = int(event_rng.integers(10, 17))
        self._cut_sign = -1.0 if int(event_rng.integers(0, 2)) == 0 else 1.0
        self._cut_angle = float(event_rng.uniform(np.deg2rad(58.0), np.deg2rad(78.0)))
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

    def _configure_reversal(self):
        self._configure_cross_intersection()

    def _configure_forklift_like_crossing(self):
        # A faster, non-reciprocal cross-aisle mover. It deliberately reuses the
        # current circular dynamic-obstacle geometry; it is 'forklift-like' in
        # kinematics/priority only, not yet a rectangular forklift footprint.
        staged = (
            ((-5.90, -0.25), (0.58, 0.00)),
            ((-3.10, 0.85), (-0.52, 0.00)),
        )
        count = min(len(staged), self.nppl)
        for i in range(count):
            self._set_human(i, *staged[i])
        self._repair_remaining_humans(count)

    def _configure_mixed(self):
        self._configure_cross_intersection()

    def _configure_scenario(self):
        name = self.scenario_name
        if name in {
            "cross_intersection",
            "human_crossing",
            "dense_human_flow",
            "mixed_local_traffic",
            "surprise_stop_go",
            "lateral_cut_in",
            "compound_nonreciprocal",
        }:
            self._configure_cross_intersection()
        elif name in {"shelf_corner", "blind_shelf_corner"}:
            self._configure_shelf_corner()
        elif name in {"hesitation", "human_hesitation"}:
            self._configure_hesitation()
        elif name == "human_reversal":
            self._configure_reversal()
        elif name == "forklift_crossing":
            self._configure_forklift_like_crossing()
        elif name == "mixed_behavior":
            self._configure_mixed()

    @staticmethod
    def _ramped_stop_go_velocity(tick: int, start: int, duration: int, speed: float) -> np.ndarray:
        """Finite-acceleration stop/restart profile with an unseen event time."""
        decel_ticks = 5
        accel_ticks = 6
        if tick < start - decel_ticks:
            factor = 1.0
        elif tick < start:
            factor = max(0.0, (start - tick) / decel_ticks)
        elif tick < start + duration:
            factor = 0.0
        elif tick < start + duration + accel_ticks:
            factor = min(1.0, (tick - (start + duration) + 1) / accel_ticks)
        else:
            factor = 1.0
        return np.array([speed * factor, 0.0], dtype=np.float32)

    def _lateral_cut_velocity(self, tick: int, start: int, speed: float, *, base_angle: float = 0.0) -> np.ndarray:
        """Seed-varying bounded cut-in with a finite heading-rate cue."""
        turn_ticks = 8
        fraction = float(np.clip((tick - start + 1) / turn_ticks, 0.0, 1.0))
        angle = base_angle + self._cut_sign * self._cut_angle * fraction
        return np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)

    def _choose_human_velocities(self):
        super()._choose_human_velocities()
        if self.nppl == 0:
            return
        tick = int(self.steps)
        if self.scenario_name in {"hesitation", "human_hesitation"}:
            # Smooth finite acceleration makes hesitation human-like and causal:
            # every controller sees the same deceleration cue before the stop.
            factor = hesitation_speed_factor(tick)
            direction = np.array([1.0, 0.0], dtype=np.float32)
            speed = min(0.46 * self.speed_scale * factor, self.human_max_speed)
            self.hv[0] = direction * speed
        elif self.scenario_name in {"mixed_behavior", "human_reversal", "mixed_local_traffic"}:
            if self.scenario_name != "human_reversal" and 24 <= tick < 35:
                self.hv[0] = 0.0
            target_idx = 0 if self.scenario_name == "human_reversal" else 1
            if self.nppl > target_idx and 45 <= tick < 62:
                speed = float(np.linalg.norm(self.hv[target_idx]))
                if speed > 1e-6:
                    self.hv[target_idx] = -self.hv[target_idx] / speed * min(speed, self.human_max_speed)
        elif self.scenario_name == "surprise_stop_go":
            speed = min(0.48 * self.speed_scale, self.human_max_speed)
            self.hv[0] = self._ramped_stop_go_velocity(
                tick, self._event_tick, self._event_duration, speed
            )
        elif self.scenario_name == "lateral_cut_in":
            speed = min(0.48 * self.speed_scale, self.human_max_speed)
            if tick < self._event_tick:
                self.hv[0] = np.array([speed, 0.0], dtype=np.float32)
            else:
                self.hv[0] = self._lateral_cut_velocity(tick, self._event_tick, speed)
        elif self.scenario_name == "compound_nonreciprocal":
            speed0 = min(0.48 * self.speed_scale, self.human_max_speed)
            self.hv[0] = self._ramped_stop_go_velocity(
                tick, self._event_tick, self._event_duration, speed0
            )
            if self.nppl > 1:
                speed1 = min(0.46 * self.speed_scale, self.human_max_speed)
                if tick < self._event_tick + 5:
                    self.hv[1] = np.array([-speed1, 0.0], dtype=np.float32)
                else:
                    self.hv[1] = self._lateral_cut_velocity(
                        tick, self._event_tick + 5, speed1, base_angle=np.pi
                    )
            if self.nppl > 2 and self._event_tick + 14 <= tick < self._event_tick + 27:
                speed2 = float(np.linalg.norm(self.hv[2]))
                if speed2 > 1e-6:
                    self.hv[2] = -self.hv[2] / speed2 * min(speed2, self.human_max_speed)


def make_scenario_world(spec, seed: int):
    return WarehouseScenarioWorld(
        spec.n_amr,
        spec.humans,
        int(seed),
        scenario_name=spec.name,
        speed_scale=spec.speed_scale,
        randomness_level=spec.randomness_level,
    )
