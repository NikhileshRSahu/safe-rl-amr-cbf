from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD
from benchmark.beast_config import load_beast_config
from benchmark.render_shared_success_case import (
    HUMAN_AMR_CLEARANCE,
    HUMAN_MAX_SPEED,
    HUMAN_MIN_SEPARATION,
    HUMAN_MIN_SPEED,
    RealisticHumanWorld,
    _rotate,
)
from benchmark.train_multi_agent_research import DT, VMAX, WMAX


@dataclass(frozen=True)
class RandomnessProfile:
    heading_sigma: float
    stop_probability: float
    reverse_probability: float
    speed_jitter: float


@dataclass(frozen=True)
class SweepScenario:
    name: str
    family: str
    humans: int = 12
    speed_scale: float = 1.0
    randomness_level: str = "baseline"


RANDOMNESS_PROFILES = {
    "baseline": RandomnessProfile(0.045, 0.0, 0.0, 0.0),
    "low": RandomnessProfile(0.080, 0.002, 0.001, 0.05),
    "medium": RandomnessProfile(0.140, 0.006, 0.003, 0.10),
    "high": RandomnessProfile(0.220, 0.012, 0.006, 0.18),
}


def build_count_scenarios():
    return [
        SweepScenario(f"count_{n}", "count", humans=n)
        for n in (12, 15, 19, 24)
    ]


def build_speed_scenarios():
    return [
        SweepScenario(f"speed_{scale:.2f}", "speed", speed_scale=scale)
        for scale in (1.0, 1.25, 1.5, 1.75)
    ]


def build_randomness_scenarios():
    return [
        SweepScenario(f"randomness_{level}", "randomness", randomness_level=level)
        for level in ("baseline", "low", "medium", "high")
    ]


def all_unique_scenarios():
    # Baseline is shared across all three sweeps and is evaluated only once.
    return [
        SweepScenario("baseline", "baseline"),
        SweepScenario("count_15", "count", humans=15),
        SweepScenario("count_19", "count", humans=19),
        SweepScenario("count_24", "count", humans=24),
        SweepScenario("speed_1.25", "speed", speed_scale=1.25),
        SweepScenario("speed_1.50", "speed", speed_scale=1.50),
        SweepScenario("speed_1.75", "speed", speed_scale=1.75),
        SweepScenario("randomness_low", "randomness", randomness_level="low"),
        SweepScenario("randomness_medium", "randomness", randomness_level="medium"),
        SweepScenario("randomness_high", "randomness", randomness_level="high"),
    ]


class SweepHumanWorld(RealisticHumanWorld):
    """RealisticHumanWorld with controlled speed/randomness knobs.

    Geometry, AMR dynamics, human body sizes and collision rules remain unchanged.
    Only the requested human stress factor varies.
    """

    def __init__(self, n_agents, n_people, seed, speed_scale=1.0, randomness_level="baseline"):
        if randomness_level not in RANDOMNESS_PROFILES:
            raise ValueError(f"unknown randomness level: {randomness_level}")
        self.speed_scale = float(speed_scale)
        self.randomness_level = randomness_level
        self.randomness = RANDOMNESS_PROFILES[randomness_level]
        self._human_stop_ticks = np.zeros(n_people, dtype=np.int32)
        super().__init__(n_agents, n_people, seed)

    @property
    def human_max_speed(self):
        return HUMAN_MAX_SPEED * self.speed_scale

    @property
    def human_min_speed(self):
        return HUMAN_MIN_SPEED * self.speed_scale

    def _spawn_realistic_humans(self):
        super()._spawn_realistic_humans()
        self._human_preferred_speed = self._human_preferred_speed * self.speed_scale
        self.hv = self.hv * self.speed_scale
        self._human_stop_ticks = np.zeros(self.nppl, dtype=np.int32)

    def _candidate_is_safe(self, person_idx, next_point, planned_points):
        if not self._human_point_is_free(next_point, extra=0.03):
            return False
        max_step = self.human_max_speed * DT
        for j in range(self.nppl):
            if j == person_idx:
                continue
            if j < len(planned_points) and planned_points[j] is not None:
                other = planned_points[j]
                required = HUMAN_MIN_SEPARATION
            else:
                other = self.hp[j]
                required = HUMAN_MIN_SEPARATION + max_step
            if np.linalg.norm(next_point - other) < required:
                return False
        for robot_pos in self.p:
            if np.linalg.norm(next_point - robot_pos) < HUMAN_AMR_CLEARANCE + 0.10:
                return False
        return True

    def _choose_human_velocities(self):
        planned_points = [None] * self.nppl
        new_velocities = np.zeros_like(self.hv)
        angle_offsets = (
            0.0, 0.28, -0.28, 0.55, -0.55, 0.85, -0.85,
            1.15, -1.15, 1.57, -1.57, math.pi,
        )
        speed_scales = (1.0, 0.82, 0.62, 0.42, 0.0)
        prof = self.randomness

        for i in range(self.nppl):
            if self._human_stop_ticks[i] > 0:
                self._human_stop_ticks[i] -= 1
                planned_points[i] = self.hp[i].copy()
                continue
            if prof.stop_probability > 0 and self.rng.random() < prof.stop_probability:
                self._human_stop_ticks[i] = int(self.rng.integers(5, 16))
                planned_points[i] = self.hp[i].copy()
                continue

            current = np.asarray(self.hv[i], dtype=np.float32)
            current_speed = float(np.linalg.norm(current))
            if current_speed < 1e-6:
                heading = float(self.rng.uniform(-math.pi, math.pi))
                current = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
            else:
                current = current / current_speed

            desired_dir = _rotate(current, float(self.rng.normal(0.0, prof.heading_sigma)))
            if prof.reverse_probability > 0 and self.rng.random() < prof.reverse_probability:
                desired_dir = -desired_dir

            repulse = np.zeros(2, dtype=np.float32)
            for j in range(self.nppl):
                if i == j:
                    continue
                delta = self.hp[i] - self.hp[j]
                dist = float(np.linalg.norm(delta))
                if 1e-6 < dist < 1.35:
                    repulse += (delta / dist) * ((1.35 - dist) / 1.35)
            for robot_pos in self.p:
                delta = self.hp[i] - robot_pos
                dist = float(np.linalg.norm(delta))
                if 1e-6 < dist < 1.75:
                    repulse += 1.25 * (delta / dist) * ((1.75 - dist) / 1.75)

            desired = desired_dir + 0.75 * repulse
            norm = float(np.linalg.norm(desired))
            desired = desired / norm if norm > 1e-6 else desired_dir

            base_speed = float(self._human_preferred_speed[i])
            jitter = 1.0
            if prof.speed_jitter > 0:
                jitter = float(np.clip(self.rng.normal(1.0, prof.speed_jitter), 0.45, 1.65))
            target_speed = float(np.clip(base_speed * jitter, self.human_min_speed, self.human_max_speed))

            best_velocity = np.zeros(2, dtype=np.float32)
            best_point = self.hp[i].copy()
            best_score = -float("inf")
            for angle in angle_offsets:
                direction = _rotate(desired, angle)
                for scale in speed_scales:
                    velocity = direction * (target_speed * scale)
                    next_point = self.hp[i] + velocity * DT
                    if not self._candidate_is_safe(i, next_point, planned_points):
                        continue
                    score = 2.0 * float(np.dot(direction, desired)) + 0.75 * scale
                    if score > best_score:
                        best_score = score
                        best_velocity = velocity.astype(np.float32)
                        best_point = next_point.astype(np.float32)
            new_velocities[i] = best_velocity
            planned_points[i] = best_point
        self.hv = new_velocities


def _episode_metrics(world, diagnostics=None):
    success = world.done & ~world.hit
    collision = world.hit
    timeout = ~(world.done)
    success_finish = world.finish_step[success]
    duration_steps = max(1, int(world.steps))
    result = {
        "success": int(success.sum()),
        "collision": int(collision.sum()),
        "timeout": int(timeout.sum()),
        "fleet_success": bool(success.all()),
        "episode_steps": duration_steps,
        "traversal_time_success_mean": (
            float(np.mean(success_finish) * DT) if success_finish.size else None
        ),
        "path_length_success_mean": (
            float(np.mean(world.path_length[success])) if success.any() else None
        ),
        "throughput_per_min": float(success.sum()) / (duration_steps * DT / 60.0),
        "cbf_interventions": int(world.interventions.sum()),
    }
    if diagnostics is not None:
        result["orca_constraints_total"] = int(
            sum(d.get("orca_constraints_total", 0) for d in diagnostics)
        )
        result["stop_yield_ticks"] = int(sum(d.get("stop_yield_ticks", 0) for d in diagnostics))
        result["recovery_count"] = int(sum(d.get("recovery_count", 0) for d in diagnostics))
    return result


def _run_sac(actor, scenario, seed):
    import torch
    world = SweepHumanWorld(
        4, scenario.humans, seed,
        speed_scale=scenario.speed_scale,
        randomness_level=scenario.randomness_level,
    )
    obs = world.reset()
    for _ in range(600):
        with torch.no_grad():
            action, _ = actor.sample(torch.tensor(np.asarray(obs), dtype=torch.float32), True)
        obs, _, done = world.step(action.numpy(), True)
        if np.all(done):
            break
    return _episode_metrics(world)


def _run_orca(config, scenario, seed):
    world = SweepHumanWorld(
        4, scenario.humans, seed,
        speed_scale=scenario.speed_scale,
        randomness_level=scenario.randomness_level,
    )
    world.reset()
    ctrls = [AStarORCADD(world, i, config) for i in range(4)]
    for _ in range(600):
        actions = np.asarray([
            ctrls[i].action(world) if not world.done[i] else [-1.0, 0.0]
            for i in range(4)
        ], dtype=np.float32)
        _, _, done = world.step(actions, False)
        if np.all(done):
            break
    return _episode_metrics(world, [c.diagnostics() for c in ctrls])


def _mean_present(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def aggregate(rows):
    n = len(rows)
    agents = 4 * n
    return {
        "episodes": n,
        "success_rate": sum(r["success"] for r in rows) / agents,
        "collision_rate": sum(r["collision"] for r in rows) / agents,
        "timeout_rate": sum(r["timeout"] for r in rows) / agents,
        "fleet_success_rate": sum(bool(r["fleet_success"]) for r in rows) / n,
        "mean_traversal_time_success": _mean_present(rows, "traversal_time_success_mean"),
        "mean_path_length_success": _mean_present(rows, "path_length_success_mean"),
        "mean_throughput_per_min": _mean_present(rows, "throughput_per_min"),
        "mean_cbf_interventions": _mean_present(rows, "cbf_interventions"),
        "mean_orca_constraints": _mean_present(rows, "orca_constraints_total"),
        "mean_stop_yield_ticks": _mean_present(rows, "stop_yield_ticks"),
        "mean_recovery_count": _mean_present(rows, "recovery_count"),
    }


def scenario_by_name(name):
    for scenario in all_unique_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(name)


def run_scenario(scenario, seeds, checkpoint, config_path, out_path):
    import torch
    from benchmark.train_multi_agent_research import Actor

    config = load_beast_config(config_path)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(ck["actor"])
    actor.eval()

    sac_rows, orca_rows = [], []
    for seed in seeds:
        sac = _run_sac(actor, scenario, seed)
        orca = _run_orca(config, scenario, seed)
        sac_rows.append(sac)
        orca_rows.append(orca)
        print(json.dumps({"seed": seed, "sac_cbf": sac, "peak_orca_dd": orca}), flush=True)

    payload = {
        "scenario": asdict(scenario),
        "seeds": list(seeds),
        "sac_cbf": aggregate(sac_rows),
        "peak_orca_dd": aggregate(orca_rows),
        "episodes": [
            {"seed": seed, "sac_cbf": s, "peak_orca_dd": o}
            for seed, s, o in zip(seeds, sac_rows, orca_rows)
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seeds", default="8100,8101,8102,8103,8104")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    payload = run_scenario(
        scenario_by_name(args.scenario), seeds, args.checkpoint, args.config, args.out
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
