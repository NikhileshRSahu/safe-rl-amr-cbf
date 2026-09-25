from __future__ import annotations

import math
from statistics import mean

import numpy as np
import torch

from benchmark.adaptive_predictive_orca import AStarAdaptivePredictiveORCADD
from benchmark.beast_classical import BeastORCAConfig
from benchmark.best_vs_best_evaluation import ControllerAggregate
from benchmark.best_vs_best_protocol import ScenarioSpec, make_paired_worlds
from benchmark.spatiotemporal_policy import reset_hidden
from benchmark.stasac_warehouse import WarehouseObservationBuilder
from benchmark.train_multi_agent_research import DT


def peak_orca_config() -> BeastORCAConfig:
    """Frozen strongest validated ORCA-DD base configuration."""
    return BeastORCAConfig(
        time_horizon=3.0,
        neighbor_distance=4.0,
        peer_margin=0.08,
        human_margin=0.12,
        static_margin=0.08,
        planning_margin=0.10,
        lookahead_distance=1.8,
        waypoint_tolerance=0.45,
        replan_interval=12,
        preferred_speed=1.0,
        command_speed_samples=7,
        command_omega_samples=15,
        smoothness_weight=0.12,
        progress_weight=4.0,
        stuck_progress_eps=0.01,
        stuck_ticks=22,
        recovery_ticks=20,
        arc_horizon=1.0,
    )


def _episode_metrics(world, *, seed: int, scenario: str, steps: int, diagnostics=None):
    success_mask = world.done & ~world.hit
    collision_mask = world.hit
    timeout_mask = ~world.done
    success = int(success_mask.sum())
    collision = int(collision_mask.sum())
    timeout = int(timeout_mask.sum())
    duration = max(int(steps), 1) * DT
    row = {
        "seed": int(seed),
        "scenario": str(scenario),
        "agents": int(world.n),
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "fleet_success": bool(success == world.n),
        "steps": int(steps),
        "throughput_per_min": float(success / (duration / 60.0)),
        "cbf_interventions": int(world.interventions.sum()),
        "path_length_success_mean": (
            float(np.mean(world.path_length[success_mask])) if success else None
        ),
        "traversal_time_success_mean": (
            float(np.mean(world.finish_step[success_mask]) * DT) if success else None
        ),
        "min_clearance": (
            float(np.min(world.min_clearance[np.isfinite(world.min_clearance)]))
            if np.any(np.isfinite(world.min_clearance)) else None
        ),
    }
    if diagnostics is not None:
        keys = set().union(*(d.keys() for d in diagnostics)) if diagnostics else set()
        for key in keys:
            values = [d.get(key, 0) for d in diagnostics]
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                row[f"ap_orca_{key}"] = float(sum(float(v) for v in values))
    return row


def run_stasac_episode(actor, spec: ScenarioSpec, *, seed: int, max_steps: int = 600):
    world, _ = make_paired_worlds(spec, int(seed))
    builder = WarehouseObservationBuilder(world)
    hidden = [torch.zeros(1, actor.hidden_dim) for _ in range(world.n)]
    steps = 0
    for _ in range(int(max_steps)):
        actions = []
        for i in range(world.n):
            if world.done[i]:
                actions.append(np.array([-1.0, 0.0], dtype=np.float32))
                hidden[i].zero_()
                continue
            ego, entity_batch = builder.observe(world, i)
            ego_t = torch.as_tensor(ego, dtype=torch.float32).unsqueeze(0)
            entities_t = torch.as_tensor(entity_batch.features, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(entity_batch.mask, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                action, _, next_hidden, _ = actor.sample(
                    ego_t, entities_t, mask_t, hidden[i], deterministic=True
                )
            actions.append(action[0].cpu().numpy().astype(np.float32))
            hidden[i] = next_hidden
        _, _, done = world.step(np.asarray(actions, dtype=np.float32), use_cbf=True)
        steps += 1
        for i in range(world.n):
            if done[i]:
                hidden[i] = reset_hidden(hidden[i], torch.tensor([True]))
        if np.all(done):
            break
    return _episode_metrics(world, seed=seed, scenario=spec.name, steps=steps)


def run_ap_orca_episode(spec: ScenarioSpec, *, seed: int, max_steps: int = 600):
    _, world = make_paired_worlds(spec, int(seed))
    cfg = peak_orca_config()
    controllers = [AStarAdaptivePredictiveORCADD(world, i, cfg) for i in range(world.n)]
    steps = 0
    for _ in range(int(max_steps)):
        actions = np.asarray([c.action(world) for c in controllers], dtype=np.float32)
        _, _, done = world.step(actions, use_cbf=True)
        steps += 1
        if np.all(done):
            break
    diagnostics = [c.diagnostics() for c in controllers]
    return _episode_metrics(
        world, seed=seed, scenario=spec.name, steps=steps, diagnostics=diagnostics
    )


def aggregate_episode_rows(rows) -> ControllerAggregate:
    rows = list(rows)
    if not rows:
        raise ValueError("rows must not be empty")
    agents = int(sum(int(r["agents"]) for r in rows))
    episodes = len(rows)
    successes = sum(int(r["success"]) for r in rows)
    collisions = sum(int(r["collision"]) for r in rows)
    timeouts = sum(int(r["timeout"]) for r in rows)
    return ControllerAggregate(
        agents=agents,
        episodes=episodes,
        success_rate=float(successes / agents),
        collision_rate=float(collisions / agents),
        timeout_rate=float(timeouts / agents),
        fleet_success_rate=float(sum(bool(r["fleet_success"]) for r in rows) / episodes),
        throughput_per_min=float(mean(float(r["throughput_per_min"]) for r in rows)),
    )
