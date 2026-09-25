from __future__ import annotations

import math

import numpy as np
import torch

from classical_baseline import AStarPlanner
from benchmark.best_vs_best_protocol import ScenarioSpec
from benchmark.human_sweep_experiment import SweepHumanWorld
from benchmark.spatiotemporal_policy import reset_hidden
from benchmark.train_multi_agent_research import DT, ROBOT_R, SHELVES, VMAX, WMAX, WORLD, wrap
from benchmark.warehouse_interaction_features import (
    EntityObservation,
    ObservationHistory,
    build_entity_batch,
)


# 2 local A* waypoint + 2 local final goal + goal distance + heading error
# + current v/w + right-of-way priority + 8 static lidar rays.
EGO_DIM = 17


def make_training_world(spec: ScenarioSpec, seed: int):
    return SweepHumanWorld(
        spec.n_amr,
        spec.humans,
        int(seed),
        speed_scale=spec.speed_scale,
        randomness_level=spec.randomness_level,
    )


def _rotate_world_to_body(vector, heading: float):
    c = math.cos(-float(heading))
    s = math.sin(-float(heading))
    x, y = float(vector[0]), float(vector[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


class WarehouseObservationBuilder:
    """Causal policy observations with common A* route context and all entities."""

    def __init__(
        self,
        world,
        *,
        perception_range: float = 6.0,
        history_len: int = 8,
        lookahead_distance: float = 1.8,
    ):
        self.perception_range = float(perception_range)
        self.lookahead_distance = float(lookahead_distance)
        self.histories = [ObservationHistory(maxlen=history_len) for _ in range(world.n)]
        self._last_history_time: list[dict[str, float]] = [dict() for _ in range(world.n)]
        self.path_indices = [0 for _ in range(world.n)]
        self.paths = []
        for i in range(world.n):
            planner = AStarPlanner(
                map_bounds=(-WORLD, WORLD, -WORLD, WORLD),
                shelves=SHELVES,
                robot_radius=ROBOT_R,
                margin=0.10,
                resolution=0.25,
            )
            path = planner.plan(tuple(world.p[i]), tuple(world.g[i])) or [
                tuple(world.p[i]), tuple(world.g[i])
            ]
            self.paths.append(path)

    def _waypoint(self, world, i: int):
        path = self.paths[i]
        p = np.asarray(world.p[i], dtype=float)
        idx = self.path_indices[i]
        while idx < len(path) - 1 and np.linalg.norm(np.asarray(path[idx], dtype=float) - p) <= 0.45:
            idx += 1
        self.path_indices[i] = idx
        j = idx
        accumulated = 0.0
        while j + 1 < len(path) and accumulated < self.lookahead_distance:
            a = np.asarray(path[j], dtype=float)
            b = np.asarray(path[j + 1], dtype=float)
            accumulated += float(np.linalg.norm(b - a))
            j += 1
        return np.asarray(path[j], dtype=np.float32)

    def _entity_observations(self, world, i: int, now: float):
        p = np.asarray(world.p[i], dtype=float)
        items: list[EntityObservation] = []
        for j in range(world.n):
            if j == i or world.done[j]:
                continue
            q = np.asarray(world.p[j], dtype=float)
            if float(np.linalg.norm(q - p)) > self.perception_range:
                continue
            velocity = np.array(
                [
                    math.cos(float(world.th[j])) * float(world.v[j]),
                    math.sin(float(world.th[j])) * float(world.v[j]),
                ],
                dtype=np.float32,
            )
            items.append(EntityObservation(f"amr-{j}", "amr", q, velocity, now, True))
        for j in range(world.nppl):
            q = np.asarray(world.hp[j], dtype=float)
            if float(np.linalg.norm(q - p)) > self.perception_range:
                continue
            items.append(
                EntityObservation(
                    f"human-{j}",
                    "human",
                    q,
                    np.asarray(world.hv[j], dtype=np.float32),
                    now,
                    True,
                )
            )
        return items

    def observe(self, world, i: int, *, now: float | None = None):
        now = float(world.steps * DT if now is None else now)
        heading = float(world.th[i])
        p = np.asarray(world.p[i], dtype=np.float32)
        goal = np.asarray(world.g[i], dtype=np.float32)
        waypoint = self._waypoint(world, i)
        goal_body = _rotate_world_to_body(goal - p, heading)
        waypoint_body = _rotate_world_to_body(waypoint - p, heading)
        goal_distance = float(np.linalg.norm(goal - p))
        goal_heading = math.atan2(float(goal[1] - p[1]), float(goal[0] - p[0]))
        heading_error = wrap(goal_heading - heading)
        rays = [world.ray(i, heading + k * math.pi / 4.0) for k in range(8)]
        ego = np.asarray(
            [
                np.clip(waypoint_body[0] / 6.0, -1.0, 1.0),
                np.clip(waypoint_body[1] / 6.0, -1.0, 1.0),
                np.clip(goal_body[0] / 20.0, -1.0, 1.0),
                np.clip(goal_body[1] / 20.0, -1.0, 1.0),
                np.clip(goal_distance / 20.0, 0.0, 1.0),
                heading_error / math.pi,
                np.clip(float(world.v[i]) / VMAX, 0.0, 1.0),
                np.clip(float(world.w[i]) / WMAX, -1.0, 1.0),
                float(world.priority[i]),
                *rays,
            ],
            dtype=np.float32,
        )
        if ego.shape != (EGO_DIM,):
            raise RuntimeError(f"ego feature contract broken: {ego.shape}")

        ego_velocity = np.array(
            [math.cos(heading) * float(world.v[i]), math.sin(heading) * float(world.v[i])],
            dtype=np.float32,
        )
        observations = self._entity_observations(world, i, now)
        history = self.histories[i]
        last_times = self._last_history_time[i]
        for obs in observations:
            previous = last_times.get(obs.entity_id)
            if previous is None or now > previous + 1e-12:
                history.update(obs.entity_id, now, obs.position, obs.velocity)
                last_times[obs.entity_id] = now
        batch = build_entity_batch(
            ego_pos=p,
            ego_vel=ego_velocity,
            entities=observations,
            now=now,
            history=history,
        )
        return ego, batch


def _actor_action(actor, ego, entity_batch, hidden, deterministic=True):
    ego_t = torch.tensor(ego, dtype=torch.float32).unsqueeze(0)
    entities_t = torch.tensor(entity_batch.features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(entity_batch.mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        action, _, next_hidden, _ = actor.sample(
            ego_t, entities_t, mask_t, hidden, deterministic=deterministic
        )
    return action[0].cpu().numpy().astype(np.float32), next_hidden


def rollout_untrained_actor_smoke(actor, world, *, steps: int = 16):
    """Short real-world integration probe; success is numerical stability only."""
    builder = WarehouseObservationBuilder(world, perception_range=100.0)
    hidden = [torch.zeros(1, actor.hidden_dim) for _ in range(world.n)]
    finite_actions = True
    max_abs_action = 0.0
    ran = 0
    for _ in range(int(steps)):
        actions = []
        for i in range(world.n):
            if world.done[i]:
                actions.append(np.array([-1.0, 0.0], np.float32))
                hidden[i].zero_()
                continue
            ego, entities = builder.observe(world, i)
            action, hidden[i] = _actor_action(actor, ego, entities, hidden[i], deterministic=True)
            finite_actions = finite_actions and bool(np.isfinite(action).all())
            max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))
            actions.append(action)
        _, _, done = world.step(np.asarray(actions, dtype=np.float32), use_cbf=True)
        ran += 1
        for i in range(world.n):
            if done[i]:
                hidden[i] = reset_hidden(hidden[i], torch.tensor([True]))
        if np.all(done):
            break
    finite_world = bool(
        np.isfinite(world.p).all()
        and np.isfinite(world.th).all()
        and np.isfinite(world.v).all()
        and np.isfinite(world.w).all()
        and np.isfinite(world.hp).all()
        and np.isfinite(world.hv).all()
    )
    return {
        "steps": ran,
        "finite_actions": bool(finite_actions),
        "finite_world_state": finite_world,
        "max_abs_action": float(max_abs_action),
    }
