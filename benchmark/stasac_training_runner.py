from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from benchmark.adaptive_predictive_orca import (
    AStarAdaptivePredictiveORCADD,
    AdaptiveORCAConfig,
)
from benchmark.beast_config import load_beast_config
from benchmark.best_vs_best_protocol import (
    ScenarioSpec,
    local_human_navigation_catalog,
    split_seed_sets,
)
from benchmark.orca_teacher import bc_coefficient, normalize_teacher_action
from benchmark.spatiotemporal_policy import STASACActor, TwinRecurrentQ, reset_hidden
from benchmark.stasac_warehouse import EGO_DIM, WarehouseObservationBuilder, make_training_world
from benchmark.train_multi_agent_research import DT, VMAX, WMAX
from benchmark.train_stasac_cbf import (
    SequenceReplay,
    recurrent_sac_update,
    save_stasac_checkpoint,
)


def training_curriculum() -> tuple[ScenarioSpec, ...]:
    """Human-focused local navigation curriculum used before validation freeze."""
    return local_human_navigation_catalog()


def training_seed_for_episode(episode_index: int) -> int:
    """Map every training episode strictly into the predeclared development set."""
    dev_seeds, _, _ = split_seed_sets(frozen=False)
    if not dev_seeds:
        raise RuntimeError("development seed set is empty")
    return int(dev_seeds[int(episode_index) % len(dev_seeds)])


def _default_orca_config():
    path = Path(__file__).resolve().parent / "frozen_peak_orca_config.json"
    return load_beast_config(path)


def _actor_action(actor, ego, entity_batch, hidden, *, deterministic=False):
    ego_t = torch.as_tensor(ego, dtype=torch.float32).unsqueeze(0)
    entity_t = torch.as_tensor(entity_batch.features, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(entity_batch.mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        action, _, next_hidden, _ = actor.sample(
            ego_t, entity_t, mask_t, hidden, deterministic=deterministic
        )
    return action[0].cpu().numpy().astype(np.float32), next_hidden


def collect_training_episode(
    actor: STASACActor,
    spec: ScenarioSpec,
    *,
    seed: int,
    max_steps: int = 600,
    teacher_mix: float = 0.0,
    use_cbf: bool = True,
    orca_config=None,
    adaptive_config: AdaptiveORCAConfig | None = None,
    deterministic_actor: bool = False,
):
    """Collect one warehouse local-navigation rollout.

    STASAC supplies the policy action. AP-ORCA is queried only for the explicit
    warm-start teacher target. Deployment never calls ORCA.
    """
    teacher_mix = float(np.clip(teacher_mix, 0.0, 1.0))
    world = make_training_world(spec, int(seed))
    builder = WarehouseObservationBuilder(world, perception_range=6.0)
    cfg = orca_config or _default_orca_config()
    teacher_controllers = [
        AStarAdaptivePredictiveORCADD(world, i, cfg, adaptive_config)
        for i in range(world.n)
    ]
    hidden = [torch.zeros(1, actor.hidden_dim) for _ in range(world.n)]
    trajectories: list[list[dict]] = [[] for _ in range(world.n)]
    agent_steps = 0

    for _ in range(int(max_steps)):
        current = [None for _ in range(world.n)]
        actions = np.tile(np.array([-1.0, 0.0], np.float32), (world.n, 1))
        active_before = ~world.done.copy()

        for i in range(world.n):
            if not active_before[i]:
                hidden[i].zero_()
                continue
            ego, entities = builder.observe(world, i)
            policy_action, next_hidden = _actor_action(
                actor,
                ego,
                entities,
                hidden[i],
                deterministic=deterministic_actor,
            )
            hidden[i] = next_hidden

            physical_teacher = teacher_controllers[i].action(world)
            normalize_teacher_action(
                v=(float(physical_teacher[0]) + 1.0) * 0.5 * VMAX,
                omega=float(physical_teacher[1]) * WMAX,
                v_max=VMAX,
                omega_max=WMAX,
            )
            teacher_action = np.asarray(physical_teacher, dtype=np.float32)
            teacher_action = np.clip(teacher_action, -1.0, 1.0)

            executed = (1.0 - teacher_mix) * policy_action + teacher_mix * teacher_action
            executed = np.clip(executed, -1.0, 1.0).astype(np.float32)
            actions[i] = executed
            current[i] = {
                "ego": ego.copy(),
                "entities": entities.features.copy(),
                "entity_mask": entities.mask.copy(),
                "action": executed.copy(),
                "teacher_action": teacher_action.copy(),
            }

        _, rewards, done = world.step(actions, use_cbf=use_cbf)
        now = world.steps * DT

        for i in range(world.n):
            if not active_before[i]:
                continue
            next_ego, next_entities = builder.observe(world, i, now=now)
            transition = current[i]
            transition.update(
                reward=float(rewards[i]),
                next_ego=next_ego.copy(),
                next_entities=next_entities.features.copy(),
                next_entity_mask=next_entities.mask.copy(),
                done=bool(done[i]),
            )
            trajectories[i].append(transition)
            agent_steps += 1
            if done[i]:
                hidden[i] = reset_hidden(hidden[i], torch.tensor([True]))

        if np.all(done):
            break

    success = world.done & ~world.hit
    summary = {
        "success": int(success.sum()),
        "collision": int(world.hit.sum()),
        "fleet_success": bool(success.all()),
        "world_steps": int(world.steps),
        "cbf_interventions": int(world.interventions.sum()),
    }
    return {
        "trajectories": trajectories,
        "agent_steps": int(agent_steps),
        "summary": summary,
    }


def train_stasac(
    *,
    agent_steps: int,
    seed: int,
    out_dir: str | Path,
    max_episode_steps: int = 600,
    burn_in: int = 4,
    train_len: int = 8,
    batch_size: int = 8,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    actor = STASACActor(EGO_DIM)
    q = TwinRecurrentQ(actor.hidden_dim)
    target_q = TwinRecurrentQ(actor.hidden_dim)
    target_q.load_state_dict(q.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=2e-4)
    q_opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    replay = SequenceReplay(capacity_episodes=512, burn_in=burn_in, train_len=train_len)

    curriculum = training_curriculum()
    total_target = max(1, int(agent_steps))
    global_agent_steps = 0
    episode_index = 0
    update_count = 0
    logs = []
    rng = np.random.default_rng(seed + 9001)

    while global_agent_steps < total_target:
        spec = curriculum[episode_index % len(curriculum)]
        dev_seed = training_seed_for_episode(episode_index)
        coeff = bc_coefficient(global_agent_steps, total_target)
        episode = collect_training_episode(
            actor,
            spec,
            seed=dev_seed,
            max_steps=max_episode_steps,
            teacher_mix=coeff,
            use_cbf=True,
            deterministic_actor=False,
        )
        global_agent_steps += episode["agent_steps"]
        episode_index += 1

        for trajectory in episode["trajectories"]:
            if len(trajectory) >= replay.window_len:
                replay.add_episode(trajectory)

        episode_updates = max(1, episode["agent_steps"] // max(16, batch_size))
        latest_metrics = None
        if len(replay) >= replay.window_len * batch_size:
            for _ in range(min(episode_updates, 32)):
                batch = replay.sample(batch_size, rng=rng)
                latest_metrics = recurrent_sac_update(
                    batch,
                    actor,
                    q,
                    target_q,
                    actor_opt,
                    q_opt,
                    alpha=0.08,
                    gamma=0.99,
                    tau=0.01,
                    bc_coeff=bc_coefficient(global_agent_steps, total_target),
                )
                update_count += 1

        row = {
            "episode": episode_index,
            "scenario": spec.name,
            "seed": int(dev_seed),
            "agent_steps": int(global_agent_steps),
            "bc_coefficient": bc_coefficient(global_agent_steps, total_target),
            **episode["summary"],
        }
        if latest_metrics:
            row.update(latest_metrics)
        logs.append(row)
        print(json.dumps(row), flush=True)

    metadata = {
        "agent_steps": int(global_agent_steps),
        "seed": int(seed),
        "updates": int(update_count),
        "architecture": "spatiotemporal_risk_attention_sac_cbf_v1",
        "benchmark_focus": "human_aware_local_navigation",
        "ego_dim": EGO_DIM,
        "teacher_warm_fraction": 0.15,
        "final_bc_coefficient": bc_coefficient(global_agent_steps, total_target),
        "training_seed_split": "development_only",
        "max_episode_steps": int(max_episode_steps),
    }
    save_stasac_checkpoint(out / "stasac_cbf.pt", actor, q, metadata=metadata)
    payload = {"metadata": metadata, "episodes": logs}
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--out", default="results/stasac_smoke")
    ap.add_argument("--max-episode-steps", type=int, default=600)
    ap.add_argument("--burn-in", type=int, default=4)
    ap.add_argument("--train-len", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    result = train_stasac(
        agent_steps=args.agent_steps,
        seed=args.seed,
        out_dir=args.out,
        max_episode_steps=args.max_episode_steps,
        burn_in=args.burn_in,
        train_len=args.train_len,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["metadata"], indent=2), flush=True)


if __name__ == "__main__":
    main()
