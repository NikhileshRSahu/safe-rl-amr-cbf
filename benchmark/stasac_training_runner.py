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
from benchmark.best_vs_best_runner import run_ap_orca_episode
from benchmark.orca_teacher import (
    normalize_teacher_action,
    performance_gated_bc_coefficient,
    should_promote_curriculum_stage,
)
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


def training_curriculum_stages() -> tuple[tuple[ScenarioSpec, ...], ...]:
    """Difficulty stages; promotion is performance gated, never time gated."""
    catalog = {s.name: s for s in training_curriculum()}
    return (
        (catalog["human_crossing"], catalog["blind_shelf_corner"]),
        (catalog["human_hesitation"], catalog["human_reversal"]),
        (catalog["forklift_crossing"], catalog["dense_human_flow"]),
        (catalog["mixed_local_traffic"],),
    )


def competence_gate_max_steps(training_max_steps: int) -> int:
    """Use a physically meaningful horizon for competence measurements.

    Smoke rollouts are intentionally truncated for CI speed.  Reusing that
    truncated horizon for policy/teacher competence makes both look like
    failures and can incorrectly remove teacher support.  The gate therefore
    gets at least 300 simulator steps, while normal 600-step research runs keep
    their full horizon unchanged.
    """
    return max(300, int(training_max_steps))


def curriculum_gate_seeds() -> tuple[int, ...]:
    """Development-only seeds withheld from gradient collection for competence gates."""
    dev_seeds, _, _ = split_seed_sets(frozen=False)
    return tuple(int(x) for x in dev_seeds[-4:])


def training_seed_for_episode(episode_index: int) -> int:
    """Map gradient episodes to development seeds excluding competence-gate seeds."""
    dev_seeds, _, _ = split_seed_sets(frozen=False)
    gate = set(curriculum_gate_seeds())
    train_seeds = tuple(int(x) for x in dev_seeds if int(x) not in gate)
    if not train_seeds:
        raise RuntimeError("development training seed set is empty")
    return int(train_seeds[int(episode_index) % len(train_seeds)])


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
    training teacher target. Deployment never calls ORCA.
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
            teacher_action = np.clip(
                np.asarray(physical_teacher, dtype=np.float32), -1.0, 1.0
            )

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


def _rate_summary(rows: list[dict]) -> dict[str, float]:
    agents = max(1, sum(int(r["agents"]) for r in rows))
    return {
        "episodes": float(len(rows)),
        "success_rate": float(sum(int(r["success"]) for r in rows) / agents),
        "collision_rate": float(sum(int(r["collision"]) for r in rows) / agents),
    }


def _policy_only_gate_rows(
    actor: STASACActor,
    stage: tuple[ScenarioSpec, ...],
    *,
    max_steps: int,
) -> list[dict]:
    rows: list[dict] = []
    for spec in stage:
        for seed in curriculum_gate_seeds()[:2]:
            result = collect_training_episode(
                actor,
                spec,
                seed=seed,
                max_steps=max_steps,
                teacher_mix=0.0,
                use_cbf=True,
                deterministic_actor=True,
            )
            rows.append(
                {
                    "scenario": spec.name,
                    "seed": int(seed),
                    "agents": int(spec.n_amr),
                    "success": int(result["summary"]["success"]),
                    "collision": int(result["summary"]["collision"]),
                }
            )
    return rows


def _teacher_gate_rows(
    stage: tuple[ScenarioSpec, ...], *, max_steps: int
) -> list[dict]:
    rows: list[dict] = []
    for spec in stage:
        for seed in curriculum_gate_seeds()[:2]:
            row = run_ap_orca_episode(spec, seed=seed, max_steps=max_steps)
            rows.append(row)
    return rows


def train_stasac(
    *,
    agent_steps: int,
    seed: int,
    out_dir: str | Path,
    max_episode_steps: int = 600,
    burn_in: int = 4,
    train_len: int = 8,
    batch_size: int = 8,
    gate_interval_episodes: int = 5,
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

    stages = training_curriculum_stages()
    total_target = max(1, int(agent_steps))
    gate_max_steps = competence_gate_max_steps(max_episode_steps)
    global_agent_steps = 0
    episode_index = 0
    update_count = 0
    current_stage = 0
    stage_episode_index = 0
    teacher_coeff = 1.0
    policy_gate_history: list[dict] = []
    teacher_gate_cache: dict[int, dict[str, float]] = {}
    gate_logs: list[dict] = []
    logs = []
    rng = np.random.default_rng(seed + 9001)

    while global_agent_steps < total_target:
        stage = stages[current_stage]
        spec = stage[stage_episode_index % len(stage)]
        dev_seed = training_seed_for_episode(episode_index)
        episode = collect_training_episode(
            actor,
            spec,
            seed=dev_seed,
            max_steps=max_episode_steps,
            teacher_mix=teacher_coeff,
            use_cbf=True,
            deterministic_actor=False,
        )
        global_agent_steps += episode["agent_steps"]
        episode_index += 1
        stage_episode_index += 1

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
                    bc_coeff=teacher_coeff,
                )
                update_count += 1

        row = {
            "episode": episode_index,
            "stage": int(current_stage),
            "scenario": spec.name,
            "seed": int(dev_seed),
            "agent_steps": int(global_agent_steps),
            "bc_coefficient": float(teacher_coeff),
            **episode["summary"],
        }
        if latest_metrics:
            row.update(latest_metrics)
        logs.append(row)
        print(json.dumps(row), flush=True)

        if stage_episode_index % max(1, int(gate_interval_episodes)) == 0:
            probe_rows = _policy_only_gate_rows(actor, stage, max_steps=gate_max_steps)
            policy_gate_history.extend(probe_rows)
            policy_gate_history = policy_gate_history[-12:]
            policy_rates = _rate_summary(policy_gate_history)

            if current_stage not in teacher_gate_cache:
                teacher_gate_cache[current_stage] = _rate_summary(
                    _teacher_gate_rows(stage, max_steps=gate_max_steps)
                )
            teacher_rates = teacher_gate_cache[current_stage]
            teacher_coeff = performance_gated_bc_coefficient(
                policy_rates["success_rate"],
                policy_rates["collision_rate"],
                teacher_rates["success_rate"],
                teacher_rates["collision_rate"],
            )
            promoted = should_promote_curriculum_stage(
                episodes=int(policy_rates["episodes"]),
                success_rate=policy_rates["success_rate"],
                collision_rate=policy_rates["collision_rate"],
            )
            gate_row = {
                "after_episode": int(episode_index),
                "stage": int(current_stage),
                "gate_max_steps": int(gate_max_steps),
                "policy": policy_rates,
                "teacher": teacher_rates,
                "next_bc_coefficient": float(teacher_coeff),
                "promoted": bool(promoted and current_stage < len(stages) - 1),
            }
            gate_logs.append(gate_row)
            print(json.dumps({"competence_gate": gate_row}), flush=True)

            if promoted and current_stage < len(stages) - 1:
                current_stage += 1
                stage_episode_index = 0
                policy_gate_history = []
                teacher_coeff = 1.0

    metadata = {
        "agent_steps": int(global_agent_steps),
        "seed": int(seed),
        "updates": int(update_count),
        "architecture": "spatiotemporal_risk_attention_sac_cbf_v1",
        "benchmark_focus": "human_aware_local_navigation",
        "teacher_schedule": "performance_gated",
        "curriculum_schedule": "safety_success_gated_4_stage",
        "final_stage": int(current_stage),
        "ego_dim": EGO_DIM,
        "final_bc_coefficient": float(teacher_coeff),
        "training_seed_split": "development_gradient_only",
        "competence_gate_seeds": list(curriculum_gate_seeds()),
        "max_episode_steps": int(max_episode_steps),
        "competence_gate_max_steps": int(gate_max_steps),
    }
    save_stasac_checkpoint(out / "stasac_cbf.pt", actor, q, metadata=metadata)
    payload = {"metadata": metadata, "episodes": logs, "competence_gates": gate_logs}
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
    ap.add_argument("--gate-interval-episodes", type=int, default=5)
    args = ap.parse_args()
    result = train_stasac(
        agent_steps=args.agent_steps,
        seed=args.seed,
        out_dir=args.out,
        max_episode_steps=args.max_episode_steps,
        burn_in=args.burn_in,
        train_len=args.train_len,
        batch_size=args.batch_size,
        gate_interval_episodes=args.gate_interval_episodes,
    )
    print(json.dumps(result["metadata"], indent=2), flush=True)


if __name__ == "__main__":
    main()
