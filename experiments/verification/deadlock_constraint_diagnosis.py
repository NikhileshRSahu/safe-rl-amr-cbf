"""experiments/deadlock_constraint_diagnosis.py

Diagnostic script (READ-ONLY on the algorithm) to answer one question:
when the robot freezes in a hard-stop deadlock, is it because ONE
constraint is very tight (over-conservative filter -> slack-relaxed QP is
the right fix), or because MULTIPLE constraints are simultaneously active
from geometrically opposing directions (reciprocal/conflicting-barrier
deadlock -> slack alone may not be enough)?

Does NOT modify cbf.py, environment.py, or the trained policy in any way.
Only calls the existing public `CBFSafetyFilter.active_barrier_constraints`
and `CBFSafetyFilter.solve` to *inspect* what happens -- it does not change
their behavior.

Usage:
    python experiments/deadlock_constraint_diagnosis.py \
        --checkpoint checkpoints/safe_sac_20260802_113318/best_model.pt \
        --episodes 3 --max-steps-per-episode 400 --deadlock-samples 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbf import CBFSafetyFilter, CBFFilterConfig, build_filter_from_config
from environment import AMRWarehouseEnv
from policy import SafeRLPolicy, PolicyConfig
from config import ROBOT_RADIUS, DYNAMIC_OBS_RADIUS, CBF_SAFETY_MARGIN, MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y, SHELVES


from evaluate import load_agent

def classify_constraint_geometry(
    A_list: List[np.ndarray], b_list: List[float], u_nom: np.ndarray
) -> Dict[str, Any]:
    n = len(A_list)
    result: Dict[str, Any] = {
        "num_active_constraints": n,
        "max_pairwise_angle_deg": 0.0,
        "opposing_pair_count": 0,
        "tightest_constraint_slack": None,
    }
    if n == 0:
        return result

    A_arr = np.array(A_list)
    b_arr = np.array(b_list)
    slacks = b_arr - A_arr @ u_nom
    result["tightest_constraint_slack"] = float(np.min(slacks))

    if n < 2:
        return result

    norms = np.linalg.norm(A_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    A_unit = A_arr / norms

    max_angle = 0.0
    opposing_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            cos_theta = float(np.clip(np.dot(A_unit[i], A_unit[j]), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(cos_theta))
            max_angle = max(max_angle, angle_deg)
            if angle_deg > 120.0:
                opposing_pairs += 1

    result["max_pairwise_angle_deg"] = max_angle
    result["opposing_pair_count"] = opposing_pairs
    return result


def run_episode_and_collect_deadlock_samples(
    env: AMRWarehouseEnv,
    agent: Any,
    cbf_filter: CBFSafetyFilter,
    max_steps: int,
    deadlock_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    obs, _ = env.reset(seed=seed)
    samples: List[Dict[str, Any]] = []

    # get max_obstacles for attention obs
    policy_config = agent.policy_config
    
    from evaluate import augment_observation

    if policy_config.use_attention_obstacles:
        obs = augment_observation(obs, env, policy_config.max_obstacles)

    for step in range(max_steps):
        action = agent.select_action(obs, deterministic=True)
        v_nom, omega_nom = float(action[0]), float(action[1])

        state_dict = env.get_state()
        robot_state = state_dict["robot_state"]
        
        A_list, b_list = cbf_filter.active_barrier_constraints(
            robot_state, env.dynamic_obstacles, SHELVES
        )
        u_nom = np.array([v_nom, omega_nom])
        v_safe, omega_safe, diag = cbf_filter.solve(
            v_nom, omega_nom, robot_state, env.dynamic_obstacles, SHELVES
        )

        is_hard_stop = (not diag["solver_success"]) and diag["num_active_constraints"] > 0

        if is_hard_stop and len(samples) < deadlock_samples:
            geometry = classify_constraint_geometry(A_list, b_list, u_nom)
            samples.append({
                "step": step,
                "robot_xy": [float(robot_state[0]), float(robot_state[1])],
                "u_nom": [v_nom, omega_nom],
                "u_safe": [v_safe, omega_safe],
                **geometry,
            })

        obs, reward, terminated, truncated, info = env.step((v_safe, omega_safe))
        if policy_config.use_attention_obstacles:
            obs = augment_observation(obs, env, policy_config.max_obstacles)
        
        if terminated or truncated:
            break
        if len(samples) >= deadlock_samples:
            pass

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps-per-episode", type=int, default=400)
    parser.add_argument("--deadlock-samples", type=int, default=3,
                         help="Max number of hard-stop timesteps to sample per episode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/deadlock_diagnosis.json")
    args = parser.parse_args()

    env = AMRWarehouseEnv(use_cbf_filter=False)
    agent = load_agent(args.checkpoint, env, device="cpu")
    
    cbf_filter = build_filter_from_config()

    all_results: List[Dict[str, Any]] = []
    for ep in range(args.episodes):
        samples = run_episode_and_collect_deadlock_samples(
            env, agent, cbf_filter,
            max_steps=args.max_steps_per_episode,
            deadlock_samples=args.deadlock_samples,
            seed=args.seed + ep,
        )
        all_results.append({"episode": ep, "deadlock_samples": samples})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Deadlock Constraint Geometry Summary ===")
    for ep_result in all_results:
        for s in ep_result["deadlock_samples"]:
            verdict = (
                "OPPOSING constraints (reciprocal deadlock)"
                if s["opposing_pair_count"] > 0
                else "single tight constraint (simple over-conservatism)"
            )
            print(
                f"  ep={ep_result['episode']} step={s['step']} "
                f"n_constraints={s['num_active_constraints']} "
                f"max_angle={s['max_pairwise_angle_deg']:.1f}deg "
                f"opposing_pairs={s['opposing_pair_count']} -> {verdict}"
            )
    print(f"\nFull data written to {args.output}")


if __name__ == "__main__":
    main()
