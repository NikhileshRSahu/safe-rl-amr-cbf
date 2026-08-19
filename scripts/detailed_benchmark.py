"""Run detailed CBF-only episodes and record per-step diagnostics.

Saves JSON with per-episode outcomes and per-step info for later comparison.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from environment import AMRWarehouseEnv
from baselines import PureCBF


def run_detailed(episodes: int, base_seed: int, max_steps: int, out_path: str):
    out = []
    env = AMRWarehouseEnv(use_cbf_filter=True, max_episode_steps=max_steps)
    env.set_curriculum_level(1.0)
    controller = PureCBF()

    for i in range(episodes):
        seed = base_seed + i
        obs, info = env.reset(seed=seed)
        if hasattr(controller, "env"):
            controller.env = env
        if hasattr(controller, "reset"):
            controller.reset()

        done = False
        ep_return = 0.0
        ep_length = 0
        success = False
        collision = False
        collision_type = "none"

        step_records: List[Dict[str, Any]] = []

        while not done and ep_length < max_steps:
            if hasattr(controller, "select_action"):
                action = controller.select_action(obs, info)
            else:
                action = controller.agent.select_action(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # record CBF diagnostics from info if present
            cbf_diag = {
                "step": ep_length,
                "cbf_intervened": bool(info.get("cbf_intervened", False)),
                "cbf_solver_success": bool(info.get("cbf_solver_success", True)),
                "tier": info.get("tier", None),
                "slack": float(info.get("slack", 0.0)) if info.get("slack", None) is not None else None,
                "barrier_value": float(info.get("barrier_value", float("nan"))) if info.get("barrier_value", None) is not None else None,
            }
            step_records.append(cbf_diag)

            ep_return += float(reward)
            ep_length += 1
            success = success or bool(info.get("goal_reached", False))
            if info.get("collision", False):
                collision = True
                collision_type = str(info.get("collision_type", "unknown"))

        episode_summary = {
            "episode_index": i,
            "seed": seed,
            "return": ep_return,
            "length": ep_length,
            "success": success,
            "collision": collision,
            "collision_type": collision_type,
            "steps": step_records,
        }
        out.append(episode_summary)
        print(f"Episode {i+1}/{episodes}: seed={seed} success={success} collision={collision} return={ep_return:.2f} len={ep_length}")

    env.close()

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved detailed results to {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=999)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out", type=str, default="benchmark_results/detailed_cbf_only_seed999.json")
    args = parser.parse_args()
    run_detailed(args.episodes, args.base_seed, args.max_steps, args.out)
