"""evaluate_improved.py

Evaluates a trained Improved SafeSACAgent (with attention-based obstacle encoder)
on AMRWarehouseEnv with deterministic actions and optional rendering.

This is the evaluation counterpart to train_improved.py -- it handles the
observation augmentation (obstacle_set / obstacle_set_mask) that the improved
policy expects.

Usage:
    python evaluate_improved.py --checkpoint checkpoints_improved/.../best_model.pt --episodes 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from environment import AMRWarehouseEnv
from safe_sac import SafeSACAgent, SafeSACConfig
from evaluate import load_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and evaluate a trained Improved SafeSACAgent checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint saved by SafeSACAgent.save_models (e.g. best_model.pt).",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of evaluation episodes to run."
    )
    parser.add_argument("--seed", type=int, default=123, help="Base random seed for evaluation.")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Rendering backend. 'human' opens a live matplotlib window; "
        "'rgb_array' renders off-screen and (optionally) saves a GIF per episode.",
    )
    parser.add_argument(
        "--save-video-dir",
        type=str,
        default=None,
        help="If set (and --render-mode rgb_array), save an episode GIF per episode to this "
        "directory. Requires the 'imageio' package; skipped with a warning if unavailable.",
    )
    parser.add_argument(
        "--no-cbf-filter",
        action="store_true",
        help="Disable the native CBF-QP safety filter (evaluate the raw, unfiltered policy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to load the agent onto.",
    )
    parser.add_argument(
        "--curriculum-level",
        type=float,
        default=1.0,
        help="Difficulty level (fraction of obstacles active).",
    )
    return parser.parse_args()


def augment_observation(
    obs: Dict[str, np.ndarray],
    env: AMRWarehouseEnv,
    max_obstacles: int,
) -> Dict[str, np.ndarray]:
    """Add obstacle_set to observation for attention encoder."""
    rx, ry, theta = env.robot_state[0], env.robot_state[1], env.robot_state[2]
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    obstacle_set = np.zeros((max_obstacles, 5), dtype=np.float32)
    mask = np.zeros(max_obstacles, dtype=bool)

    num_active = int(min(len(env.dynamic_obstacles), max_obstacles))
    for i in range(num_active):
        obs_i = env.dynamic_obstacles[i]
        ox, oy = obs_i[0], obs_i[1]
        otheta, ospeed = obs_i[2], obs_i[3]

        dx_world = ox - rx
        dy_world = oy - ry
        rel_x = cos_t * dx_world + sin_t * dy_world
        rel_y = -sin_t * dx_world + cos_t * dy_world

        vox = ospeed * math.cos(otheta)
        voy = ospeed * math.sin(otheta)
        rel_vx = cos_t * vox + sin_t * voy
        rel_vy = -sin_t * vox + cos_t * voy

        obstacle_set[i] = [
            rel_x / 20.0,
            rel_y / 20.0,
            rel_vx / 2.0,
            rel_vy / 2.0,
            env.cbf_filter.config.dynamic_obs_radius / 2.0,
        ]
        mask[i] = True

    obs = dict(obs)
    obs["obstacle_set"] = obstacle_set
    obs["obstacle_set_mask"] = mask
    return obs


def run_episode(
    agent: SafeSACAgent,
    env: AMRWarehouseEnv,
    episode_seed: int,
    collect_frames: bool,
) -> Dict[str, Any]:
    obs, _info = env.reset(seed=episode_seed)
    max_obstacles = getattr(agent.policy_config, "max_obstacles", 0)
    if max_obstacles > 0:
        obs = augment_observation(obs, env, max_obstacles)

    done = False
    episode_return = 0.0
    episode_length = 0
    success = False
    collision = False
    collision_type = "none"
    cbf_interventions = 0
    frames: List[np.ndarray] = []

    if collect_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if max_obstacles > 0:
            obs = augment_observation(obs, env, max_obstacles)

        done = terminated or truncated
        episode_return += float(reward)
        episode_length += 1
        success = success or bool(info.get("goal_reached", False))
        if info.get("collision", False):
            collision = True
            collision_type = str(info.get("collision_type", "unknown"))
        cbf_interventions += int(info.get("cbf_intervened", False))

        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    result: Dict[str, Any] = {
        "return": episode_return,
        "length": episode_length,
        "success": success,
        "collision": collision,
        "collision_type": collision_type,
        "cbf_interventions": cbf_interventions,
    }
    if collect_frames:
        result["frames"] = frames
    return result


def _save_episode_gif(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(
            f"[evaluate_improved.py] 'imageio' is not installed; skipping GIF save for {out_path}. "
            "Install with `pip install imageio` to enable this."
        )
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[evaluate_improved.py] Saved episode video to {out_path}")


def evaluate(args: argparse.Namespace) -> None:
    env = AMRWarehouseEnv(
        render_mode=args.render_mode,
        use_cbf_filter=not args.no_cbf_filter,
    )
    env.set_curriculum_level(args.curriculum_level)
    agent = load_agent(args.checkpoint, env, device=args.device)

    print(f"[evaluate_improved.py] Checkpoint:   {args.checkpoint}")
    print(f"[evaluate_improved.py] Device:       {args.device}")
    print(f"[evaluate_improved.py] Render mode:  {args.render_mode}")
    print(f"[evaluate_improved.py] CBF filter:   {'enabled' if not args.no_cbf_filter else 'DISABLED'}")
    print(f"[evaluate_improved.py] Curriculum:   {args.curriculum_level}")
    print(f"[evaluate_improved.py] Episodes:     {args.episodes}\n")

    collect_frames = args.render_mode == "rgb_array" and args.save_video_dir is not None
    save_video_dir = Path(args.save_video_dir) if args.save_video_dir else None

    returns: List[float] = []
    successes: List[bool] = []
    collisions: List[bool] = []
    interventions: List[int] = []

    for episode_idx in range(args.episodes):
        result = run_episode(
            agent, env, episode_seed=args.seed + episode_idx, collect_frames=collect_frames
        )

        returns.append(result["return"])
        successes.append(result["success"])
        collisions.append(result["collision"])
        interventions.append(result["cbf_interventions"])

        status = "SUCCESS" if result["success"] else ("COLLISION" if result["collision"] else "TIMEOUT")
        print(
            f"[episode {episode_idx + 1:>3}/{args.episodes}] "
            f"status={status:<10} "
            f"return={result['return']:8.2f}  "
            f"length={result['length']:4d}  "
            f"collision_type={result['collision_type']:<18} "
            f"cbf_interventions={result['cbf_interventions']:4d}"
        )

        if collect_frames and save_video_dir is not None:
            _save_episode_gif(
                result["frames"],
                save_video_dir / f"episode_{episode_idx + 1:03d}.gif",
                fps=env.metadata.get("render_fps", 10),
            )

    print("\n[evaluate_improved.py] ==== Summary over {} episodes ====".format(args.episodes))
    print(f"  Mean return:            {np.mean(returns):.2f} (+/- {np.std(returns):.2f})")
    print(f"  Success rate:           {np.mean(successes):.2%}")
    print(f"  Collision rate:         {np.mean(collisions):.2%}")
    print(f"  Mean CBF interventions: {np.mean(interventions):.2f} per episode")

    env.close()


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate(cli_args)
