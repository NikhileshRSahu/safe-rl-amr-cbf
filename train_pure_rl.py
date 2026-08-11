"""train_pure_rl.py

Train a Pure RL agent (NO CBF safety filter) for ablation comparison.
Uses the same improved architecture (attention-based obstacle encoder,
curriculum learning, deeper network) as train_improved.py, but disables
the CBF-QP filter in the environment.

This lets you fairly compare:
    - Pure RL   (this script)  vs
    - RL + CBF  (train_improved.py)
    - Classical baselines       (benchmark.py)

Usage:
    python train_pure_rl.py --timesteps 1000000 --save-dir checkpoints_pure_rl
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import trange
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from environment import AMRWarehouseEnv
from policy import ActionBounds, PolicyConfig
from safe_sac import SafeSACAgent, SafeSACConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pure RL (no CBF) agent.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--curriculum-start", type=float, default=0.3)
    parser.add_argument("--curriculum-end", type=float, default=1.0)
    parser.add_argument("--curriculum-advance-threshold", type=float, default=0.70)
    parser.add_argument("--curriculum-advance-window", type=int, default=50)
    parser.add_argument("--log-dir", type=str, default="runs_pure_rl")
    parser.add_argument("--save-dir", type=str, default="checkpoints_pure_rl")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--print-freq", type=int, default=1_000)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy_config(env: AMRWarehouseEnv) -> PolicyConfig:
    obs_space = env.observation_space
    return PolicyConfig(
        robot_state_dim=int(obs_space["robot_state"].shape[0]),
        goal_dim=int(obs_space["goal"].shape[0]),
        lidar_dim=int(obs_space["lidar"].shape[0]),
        obstacle_dim=0,
        use_attention_obstacles=True,
        max_obstacles=env.dynamic_obstacles.shape[0],
        obstacle_feature_dim=5,
        attention_heads=4,
        action_dim=int(env.action_space.shape[0]),
        critic_type="q",
        use_auto_entropy_tuning=True,
        head_hidden=(512, 512, 256),
        encoder_hidden=256,
        encoder_out=128,
        aux_heads=("collision_risk",),
        aux_hidden_dim=64,
        action_bounds=ActionBounds(v_min=-1.0, v_max=1.0, omega_min=-1.0, omega_max=1.0),
    )


def build_agent(env: AMRWarehouseEnv, args: argparse.Namespace) -> SafeSACAgent:
    policy_config = build_policy_config(env)
    sac_config = SafeSACConfig(
        gamma=args.gamma, tau=args.tau,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        max_grad_norm=args.max_grad_norm, device=args.device,
    )
    # Build expanded observation space so replay buffer stores augmented obs
    obs_space = build_observation_space_with_obstacles(env, policy_config.max_obstacles)
    return SafeSACAgent(
        policy_config=policy_config,
        observation_space=obs_space,
        action_dim=int(env.action_space.shape[0]),
        sac_config=sac_config,
        replay_buffer_size=args.buffer_size,
    )


def build_observation_space_with_obstacles(env: AMRWarehouseEnv, max_obstacles: int):
    original = env.observation_space
    new_spaces = dict(original.spaces)
    new_spaces["obstacle_set"] = __import__('gymnasium').spaces.Box(
        low=-1.0, high=1.0, shape=(max_obstacles, 5), dtype=np.float32
    )
    new_spaces["obstacle_set_mask"] = __import__('gymnasium').spaces.Box(
        low=0, high=1, shape=(max_obstacles,), dtype=bool
    )
    return __import__('gymnasium').spaces.Dict(new_spaces)


def augment_observation(
    obs: Dict[str, np.ndarray], env: AMRWarehouseEnv, max_obstacles: int
) -> Dict[str, np.ndarray]:
    rx, ry, theta = env.robot_state[0], env.robot_state[1], env.robot_state[2]
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    obstacle_set = np.zeros((max_obstacles, 5), dtype=np.float32)
    mask = np.zeros(max_obstacles, dtype=bool)
    num_active = int(min(len(env.dynamic_obstacles), max_obstacles))
    for i in range(num_active):
        obs_i = env.dynamic_obstacles[i]
        ox, oy = obs_i[0], obs_i[1]
        otheta, ospeed = obs_i[2], obs_i[3]
        dx_world, dy_world = ox - rx, oy - ry
        rel_x = cos_t * dx_world + sin_t * dy_world
        rel_y = -sin_t * dx_world + cos_t * dy_world
        vox = ospeed * math.cos(otheta)
        voy = ospeed * math.sin(otheta)
        rel_vx = cos_t * vox + sin_t * voy
        rel_vy = -sin_t * vox + cos_t * voy
        obstacle_set[i] = [rel_x / 20.0, rel_y / 20.0, rel_vx / 2.0, rel_vy / 2.0,
                           env.cbf_filter.config.dynamic_obs_radius / 2.0]
        mask[i] = True
    obs = dict(obs)
    obs["obstacle_set"] = obstacle_set
    obs["obstacle_set_mask"] = mask
    return obs


def evaluate_agent(agent, eval_env, num_episodes, seed, augment=True):
    was_training = agent.policy.training
    agent.eval()
    episode_returns, episode_lengths, successes, collisions = [], [], [], []
    for episode_idx in range(num_episodes):
        obs, _ = eval_env.reset(seed=seed + episode_idx)
        if augment:
            obs = augment_observation(obs, eval_env, agent.policy_config.max_obstacles)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_success = False
        ep_collision = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if augment:
                obs = augment_observation(obs, eval_env, agent.policy_config.max_obstacles)
            done = terminated or truncated
            ep_return += float(reward)
            ep_length += 1
            ep_success = ep_success or bool(info.get("goal_reached", False))
            ep_collision = ep_collision or bool(info.get("collision", False))
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        successes.append(float(ep_success))
        collisions.append(float(ep_collision))
    if was_training:
        agent.train()
    return {
        "mean_return": float(np.mean(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
    }


def train(args: argparse.Namespace) -> None:
    run_name = args.run_name or time.strftime("pure_rl_%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / run_name
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    # CRITICAL: Pure RL has NO CBF filter
    env = AMRWarehouseEnv(use_cbf_filter=False)
    eval_env = AMRWarehouseEnv(use_cbf_filter=False)

    curriculum_level = args.curriculum_start
    env.set_curriculum_level(curriculum_level)
    eval_env.set_curriculum_level(curriculum_level)

    agent = build_agent(env, args)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"[train_pure_rl.py] Run name:        {run_name}")
    print(f"[train_pure_rl.py] Device:          {agent.device}")
    print(f"[train_pure_rl.py] CBF filter:      DISABLED (Pure RL ablation)")
    print(f"[train_pure_rl.py] Curriculum:      start={args.curriculum_start}, end={args.curriculum_end}")

    obs, _ = env.reset(seed=args.seed)
    obs = augment_observation(obs, env, agent.policy_config.max_obstacles)

    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[float] = deque(maxlen=args.curriculum_advance_window)
    recent_collisions: Deque[float] = deque(maxlen=100)
    best_success_rate = -float("inf")
    best_eval_return = -float("inf")

    start_time = time.time()
    last_print_time = start_time
    last_print_step = 0

    step_iterator = trange(1, args.timesteps + 1, desc="train", unit="step") if _TQDM_AVAILABLE else range(1, args.timesteps + 1)

    for global_step in step_iterator:
        if global_step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = augment_observation(next_obs, env, agent.policy_config.max_obstacles)
        done = terminated or truncated

        cost = 1.0 if info.get("collision", False) else 0.0
        barrier_value = float(info.get("barrier_value", 0.0))

        agent.replay_buffer.add(
            obs=obs, action=action, reward=float(reward),
            next_obs=next_obs, done=float(terminated),
            cost=cost, barrier_value=barrier_value,
        )
        obs = next_obs
        episode_return += float(reward)
        episode_length += 1

        if done:
            episode_index += 1
            success = bool(info.get("goal_reached", False))
            collision = bool(info.get("collision", False))
            recent_returns.append(episode_return)
            recent_successes.append(float(success))
            recent_collisions.append(float(collision))
            writer.add_scalar("rollout/episode_return", episode_return, global_step)
            writer.add_scalar("rollout/episode_length", episode_length, global_step)
            writer.add_scalar("rollout/success", float(success), global_step)
            writer.add_scalar("rollout/collision", float(collision), global_step)
            writer.add_scalar("rollout/curriculum_level", curriculum_level, global_step)

            if (len(recent_successes) >= args.curriculum_advance_window
                    and curriculum_level < args.curriculum_end):
                recent_success_rate = float(np.mean(list(recent_successes)))
                if recent_success_rate >= args.curriculum_advance_threshold:
                    curriculum_level = min(curriculum_level + 0.1, args.curriculum_end)
                    env.set_curriculum_level(curriculum_level)
                    eval_env.set_curriculum_level(curriculum_level)
                    print(f"\n[CURRICULUM] Advanced to level {curriculum_level:.1f} at step {global_step}")
                    recent_successes.clear()

            obs, _ = env.reset()
            obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
            episode_return = 0.0
            episode_length = 0

        if (len(agent.replay_buffer) > args.batch_size
                and global_step > args.learning_starts
                and global_step % args.train_freq == 0):
            for _ in range(args.updates_per_step):
                stats = agent.update(args.batch_size)
            for key, value in stats.items():
                writer.add_scalar(f"train/{key}", value, global_step)

        if global_step % args.eval_freq == 0:
            eval_stats = evaluate_agent(agent, eval_env, num_episodes=args.eval_episodes,
                                         seed=args.seed + 10_000)
            for key, value in eval_stats.items():
                writer.add_scalar(f"eval/{key}", value, global_step)
            print(
                f"\n[eval @ step {global_step:>9,}] "
                f"return={eval_stats['mean_return']:.2f}  "
                f"success_rate={eval_stats['success_rate']:.2%}  "
                f"collision_rate={eval_stats['collision_rate']:.2%}  "
                f"length={eval_stats['mean_length']:.1f}  "
                f"curriculum={curriculum_level:.1f}"
            )
            improved = (eval_stats["success_rate"] > best_success_rate
                        or (eval_stats["success_rate"] == best_success_rate
                            and eval_stats["mean_return"] > best_eval_return))
            if improved:
                best_success_rate = eval_stats["success_rate"]
                best_eval_return = eval_stats["mean_return"]
                best_path = save_dir / "best_model.pt"
                agent.save_models(best_path)
                print(f"[train_pure_rl.py] New best model saved to {best_path}")
            periodic_path = save_dir / f"model_step_{global_step}.pt"
            agent.save_models(periodic_path)

        if not _TQDM_AVAILABLE and global_step % args.print_freq == 0:
            now = time.time()
            fps = (global_step - last_print_step) / max(now - last_print_time, 1e-8)
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            mean_success = float(np.mean(list(recent_successes)[-100:])) if recent_successes else float("nan")
            mean_collision = float(np.mean(recent_collisions)) if recent_collisions else float("nan")
            elapsed = now - start_time
            print(
                f"[step {global_step:>9,}/{args.timesteps:,}] "
                f"fps={fps:6.1f}  elapsed={elapsed/60:6.1f}m  "
                f"episodes={episode_index:>6,}  "
                f"recent_return={mean_return:7.2f}  "
                f"recent_success={mean_success:6.2%}  "
                f"recent_collision={mean_collision:6.2%}  "
                f"buffer={len(agent.replay_buffer):>9,}  "
                f"curriculum={curriculum_level:.1f}"
            )
            last_print_time = now
            last_print_step = global_step

    final_path = save_dir / "final_model.pt"
    agent.save_models(final_path)
    print(f"[train_pure_rl.py] Training complete. Final model saved to {final_path}")
    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
