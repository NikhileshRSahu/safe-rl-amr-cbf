"""train.py

Main executable training script for the Hybrid SAC + CBF Safe RL AMR agent.

Ties together ``environment.AMRWarehouseEnv`` (native CBF-QP filtered
dynamics), ``policy.SafeRLPolicy`` / ``policy.PolicyConfig``, and
``safe_sac.SafeSACAgent`` into a standard off-policy training loop with
periodic deterministic evaluation, TensorBoard logging, and best-model
checkpointing.

Usage:
    python train.py --timesteps 500000 --batch-size 256 --eval-freq 10000
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import trange

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency.
    _TQDM_AVAILABLE = False

from environment import AMRWarehouseEnv
from policy import ActionBounds, PolicyConfig
from safe_sac import SafeSACAgent, SafeSACConfig


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training run.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a Hybrid SAC + CBF agent on AMRWarehouseEnv."
    )

    # -- Run control ------------------------------------------------------- #
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help="Total environment steps to train for."
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5_000,
        help="Number of environment steps of pure exploration before SAC updates begin.",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Number of environment steps between each agent.update() call.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="SAC minibatch size.")
    parser.add_argument(
        "--buffer-size", type=int, default=1_000_000, help="Replay buffer capacity."
    )

    # -- Evaluation ---------------------------------------------------------- #
    parser.add_argument(
        "--eval-freq", type=int, default=10_000, help="Environment steps between evaluation runs."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of episodes per evaluation run."
    )

    # -- SAC / optimization hyperparameters ---------------------------------- #
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network Polyak coefficient.")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor optimizer learning rate.")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Critic optimizer learning rate.")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="Entropy temperature learning rate.")
    parser.add_argument(
        "--max-grad-norm", type=float, default=10.0, help="Global gradient-clipping norm."
    )

    # -- Logging / checkpointing ---------------------------------------------- #
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory.")
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="Directory for saved model checkpoints."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (defaults to a timestamp). Used for the log/checkpoint subfolder.",
    )
    parser.add_argument(
        "--print-freq", type=int, default=1_000, help="Environment steps between console progress prints."
    )

    # -- Device / environment -------------------------------------------------- #
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for the agent's networks.",
    )
    parser.add_argument(
        "--no-cbf-filter",
        action="store_true",
        help="Disable the native CBF-QP safety filter in the environment (unfiltered baseline).",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int) -> None:
    """Seeds every RNG involved in the training run for reproducibility.

    Args:
        seed: Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Agent / environment construction
# --------------------------------------------------------------------------- #

def build_policy_config(env: AMRWarehouseEnv) -> PolicyConfig:
    """Builds the ``PolicyConfig`` matching ``env``'s observation/action layout.

    Configures ``critic_type="q"`` (required for SAC's twin-Q target) and
    ``use_auto_entropy_tuning=True`` (learned temperature), and sets
    ``action_bounds`` to the environment's normalized ``[-1, 1]`` action
    range so the actor's tanh-squashed output is directly usable by
    ``AMRWarehouseEnv.step`` without any extra rescaling.

    Args:
        env: The (unwrapped) environment instance, used to read observation
            dimensionalities off ``env.observation_space``.

    Returns:
        A fully populated ``PolicyConfig``.
    """
    obs_space = env.observation_space
    robot_state_dim = int(obs_space["robot_state"].shape[0])
    goal_dim = int(obs_space["goal"].shape[0])
    lidar_dim = int(obs_space["lidar"].shape[0])

    return PolicyConfig(
        robot_state_dim=robot_state_dim,
        goal_dim=goal_dim,
        lidar_dim=lidar_dim,
        obstacle_dim=0,
        action_dim=int(env.action_space.shape[0]),
        critic_type="q",
        use_auto_entropy_tuning=True,
        # Normalized [-1, 1] bounds for both [v, omega] so the actor's
        # squashed output matches AMRWarehouseEnv's expected action_space
        # (AMRWarehouseEnv internally rescales this to physical V/OMEGA
        # bounds before the CBF filter sees it). Note: v_min == v_max
        # would be a degenerate (zero-width) action dimension, so this is
        # symmetric [-1, 1] rather than the literal [-1, -1] sometimes
        # mis-typed for this field.
        action_bounds=ActionBounds(v_min=-1.0, v_max=1.0, omega_min=-1.0, omega_max=1.0),
    )


def build_agent(env: AMRWarehouseEnv, args: argparse.Namespace) -> SafeSACAgent:
    """Constructs the ``SafeSACAgent`` wired up to ``env`` and CLI args.

    Args:
        env: The training environment (used for its observation/action
            spaces).
        args: Parsed command-line arguments.

    Returns:
        A fully initialized ``SafeSACAgent``.
    """
    policy_config = build_policy_config(env)
    sac_config = SafeSACConfig(
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )
    return SafeSACAgent(
        policy_config=policy_config,
        observation_space=env.observation_space,
        action_dim=int(env.action_space.shape[0]),
        sac_config=sac_config,
        replay_buffer_size=args.buffer_size,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_agent(
    agent: SafeSACAgent,
    eval_env: AMRWarehouseEnv,
    num_episodes: int,
    seed: int,
) -> Dict[str, float]:
    """Runs deterministic evaluation episodes and aggregates diagnostics.

    Args:
        agent: The agent under evaluation. ``select_action`` is called with
            ``deterministic=True``.
        eval_env: A separate environment instance used only for evaluation,
            so evaluation never perturbs the training environment's state
            or RNG stream.
        num_episodes: Number of episodes to run.
        seed: Base seed for the evaluation episodes (each episode uses
            ``seed + episode_index`` so evaluation is reproducible but not
            degenerate across episodes).

    Returns:
        Dict with keys ``mean_return``, ``mean_length``, ``success_rate``,
        ``collision_rate``, and ``mean_cbf_interventions`` (average number
        of CBF-filter interventions per episode).
    """
    was_training = agent.policy.training
    agent.eval()

    episode_returns: list = []
    episode_lengths: list = []
    successes: list = []
    collisions: list = []
    cbf_intervention_counts: list = []

    for episode_idx in range(num_episodes):
        obs, _info = eval_env.reset(seed=seed + episode_idx)
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_success = False
        episode_collision = False
        episode_cbf_interventions = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            episode_return += float(reward)
            episode_length += 1
            episode_success = episode_success or bool(info.get("goal_reached", False))
            episode_collision = episode_collision or bool(info.get("collision", False))
            episode_cbf_interventions += int(info.get("cbf_intervened", False))

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        successes.append(float(episode_success))
        collisions.append(float(episode_collision))
        cbf_intervention_counts.append(episode_cbf_interventions)

    if was_training:
        agent.train()

    return {
        "mean_return": float(np.mean(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_cbf_interventions": float(np.mean(cbf_intervention_counts)),
    }


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    """Runs the full training loop: rollout collection, SAC updates, and eval.

    Args:
        args: Parsed command-line arguments.
    """
    run_name = args.run_name or time.strftime("safe_sac_%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / run_name
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    env = AMRWarehouseEnv(use_cbf_filter=not args.no_cbf_filter)
    eval_env = AMRWarehouseEnv(use_cbf_filter=not args.no_cbf_filter)

    agent = build_agent(env, args)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"[train.py] Run name:        {run_name}")
    print(f"[train.py] Device:          {agent.device}")
    print(f"[train.py] Log dir:         {log_dir}")
    print(f"[train.py] Checkpoint dir:  {save_dir}")
    print(f"[train.py] CBF filter:      {'enabled' if not args.no_cbf_filter else 'DISABLED'}")

    obs, _info = env.reset(seed=args.seed)

    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[float] = deque(maxlen=100)
    recent_collisions: Deque[float] = deque(maxlen=100)

    best_success_rate = -float("inf")
    best_eval_return = -float("inf")

    start_time = time.time()
    last_print_time = start_time
    last_print_step = 0

    step_iterator = (
        trange(1, args.timesteps + 1, desc="train", unit="step")
        if _TQDM_AVAILABLE
        else range(1, args.timesteps + 1)
    )

    for global_step in step_iterator:
        # -- 1. Act -------------------------------------------------------- #
        if global_step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        cost = 1.0 if info.get("collision", False) else 0.0
        barrier_value = float(info.get("barrier_value", 0.0))

        agent.replay_buffer.add(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=float(terminated),
            cost=cost,
            barrier_value=barrier_value,
        )

        obs = next_obs
        episode_return += float(reward)
        episode_length += 1

        # -- 2. Episode bookkeeping ------------------------------------------ #
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

            obs, _info = env.reset()
            episode_return = 0.0
            episode_length = 0

        # -- 3. SAC update -------------------------------------------------- #
        if (
            len(agent.replay_buffer) > args.batch_size
            and global_step > args.learning_starts
            and global_step % args.train_freq == 0
        ):
            stats = agent.update(args.batch_size)
            for key, value in stats.items():
                writer.add_scalar(f"train/{key}", value, global_step)

        # -- 4. Evaluation ---------------------------------------------------- #
        if global_step % args.eval_freq == 0:
            eval_stats = evaluate_agent(
                agent, eval_env, num_episodes=args.eval_episodes, seed=args.seed + 10_000
            )
            for key, value in eval_stats.items():
                writer.add_scalar(f"eval/{key}", value, global_step)

            print(
                f"\n[eval @ step {global_step:>9,}] "
                f"return={eval_stats['mean_return']:.2f}  "
                f"success_rate={eval_stats['success_rate']:.2%}  "
                f"collision_rate={eval_stats['collision_rate']:.2%}  "
                f"avg_cbf_interventions={eval_stats['mean_cbf_interventions']:.2f}  "
                f"length={eval_stats['mean_length']:.1f}"
            )

            improved = (
                eval_stats["success_rate"] > best_success_rate
                or (
                    eval_stats["success_rate"] == best_success_rate
                    and eval_stats["mean_return"] > best_eval_return
                )
            )
            if improved:
                best_success_rate = eval_stats["success_rate"]
                best_eval_return = eval_stats["mean_return"]
                best_path = save_dir / "best_model.pt"
                agent.save_models(best_path)
                print(
                    f"[train.py] New best model saved to {best_path} "
                    f"(success_rate={best_success_rate:.2%}, return={best_eval_return:.2f})"
                )

        # -- 5. Console progress --------------------------------------------- #
        if not _TQDM_AVAILABLE and global_step % args.print_freq == 0:
            now = time.time()
            fps = (global_step - last_print_step) / max(now - last_print_time, 1e-8)
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            mean_success = float(np.mean(recent_successes)) if recent_successes else float("nan")
            mean_collision = float(np.mean(recent_collisions)) if recent_collisions else float("nan")
            elapsed = now - start_time
            print(
                f"[step {global_step:>9,}/{args.timesteps:,}] "
                f"fps={fps:6.1f}  elapsed={elapsed/60:6.1f}m  "
                f"episodes={episode_index:>6,}  "
                f"recent_return={mean_return:7.2f}  "
                f"recent_success={mean_success:6.2%}  "
                f"recent_collision={mean_collision:6.2%}  "
                f"buffer={len(agent.replay_buffer):>9,}"
            )
            last_print_time = now
            last_print_step = global_step

    # -- Final checkpoint & cleanup ------------------------------------------- #
    final_path = save_dir / "final_model.pt"
    agent.save_models(final_path)
    print(f"[train.py] Training complete. Final model saved to {final_path}")

    writer.close()
    env.close()
    eval_env.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)