"""train.py

Canonical training script for the Hybrid SAC + CBF Safe RL AMR agent.

This is the single training entry point intended for the final project repository.
It contains the architecture fixes used in the final experiments plus corrected
checkpoint/resume, evaluation, entropy-floor, and critic-warmup handling.

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
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from config import V_MIN, V_MAX, OMEGA_MAX, GOAL_TOLERANCE, HER_ENSURE_HINDSIGHT_SUCCESS

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):  # TensorBoard is optional for headless smoke tests.
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def flush(self): pass
        def close(self): pass

try:
    from tqdm import trange

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency.
    _TQDM_AVAILABLE = False

from environment import AMRWarehouseEnv
from her_buffer import HEREpisodeBuffer
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
        "--target-entropy",
        type=float,
        default=None,
        help="Target entropy for SAC's automatic temperature tuning. Defaults to -action_dim (=-2 for this "
        "env). Raise toward 0 (e.g. -1.0) for more exploration; lower for faster convergence once the "
        "policy has already learned basic navigation. Diagnostic note: at the default of -2 the policy "
        "was reaching target entropy by step ~18k (before learning goal-directed behavior) in 30k runs, "
        "causing premature exploitation of a spin-in-place fixed point.",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=None,
        help="Optional hard floor on SAC entropy temperature alpha. Leave unset for the final "
        "selected training configuration unless deliberately running an alpha-floor ablation. "
        "When set, the real entropy_temperature.log_alpha parameter is clamped after updates.",
    )
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
    parser.add_argument(
        "--lookahead-distance",
        type=float,
        default=None,
        help="Lookahead offset (m) used when computing CBF Lie derivatives. Overrides the value "
        "wired in build_filter_from_config() (config.LOOKAHEAD_DISTANCE=0.25). Set to 0.0 to "
        "reproduce the historical L=0 behavior where the omega column of Lg_h is identically "
        "zero and the QP can only brake, never steer -- useful as a controlled A/B baseline.",
    )
    parser.add_argument(
        "--enable-cbf-actor-reg",
        action="store_true",
        help="Enable the CBF actor-consistency regularizer added to the SAC actor loss "
        "(see cbf.CBFSafetyFilter.actor_consistency_loss). Opt-in and OFF by default so "
        "existing runs/configs are reproduced exactly unless explicitly requested. Has no "
        "effect if --no-cbf-filter is also set (nothing to be consistent with).",
    )
    parser.add_argument(
        "--cbf-reg-weight",
        type=float,
        default=None,
        help="Weight on the CBF actor-consistency regularizer. Defaults to "
        "config.CBF_ACTOR_REG_WEIGHT if --enable-cbf-actor-reg is set and this is omitted.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.05,
        help="Multiplier applied to sampled rewards inside the SAC critic/actor "
        "objective only (SafeSACConfig.reward_scale; standard SAC 'reward_scale' "
        "hyperparameter). Does not affect env rewards, episode returns, or "
        "eval/TensorBoard logging of raw reward -- only the internal Q-value "
        "scale the actor optimizes against. Default 0.05 brings this project's "
        "theoretical Q ceiling (~2000, from mean_step_reward/(1-gamma) with the "
        "current REWARD_CONFIG) down to ~100. See SafeSACConfig.reward_scale "
        "docstring for the measured critic-instability evidence motivating this.",
    )
    parser.add_argument(
        "--critic-warmup-steps",
        type=int,
        default=0,
        help="Number of initial agent.update() calls that update only the "
        "critic (freeze_actor=True), before actor/alpha updates begin. "
        "Meant to pair with --warm-start-checkpoint: a freshly-initialized "
        "critic paired with a pretrained (e.g. behavior-cloned) actor "
        "produces noisy early Q-estimates that can immediately erode the "
        "pretrained actor's good behavior if the actor starts updating "
        "against that noise right away (observed empirically, 2026-08: "
        "eval collision rate swung 20%%->90%%->10%% and progress-per-episode "
        "collapsed within 40k steps of naive fine-tuning with no warmup). "
        "0 disables this (actor updates from step 1, the pre-existing "
        "behavior). A few thousand is a reasonable starting point when "
        "using --warm-start-checkpoint.",
    )
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=None,
        help="Number of env steps over which AMRWarehouseEnv.curriculum_level is "
        "ramped linearly from --curriculum-start-level to 1.0, then held at 1.0 "
        "for the rest of training. Defaults to 40%% of --timesteps. Pass 0 to "
        "disable the curriculum (curriculum_level stays at 1.0 throughout, the "
        "prior un-curriculumed behavior). The env already implements "
        "set_curriculum_level() (scales active dynamic-obstacle count and lidar "
        "noise/dropout) but train.py never called it before this flag existed, "
        "so every prior run trained at full difficulty (all dynamic obstacles, "
        "full sensor noise) from step 0 -- including during the pure-random "
        "--learning-starts warmup, when the replay buffer is seeded almost "
        "entirely with collision transitions in a fully-noisy, fully-populated "
        "warehouse. Verified via hand-scripted-policy comparison (2026-08) that "
        "a lidar-aware obstacle-avoiding goal-seeking controller scores clearly "
        "better under the *existing* reward function than either a naive "
        "goal-blind controller or the collapsed 'turn hard, rarely crash' "
        "policy these runs converge to -- i.e. the reward function is not the "
        "problem, so easing exploration via curriculum is the next lever.",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint used to initialize compatible policy/target weights before "
        "starting a new training run. Optimizer states are not restored in warm-start mode. "
        "For an exact continuation including optimizer state and update count, use "
        "--resume-checkpoint instead.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume policy/target/optimizer/update-count state from a checkpoint saved by "
        "SafeSACAgent.save_models(). Replay-buffer contents are not stored in the checkpoint, "
        "so use --resume-replay-warmup to refill the replay buffer before updates restart. "
        "Do not combine this with --warm-start-checkpoint.",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Environment step represented by --resume-checkpoint. With --timesteps 120000 and "
        "--resume-step 90000, training continues from 90001 through 120000.",
    )
    parser.add_argument(
        "--resume-replay-warmup",
        type=int,
        default=5_000,
        help="New transitions to collect after --resume-checkpoint before SAC updates resume, because "
        "the replay buffer itself is not checkpointed. The resumed policy is used during refill; "
        "actions are not replaced with random exploration.",
    )
    parser.add_argument(
        "--curriculum-start-level",
        type=float,
        default=0.15,
        help="Starting AMRWarehouseEnv.curriculum_level (0-1) at step 0 when "
        "--curriculum-steps > 0. 0.15 keeps ~1-2 dynamic obstacles active "
        "(NUM_DYNAMIC_OBSTACLES * level) and reduced lidar noise/dropout early "
        "on, rather than the full set from step 0.",
    )
    parser.add_argument(
        "--her-ensure-hindsight-success",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=None,
        help="Guarantee at least one genuine reached=True transition per distinct HER "
        "relabel-goal index per episode (see her_buffer.HEREpisodeBuffer.relabel_and_push). "
        "Defaults to config.HER_ENSURE_HINDSIGHT_SUCCESS (True) if omitted. Pass "
        "--her-ensure-hindsight-success=false to reproduce the previous, success-starved "
        "behavior for an A/B comparison.",
    )
    parser.add_argument(
        "--her-strategy",
        type=str,
        default="future",
        choices=["future", "episode"],
        help="HER relabel-goal sampling pool. 'future' (default, unchanged prior behavior): "
        "sample candidate goals only from steps after the current one in the episode. "
        "'episode': sample from anywhere in the episode (before or after). Added after "
        "measure_episode_displacement.py showed 'future' skip_ratio averaging ~78%% on the "
        "prior 30k-step checkpoint while 'episode' would average ~25%% on the same rollouts. "
        "Default stays 'future' so existing runs are unaffected unless this is passed explicitly.",
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

def build_policy_config(env: AMRWarehouseEnv, args: argparse.Namespace) -> PolicyConfig:
    """Builds the ``PolicyConfig`` matching ``env``'s observation/action layout.

    Configures ``critic_type="q"`` (required for SAC's twin-Q target) and
    ``use_auto_entropy_tuning=True`` (learned temperature), and sets
    ``action_bounds`` to the environment's normalized ``[-1, 1]`` action
    range so the actor's tanh-squashed output is directly usable by
    ``AMRWarehouseEnv.step`` without any extra rescaling.

    Args:
        env: The (unwrapped) environment instance, used to read observation
            dimensionalities off ``env.observation_space``.
        args: Parsed CLI args. ``args.target_entropy`` (if not None) overrides
            the default ``-action_dim`` target used by automatic alpha tuning.

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
        # Allow the CLI to override the default -action_dim=-2 target. At -2
        # the policy was reaching target entropy by ~18k gradient steps in
        # 30k runs -- before it had learned goal-directed behavior -- causing
        # premature exploitation of a spin-in-place fixed point.
        target_entropy=args.target_entropy,
    )


def build_agent(env: AMRWarehouseEnv, args: argparse.Namespace) -> SafeSACAgent:
    """Constructs the ``SafeSACAgent`` wired up to ``env`` and CLI args.

    Args:
        env: The training environment (used for its observation/action
            spaces, and -- if ``--enable-cbf-actor-reg`` is set -- its CBF
            filter instance).
        args: Parsed command-line arguments.

    Returns:
        A fully initialized ``SafeSACAgent``.
    """
    from config import NUM_DYNAMIC_OBSTACLES, CBF_ACTOR_REG_WEIGHT

    policy_config = build_policy_config(env, args)
    cbf_reg_weight = args.cbf_reg_weight if args.cbf_reg_weight is not None else CBF_ACTOR_REG_WEIGHT
    sac_config = SafeSACConfig(
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        cbf_reg_weight=cbf_reg_weight if args.enable_cbf_actor_reg else 0.0,
        reward_scale=args.reward_scale,
    )
    use_cbf_reg = args.enable_cbf_actor_reg and not args.no_cbf_filter
    if args.enable_cbf_actor_reg and args.no_cbf_filter:
        print(
            "[train.py] --enable-cbf-actor-reg was set but --no-cbf-filter disables the "
            "filter entirely; the regularizer has nothing to be consistent with, so it "
            "is being skipped for this run."
        )
    agent = SafeSACAgent(
        policy_config=policy_config,
        observation_space=env.observation_space,
        action_dim=int(env.action_space.shape[0]),
        sac_config=sac_config,
        replay_buffer_size=args.buffer_size,
        cbf_filter=env.cbf_filter if use_cbf_reg else None,
        max_dynamic_obstacles=NUM_DYNAMIC_OBSTACLES if use_cbf_reg else 0,
    )
    if args.warm_start_checkpoint:
        import torch as _torch
        warm_ckpt = _torch.load(args.warm_start_checkpoint, map_location=args.device, weights_only=False)
        state_dict = warm_ckpt["policy_state_dict"] if "policy_state_dict" in warm_ckpt else warm_ckpt
        # strict=False allows actor-only / BC checkpoints as well as full
        # SafeSAC checkpoints. Any compatible keys present in the checkpoint
        # are loaded; optimizer states are intentionally not restored here.
        missing_policy, unexpected_policy = agent.policy.load_state_dict(state_dict, strict=False)
        missing_target, unexpected_target = agent.target_policy.load_state_dict(state_dict, strict=False)
        print(f"[train.py] Warm-started actor/target from {args.warm_start_checkpoint}")
        print(f"[train.py]   (not loaded, kept at fresh init: {sorted(set(k.split('.')[0] for k in missing_policy))})")
    return agent


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

def evaluate_trajectories(
    agent: SafeSACAgent,
    eval_env: AMRWarehouseEnv,
    num_episodes: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, float]:
    """Run deterministic evaluation and save detailed trajectories.

    This diagnostic is designed to determine whether failures are:
      1. Early collisions,
      2. Late collisions after making progress,
      3. Timeout / wandering behavior,
      4. Successful navigation.

    Saves:
        - trajectory_steps.csv
        - episode_summary.csv
        - trajectory_plots.png

    Returns:
        The same aggregate evaluation metrics formerly produced by
        ``evaluate_agent``. Keeping metrics and trajectory diagnostics in a
        single pass avoids running every evaluation episode twice.
    """

    import csv
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = agent.policy.training
    agent.eval()

    trajectory_rows = []
    episode_rows = []

    for episode_idx in range(num_episodes):

        obs, _info = eval_env.reset(seed=seed + episode_idx)

        done = False
        episode_return = 0.0
        episode_length = 0

        start_distance = float(
            np.linalg.norm(
                eval_env.robot_state[:2] - eval_env.goal_pos
            )
        )

        min_distance = start_distance
        max_progress = 0.0

        collision = False
        collision_type = "none"
        success = False
        episode_cbf_interventions = 0

        while not done:

            # ---------------------------------------------------------
            # State BEFORE taking action
            # ---------------------------------------------------------

            robot_x = float(eval_env.robot_state[0])
            robot_y = float(eval_env.robot_state[1])
            robot_theta = float(eval_env.robot_state[2])

            goal_x = float(eval_env.goal_pos[0])
            goal_y = float(eval_env.goal_pos[1])

            distance_before = float(
                np.linalg.norm(
                    eval_env.robot_state[:2] - eval_env.goal_pos
                )
            )

            # ---------------------------------------------------------
            # Deterministic policy action
            # ---------------------------------------------------------

            action = agent.select_action(
                obs,
                deterministic=True,
            )

            action = np.asarray(action, dtype=np.float32)

            # ---------------------------------------------------------
            # Environment step
            # ---------------------------------------------------------

            next_obs, reward, terminated, truncated, info = eval_env.step(
                action
            )

            done = terminated or truncated

            episode_length += 1
            episode_return += float(reward)

            # ---------------------------------------------------------
            # State AFTER environment step
            # ---------------------------------------------------------

            next_robot_x = float(eval_env.robot_state[0])
            next_robot_y = float(eval_env.robot_state[1])
            next_robot_theta = float(eval_env.robot_state[2])

            distance_after = float(
                info.get(
                    "distance_to_goal",
                    np.linalg.norm(
                        eval_env.robot_state[:2] - eval_env.goal_pos
                    ),
                )
            )

            min_distance = min(min_distance, distance_after)

            progress = start_distance - distance_after
            max_progress = max(max_progress, progress)

            # ---------------------------------------------------------
            # Terminal information
            # ---------------------------------------------------------

            if info.get("collision", False):
                collision = True
                collision_type = info.get(
                    "collision_type",
                    "unknown",
                )

            if info.get("goal_reached", False):
                success = True

            episode_cbf_interventions += int(info.get("cbf_intervened", False))

            # ---------------------------------------------------------
            # Record timestep
            # ---------------------------------------------------------

            trajectory_rows.append(
                {
                    "episode": episode_idx,
                    "step": episode_length,

                    "robot_x": robot_x,
                    "robot_y": robot_y,
                    "robot_theta": robot_theta,

                    "next_robot_x": next_robot_x,
                    "next_robot_y": next_robot_y,
                    "next_robot_theta": next_robot_theta,

                    "goal_x": goal_x,
                    "goal_y": goal_y,

                    "distance_before": distance_before,
                    "distance_after": distance_after,

                    "progress_from_start": progress,

                    "action_linear": float(action[0]),
                    "action_angular": float(action[1]),

                    "safe_action_linear": float(
                        info["safe_action"][0]
                    ),
                    "safe_action_angular": float(
                        info["safe_action"][1]
                    ),

                    "v_actual": float(
                        info.get("v_actual", 0.0)
                    ),
                    "omega_actual": float(
                        info.get("omega_actual", 0.0)
                    ),

                    "reward": float(reward),

                    "collision": bool(
                        info.get("collision", False)
                    ),

                    "collision_type": info.get(
                        "collision_type",
                        "none",
                    ),

                    "goal_reached": bool(
                        info.get("goal_reached", False)
                    ),

                    "cbf_intervened": bool(
                        info.get("cbf_intervened", False)
                    ),

                    "cbf_num_active_constraints": int(
                        info.get(
                            "cbf_num_active_constraints",
                            0,
                        )
                    ),

                    "barrier_value": float(
                        info.get(
                            "barrier_value",
                            0.0,
                        )
                    ),

                    "cbf_slack": float(
                        info.get("cbf_slack") if info.get("cbf_slack") is not None else 0.0
                    ),

                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            obs = next_obs

        # -------------------------------------------------------------
        # Episode classification
        # -------------------------------------------------------------

        if success:
            outcome = "success"

        elif collision:
            outcome = "collision"

        elif episode_length >= eval_env.max_episode_steps:
            outcome = "timeout"

        else:
            outcome = "other"

        # -------------------------------------------------------------
        # Episode summary
        # -------------------------------------------------------------

        episode_rows.append(
            {
                "episode": episode_idx,

                "outcome": outcome,

                "length": episode_length,

                "return": episode_return,

                "start_distance": start_distance,

                "final_distance": distance_after,

                "minimum_distance": min_distance,

                "max_progress": max_progress,

                "success": success,

                "collision": collision,

                "collision_type": collision_type,

                "final_x": float(eval_env.robot_state[0]),
                "final_y": float(eval_env.robot_state[1]),

                "goal_x": float(eval_env.goal_pos[0]),
                "goal_y": float(eval_env.goal_pos[1]),

                "cbf_interventions": episode_cbf_interventions,
            }
        )

    # -----------------------------------------------------------------
    # Restore training mode
    # -----------------------------------------------------------------

    if was_training:
        agent.train()

    # -----------------------------------------------------------------
    # Save timestep-level CSV
    # -----------------------------------------------------------------

    trajectory_file = output_dir / "trajectory_steps.csv"

    if trajectory_rows:

        with open(
            trajectory_file,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=trajectory_rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(trajectory_rows)

    # -----------------------------------------------------------------
    # Save episode summary CSV
    # -----------------------------------------------------------------

    summary_file = output_dir / "episode_summary.csv"

    if episode_rows:

        with open(
            summary_file,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=episode_rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(episode_rows)

    # -----------------------------------------------------------------
    # Print diagnostic summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 75)
    print("TRAJECTORY DIAGNOSTIC")
    print("=" * 75)

    for row in episode_rows:

        print(
            f"Episode {row['episode']:>3}: "
            f"{row['outcome']:<9} "
            f"length={row['length']:>4} "
            f"start_dist={row['start_distance']:.2f} "
            f"final_dist={row['final_distance']:.2f} "
            f"min_dist={row['minimum_distance']:.2f} "
            f"progress={row['max_progress']:.2f}"
            + (
                f" collision={row['collision_type']}"
                if row["collision"]
                else ""
            )
        )

    # -----------------------------------------------------------------
    # Aggregate diagnostics
    # -----------------------------------------------------------------

    collision_lengths = [
        r["length"]
        for r in episode_rows
        if r["outcome"] == "collision"
    ]

    success_lengths = [
        r["length"]
        for r in episode_rows
        if r["outcome"] == "success"
    ]

    timeout_lengths = [
        r["length"]
        for r in episode_rows
        if r["outcome"] == "timeout"
    ]

    print("\n--- Outcome counts ---")

    for outcome in [
        "success",
        "collision",
        "timeout",
        "other",
    ]:

        count = sum(
            r["outcome"] == outcome
            for r in episode_rows
        )

        print(
            f"{outcome:<10}: "
            f"{count}/{len(episode_rows)}"
        )

    print("\n--- Episode lengths ---")

    if collision_lengths:
        print(
            f"Collision: mean={np.mean(collision_lengths):.1f}, "
            f"median={np.median(collision_lengths):.1f}"
        )

    if success_lengths:
        print(
            f"Success:   mean={np.mean(success_lengths):.1f}, "
            f"median={np.median(success_lengths):.1f}"
        )

    if timeout_lengths:
        print(
            f"Timeout:   mean={np.mean(timeout_lengths):.1f}, "
            f"median={np.median(timeout_lengths):.1f}"
        )

    # -----------------------------------------------------------------
    # Collision type distribution
    # -----------------------------------------------------------------

    collision_types = {}

    for row in episode_rows:

        if row["collision"]:

            ctype = row["collision_type"]

            collision_types[ctype] = (
                collision_types.get(ctype, 0) + 1
            )

    print("\n--- Collision types ---")

    if collision_types:

        for ctype, count in collision_types.items():

            print(
                f"{ctype:<20}: "
                f"{count}"
            )

    else:

        print("No collisions.")

    # -----------------------------------------------------------------
    # Plot trajectories
    # -----------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 11),
    )

    # ================================================================
    # Plot 1: XY trajectories
    # ================================================================

    ax = axes[0, 0]

    for episode_idx in range(num_episodes):

        rows = [
            r
            for r in trajectory_rows
            if r["episode"] == episode_idx
        ]

        if not rows:
            continue

        xs = [r["robot_x"] for r in rows]
        ys = [r["robot_y"] for r in rows]

        ax.plot(
            xs,
            ys,
            linewidth=1.5,
            label=f"Ep {episode_idx}",
        )

        ax.scatter(
            xs[0],
            ys[0],
            marker="o",
            s=35,
        )

        ax.scatter(
            xs[-1],
            ys[-1],
            marker="x",
            s=50,
        )

        ax.scatter(
            rows[0]["goal_x"],
            rows[0]["goal_y"],
            marker="*",
            s=100,
        )

    ax.set_title("Robot Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # ================================================================
    # Plot 2: Distance to goal
    # ================================================================

    ax = axes[0, 1]

    for episode_idx in range(num_episodes):

        rows = [
            r
            for r in trajectory_rows
            if r["episode"] == episode_idx
        ]

        if not rows:
            continue

        steps = [
            r["step"]
            for r in rows
        ]

        distances = [
            r["distance_after"]
            for r in rows
        ]

        ax.plot(
            steps,
            distances,
            label=f"Ep {episode_idx}",
        )

    ax.set_title("Distance to Goal")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Distance")
    ax.grid(True)

    # ================================================================
    # Plot 3: Actions
    # ================================================================

    ax = axes[1, 0]

    for episode_idx in range(
        min(num_episodes, 5)
    ):

        rows = [
            r
            for r in trajectory_rows
            if r["episode"] == episode_idx
        ]

        if not rows:
            continue

        steps = [
            r["step"]
            for r in rows
        ]

        linear = [
            r["action_linear"]
            for r in rows
        ]

        angular = [
            r["action_angular"]
            for r in rows
        ]

        ax.plot(
            steps,
            linear,
            label=f"Ep {episode_idx} linear",
        )

        ax.plot(
            steps,
            angular,
            linestyle="--",
            label=f"Ep {episode_idx} angular",
        )

    ax.set_title("Policy Actions")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Normalized Action")
    ax.grid(True)

    # ================================================================
    # Plot 4: Barrier value
    # ================================================================

    ax = axes[1, 1]

    for episode_idx in range(
        min(num_episodes, 5)
    ):

        rows = [
            r
            for r in trajectory_rows
            if r["episode"] == episode_idx
        ]

        if not rows:
            continue

        steps = [
            r["step"]
            for r in rows
        ]

        barriers = [
            r["barrier_value"]
            for r in rows
        ]

        ax.plot(
            steps,
            barriers,
            label=f"Ep {episode_idx}",
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title("CBF Barrier Value")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Barrier Value")
    ax.grid(True)

    plt.tight_layout()

    plot_file = output_dir / "trajectory_plots.png"

    fig.savefig(
        plot_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nFiles saved:")
    print(f"  {trajectory_file}")
    print(f"  {summary_file}")
    print(f"  {plot_file}")

    print("=" * 75)

    return {
        "mean_return": float(np.mean([r["return"] for r in episode_rows])),
        "mean_length": float(np.mean([r["length"] for r in episode_rows])),
        "success_rate": float(np.mean([float(r["success"]) for r in episode_rows])),
        "collision_rate": float(np.mean([float(r["collision"]) for r in episode_rows])),
        "mean_cbf_interventions": float(np.mean([r["cbf_interventions"] for r in episode_rows])),
    }

# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def _get_log_alpha_parameter(agent: SafeSACAgent):
    """Return the actual learnable SAC log-alpha parameter, if enabled."""
    entropy_module = getattr(agent.policy, "entropy_temperature", None)
    if entropy_module is None:
        return None
    return getattr(entropy_module, "log_alpha", None)


def _apply_alpha_floor(
    agent: SafeSACAgent,
    alpha_min: float | None,
    *,
    reset_optimizer_if_clamped: bool = False,
) -> tuple[float | None, float | None, bool]:
    """Clamp the *real* EntropyTemperature.log_alpha parameter.

    The previous training script checked ``agent.policy.log_alpha``. That
    attribute does not exist in this policy: log_alpha lives under
    ``agent.policy.entropy_temperature.log_alpha``. Consequently --alpha-min
    was silently ineffective.
    """
    log_alpha = _get_log_alpha_parameter(agent)
    if alpha_min is None or log_alpha is None:
        return None, None, False
    if alpha_min <= 0.0:
        raise ValueError("--alpha-min must be > 0")

    import math
    before = float(log_alpha.detach().exp().cpu().item())
    floor_log = math.log(float(alpha_min))
    clamped = before < float(alpha_min)
    with torch.no_grad():
        log_alpha.clamp_(min=floor_log)
    after = float(log_alpha.detach().exp().cpu().item())

    # A resumed optimizer can contain many steps of momentum pushing alpha
    # downward. If a checkpoint starts below the new floor, clear only the
    # alpha optimizer state once so stale momentum does not fight the clamp.
    if clamped and reset_optimizer_if_clamped and agent.alpha_optimizer is not None:
        agent.alpha_optimizer.state.clear()

    return before, after, clamped


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

    if args.warm_start_checkpoint and args.resume_checkpoint:
        raise ValueError("Use either --warm-start-checkpoint or --resume-checkpoint, not both.")
    if args.resume_checkpoint and args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if args.resume_checkpoint and args.resume_step >= args.timesteps:
        raise ValueError("For resume mode, --timesteps is the final target step and must be > --resume-step.")

    start_step = int(args.resume_step) if args.resume_checkpoint else 0

    env = AMRWarehouseEnv(
        use_cbf_filter=not args.no_cbf_filter,
        cbf_lookahead_distance=args.lookahead_distance,
    )
    eval_env = AMRWarehouseEnv(
        use_cbf_filter=not args.no_cbf_filter,
        cbf_lookahead_distance=args.lookahead_distance,
    )

    # Train and evaluate under the exact controller configuration that passed
    # the final safety ablations.  Previously circulation was enabled only in
    # ad-hoc evaluation scripts, so a newly-trained actor saw a different
    # safety projection than the one used at deployment/evaluation.
    for _env in (env, eval_env):
        _env.cbf_filter.config.enable_circulation = True
        _env.cbf_filter.config.circulation_threshold = 0.5
        _env.cbf_filter.config.circulation_gain = 0.6

    agent = build_agent(env, args)

    if args.resume_checkpoint:
        agent.load_models(args.resume_checkpoint, map_location=args.device)
        print(f"[train.py] Resumed model/optimizers from {args.resume_checkpoint}")
        print(f"[train.py] Resume env step:  {start_step:,}")
        print(f"[train.py] Resume updates:   {getattr(agent, '_update_count', 0):,}")
        print(f"[train.py] Replay refill:    {args.resume_replay_warmup:,} new transitions before updates")

    # Enforce the entropy floor immediately as well as after every update.
    # This matters when resuming the 90k checkpoint whose alpha had already
    # fallen below the requested floor due to the old attribute-path bug.
    alpha_before, alpha_after, alpha_was_clamped = _apply_alpha_floor(
        agent, args.alpha_min, reset_optimizer_if_clamped=bool(args.resume_checkpoint)
    )
    if alpha_before is not None:
        print(
            f"[train.py] Alpha:           {alpha_before:.6f} -> {alpha_after:.6f}"
            + (" (floor applied; alpha optimizer momentum reset)" if alpha_was_clamped and args.resume_checkpoint else "")
        )

    writer = SummaryWriter(log_dir=str(log_dir))

    _active_L = args.lookahead_distance if args.lookahead_distance is not None else 0.25
    print(f"[train.py] Run name:        {run_name}")
    if args.resume_checkpoint:
        print(f"[train.py] Step range:      {start_step + 1:,} -> {args.timesteps:,}")
    print(f"[train.py] Device:          {agent.device}")
    if str(agent.device).startswith("cpu") and torch.cuda.is_available():
        print("[train.py] WARNING: CUDA is available but this run is using CPU. Pass --device cuda to use the GPU for networks.")
    elif str(agent.device).startswith("cpu"):
        print("[train.py] NOTE: PyTorch reports no usable CUDA device; neural-network updates will run on CPU.")
    print(f"[train.py] Log dir:         {log_dir}")
    print(f"[train.py] Checkpoint dir:  {save_dir}")
    print(f"[train.py] CBF filter:      {'enabled' if not args.no_cbf_filter else 'DISABLED'}")
    print(f"[train.py] CBF lookahead:   L={_active_L:.3f}m {'(default)' if args.lookahead_distance is None else '(overridden via --lookahead-distance)'}")
    print("[train.py] Circulation:     ON (threshold=0.5, gain=0.6)")
    print(f"[train.py] Hard escape:     {'ON' if env.cbf_filter.config.enable_hard_escape else 'OFF'}")
    print(f"[train.py] HER strategy:    {args.her_strategy}")

    her_buffer = HEREpisodeBuffer()

    # -- Curriculum ramp -------------------------------------------------- #
    # See --curriculum-steps help text for why this exists: the env supports
    # set_curriculum_level() but no prior run ever called it, so training
    # always started at full difficulty (curriculum_level=1.0) from step 0.
    curriculum_steps = args.curriculum_steps if args.curriculum_steps is not None else int(0.4 * args.timesteps)
    curriculum_start_level = args.curriculum_start_level

    def _curriculum_level_at(step: int) -> float:
        if curriculum_steps <= 0:
            return 1.0
        frac = min(1.0, step / curriculum_steps)
        return curriculum_start_level + frac * (1.0 - curriculum_start_level)

    if curriculum_steps > 0:
        print(
            f"[train.py] Curriculum:      level {curriculum_start_level:.2f} -> 1.00 "
            f"over first {curriculum_steps:,} steps"
        )
    else:
        print("[train.py] Curriculum:      DISABLED (curriculum_level=1.0 throughout)")

    # Curriculum level is fixed for the whole episode.  Set it BEFORE reset
    # so obstacle sampling, sensing noise, and active-obstacle count agree from
    # the very first transition.
    env.set_curriculum_level(_curriculum_level_at(start_step + 1))
    obs, _info = env.reset(seed=args.seed + start_step)

    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    recent_returns: deque = deque(maxlen=100)
    recent_successes: deque = deque(maxlen=100)
    recent_collisions: deque = deque(maxlen=100)

    best_success_rate = -float("inf")
    best_eval_return = -float("inf")

    start_time = time.time()
    last_print_time = start_time
    last_print_step = start_step
    session_update_count = 0

    step_iterator = (
        trange(start_step + 1, args.timesteps + 1, desc="train", unit="step", initial=start_step, total=args.timesteps)
        if _TQDM_AVAILABLE
        else range(start_step + 1, args.timesteps + 1)
    )

    for global_step in step_iterator:
        # -- 0. Curriculum -------------------------------------------------- #
        # The difficulty is intentionally frozen within each episode.  It is
        # updated immediately before the next reset in the episode-bookkeeping
        # block below; changing obstacle count mid-episode created discontinuous
        # dynamics and stale/frozen obstacle slots.

        # -- 1. Act -------------------------------------------------------- #
        if (not args.resume_checkpoint) and global_step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        cost = 1.0 if info.get("collision", False) else 0.0
        barrier_value = float(info.get("barrier_value", 0.0))

        # The MDP action is the command selected by SAC and passed into
        # env.step().  The CBF, actuator lag, slip, and noise are part of the
        # environment transition.  Storing the post-filter/post-lag action here
        # trains Q(s,a_executed) while the actor and Bellman target query
        # Q(s,a_policy), which is an action-semantics mismatch.  Keep
        # info["safe_action"] only for diagnostics and store the policy action.
        policy_action = np.asarray(action, dtype=np.float32).copy()

        her_buffer.add(
            obs=obs,
            action=policy_action,
            reward=float(reward),
            next_obs=next_obs,
            done=float(terminated),
            cost=cost,
            barrier_value=barrier_value,
            is_goal=bool(info.get("goal_reached", False)),
            is_collision=bool(info.get("collision", False)),
            max_diag=env.max_diag,
            info=info,
            # Pre-action physical state the CBF filter was conditioned on
            # this step (see AMRWarehouseEnv.step()'s capture point) --
            # goal-independent, so her_buffer.py's relabel_and_push copies
            # it through unchanged onto every relabeled variant of this
            # transition. Only needed when the CBF actor-consistency
            # regularizer is enabled; harmless (stored as None, ignored by
            # the replay buffer) otherwise.
            robot_state_raw=info.get("cbf_robot_state"),
            dynamic_obstacles_raw=info.get("cbf_dynamic_obstacles"),
        )

        obs = next_obs
        episode_return += float(reward)
        episode_length += 1

        # -- 2. Episode bookkeeping ------------------------------------------ #
        if done:
            ensure_hs = (
                args.her_ensure_hindsight_success
                if args.her_ensure_hindsight_success is not None
                else HER_ENSURE_HINDSIGHT_SUCCESS
            )
            her_stats = her_buffer.relabel_and_push(
                agent.replay_buffer,
                k=4,
                strategy=args.her_strategy,
                min_relabel_distance=2.0 * GOAL_TOLERANCE,
                ensure_hindsight_success=ensure_hs,
            )
            her_buffer.clear()

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
            writer.add_scalar("her/fraction_reached", her_stats.fraction_reached, global_step)
            writer.add_scalar("her/skip_ratio", her_stats.skip_ratio, global_step)
            writer.add_scalar("her/total_relabeled", her_stats.total_relabeled, global_step)
            writer.add_scalar("her/net_episode_displacement", her_stats.net_episode_displacement, global_step)
            writer.add_scalar("her/hindsight_success_injected", her_stats.hindsight_success_injected, global_step)

            # Advance curriculum only at an episode boundary, then sample a
            # fresh obstacle set at that fixed difficulty.
            env.set_curriculum_level(_curriculum_level_at(global_step + 1))
            obs, _info = env.reset()
            episode_return = 0.0
            episode_length = 0

        # -- 3. SAC update -------------------------------------------------- #
        if args.resume_checkpoint:
            update_time_ready = (global_step - start_step) > args.resume_replay_warmup
        else:
            update_time_ready = global_step > args.learning_starts

        if (
            len(agent.replay_buffer) > args.batch_size
            and update_time_ready
            and global_step % args.train_freq == 0
        ):
            # critic_warmup_steps counts actual UPDATE CALLS, not environment
            # steps. In the old script, learning-starts=5000 together with
            # critic-warmup-steps=5000 resulted in zero frozen-actor updates.
            freeze_actor = session_update_count < args.critic_warmup_steps
            stats = agent.update(args.batch_size, freeze_actor=freeze_actor)
            session_update_count += 1
            for key, value in stats.items():
                writer.add_scalar(f"train/{key}", value, global_step)

            alpha_pre, alpha_post, _ = _apply_alpha_floor(agent, args.alpha_min)
            if alpha_pre is not None:
                writer.add_scalar("train/alpha_pre_clamp", alpha_pre, global_step)
                writer.add_scalar("train/alpha", alpha_post, global_step)

        # -- 4. Evaluation ---------------------------------------------------- #
        if global_step % args.eval_freq == 0:
            eval_wall_start = time.time()
            trajectory_dir = save_dir / f"trajectory_eval_{global_step}"

            # ONE deterministic pass now produces both aggregate metrics and
            # trajectory diagnostics. The old code ran evaluate_agent(...) and
            # then evaluate_trajectories(...) on the same seeds, doubling the
            # expensive CBF/SLSQP evaluation workload (30 + 30 episodes).
            eval_stats = evaluate_trajectories(
                agent=agent,
                eval_env=eval_env,
                num_episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                output_dir=trajectory_dir,
            )
            eval_wall_s = time.time() - eval_wall_start

            for key, value in eval_stats.items():
                writer.add_scalar(f"eval/{key}", value, global_step)
            writer.add_scalar("eval/wall_seconds", eval_wall_s, global_step)

            print(
                f"\n[eval @ step {global_step:>9,}] "
                f"return={eval_stats['mean_return']:.2f}  "
                f"success_rate={eval_stats['success_rate']:.2%}  "
                f"collision_rate={eval_stats['collision_rate']:.2%}  "
                f"avg_cbf_interventions={eval_stats['mean_cbf_interventions']:.2f}  "
                f"length={eval_stats['mean_length']:.1f}  "
                f"eval_wall={eval_wall_s/60:.1f}m"
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

            # Always keep an evaluation-boundary checkpoint for interruption
            # recovery, even when it is not the best model.
            latest_path = save_dir / "latest_model.pt"
            agent.save_models(latest_path)
            state_path = save_dir / "training_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "global_step": global_step,
                        "update_count": int(getattr(agent, "_update_count", 0)),
                        "best_success_rate": best_success_rate,
                        "best_eval_return": best_eval_return,
                        "latest_checkpoint": str(latest_path),
                    },
                    indent=2,
                )
            )

            # tqdm counts evaluation wall time as if it were training time.
            # unpause() removes that pause from its rate/ETA estimate.
            if _TQDM_AVAILABLE and hasattr(step_iterator, "unpause"):
                step_iterator.unpause()

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
    (save_dir / "training_state.json").write_text(
        json.dumps(
            {
                "global_step": args.timesteps,
                "update_count": int(getattr(agent, "_update_count", 0)),
                "best_success_rate": best_success_rate,
                "best_eval_return": best_eval_return,
                "latest_checkpoint": str(final_path),
            },
            indent=2,
        )
    )
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