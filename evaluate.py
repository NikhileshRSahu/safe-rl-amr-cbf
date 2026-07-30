"""evaluate.py

Loads a trained ``SafeSACAgent`` checkpoint and runs it in
``AMRWarehouseEnv`` with live (or off-screen) rendering, printing a
per-episode summary of return, length, success, collisions, and CBF
safety-filter interventions.

Usage:
    python evaluate.py --checkpoint checkpoints/safe_sac_.../best_model.pt --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from environment import AMRWarehouseEnv
from safe_sac import SafeSACAgent, SafeSACConfig


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the evaluation run.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Render and evaluate a trained SafeSACAgent checkpoint."
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
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Agent loading
# --------------------------------------------------------------------------- #

def load_agent(
    checkpoint_path: str, env: AMRWarehouseEnv, device: str
) -> SafeSACAgent:
    """Reconstructs a ``SafeSACAgent`` matching a checkpoint's saved architecture.

    The checkpoint (written by ``SafeSACAgent.save_models``) stores the
    exact ``PolicyConfig`` and ``SafeSACConfig`` used at training time, so
    the agent is first built with that identical configuration and *then*
    its weights are loaded -- this guarantees the actor/critic/target
    network shapes line up with the saved ``state_dict``s.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        env: The environment instance, used to size the replay buffer and
            confirm the observation/action spaces the policy expects.
        device: Torch device to load the agent onto.

    Returns:
        A ``SafeSACAgent`` with weights restored from the checkpoint, set
        to evaluation mode.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        KeyError: If the checkpoint is missing the expected
            ``policy_config`` entry (i.e. was not produced by
            ``SafeSACAgent.save_models``).
    """
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

    raw_checkpoint: Dict[str, Any] = torch.load(
        checkpoint_path_obj, map_location=device, weights_only=False
    )
    if "policy_config" not in raw_checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path_obj} has no 'policy_config' entry; "
            "it was not produced by SafeSACAgent.save_models()."
        )

    policy_config = raw_checkpoint["policy_config"]
    sac_config: SafeSACConfig = raw_checkpoint.get("sac_config", SafeSACConfig())
    sac_config.device = device

    agent = SafeSACAgent(
        policy_config=policy_config,
        observation_space=env.observation_space,
        action_dim=int(env.action_space.shape[0]),
        sac_config=sac_config,
        replay_buffer_size=1,  # Evaluation only; no training data is collected.
    )
    agent.load_models(checkpoint_path_obj, map_location=device)
    agent.eval()
    return agent


# --------------------------------------------------------------------------- #
# Optional video saving
# --------------------------------------------------------------------------- #

def _save_episode_gif(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    """Saves a list of RGB frames as an animated GIF, if ``imageio`` is available.

    Args:
        frames: List of ``(H, W, 3)`` uint8 RGB frames.
        out_path: Destination ``.gif`` path.
        fps: Playback frame rate.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(
            f"[evaluate.py] 'imageio' is not installed; skipping GIF save for {out_path}. "
            "Install with `pip install imageio` to enable this."
        )
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[evaluate.py] Saved episode video to {out_path}")


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #

def run_episode(
    agent: SafeSACAgent,
    env: AMRWarehouseEnv,
    episode_seed: int,
    collect_frames: bool,
) -> Dict[str, Any]:
    """Runs a single deterministic evaluation episode.

    Args:
        agent: The loaded, evaluation-mode agent.
        env: The rendering environment.
        episode_seed: Seed passed to ``env.reset``.
        collect_frames: If ``True``, calls ``env.render()`` every step and
            collects the returned RGB frames (only meaningful when
            ``env.render_mode == "rgb_array"``).

    Returns:
        Dict with keys ``return``, ``length``, ``success``, ``collision``,
        ``collision_type``, ``cbf_interventions``, and (if
        ``collect_frames``) ``frames``.
    """
    obs, _info = env.reset(seed=episode_seed)
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


def evaluate(args: argparse.Namespace) -> None:
    """Loads the checkpoint and runs the full evaluation loop with rendering.

    Args:
        args: Parsed command-line arguments.
    """
    env = AMRWarehouseEnv(
        render_mode=args.render_mode, use_cbf_filter=not args.no_cbf_filter
    )
    agent = load_agent(args.checkpoint, env, device=args.device)

    print(f"[evaluate.py] Checkpoint:   {args.checkpoint}")
    print(f"[evaluate.py] Device:       {args.device}")
    print(f"[evaluate.py] Render mode:  {args.render_mode}")
    print(f"[evaluate.py] CBF filter:   {'enabled' if not args.no_cbf_filter else 'DISABLED'}")
    print(f"[evaluate.py] Episodes:     {args.episodes}\n")

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

    print("\n[evaluate.py] ==== Summary over {} episodes ====".format(args.episodes))
    print(f"  Mean return:            {np.mean(returns):.2f} (+/- {np.std(returns):.2f})")
    print(f"  Success rate:           {np.mean(successes):.2%}")
    print(f"  Collision rate:         {np.mean(collisions):.2%}")
    print(f"  Mean CBF interventions: {np.mean(interventions):.2f} per episode")

    env.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cli_args = parse_args()
    evaluate(cli_args)