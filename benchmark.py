"""benchmark.py

Unified benchmarking framework that evaluates ALL algorithms on the SAME
set of seeded test scenarios and produces comparison metrics + plots.

Algorithms compared:
    - PID
    - Potential Field
    - A*
    - DWA
    - RRT
    - MPC
    - Pure CBF (CBF-only reactive)
    - Pure RL (RL without CBF filter)
    - RL + CBF (our approach)

Metrics collected per algorithm:
    - Success rate (% episodes reaching goal)
    - Collision rate (% episodes with collision)
    - Mean return
    - Mean episode length
    - Mean path efficiency (goal distance / path length)
    - Mean CBF interventions per episode (RL+CBF only)

Usage:
    # 1. Train the improved RL+CBF agent:
    python train_improved.py --timesteps 1000000 --save-dir checkpoints_improved

    # 2. Run benchmark:
    python benchmark.py \
        --rl-checkpoint checkpoints_improved/.../best_model.pt \
        --episodes 50 \
        --output-dir benchmark_results
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from environment import AMRWarehouseEnv
from safe_sac import SafeSACAgent, SafeSACConfig
from evaluate import load_agent
from baselines import (
    BaseBaseline,
    make_baseline,
    PIDController,
    PotentialField,
    AStarPlanner,
    DWA,
    RRTPlanner,
    MPCBaseline,
    PureCBF,
)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all AMR navigation algorithms.")

    # RL checkpoints
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to trained RL+CBF checkpoint.")
    parser.add_argument("--pure-rl-checkpoint", type=str, default=None,
                        help="Path to trained Pure RL checkpoint (no CBF).")

    # Benchmark control
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of test episodes per algorithm.")
    parser.add_argument("--seed", type=int, default=999,
                        help="Base seed for test scenarios.")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode.")
    parser.add_argument("--curriculum-level", type=float, default=1.0,
                        help="Difficulty level (1.0 = full obstacles).")

    # Output
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results and plots.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for RL agents.")

    # Algorithm selection (default: run all)
    parser.add_argument("--algorithms", type=str, nargs="+", default=None,
                        help="List of algorithms to benchmark. Default: all.")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Test scenario generation
# --------------------------------------------------------------------------- #

def generate_test_scenarios(
    num_episodes: int, base_seed: int
) -> List[Dict[str, Any]]:
    """Generate a fixed set of test scenarios for reproducible evaluation.

    Each scenario is a seed value. All algorithms run on the same seeds
    so they face identical obstacle configurations, start poses, and goals.
    """
    return [{"seed": base_seed + i} for i in range(num_episodes)]


# --------------------------------------------------------------------------- #
# RL agent wrapper with observation augmentation
# --------------------------------------------------------------------------- #

def augment_observation(
    obs: Dict[str, np.ndarray],
    env: AMRWarehouseEnv,
    max_obstacles: int,
) -> Dict[str, np.ndarray]:
    """Add obstacle_set to observation for attention encoder."""
    import math
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


class RLAgentWrapper:
    """Wraps SafeSACAgent to match baseline interface."""

    def __init__(self, agent: SafeSACAgent, env: AMRWarehouseEnv, use_augment: bool = True):
        self.agent = agent
        self.env = env
        self.use_augment = use_augment
        self.max_obstacles = getattr(agent.policy_config, "max_obstacles", 0)

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        if self.use_augment and self.max_obstacles > 0:
            obs = augment_observation(obs, self.env, self.max_obstacles)
        return self.agent.select_action(obs, deterministic=True)

    def reset(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Single episode evaluation
# --------------------------------------------------------------------------- #

def run_episode(
    controller: Any,
    env: AMRWarehouseEnv,
    scenario: Dict[str, Any],
    max_steps: int,
) -> Dict[str, Any]:
    """Run one episode and return metrics."""
    seed = scenario["seed"]
    obs, info = env.reset(seed=seed)

    # Augment obs for RL agents that need it
    if hasattr(controller, "env") and hasattr(controller, "use_augment"):
        if controller.use_augment:
            obs = augment_observation(obs, env, controller.max_obstacles)

    if hasattr(controller, "reset"):
        controller.reset()

    done = False
    ep_return = 0.0
    ep_length = 0
    success = False
    collision = False
    collision_type = "none"
    cbf_interventions = 0
    path_coords: List[Tuple[float, float]] = []

    while not done and ep_length < max_steps:
        if hasattr(controller, "select_action"):
            action = controller.select_action(obs, info)
        else:
            action = controller.agent.select_action(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # Re-augment for RL on next step
        if hasattr(controller, "env") and hasattr(controller, "use_augment"):
            if controller.use_augment:
                obs = augment_observation(obs, env, controller.max_obstacles)

        done = terminated or truncated
        ep_return += float(reward)
        ep_length += 1
        success = success or bool(info.get("goal_reached", False))
        if info.get("collision", False):
            collision = True
            collision_type = str(info.get("collision_type", "unknown"))
        cbf_interventions += int(info.get("cbf_intervened", False))
        path_coords.append((float(env.robot_state[0]), float(env.robot_state[1])))

    # Path efficiency: straight-line distance / actual path length
    start_pos = env.trajectory_history[0] if env.trajectory_history else (0.0, 0.0)
    goal_pos = env.goal_pos
    straight_dist = distance(start_pos, goal_pos)
    path_len = 0.0
    for i in range(1, len(path_coords)):
        path_len += math.hypot(path_coords[i][0] - path_coords[i-1][0],
                               path_coords[i][1] - path_coords[i-1][1])
    efficiency = straight_dist / max(path_len, 1e-6) if path_len > 0 else 0.0

    return {
        "return": ep_return,
        "length": ep_length,
        "success": success,
        "collision": collision,
        "collision_type": collision_type,
        "cbf_interventions": cbf_interventions,
        "efficiency": efficiency,
        "path_length": path_len,
        "straight_dist": straight_dist,
    }


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --------------------------------------------------------------------------- #
# Algorithm benchmark runner
# --------------------------------------------------------------------------- #

def benchmark_algorithm(
    name: str,
    controller: Any,
    env: AMRWarehouseEnv,
    scenarios: List[Dict[str, Any]],
    max_steps: int,
) -> Dict[str, Any]:
    """Run all scenarios for one algorithm and aggregate metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    results: List[Dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        result = run_episode(controller, env, scenario, max_steps)
        results.append(result)
        status = "SUCCESS" if result["success"] else ("COLLISION" if result["collision"] else "TIMEOUT")
        print(f"  Episode {i+1:>3}/{len(scenarios)}: {status:<10} "
              f"return={result['return']:8.2f}  length={result['length']:4d} "
              f"efficiency={result['efficiency']:.3f}")

    returns = [r["return"] for r in results]
    lengths = [r["length"] for r in results]
    successes = [r["success"] for r in results]
    collisions = [r["collision"] for r in results]
    efficiencies = [r["efficiency"] for r in results]
    cbf_counts = [r["cbf_interventions"] for r in results]

    summary = {
        "algorithm": name,
        "episodes": len(scenarios),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "mean_cbf_interventions": float(np.mean(cbf_counts)),
        "results": results,
    }

    print(f"\n  Summary: success={summary['success_rate']:.1%}  "
          f"collision={summary['collision_rate']:.1%}  "
          f"return={summary['mean_return']:.2f} (+/- {summary['std_return']:.2f})  "
          f"efficiency={summary['mean_efficiency']:.3f}")

    return summary


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_comparison(all_summaries: List[Dict[str, Any]], output_dir: Path) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        print("[benchmark.py] matplotlib not available, skipping plots.")
        return

    algorithms = [s["algorithm"] for s in all_summaries]
    success_rates = [s["success_rate"] * 100 for s in all_summaries]
    collision_rates = [s["collision_rate"] * 100 for s in all_summaries]
    mean_returns = [s["mean_return"] for s in all_summaries]
    mean_efficiencies = [s["mean_efficiency"] for s in all_summaries]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AMR Navigation Algorithm Comparison", fontsize=16, fontweight="bold")

    # Success rate
    ax = axes[0, 0]
    bars = ax.bar(algorithms, success_rates, color="seagreen", edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Success Rate (higher is better)")
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Collision rate
    ax = axes[0, 1]
    bars = ax.bar(algorithms, collision_rates, color="firebrick", edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Collision Rate (lower is better)")
    for bar, val in zip(bars, collision_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Mean return
    ax = axes[1, 0]
    bars = ax.bar(algorithms, mean_returns, color="steelblue", edgecolor="black")
    ax.set_ylabel("Mean Return")
    ax.set_title("Mean Episode Return (higher is better)")
    for bar, val in zip(bars, mean_returns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(0.01, 0.02 * max(mean_returns)),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Efficiency
    ax = axes[1, 1]
    bars = ax.bar(algorithms, mean_efficiencies, color="goldenrod", edgecolor="black")
    ax.set_ylabel("Path Efficiency")
    ax.set_ylim(0, 1.1)
    ax.set_title("Path Efficiency (higher is better)")
    for bar, val in zip(bars, mean_efficiencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "benchmark_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[benchmark.py] Saved comparison plot to {plot_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    import math
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = generate_test_scenarios(args.episodes, args.seed)

    # Build algorithms to benchmark
    algorithms_to_run: List[Tuple[str, Any]] = []

    if args.algorithms is None:
        algo_names = ["pid", "potential_field", "astar", "dwa", "rrt", "mpc", "cbf_only"]
        if args.rl_checkpoint:
            algo_names.append("rl_cbf")
        if args.pure_rl_checkpoint:
            algo_names.append("pure_rl")
    else:
        algo_names = args.algorithms

    for name in algo_names:
        if name in ["rl_cbf", "rl+cbf"] and args.rl_checkpoint:
            env = AMRWarehouseEnv(use_cbf_filter=True, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            agent = load_agent(args.rl_checkpoint, env, device=args.device)
            wrapper = RLAgentWrapper(agent, env, use_augment=True)
            algorithms_to_run.append(("RL + CBF", wrapper))
        elif name in ["pure_rl"] and args.pure_rl_checkpoint:
            env = AMRWarehouseEnv(use_cbf_filter=False, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            agent = load_agent(args.pure_rl_checkpoint, env, device=args.device)
            wrapper = RLAgentWrapper(agent, env, use_augment=True)
            algorithms_to_run.append(("Pure RL", wrapper))
        elif name == "pid":
            env = AMRWarehouseEnv(use_cbf_filter=False, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            algorithms_to_run.append(("PID", PIDController()))
        elif name == "potential_field":
            env = AMRWarehouseEnv(use_cbf_filter=False, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            algorithms_to_run.append(("Potential Field", PotentialField()))
        elif name == "astar":
            env = AMRWarehouseEnv(use_cbf_filter=False, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            algorithms_to_run.append(("A*", AStarPlanner()))
        elif name == "dwa":
            env = AMRWarehouseEnv(use_cbf_filter=False, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            algorithms_to_run.append(("DWA", DWA()))
        elif name == "rrt":
            env = AMRWarehouseEnv(use_cbf_filter=False, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            algorithms_to_run.append(("RRT", RRTPlanner()))
        elif name == "mpc":
            env = AMRWarehouseEnv(use_cbf_filter=False, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            algorithms_to_run.append(("MPC", MPCBaseline()))
        elif name == "cbf_only":
            env = AMRWarehouseEnv(use_cbf_filter=True, max_episode_steps=args.max_steps)
            env.set_curriculum_level(args.curriculum_level)
            algorithms_to_run.append(("CBF Only", PureCBF()))
        else:
            print(f"[benchmark.py] Warning: unknown algorithm '{name}', skipping.")

    print(f"[benchmark.py] Benchmarking {len(algorithms_to_run)} algorithms")
    print(f"[benchmark.py] Episodes per algorithm: {args.episodes}")
    print(f"[benchmark.py] Curriculum level: {args.curriculum_level}")
    print(f"[benchmark.py] Max steps per episode: {args.max_steps}")

    all_summaries: List[Dict[str, Any]] = []

    for name, controller in algorithms_to_run:
        # Create a fresh env for each algorithm
        env = AMRWarehouseEnv(
            use_cbf_filter=(name == "RL + CBF" or name == "CBF Only"),
            max_episode_steps=args.max_steps,
        )
        env.set_curriculum_level(args.curriculum_level)

        # Update wrapper's env reference if needed
        if hasattr(controller, "env"):
            controller.env = env

        summary = benchmark_algorithm(name, controller, env, scenarios, args.max_steps)
        all_summaries.append(summary)
        env.close()

    # Save JSON results
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        # Don't save per-episode results in JSON to keep it compact
        json.dump([
            {k: v for k, v in s.items() if k != "results"}
            for s in all_summaries
        ], f, indent=2)
    print(f"\n[benchmark.py] Saved results to {json_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Algorithm':<18} {'Success':>10} {'Collision':>10} {'Return':>12} {'Efficiency':>12}")
    print(f"{'-'*80}")
    for s in all_summaries:
        print(f"{s['algorithm']:<18} "
              f"{s['success_rate']:>9.1%} "
              f"{s['collision_rate']:>9.1%} "
              f"{s['mean_return']:>11.2f} "
              f"{s['mean_efficiency']:>11.3f}")
    print(f"{'='*80}")

    # Generate plots
    plot_comparison(all_summaries, output_dir)

    print("\n[benchmark.py] Benchmark complete!")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
