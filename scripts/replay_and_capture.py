"""Replay episodes using the PureCBF baseline to find collision steps and
optionally dump the final N states as NPZ snapshots for analysis.

Usage:
    python scripts/replay_and_capture.py --seed 1002 --dump-last 8
    python scripts/replay_and_capture.py --seeds 1002 1003 1005 1018 --dump-last 8
"""

from pathlib import Path
import argparse
import numpy as np
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from environment import AMRWarehouseEnv


def replay_and_dump(seed: int, dump_last: int = 8, out_dir: Path = None):
    if out_dir is None:
        out_dir = repo_root / "diagnostics" / "cbf_failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = AMRWarehouseEnv(use_cbf_filter=True)
    obs, info = env.reset(seed=seed)
    from baselines import PureCBF
    controller = PureCBF()

    states = []
    step = 0
    collision_step = None
    while True:
        action = controller.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        states.append(env.get_state())

        if terminated and info.get("collision", False):
            collision_step = step
            break
        if truncated:
            break

    # dump last N states
    last_states = states[-dump_last:]
    for s in last_states:
        stepnum = s["step_count"]
        outp = out_dir / f"seed_{seed}_step_{stepnum}_replay.npz"
        np.savez_compressed(
            outp,
            robot_state=s["robot_state"],
            dynamic_obstacles=s["dynamic_obstacles"],
            last_cbf_diag=s.get("last_cbf_diagnostics", {}),
            step=np.array([stepnum]),
        )

    env.close()
    return collision_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--dump-last", type=int, default=8)
    args = p.parse_args()

    for seed in args.seeds:
        print(f"Replaying seed={seed} ...")
        coll = replay_and_dump(seed, dump_last=args.dump_last)
        print(f"  collision_step: {coll}")


if __name__ == "__main__":
    main()
