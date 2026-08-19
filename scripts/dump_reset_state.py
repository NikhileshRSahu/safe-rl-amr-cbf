"""Dump environment reset state for a given seed to an NPZ file.

Usage:
    python scripts/dump_reset_state.py --seed 1018 --out /tmp/state1.npz
"""
import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
from environment import AMRWarehouseEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    env = AMRWarehouseEnv(use_cbf_filter=True)
    obs, info = env.reset(seed=args.seed)
    state = env.get_state()
    outp = Path(args.out)
    np.savez_compressed(outp, robot_state=state["robot_state"], dynamic_obstacles=state["dynamic_obstacles"], goal_pos=state.get("goal_pos", env.goal_pos))
    print(f"Wrote {outp}")
    env.close()


if __name__ == '__main__':
    main()
