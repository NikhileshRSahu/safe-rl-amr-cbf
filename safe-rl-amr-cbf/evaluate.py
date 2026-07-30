import argparse
import cvxpy as cp
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(
        description="Evaluate Safe RL + CBF on AMR"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--use-cbf",
        action="store_true",
        help="Enable Control Barrier Function safety filter",
    )
    group.add_argument(
        "--no-cbf",
        action="store_false",
        dest="use_cbf",
        help="Disable safety filter (raw RL policy)",
    )

    args = parser.parse_args()

    # 2. Status Output
    if args.use_cbf:
        print("[INFO] Running evaluation WITH Control Barrier Function (CBF)...")
    else:
        print("[INFO] Running evaluation WITHOUT CBF (Raw RL Policy)...")

    # 3. Add your environment loading, policy loading, and loop here
    # Example:
    # env = gym.make("YourWarehouseEnv-v0")
    # model = PPO.load("trained_model.zip")
    # ...


if __name__ == "__main__":
    main()
