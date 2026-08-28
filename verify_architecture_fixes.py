"""Fast verification of the architecture fixes used by the next training run."""
from pathlib import Path
import math
import numpy as np
from environment import AMRWarehouseEnv
from config import ROBOT_RADIUS, DYNAMIC_OBS_RADIUS, CBF_SAFETY_MARGIN


def main():
    env = AMRWarehouseEnv(max_episode_steps=20)
    env.cbf_filter.config.enable_circulation = True
    env.cbf_filter.config.circulation_threshold = 0.5
    env.cbf_filter.config.circulation_gain = 0.6

    # Curriculum masking / active count.
    env.set_curriculum_level(0.15)
    obs0, _ = env.reset(seed=123)
    assert env._num_active_dynamic_obstacles() == 1
    assert np.all(np.linalg.norm(env.dynamic_obstacles[1:, :2], axis=1) > 1000.0)

    # Dynamic obstacle starts inside the live CBF's initial safe set.
    L = env.cbf_filter.config.lookahead_distance
    pL = env.robot_state[:2] + L * np.array([
        math.cos(env.robot_state[2]), math.sin(env.robot_state[2])
    ])
    R = (ROBOT_RADIUS + DYNAMIC_OBS_RADIUS + CBF_SAFETY_MARGIN
         + env.cbf_filter.config.tracking_uncertainty + L)
    assert np.linalg.norm(env.dynamic_obstacles[0, :2] - pL) >= R - 1e-9

    # Exactly one observation-step of latency.
    reset_robot_obs = obs0["robot_state"].copy()
    a = np.array([1.0, 0.0], dtype=np.float32)
    obs1, *_ = env.step(a)
    true_after1 = env._get_observation()["robot_state"].copy()
    obs2, *_ = env.step(a)
    assert np.allclose(obs1["robot_state"], reset_robot_obs)
    assert np.allclose(obs2["robot_state"], true_after1)

    # Static source checks for the two training-loop semantics fixes.
    train_src = (Path(__file__).parent / "train.py").read_text(encoding="utf-8")
    assert "action=policy_action" in train_src
    assert "action=safe_action" not in train_src
    assert "env.set_curriculum_level(_curriculum_level_at(global_step + 1))" in train_src
    assert "env.set_curriculum_level(_curriculum_level_at(global_step))" not in train_src
    assert "enable_circulation = True" in train_src

    print("PASS: one-step latency")
    print("PASS: curriculum masks inactive obstacles")
    print("PASS: curriculum has at least one active obstacle at level 0.15")
    print("PASS: active dynamic obstacle starts inside CBF-safe initial set")
    print("PASS: replay stores policy action, not post-filter actuator action")
    print("PASS: curriculum updates only at episode boundaries")
    print("PASS: training/evaluation controller enables circulation")
    print("ALL ARCHITECTURE CHECKS PASSED")

if __name__ == "__main__":
    main()
