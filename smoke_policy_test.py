"""
Policy <-> Environment integration smoke test.

Verifies that SafeRLPolicy's dict-based observation contract actually
lines up with what AMRWarehouseEnv produces: correct keys, correct shapes,
and a working forward() / evaluate_actions() / predict() pass using REAL
observations pulled straight out of env.reset() / env.step() (not
hand-built dummy tensors).

Run this after the env-level smoke_test.py has already passed.
"""

import numpy as np
import torch

from environment import AMRWarehouseEnv
from policy import SafeRLPolicy, PolicyConfig, ActionBounds
from config import V_MIN, V_MAX, OMEGA_MIN, OMEGA_MAX


def obs_dict_to_tensor_dict(obs: dict) -> dict:
    """Convert one env observation (dict of np arrays) into a batch-of-1
    dict of torch tensors, matching what SafeRLPolicy expects."""
    return {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}


def run_policy_smoke_test() -> None:
    print("Initializing environment...")
    env = AMRWarehouseEnv(render_mode=None)
    obs, info = env.reset()

    print("Building PolicyConfig from config.py (not defaults)...")
    # Option (a): the policy's action_bounds are set to [-1, 1] on both dims,
    # matching env.action_space exactly. The policy never sees physical
    # units. env.step() is solely responsible for converting the [-1,1]
    # action into physical [v, omega] before handing it to the CBF-QP
    # filter -- see env.step()'s v_nom/omega_nom lines. This keeps the
    # policy's native output space identical to the Gym action_space,
    # which is what SB3's own ActorCriticPolicy assumes, and removes the
    # manual rescaling that used to live in this test script.
    policy_config = PolicyConfig(
        robot_state_dim=obs["robot_state"].shape[0],   # 5
        goal_dim=obs["goal"].shape[0],                 # 4
        lidar_dim=obs["lidar"].shape[0],                # 192
        obstacle_dim=0,
        action_dim=2,
        action_bounds=ActionBounds(
            v_min=-1.0, v_max=1.0,
            omega_min=-1.0, omega_max=1.0,
        ),
    )
    print(f"  robot_state_dim={policy_config.robot_state_dim}, "
          f"goal_dim={policy_config.goal_dim}, lidar_dim={policy_config.lidar_dim}")
    print(f"  action_bounds v=({policy_config.action_bounds.v_min},{policy_config.action_bounds.v_max}) "
          f"omega=({policy_config.action_bounds.omega_min},{policy_config.action_bounds.omega_max})  "
          f"[normalized, matches env.action_space]")

    policy = SafeRLPolicy(policy_config)
    policy.eval()

    print("\nTesting forward() with a real observation from env.reset()...")
    obs_tensors = obs_dict_to_tensor_dict(obs)
    action, value, log_prob = policy(obs_tensors, deterministic=False)
    assert action.shape == (1, 2), f"Unexpected action shape: {action.shape}"
    assert value.shape == (1,), f"Unexpected value shape: {value.shape}"
    assert log_prob.shape == (1,), f"Unexpected log_prob shape: {log_prob.shape}"
    print(f"  action={action.detach().numpy().ravel()}, value={value.item():.4f}, log_prob={log_prob.item():.4f}")

    # Check action respects the NORMALIZED [-1, 1] bounds -- this is u_rl,
    # which now lives in the exact same space as env.action_space.
    v_out, omega_out = action[0, 0].item(), action[0, 1].item()
    assert -1.0 - 1e-4 <= v_out <= 1.0 + 1e-4, f"v={v_out} outside [-1,1]"
    assert -1.0 - 1e-4 <= omega_out <= 1.0 + 1e-4, f"omega={omega_out} outside [-1,1]"
    print("  action respects normalized [-1,1] bounds (matches env.action_space).")

    print("\nCross-checking env.action_space.contains(action)...")
    action_np = action.detach().numpy().ravel().astype(np.float32)
    assert env.action_space.contains(action_np), f"Policy action {action_np} not in env.action_space!"
    print("  env.action_space.contains(action) == True")

    print("\nSanity-checking the CBF filter directly (converting to physical units the "
          "same way env.step() does internally, purely for this diagnostic call)...")
    v_nom = V_MIN + 0.5 * (v_out + 1.0) * (V_MAX - V_MIN)
    omega_nom = omega_out * OMEGA_MAX
    from config import SHELVES
    v_safe, omega_safe, cbf_diag = env.cbf_filter.solve(
        v_nom, omega_nom, env.robot_state, env.dynamic_obstacles, SHELVES
    )
    print(f"  physical u_rl=({v_nom:.3f}, {omega_nom:.3f}) -> u_safe=({v_safe:.3f}, {omega_safe:.3f})")
    print(f"  diagnostics: {cbf_diag}")

    print("\nTesting step() -- policy action is fed to env.step() DIRECTLY, no manual rescaling...")
    next_obs, reward, terminated, truncated, info = env.step(action_np)
    assert isinstance(next_obs, dict) and set(next_obs.keys()) == {"robot_state", "goal", "lidar"}
    print(f"  step successful, reward={reward:.3f}")

    print("\nTesting evaluate_actions() (used by PPO update)...")
    taken_action = action.detach()
    value2, log_prob2, entropy2 = policy.evaluate_actions(obs_tensors, taken_action)
    assert torch.allclose(log_prob2, log_prob, atol=1e-4), "log_prob mismatch between forward() and evaluate_actions()"
    print(f"  value={value2.item():.4f}, log_prob={log_prob2.item():.4f}, entropy={entropy2.item():.4f}")
    print("  log_prob consistent between forward() and evaluate_actions().")

    print("\nTesting predict() (inference-time API)...")
    pred_action, hidden_state = policy.predict(obs_tensors, deterministic=True)
    assert pred_action.shape == (1, 2)
    assert hidden_state is None, "hidden_state should be None for non-recurrent policy"
    print(f"  predict() action={pred_action.detach().numpy().ravel()}")

    print("\nTesting a short rollout (5 steps) with real policy actions...")
    obs, _ = env.reset()
    for t in range(5):
        obs_tensors = obs_dict_to_tensor_dict(obs)
        with torch.no_grad():
            action, value, log_prob = policy(obs_tensors, deterministic=False)
        action_np = action.numpy().ravel().astype(np.float32)
        assert env.action_space.contains(action_np), f"t={t}: action {action_np} outside env.action_space"
        obs, reward, terminated, truncated, info = env.step(action_np)
        print(f"  t={t} reward={reward:.3f} barrier={info['barrier_value']:.3f} "
              f"safe={info['is_safe']} cbf_intervened={info['cbf_intervened']}")
        if terminated or truncated:
            print("  episode ended early, resetting.")
            obs, _ = env.reset()

    print("\n✅ All policy<->env integration smoke tests PASSED!")


if __name__ == "__main__":
    run_policy_smoke_test()