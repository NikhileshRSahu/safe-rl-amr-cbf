import numpy as np
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbf import build_filter_from_config
from environment import AMRWarehouseEnv
from evaluate import load_agent, augment_observation
from config import SHELVES

cbf = build_filter_from_config()
cbf.config.enable_slack = True
cbf.config.slack_max = 0.15

env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
env.cbf_filter = cbf

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = load_agent("checkpoints/safe_sac_20260802_113318/best_model.pt", env, device=device)

deltas_used = []
near_miss_count = 0

print("Running 50 episodes...")
for ep in range(50):
    obs, info = env.reset(seed=42 + ep)
    if agent.policy_config.use_attention_obstacles:
        obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
        
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        diag = env.last_cbf_diagnostics
        
        if diag.get("tier") == "slack" and diag.get("slack_used") is not None:
            deltas_used.append(diag["slack_used"])
        elif diag.get("tier") == "hard_stop":
            # Test what it would have taken with unbounded slack
            cbf.config.slack_max = 1000.0
            # We can reconstruct the constraints via active_barrier_constraints
            A_list, b_list = cbf.active_barrier_constraints(env.robot_state, env.dynamic_obstacles, SHELVES)
            v_safe, omega_safe, delta_req, succ = cbf._solve_scipy_slack(action[0], action[1], A_list, b_list)
            cbf.config.slack_max = 0.15
            if succ and delta_req > 0.15:
                near_miss_count += 1
                
        if agent.policy_config.use_attention_obstacles:
            obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
        done = terminated or truncated

deltas_used = np.array(deltas_used)
print(f"n_rescued={len(deltas_used)}")
if len(deltas_used) > 0:
    print(f"delta stats: mean={deltas_used.mean():.4f}, median={np.median(deltas_used):.4f}, "
          f"p90={np.percentile(deltas_used,90):.4f}, max={deltas_used.max():.4f}")
    print(f"fraction within 10% of slack_max (0.135-0.15): "
          f"{((deltas_used > 0.135).sum() / len(deltas_used)):.2%}")
print(f"hard_stops that needed >slack_max (near-misses): {near_miss_count}")

np.save("experiments/onpolicy_deltas.npy", deltas_used)
