import os
import sys
from pathlib import Path
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbf import build_filter_from_config
from environment import AMRWarehouseEnv
from evaluate import load_agent, augment_observation
from config import SHELVES

def run_reconciliation():
    cbf = build_filter_from_config()
    cbf.config.enable_slack = True
    cbf.config.slack_max = 0.15

    env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
    env.cbf_filter = cbf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_agent("checkpoints/safe_sac_20260802_113318/best_model.pt", env, device=device)

    reasons = Counter()
    total_hard_stops = 0

    print("Running 50 episodes for strict reconciliation...")
    for ep in range(50):
        obs, info = env.reset(seed=42 + ep)
        if agent.policy_config.use_attention_obstacles:
            obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
            
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            # Capture BEFORE step
            curr_robot_state = env.robot_state.copy()
            curr_dyn_obs = env.dynamic_obstacles.copy() if env.dynamic_obstacles is not None else []
            
            obs_next, reward, terminated, truncated, info = env.step(action)
            diag = env.last_cbf_diagnostics
            
            if diag.get("tier") == "hard_stop":
                total_hard_stops += 1
                
                # Test what it would have taken with unbounded slack ON THE STATE THAT FAILED
                cbf.config.slack_max = 1000.0
                A_list, b_list = cbf.active_barrier_constraints(curr_robot_state, curr_dyn_obs, SHELVES)
                v_safe, omega_safe, delta_req, succ = cbf._solve_scipy_slack(action[0], action[1], A_list, b_list)
                cbf.config.slack_max = 0.15
                
                if succ:
                    if delta_req > 0.15:
                        reasons["formally_exceeds_slack_max"] += 1
                    else:
                        reasons["bounded_solver_failed_but_unbounded_succeeded_within_bound"] += 1
                else:
                    reasons["solver_error_no_delta_estimate"] += 1
                    
            if agent.policy_config.use_attention_obstacles:
                obs = augment_observation(obs_next, env, agent.policy_config.max_obstacles)
            else:
                obs = obs_next
                
            done = terminated or truncated

    print(f"\nTotal Hard Stops Logged: {total_hard_stops} (Expected ~697)")
    for k, v in reasons.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_reconciliation()
