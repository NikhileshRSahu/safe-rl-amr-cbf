import sys
from pathlib import Path
import numpy as np
import torch
import scipy.optimize as opt
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbf import build_filter_from_config
from environment import AMRWarehouseEnv
from evaluate import load_agent, augment_observation
from config import SHELVES

def run_probe():
    cbf = build_filter_from_config()
    cfg = cbf.config
    cfg.enable_slack = True
    cfg.slack_max = 0.15
    
    env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
    env.cbf_filter = cbf
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_agent("checkpoints/safe_sac_20260802_113318/best_model.pt", env, device=device)
    
    messages = Counter()
    resolved_by_tuning = 0
    total_108 = 0
    
    print("Running to capture the 108 solver failures...")
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
                cbf.config.slack_max = 1000.0
                A_list, b_list = cbf.active_barrier_constraints(curr_robot_state, curr_dyn_obs, SHELVES)
                _, _, delta_req, succ = cbf._solve_scipy_slack(action[0], action[1], A_list, b_list)
                cbf.config.slack_max = 0.15
                
                if succ and delta_req <= 0.15:
                    total_108 += 1
                    
                    u_nom = np.array([action[0], action[1]])
                    A_mat = np.array(A_list)
                    b_vec = np.array(b_list)
                    x0 = np.array([action[0], action[1], 0.0])
                    
                    def objective(x):
                        return 0.5 * np.sum((x[:2] - u_nom)**2) + cfg.slack_penalty_weight * (x[2]**2)
                    def constraint(x):
                        return b_vec + x[2] - np.dot(A_mat, x[:2])
                    
                    bounds = (cfg.v_bounds, cfg.omega_bounds, (0.0, 0.15))
                    cons = {"type": "ineq", "fun": constraint}
                    
                    res_original = opt.minimize(
                        objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 50, "ftol": 1e-3}
                    )
                    messages[res_original.message] += 1
                    
                    res_tuned = opt.minimize(
                        objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 500, "ftol": 1e-6}
                    )
                    if res_tuned.success and res_tuned.x[2] <= 0.15:
                        resolved_by_tuning += 1

            if agent.policy_config.use_attention_obstacles:
                obs = augment_observation(obs_next, env, agent.policy_config.max_obstacles)
            else:
                obs = obs_next
            done = terminated or truncated

    print(f"\nTotal 108-cases found: {total_108}")
    print("Messages from original SLSQP failure:")
    for msg, count in messages.items():
        print(f"  {count}x: {msg}")
    print(f"Cases resolved by maxiter=500, ftol=1e-6: {resolved_by_tuning} / {total_108}")

if __name__ == "__main__":
    run_probe()
