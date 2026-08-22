import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import scipy.optimize as opt
import torch
from cbf import build_filter_from_config
from environment import AMRWarehouseEnv
from evaluate import load_agent
from config import SHELVES

cbf = build_filter_from_config()
cbf.config.enable_slack = True
cbf.config.slack_max = 0.15

env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
env.cbf_filter = cbf
agent = load_agent('checkpoints/safe_sac_20260802_113318/best_model.pt', env, device='cpu')

results = []

for ep in range(50):
    obs, info = env.reset(seed=42 + ep)
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        
        # Buffer t state
        curr_robot_state = env.robot_state.copy()
        curr_dyn_obs = env.dynamic_obstacles.copy() if env.dynamic_obstacles is not None else []
        
        v, w, diag = cbf.solve(action[0], action[1], curr_robot_state, curr_dyn_obs, SHELVES)
        
        if diag.get("tier") == "hard_stop":
            # Direct unbounded minimization
            cfg = cbf.config
            u_nom = np.array([action[0], action[1]])
            A_list, b_list = cbf.active_barrier_constraints(curr_robot_state, curr_dyn_obs, SHELVES)
            A_mat = np.array(A_list)
            b_vec = np.array(b_list)
            
            x0 = np.array([action[0], action[1], 0.0])
            def objective(x): return 0.5 * np.sum((x[:2] - u_nom)**2) + cfg.slack_penalty_weight * (x[2]**2)
            def constraint(x): return b_vec + x[2] - np.dot(A_mat, x[:2])
            
            bounds = (cfg.v_bounds, cfg.omega_bounds, (0.0, 1000.0))
            cons = {"type": "ineq", "fun": constraint}
            
            res = opt.minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                options={"maxiter": 50, "ftol": 1e-3}
            )
            
            results.append({
                "ep": ep,
                "delta_required": float(res.x[2]) if res.success else None,
                "success": res.success,
                "status": res.status,
                "message": res.message
            })
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

with open('experiments/hardstop_details.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["ep", "delta_required", "success", "status", "message"])
    writer.writeheader()
    writer.writerows(results)

successes = [r for r in results if r["success"]]
failures = [r for r in results if not r["success"]]

print(f"Total hard stops: {len(results)}")
print(f"Solver successes (found a delta): {len(successes)}")
print(f"Solver failures: {len(failures)}")

if len(failures) > 0:
    for f in failures:
        print(f"Failure: {f['message']}")

deltas = np.array([r["delta_required"] for r in successes])
print(f"Delta distribution for successes:")
print(f"  < 0.15: {(deltas <= 0.15).sum()}")
print(f"  0.15 - 0.20: {((deltas > 0.15) & (deltas <= 0.20)).sum()}")
print(f"  > 0.20: {(deltas > 0.20).sum()}")
print(f"  Max delta: {deltas.max():.4f}")
print(f"  Mean delta: {deltas.mean():.4f}")
