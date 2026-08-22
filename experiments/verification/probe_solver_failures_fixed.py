import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbf import build_filter_from_config
from environment import AMRWarehouseEnv
from evaluate import load_agent
from config import SHELVES
from collections import Counter
import numpy as np

cbf = build_filter_from_config()
cbf.config.enable_slack = True
cbf.config.slack_max = 0.15
env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
env.cbf_filter = cbf
agent = load_agent('checkpoints/safe_sac_20260802_113318/best_model.pt', env, device='cpu')

reasons = Counter()

for ep in range(50):
    obs, info = env.reset(seed=42 + ep)
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        
        # Capture state BEFORE step
        rx, ry, theta, v_curr, omega_curr = env.robot_state
        dyn_obs = []
        for i in range(env.num_obstacles):
            ox, oy = env.obstacles_x[i], env.obstacles_y[i]
            ov, otheta = env.obstacles_v[i], env.obstacles_theta[i]
            o_radius = env.obstacles_radius[i]
            o_active = env.obstacles_active[i]
            if o_active:
                dyn_obs.append(np.array([ox, oy, otheta, ov, omega_curr, o_radius]))
                
        # Manually invoke solve to get the diagnosis without changing state
        v, w, diag = cbf.solve(action[0], action[1], env.robot_state, dyn_obs, SHELVES)
        
        if diag.get("tier") == "hard_stop":
            cbf.config.slack_max = 1000.0
            A_list, b_list = cbf.active_barrier_constraints(env.robot_state, dyn_obs, SHELVES)
            _, _, delta_req, succ = cbf._solve_scipy_slack(action[0], action[1], A_list, b_list)
            cbf.config.slack_max = 0.15
            
            if succ:
                if delta_req > 0.15:
                    reasons["formally_exceeds_slack_max"] += 1
                else:
                    reasons["bounded_solver_failed_but_unbounded_succeeded_within_bound"] += 1
            else:
                reasons["solver_error_no_delta_estimate"] += 1
                
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

for k, v in reasons.items():
    print(f"{k}: {v}")
print(f"Total: {sum(reasons.values())}")
