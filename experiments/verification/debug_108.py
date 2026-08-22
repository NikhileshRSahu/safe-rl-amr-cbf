import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbf import build_filter_from_config
from environment import AMRWarehouseEnv
from evaluate import load_agent
from config import SHELVES
import scipy.optimize as opt
import numpy as np

cbf = build_filter_from_config()
cbf.config.enable_slack = True
cbf.config.slack_max = 0.15
env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
env.cbf_filter = cbf
agent = load_agent('checkpoints/safe_sac_20260802_113318/best_model.pt', env, device='cpu')

for ep in range(50):
    obs, info = env.reset(seed=42 + ep)
    done = False
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        diag = env.last_cbf_diagnostics
        
        if diag.get('tier') == 'hard_stop':
            cbf.config.slack_max = 1000.0
            A_list, b_list = cbf.active_barrier_constraints(env.robot_state, env.dynamic_obstacles, SHELVES)
            v, w, delta, succ = cbf._solve_scipy_slack(action[0], action[1], A_list, b_list)
            cbf.config.slack_max = 0.15
            if succ and delta <= 0.15:
                print('FOUND ONE 108 case!')
                
                # Test the original bounding again
                v2, w2, delta2, succ2 = cbf._solve_scipy_slack(action[0], action[1], A_list, b_list)
                print(f'Bounded solver returned succ2={succ2}, delta2={delta2}')
                
                # Manual opt to get message
                cfg = cbf.config
                u_nom = np.array([action[0], action[1]])
                A_mat = np.array(A_list)
                b_vec = np.array(b_list)
                x0 = np.array([action[0], action[1], 0.0])
                
                def objective(x): return 0.5 * np.sum((x[:2] - u_nom)**2) + cfg.slack_penalty_weight * (x[2]**2)
                def constraint(x): return b_vec + x[2] - np.dot(A_mat, x[:2])
                
                bounds = (cfg.v_bounds, cfg.omega_bounds, (0.0, 0.15))
                cons = {"type": "ineq", "fun": constraint}
                
                res_original = opt.minimize(
                    objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 50, "ftol": 1e-3}
                )
                print(f'Manual run success={res_original.success}, message={res_original.message}')
                sys.exit(0)
        done = terminated or truncated
