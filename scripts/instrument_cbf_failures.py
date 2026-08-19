import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
from environment import AMRWarehouseEnv
from config import SHELVES

OUTPUT_DIR = repo_root / "diagnostics" / "cbf_failures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seeds corresponding to episodes 4,5,20 from earlier benchmark (base_seed=999)
seeds = [1002, 1003, 1018]

for seed in seeds:
    print(f"Processing seed={seed}")
    env = AMRWarehouseEnv(use_cbf_filter=True)
    obs, info = env.reset(seed=seed)
    step = 0
    dumped = 0
    while True:
        # Use PureCBF nominal controller to reproduce earlier runs (same as benchmark)
        # But we want to capture any solver failures regardless of action source; use a simple nominal action from env to match previous runs
        # For replay, use the PureCBF baseline behavior
        from baselines import PureCBF
        controller = PureCBF()
        action = controller.select_action(obs, info)

        # compute v_nom, omega_nom as env.step does
        a_v, a_omega = float(action[0]), float(action[1])
        from config import V_MIN, V_MAX, OMEGA_MAX
        v_nom = V_MIN + 0.5 * (a_v + 1.0) * (V_MAX - V_MIN)
        omega_nom = a_omega * OMEGA_MAX

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        solver_ok = info.get("cbf_solver_success")
        intervened = info.get("cbf_intervened")

        if solver_ok is False:
            # collect A, b
            A_list, b_list = env.cbf_filter.active_barrier_constraints(env.robot_state, env.dynamic_obstacles, SHELVES)
            out_path = OUTPUT_DIR / f"seed_{seed}_step_{step}.npz"
            np.savez_compressed(
                out_path,
                A_list=A_list,
                b_list=b_list,
                robot_state=env.robot_state,
                dynamic_obstacles=env.dynamic_obstacles,
                shelves=np.array(SHELVES, dtype=object),
                v_nom=v_nom,
                omega_nom=omega_nom,
                last_cbf_diag=env.last_cbf_diagnostics,
                step=np.array([step]),
            )
            print(f"  Dumped failure data to {out_path}")
            dumped += 1

        if terminated or truncated:
            break

    env.close()
    if dumped == 0:
        print(f"  No solver failures detected for seed={seed}")
    else:
        print(f"  Total dumps for seed={seed}: {dumped}")

print("Done")
