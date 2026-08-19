import sys
from pathlib import Path
# Ensure repo root is on sys.path when running from scripts/ directory
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from environment import AMRWarehouseEnv
from baselines import PureCBF

seeds = [999 + 3, 999 + 4, 999 + 19]  # episodes 4,5,20 from earlier run

for seed in seeds:
    print(f"\n--- Diagnostic run for seed={seed} ---")
    env = AMRWarehouseEnv(use_cbf_filter=True)
    obs, info = env.reset(seed=seed)
    controller = PureCBF()
    done = False
    step = 0
    while not done and step < env.max_episode_steps:
        action = controller.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        print(f"step={step:03d} interv={info.get('cbf_intervened')} solver_ok={info.get('cbf_solver_success')} active={info.get('cbf_num_active_constraints')} barrier={info.get('barrier_value'):.4f} collision={info.get('collision')} type={info.get('collision_type')} v_act={info.get('v_actual'):.3f} w_act={info.get('omega_actual'):.3f}")
        if info.get('collision'):
            print(f"Collision occurred at step {step} (seed={seed})\n")
            break
        if info.get('goal_reached'):
            print(f"Goal reached at step {step} (seed={seed})\n")
            break
    env.close()
print('\nDiagnostic script complete')
