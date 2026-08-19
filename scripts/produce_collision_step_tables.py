import json
from pathlib import Path
import numpy as np
from math import hypot

repo_root = Path(__file__).resolve().parent.parent
post_path = repo_root / "benchmark_results" / "detailed_cbf_only_seed999_with_diags.json"
diag_dir = repo_root / "diagnostics" / "cbf_failures"
out_path = repo_root / "benchmark_results" / "collision_step_tables_seed999.txt"

SEEDS = [1002,1003,1005,1018]
LAST_STEPS = 15

def nearest_dynamic_dist(robot_xy, dyn_obs):
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    if len(dyn_obs)==0:
        return float('inf')
    return min(hypot(rx - float(o[0]), ry - float(o[1])) for o in dyn_obs)

def load_npz_for(seed, step):
    fn = f"seed_{seed}_step_{step}.npz"
    p = diag_dir / fn
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)

def main():
    post = json.loads(post_path.read_text())
    by_seed = {e['seed']: e for e in post}

    lines = []
    for seed in SEEDS:
        ep = by_seed.get(seed)
        if ep is None:
            lines.append(f"Seed {seed}: not found in post results")
            continue
        if not ep.get('collision'):
            lines.append(f"Seed {seed}: no collision in post results")
            continue
        steps = ep.get('steps', [])
        last = steps[-LAST_STEPS:]
        lines.append(f"Seed {seed} - last {LAST_STEPS} steps (step_index,tier,slack,barrier_value,nearest_dyn_dist)")
        for s in last:
            step_idx = s.get('step')
            tier = s.get('tier')
            slack = s.get('slack')
            barrier = s.get('barrier_value')
            # try to load diagnostic npz for this step to compute nearest dynamic obstacle distance
            npz = load_npz_for(seed, step_idx)
            if npz is not None and 'dynamic_obstacles' in npz and 'robot_state' in npz:
                dyn = npz['dynamic_obstacles']
                robot_state = npz['robot_state']
                dist = nearest_dynamic_dist(robot_state, dyn)
                dist_str = f"{dist:.6f}"
            else:
                dist_str = "missing_npz"
            lines.append(f"{step_idx},{tier},{slack},{barrier},{dist_str}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote collision step tables to: {out_path}")

if __name__ == '__main__':
    main()
