import json
from pathlib import Path
import subprocess
import sys
import math

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from config import DT


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip(), res.stderr.strip(), res.returncode


def get_collision_step(seed):
    python_exe = repo_root / 'venv' / 'Scripts' / 'python.exe'
    cmd = f"{python_exe} scripts/replay_and_capture.py --seeds {seed} --dump-last 4"
    out, err, code = run_cmd(cmd)
    for line in out.splitlines():
        if "collision_step" in line:
            try:
                return int(line.split(":")[-1].strip())
            except Exception:
                continue
    return None


def load_feasibility(seed):
    p = repo_root / f"diagnostics/feasibility_{seed}.json"
    if not p.exists():
        # run analyzer
        cmd = f"$env:PYTHONPATH='.'; {repo_root / 'venv' / 'Scripts' / 'python.exe'} scripts/feasibility_analysis.py --seed {seed} --start-step 1 --out diagnostics/feasibility_{seed}.json"
        run_cmd(cmd)
    with open(p, 'r') as f:
        return json.load(f)


def extract_last_feasible(results):
    # results is list of dicts ordered by step
    last_feasible = None
    for i, r in enumerate(results):
        if r.get('greedy_feasible'):
            # if next exists and is False, this is the flip point
            if i + 1 < len(results) and not results[i+1].get('greedy_feasible'):
                return r
            last_feasible = r
    return last_feasible


def find_npz_for_step(seed, step):
    import glob
    p = repo_root / 'diagnostics' / 'cbf_failures'
    candidates = list(p.glob(f"seed_{seed}_step_{step}*.npz"))
    return candidates[0] if candidates else None


def compute_closing_speed(npz_path):
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    robot = data['robot_state']
    dyn = data['dynamic_obstacles']
    rx, ry, rtheta, rv, romega = [float(x) for x in robot]
    # robot velocity vector
    vrx = rv * math.cos(rtheta)
    vry = rv * math.sin(rtheta)
    # find nearest obstacle
    best = None
    best_d = float('inf')
    for obs in dyn:
        ox, oy, otheta, ospeed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        d = math.hypot(rx - ox, ry - oy)
        if d < best_d:
            best_d = d
            best = (ox, oy, otheta, ospeed)
    if best is None:
        return None, None
    ox, oy, otheta, ospeed = best
    vox = ospeed * math.cos(otheta)
    voy = ospeed * math.sin(otheta)
    relx = ox - rx
    rely = oy - ry
    dist = math.hypot(relx, rely) - (0.30 + 0.30)  # robot+obs radii default; approximate
    # relative velocity along line-of-centers
    vrelx = vox - vrx
    vrely = voy - vry
    # closing rate positive if distance decreasing
    if math.hypot(relx, rely) > 1e-6:
        closing = (vrelx * relx + vrely * rely) / math.hypot(relx, rely)
    else:
        closing = float('inf')
    return dist, closing


def main():
    seeds = [1002, 1003, 1005, 1018]
    rows = []
    for s in seeds:
        results = load_feasibility(s)
        last = extract_last_feasible(results)
        if last is None:
            last_step = None
            nearest = None
            closing = None
        else:
            last_step = int(last['step'])
            npz = find_npz_for_step(s, last_step)
            if npz:
                nearest, closing = compute_closing_speed(npz)
            else:
                nearest, closing = last.get('nearest_dyn_margin'), None

        coll = get_collision_step(s)
        gap_steps = None
        gap_seconds = None
        if last_step is not None and coll is not None:
            gap_steps = coll - last_step
            gap_seconds = gap_steps * DT

        rows.append({
            'seed': s,
            'last_feasible_step': last_step,
            'collision_step': coll,
            'gap_steps': gap_steps,
            'gap_seconds': gap_seconds,
            'nearest_margin_at_last': nearest,
            'closing_speed_at_last': closing,
        })

    # print table
    print("seed | last_feasible | collision | gap_steps | gap_secs | nearest_m | closing_speed")
    for r in rows:
        print(f"{r['seed']} | {r['last_feasible_step']} | {r['collision_step']} | {r['gap_steps']} | {r['gap_seconds']} | {r['nearest_margin_at_last']:.3f} | {r['closing_speed_at_last']:.3f}" )


if __name__ == '__main__':
    main()
