"""Feasibility analysis for CBF failure dumps.

Checks whether, from a dumped robot/obstacle state, any control sequence
within actuator limits (and respecting accel/alpha limits) can avoid a
collision with the nearest dynamic obstacle using a short greedy search.

Usage:
    python scripts/feasibility_analysis.py --seed 1018 --start-step 66

This is intentionally conservative / approximate: it attempts a best-effort
avoidance policy under the physical limits. If this script reports
"UNAVOIDABLE" for a state, that is strong evidence the encounter is
physically infeasible for the robot's bounds.
"""

from pathlib import Path
import argparse
import numpy as np
import math
import json

repo_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(repo_root))

from config import (
    DT,
    V_MIN,
    V_MAX,
    OMEGA_MIN,
    OMEGA_MAX,
    ACCEL_MAX,
    ALPHA_MAX,
    ROBOT_RADIUS,
    DYNAMIC_OBS_RADIUS,
)

from utils import distance, circle_circle_collision


def _extract_step_from_stem(stem: str):
    # stem examples: 'seed_1018_step_67', 'seed_1018_step_67_replay'
    parts = stem.split("_")
    # find last integer token
    for tok in reversed(parts):
        try:
            return int(tok)
        except Exception:
            continue
    return 0


def load_dumps_for_seed(seed: int, dumps_dir: Path):
    files = list(dumps_dir.glob(f"seed_{seed}_step_*.npz"))
    files = sorted(files, key=lambda p: _extract_step_from_stem(p.stem))
    return files


def simulate_greedy_avoidance(robot_state, dynamic_obstacles, horizon=30, samples=7):
    """Greedy forward simulation: at each timestep choose the reachable
    control (on a small grid) that maximizes the minimum distance to
    dynamic obstacles one step ahead. Returns True if no collision over
    the horizon, False if collision occurs for all attempted choices.
    """
    rx, ry, rtheta, rv, romega = [float(x) for x in robot_state]
    # take only first dynamic obstacle (closest) for this conservative check
    dyn = np.copy(dynamic_obstacles)
    # find nearest dynamic obstacle index by current distance
    if dyn.size == 0:
        return True

    for t in range(horizon):
        # reachable control bounds given accel limits
        v_min_reach = max(V_MIN, rv - ACCEL_MAX * DT)
        v_max_reach = min(V_MAX, rv + ACCEL_MAX * DT)
        w_min_reach = max(OMEGA_MIN, romega - ALPHA_MAX * DT)
        w_max_reach = min(OMEGA_MAX, romega + ALPHA_MAX * DT)

        v_candidates = np.linspace(v_min_reach, v_max_reach, samples)
        w_candidates = np.linspace(w_min_reach, w_max_reach, samples)

        best_margin = -1e9
        best_u = (rv, romega)
        best_next = None

        for v_c in v_candidates:
            for w_c in w_candidates:
                # simulate one step
                nx = rx + v_c * math.cos(rtheta) * DT
                ny = ry + v_c * math.sin(rtheta) * DT
                nth = (rtheta + w_c * DT + math.pi) % (2 * math.pi) - math.pi

                # propagate obstacles one step (constant vel)
                min_margin = 1e9
                for obs in dyn:
                    ox, oy, otheta, ospeed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
                    ox_n = ox + ospeed * math.cos(otheta) * DT
                    oy_n = oy + ospeed * math.sin(otheta) * DT
                    d = math.hypot(nx - ox_n, ny - oy_n) - (ROBOT_RADIUS + DYNAMIC_OBS_RADIUS)
                    min_margin = min(min_margin, d)

                if min_margin > best_margin:
                    best_margin = min_margin
                    best_u = (v_c, w_c)
                    best_next = (nx, ny, nth)

        # apply best control
        rv, romega = best_u
        rx, ry, rtheta = best_next

        # advance all dynamic obstacles in place for next iteration
        for i in range(len(dyn)):
            ox, oy, otheta, ospeed = float(dyn[i][0]), float(dyn[i][1]), float(dyn[i][2]), float(dyn[i][3])
            dyn[i][0] = ox + ospeed * math.cos(otheta) * DT
            dyn[i][1] = oy + ospeed * math.sin(otheta) * DT

        # collision check
        for obs in dyn:
            if circle_circle_collision((rx, ry), ROBOT_RADIUS, (float(obs[0]), float(obs[1])), DYNAMIC_OBS_RADIUS):
                return False

    return True


def analyze_seed_step(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    robot_state = data["robot_state"]
    dynamic_obstacles = data["dynamic_obstacles"]
    last_cbf = data.get("last_cbf_diag", data.get("last_cbf_diagnostics", {}))
    v_nom = float(data.get("v_nom", 0.0))
    omega_nom = float(data.get("omega_nom", 0.0))

    # compute nearest dynamic obstacle distance
    rx, ry = float(robot_state[0]), float(robot_state[1])
    nearest = float("inf")
    for obs in dynamic_obstacles:
        d = math.hypot(rx - float(obs[0]), ry - float(obs[1])) - (ROBOT_RADIUS + DYNAMIC_OBS_RADIUS)
        nearest = min(nearest, d)

    slack = last_cbf.get("slack") if isinstance(last_cbf, dict) else None
    barrier_val = last_cbf.get("barrier_value") if isinstance(last_cbf, dict) else None

    feasible = simulate_greedy_avoidance(robot_state, dynamic_obstacles)

    return {
        "file": str(npz_path.name),
        "step": int(data.get("step", np.array([0]))[0]) if "step" in data else None,
        "nearest_dyn_margin": float(nearest),
        "slack": float(slack) if slack is not None else None,
        "barrier_value": float(barrier_val) if barrier_val is not None else None,
        "greedy_feasible": bool(feasible),
        "v_nom": v_nom,
        "omega_nom": omega_nom,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--start-step", type=int, default=None)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    dumps_dir = repo_root / "diagnostics" / "cbf_failures"
    files = load_dumps_for_seed(args.seed, dumps_dir)
    if not files:
        print(f"No dumps found for seed {args.seed} in {dumps_dir}")
        return

    results = []
    for f in files:
        stepnum = _extract_step_from_stem(f.stem)
        if args.start_step is not None and stepnum < args.start_step:
            continue
        print(f"Analyzing {f.name}...")
        r = analyze_seed_step(f)
        results.append(r)
        print(json.dumps(r, indent=2))

    if args.out:
        outp = Path(args.out)
        outp.write_text(json.dumps(results, indent=2))
        print(f"Wrote report to {outp}")


if __name__ == "__main__":
    main()
