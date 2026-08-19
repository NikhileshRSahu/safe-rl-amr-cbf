import numpy as np
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scipy.optimize import linprog
from config import ROBOT_RADIUS, DYNAMIC_OBS_RADIUS, CBF_SAFETY_MARGIN, SHELVES

DUMP_DIR = repo_root / "diagnostics" / "cbf_failures"
files = ["seed_1002_step_27.npz", "seed_1002_step_32.npz"]

def nearest_shelf_distance(robot_xy, shelves):
    rx, ry = robot_xy[:2]
    dmin = float('inf')
    closest = None
    for rect in shelves:
        xmin, ymin, xmax, ymax = rect
        # distance from point to rectangle edge (0 if inside)
        dx = max(xmin - rx, 0, rx - xmax)
        dy = max(ymin - ry, 0, ry - ymax)
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < dmin:
            dmin = dist
            closest = rect
    return dmin

for fn in files:
    path = DUMP_DIR / fn
    if not path.exists():
        print(f"Missing dump: {path}")
        continue
    data = np.load(path, allow_pickle=True)
    A_list = data['A_list']
    b_list = data['b_list']
    robot_state = data['robot_state']
    dyn = data['dynamic_obstacles']

    # compute nearest dynamic obstacle center distance
    rx, ry = float(robot_state[0]), float(robot_state[1])
    dists = [np.hypot(rx - float(o[0]), ry - float(o[1])) for o in dyn]
    nearest_dyn = min(dists) if len(dists)>0 else float('inf')
    # distance margin
    safety_thresh = ROBOT_RADIUS + DYNAMIC_OBS_RADIUS + CBF_SAFETY_MARGIN

    shelf_dist = nearest_shelf_distance(robot_state, SHELVES)

    print(f"\nFile: {fn}")
    print(f" robot pos=({rx:.3f},{ry:.3f})")
    print(f" nearest_dyn_center_dist={nearest_dyn:.4f}, safety_thresh={safety_thresh:.4f}")
    print(f" nearest_shelf_edge_dist={shelf_dist:.4f}")

    # Prepare A,b for linprog feasibility: find u in bounds satisfying A u <= b
    if isinstance(A_list, np.ndarray) and A_list.size>0:
        A = np.array(A_list)
        b = np.array(b_list)
        # linprog minimizes c^T u subject to A_ub u <= b_ub and bounds
        c = np.zeros(A.shape[1])
        # bounds from config
        from config import V_MIN, V_MAX, OMEGA_MIN, OMEGA_MAX
        bounds = [(V_MIN, V_MAX), (OMEGA_MIN, OMEGA_MAX)]
        # run linprog
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            feasible = res.success
            print(f" linprog feasibility: success={res.success}, status={res.status}, message={res.message}")
            if feasible:
                print(f"  feasible u={res.x}")
            else:
                print("  no feasible u found by linprog")
        except Exception as e:
            print(" linprog exception:", e)
    else:
        print(" No active constraints in dump (A_list empty)")

# Grep cbf.py for slack-related terms
import re
cbf_path = repo_root / 'cbf.py'
text = cbf_path.read_text()
matches = re.findall(r"\b(slack|SLACK_WEIGHT|delta)\b", text)
print("\ncbf.py slack-related matches:", matches)
