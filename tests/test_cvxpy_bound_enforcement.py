import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from cbf import CBFFilterConfig, build_filter_from_config
from tests.test_cbf_correctness import TEST_CASES

# Construct an over-constrained case that needs delta > slack_max
# We can just take one of the 89 hard stops from TEST_CASES!
cbf = build_filter_from_config()
cbf.config.enable_slack = True
cbf.config.slack_max = 0.15
cbf.config.backend = "cvxpy"

found_hardstop = False
for robot_state, dyn_obs, shelves, v_nom, omega_nom in TEST_CASES:
    v_safe, omega_safe, diag = cbf.solve(v_nom, omega_nom, robot_state, dyn_obs, shelves)
    if diag["tier"] == "hard_stop":
        found_hardstop = True
        break

assert found_hardstop, "Could not find a hard_stop case"
assert diag["tier"] == "hard_stop", f"Expected hard_stop, got {diag}"
assert (v_safe, omega_safe) == (0.0, 0.0)
print("PASS (slack_max=0.15):", diag)

# Now probe with absurdly small slack_max
cbf2 = build_filter_from_config()
cbf2.config.enable_slack = True
cbf2.config.slack_max = 1e-6
cbf2.config.backend = "cvxpy"
v2, w2, diag2 = cbf2.solve(v_nom, omega_nom, robot_state, dyn_obs, shelves)
print("Tiny slack_max case:", diag2)
