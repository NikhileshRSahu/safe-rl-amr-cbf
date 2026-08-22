import math
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cbf import build_filter_from_config, recommended_slack_max
from tests.test_cbf_correctness import TEST_CASES
from utils import (
    compute_barrier_value,
    closest_point_on_rectangle,
    squared_distance,
)
from config import SHELVES, MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y

def test_slack_physical_safety():
    f = build_filter_from_config()
    cfg = f.config
    cfg.enable_slack = True
    cfg.slack_max = recommended_slack_max(cfg.gamma, cfg.safety_margin)
    
    slack_used_count = 0
    total_delta = 0.0
    max_delta = 0.0
    
    for robot_state, dyn_obs, shelves, v_nom, omega_nom in TEST_CASES:
        rx, ry, theta = robot_state[0], robot_state[1], robot_state[2]
        
        v_safe, omega_safe, diag = f.solve(v_nom, omega_nom, robot_state, dyn_obs, shelves)
        delta_used = diag.get("slack_used") or 0.0
        
        if delta_used > 1e-6:
            slack_used_count += 1
            total_delta += delta_used
            if delta_used > max_delta:
                max_delta = delta_used
                
            # Assert the true physical h never goes negative under this worst-case delta
            # The worst-case buffered h is -delta_used / gamma
            worst_case_buffered_h = -delta_used / cfg.gamma
            
            # For dynamic obstacles:
            safe_radius_dyn_buffered = cfg.robot_radius + cfg.dynamic_obs_radius + cfg.safety_margin + cfg.tracking_uncertainty
            # physical radius doesn't include safety_margin or tracking_uncertainty
            safe_radius_dyn_physical = cfg.robot_radius + cfg.dynamic_obs_radius
            
            # distance^2 = h_buffered + safe_radius_dyn_buffered^2
            worst_case_dist_sq_dyn = worst_case_buffered_h + safe_radius_dyn_buffered**2
            worst_case_true_h_dyn = worst_case_dist_sq_dyn - safe_radius_dyn_physical**2
            assert worst_case_true_h_dyn >= -1e-5, f"True physical dyn h violated! {worst_case_true_h_dyn}"
            
            # For static obstacles:
            safe_radius_static_buffered = cfg.robot_radius + cfg.safety_margin
            safe_radius_static_physical = cfg.robot_radius
            
            worst_case_dist_sq_static = worst_case_buffered_h + safe_radius_static_buffered**2
            worst_case_true_h_static = worst_case_dist_sq_static - safe_radius_static_physical**2
            assert worst_case_true_h_static >= -1e-5, f"True physical static h violated! {worst_case_true_h_static}"

    print("\n=== Slack Physical Safety Results ===")
    print(f"Cases using slack: {slack_used_count} / {len(TEST_CASES)}")
    if slack_used_count > 0:
        print(f"Average delta used: {total_delta / slack_used_count:.6f}")
        print(f"Max delta used: {max_delta:.6f}")
    print("Confirmed 0 true physical-safety violations (worst-case true h >= 0).")

if __name__ == "__main__":
    test_slack_physical_safety()
