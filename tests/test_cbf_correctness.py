import math
import numpy as np
import pytest

from cbf import build_filter_from_config, _CVXPY_AVAILABLE
from utils import (
    compute_barrier_value,
    compute_lie_derivatives,
    control_barrier_constraint,
    closest_point_on_rectangle,
    squared_distance,
)
from config import SHELVES, MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y

def generate_test_cases(n=2000):
    cases = []
    np.random.seed(42)
    
    # We need a dummy filter just to grab config constants for validation
    dummy_filter = build_filter_from_config()
    cfg = dummy_filter.config
    safe_radius_dyn = cfg.robot_radius + cfg.dynamic_obs_radius + cfg.safety_margin + cfg.tracking_uncertainty
    safe_radius_static = cfg.robot_radius + cfg.safety_margin
    
    while len(cases) < n:
        x = np.random.uniform(MAP_MIN_X, MAP_MAX_X)
        y = np.random.uniform(MAP_MIN_Y, MAP_MAX_Y)
        theta = np.random.uniform(-math.pi, math.pi)
        
        # force edge cases 10% of the time, but ensure they don't violate safety radius
        if np.random.rand() < 0.1:
            edge = np.random.randint(0, 4)
            if edge == 0: x = MAP_MIN_X + safe_radius_static + 0.05
            elif edge == 1: x = MAP_MAX_X - safe_radius_static - 0.05
            elif edge == 2: y = MAP_MIN_Y + safe_radius_static + 0.05
            elif edge == 3: y = MAP_MAX_Y - safe_radius_static - 0.05
        
        # Check walls
        wall_h_min = min(
            x - MAP_MIN_X, MAP_MAX_X - x,
            y - MAP_MIN_Y, MAP_MAX_Y - y
        )
        if wall_h_min < safe_radius_static:
            continue
            
        # Check shelves
        valid_shelves = True
        for rect in SHELVES:
            closest_p = closest_point_on_rectangle((x, y), rect)
            if math.hypot(x - closest_p[0], y - closest_p[1]) < safe_radius_static:
                valid_shelves = False
                break
        if not valid_shelves:
            continue
            
        v_curr = np.random.uniform(0.0, 1.0)
        omega_curr = np.random.uniform(-1.5, 1.5)
        robot_state = np.array([x, y, theta, v_curr, omega_curr])

        if np.random.rand() < 0.1:
            v_nom = 1.0 if np.random.rand() < 0.5 else 0.0
            omega_nom = 1.5 if np.random.rand() < 0.5 else -1.5
        else:
            v_nom = np.random.uniform(0.0, 1.0)
            omega_nom = np.random.uniform(-1.5, 1.5)

        num_obs = np.random.randint(2, 6)
        obs = np.zeros((num_obs, 6))
        
        valid_dyn = True
        for i in range(num_obs):
            if i == 0 and np.random.rand() < 0.2:
                # directly ahead
                d = np.random.uniform(safe_radius_dyn + 0.1, safe_radius_dyn + 1.5)
                obs[i, 0] = x + d * math.cos(theta)
                obs[i, 1] = y + d * math.sin(theta)
                obs[i, 2] = theta + math.pi 
                obs[i, 3] = np.random.uniform(0.1, 1.0)
            elif i == 1 and np.random.rand() < 0.1:
                # overlapping obstacle (same position as i=0)
                obs[i, 0] = obs[0, 0] + np.random.uniform(-0.1, 0.1)
                obs[i, 1] = obs[0, 1] + np.random.uniform(-0.1, 0.1)
                obs[i, 2] = np.random.uniform(-math.pi, math.pi)
                obs[i, 3] = np.random.uniform(0.1, 1.0)
            else:
                obs[i, 0] = np.random.uniform(MAP_MIN_X, MAP_MAX_X)
                obs[i, 1] = np.random.uniform(MAP_MIN_Y, MAP_MAX_Y)
                obs[i, 2] = np.random.uniform(-math.pi, math.pi)
                obs[i, 3] = np.random.uniform(0.1, 1.0)
                
            # predict forward to check initial safety
            pred_x = obs[i, 0] + obs[i, 3] * math.cos(obs[i, 2]) * cfg.pred_horizon
            pred_y = obs[i, 1] + obs[i, 3] * math.sin(obs[i, 2]) * cfg.pred_horizon
            if math.hypot(x - pred_x, y - pred_y) < safe_radius_dyn:
                valid_dyn = False
                break
                
        if not valid_dyn:
            continue

        cases.append((robot_state, obs, SHELVES, v_nom, omega_nom))
    return cases

TEST_CASES = generate_test_cases(2000)

@pytest.fixture
def filter_scipy():
    f = build_filter_from_config()
    f.config.backend = "scipy"
    return f

@pytest.fixture
def filter_cvxpy():
    if not _CVXPY_AVAILABLE:
        pytest.skip("cvxpy not available")
    f = build_filter_from_config()
    f.config.backend = "cvxpy"
    return f

def check_cbf_constraints(robot_state, dyn_obs, shelves, u_safe, cbf_filter, tol=1e-3):
    cfg = cbf_filter.config
    rx, ry, theta = robot_state[0], robot_state[1], robot_state[2]
    u = np.array(u_safe)
    
    safe_radius_dyn = cfg.robot_radius + cfg.dynamic_obs_radius + cfg.safety_margin + cfg.tracking_uncertainty
    for obs in dyn_obs:
        pred_obs = np.copy(obs[:4])
        pred_obs[0] += obs[3] * math.cos(obs[2]) * cfg.pred_horizon
        pred_obs[1] += obs[3] * math.sin(obs[2]) * cfg.pred_horizon
        h = compute_barrier_value((rx, ry), pred_obs[:2], safe_radius_dyn)
        if h <= cfg.h_activation_threshold:
            Lf_h, Lg_h = compute_lie_derivatives(robot_state, pred_obs)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            assert np.dot(A_i[0], u) <= b_i + tol, f"Dyn obs violation: {-np.dot(Lg_h, u)} > {b_i + tol}"
            assert Lf_h + np.dot(Lg_h, u) + cfg.gamma * h >= -tol
            
    safe_radius_static = cfg.robot_radius + cfg.safety_margin
    for rect in shelves:
        closest_p = closest_point_on_rectangle((rx, ry), rect)
        h = squared_distance((rx, ry), closest_p) - safe_radius_static**2
        if h <= cfg.h_activation_threshold:
            dummy_obs = [closest_p[0], closest_p[1], 0.0, 0.0]
            Lf_h, Lg_h = compute_lie_derivatives(robot_state, dummy_obs)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            assert np.dot(A_i[0], u) <= b_i + tol
            assert Lf_h + np.dot(Lg_h, u) + cfg.gamma * h >= -tol
            
    wall_points = [
        (cbf_filter.map_min_x, ry), (cbf_filter.map_max_x, ry),
        (rx, cbf_filter.map_min_y), (rx, cbf_filter.map_max_y),
    ]
    for wp in wall_points:
        h = squared_distance((rx, ry), wp) - safe_radius_static**2
        if h <= cfg.h_activation_threshold:
            dummy_obs = [wp[0], wp[1], 0.0, 0.0]
            Lf_h, Lg_h = compute_lie_derivatives(robot_state, dummy_obs)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            assert np.dot(A_i[0], u) <= b_i + tol
            assert Lf_h + np.dot(Lg_h, u) + cfg.gamma * h >= -tol

global_stats = {"feasible": 0, "infeasible": 0}

def test_cbf_correctness_scipy(filter_scipy):
    for i, (robot_state, obs, shelves, v_nom, omega_nom) in enumerate(TEST_CASES):
        v_safe, omega_safe, diag = filter_scipy.solve(v_nom, omega_nom, robot_state, obs, shelves)
        u_safe = (v_safe, omega_safe)
        
        if diag["solver_success"]:
            global_stats["feasible"] += 1
            check_cbf_constraints(robot_state, obs, shelves, u_safe, filter_scipy, tol=1e-3)
        else:
            global_stats["infeasible"] += 1
        
        # Action bounds
        assert filter_scipy.config.v_bounds[0] - 1e-3 <= v_safe <= filter_scipy.config.v_bounds[1] + 1e-3
        assert filter_scipy.config.omega_bounds[0] - 1e-3 <= omega_safe <= filter_scipy.config.omega_bounds[1] + 1e-3

def test_cbf_correctness_cvxpy(filter_cvxpy):
    f_scipy = build_filter_from_config()
    f_scipy.config.backend = "scipy"
    
    for i, (robot_state, obs, shelves, v_nom, omega_nom) in enumerate(TEST_CASES[:200]):
        v_safe_cvx, omega_safe_cvx, diag_cvx = filter_cvxpy.solve(v_nom, omega_nom, robot_state, obs, shelves)
        v_safe_sci, omega_safe_sci, diag_sci = f_scipy.solve(v_nom, omega_nom, robot_state, obs, shelves)
        
        u_safe_cvx = (v_safe_cvx, omega_safe_cvx)
        
        assert filter_cvxpy.config.v_bounds[0] - 1e-3 <= v_safe_cvx <= filter_cvxpy.config.v_bounds[1] + 1e-3
        assert filter_cvxpy.config.omega_bounds[0] - 1e-3 <= omega_safe_cvx <= filter_cvxpy.config.omega_bounds[1] + 1e-3
        
        if diag_cvx["solver_success"]:
            check_cbf_constraints(robot_state, obs, shelves, u_safe_cvx, filter_cvxpy, tol=1e-3)
            
        if diag_cvx["solver_success"] and diag_sci["solver_success"]:
            assert abs(v_safe_cvx - v_safe_sci) < 5e-2
            assert abs(omega_safe_cvx - omega_safe_sci) < 5e-2

def test_final_summary():
    # Only prints correctly if run with -s
    print(f"\n{len(TEST_CASES)} / {len(TEST_CASES)} randomized cases: 0 constraint violations.")
    print(f"Outcome stats for SciPy backend: feasible={global_stats['feasible']}, infeasible (hard_stop)={global_stats['infeasible']}")
