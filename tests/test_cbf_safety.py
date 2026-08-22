import math
import numpy as np
import pytest

from utils import compute_lie_derivatives
from cbf import CBFSafetyFilter, CBFFilterConfig
from environment import AMRWarehouseEnv

def test_hocbf_lookahead_derivatives():
    """Verify that using a lookahead distance > 0 gives a non-zero angular coefficient."""
    robot_pose = [0.0, 0.0, 0.0]  # pointing right (0 rad)
    obs_state = [1.0, 1.0, 0.0, 0.0]      # obstacle off-center (ahead and to the left)
    
    # 1. Zero lookahead (classic CBF)
    Lf_0, Lg_0 = compute_lie_derivatives(robot_pose, obs_state, lookahead_distance=0.0)
    assert Lg_0[1] == 0.0, "Classic CBF must have zero angular authority"
    
    # 2. Non-zero lookahead (HOCBF)
    Lf_L, Lg_L = compute_lie_derivatives(robot_pose, obs_state, lookahead_distance=0.25)
    assert abs(Lg_L[1]) > 0.0, f"Lookahead CBF must have non-zero angular authority, got Lg_L={Lg_L}"

def test_singular_lie_derivative_guard():
    """Verify that compute_lie_derivatives handles the degenerate case when lookahead point == obstacle."""
    # Place obstacle exactly at the lookahead point
    lookahead = 0.25
    robot_pose = [0.0, 0.0, 0.0]
    obs_state = [lookahead, 0.0, 0.0, 0.0]
    
    Lf, Lg = compute_lie_derivatives(robot_pose, obs_state, lookahead_distance=lookahead)
    
    # Degenerate case should trigger the singular guard, giving non-zero gradient to force a stop
    assert abs(Lg[0]) > 0 or abs(Lg[1]) > 0, "Singular guard failed to provide a valid gradient"
    assert Lf < -100, "Singular guard failed to impose a tight stopping constraint"

def test_no_nan_in_observations():
    """Verify observations don't contain NaN when taking a sequence of random actions."""
    env = AMRWarehouseEnv(use_cbf_filter=False)
    obs, _ = env.reset(seed=42)
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Check all arrays in dict for NaN
        for key, value in obs.items():
            assert not np.isnan(value).any(), f"NaN detected in observation key {key}"
            
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

def test_cbf_filter_bounds():
    """Verify CBF filter always returns actions within bounds."""
    cfg = CBFFilterConfig(
        robot_radius=0.3,
        dynamic_obs_radius=0.3,
        safety_margin=0.1,
        gamma=1.0,
        lookahead_distance=0.25,
        v_bounds=(0.0, 1.0),
        omega_bounds=(-1.5, 1.5)
    )
    filter = CBFSafetyFilter(cfg, map_bounds=(-10, -10, 10, 10))
    
    robot_state = [0.0, 0.0, 0.0, 0.5, 0.0]
    dynamic_obs = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    static_shelves = []
    
    v_nom, omega_nom = 1.0, 0.0
    safe_v, safe_w, info = filter.solve(v_nom, omega_nom, robot_state, dynamic_obs, static_shelves)
    
    assert 0.0 <= safe_v <= 1.0, f"safe_v {safe_v} out of bounds"
    assert -1.5 <= safe_w <= 1.5, f"safe_w {safe_w} out of bounds"
