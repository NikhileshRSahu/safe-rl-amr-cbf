import numpy as np

from benchmark.progressive_velocity_baselines import (
    AStarRVODD,
    AStarVODD,
    finite_horizon_collision,
    rvo_peer_effective_velocity,
)
from benchmark.beast_classical import AStarORCADD


def test_finite_horizon_collision_detects_closing_motion():
    assert finite_horizon_collision(
        ego_pos=np.array([0.0, 0.0]),
        ego_vel=np.array([1.0, 0.0]),
        other_pos=np.array([2.0, 0.0]),
        other_vel=np.array([0.0, 0.0]),
        combined_radius=0.6,
        horizon=3.0,
    )


def test_finite_horizon_collision_rejects_separating_motion():
    assert not finite_horizon_collision(
        ego_pos=np.array([0.0, 0.0]),
        ego_vel=np.array([-1.0, 0.0]),
        other_pos=np.array([2.0, 0.0]),
        other_vel=np.array([0.0, 0.0]),
        combined_radius=0.6,
        horizon=3.0,
    )


def test_finite_horizon_clamps_closest_approach_to_horizon():
    assert not finite_horizon_collision(
        ego_pos=np.array([0.0, 0.0]),
        ego_vel=np.array([0.1, 0.0]),
        other_pos=np.array([10.0, 0.0]),
        other_vel=np.array([0.0, 0.0]),
        combined_radius=0.6,
        horizon=2.0,
    )


def test_rvo_peer_transform_matches_reciprocal_velocity_obstacle_definition():
    current = np.array([0.8, -0.2])
    candidate = np.array([0.3, 0.4])
    expected = 2.0 * candidate - current
    np.testing.assert_allclose(rvo_peer_effective_velocity(candidate, current), expected)


def test_vo_rvo_and_orca_are_distinct_controller_classes():
    assert AStarVODD is not AStarRVODD
    assert AStarVODD is not AStarORCADD
    assert AStarRVODD is not AStarORCADD
