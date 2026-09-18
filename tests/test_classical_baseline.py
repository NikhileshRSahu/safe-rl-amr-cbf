import math
import numpy as np

from classical_baseline import (
    AStarPlanner,
    AStarDWAController,
    DWAConfig,
    physical_to_normalized_action,
)


def test_physical_to_normalized_action_maps_limits():
    assert np.allclose(
        physical_to_normalized_action(0.0, -1.5, v_min=0.0, v_max=1.0, omega_max=1.5),
        [-1.0, -1.0],
    )
    assert np.allclose(
        physical_to_normalized_action(1.0, 1.5, v_min=0.0, v_max=1.0, omega_max=1.5),
        [1.0, 1.0],
    )
    assert np.allclose(
        physical_to_normalized_action(0.5, 0.0, v_min=0.0, v_max=1.0, omega_max=1.5),
        [0.0, 0.0],
    )


def test_astar_returns_collision_free_path_around_shelf():
    planner = AStarPlanner(
        map_bounds=(-5.0, 5.0, -5.0, 5.0),
        shelves=[(-0.8, -3.0, 0.8, 3.0)],
        robot_radius=0.25,
        margin=0.10,
        resolution=0.25,
    )
    path = planner.plan((-4.0, 0.0), (4.0, 0.0))
    assert path
    assert np.linalg.norm(np.asarray(path[0]) - np.array([-4.0, 0.0])) < 0.4
    assert np.linalg.norm(np.asarray(path[-1]) - np.array([4.0, 0.0])) < 0.4
    assert all(planner.is_free_world(p) for p in path)
    assert any(abs(y) > 3.25 for _, y in path)


def test_astar_returns_empty_when_shelf_separates_map():
    planner = AStarPlanner(
        map_bounds=(-2.0, 2.0, -2.0, 2.0),
        shelves=[(-0.3, -2.0, 0.3, 2.0)],
        robot_radius=0.25,
        margin=0.10,
        resolution=0.20,
    )
    assert planner.plan((-1.5, 0.0), (1.5, 0.0)) == []


def test_dwa_command_stays_within_normalized_bounds():
    ctrl = AStarDWAController(
        map_bounds=(-5.0, 5.0, -5.0, 5.0),
        shelves=[],
        robot_radius=0.30,
        dynamic_obstacle_radius=0.35,
        v_min=0.0,
        v_max=1.0,
        omega_min=-1.5,
        omega_max=1.5,
        accel_max=1.5,
        alpha_max=2.5,
        dt=0.1,
        config=DWAConfig(v_samples=5, omega_samples=9, horizon=1.0),
    )
    assert ctrl.reset((-4.0, 0.0), (4.0, 0.0))
    robot = np.array([-4.0, 0.0, 0.0, 0.2, 0.0], dtype=float)
    action = ctrl.command(robot, np.zeros((0, 6), dtype=float))
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_dwa_rejects_straight_collision_with_moving_obstacle():
    ctrl = AStarDWAController(
        map_bounds=(-5.0, 5.0, -5.0, 5.0),
        shelves=[],
        robot_radius=0.30,
        dynamic_obstacle_radius=0.35,
        v_min=0.0,
        v_max=1.0,
        omega_min=-1.5,
        omega_max=1.5,
        accel_max=2.0,
        alpha_max=4.0,
        dt=0.1,
        config=DWAConfig(v_samples=6, omega_samples=15, horizon=1.6, safety_margin=0.10),
    )
    assert ctrl.reset((-2.0, 0.0), (3.0, 0.0))
    robot = np.array([-2.0, 0.0, 0.0, 0.5, 0.0], dtype=float)
    dyn = np.array([[-0.30, -0.70, math.pi / 2.0, 0.50, -0.30, 4.0]], dtype=float)
    action = ctrl.command(robot, dyn)
    v, w = ctrl.normalized_to_physical(action)
    assert ctrl.trajectory_is_safe(robot, v, w, dyn)
    assert abs(w) > 1e-3 or v < 0.45
