import numpy as np

from benchmark.dd_motion import (
    arc_static_clearance,
    reachable_commands,
    simulate_unicycle_arc,
)


def test_zero_omega_produces_straight_arc():
    arc = simulate_unicycle_arc(np.array([0.0, 0.0, 0.0]), 1.0, 0.0, dt=0.1, horizon=1.0)
    assert np.allclose(arc[-1, :2], [1.0, 0.0], atol=1e-2)


def test_nonzero_omega_produces_curved_arc():
    arc = simulate_unicycle_arc(np.array([0.0, 0.0, 0.0]), 1.0, 1.0, dt=0.05, horizon=1.0)
    assert arc[-1, 1] > 0.1


def test_arc_clearance_detects_turn_into_shelf():
    shelves = [(0.7, 0.35, 1.5, 1.2)]
    arc = simulate_unicycle_arc(np.array([0.0, 0.0, 0.0]), 1.0, 1.0, dt=0.05, horizon=1.0)
    clearance = arc_static_clearance(arc, shelves, world_bound=10.0, robot_radius=0.3)
    assert clearance < 0.1


def test_reachable_commands_are_bounded_and_include_stop():
    cmds = reachable_commands(
        0.5,
        0.2,
        preferred_speed=1.0,
        preferred_omega=1.2,
        v_max=1.0,
        w_max=1.5,
        speed_samples=5,
        omega_samples=7,
    )
    assert (0.0, 0.0) in cmds
    assert all(0.0 <= v <= 1.0 and -1.5 <= w <= 1.5 for v, w in cmds)
