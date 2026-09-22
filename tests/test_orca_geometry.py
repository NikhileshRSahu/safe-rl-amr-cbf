import numpy as np

from benchmark.orca_geometry import (
    OrcaLine,
    build_orca_line,
    project_velocity_to_orca_halfplanes,
    satisfies_orca_line,
)


def test_head_on_constraint_rejects_collision_course():
    rel_p = np.array([2.0, 0.0])
    rel_v = np.array([-2.0, 0.0])
    line = build_orca_line(
        rel_p,
        rel_v,
        combined_radius=0.6,
        time_horizon=2.0,
        responsibility=0.5,
    )
    assert not satisfies_orca_line(np.array([-2.0, 0.0]), line)


def test_safe_lateral_velocity_satisfies_constraint():
    rel_p = np.array([2.0, 0.0])
    rel_v = np.array([-2.0, 0.0])
    line = build_orca_line(
        rel_p,
        rel_v,
        combined_radius=0.6,
        time_horizon=2.0,
        responsibility=0.5,
    )
    assert satisfies_orca_line(np.array([0.0, 0.8]), line)


def test_projection_returns_closest_feasible_velocity():
    lines = [
        OrcaLine(point=np.array([0.2, 0.0]), normal=np.array([-1.0, 0.0])),
    ]
    v = project_velocity_to_orca_halfplanes(
        np.array([1.0, 0.0]),
        lines,
        max_speed=1.0,
    )
    assert v is not None
    assert v[0] <= 0.2 + 1e-6


def test_projection_respects_speed_limit():
    v = project_velocity_to_orca_halfplanes(
        np.array([3.0, 0.0]),
        [],
        max_speed=1.0,
    )
    assert v is not None
    assert np.linalg.norm(v) <= 1.0 + 1e-9


def test_projection_returns_none_for_infeasible_halfplanes():
    lines = [
        OrcaLine(point=np.array([0.8, 0.0]), normal=np.array([1.0, 0.0])),
        OrcaLine(point=np.array([-0.8, 0.0]), normal=np.array([-1.0, 0.0])),
    ]
    assert project_velocity_to_orca_halfplanes(np.zeros(2), lines, max_speed=0.5) is None
