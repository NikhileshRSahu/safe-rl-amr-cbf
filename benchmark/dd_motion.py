from __future__ import annotations

import math
import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def simulate_unicycle_arc(
    pose,
    v_cmd: float,
    omega_cmd: float,
    dt: float,
    horizon: float,
) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (3,):
        raise ValueError("pose must be [x,y,theta]")
    if dt <= 0 or horizon < 0:
        raise ValueError("dt must be positive and horizon non-negative")
    steps = max(1, int(math.ceil(horizon / dt)))
    x, y, th = map(float, pose)
    out = [np.array([x, y, th], dtype=float)]
    for _ in range(steps):
        th = wrap(th + float(omega_cmd) * dt)
        x += math.cos(th) * float(v_cmd) * dt
        y += math.sin(th) * float(v_cmd) * dt
        out.append(np.array([x, y, th], dtype=float))
    return np.asarray(out)


def command_terminal_velocity(
    theta: float,
    v_cmd: float,
    omega_cmd: float,
    dt: float,
) -> np.ndarray:
    th = wrap(float(theta) + float(omega_cmd) * float(dt))
    return np.array(
        [math.cos(th) * float(v_cmd), math.sin(th) * float(v_cmd)],
        dtype=float,
    )


def _point_rect_distance(x: float, y: float, rect) -> float:
    xmin, ymin, xmax, ymax = rect
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def arc_static_clearance(
    arc: np.ndarray,
    shelves,
    world_bound: float,
    robot_radius: float,
) -> float:
    arc = np.asarray(arc, dtype=float)
    best = float("inf")
    for q in arc[:, :2]:
        wall = min(
            q[0] + world_bound - robot_radius,
            world_bound - q[0] - robot_radius,
            q[1] + world_bound - robot_radius,
            world_bound - q[1] - robot_radius,
        )
        best = min(best, float(wall))
        for rect in shelves:
            best = min(
                best,
                _point_rect_distance(float(q[0]), float(q[1]), rect) - robot_radius,
            )
    return float(best)


def reachable_commands(
    current_speed: float,
    current_omega: float,
    *,
    preferred_speed: float,
    preferred_omega: float,
    v_max: float,
    w_max: float,
    speed_samples: int,
    omega_samples: int,
):
    if speed_samples < 2 or omega_samples < 3:
        raise ValueError("need at least 2 speed and 3 omega samples")
    speeds = np.unique(
        np.clip(
            np.r_[
                0.0,
                current_speed,
                preferred_speed,
                np.linspace(0.0, v_max, speed_samples),
            ],
            0.0,
            v_max,
        )
    )
    omegas = np.unique(
        np.clip(
            np.r_[
                0.0,
                current_omega,
                preferred_omega,
                np.linspace(-w_max, w_max, omega_samples),
            ],
            -w_max,
            w_max,
        )
    )
    cmds = {(float(v), float(w)) for v in speeds for w in omegas}
    cmds.add((0.0, 0.0))
    return sorted(
        cmds,
        key=lambda x: (
            x[0] != 0.0 or x[1] != 0.0,
            abs(x[0] - preferred_speed) + 0.25 * abs(x[1] - preferred_omega),
            x[0],
            x[1],
        ),
    )
