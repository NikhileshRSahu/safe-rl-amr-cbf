"""
Utility functions for Safe Reinforcement Learning Autonomous Mobile Robot (AMR).

This module provides geometric computations, coordinate frame transformations,
collision detection algorithms, Control Barrier Function (CBF) helpers, random
pose sampling, warehouse environment helpers, and math/randomness utility functions
for the unicycle mobile robot.

Mathematical Reference
----------------------
1. Continuous Angle Normalization:
   \text{wrap\_to\_pi}(\theta) = \text{atan2}(\sin(\theta), \cos(\theta)) \in [-\pi, \pi]

2. 2D Coordinate Transformation (World to Robot Frame):
   \begin{bmatrix} x_r \\ y_r \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_w - x_{\text{robot}} \\ y_w - y_{\text{robot}} \end{bmatrix}

3. Axis-Aligned Rectangle Distance:
   d(\mathbf{p}, \mathcal{R}) = \sqrt{ \max(0, x_{\text{min}} - x)^2 + \max(0, x - x_{\text{max}})^2 + \max(0, y_{\text{min}} - y)^2 + \max(0, y - y_{\text{max}})^2 }

4. Control Barrier Function Safety Index:
   h(\mathbf{x}) = \|\mathbf{p}_{\text{robot}} - \mathbf{p}_{\text{obs}}\|^2 - r_{\text{safe}}^2 \ge 0
"""

import math
import random
from typing import List, Optional, Tuple, Union, Sequence
import numpy as np

from config import (
    DYNAMIC_OBS_RADIUS,
    DYNAMIC_OBS_SPEED_MAX,
    DYNAMIC_OBS_SPEED_MIN,
    GOAL_TOLERANCE,
    MAP_MAX_X,
    MAP_MAX_Y,
    MAP_MIN_X,
    MAP_MIN_Y,
    NUM_DYNAMIC_OBSTACLES,
    OMEGA_MAX,
    OMEGA_MIN,
    RANDOM_SEED,
    ROBOT_RADIUS,
    SHELVES,
    V_MAX,
    V_MIN,
)

__all__ = [
    # Geometry & Math
    "distance", "squared_distance", "normalize_angle", "angle_difference", "rotation_matrix",
    "heading_vector", "normalize_vector", "project_point_to_segment", "closest_point_on_rectangle",
    "signed_distance", "angle_between_vectors", "polar_to_cartesian", "cartesian_to_polar",
    "clamp", "lerp", "wrap_to_pi",
    
    # Numerical Stability
    "is_near_zero", "safe_divide", "safe_norm",
    
    # Coordinate Transforms
    "world_to_robot", "robot_to_world",
    
    # Kinematics & Unicycle
    "clip_velocity", "propagate_unicycle", "state_jacobian", "control_jacobian",
    
    # Collisions & LiDAR
    "circle_circle_collision", "point_inside_rectangle", "rectangle_distance", 
    "circle_rectangle_collision", "ray_circle_intersection", "ray_rectangle_intersection",
    
    # Control Barrier Functions (CBF)
    "compute_barrier_value", "barrier_gradient", "compute_lie_derivatives", 
    "compute_barrier_derivative", "control_barrier_constraint",
    
    # Dynamic Obstacles & Time-to-Collision
    "closest_obstacle", "predict_obstacle_position", "predict_trajectory",
    "relative_velocity", "compute_ttc", "bounce_from_wall",
    
    # Environment & RL Rewards
    "inside_map", "outside_map", "minimum_distance_to_shelves", 
    "goal_reached", "heading_to_goal", "compute_progress_reward", 
    "compute_safety_reward", "normalize_observation",
    
    # Sampling
    "random_robot_pose", "random_goal_pose", "random_dynamic_obstacle_positions", "seed_everything",
    
    # Batch & Evaluation Utilities
    "batch_distance", "batch_collision", "path_length", "control_effort"
]

# Type Aliases
Point2D = Union[np.ndarray, Tuple[float, float], List[float]]
Pose3D = Union[np.ndarray, Tuple[float, float, float], List[float]]
RectBounds = Tuple[float, float, float, float]


# ==============================================================================
# 1. Numerical Stability Utilities
# ==============================================================================

def is_near_zero(val: float, eps: float = 1e-8) -> bool:
    """Check if a scalar is close to zero."""
    return abs(val) < eps


def safe_divide(num: float, den: float, eps: float = 1e-8) -> float:
    """Safely divide two numbers, avoiding division by zero."""
    if is_near_zero(den, eps):
        return num / math.copysign(eps, den)
    return num / den


def safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    """Compute vector norm safely avoiding zero-gradients in auto-diff pipelines."""
    sq_norm = np.dot(v, v)
    return math.sqrt(max(sq_norm, eps**2))


# ==============================================================================
# 2. Geometry & Coordinate Utilities
# ==============================================================================

def _validate_rect(rect: RectBounds) -> None:
    """Internal helper to validate rectangle bounds."""
    if rect[0] > rect[2] or rect[1] > rect[3]:
        raise ValueError(f"Invalid rectangle bounds: xmin <= xmax and ymin <= ymax violated in {rect}")


def distance(p1: Point2D, p2: Point2D) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def squared_distance(p1: Point2D, p2: Point2D) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(dx * dx + dy * dy)


def normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(angle, (float, int)):
        return math.atan2(math.sin(angle), math.cos(angle))
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_difference(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return float(diff)


def angle_between_vectors(v1: Point2D, v2: Point2D) -> float:
    """Compute the absolute angle in [0, pi] between two 2D vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return abs(math.atan2(det, dot))


def polar_to_cartesian(r: float, theta: float) -> np.ndarray:
    """Convert polar coordinates (radius, angle) to Cartesian (x, y)."""
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """Convert Cartesian coordinates (x, y) to polar (radius, angle)."""
    return math.hypot(x, y), math.atan2(y, x)


def rotation_matrix(theta: float) -> np.ndarray:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)


def heading_vector(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)


def normalize_vector(v: Point2D, eps: float = 1e-8) -> np.ndarray:
    v_arr = np.asarray(v, dtype=np.float64)
    norm = math.hypot(v_arr[0], v_arr[1]) if v_arr.size == 2 else float(np.linalg.norm(v_arr))
    if norm < eps:
        return np.zeros_like(v_arr)
    return v_arr / norm


def project_point_to_segment(point: Point2D, a: Point2D, b: Point2D) -> np.ndarray:
    px, py = point[0], point[1]
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]

    dx, dy = bx - ax, by - ay
    if math.isclose(dx, 0.0, abs_tol=1e-8) and math.isclose(dy, 0.0, abs_tol=1e-8):
        return np.array([ax, ay], dtype=np.float64)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return np.array([ax + t * dx, ay + t * dy], dtype=np.float64)


def closest_point_on_rectangle(point: Point2D, rect: RectBounds) -> np.ndarray:
    """Find the closest point on/in an axis-aligned rectangle to a given point."""
    _validate_rect(rect)
    cx = max(rect[0], min(point[0], rect[2]))
    cy = max(rect[1], min(point[1], rect[3]))
    return np.array([cx, cy], dtype=np.float64)


def world_to_robot(point_world: Union[Point2D, np.ndarray], robot_pose: Pose3D) -> np.ndarray:
    pw = np.asarray(point_world, dtype=np.float64)
    rx, ry, theta = robot_pose[0], robot_pose[1], robot_pose[2]
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    if pw.ndim == 1:
        dx, dy = pw[0] - rx, pw[1] - ry
        return np.array([cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy], dtype=np.float64)

    dx, dy = pw[:, 0] - rx, pw[:, 1] - ry
    return np.column_stack((cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy))


def robot_to_world(point_robot: Union[Point2D, np.ndarray], robot_pose: Pose3D) -> np.ndarray:
    pr = np.asarray(point_robot, dtype=np.float64)
    rx, ry, theta = robot_pose[0], robot_pose[1], robot_pose[2]
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    if pr.ndim == 1:
        return np.array([rx + cos_t * pr[0] - sin_t * pr[1],
                         ry + sin_t * pr[0] + cos_t * pr[1]], dtype=np.float64)

    return np.column_stack((rx + cos_t * pr[:, 0] - sin_t * pr[:, 1],
                            ry + sin_t * pr[:, 0] + cos_t * pr[:, 1]))


# ==============================================================================
# 3. Kinematics & Unicycle Dynamics
# ==============================================================================

def clip_velocity(v: float, omega: float) -> Tuple[float, float]:
    return float(max(V_MIN, min(V_MAX, v))), float(max(OMEGA_MIN, min(OMEGA_MAX, omega)))


def propagate_unicycle(pose: Pose3D, v: float, omega: float, dt: float) -> np.ndarray:
    """
    Propagate unicycle kinematics forward in time using exact integration.
    x(t+dt) = x(t) + v/w * (sin(theta + w*dt) - sin(theta)) if w != 0.
    """
    x, y, theta = pose[0], pose[1], pose[2]
    
    if is_near_zero(omega):
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt
        theta_new = theta
    else:
        theta_new = theta + omega * dt
        x_new = x + (v / omega) * (math.sin(theta_new) - math.sin(theta))
        y_new = y - (v / omega) * (math.cos(theta_new) - math.cos(theta))
        
    return np.array([x_new, y_new, normalize_angle(theta_new)], dtype=np.float64)


def control_jacobian(theta: float) -> np.ndarray:
    """
    Compute control Jacobian G(x) for unicycle model mapping [v, w]^T to [\dot{x}, \dot{y}, \dot{\theta}]^T.
    """
    return np.array([
        [math.cos(theta), 0.0],
        [math.sin(theta), 0.0],
        [0.0, 1.0]
    ], dtype=np.float64)


def state_jacobian(v: float, theta: float) -> np.ndarray:
    """
    Compute state Jacobian df/dx for unicycle model.
    """
    return np.array([
        [0.0, 0.0, -v * math.sin(theta)],
        [0.0, 0.0,  v * math.cos(theta)],
        [0.0, 0.0,  0.0]
    ], dtype=np.float64)


# ==============================================================================
# 4. Raycasting & LiDAR Collision Detection
# ==============================================================================

def signed_distance(point: Point2D, rect: RectBounds) -> float:
    """Compute SDF to a rectangle. Negative if inside, positive if outside."""
    _validate_rect(rect)
    dx = max(rect[0] - point[0], point[0] - rect[2])
    dy = max(rect[1] - point[1], point[1] - rect[3])
    
    if dx <= 0 and dy <= 0:
        return max(dx, dy)
    return math.hypot(max(dx, 0), max(dy, 0))


def ray_circle_intersection(ray_origin: Point2D, ray_dir: Point2D, center: Point2D, radius: float) -> float:
    """Return distance to intersection of ray with circle, or inf if no intersection."""
    oc_x = ray_origin[0] - center[0]
    oc_y = ray_origin[1] - center[1]
    
    b = 2.0 * (oc_x * ray_dir[0] + oc_y * ray_dir[1])
    c = (oc_x * oc_x + oc_y * oc_y) - radius * radius
    desc = b * b - 4 * c
    
    if desc < 0:
        return float('inf')
        
    sqrt_desc = math.sqrt(desc)
    t1 = (-b - sqrt_desc) / 2.0
    t2 = (-b + sqrt_desc) / 2.0
    
    if t1 > 0: return t1
    if t2 > 0: return t2
    return float('inf')


def ray_rectangle_intersection(ray_origin: Point2D, ray_dir: Point2D, rect: RectBounds) -> float:
    """LiDAR simulation: Return distance to intersection of ray with axis-aligned rectangle."""
    _validate_rect(rect)
    tmin, tmax = float('-inf'), float('inf')
    
    for i in range(2):
        if is_near_zero(ray_dir[i]):
            if ray_origin[i] < rect[i] or ray_origin[i] > rect[i+2]:
                return float('inf')
        else:
            invD = 1.0 / ray_dir[i]
            t0 = (rect[i] - ray_origin[i]) * invD
            t1 = (rect[i+2] - ray_origin[i]) * invD
            if t0 > t1: t0, t1 = t1, t0
            tmin = max(tmin, t0)
            tmax = min(tmax, t1)
            
            if tmax < tmin:
                return float('inf')
                
    return tmin if tmin > 0 else float('inf')


# ==============================================================================
# 5. Advanced Control Barrier Functions (CBF)
# ==============================================================================

def compute_barrier_value(robot_position: Point2D, obstacle_position: Point2D, safe_radius: float) -> float:
    return squared_distance(robot_position, obstacle_position) - (safe_radius * safe_radius)


def barrier_gradient(robot_position: Point2D, obstacle_position: Point2D) -> np.ndarray:
    return 2.0 * np.array([
        robot_position[0] - obstacle_position[0],
        robot_position[1] - obstacle_position[1]
    ], dtype=np.float64)


def compute_lie_derivatives(
    robot_pose: Pose3D,
    obs_state: Sequence[float],
    # lookahead_distance: DEAD as of 2026-08-23. Introduced in ccab3ab8 alongside
    # an undocumented slack-relaxed QP variant (see project handoff doc). No live
    # call site in cbf.py/environment.py/safe_sac.py passes a nonzero value --
    # verified via repo-wide grep. Do not pass nonzero here without a full audit.
    lookahead_distance: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Compute Lie derivatives Lf_h and Lg_h for the unicycle CBF formulation.

    Uses a **lookahead-point barrier** when ``lookahead_distance > 0``.  The
    barrier is evaluated at::

        p_L = [rx + L·cos(θ),  ry + L·sin(θ)]

    instead of the robot centre.  The time derivative of p_L is::

        ṗ_Lx = cos(θ)·v  −  L·sin(θ)·ω
        ṗ_Ly = sin(θ)·v  +  L·cos(θ)·ω

    which means both v *and* ω appear in Lg_h — giving the CBF-QP direct
    authority over angular velocity without a full second-order HOCBF.
    When ``lookahead_distance == 0`` the function reduces to the original
    robot-centre formulation (Lg_h[1] = 0, ω unconstrained).

    Args:
        robot_pose: ``[x, y, theta, ...]`` — at least 3 elements.
        obs_state: ``[ox, oy, obs_theta, obs_speed, ...]``.
        lookahead_distance: Distance L (m) of the lookahead point ahead of
            the robot along its heading.  ``0`` → classic robot-centre CBF.

    Returns:
        ``(Lf_h, Lg_h)`` where ``Lf_h`` is a scalar and ``Lg_h`` is a
        ``(2,)`` array for ``[v, ω]``.
    """
    rx, ry, theta = robot_pose[0], robot_pose[1], robot_pose[2]
    ox, oy, obs_theta, obs_v = obs_state[0], obs_state[1], obs_state[2], obs_state[3]

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    L = float(lookahead_distance)

    # Lookahead point (equals robot centre when L == 0).
    pLx = rx + L * cos_t
    pLy = ry + L * sin_t

    # Obstacle velocity.
    vox = obs_v * math.cos(obs_theta)
    voy = obs_v * math.sin(obs_theta)

    # Barrier gradient ∇h w.r.t. the lookahead point.
    dh_dx = 2.0 * (pLx - ox)
    dh_dy = 2.0 * (pLy - oy)

    # Lf_h: drift term — robot drift f(x) is zero for position in the
    # control-affine unicycle model; only obstacle motion contributes.
    Lf_h = dh_dx * (-vox) + dh_dy * (-voy)

    # Lg_h = ∇h · G_L(x) where G_L is the control Jacobian of p_L w.r.t. [v, ω]:
    #   ∂pLx/∂v  = cos(θ),     ∂pLx/∂ω = −L·sin(θ)
    #   ∂pLy/∂v  = sin(θ),     ∂pLy/∂ω =  L·cos(θ)
    Lg_h_v = dh_dx * cos_t + dh_dy * sin_t
    Lg_h_w = dh_dx * (-L * sin_t) + dh_dy * (L * cos_t)
    Lg_h = np.array([Lg_h_v, Lg_h_w], dtype=np.float64)

    # --- Singular guard ----------------------------------------------------- #
    # When the lookahead point coincides with the obstacle (h ≈ −R²), the
    # gradient vanishes and the constraint becomes trivially satisfiable by
    # any action.  In that degenerate case we emit a maximally conservative
    # constraint that forces the QP to stop the robot.
    if math.hypot(Lg_h_v, Lg_h_w) < 1e-8:
        # Override with a v-stop constraint: Lf_h set very negative so the
        # right-hand side forces v → 0.
        Lg_h = np.array([1.0, 0.0], dtype=np.float64)
        Lf_h = -1e3

    return float(Lf_h), Lg_h


def compute_barrier_derivative(Lf_h: float, Lg_h: np.ndarray, control: Sequence[float]) -> float:
    """Compute \dot{h} = Lf_h + Lg_h * u."""
    return Lf_h + np.dot(Lg_h, control)


def control_barrier_constraint(Lf_h: float, Lg_h: np.ndarray, h: float, alpha: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Construct the linear QP constraints for CBF safety filter.
    Returns (A, b) such that A * u \le b guarantees safety.
    Constraint: Lf_h + Lg_h * u + \alpha(h) \ge 0  ==>  -Lg_h * u \le Lf_h + \alpha h
    """
    A = -np.asarray(Lg_h).reshape(1, -1)
    b = float(Lf_h + alpha * h)
    return A, b


# ==============================================================================
# 6. Dynamic Obstacles & Time-To-Collision
# ==============================================================================

def relative_velocity(v1: float, theta1: float, v2: float, theta2: float) -> np.ndarray:
    """Compute relative 2D velocity vector between two dynamic entities."""
    return np.array([
        v1 * math.cos(theta1) - v2 * math.cos(theta2),
        v1 * math.sin(theta1) - v2 * math.sin(theta2)
    ], dtype=np.float64)


def compute_ttc(robot_pos: Point2D, robot_vel_vec: Point2D, obs_pos: Point2D, obs_vel_vec: Point2D) -> float:
    """
    Compute Time-To-Collision (TTC) using relative kinematics.
    TTC = - (dp . dv) / ||dv||^2
    """
    dp = np.array([obs_pos[0] - robot_pos[0], obs_pos[1] - robot_pos[1]])
    dv = np.array([robot_vel_vec[0] - obs_vel_vec[0], robot_vel_vec[1] - obs_vel_vec[1]])
    
    v_sq = np.dot(dv, dv)
    if v_sq < 1e-8:
        return float('inf')
        
    ttc = np.dot(dp, dv) / v_sq
    return float(ttc) if ttc > 0 else float('inf')


def predict_trajectory(obstacle_state: Sequence[float], horizon: int, dt: float) -> np.ndarray:
    """Predict future N=horizon positions [x,y] of an obstacle."""
    x, y, theta, v = obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[3]
    vx = v * math.cos(theta)
    vy = v * math.sin(theta)
    
    t_steps = np.arange(1, horizon + 1) * dt
    traj_x = x + vx * t_steps
    traj_y = y + vy * t_steps
    return np.column_stack((traj_x, traj_y))


def bounce_from_wall(
    obstacle_state: Sequence[float],
    bounds: RectBounds = (MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y),
) -> np.ndarray:
    """Apply elastic collision reflection to dynamic obstacles hitting map boundaries.

    Args:
        obstacle_state: ``[x, y, theta, v]``.
        bounds: ``(xmin, ymin, xmax, ymax)`` — follows the same ``RectBounds``
            convention used everywhere else in this module.

    .. note::
        Previous default was ``(MAP_MIN_X, MAP_MAX_X, MAP_MIN_Y, MAP_MAX_Y)``
        which destructured into ``xmin=MAP_MIN_X, xmax=MAP_MAX_X,
        ymin=MAP_MIN_Y, ymax=MAP_MAX_Y`` — matching the intended semantics
        only by coincidence.  The default is now the canonical
        ``(xmin, ymin, xmax, ymax)`` order consistent with :data:`SHELVES`
        and :func:`closest_point_on_rectangle`.
    """
    x, y, theta, v = obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[3]
    xmin, ymin, xmax, ymax = bounds  # canonical (xmin,ymin,xmax,ymax)

    if x <= xmin or x >= xmax:
        theta = normalize_angle(math.pi - theta)
    if y <= ymin or y >= ymax:
        theta = normalize_angle(-theta)

    return np.array([clamp(x, xmin, xmax), clamp(y, ymin, ymax), theta, v], dtype=np.float64)

# ... [Geometry Collisions, Random Sampling, Warehouse bounds remain identical to previous implementation block] ...

# ==============================================================================
# 7. Reinforcement Learning Rewards & Normalization
# ==============================================================================

def compute_progress_reward(current_pos: Point2D, prev_pos: Point2D, goal_pos: Point2D, scale: float = 1.0) -> float:
    """Dense reward based on progress made towards the goal."""
    prev_dist = distance(prev_pos, goal_pos)
    curr_dist = distance(current_pos, goal_pos)
    return float((prev_dist - curr_dist) * scale)


def compute_safety_reward(closest_dist: float, safe_margin: float = 0.5, penalty: float = -10.0) -> float:
    """Continuous exponential penalty as the robot gets uncomfortably close to obstacles."""
    if closest_dist < 0:
        return penalty # Collision
    if closest_dist < safe_margin:
        return penalty * math.exp(-3.0 * (closest_dist / safe_margin))
    return 0.0


def normalize_observation(val: Union[float, np.ndarray], min_val: float, max_val: float) -> Union[float, np.ndarray]:
    """Min-Max scale observation features into [-1, 1]."""
    return 2.0 * (val - min_val) / safe_divide(max_val - min_val, 1.0) - 1.0


# ==============================================================================
# 8. Batch & Evaluation Metrics (For Vectorized RL)
# ==============================================================================

def batch_distance(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Compute vectorized Euclidean distances for N pairs of points."""
    diff = points1[:, :2] - points2[:, :2]
    return np.sqrt(np.sum(diff * diff, axis=1))


def batch_collision(robot_positions: np.ndarray, obs_positions: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorized boolean collision detection for multiple agents/envs."""
    diff = robot_positions[:, :2] - obs_positions[:, :2]
    sq_dist = np.sum(diff * diff, axis=1)
    return sq_dist <= (threshold * threshold)


def path_length(path_coords: np.ndarray) -> float:
    """Calculate cumulative path distance for evaluation metrics."""
    if len(path_coords) < 2: return 0.0
    diffs = np.diff(path_coords[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def control_effort(action_history: np.ndarray) -> float:
    """Metric for evaluating smoothness / energy consumption in Safe RL."""
    return float(np.sum(np.linalg.norm(action_history, axis=1)))

# ==============================================================================
# 9. Scalar Helpers (clamp / lerp / angle wrap)
# ==============================================================================

def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp a scalar value into the inclusive range ``[lo, hi]``."""
    return max(lo, min(hi, val))


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between ``a`` and ``b`` by fraction ``t`` (unclamped)."""
    return a + (b - a) * t


def wrap_to_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap an angle (radians) into ``[-pi, pi]``. Alias of :func:`normalize_angle`."""
    return normalize_angle(angle)


# ==============================================================================
# 10. Collision Geometry
# ==============================================================================

def point_inside_rectangle(point: Point2D, rect: RectBounds) -> bool:
    """Check whether ``point`` lies on/inside an axis-aligned rectangle."""
    _validate_rect(rect)
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def circle_circle_collision(c1: Point2D, r1: float, c2: Point2D, r2: float) -> bool:
    """Check whether two circles (centers ``c1``/``c2``, radii ``r1``/``r2``) overlap."""
    return squared_distance(c1, c2) <= (r1 + r2) ** 2


def circle_rectangle_collision(center: Point2D, radius: float, rect: RectBounds) -> bool:
    """Check whether a circle overlaps an axis-aligned rectangle."""
    closest = closest_point_on_rectangle(center, rect)
    return squared_distance(center, closest) <= radius * radius


def rectangle_distance(point: Point2D, rect: RectBounds) -> float:
    """Euclidean distance from ``point`` to the nearest edge of a rectangle.

    Returns ``0.0`` if ``point`` is on/inside the rectangle. See module
    docstring formula (3) for the closed-form definition.
    """
    _validate_rect(rect)
    dx = max(rect[0] - point[0], 0.0, point[0] - rect[2])
    dy = max(rect[1] - point[1], 0.0, point[1] - rect[3])
    return math.hypot(dx, dy)


# ==============================================================================
# 11. Warehouse / Map Boundary Helpers
# ==============================================================================

def inside_map(point: Point2D, margin: float = 0.0) -> bool:
    """Check whether ``point`` lies within the warehouse bounds, inset by ``margin``."""
    x, y = point[0], point[1]
    return (
        (MAP_MIN_X + margin) <= x <= (MAP_MAX_X - margin)
        and (MAP_MIN_Y + margin) <= y <= (MAP_MAX_Y - margin)
    )


def outside_map(point: Point2D, margin: float = 0.0) -> bool:
    """Check whether ``point`` lies outside the warehouse bounds, inset by ``margin``."""
    return not inside_map(point, margin=margin)


def minimum_distance_to_shelves(point: Point2D, shelves: Sequence[RectBounds]) -> float:
    """Minimum :func:`rectangle_distance` from ``point`` over all ``shelves``."""
    if not shelves:
        return float("inf")
    return min(rectangle_distance(point, rect) for rect in shelves)


# ==============================================================================
# 12. Goal / Heading Helpers
# ==============================================================================

def goal_reached(robot_pos: Point2D, goal_pos: Point2D, tolerance: float = GOAL_TOLERANCE) -> bool:
    """Check whether the robot is within ``tolerance`` meters of the goal."""
    return distance(robot_pos, goal_pos) <= tolerance


def heading_to_goal(robot_pose: Pose3D, goal_pos: Point2D) -> float:
    """Signed heading error (radians, in ``[-pi, pi]``) from robot heading to goal bearing.

    ``0`` means the robot is pointed directly at the goal; ``+-pi`` means it is
    pointed directly away from it.
    """
    rx, ry, theta = robot_pose[0], robot_pose[1], robot_pose[2]
    bearing = math.atan2(goal_pos[1] - ry, goal_pos[0] - rx)
    return angle_difference(bearing, theta)


# ==============================================================================
# 13. Dynamic Obstacle Prediction
# ==============================================================================

def closest_obstacle(
    robot_pos: Point2D, obstacles: Union[np.ndarray, Sequence[Sequence[float]]]
) -> Tuple[Optional[np.ndarray], float]:
    """Find the nearest obstacle to ``robot_pos``.

    Args:
        robot_pos: ``(x, y)`` robot position.
        obstacles: Array-like of shape ``[N, >=2]``, rows ``[x, y, ...]``.

    Returns:
        Tuple ``(closest_position, distance)``. ``(None, inf)`` if ``obstacles``
        is empty.
    """
    obstacles_arr = np.asarray(obstacles, dtype=np.float64)
    if obstacles_arr.size == 0 or obstacles_arr.shape[0] == 0:
        return None, float("inf")

    robot_xy = np.asarray([robot_pos[0], robot_pos[1]], dtype=np.float64)
    diffs = obstacles_arr[:, :2] - robot_xy
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    idx = int(np.argmin(dists))
    return obstacles_arr[idx, :2].copy(), float(dists[idx])


def predict_obstacle_position(obstacle_state: Sequence[float], dt: float) -> np.ndarray:
    """Predict an obstacle's ``(x, y)`` position ``dt`` seconds ahead under constant velocity.

    Args:
        obstacle_state: ``[x, y, theta, v, ...]`` obstacle row.
        dt: Time horizon in seconds.

    Returns:
        Predicted ``[x, y]`` position.
    """
    x, y, theta, v = obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[3]
    return np.array([x + v * math.cos(theta) * dt, y + v * math.sin(theta) * dt], dtype=np.float64)


# ==============================================================================
# 14. Random Pose Sampling (Rejection Sampling Against Shelves)
# ==============================================================================

def random_robot_pose(
    rng: np.random.Generator,
    shelves: Sequence[RectBounds],
    margin: Optional[float] = None,
    max_attempts: int = 100,
) -> np.ndarray:
    """Sample a random collision-free ``[x, y, theta]`` robot pose via rejection sampling.

    Args:
        rng: NumPy ``Generator`` (e.g. ``env.np_random``) used for sampling.
        shelves: Static shelf rectangles to avoid.
        margin: Clearance radius used for the shelf-collision check. Defaults
            to :data:`ROBOT_RADIUS`.
        max_attempts: Maximum rejection-sampling attempts before giving up.

    Returns:
        ``np.ndarray`` of shape ``(3,)``: ``[x, y, theta]``.

    Raises:
        RuntimeError: If no collision-free pose is found within
            ``max_attempts`` tries.
    """
    clearance = ROBOT_RADIUS if margin is None else margin
    for _ in range(max_attempts):
        x = rng.uniform(MAP_MIN_X + clearance, MAP_MAX_X - clearance)
        y = rng.uniform(MAP_MIN_Y + clearance, MAP_MAX_Y - clearance)
        if not any(circle_rectangle_collision((x, y), clearance, rect) for rect in shelves):
            theta = rng.uniform(-math.pi, math.pi)
            return np.array([x, y, theta], dtype=np.float64)
    raise RuntimeError(
        f"random_robot_pose: failed to find a collision-free pose after {max_attempts} attempts."
    )


def random_goal_pose(
    rng: np.random.Generator,
    robot_pos: Point2D,
    shelves: Sequence[RectBounds],
    min_dist_from_robot: float = 1.0,
    margin: Optional[float] = None,
    max_attempts: int = 100,
) -> np.ndarray:
    """Sample a random collision-free ``[x, y]`` goal position via rejection sampling.

    Args:
        rng: NumPy ``Generator`` used for sampling.
        robot_pos: ``(x, y)`` position the goal must be sufficiently far from.
        shelves: Static shelf rectangles to avoid.
        min_dist_from_robot: Minimum allowed distance from ``robot_pos``.
        margin: Clearance radius used for the shelf-collision check. Defaults
            to :data:`GOAL_TOLERANCE`.
        max_attempts: Maximum rejection-sampling attempts before giving up.

    Returns:
        ``np.ndarray`` of shape ``(2,)``: ``[x, y]``.

    Raises:
        RuntimeError: If no valid goal is found within ``max_attempts`` tries.
    """
    clearance = GOAL_TOLERANCE if margin is None else margin
    for _ in range(max_attempts):
        x = rng.uniform(MAP_MIN_X + clearance, MAP_MAX_X - clearance)
        y = rng.uniform(MAP_MIN_Y + clearance, MAP_MAX_Y - clearance)
        if distance((x, y), robot_pos) < min_dist_from_robot:
            continue
        if any(circle_rectangle_collision((x, y), clearance, rect) for rect in shelves):
            continue
        return np.array([x, y], dtype=np.float64)
    raise RuntimeError(
        f"random_goal_pose: failed to find a valid goal after {max_attempts} attempts."
    )


def random_dynamic_obstacle_positions(
    rng: np.random.Generator,
    num_obstacles: int,
    robot_pos: Point2D,
    goal_pos: Point2D,
    shelves: Sequence[RectBounds],
    min_dist_from_robot: float = 1.0,
    min_dist_from_goal: float = 1.0,
    max_attempts: int = 100,
) -> np.ndarray:
    """Sample ``num_obstacles`` collision-free dynamic-obstacle spawn states.

    Args:
        rng: NumPy ``Generator`` used for sampling.
        num_obstacles: Number of obstacles to place.
        robot_pos: ``(x, y)`` robot position to keep clear of.
        goal_pos: ``(x, y)`` goal position to keep clear of.
        shelves: Static shelf rectangles to avoid.
        min_dist_from_robot: Minimum allowed distance from ``robot_pos``.
        min_dist_from_goal: Minimum allowed distance from ``goal_pos``.
        max_attempts: Maximum rejection-sampling attempts per obstacle.

    Returns:
        ``np.ndarray`` of shape ``(num_obstacles, 4)``, rows ``[x, y, theta, speed]``.

    Raises:
        RuntimeError: If any obstacle cannot be placed within ``max_attempts``
            tries.
    """
    positions = np.zeros((num_obstacles, 4), dtype=np.float64)
    for i in range(num_obstacles):
        placed = False
        for _ in range(max_attempts):
            x = rng.uniform(MAP_MIN_X + DYNAMIC_OBS_RADIUS, MAP_MAX_X - DYNAMIC_OBS_RADIUS)
            y = rng.uniform(MAP_MIN_Y + DYNAMIC_OBS_RADIUS, MAP_MAX_Y - DYNAMIC_OBS_RADIUS)
            if distance((x, y), robot_pos) < min_dist_from_robot:
                continue
            if distance((x, y), goal_pos) < min_dist_from_goal:
                continue
            if any(circle_rectangle_collision((x, y), DYNAMIC_OBS_RADIUS, rect) for rect in shelves):
                continue
            theta = rng.uniform(-math.pi, math.pi)
            speed = rng.uniform(DYNAMIC_OBS_SPEED_MIN, DYNAMIC_OBS_SPEED_MAX)
            positions[i] = [x, y, theta, speed]
            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"random_dynamic_obstacle_positions: failed to place obstacle {i} "
                f"after {max_attempts} attempts."
            )
    return positions


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed the ``random`` and ``numpy`` global RNGs (this module has no torch dependency)."""
    random.seed(seed)
    np.random.seed(seed)