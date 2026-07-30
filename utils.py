"""
Utility functions for Safe Reinforcement Learning Autonomous Mobile Robot (AMR).

This module provides geometric computations, coordinate frame transformations,
collision detection algorithms, Control Barrier Function (CBF) helpers, random
pose sampling, warehouse environment helpers, and math/randomness utility functions
for the unicycle mobile robot.
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
    return abs(val) < eps

def safe_divide(num: float, den: float, eps: float = 1e-8) -> float:
    if is_near_zero(den, eps):
        return num / math.copysign(eps, den)
    return num / den

def safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    sq_norm = np.dot(v, v)
    return math.sqrt(max(sq_norm, eps**2))

def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))

def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

# ==============================================================================
# 2. Geometry & Coordinate Utilities
# ==============================================================================

def _validate_rect(rect: RectBounds) -> None:
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

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi

def angle_difference(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return float(diff)

def angle_between_vectors(v1: Point2D, v2: Point2D) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return abs(math.atan2(det, dot))

def polar_to_cartesian(r: float, theta: float) -> np.ndarray:
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)

def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
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
    return np.array([
        [math.cos(theta), 0.0],
        [math.sin(theta), 0.0],
        [0.0, 1.0]
    ], dtype=np.float64)

def state_jacobian(v: float, theta: float) -> np.ndarray:
    return np.array([
        [0.0, 0.0, -v * math.sin(theta)],
        [0.0, 0.0,  v * math.cos(theta)],
        [0.0, 0.0,  0.0]
    ], dtype=np.float64)

# ==============================================================================
# 4. Raycasting & LiDAR Collision Detection
# ==============================================================================

def signed_distance(point: Point2D, rect: RectBounds) -> float:
    _validate_rect(rect)
    dx = max(rect[0] - point[0], point[0] - rect[2])
    dy = max(rect[1] - point[1], point[1] - rect[3])
    if dx <= 0 and dy <= 0:
        return max(dx, dy)
    return math.hypot(max(dx, 0), max(dy, 0))

def ray_circle_intersection(ray_origin: Point2D, ray_dir: Point2D, center: Point2D, radius: float) -> float:
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

def compute_lie_derivatives(robot_pose: Pose3D, obs_state: Sequence[float]) -> Tuple[float, np.ndarray]:
    rx, ry, theta = robot_pose[0], robot_pose[1], robot_pose[2]
    ox, oy, obs_theta, obs_v = obs_state[0], obs_state[1], obs_state[2], obs_state[3]
    
    vox = obs_v * math.cos(obs_theta)
    voy = obs_v * math.sin(obs_theta)
    
    dh_dx = 2 * (rx - ox)
    dh_dy = 2 * (ry - oy)
    
    Lf_h = dh_dx * (-vox) + dh_dy * (-voy)
    Lg_h = np.array([dh_dx * math.cos(theta) + dh_dy * math.sin(theta), 0.0])
    
    return float(Lf_h), Lg_h

def compute_barrier_derivative(Lf_h: float, Lg_h: np.ndarray, control: Sequence[float]) -> float:
    return Lf_h + np.dot(Lg_h, control)

def control_barrier_constraint(Lf_h: float, Lg_h: np.ndarray, h: float, alpha: float = 1.0) -> Tuple[np.ndarray, float]:
    A = -np.asarray(Lg_h).reshape(1, -1)
    b = float(Lf_h + alpha * h)
    return A, b

# ==============================================================================
# 6. Dynamic Obstacles & Time-To-Collision
# ==============================================================================

def relative_velocity(v1: float, theta1: float, v2: float, theta2: float) -> np.ndarray:
    return np.array([
        v1 * math.cos(theta1) - v2 * math.cos(theta2),
        v1 * math.sin(theta1) - v2 * math.sin(theta2)
    ], dtype=np.float64)

def compute_ttc(robot_pos: Point2D, robot_vel_vec: Point2D, obs_pos: Point2D, obs_vel_vec: Point2D) -> float:
    dp = np.array([obs_pos[0] - robot_pos[0], obs_pos[1] - robot_pos[1]])
    dv = np.array([robot_vel_vec[0] - obs_vel_vec[0], robot_vel_vec[1] - obs_vel_vec[1]])
    
    v_sq = np.dot(dv, dv)
    if v_sq < 1e-8:
        return float('inf')
        
    ttc = np.dot(dp, dv) / v_sq
    return float(ttc) if ttc > 0 else float('inf')

def predict_trajectory(obstacle_state: Sequence[float], horizon: int, dt: float) -> np.ndarray:
    x, y, theta, v = obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[3]
    vx = v * math.cos(theta)
    vy = v * math.sin(theta)
    
    t_steps = np.arange(1, horizon + 1) * dt
    traj_x = x + vx * t_steps
    traj_y = y + vy * t_steps
    return np.column_stack((traj_x, traj_y))

def bounce_from_wall(obstacle_state: Sequence[float], bounds: RectBounds = (MAP_MIN_X, MAP_MAX_X, MAP_MIN_Y, MAP_MAX_Y)) -> np.ndarray:
    x, y, theta, v = obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[3]
    xmin, xmax, ymin, ymax = bounds
    
    if x <= xmin or x >= xmax:
        theta = normalize_angle(math.pi - theta)
    if y <= ymin or y >= ymax:
        theta = normalize_angle(-theta)
        
    return np.array([clamp(x, xmin, xmax), clamp(y, ymin, ymax), theta, v], dtype=np.float64)

def predict_obstacle_position(obstacle_state: Sequence[float], time_ahead: float) -> np.ndarray:
    x, y, theta, v = obstacle_state[0], obstacle_state[1], obstacle_state[2], obstacle_state[3]
    return np.array([x + v * math.cos(theta) * time_ahead, y + v * math.sin(theta) * time_ahead])


# ==============================================================================
# 7. Geometry Collisions & Environment Helpers
# ==============================================================================

def circle_circle_collision(p1: Point2D, r1: float, p2: Point2D, r2: float) -> bool:
    return squared_distance(p1, p2) <= (r1 + r2) ** 2

def point_inside_rectangle(p: Point2D, rect: RectBounds) -> bool:
    return rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]

def rectangle_distance(p: Point2D, rect: RectBounds) -> float:
    closest = closest_point_on_rectangle(p, rect)
    return distance(p, closest)

def circle_rectangle_collision(c_pos: Point2D, c_rad: float, rect: RectBounds) -> bool:
    closest = closest_point_on_rectangle(c_pos, rect)
    return squared_distance(c_pos, closest) <= (c_rad ** 2)

def inside_map(pos: Point2D, margin: float = 0.0) -> bool:
    return (MAP_MIN_X + margin <= pos[0] <= MAP_MAX_X - margin and
            MAP_MIN_Y + margin <= pos[1] <= MAP_MAX_Y - margin)

def outside_map(pos: Point2D, margin: float = 0.0) -> bool:
    return not inside_map(pos, margin)

def goal_reached(pos: Point2D, goal: Point2D, tolerance: float) -> bool:
    return distance(pos, goal) <= tolerance

def heading_to_goal(robot_pose: Pose3D, goal_pos: Point2D) -> float:
    target_heading = math.atan2(goal_pos[1] - robot_pose[1], goal_pos[0] - robot_pose[0])
    return angle_difference(target_heading, robot_pose[2])

def closest_obstacle(robot_pos: Point2D, obstacles: np.ndarray) -> np.ndarray:
    if len(obstacles) == 0:
        return np.array([])
    dists = [squared_distance(robot_pos, obs[:2]) for obs in obstacles]
    return obstacles[np.argmin(dists)]

# ==============================================================================
# 8. Random Sampling Utilities
# ==============================================================================

def minimum_distance_to_shelves(pos: Point2D, shelves: List[RectBounds]) -> float:
    if not shelves: 
        return float('inf')
    return min(rectangle_distance(pos, s) for s in shelves)

def random_robot_pose(rng, shelves: List[RectBounds], max_tries: int = 100) -> np.ndarray:
    for _ in range(max_tries):
        x = rng.uniform(MAP_MIN_X + 1.0, MAP_MAX_X - 1.0)
        y = rng.uniform(MAP_MIN_Y + 1.0, MAP_MAX_Y - 1.0)
        if minimum_distance_to_shelves((x, y), shelves) > ROBOT_RADIUS + 0.5:
            return np.array([x, y, rng.uniform(-math.pi, math.pi)], dtype=np.float64)
    raise RuntimeError("Failed to sample safe robot pose after maximum attempts.")

def random_goal_pose(rng, robot_pos: Point2D, shelves: List[RectBounds], max_tries: int = 100) -> np.ndarray:
    for _ in range(max_tries):
        x = rng.uniform(MAP_MIN_X + 1.0, MAP_MAX_X - 1.0)
        y = rng.uniform(MAP_MIN_Y + 1.0, MAP_MAX_Y - 1.0)
        if (distance((x, y), robot_pos) > 2.0 and 
            minimum_distance_to_shelves((x, y), shelves) > ROBOT_RADIUS + 0.5):
            return np.array([x, y], dtype=np.float64)
    raise RuntimeError("Failed to sample safe goal pose after maximum attempts.")

def random_dynamic_obstacle_positions(rng, num_obstacles: int, robot_pos: Point2D, goal_pos: Point2D, shelves: List[RectBounds], max_tries: int = 100) -> List[np.ndarray]:
    obs = []
    for _ in range(num_obstacles):
        for _ in range(max_tries):
            x = rng.uniform(MAP_MIN_X + 1.0, MAP_MAX_X - 1.0)
            y = rng.uniform(MAP_MIN_Y + 1.0, MAP_MAX_Y - 1.0)
            if (distance((x, y), robot_pos) > 2.0 and distance((x, y), goal_pos) > 1.0 and
                minimum_distance_to_shelves((x, y), shelves) > DYNAMIC_OBS_RADIUS + 0.2):
                speed = rng.uniform(DYNAMIC_OBS_SPEED_MIN, DYNAMIC_OBS_SPEED_MAX)
                theta = rng.uniform(-math.pi, math.pi)
                obs.append(np.array([x, y, theta, speed], dtype=np.float64))
                break
    if not obs:
        raise RuntimeError("Failed to sample dynamic obstacles.")
    return obs

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

# ==============================================================================
# 9. Reinforcement Learning Rewards & Normalization
# ==============================================================================

def compute_progress_reward(current_pos: Point2D, prev_pos: Point2D, goal_pos: Point2D, scale: float = 1.0) -> float:
    prev_dist = distance(prev_pos, goal_pos)
    curr_dist = distance(current_pos, goal_pos)
    return float((prev_dist - curr_dist) * scale)

def compute_safety_reward(closest_dist: float, safe_margin: float = 0.5, penalty: float = -10.0) -> float:
    if closest_dist < 0:
        return penalty 
    if closest_dist < safe_margin:
        return penalty * math.exp(-3.0 * (closest_dist / safe_margin))
    return 0.0

def normalize_observation(val: Union[float, np.ndarray], min_val: float, max_val: float) -> Union[float, np.ndarray]:
    """Min-Max scale observation features into [-1, 1] using safe_divide fix."""
    return 2.0 * safe_divide(val - min_val, max_val - min_val) - 1.0

# ==============================================================================
# 10. Batch & Evaluation Metrics
# ==============================================================================

def batch_distance(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    diff = points1[:, :2] - points2[:, :2]
    return np.sqrt(np.sum(diff * diff, axis=1))

def batch_collision(robot_positions: np.ndarray, obs_positions: np.ndarray, threshold: float) -> np.ndarray:
    diff = robot_positions[:, :2] - obs_positions[:, :2]
    sq_dist = np.sum(diff * diff, axis=1)
    return sq_dist <= (threshold * threshold)

def path_length(path_coords: np.ndarray) -> float:
    if len(path_coords) < 2: return 0.0
    diffs = np.diff(path_coords[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def control_effort(action_history: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(action_history, axis=1)))