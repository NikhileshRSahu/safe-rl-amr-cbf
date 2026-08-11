"""Classical baseline algorithms for AMR warehouse navigation.

Each baseline implements a common ``select_action(obs, info) -> np.ndarray`` interface
so they can be dropped into the same ``AMRWarehouseEnv`` evaluation loop as the RL agent.

Implemented baselines:
    - PIDController: Proportional navigation to goal with obstacle braking.
    - PotentialField: Attractive goal field + repulsive obstacle fields (suffers from local minima).
    - AStarPlanner: Grid A* with reactive path following (slow replanning, struggles with dynamics).
    - DWA: Dynamic Window Approach local planner (good but sensitive to state noise).
    - RRTPlanner: Rapidly-exploring Random Tree global planner (struggles with dynamic obstacles).
    - MPCBaseline: Simple Model Predictive Control with short horizon.
    - PureCBF: CBF filter with only a simple goal-seeking nominal controller (reactive).
    - PureRL: RL policy without CBF safety filter (for ablation).

All baselines operate on the *same* noisy, delayed observations that the RL policy sees,
so the comparison is fair.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    DT,
    DYNAMIC_OBS_RADIUS,
    GOAL_TOLERANCE,
    MAP_MAX_X,
    MAP_MAX_Y,
    MAP_MIN_X,
    MAP_MIN_Y,
    OMEGA_MAX,
    OMEGA_MIN,
    ROBOT_RADIUS,
    SHELVES,
    V_MAX,
    V_MIN,
)
from utils import (
    closest_point_on_rectangle,
    distance,
    heading_to_goal,
    normalize_angle,
    propagate_unicycle,
    rectangle_distance,
)


# =============================================================================
# Base class
# =============================================================================

class BaseBaseline:
    """Common interface for all baseline controllers."""

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        """Return a normalized action in [-1, 1]^2 from the observation dict.

        Args:
            obs: Observation dict from ``AMRWarehouseEnv``.
            info: Info dict from ``AMRWarehouseEnv``.

        Returns:
            Normalized action ``[a_v, a_omega]`` in ``[-1, 1]``.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of each episode."""
        pass


# =============================================================================
# Helper: denormalize observations back to physical units
# =============================================================================

def _denorm_robot_state(robot_obs: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Map normalized robot_state obs back to physical units."""
    rx = float(np.clip(robot_obs[0], -1, 1)) * 0.5 + 0.5  # approx back to MAP_MIN_X..MAP_MAX_X
    # Better: we know the normalization in env.py
    # robot_obs = [rx_norm, ry_norm, theta/pi, v_norm, omega/OMEGA_MAX]
    rx = float(robot_obs[0]) * (MAP_MAX_X - MAP_MIN_X) * 0.5 + (MAP_MAX_X + MAP_MIN_X) * 0.5
    ry = float(robot_obs[1]) * (MAP_MAX_Y - MAP_MIN_Y) * 0.5 + (MAP_MAX_Y + MAP_MIN_Y) * 0.5
    theta = float(robot_obs[2]) * math.pi
    v = float(robot_obs[3]) * (V_MAX - V_MIN) * 0.5 + (V_MAX + V_MIN) * 0.5
    omega = float(robot_obs[4]) * OMEGA_MAX
    return rx, ry, theta, v, omega


def _denorm_goal(goal_obs: np.ndarray, robot_pose: Tuple[float, float, float]) -> Tuple[float, float]:
    """Map normalized goal obs back to world coordinates.

    goal_obs = [rel_x/max_diag, rel_y/max_diag, dist/max_diag, heading_error/pi]
    We use the relative robot-frame coords and rotate back to world frame.
    """
    max_diag = math.hypot(MAP_MAX_X - MAP_MIN_X, MAP_MAX_Y - MAP_MIN_Y)
    rel_x = float(goal_obs[0]) * max_diag
    rel_y = float(goal_obs[1]) * max_diag
    rx, ry, theta = robot_pose
    # robot_to_world: [rx + cos_t*rel_x - sin_t*rel_y, ry + sin_t*rel_x + cos_t*rel_y]
    gx = rx + math.cos(theta) * rel_x - math.sin(theta) * rel_y
    gy = ry + math.sin(theta) * rel_x + math.cos(theta) * rel_y
    return gx, gy


def _extract_state(obs: Dict[str, np.ndarray]) -> Tuple[Tuple[float, float, float, float, float], Tuple[float, float]]:
    """Extract physical robot state and goal from observation dict."""
    robot_obs = obs["robot_state"]
    goal_obs = obs["goal"]
    pose = _denorm_robot_state(robot_obs)
    goal = _denorm_goal(goal_obs, pose[:3])
    return pose, goal


def _extract_lidar(obs: Dict[str, np.ndarray], max_range: float = 10.0) -> np.ndarray:
    """Extract LiDAR scan from observation and denormalize to physical distances."""
    lidar_norm = obs["lidar"]
    # Normalized: 2*(scan/max_range) - 1  -> scan = (lidar_norm + 1) * max_range / 2
    scan = (lidar_norm + 1.0) * max_range / 2.0
    return np.clip(scan, 0.0, max_range)


def _norm_action(v: float, omega: float) -> np.ndarray:
    """Map physical (v, omega) to normalized [-1, 1] action."""
    a_v = 2.0 * (v - V_MIN) / max(V_MAX - V_MIN, 1e-8) - 1.0
    a_omega = omega / OMEGA_MAX
    return np.array([np.clip(a_v, -1.0, 1.0), np.clip(a_omega, -1.0, 1.0)], dtype=np.float32)


# =============================================================================
# 1. PID Controller (simple go-to-goal with obstacle braking)
# =============================================================================

class PIDController(BaseBaseline):
    """Simple proportional controller that drives toward goal while braking near obstacles.

    Uses LiDAR to detect frontal obstacles and reduces speed proportionally.
    Struggles with: local minima, dynamic obstacles, narrow passages.
    """

    def __init__(
        self,
        kv: float = 0.8,
        komega: float = 1.5,
        safety_margin: float = 1.0,
        braking_gain: float = 2.0,
    ) -> None:
        self.kv = kv
        self.komega = komega
        self.safety_margin = safety_margin
        self.braking_gain = braking_gain

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        pose, goal = _extract_state(obs)
        rx, ry, theta, v, omega = pose
        gx, gy = goal

        heading_err = heading_to_goal((rx, ry, theta), (gx, gy))
        dist_to_goal = distance((rx, ry), (gx, gy))

        # Base control
        v_cmd = min(self.kv * dist_to_goal, V_MAX)
        omega_cmd = np.clip(self.komega * heading_err, OMEGA_MIN, OMEGA_MAX)

        # Obstacle braking from LiDAR
        lidar = _extract_lidar(obs)
        num_rays = len(lidar)
        # Front sector: -30 to +30 degrees
        front_indices = list(range(num_rays - 6, num_rays)) + list(range(0, 7))
        front_dists = [lidar[i % num_rays] for i in front_indices]
        min_front = min(front_dists)

        if min_front < self.safety_margin:
            brake = math.exp(-self.braking_gain * (self.safety_margin - min_front))
            v_cmd *= brake
            # Turn away from closest obstacle in front half
            left_min = min(lidar[num_rays // 2 - 8 : num_rays // 2 + 1]) if num_rays > 16 else min_front
            right_min = min(lidar[0:9]) if num_rays > 16 else min_front
            if left_min < right_min:
                omega_cmd = -abs(omega_cmd) - 0.5  # turn right
            else:
                omega_cmd = abs(omega_cmd) + 0.5   # turn left

        return _norm_action(v_cmd, omega_cmd)


# =============================================================================
# 2. Artificial Potential Field
# =============================================================================

class PotentialField(BaseBaseline):
    """Artificial Potential Field: attractive goal + repulsive obstacles.

    Classic approach that often gets stuck in local minima, oscillates in
    narrow corridors, and cannot handle dynamic obstacles well.
    """

    def __init__(
        self,
        att_gain: float = 1.0,
        rep_gain: float = 3.0,
        rep_range: float = 3.0,
        max_speed: float = V_MAX,
    ) -> None:
        self.att_gain = att_gain
        self.rep_gain = rep_gain
        self.rep_range = rep_range
        self.max_speed = max_speed

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        pose, goal = _extract_state(obs)
        rx, ry, theta, v, omega = pose
        gx, gy = goal

        # Attractive force toward goal
        dx = gx - rx
        dy = gy - ry
        dist_g = math.hypot(dx, dy)
        if dist_g < 0.1:
            return _norm_action(0.0, 0.0)

        f_att_x = self.att_gain * dx / dist_g
        f_att_y = self.att_gain * dy / dist_g

        # Repulsive forces from LiDAR
        lidar = _extract_lidar(obs)
        num_rays = len(lidar)
        angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)

        f_rep_x, f_rep_y = 0.0, 0.0
        for i, d in enumerate(lidar):
            if d >= self.rep_range or d <= 0.05:
                continue
            # Repulsive magnitude
            rep = self.rep_gain * (1.0 / d - 1.0 / self.rep_range) / (d * d)
            # Direction: away from obstacle = opposite of ray direction in world frame
            obs_angle_world = normalize_angle(theta + angles[i] + math.pi)
            f_rep_x += rep * math.cos(obs_angle_world)
            f_rep_y += rep * math.sin(obs_angle_world)

        # Repulsive from static shelves (using known map)
        for rect in SHELVES:
            cp = closest_point_on_rectangle((rx, ry), rect)
            d_shelf = distance((rx, ry), cp)
            if d_shelf < self.rep_range and d_shelf > 0.05:
                rep = self.rep_gain * (1.0 / d_shelf - 1.0 / self.rep_range) / (d_shelf * d_shelf)
                f_rep_x += rep * (rx - cp[0]) / d_shelf
                f_rep_y += rep * (ry - cp[1]) / d_shelf

        f_x = f_att_x + f_rep_x
        f_y = f_att_y + f_rep_y

        # Convert force to velocity commands
        desired_heading = math.atan2(f_y, f_x)
        heading_err = normalize_angle(desired_heading - theta)
        force_mag = math.hypot(f_x, f_y)

        v_cmd = min(force_mag, self.max_speed)
        omega_cmd = np.clip(2.0 * heading_err, OMEGA_MIN, OMEGA_MAX)

        return _norm_action(v_cmd, omega_cmd)


# =============================================================================
# 3. A* Planner
# =============================================================================

class AStarPlanner(BaseBaseline):
    """Grid-based A* with reactive pure-pursuit path following.

    Plans on a coarse grid from a known map. Replanning is slow and expensive,
    so it only replans every N steps. Dynamic obstacles are NOT in the map,
    so the planner is blind to them. Uses LiDAR for emergency braking only.
    """

    def __init__(
        self,
        grid_res: float = 0.5,
        replan_interval: int = 20,
        lookahead: float = 1.5,
    ) -> None:
        self.grid_res = grid_res
        self.replan_interval = replan_interval
        self.lookahead = lookahead
        self.path: List[Tuple[int, int]] = []
        self.step_count = 0
        self.last_goal: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.path = []
        self.step_count = 0
        self.last_goal = None

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - MAP_MIN_X) / self.grid_res)
        gy = int((y - MAP_MIN_Y) / self.grid_res)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = MAP_MIN_X + (gx + 0.5) * self.grid_res
        y = MAP_MIN_Y + (gy + 0.5) * self.grid_res
        return x, y

    def _is_free(self, gx: int, gy: int) -> bool:
        x, y = self._grid_to_world(gx, gy)
        if not (MAP_MIN_X + ROBOT_RADIUS <= x <= MAP_MAX_X - ROBOT_RADIUS):
            return False
        if not (MAP_MIN_Y + ROBOT_RADIUS <= y <= MAP_MAX_Y - ROBOT_RADIUS):
            return False
        for rect in SHELVES:
            if rectangle_distance((x, y), rect) < ROBOT_RADIUS + 0.1:
                return False
        return True

    def _astar(self, start_g: Tuple[int, int], goal_g: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Simple A* on grid."""
        import heapq

        if start_g == goal_g:
            return [start_g]

        nx = int((MAP_MAX_X - MAP_MIN_X) / self.grid_res) + 1
        ny = int((MAP_MAX_Y - MAP_MIN_Y) / self.grid_res) + 1

        open_set = [(0.0, 0, start_g)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_g: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start_g: distance(start_g, goal_g)}
        counter = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal_g:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny):
                    continue
                if not self._is_free(neighbor[0], neighbor[1]):
                    continue

                move_cost = math.hypot(dx, dy) * self.grid_res
                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + distance(neighbor, goal_g) * self.grid_res
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1

        return []  # No path found

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        pose, goal = _extract_state(obs)
        rx, ry, theta, v, omega = pose
        gx, gy = goal

        self.step_count += 1

        # Replan if goal changed or interval reached or path empty
        need_replan = (
            self.last_goal is None or
            distance((gx, gy), self.last_goal) > 0.5 or
            self.step_count % self.replan_interval == 0 or
            not self.path
        )

        if need_replan:
            start_g = self._world_to_grid(rx, ry)
            goal_g = self._world_to_grid(gx, gy)
            self.path = self._astar(start_g, goal_g)
            self.last_goal = (gx, gy)

        if not self.path:
            # No path - emergency stop
            return _norm_action(0.0, 0.0)

        # Pure pursuit: find waypoint ahead on path
        # Find closest point on path
        best_idx = 0
        best_dist = float("inf")
        for i, pg in enumerate(self.path):
            wx, wy = self._grid_to_world(pg[0], pg[1])
            d = distance((rx, ry), (wx, wy))
            if d < best_dist:
                best_dist = d
                best_idx = i

        # Lookahead
        target_idx = min(best_idx + max(1, int(self.lookahead / self.grid_res)), len(self.path) - 1)
        tx, ty = self._grid_to_world(self.path[target_idx][0], self.path[target_idx][1])

        heading_err = heading_to_goal((rx, ry, theta), (tx, ty))
        dist_to_target = distance((rx, ry), (tx, ty))

        v_cmd = min(0.8 * dist_to_target, V_MAX * 0.8)
        omega_cmd = np.clip(2.0 * heading_err, OMEGA_MIN, OMEGA_MAX)

        # Emergency braking from dynamic obstacles (A* doesn't see them!)
        lidar = _extract_lidar(obs)
        if len(lidar) > 8:
            front_sector = np.concatenate([lidar[-4:], lidar[:5]])
            front_min = float(np.min(front_sector))
        else:
            front_min = float(np.min(lidar))
        if front_min < 1.0:
            v_cmd *= 0.3

        return _norm_action(v_cmd, omega_cmd)


# =============================================================================
# 4. Dynamic Window Approach (DWA)
# =============================================================================

class DWA(BaseBaseline):
    """Dynamic Window Approach - samples velocity commands and scores them.

    Good local planner but needs accurate state estimation. With noisy/delayed
    observations and dynamic obstacles, performance degrades significantly.
    """

    def __init__(
        self,
        v_samples: int = 15,
        omega_samples: int = 21,
        pred_horizon: float = 2.0,
        goal_weight: float = 0.8,
        speed_weight: float = 0.2,
        obstacle_weight: float = 2.0,
    ) -> None:
        self.v_samples = v_samples
        self.omega_samples = omega_samples
        self.pred_horizon = pred_horizon
        self.goal_weight = goal_weight
        self.speed_weight = speed_weight
        self.obstacle_weight = obstacle_weight

    def _simulate(self, pose: Tuple[float, float, float], v: float, omega: float, dt: float, steps: int) -> List[Tuple[float, float]]:
        traj = [(pose[0], pose[1])]
        x, y, theta = pose
        for _ in range(steps):
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta = normalize_angle(theta + omega * dt)
            traj.append((x, y))
        return traj

    def _score_trajectory(
        self,
        traj: List[Tuple[float, float]],
        goal: Tuple[float, float],
        obs: Dict[str, np.ndarray],
    ) -> float:
        # Goal heading score
        final_x, final_y = traj[-1]
        dist_to_goal = distance((final_x, final_y), goal)
        goal_score = -dist_to_goal

        # Speed score (prefer higher forward speed)
        # We don't have velocity in traj, assume constant
        speed_score = 0.0  # set externally

        # Obstacle clearance score from LiDAR at final position
        # Approximate: use current LiDAR (delayed/noisy!)
        lidar = _extract_lidar(obs)
        min_lidar = float(np.min(lidar))
        obstacle_score = min_lidar if min_lidar > ROBOT_RADIUS else -10.0

        return self.goal_weight * goal_score + self.obstacle_weight * obstacle_score

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        pose, goal = _extract_state(obs)
        rx, ry, theta, v_cur, omega_cur = pose

        # Dynamic window
        v_min_win = max(V_MIN, v_cur - 1.5 * DT)
        v_max_win = min(V_MAX, v_cur + 1.5 * DT)
        omega_min_win = max(OMEGA_MIN, omega_cur - 2.5 * DT)
        omega_max_win = min(OMEGA_MAX, omega_cur + 2.5 * DT)

        best_score = -float("inf")
        best_v, best_omega = 0.0, 0.0

        num_steps = max(1, int(self.pred_horizon / DT))

        for v in np.linspace(v_min_win, v_max_win, self.v_samples):
            for omega in np.linspace(omega_min_win, omega_max_win, self.omega_samples):
                traj = self._simulate((rx, ry, theta), v, omega, DT, num_steps)
                score = self._score_trajectory(traj, goal, obs)
                # Add speed bonus
                score += self.speed_weight * abs(v)

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_omega = omega

        return _norm_action(best_v, best_omega)


# =============================================================================
# 5. RRT Planner
# =============================================================================

class RRTPlanner(BaseBaseline):
    """Rapidly-exploring Random Tree with reactive path following.

    Plans a global path through static obstacles but does NOT replan for dynamic
    obstacles. Slow replanning makes it vulnerable to moving obstacles.
    """

    def __init__(
        self,
        step_size: float = 0.8,
        max_iter: int = 300,
        replan_interval: int = 50,
    ) -> None:
        self.step_size = step_size
        self.max_iter = max_iter
        self.replan_interval = replan_interval
        self.path: List[Tuple[float, float]] = []
        self.step_count = 0
        self.last_goal: Optional[Tuple[float, float]] = None
        self.rng = np.random.default_rng(42)

    def reset(self) -> None:
        self.path = []
        self.step_count = 0
        self.last_goal = None

    def _collision_free(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        # Simple check: sample points along line segment
        dist = distance(p1, p2)
        if dist < 1e-6:
            return True
        n_checks = max(1, int(dist / 0.1))
        for i in range(n_checks + 1):
            t = i / n_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not (MAP_MIN_X + ROBOT_RADIUS <= x <= MAP_MAX_X - ROBOT_RADIUS):
                return False
            if not (MAP_MIN_Y + ROBOT_RADIUS <= y <= MAP_MAX_Y - ROBOT_RADIUS):
                return False
            for rect in SHELVES:
                if rectangle_distance((x, y), rect) < ROBOT_RADIUS + 0.05:
                    return False
        return True

    def _rrt(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        nodes: List[Tuple[float, float]] = [start]
        parents: Dict[int, int] = {}

        for _ in range(self.max_iter):
            # Sample random point (with goal bias)
            if self.rng.random() < 0.15:
                sample = goal
            else:
                sample = (
                    self.rng.uniform(MAP_MIN_X + ROBOT_RADIUS, MAP_MAX_X - ROBOT_RADIUS),
                    self.rng.uniform(MAP_MIN_Y + ROBOT_RADIUS, MAP_MAX_Y - ROBOT_RADIUS),
                )

            # Find nearest node
            nearest_idx = min(range(len(nodes)), key=lambda i: distance(nodes[i], sample))
            nearest = nodes[nearest_idx]

            # Steer toward sample
            dist = distance(nearest, sample)
            if dist < 1e-6:
                continue
            t = min(self.step_size / dist, 1.0)
            new_node = (nearest[0] + t * (sample[0] - nearest[0]),
                        nearest[1] + t * (sample[1] - nearest[1]))

            if self._collision_free(nearest, new_node):
                new_idx = len(nodes)
                nodes.append(new_node)
                parents[new_idx] = nearest_idx

                if distance(new_node, goal) < self.step_size:
                    # Reconstruct path
                    path = [goal]
                    idx = new_idx
                    while idx in parents:
                        path.append(nodes[idx])
                        idx = parents[idx]
                    path.append(start)
                    path.reverse()
                    return path

        # Return path to closest node to goal
        best_idx = min(range(len(nodes)), key=lambda i: distance(nodes[i], goal))
        path = []
        idx = best_idx
        while idx in parents:
            path.append(nodes[idx])
            idx = parents[idx]
        path.append(start)
        path.reverse()
        return path

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        pose, goal = _extract_state(obs)
        rx, ry, theta, v, omega = pose
        gx, gy = goal

        self.step_count += 1

        need_replan = (
            self.last_goal is None or
            distance((gx, gy), self.last_goal) > 0.5 or
            self.step_count % self.replan_interval == 0 or
            not self.path
        )

        if need_replan:
            self.path = self._rrt((rx, ry), (gx, gy))
            self.last_goal = (gx, gy)

        if not self.path:
            return _norm_action(0.0, 0.0)

        # Pure pursuit
        best_idx = 0
        best_dist = float("inf")
        for i, (wx, wy) in enumerate(self.path):
            d = distance((rx, ry), (wx, wy))
            if d < best_dist:
                best_dist = d
                best_idx = i

        target_idx = min(best_idx + 3, len(self.path) - 1)
        tx, ty = self.path[target_idx]

        heading_err = heading_to_goal((rx, ry, theta), (tx, ty))
        dist_to_target = distance((rx, ry), (tx, ty))

        v_cmd = min(0.7 * dist_to_target, V_MAX * 0.7)
        omega_cmd = np.clip(2.5 * heading_err, OMEGA_MIN, OMEGA_MAX)

        # Dynamic obstacle braking (RRT is blind to them)
        lidar = _extract_lidar(obs)
        front_min = min(list(lidar[-3:]) + list(lidar[:4])) if len(lidar) > 6 else float(np.min(lidar))
        if front_min < 0.8:
            v_cmd *= 0.2

        return _norm_action(v_cmd, omega_cmd)


# =============================================================================
# 6. MPC Baseline
# =============================================================================

class MPCBaseline(BaseBaseline):
    """Simple Model Predictive Control with short horizon.

    Optimizes a trajectory over a receding horizon. With a short horizon and
    simple dynamics, it handles static obstacles but struggles with dynamic
    ones and noisy state estimates.
    """

    def __init__(
        self,
        horizon: int = 8,
        dt: float = DT,
        v_samples: int = 7,
        omega_samples: int = 11,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.v_samples = v_samples
        self.omega_samples = omega_samples

    def _simulate_open_loop(
        self,
        pose: Tuple[float, float, float],
        v_seq: List[float],
        omega_seq: List[float],
    ) -> List[Tuple[float, float]]:
        traj = [(pose[0], pose[1])]
        x, y, theta = pose
        for v, omega in zip(v_seq, omega_seq):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta = normalize_angle(theta + omega * self.dt)
            traj.append((x, y))
        return traj

    def _cost(
        self,
        traj: List[Tuple[float, float]],
        goal: Tuple[float, float],
        v_seq: List[float],
        omega_seq: List[float],
        obs: Dict[str, np.ndarray],
    ) -> float:
        # Terminal cost: distance to goal
        cost = 2.0 * distance(traj[-1], goal)

        # Running costs
        for i, (x, y) in enumerate(traj[:-1]):
            cost += 0.1 * distance((x, y), goal)
            # Control effort
            cost += 0.05 * (v_seq[i]**2 + omega_seq[i]**2)
            # Obstacle avoidance from LiDAR (only current position, approximated)
            for rect in SHELVES:
                cp = closest_point_on_rectangle((x, y), rect)
                d = distance((x, y), cp)
                if d < 1.0:
                    cost += 5.0 * max(0, 1.0 - d)**2

        # LiDAR penalty at current position
        lidar = _extract_lidar(obs)
        min_dist = float(np.min(lidar))
        if min_dist < 1.5:
            cost += 3.0 * (1.5 - min_dist)**2

        return cost

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        pose, goal = _extract_state(obs)
        rx, ry, theta, v_cur, omega_cur = pose

        best_cost = float("inf")
        best_v, best_omega = 0.0, 0.0

        v_options = np.linspace(max(V_MIN, v_cur - 0.5), min(V_MAX, v_cur + 0.5), self.v_samples)
        omega_options = np.linspace(max(OMEGA_MIN, omega_cur - 1.0), min(OMEGA_MAX, omega_cur + 1.0), self.omega_samples)

        for v in v_options:
            for omega in omega_options:
                v_seq = [v] * self.horizon
                omega_seq = [omega] * self.horizon
                traj = self._simulate_open_loop((rx, ry, theta), v_seq, omega_seq)
                c = self._cost(traj, goal, v_seq, omega_seq, obs)
                if c < best_cost:
                    best_cost = c
                    best_v = v
                    best_omega = omega
            for omega in omega_seq:
                v_seq = [v] * self.horizon
                omega_seq = [omega] * self.horizon
                traj = self._simulate_open_loop((rx, ry, theta), v_seq, omega_seq)
                c = self._cost(traj, goal, v_seq, omega_seq, obs)
                if c < best_cost:
                    best_cost = c
                    best_v = v
                    best_omega = omega

        return _norm_action(best_v, best_omega)


# =============================================================================
# 7. Pure CBF (reactive safety filter only)
# =============================================================================

class PureCBF(BaseBaseline):
    """CBF safety filter with a very simple nominal controller (go-to-goal).

    The nominal controller is just a proportional goal seeker. The CBF overrides
    it when near obstacles. Without learning, the nominal controller has no notion
    of obstacle avoidance, so the CBF is constantly intervening - leading to
    conservative/slow behavior and potential deadlock.
    """

    def __init__(self, kv: float = 0.6, komega: float = 1.2) -> None:
        self.kv = kv
        self.komega = komega

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        pose, goal = _extract_state(obs)
        rx, ry, theta, v, omega = pose
        gx, gy = goal

        dist_to_goal = distance((rx, ry), (gx, gy))
        heading_err = heading_to_goal((rx, ry, theta), (gx, gy))

        v_cmd = min(self.kv * dist_to_goal, V_MAX)
        omega_cmd = np.clip(self.komega * heading_err, OMEGA_MIN, OMEGA_MAX)

        return _norm_action(v_cmd, omega_cmd)


# =============================================================================
# 8. Pure RL wrapper (policy without CBF)
# =============================================================================

class PureRLWrapper(BaseBaseline):
    """Wraps a trained RL policy but runs WITHOUT the CBF safety filter.

    Used for ablation study. Expect higher collision rates.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        return self.agent.select_action(obs, deterministic=True)


# =============================================================================
# Registry
# =============================================================================

BASELINE_REGISTRY = {
    "pid": PIDController,
    "potential_field": PotentialField,
    "astar": AStarPlanner,
    "dwa": DWA,
    "rrt": RRTPlanner,
    "mpc": MPCBaseline,
    "cbf_only": PureCBF,
}


def make_baseline(name: str, **kwargs: Any) -> BaseBaseline:
    """Factory for baseline controllers."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](**kwargs)
