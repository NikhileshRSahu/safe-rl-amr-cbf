"""Classical A* + Dynamic Window Approach baseline for AMR navigation."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _point_rect_distance(x: float, y: float, rect: Rect) -> float:
    xmin, ymin, xmax, ymax = rect
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def physical_to_normalized_action(
    v: float,
    omega: float,
    *,
    v_min: float,
    v_max: float,
    omega_max: float,
) -> np.ndarray:
    if v_max <= v_min:
        raise ValueError("v_max must be greater than v_min")
    if omega_max <= 0.0:
        raise ValueError("omega_max must be positive")
    a_v = 2.0 * (float(v) - v_min) / (v_max - v_min) - 1.0
    a_w = float(omega) / omega_max
    return np.clip(np.array([a_v, a_w], dtype=np.float32), -1.0, 1.0)


class AStarPlanner:
    """8-connected grid A* over inflated static warehouse geometry."""

    _MOVES = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(
        self,
        *,
        map_bounds: Tuple[float, float, float, float],
        shelves: Sequence[Rect],
        robot_radius: float,
        margin: float = 0.10,
        resolution: float = 0.25,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.xmin, self.xmax, self.ymin, self.ymax = map_bounds
        self.shelves = [tuple(map(float, r)) for r in shelves]
        self.inflation = float(robot_radius + margin)
        self.resolution = float(resolution)
        self.nx = int(round((self.xmax - self.xmin) / self.resolution)) + 1
        self.ny = int(round((self.ymax - self.ymin) / self.resolution)) + 1

    def world_to_grid(self, point: Point) -> Tuple[int, int]:
        x, y = point
        return (
            int(round((x - self.xmin) / self.resolution)),
            int(round((y - self.ymin) / self.resolution)),
        )

    def grid_to_world(self, node: Tuple[int, int]) -> Point:
        ix, iy = node
        return (
            self.xmin + ix * self.resolution,
            self.ymin + iy * self.resolution,
        )

    def in_grid(self, node: Tuple[int, int]) -> bool:
        ix, iy = node
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def is_free_world(self, point: Point) -> bool:
        x, y = map(float, point)
        margin = self.inflation
        if (
            x < self.xmin + margin
            or x > self.xmax - margin
            or y < self.ymin + margin
            or y > self.ymax - margin
        ):
            return False
        for xmin, ymin, xmax, ymax in self.shelves:
            if (
                xmin - margin <= x <= xmax + margin
                and ymin - margin <= y <= ymax + margin
            ):
                return False
        return True

    def _free(self, node: Tuple[int, int]) -> bool:
        return self.in_grid(node) and self.is_free_world(self.grid_to_world(node))

    def _nearest_free(
        self, node: Tuple[int, int], radius_cells: int = 8
    ) -> Tuple[int, int] | None:
        if self._free(node):
            return node
        ix, iy = node
        for radius in range(1, radius_cells + 1):
            candidates = []
            for dx in range(-radius, radius + 1):
                candidates.append((ix + dx, iy - radius))
                candidates.append((ix + dx, iy + radius))
            for dy in range(-radius + 1, radius):
                candidates.append((ix - radius, iy + dy))
                candidates.append((ix + radius, iy + dy))
            free = [candidate for candidate in candidates if self._free(candidate)]
            if free:
                return min(
                    free,
                    key=lambda candidate: (
                        (candidate[0] - ix) ** 2 + (candidate[1] - iy) ** 2
                    ),
                )
        return None

    def plan(self, start: Point, goal: Point) -> List[Point]:
        start_node = self._nearest_free(self.world_to_grid(start))
        goal_node = self._nearest_free(self.world_to_grid(goal))
        if start_node is None or goal_node is None:
            return []

        def heuristic(node: Tuple[int, int]) -> float:
            return math.hypot(node[0] - goal_node[0], node[1] - goal_node[1])

        queue = [(heuristic(start_node), 0.0, start_node)]
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        cost_so_far: dict[Tuple[int, int], float] = {start_node: 0.0}
        closed: set[Tuple[int, int]] = set()

        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current in closed:
                continue
            if current == goal_node:
                nodes = [current]
                while nodes[-1] in came_from:
                    nodes.append(came_from[nodes[-1]])
                nodes.reverse()
                return [self.grid_to_world(node) for node in nodes]
            closed.add(current)

            for dx, dy, move_cost in self._MOVES:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._free(nxt) or nxt in closed:
                    continue
                if dx and dy:
                    if (
                        not self._free((current[0] + dx, current[1]))
                        or not self._free((current[0], current[1] + dy))
                    ):
                        continue
                candidate_cost = current_cost + move_cost
                if candidate_cost < cost_so_far.get(nxt, float("inf")):
                    cost_so_far[nxt] = candidate_cost
                    came_from[nxt] = current
                    heapq.heappush(
                        queue,
                        (
                            candidate_cost + heuristic(nxt),
                            candidate_cost,
                            nxt,
                        ),
                    )
        return []


@dataclass(frozen=True)
class DWAConfig:
    v_samples: int = 7
    omega_samples: int = 21
    horizon: float = 1.5
    safety_margin: float = 0.10
    waypoint_tolerance: float = 0.55
    lookahead_distance: float = 1.5
    progress_weight: float = 3.0
    heading_weight: float = 1.2
    clearance_weight: float = 1.5
    speed_weight: float = 0.35
    smoothness_weight: float = 0.20
    replan_interval: int = 25


class AStarDWAController:
    """A* global path with a DWA local velocity controller."""

    def __init__(
        self,
        *,
        map_bounds: Tuple[float, float, float, float],
        shelves: Sequence[Rect],
        robot_radius: float,
        dynamic_obstacle_radius: float,
        v_min: float,
        v_max: float,
        omega_min: float,
        omega_max: float,
        accel_max: float,
        alpha_max: float,
        dt: float,
        config: DWAConfig | None = None,
        grid_resolution: float = 0.25,
        planning_margin: float = 0.10,
    ) -> None:
        self.map_bounds = tuple(map(float, map_bounds))
        self.shelves = [tuple(map(float, r)) for r in shelves]
        self.robot_radius = float(robot_radius)
        self.dynamic_obstacle_radius = float(dynamic_obstacle_radius)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.accel_max = float(accel_max)
        self.alpha_max = float(alpha_max)
        self.dt = float(dt)
        self.config = config or DWAConfig()
        self.planner = AStarPlanner(
            map_bounds=self.map_bounds,
            shelves=self.shelves,
            robot_radius=self.robot_radius,
            margin=planning_margin,
            resolution=grid_resolution,
        )
        self.path: List[Point] = []
        self.goal: Point | None = None
        self.path_index = 0
        self.prev_physical = np.array([0.0, 0.0], dtype=float)
        self.command_count = 0

    def reset(self, start_xy: Point, goal_xy: Point) -> bool:
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        self.path = self.planner.plan(
            (float(start_xy[0]), float(start_xy[1])),
            self.goal,
        )
        self.path_index = 0
        self.prev_physical[:] = 0.0
        self.command_count = 0
        return bool(self.path)

    def plan(self, start_xy: Point, goal_xy: Point) -> List[Point]:
        return self.planner.plan(start_xy, goal_xy)

    def normalized_to_physical(self, action: np.ndarray) -> Tuple[float, float]:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        v = self.v_min + 0.5 * (action[0] + 1.0) * (self.v_max - self.v_min)
        omega = action[1] * self.omega_max
        return float(v), float(omega)

    def _target_waypoint(self, robot_xy: np.ndarray) -> Point:
        if not self.path:
            return (
                self.goal
                if self.goal is not None
                else (float(robot_xy[0]), float(robot_xy[1]))
            )

        while self.path_index < len(self.path) - 1:
            point = np.asarray(self.path[self.path_index], dtype=float)
            if np.linalg.norm(point - robot_xy) > self.config.waypoint_tolerance:
                break
            self.path_index += 1

        index = self.path_index
        accumulated = 0.0
        last = np.asarray(self.path[index], dtype=float)
        while (
            index + 1 < len(self.path)
            and accumulated < self.config.lookahead_distance
        ):
            nxt = np.asarray(self.path[index + 1], dtype=float)
            accumulated += float(np.linalg.norm(nxt - last))
            index += 1
            last = nxt
        return self.path[index]

    def _pose_clearance(
        self, x: float, y: float, dynamic_xy: np.ndarray
    ) -> float:
        xmin, xmax, ymin, ymax = self.map_bounds
        wall_clearance = (
            min(x - xmin, xmax - x, y - ymin, ymax - y)
            - self.robot_radius
        )
        shelf_clearance = min(
            (
                _point_rect_distance(x, y, rect) - self.robot_radius
                for rect in self.shelves
            ),
            default=float("inf"),
        )
        dynamic_clearance = min(
            (
                math.hypot(x - ox, y - oy)
                - self.robot_radius
                - self.dynamic_obstacle_radius
                for ox, oy in dynamic_xy
            ),
            default=float("inf"),
        )
        return min(wall_clearance, shelf_clearance, dynamic_clearance)

    def _simulate(
        self,
        robot_state: np.ndarray,
        v: float,
        omega: float,
        dynamic_obstacles: np.ndarray,
    ) -> Tuple[float, float, float, float, bool]:
        x, y, theta = map(float, robot_state[:3])
        min_clearance = float("inf")
        steps = max(1, int(math.ceil(self.config.horizon / self.dt)))
        dynamic_obstacles = np.asarray(dynamic_obstacles, dtype=float)

        for step in range(1, steps + 1):
            theta = _angle_wrap(theta + omega * self.dt)
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            t = step * self.dt

            if dynamic_obstacles.size:
                dyn_x = (
                    dynamic_obstacles[:, 0]
                    + dynamic_obstacles[:, 3]
                    * np.cos(dynamic_obstacles[:, 2])
                    * t
                )
                dyn_y = (
                    dynamic_obstacles[:, 1]
                    + dynamic_obstacles[:, 3]
                    * np.sin(dynamic_obstacles[:, 2])
                    * t
                )
                dynamic_xy = np.column_stack((dyn_x, dyn_y))
            else:
                dynamic_xy = np.empty((0, 2), dtype=float)

            clearance = self._pose_clearance(x, y, dynamic_xy)
            min_clearance = min(min_clearance, clearance)
            if clearance < self.config.safety_margin:
                return x, y, theta, min_clearance, False

        return x, y, theta, min_clearance, True

    def trajectory_is_safe(
        self,
        robot_state: np.ndarray,
        v: float,
        omega: float,
        dynamic_obstacles: np.ndarray,
    ) -> bool:
        return bool(
            self._simulate(
                robot_state,
                float(v),
                float(omega),
                dynamic_obstacles,
            )[4]
        )

    def command(
        self,
        robot_state: np.ndarray,
        dynamic_obstacles: np.ndarray,
    ) -> np.ndarray:
        if not self.path or self.goal is None:
            return physical_to_normalized_action(
                0.0,
                0.0,
                v_min=self.v_min,
                v_max=self.v_max,
                omega_max=self.omega_max,
            )

        self.command_count += 1
        robot_xy = np.asarray(robot_state[:2], dtype=float)

        if (
            self.command_count
            % max(1, self.config.replan_interval)
            == 0
        ):
            replanned = self.planner.plan(
                (float(robot_xy[0]), float(robot_xy[1])),
                self.goal,
            )
            if replanned:
                self.path = replanned
                self.path_index = 0

        waypoint = np.asarray(self._target_waypoint(robot_xy), dtype=float)
        current_v = float(robot_state[3])
        current_omega = float(robot_state[4])

        v_low = max(
            self.v_min,
            current_v - self.accel_max * self.dt,
        )
        v_high = min(
            self.v_max,
            current_v + self.accel_max * self.dt,
        )
        omega_low = max(
            self.omega_min,
            current_omega - self.alpha_max * self.dt,
        )
        omega_high = min(
            self.omega_max,
            current_omega + self.alpha_max * self.dt,
        )

        v_candidates = np.linspace(
            v_low,
            v_high,
            max(2, self.config.v_samples),
        )
        omega_candidates = np.linspace(
            omega_low,
            omega_high,
            max(3, self.config.omega_samples),
        )

        best_score = -float("inf")
        best = (0.0, 0.0)
        start_distance = float(np.linalg.norm(waypoint - robot_xy))

        for v in v_candidates:
            for omega in omega_candidates:
                x, y, theta, clearance, safe = self._simulate(
                    robot_state,
                    float(v),
                    float(omega),
                    dynamic_obstacles,
                )
                if not safe:
                    continue

                final_xy = np.array([x, y], dtype=float)
                progress = start_distance - float(
                    np.linalg.norm(waypoint - final_xy)
                )
                target_heading = math.atan2(
                    waypoint[1] - y,
                    waypoint[0] - x,
                )
                heading_score = math.cos(
                    _angle_wrap(target_heading - theta)
                )
                clearance_score = min(
                    max(clearance, 0.0),
                    2.0,
                )
                speed_score = (
                    0.0
                    if self.v_max <= self.v_min
                    else (float(v) - self.v_min)
                    / (self.v_max - self.v_min)
                )
                smoothness = (
                    abs(float(v) - self.prev_physical[0])
                    + 0.25
                    * abs(float(omega) - self.prev_physical[1])
                )
                score = (
                    self.config.progress_weight * progress
                    + self.config.heading_weight * heading_score
                    + self.config.clearance_weight * clearance_score
                    + self.config.speed_weight * speed_score
                    - self.config.smoothness_weight * smoothness
                )
                if score > best_score:
                    best_score = score
                    best = (float(v), float(omega))

        self.prev_physical[:] = best
        return physical_to_normalized_action(
            best[0],
            best[1],
            v_min=self.v_min,
            v_max=self.v_max,
            omega_max=self.omega_max,
        )
