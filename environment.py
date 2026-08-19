"""Autonomous Mobile Robot (AMR) Warehouse Environment for Safe Reinforcement Learning.

This environment models a Dynamic Differential-Drive AMR navigating a warehouse.
It natively integrates a Control Barrier Function (CBF-QP) Safety Filter within the 
environment step, ensuring physical collision avoidance overrides unsafe RL actions.

Key Research Features:
    - Native CBF-QP Safety Filter with Predictive Horizons & Uncertainty Margins
    - CBF constraints for Dynamic Agents, Static Shelves, and Boundary Walls
    - Actuator Dynamics: Motor inertia, Gaussian encoder noise, and Wheel Slip
    - Partial Observability: 3-Frame LiDAR Stacking with beam dropout and latency
    - Advanced Social Force Model for Dynamic Obstacles
    - Deterministic Replay mechanics (get_state/set_state)
"""

import copy
import math
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle, Rectangle
import numpy as np

from cbf import CBFSafetyFilter, build_filter_from_config

from config import (
    ACCEL_MAX,
    ACTION_DIM,
    ALPHA_MAX,
    CBF_SAFETY_MARGIN,
    DT,
    DYNAMIC_OBS_RADIUS,
    DYNAMIC_OBS_SPEED_MAX,
    DYNAMIC_OBS_SPEED_MIN,
    GOAL_TOLERANCE,
    MAP_MAX_X,
    MAP_MAX_Y,
    MAP_MIN_X,
    MAP_MIN_Y,
    MAX_EPISODE_STEPS,
    NUM_DYNAMIC_OBSTACLES,
    OMEGA_MAX,
    OMEGA_MIN,
    RANDOM_SEED,
    REWARD_CONFIG,
    ROBOT_RADIUS,
    SHELVES,
    V_MAX,
    V_MIN,
    VISUALIZATION_CONFIG,
)

from utils import (
    circle_circle_collision,
    circle_rectangle_collision,
    clamp,
    closest_obstacle,
    compute_barrier_value,
    distance,
    goal_reached,
    heading_to_goal,
    heading_vector,
    inside_map,
    normalize_angle,
    normalize_observation,
    outside_map,
    random_dynamic_obstacle_positions,
    random_goal_pose,
    random_robot_pose,
    ray_circle_intersection,
    ray_rectangle_intersection,
    rectangle_distance,
    world_to_robot,
)

logger = logging.getLogger(__name__)


class AMRWarehouseEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": VISUALIZATION_CONFIG.FPS,
    }

    def __init__(self, render_mode: Optional[str] = None, max_episode_steps: int = MAX_EPISODE_STEPS, use_cbf_filter: bool = True) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.use_cbf_filter = use_cbf_filter

        # Hardware, Sensing & Control Config
        self.WHEEL_BASE = 0.5
        self.WHEEL_RADIUS = 0.1
        self.ACTUATOR_TAU = 0.15      # Motor inertia lag
        self.NUM_LIDAR_RAYS = 64
        self.MAX_LIDAR_RANGE = 10.0
        self.LIDAR_DROPOUT_RATE = 0.05
        self.FRAME_STACK = 3
        self.LATENCY_STEPS = 1        # Sensor/Network Latency Delay

        self.lidar_angles = np.linspace(-math.pi, math.pi, self.NUM_LIDAR_RAYS, endpoint=False)
        self.curriculum_level = 1.0

        # CBF-QP Safety Filter (see cbf.py). Built once here from config.py
        # so environment.py, safe_sac.py, and any standalone diagnostics
        # all share the exact same filter behavior.
        self.cbf_filter: CBFSafetyFilter = build_filter_from_config()
        self.last_cbf_diagnostics: Dict[str, Any] = {
            "intervened": False, "num_active_constraints": 0, "solver_success": True,
            "tier": "strict", "slack": 0.0,
        }

        # Action & Observation Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "robot_state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "goal": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "lidar": spaces.Box(low=-1.0, high=1.0, shape=(self.NUM_LIDAR_RAYS * self.FRAME_STACK,), dtype=np.float32),
        })

        # State Variables
        self.robot_state: np.ndarray = np.zeros(5, dtype=np.float64)
        self.goal_pos: np.ndarray = np.zeros(2, dtype=np.float64)
        self.dynamic_obstacles: np.ndarray = np.zeros((NUM_DYNAMIC_OBSTACLES, 6), dtype=np.float64)
        
        self.step_count: int = 0
        self.prev_distance_to_goal: float = 0.0
        self.trajectory_history: List[Tuple[float, float]] = []
        
        self.actual_v: float = 0.0
        self.actual_omega: float = 0.0
        self.prev_action = np.zeros(2)
        
        # Buffers
        self.lidar_history = deque(maxlen=self.FRAME_STACK)
        self.obs_buffer: List[Dict[str, np.ndarray]] = []

        self.max_diag = math.hypot(MAP_MAX_X - MAP_MIN_X, MAP_MAX_Y - MAP_MIN_Y)
        self.np_random = np.random.default_rng(RANDOM_SEED)

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def set_curriculum_level(self, level: float) -> None:
        self.curriculum_level = clamp(level, 0.0, 1.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.WHEEL_RADIUS = self.np_random.uniform(0.09, 0.11)
        self.WHEEL_BASE = self.np_random.uniform(0.45, 0.55)

        self.step_count = 0
        self.trajectory_history.clear()
        self.actual_v = 0.0
        self.actual_omega = 0.0
        self.prev_action = np.zeros(2)
        self.last_cbf_diagnostics = {
            "intervened": False, "num_active_constraints": 0, "solver_success": True,
            "tier": "strict", "slack": 0.0,
        }

        self._sample_robot()
        self._sample_goal()
        self._sample_dynamic_obstacles()

        self.prev_distance_to_goal = distance(self.robot_state[:2], self.goal_pos)
        self.trajectory_history.append((float(self.robot_state[0]), float(self.robot_state[1])))

        initial_lidar = self._get_lidar_scan()
        for _ in range(self.FRAME_STACK):
            self.lidar_history.append(initial_lidar)

        initial_obs = self._get_observation()
        self.obs_buffer = [initial_obs for _ in range(self.LATENCY_STEPS + 1)]

        return self.obs_buffer[0], self._get_info(False, False, "none", 0.0, 0.0, {})

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        a_v, a_omega = clamp(float(action[0]), -1.0, 1.0), clamp(float(action[1]), -1.0, 1.0)
        v_nom = V_MIN + 0.5 * (a_v + 1.0) * (V_MAX - V_MIN)
        omega_nom = a_omega * OMEGA_MAX

        # 1. CBF-QP Safety Filter (see cbf.py for the constraint assembly
        #    and solver; this env only owns the state that goes INTO it).
        if self.use_cbf_filter:
            v_cmd, omega_cmd, self.last_cbf_diagnostics = self.cbf_filter.solve(
                v_nom, omega_nom, self.robot_state, self.dynamic_obstacles, SHELVES
            )
        else:
            v_cmd, omega_cmd = v_nom, omega_nom
            self.last_cbf_diagnostics = {
                "intervened": False, "num_active_constraints": 0, "solver_success": True,
                "tier": "strict", "slack": 0.0,
            }

        # 2. Differential Kinematics with Friction/Slip
        is_collision, collision_type = self._update_robot_dynamics(v_cmd, omega_cmd)
        self.trajectory_history.append((float(self.robot_state[0]), float(self.robot_state[1])))
        
        # 3. Environment Dynamics
        self._update_social_force_obstacles()
        
        if not is_collision:
            is_collision, collision_type = self._check_collisions(self.robot_state)

        is_goal = goal_reached(self.robot_state[:2], self.goal_pos, GOAL_TOLERANCE)
        terminated = is_collision or is_goal
        truncated = self.step_count >= self.max_episode_steps

        # 4. Reward & Metrics
        reward, reward_breakdown = self._calculate_reward(is_goal, is_collision, action)

        self.prev_distance_to_goal = distance(self.robot_state[:2], self.goal_pos)
        self.prev_action = action
        
        self.lidar_history.append(self._get_lidar_scan())
        current_obs = self._get_observation()
        
        # Latency Handling
        self.obs_buffer.append(current_obs)
        delayed_obs = self.obs_buffer.pop(0)

        info = self._get_info(is_collision, is_goal, collision_type, self.actual_v, self.actual_omega, reward_breakdown)

        if self.render_mode == "human":
            self.render()

        return delayed_obs, reward, terminated, truncated, info

    # =========================================================================
    # Differential Drive & Physical Actuation
    # =========================================================================

    def _update_robot_dynamics(self, v_cmd: float, omega_cmd: float) -> Tuple[bool, str]:
        x, y, theta, _, _ = self.robot_state

        # Motor Lag Filter
        alpha_lag = DT / (self.ACTUATOR_TAU + DT)
        v_target = self.actual_v + alpha_lag * (v_cmd - self.actual_v)
        omega_target = self.actual_omega + alpha_lag * (omega_cmd - self.actual_omega)

        # Inverse Kinematics
        w_right_target = (v_target + (omega_target * self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS
        w_left_target  = (v_target - (omega_target * self.WHEEL_BASE / 2.0)) / self.WHEEL_RADIUS

        # Motor Encoder Noise
        noise_scale = 0.05 * self.curriculum_level
        w_right_actual = w_right_target + self.np_random.normal(0, noise_scale)
        w_left_actual  = w_left_target + self.np_random.normal(0, noise_scale)

        # Wheel Slip Mechanics
        w_right_actual *= self.np_random.uniform(0.92, 1.0)
        w_left_actual  *= self.np_random.uniform(0.92, 1.0)

        # Forward Kinematics
        self.actual_v = (self.WHEEL_RADIUS / 2.0) * (w_right_actual + w_left_actual)
        self.actual_omega = (self.WHEEL_RADIUS / self.WHEEL_BASE) * (w_right_actual - w_left_actual)

        dx = self.actual_v * math.cos(theta) * DT
        dy = self.actual_v * math.sin(theta) * DT
        dtheta = self.actual_omega * DT

        test_state = np.array([x + dx, y + dy, normalize_angle(theta + dtheta), self.actual_v, self.actual_omega], dtype=np.float64)
        
        is_collision, c_type = self._check_collisions(test_state)
        if is_collision:
            self.actual_v, self.actual_omega = 0.0, 0.0
            self.robot_state[3], self.robot_state[4] = 0.0, 0.0
            return True, c_type
            
        self.robot_state = test_state
        return False, "none"

    # =========================================================================
    # Environment Updates
    # =========================================================================

    def _update_social_force_obstacles(self) -> None:
        new_obstacles = np.copy(self.dynamic_obstacles)
        num_active = int(NUM_DYNAMIC_OBSTACLES * self.curriculum_level)
        
        for i in range(num_active):
            ox, oy, otheta, ospeed, gx, gy = self.dynamic_obstacles[i]

            RELAX_TIME = 0.5
            DESIRED_SPEED = DYNAMIC_OBS_SPEED_MAX
            
            dist_to_goal = math.hypot(gx - ox, gy - oy)
            if dist_to_goal < 1.0:
                try: gx, gy = random_goal_pose(rng=self.np_random, robot_pos=(ox, oy), shelves=SHELVES)
                except RuntimeError: pass

            dir_x, dir_y = (gx - ox) / dist_to_goal, (gy - oy) / dist_to_goal
            f_goal_x = (dir_x * DESIRED_SPEED - ospeed * math.cos(otheta)) / RELAX_TIME
            f_goal_y = (dir_y * DESIRED_SPEED - ospeed * math.sin(otheta)) / RELAX_TIME

            f_rep_x, f_rep_y = 0.0, 0.0
            for j in range(num_active):
                if i == j: continue
                jx, jy = self.dynamic_obstacles[j][:2]
                dx, dy = ox - jx, oy - jy
                dist = math.hypot(dx, dy)
                if dist < 2.5:
                    force = 2.0 * math.exp(-dist / 0.5)
                    f_rep_x += force * (dx / dist)
                    f_rep_y += force * (dy / dist)

            vx = ospeed * math.cos(otheta) + (f_goal_x + f_rep_x) * DT
            vy = ospeed * math.sin(otheta) + (f_goal_y + f_rep_y) * DT
            
            ospeed_new = clamp(math.hypot(vx, vy), DYNAMIC_OBS_SPEED_MIN, DYNAMIC_OBS_SPEED_MAX)
            otheta_new = math.atan2(vy, vx)

            ox_next = ox + ospeed_new * math.cos(otheta_new) * DT
            oy_next = oy + ospeed_new * math.sin(otheta_new) * DT

            if outside_map((ox_next, oy_next), margin=DYNAMIC_OBS_RADIUS) or any(circle_rectangle_collision((ox_next, oy_next), DYNAMIC_OBS_RADIUS, rect) for rect in SHELVES):
                otheta_new = float(normalize_angle(otheta_new + math.pi + self.np_random.uniform(-0.5, 0.5)))
                ox_next, oy_next = ox, oy

            new_obstacles[i] = np.array([ox_next, oy_next, otheta_new, ospeed_new, gx, gy], dtype=np.float64)
            
        self.dynamic_obstacles = new_obstacles

    def _check_collisions(self, state_to_check: np.ndarray) -> Tuple[bool, str]:
        rx, ry = state_to_check[0], state_to_check[1]
        if not inside_map((rx, ry), margin=ROBOT_RADIUS): return True, "wall"
        if any(circle_rectangle_collision((rx, ry), ROBOT_RADIUS, rect) for rect in SHELVES): return True, "shelf"
        if any(circle_circle_collision((rx, ry), ROBOT_RADIUS, obs[:2], DYNAMIC_OBS_RADIUS) for obs in self.dynamic_obstacles): return True, "dynamic_obstacle"
        return False, "none"

    def _compute_cbf_barriers(self) -> Dict[str, float]:
        r_pos = self.robot_state[:2]
        dyn_barriers = [compute_barrier_value(r_pos, obs[:2], ROBOT_RADIUS + DYNAMIC_OBS_RADIUS) for obs in self.dynamic_obstacles]
        shelf_barriers = [(rectangle_distance(r_pos, rect) ** 2) - (ROBOT_RADIUS ** 2) for rect in SHELVES]
        rx, ry = r_pos[0], r_pos[1]
        wall_barriers = [rx - (MAP_MIN_X + ROBOT_RADIUS), (MAP_MAX_X - ROBOT_RADIUS) - rx, ry - (MAP_MIN_Y + ROBOT_RADIUS), (MAP_MAX_Y - ROBOT_RADIUS) - ry]
        
        min_dyn = min(dyn_barriers) if dyn_barriers else float("inf")
        min_shelf = min(shelf_barriers) if shelf_barriers else float("inf")
        return {"overall_min_barrier": float(min(min_dyn, min_shelf, min(wall_barriers)))}

    def _get_lidar_scan(self) -> np.ndarray:
        scan = np.full(self.NUM_LIDAR_RAYS, self.MAX_LIDAR_RANGE, dtype=np.float32)
        rx, ry, rtheta = self.robot_state[:3]
        origin = (rx, ry)

        for i, angle in enumerate(self.lidar_angles):
            ray_dir = heading_vector(rtheta + angle)
            min_dist = self.MAX_LIDAR_RANGE
            
            tx = (MAP_MAX_X - origin[0]) / ray_dir[0] if ray_dir[0] > 0 else (MAP_MIN_X - origin[0]) / ray_dir[0] if ray_dir[0] < 0 else float('inf')
            ty = (MAP_MAX_Y - origin[1]) / ray_dir[1] if ray_dir[1] > 0 else (MAP_MIN_Y - origin[1]) / ray_dir[1] if ray_dir[1] < 0 else float('inf')
            if min(tx, ty) < min_dist: min_dist = min(tx, ty)

            for rect in SHELVES:
                d = ray_rectangle_intersection(origin, ray_dir, rect)
                if d < min_dist: min_dist = d
            for obs in self.dynamic_obstacles:
                d = ray_circle_intersection(origin, ray_dir, obs[:2], DYNAMIC_OBS_RADIUS)
                if d < min_dist: min_dist = d
                
            scan[i] = min_dist

        noise = self.np_random.normal(0, 0.05 * self.curriculum_level, self.NUM_LIDAR_RAYS)
        scan = np.clip(scan + noise, 0.0, self.MAX_LIDAR_RANGE)
        
        dropout_mask = self.np_random.random(self.NUM_LIDAR_RAYS) < (self.LIDAR_DROPOUT_RATE * self.curriculum_level)
        scan[dropout_mask] = self.MAX_LIDAR_RANGE
        
        return normalize_observation(scan, 0.0, self.MAX_LIDAR_RANGE)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        rx, ry, theta, _, _ = self.robot_state
        gx, gy = self.goal_pos

        robot_obs = np.array([
            normalize_observation(rx, MAP_MIN_X, MAP_MAX_X),
            normalize_observation(ry, MAP_MIN_Y, MAP_MAX_Y),
            theta / math.pi,
            normalize_observation(self.actual_v, V_MIN, V_MAX),
            clamp(self.actual_omega / OMEGA_MAX, -1.0, 1.0),
        ], dtype=np.float32)

        goal_rel_robot = world_to_robot((gx, gy), self.robot_state[:3])
        goal_obs = np.array([
            clamp(goal_rel_robot[0] / self.max_diag, -1.0, 1.0),
            clamp(goal_rel_robot[1] / self.max_diag, -1.0, 1.0),
            clamp(distance((rx, ry), (gx, gy)) / self.max_diag, 0.0, 1.0),
            heading_to_goal(self.robot_state[:3], self.goal_pos) / math.pi,
        ], dtype=np.float32)

        lidar_obs = np.concatenate(list(self.lidar_history)).astype(np.float32)

        return {
            "robot_state": robot_obs,
            "goal": goal_obs,
            "lidar": lidar_obs,
        }

    def _calculate_reward(self, is_goal: bool, is_collision: bool, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        breakdown = {k: 0.0 for k in ["r_progress", "r_goal", "r_collision", "r_time", "r_heading", "r_energy", "r_oscillation", "r_deadlock", "r_safety"]}

        if is_goal:
            breakdown["r_goal"] = REWARD_CONFIG.SUCCESS_REWARD
            return float(REWARD_CONFIG.SUCCESS_REWARD), breakdown
        if is_collision:
            breakdown["r_collision"] = -REWARD_CONFIG.COLLISION_PENALTY
            return float(-REWARD_CONFIG.COLLISION_PENALTY), breakdown

        current_dist = distance(self.robot_state[:2], self.goal_pos)
        breakdown["r_progress"] = float(REWARD_CONFIG.PROGRESS_WEIGHT * (self.prev_distance_to_goal - current_dist))
        breakdown["r_heading"] = float(REWARD_CONFIG.HEADING_WEIGHT * math.cos(heading_to_goal(self.robot_state[:3], self.goal_pos)))
        breakdown["r_time"] = float(-REWARD_CONFIG.TIME_PENALTY)
        breakdown["r_energy"] = float(-0.01 * (action[0]**2 + action[1]**2))
        breakdown["r_oscillation"] = float(-0.05 * abs(action[1] - self.prev_action[1]))
        
        if abs(self.actual_v) < 0.05 and current_dist > GOAL_TOLERANCE: breakdown["r_deadlock"] = -0.5

        min_barrier = self._compute_cbf_barriers()["overall_min_barrier"]
        if min_barrier < CBF_SAFETY_MARGIN:
            breakdown["r_safety"] = float(-REWARD_CONFIG.CBF_INTERVENTION_PENALTY * (CBF_SAFETY_MARGIN - min_barrier))

        return float(sum(breakdown.values())), breakdown

    def _get_info(self, is_collision: bool, is_goal: bool, collision_type: str, v_act: float, w_act: float, rew: Dict[str, float]) -> Dict[str, Any]:
        cbf_dict = self._compute_cbf_barriers()
        return {
            "episode_step": self.step_count, "distance_to_goal": float(distance(self.robot_state[:2], self.goal_pos)),
            "goal_reached": is_goal, "collision": is_collision, "collision_type": collision_type,
            "barrier_value": float(cbf_dict["overall_min_barrier"]), "reward_breakdown": rew,
            "v_actual": float(v_act), "omega_actual": float(w_act), "is_safe": cbf_dict["overall_min_barrier"] >= 0.0,
            "cbf_intervened": self.last_cbf_diagnostics["intervened"],
            "cbf_num_active_constraints": self.last_cbf_diagnostics["num_active_constraints"],
            "cbf_solver_success": self.last_cbf_diagnostics["solver_success"],
            # Expose new diagnostics added by CBFSafetyFilter.solve()
            "cbf_tier": self.last_cbf_diagnostics.get("tier"),
            "cbf_slack": self.last_cbf_diagnostics.get("slack"),
            # Backwards-compatible top-level keys for downstream scripts
            "tier": self.last_cbf_diagnostics.get("tier"),
            "slack": self.last_cbf_diagnostics.get("slack"),
        }

    # =========================================================================
    # Research Deterministic Replay Utilities
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        return {
            "robot_state": np.copy(self.robot_state), "goal_pos": np.copy(self.goal_pos), "dynamic_obstacles": np.copy(self.dynamic_obstacles),
            "step_count": self.step_count, "actual_v": self.actual_v, "actual_omega": self.actual_omega,
            "obs_buffer": [{k: np.copy(v) for k, v in obs.items()} for obs in self.obs_buffer],
            "lidar_history": [np.copy(scan) for scan in self.lidar_history],
            "last_cbf_diagnostics": copy.deepcopy(self.last_cbf_diagnostics),
            "rng_state": copy.deepcopy(self.np_random.bit_generator.state)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.robot_state = np.copy(state["robot_state"])
        self.goal_pos = np.copy(state["goal_pos"])
        self.dynamic_obstacles = np.copy(state["dynamic_obstacles"])
        self.step_count = state["step_count"]
        self.actual_v = state["actual_v"]
        self.actual_omega = state["actual_omega"]
        self.obs_buffer = [{k: np.copy(v) for k, v in obs.items()} for obs in state["obs_buffer"]]
        self.lidar_history = deque([np.copy(scan) for scan in state["lidar_history"]], maxlen=self.FRAME_STACK)
        self.last_cbf_diagnostics = copy.deepcopy(state.get("last_cbf_diagnostics", {
            "intervened": False, "num_active_constraints": 0, "solver_success": True,
        }))
        self.np_random.bit_generator.state = state["rng_state"]

    # =========================================================================
    # Internal Sampling & Rendering 
    # =========================================================================

    def _sample_robot(self) -> None:
        try: init_pose = random_robot_pose(rng=self.np_random, shelves=SHELVES)
        except RuntimeError: init_pose = np.array([0.0, -9.0, 0.0], dtype=np.float64)
        self.robot_state = np.array([init_pose[0], init_pose[1], init_pose[2], 0.0, 0.0], dtype=np.float64)

    def _sample_goal(self) -> None:
        try: self.goal_pos = random_goal_pose(rng=self.np_random, robot_pos=self.robot_state[:2], shelves=SHELVES)
        except RuntimeError: self.goal_pos = np.array([0.0, 8.0], dtype=np.float64)

    def _sample_dynamic_obstacles(self) -> None:
        for i in range(NUM_DYNAMIC_OBSTACLES):
            try:
                pos = random_dynamic_obstacle_positions(rng=self.np_random, num_obstacles=1, robot_pos=self.robot_state[:2], goal_pos=self.goal_pos, shelves=SHELVES)[0]
                goal = random_goal_pose(rng=self.np_random, robot_pos=pos[:2], shelves=SHELVES)
                self.dynamic_obstacles[i] = np.array([pos[0], pos[1], pos[2], pos[3], goal[0], goal[1]])
            except RuntimeError: pass

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None: return None
        if self.fig is None or self.ax is None:
            matplotlib.use("Agg" if self.render_mode == "rgb_array" else "TkAgg")
            self.fig, self.ax = plt.subplots(figsize=VISUALIZATION_CONFIG.FIGURE_SIZE)
            if self.render_mode == "human": plt.ion()
        self.ax.clear()
        
        self.ax.set_xlim(MAP_MIN_X - 0.5, MAP_MAX_X + 0.5)
        self.ax.set_ylim(MAP_MIN_Y - 0.5, MAP_MAX_Y + 0.5)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"AMR Safe RL Warehouse Navigation - Step {self.step_count}", fontsize=12, fontweight="bold")
        
        self.ax.add_patch(Rectangle((MAP_MIN_X, MAP_MIN_Y), MAP_MAX_X - MAP_MIN_X, MAP_MAX_Y - MAP_MIN_Y, fill=False, edgecolor="black", linewidth=2.5))
        for rect in SHELVES: self.ax.add_patch(Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], facecolor=VISUALIZATION_CONFIG.COLOR_SHELF, edgecolor="black", alpha=0.85))

        self.ax.add_patch(Circle(self.goal_pos, GOAL_TOLERANCE, facecolor=VISUALIZATION_CONFIG.COLOR_GOAL, edgecolor="darkgreen", alpha=0.6))
        
        for ox, oy, otheta, ospeed, gx, gy in self.dynamic_obstacles[:int(NUM_DYNAMIC_OBSTACLES * self.curriculum_level)]:
            self.ax.add_patch(Circle((ox, oy), DYNAMIC_OBS_RADIUS, facecolor=VISUALIZATION_CONFIG.COLOR_DYNAMIC_OBS, edgecolor="maroon", alpha=0.8))
            arrow_len = max(0.4, ospeed * 0.5)
            self.ax.arrow(ox, oy, arrow_len * math.cos(otheta), arrow_len * math.sin(otheta), head_width=0.15, head_length=0.15, fc="black", ec="black")

        rx, ry, rtheta = self.robot_state[:3]
        self.ax.add_patch(Circle((rx, ry), ROBOT_RADIUS + CBF_SAFETY_MARGIN, fill=False, edgecolor=VISUALIZATION_CONFIG.COLOR_SAFETY_BARRIER, linestyle=":", linewidth=2.0))
        self.ax.add_patch(Circle((rx, ry), ROBOT_RADIUS, facecolor=VISUALIZATION_CONFIG.COLOR_ROBOT, edgecolor="navy", alpha=0.9))
        self.ax.arrow(rx, ry, (ROBOT_RADIUS + 0.2) * math.cos(rtheta), (ROBOT_RADIUS + 0.2) * math.sin(rtheta), head_width=0.15, head_length=0.15, fc="yellow", ec="navy")

        if self.render_mode == "human":
            plt.draw()
            plt.pause(1.0 / VISUALIZATION_CONFIG.FPS)
            return None
        elif self.render_mode == "rgb_array":
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None