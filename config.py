"""
Configuration module for Safe Reinforcement Learning Autonomous Mobile Robot (AMR).

This module defines all physical, dynamic, safety, environment, observation,
visualization, and training hyperparameters for a 20x20m warehouse AMR controlled
via Proximal Policy Optimization (PPO) filtered by Control Barrier Functions (CBF).

Mathematical Formulation
------------------------
1. Robot Kinematics (Unicycle Model):
   \dot{x} = v \cos(\theta)
   \dot{y} = v \sin(\theta)
   \dot{\theta} = \omega
   where state \mathbf{x} = [x, y, \theta, v, \omega]^T and control input \mathbf{u} = [v_{\text{cmd}}, \omega_{\text{cmd}}]^T
   or continuous velocity integration.

2. Control Barrier Function (CBF) & Safe Set:
   A safe set \mathcal{C} is defined by the zero-superlevel set of a continuously
   differentiable function h: \mathbb{R}^n \rightarrow \mathbb{R}:
   \mathcal{C} = \{ \mathbf{x} \in \mathbb{R}^n : h(\mathbf{x}) \ge 0 \}

   By Nagumo's Theorem and Forward Invariance, the system remains safe for all t >= 0
   if:
   \dot{h}(\mathbf{x}, \mathbf{u}) + \gamma(h(\mathbf{x})) \ge 0

   For a control-affine system \dot{\mathbf{x}} = f(\mathbf{x}) + g(\mathbf{x})\mathbf{u},
   the CBF constraint becomes a linear inequality in control input \mathbf{u}:
   L_f h(\mathbf{x}) + L_g h(\mathbf{x}) \mathbf{u} + \gamma h(\mathbf{x}) \ge 0

3. Safe RL Safety Filter (Quadratic Program):
   \min_{\mathbf{u}, \delta} \frac{1}{2} \|\mathbf{u} - \mathbf{u}_{\text{RL}}\|_2^2 + w_{\text{slack}} \delta^2
   subject to:
   L_f h_i(\mathbf{x}) + L_g h_i(\mathbf{x}) \mathbf{u} + \gamma h_i(\mathbf{x}) \ge -\delta, \quad \forall i
   \mathbf{u}_{\text{min}} \le \mathbf{u} \le \mathbf{u}_{\text{max}}
   \delta \ge 0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, List, Tuple
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """
    Parameters governing time discretization and simulation execution.

    Attributes
    ----------
    DT : float
        Discrete integration time step in seconds (10 Hz control loop).
    MAX_EPISODE_STEPS : int
        Maximum allowable steps per training/evaluation episode.
    RANDOM_SEED : int
        Global random seed for reproducibility across RNGs.
    """

    DT: float = 0.1
    MAX_EPISODE_STEPS: int = 300
    RANDOM_SEED: int = 42


@dataclass(frozen=True)
class RobotConfig:
    """
    Physical dimensions and kinematic velocity limits for the unicycle AMR.

    Attributes
    ----------
    RADIUS : float
        Bounding collision radius of the circular AMR in meters.
    V_MIN : float
        Minimum linear velocity command in m/s.
    V_MAX : float
        Maximum linear velocity command in m/s.
    OMEGA_MIN : float
        Minimum angular velocity command in rad/s.
    OMEGA_MAX : float
        Maximum angular velocity command in rad/s.
    ACCEL_MAX : float
        Maximum linear acceleration limit in m/s^2.
    ALPHA_MAX : float
        Maximum angular acceleration limit in rad/s^2.
    """

    RADIUS: float = 0.30
    V_MIN: float = 0.0
    V_MAX: float = 1.0
    OMEGA_MIN: float = -1.5
    OMEGA_MAX: float = 1.5
    ACCEL_MAX: float = 1.5
    ALPHA_MAX: float = 2.5


@dataclass(frozen=True)
class WarehouseConfig:
    """
    Warehouse environment setup, bounding box, rectangular shelf layout, and targets.

    Attributes
    ----------
    MAP_MIN_X : float
        Minimum X coordinate of warehouse environment boundary in meters.
    MAP_MAX_X : float
        Maximum X coordinate of warehouse environment boundary in meters.
    MAP_MIN_Y : float
        Minimum Y coordinate of warehouse environment boundary in meters.
    MAP_MAX_Y : float
        Maximum Y coordinate of warehouse environment boundary in meters.
    GOAL_TOLERANCE : float
        Radius tolerance around target destination to declare goal reached (meters).
    SHELVES : List[Tuple[float, float, float, float]]
        List of rectangular warehouse shelves defined as (xmin, ymin, xmax, ymax).
    WALL_MARGIN : float
        Safety clearance buffer from boundary walls in meters.
    """

    MAP_MIN_X: float = -10.0
    MAP_MAX_X: float = 10.0
    MAP_MIN_Y: float = -10.0
    MAP_MAX_Y: float = 10.0
    GOAL_TOLERANCE: float = 0.30
    SHELVES: List[Tuple[float, float, float, float]] = field(
        default_factory=lambda: [
            # Rectangular warehouse shelf racks: (xmin, ymin, xmax, ymax)
            (-8.0, -8.0, -6.0, -2.0),
            (-8.0, 2.0, -6.0, 8.0),
            (-3.0, -8.0, -1.0, -2.0),
            (-3.0, 2.0, -1.0, 8.0),
            (1.0, -8.0, 3.0, -2.0),
            (1.0, 2.0, 3.0, 8.0),
            (6.0, -8.0, 8.0, -2.0),
            (6.0, 2.0, 8.0, 8.0),
        ]
    )
    WALL_MARGIN: float = 0.10


@dataclass(frozen=True)
class DynamicObstacleConfig:
    """
    Configuration parameters for moving obstacles (e.g., human workers, other AMRs).

    Attributes
    ----------
    NUM_OBSTACLES : int
        Total number of dynamic obstacles in the warehouse.
    RADIUS : float
        Bounding radius of each dynamic obstacle in meters.
    SPEED_MIN : float
        Minimum speed of dynamic obstacles in m/s.
    SPEED_MAX : float
        Maximum speed of dynamic obstacles in m/s.
    DETECTION_RADIUS : float
        Maximum sensing/detection radius for dynamic obstacles in meters.
    """

    NUM_OBSTACLES: int = 5
    RADIUS: float = 0.30
    SPEED_MIN: float = 0.20
    SPEED_MAX: float = 0.80
    DETECTION_RADIUS: float = 4.0


@dataclass(frozen=True)
class ObservationConfig:
    """
    Observation and action space dimensions for RL policy inputs and outputs.

    Attributes
    ----------
    ACTION_DIM : int
        Dimensionality of continuous action space: [v_cmd, omega_cmd].
    ROBOT_STATE_DIM : int
        Ego-robot state features: [x, y, theta, v, omega].
    GOAL_STATE_DIM : int
        Relative target features: [rel_x, rel_y, distance, heading_error].
    DYNAMIC_OBS_DIM : int
        Features per dynamic obstacle within sensing radius: [rel_x, rel_y, vx, vy, dist].
    MAX_DYNAMIC_OBSTACLES : int
        Maximum number of dynamic obstacles tracked in the state vector.
    TOTAL_OBS_DIM : int
        Total flattened state observation vector dimension (5 + 4 + 5*5 = 34).
    """

    ACTION_DIM: int = 2
    ROBOT_STATE_DIM: int = 5
    GOAL_STATE_DIM: int = 4
    DYNAMIC_OBS_DIM: int = 5
    MAX_DYNAMIC_OBSTACLES: int = 5
    TOTAL_OBS_DIM: int = 34


@dataclass(frozen=True)
class VisualizationConfig:
    """
    Color scheme, rendering window dimensions, and playback FPS for visualization.

    Attributes
    ----------
    FIGURE_SIZE : Tuple[int, int]
        Matplotlib figure width and height in inches.
    FPS : int
        Frames per second for live animation rendering.
    COLOR_ROBOT : str
        Hex color code for ego autonomous mobile robot.
    COLOR_GOAL : str
        Hex color code for target destination.
    COLOR_SHELF : str
        Hex color code for static rectangular warehouse shelves.
    COLOR_DYNAMIC_OBS : str
        Hex color code for dynamic moving obstacles.
    COLOR_TRAJECTORY : str
        Hex color code for robot trajectory path.
    COLOR_SAFETY_BARRIER : str
        Hex color code for CBF safety clearance boundary.
    """

    FIGURE_SIZE: Tuple[int, int] = (10, 10)
    FPS: int = 10
    COLOR_ROBOT: str = "#1f77b4"
    COLOR_GOAL: str = "#2ca02c"
    COLOR_SHELF: str = "#4a4a4a"
    COLOR_DYNAMIC_OBS: str = "#d62728"
    COLOR_TRAJECTORY: str = "#9467bd"
    COLOR_SAFETY_BARRIER: str = "#ff7f0e"


@dataclass(frozen=True)
class CBFConfig:
    """
    Control Barrier Function (CBF) and OSQP solver configuration parameters.

    Attributes
    ----------
    SAFETY_MARGIN : float
        Additional safety distance buffer added to robot and obstacle boundaries.
    GAMMA : float
        Class K linear gain for CBF constraint relaxation rate: \dot{h} + \gamma h \ge 0.
    HOCBF_GAMMA1 : float
        First decay constant for High-Order CBF (relative degree 2 systems).
    HOCBF_GAMMA2 : float
        Second decay constant for High-Order CBF (relative degree 2 systems).
    SLACK_WEIGHT : float
        Quadratic penalty weight w_{slack} in the QP objective for slack variables.
    QP_SOLVER : str
        Solver name passed to CVXPY (e.g., 'OSQP').
    QP_VERBOSE : bool
        Flag to enable OSQP solver console debug output.
    QP_MAX_ITER : int
        Maximum allowable iterations for OSQP solver.
    QP_ABSTOL : float
        Absolute convergence tolerance for OSQP solver.
    """

    SAFETY_MARGIN: float = 0.15
    GAMMA: float = 2.0
    HOCBF_GAMMA1: float = 3.0
    HOCBF_GAMMA2: float = 3.0
    SLACK_WEIGHT: float = 1e6
    QP_SOLVER: str = "OSQP"
    QP_VERBOSE: bool = False
    QP_MAX_ITER: int = 4000
    QP_ABSTOL: float = 1e-5


@dataclass(frozen=True)
class RewardConfig:
    """
    Reward function coefficients for Reinforcement Learning formulation.

    Attributes
    ----------
    SUCCESS_REWARD : float
        Bonus assigned upon successfully reaching the target location.
    COLLISION_PENALTY : float
        Severe penalty applied upon colliding with obstacles or walls.
    PROGRESS_WEIGHT : float
        Multiplier for step-wise reduction in Euclidean distance to goal.
    HEADING_WEIGHT : float
        Multiplier rewarding alignment of robot heading with target direction.
    ACTION_PENALTY_WEIGHT : float
        Penalty applied to high control effort to promote smooth motion.
    TIME_PENALTY : float
        Per-step penalty to encourage minimal-time target reaching.
    CBF_INTERVENTION_PENALTY : float
        Penalty applied when the CBF safety filter alters the nominal RL action.
    """

    SUCCESS_REWARD: float = 150.0
    COLLISION_PENALTY: float = 100.0
    PROGRESS_WEIGHT: float = 5.0
    HEADING_WEIGHT: float = 0.2
    ACTION_PENALTY_WEIGHT: float = 0.05
    TIME_PENALTY: float = 0.1
    CBF_INTERVENTION_PENALTY: float = 0.2


@dataclass(frozen=True)
class PPOConfig:
    """
    Hyperparameters for Stable-Baselines3 Proximal Policy Optimization (PPO).

    Attributes
    ----------
    LEARNING_RATE : float
        Initial learning rate for actor and critic networks.
    N_STEPS : int
        Number of environment steps to collect per update rollout.
    BATCH_SIZE : int
        Minibatch size for gradient descent optimization.
    N_EPOCHS : int
        Number of optimization epochs per rollout update.
    GAMMA : float
        Discount factor for future rewards.
    GAE_LAMBDA : float
        Factor for trade-off of bias vs variance in Generalized Advantage Estimator.
    CLIP_RANGE : float
        Clipping parameter for PPO surrogate objective.
    ENT_COEF : float
        Entropy coefficient for exploration promotion.
    VF_COEF : float
        Value function loss term weight in total loss.
    MAX_GRAD_NORM : float
        Maximum gradient norm threshold for clipping.
    TOTAL_TIMESTEPS : int
        Total training environment steps.
    POLICY_TYPE : str
        Policy neural network architecture type (e.g., 'MlpPolicy').
    """

    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.005
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    TOTAL_TIMESTEPS: int = 100_000
    POLICY_TYPE: str = "MlpPolicy"


@dataclass(frozen=True)
class PathConfig:
    """
    Directory paths for logs, checkpoints, plots, and models.

    Attributes
    ----------
    BASE_DIR : Path
        Root directory of the project workspace.
    LOG_DIR : Path
        Directory for training performance logs.
    MODEL_DIR : Path
        Directory for saving trained SB3 model weights.
    TENSORBOARD_DIR : Path
        Directory for TensorBoard experiment metrics.
    PLOT_DIR : Path
        Directory for generated trajectory and evaluation plots.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent
    LOG_DIR: Path = BASE_DIR / "logs"
    MODEL_DIR: Path = BASE_DIR / "models"
    TENSORBOARD_DIR: Path = BASE_DIR / "tensorboard_logs"
    PLOT_DIR: Path = BASE_DIR / "plots"

    def create_directories(self) -> None:
        """Create output folders if they do not already exist."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOT_DIR.mkdir(parents=True, exist_ok=True)


# Instantiate Global Configuration Objects
SIM_CONFIG: Final[SimulationConfig] = SimulationConfig()
ROBOT_CONFIG: Final[RobotConfig] = RobotConfig()
WAREHOUSE_CONFIG: Final[WarehouseConfig] = WarehouseConfig()
DYNAMIC_OBS_CONFIG: Final[DynamicObstacleConfig] = DynamicObstacleConfig()
OBS_CONFIG: Final[ObservationConfig] = ObservationConfig()
VISUALIZATION_CONFIG: Final[VisualizationConfig] = VisualizationConfig()
CBF_CONFIG: Final[CBFConfig] = CBFConfig()
REWARD_CONFIG: Final[RewardConfig] = RewardConfig()
PPO_CONFIG: Final[PPOConfig] = PPOConfig()
PATH_CONFIG: Final[PathConfig] = PathConfig()

# Global Direct Constants (for modular importing across environment, dynamics, and CBF nodes)
DT: Final[float] = SIM_CONFIG.DT
MAX_EPISODE_STEPS: Final[int] = SIM_CONFIG.MAX_EPISODE_STEPS
RANDOM_SEED: Final[int] = SIM_CONFIG.RANDOM_SEED

ROBOT_RADIUS: Final[float] = ROBOT_CONFIG.RADIUS
V_MIN: Final[float] = ROBOT_CONFIG.V_MIN
V_MAX: Final[float] = ROBOT_CONFIG.V_MAX
OMEGA_MIN: Final[float] = ROBOT_CONFIG.OMEGA_MIN
OMEGA_MAX: Final[float] = ROBOT_CONFIG.OMEGA_MAX
ACCEL_MAX: Final[float] = ROBOT_CONFIG.ACCEL_MAX
ALPHA_MAX: Final[float] = ROBOT_CONFIG.ALPHA_MAX

MAP_MIN_X: Final[float] = WAREHOUSE_CONFIG.MAP_MIN_X
MAP_MAX_X: Final[float] = WAREHOUSE_CONFIG.MAP_MAX_X
MAP_MIN_Y: Final[float] = WAREHOUSE_CONFIG.MAP_MIN_Y
MAP_MAX_Y: Final[float] = WAREHOUSE_CONFIG.MAP_MAX_Y
GOAL_TOLERANCE: Final[float] = WAREHOUSE_CONFIG.GOAL_TOLERANCE
SHELVES: Final[List[Tuple[float, float, float, float]]] = WAREHOUSE_CONFIG.SHELVES

NUM_DYNAMIC_OBSTACLES: Final[int] = DYNAMIC_OBS_CONFIG.NUM_OBSTACLES
DYNAMIC_OBS_RADIUS: Final[float] = DYNAMIC_OBS_CONFIG.RADIUS
DYNAMIC_OBS_SPEED_MIN: Final[float] = DYNAMIC_OBS_CONFIG.SPEED_MIN
DYNAMIC_OBS_SPEED_MAX: Final[float] = DYNAMIC_OBS_CONFIG.SPEED_MAX
DYNAMIC_OBS_DETECTION_RADIUS: Final[float] = DYNAMIC_OBS_CONFIG.DETECTION_RADIUS

ACTION_DIM: Final[int] = OBS_CONFIG.ACTION_DIM
TOTAL_OBS_DIM: Final[int] = OBS_CONFIG.TOTAL_OBS_DIM

CBF_SAFETY_MARGIN: Final[float] = CBF_CONFIG.SAFETY_MARGIN
CBF_GAMMA: Final[float] = CBF_CONFIG.GAMMA
QP_SLACK_WEIGHT: Final[float] = CBF_CONFIG.SLACK_WEIGHT

TOTAL_TIMESTEPS: Final[int] = PPO_CONFIG.TOTAL_TIMESTEPS

# Ensure output directories exist upon module load
PATH_CONFIG.create_directories()