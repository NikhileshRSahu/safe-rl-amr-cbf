"""cbf.py

Standalone Control Barrier Function (CBF) Quadratic-Program safety filter
for the unicycle AMR.

This module extracts the safety-filtering logic that previously lived as a
private method on ``AMRWarehouseEnv`` (``_solve_cbf_qp``) into a
self-contained, reusable class. The reasons for this split:

    1. ``safe_sac.py`` (Hybrid SAC + CBF) needs to call the *exact same*
       filter the environment uses, both during environment.step() and
       potentially during off-policy training-time filtering. A private
       env method can't be shared cleanly across those call sites.
    2. It makes the filter independently unit-testable (see the
       CBF-active edge case discussed in the thesis smoke-test plan)
       without spinning up a full Gymnasium environment.
    3. It matches the architecture diagrams in the thesis document
       (Figures 1, 5, 6, 7) where the "CBF Safety Layer" is drawn as its
       own block, not folded into the environment step.

Solver backends
----------------
Two backends are supported behind the same public interface:

    * ``"scipy"`` (default) -- ``scipy.optimize.minimize`` with SLSQP.
      This is the backend that has been validated against the project's
      smoke tests (env.reset/step/get_state/set_state/render all pass
      with this backend). Kept as the default so nothing that already
      works changes behavior.
    * ``"cvxpy"`` -- solves the QP via CVXPY with the OSQP solver, which
      is what ``config.CBFConfig`` (``QP_SOLVER = "OSQP"``) originally
      specified. This backend is optional and import-guarded: if
      ``cvxpy``/``osqp`` are not installed, constructing a filter with
      ``backend="cvxpy"`` raises a clear ``ImportError`` at construction
      time rather than failing silently or crashing training mid-run.

No slack variable is implemented in either backend (matching the
project's deliberate choice to keep a hard-stop fallback for the
baseline). If the QP is infeasible, ``solve()`` returns ``(0.0, 0.0)``
-- see ``CBFSafetyFilter.solve``'s docstring for the TODO on revisiting
this if infeasible QPs turn out to be frequent with dense dynamic
obstacles.

Public interface
-----------------
    filt = CBFSafetyFilter(...)                      # once, at env init
    v_safe, omega_safe, diag = filt.solve(            # every env.step()
        v_nom, omega_nom, robot_state, dynamic_obstacles, shelves
    )

``diag`` is a small dict of diagnostics (whether the filter intervened,
how many constraints were active, solver success) intended for logging /
thesis plots (e.g. "CBF intervention rate over an episode").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize as opt

from utils import (
    closest_point_on_rectangle,
    compute_barrier_value,
    compute_lie_derivatives,
    control_barrier_constraint,
    squared_distance,
)

try:  # Optional dependency -- see module docstring.
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent.
    cp = None  # type: ignore[assignment]
    _CVXPY_AVAILABLE = False


RectBounds = Tuple[float, float, float, float]

def recommended_slack_max(gamma: float, safety_margin: float, headroom: float = 0.5) -> float:
    """Computes a theoretically bounded slack_max for a given CBF configuration.
    
    The strict CBF constraint is: L_f h + L_g h * u + gamma * h >= 0
    When slack is allowed, the constraint becomes: L_f h + L_g h * u + gamma * h >= -delta
    Under this relaxed constraint, as it stays active, the barrier value h is lower-bounded
    by the equilibrium of the linear ODE h_dot = -gamma * h - delta, which is h >= -delta / gamma.
    
    To ensure the robot's TRUE physical collision radius (robot_radius + obs_radius) is never
    compromised, we must guarantee that this worst-case erosion never exceeds the buffered
    safety_margin. That is: delta / gamma <= safety_margin.
    
    This function sets delta_max (slack_max) to a fraction (headroom) of that bound:
    slack_max = headroom * gamma * safety_margin.
    
    Args:
        gamma: CBF rate constant.
        safety_margin: The buffered margin (meters) around the physical radius.
        headroom: Safety fraction of the total margin to allow eroding (default: 0.5).
        
    Returns:
        A mathematically bounded slack_max value.
    """
    return headroom * gamma * safety_margin

@dataclass
class CBFFilterConfig:
    """Static configuration for :class:`CBFSafetyFilter`.

    Mirrors the relevant fields of ``config.CBFConfig`` plus the
    prediction-horizon / uncertainty-margin parameters that previously
    lived as hardcoded attributes on ``AMRWarehouseEnv``.

    Attributes:
        robot_radius: Collision radius of the ego robot (m).
        dynamic_obs_radius: Collision radius of a dynamic obstacle (m).
        safety_margin: Additional CBF safety buffer added on top of the
            physical radii (m). Corresponds to ``CBF_SAFETY_MARGIN``.
        gamma: Class-K linear gain in the CBF constraint
            ``ḣ + gamma*h >= 0``. Corresponds to ``CBF_GAMMA``.
        pred_horizon: Lookahead time (s) used to linearly predict dynamic
            obstacle positions before building their CBF constraint.
        tracking_uncertainty: Additional radius (m) added to dynamic
            obstacles' safe radius to account for perception/tracking
            uncertainty in the predicted position.
        h_activation_threshold: Constraints are only added to the QP when
            the barrier value ``h`` is below this threshold; barriers far
            above zero cannot become active within one control step and
            are skipped purely as a solver-speed optimization.
        v_bounds: ``(v_min, v_max)`` physical bounds for the QP.
        omega_bounds: ``(omega_min, omega_max)`` physical bounds for the QP.
        backend: ``"scipy"`` (default, SLSQP) or ``"cvxpy"`` (OSQP).
        max_iter: Solver iteration cap.
        infeasible_fallback: Action returned when the solver fails to find
            a feasible point. Defaults to a hard stop ``(0.0, 0.0)``, per
            the project's deliberate no-slack design decision.
    """

    robot_radius: float
    dynamic_obs_radius: float
    safety_margin: float
    gamma: float = 1.0
    pred_horizon: float = 0.5
    tracking_uncertainty: float = 0.10
    h_activation_threshold: float = 2.0
    v_bounds: Tuple[float, float] = (0.0, 1.0)
    omega_bounds: Tuple[float, float] = (-1.5, 1.5)
    backend: str = "scipy"
    max_iter: int = 50
    infeasible_fallback: Tuple[float, float] = (0.0, 0.0)
    enable_slack: bool = False
    slack_max: float = 0.0
    slack_penalty_weight: float = 1e5

    def __post_init__(self) -> None:
        if self.backend not in ("scipy", "cvxpy"):
            raise ValueError(f"Unknown backend: {self.backend!r}. Use 'scipy' or 'cvxpy'.")
        if self.backend == "cvxpy" and not _CVXPY_AVAILABLE:
            raise ImportError(
                "backend='cvxpy' requires the 'cvxpy' and 'osqp' packages, "
                "which are not installed. Install them (`pip install cvxpy osqp`) "
                "or use backend='scipy' (the default, validated backend)."
            )


class CBFSafetyFilter:
    """Minimally-invasive CBF-QP safety filter for a unicycle robot.

    Given a nominal (possibly unsafe) RL-proposed action ``u_nom = [v, ω]``,
    solves

        minimize_u   0.5 * ||u - u_nom||^2
        subject to   Lf h_i(x) + Lg h_i(x)·u + gamma * h_i(x) >= 0   for all i
                      v_min <= u[0] <= v_max
                      omega_min <= u[1] <= omega_max

    over every active barrier ``h_i`` -- dynamic obstacles (predicted
    forward by ``pred_horizon``), static shelves, and map boundary walls
    -- and returns the closest safe action.

    This class holds no simulation state of its own; every quantity it
    needs (robot pose, obstacle positions, shelf geometry, wall bounds) is
    passed into :meth:`solve` each call, so a single filter instance can
    safely be shared across an env, a training-time filter, and unit
    tests without any risk of stale internal state.
    """

    def __init__(self, config: CBFFilterConfig, map_bounds: RectBounds) -> None:
        """Initializes the filter.

        Args:
            config: Filter configuration (radii, margins, gains, bounds).
            map_bounds: ``(xmin, ymin, xmax, ymax)`` warehouse boundary,
                used to build the four wall-barrier constraints.
        """
        self.config = config
        self.map_min_x, self.map_min_y, self.map_max_x, self.map_max_y = map_bounds

    # ------------------------------------------------------------------ #
    # Constraint assembly
    # ------------------------------------------------------------------ #

    def _dynamic_obstacle_constraints(
        self, robot_state: Sequence[float], dynamic_obstacles: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """Builds one linear CBF constraint per active dynamic obstacle.

        Obstacle positions are linearly predicted ``pred_horizon`` seconds
        forward (constant-velocity assumption) before the barrier value is
        computed, and the safe radius is inflated by
        ``tracking_uncertainty`` to account for that prediction's error.

        Args:
            robot_state: ``[x, y, theta, v, omega]`` (or any sequence
                whose first 3 entries are the pose).
            dynamic_obstacles: Array of shape ``[N, >=4]`` where each row
                is ``[x, y, theta, speed, ...]``.

        Returns:
            List of ``(A_row, b)`` pairs, each satisfying ``A_row @ u <= b``.
        """
        cfg = self.config
        constraints: List[Tuple[np.ndarray, float]] = []
        robot_xy = robot_state[:2]

        for obs in dynamic_obstacles:
            pred_obs = np.copy(obs[:4])
            pred_obs[0] += obs[3] * math.cos(obs[2]) * cfg.pred_horizon
            pred_obs[1] += obs[3] * math.sin(obs[2]) * cfg.pred_horizon

            safe_radius = (
                cfg.robot_radius + cfg.dynamic_obs_radius + cfg.safety_margin + cfg.tracking_uncertainty
            )
            h = compute_barrier_value(robot_xy, pred_obs[:2], safe_radius)
            if h > cfg.h_activation_threshold:
                continue

            Lf_h, Lg_h = compute_lie_derivatives(robot_state, pred_obs)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            constraints.append((A_i[0], b_i))

        return constraints

    def _static_shelf_constraints(
        self, robot_state: Sequence[float], shelves: Sequence[RectBounds]
    ) -> List[Tuple[np.ndarray, float]]:
        """Builds one linear CBF constraint per nearby static shelf.

        Args:
            robot_state: ``[x, y, theta, ...]``.
            shelves: Sequence of ``(xmin, ymin, xmax, ymax)`` rectangles.

        Returns:
            List of ``(A_row, b)`` pairs, each satisfying ``A_row @ u <= b``.
        """
        cfg = self.config
        constraints: List[Tuple[np.ndarray, float]] = []
        robot_xy = robot_state[:2]
        safe_radius_static = cfg.robot_radius + cfg.safety_margin

        for rect in shelves:
            closest_p = closest_point_on_rectangle(robot_xy, rect)
            h = squared_distance(robot_xy, closest_p) - safe_radius_static**2
            if h > cfg.h_activation_threshold:
                continue

            dummy_static_obs = [closest_p[0], closest_p[1], 0.0, 0.0]
            Lf_h, Lg_h = compute_lie_derivatives(robot_state, dummy_static_obs)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            constraints.append((A_i[0], b_i))

        return constraints

    def _wall_constraints(self, robot_state: Sequence[float]) -> List[Tuple[np.ndarray, float]]:
        """Builds one linear CBF constraint per nearby map boundary wall.

        Args:
            robot_state: ``[x, y, theta, ...]``.

        Returns:
            List of ``(A_row, b)`` pairs, each satisfying ``A_row @ u <= b``.
        """
        cfg = self.config
        rx, ry = robot_state[0], robot_state[1]
        safe_radius_static = cfg.robot_radius + cfg.safety_margin
        wall_points = [
            (self.map_min_x, ry), (self.map_max_x, ry),
            (rx, self.map_min_y), (rx, self.map_max_y),
        ]

        constraints: List[Tuple[np.ndarray, float]] = []
        for wp in wall_points:
            h = squared_distance((rx, ry), wp) - safe_radius_static**2
            if h > cfg.h_activation_threshold:
                continue

            dummy_static_obs = [wp[0], wp[1], 0.0, 0.0]
            Lf_h, Lg_h = compute_lie_derivatives(robot_state, dummy_static_obs)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            constraints.append((A_i[0], b_i))

        return constraints

    def active_barrier_constraints(
        self,
        robot_state: Sequence[float],
        dynamic_obstacles: np.ndarray,
        shelves: Sequence[RectBounds],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Assembles every active CBF constraint into flat ``(A_list, b_list)``.

        Exposed as a public method (in addition to being used internally by
        :meth:`solve`) so training code or diagnostics can inspect exactly
        which constraints are active at a given state, e.g. for a thesis
        plot of "number of active CBF constraints per episode step."

        Args:
            robot_state: ``[x, y, theta, v, omega]``.
            dynamic_obstacles: Array of shape ``[N, >=4]``.
            shelves: Sequence of ``(xmin, ymin, xmax, ymax)`` rectangles.

        Returns:
            ``(A_list, b_list)`` such that stacking gives ``A @ u <= b``.
        """
        all_constraints = (
            self._dynamic_obstacle_constraints(robot_state, dynamic_obstacles)
            + self._static_shelf_constraints(robot_state, shelves)
            + self._wall_constraints(robot_state)
        )
        A_list = [c[0] for c in all_constraints]
        b_list = [c[1] for c in all_constraints]
        return A_list, b_list

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #

    def solve(
        self,
        v_nom: float,
        omega_nom: float,
        robot_state: Sequence[float],
        dynamic_obstacles: np.ndarray,
        shelves: Sequence[RectBounds],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Filters a nominal action into the closest safe action.

        Args:
            v_nom: Nominal (RL-proposed) linear velocity, physical units.
            omega_nom: Nominal (RL-proposed) angular velocity, physical
                units.
            robot_state: ``[x, y, theta, v, omega]`` current robot state.
            dynamic_obstacles: Array of shape ``[N, >=4]``, rows
                ``[x, y, theta, speed, ...]``.
            shelves: Sequence of ``(xmin, ymin, xmax, ymax)`` rectangles.

        Returns:
            Tuple ``(v_safe, omega_safe, diagnostics)`` where
            ``diagnostics`` contains:
                * ``"intervened"``: whether ``(v_safe, omega_safe)`` differs
                  from ``(v_nom, omega_nom)`` beyond a small numerical
                  tolerance.
                * ``"num_active_constraints"``: number of CBF constraints
                  passed to the solver.
                * ``"solver_success"``: whether the QP solver reported
                  success (always ``True`` when there were zero active
                  constraints, since the nominal action is returned
                  unfiltered in that case).
        """
        A_list, b_list = self.active_barrier_constraints(robot_state, dynamic_obstacles, shelves)

        if len(A_list) == 0:
            diag: Dict[str, Any] = {
                "intervened": False,
                "num_active_constraints": 0,
                "solver_success": True,
            }
            if self.config.enable_slack:
                diag["tier"] = "strict"
                diag["slack_used"] = 0.0
            return v_nom, omega_nom, diag

        if self.config.backend == "cvxpy":
            v_safe, omega_safe, success = self._solve_cvxpy(v_nom, omega_nom, A_list, b_list)
        else:
            v_safe, omega_safe, success = self._solve_scipy(v_nom, omega_nom, A_list, b_list)

        intervened = (abs(v_safe - v_nom) > 1e-6) or (abs(omega_safe - omega_nom) > 1e-6)
        diagnostics = {
            "intervened": intervened,
            "num_active_constraints": len(A_list),
            "solver_success": success,
        }

        if success:
            if self.config.enable_slack:
                diagnostics["tier"] = "strict"
                diagnostics["slack_used"] = 0.0
            return v_safe, omega_safe, diagnostics

        if self.config.enable_slack:
            if self.config.backend == "cvxpy":
                v_safe_s, omega_safe_s, delta_used, success_s = self._solve_cvxpy_slack(v_nom, omega_nom, A_list, b_list)
            else:
                v_safe_s, omega_safe_s, delta_used, success_s = self._solve_scipy_slack(v_nom, omega_nom, A_list, b_list)
            if success_s:
                intervened_s = (abs(v_safe_s - v_nom) > 1e-6) or (abs(omega_safe_s - omega_nom) > 1e-6)
                diagnostics["intervened"] = intervened_s
                diagnostics["solver_success"] = True
                diagnostics["tier"] = "slack"
                diagnostics["slack_used"] = delta_used
                return v_safe_s, omega_safe_s, diagnostics

        if self.config.enable_slack:
            diagnostics["tier"] = "hard_stop"
            diagnostics["slack_used"] = None

        return v_safe, omega_safe, diagnostics

    def _solve_scipy_slack(
        self, v_nom: float, omega_nom: float, A_list: List[np.ndarray], b_list: List[float]
    ) -> Tuple[float, float, float, bool]:
        cfg = self.config
        u_nom = np.array([v_nom, omega_nom])
        A_mat = np.array(A_list)
        b_vec = np.array(b_list)

        # Decision variable: x = [v, omega, delta]
        x0 = np.array([v_nom, omega_nom, 0.0])

        def objective(x: np.ndarray) -> float:
            return 0.5 * np.sum((x[:2] - u_nom) ** 2) + cfg.slack_penalty_weight * (x[2] ** 2)

        def constraint(x: np.ndarray) -> np.ndarray:
            # A_i @ u <= b_i + delta  =>  b_i + delta - A_i @ u >= 0
            return b_vec + x[2] - np.dot(A_mat, x[:2])

        bounds = (cfg.v_bounds, cfg.omega_bounds, (0.0, cfg.slack_max))
        cons = {"type": "ineq", "fun": constraint}

        res = opt.minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": cfg.max_iter, "ftol": 1e-3},
        )

        if res.success:
            return float(res.x[0]), float(res.x[1]), float(res.x[2]), True
        
        fb_v, fb_omega = cfg.infeasible_fallback
        return fb_v, fb_omega, 0.0, False

    def _solve_scipy(
        self, v_nom: float, omega_nom: float, A_list: List[np.ndarray], b_list: List[float]
    ) -> Tuple[float, float, bool]:
        """SLSQP backend (default, validated). See module docstring."""
        cfg = self.config
        u_nom = np.array([v_nom, omega_nom])
        A_mat = np.array(A_list)
        b_vec = np.array(b_list)

        def objective(u: np.ndarray) -> float:
            return 0.5 * np.sum((u - u_nom) ** 2)

        def constraint(u: np.ndarray) -> np.ndarray:
            return b_vec - np.dot(A_mat, u)

        bounds = (cfg.v_bounds, cfg.omega_bounds)
        cons = {"type": "ineq", "fun": constraint}

        res = opt.minimize(
            objective, u_nom, method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": cfg.max_iter, "ftol": 1e-3},
        )

        if res.success:
            return float(res.x[0]), float(res.x[1]), True
        # TODO: hard fallback, no slack variable in the QP -- revisit if
        # infeasible QPs (e.g. robot boxed in by dense dynamic obstacles)
        # turn out to be frequent in practice; a slack-relaxed QP would
        # degrade more gracefully than a full stop.
        fb_v, fb_omega = cfg.infeasible_fallback
        return fb_v, fb_omega, False

    def _solve_cvxpy(
        self, v_nom: float, omega_nom: float, A_list: List[np.ndarray], b_list: List[float]
    ) -> Tuple[float, float, bool]:
        """CVXPY + OSQP backend (optional). See module docstring.

        Raises:
            ImportError: If cvxpy/osqp are not installed. This should be
                unreachable in practice because ``CBFFilterConfig`` already
                validates backend availability at construction time, but
                is kept as a defensive check.
        """
        if not _CVXPY_AVAILABLE:  # pragma: no cover - defensive.
            raise ImportError("cvxpy backend selected but cvxpy is not installed.")

        cfg = self.config
        u_nom = np.array([v_nom, omega_nom])
        A_mat = np.array(A_list)
        b_vec = np.array(b_list)

        u = cp.Variable(2)
        objective = cp.Minimize(0.5 * cp.sum_squares(u - u_nom))
        constraints = [
            A_mat @ u <= b_vec,
            u[0] >= cfg.v_bounds[0], u[0] <= cfg.v_bounds[1],
            u[1] >= cfg.omega_bounds[0], u[1] <= cfg.omega_bounds[1],
        ]
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, max_iter=cfg.max_iter)
        except cp.error.SolverError:
            fb_v, fb_omega = cfg.infeasible_fallback
            return fb_v, fb_omega, False

        if u.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
            fb_v, fb_omega = cfg.infeasible_fallback
            return fb_v, fb_omega, False

        return float(u.value[0]), float(u.value[1]), True

    def _solve_cvxpy_slack(
        self, v_nom: float, omega_nom: float, A_list: List[np.ndarray], b_list: List[float]
    ) -> Tuple[float, float, float, bool]:
        if not _CVXPY_AVAILABLE:  # pragma: no cover - defensive.
            raise ImportError("cvxpy backend selected but cvxpy is not installed.")

        cfg = self.config
        u_nom = np.array([v_nom, omega_nom])
        A_mat = np.array(A_list)
        b_vec = np.array(b_list)

        u = cp.Variable(2)
        delta = cp.Variable(nonneg=True)
        objective = cp.Minimize(0.5 * cp.sum_squares(u - u_nom) + cfg.slack_penalty_weight * cp.square(delta))
        constraints = [
            A_mat @ u <= b_vec + delta,
            delta <= cfg.slack_max,
            u[0] >= cfg.v_bounds[0], u[0] <= cfg.v_bounds[1],
            u[1] >= cfg.omega_bounds[0], u[1] <= cfg.omega_bounds[1],
        ]
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, max_iter=cfg.max_iter)
        except cp.error.SolverError:
            fb_v, fb_omega = cfg.infeasible_fallback
            return fb_v, fb_omega, 0.0, False

        if u.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
            fb_v, fb_omega = cfg.infeasible_fallback
            return fb_v, fb_omega, 0.0, False

        return float(u.value[0]), float(u.value[1]), float(delta.value), True


def build_filter_from_config() -> CBFSafetyFilter:
    """Convenience constructor wiring :class:`CBFSafetyFilter` to ``config.py``.

    Keeps the "which constants come from where" decision in one place
    rather than duplicated at every call site (``environment.py``,
    ``safe_sac.py``, tests).

    Returns:
        A :class:`CBFSafetyFilter` configured from the project's global
        ``config.py`` values, using the default (validated) SLSQP backend.
    """
    from config import (
        CBF_GAMMA,
        CBF_SAFETY_MARGIN,
        DYNAMIC_OBS_RADIUS,
        MAP_MAX_X,
        MAP_MAX_Y,
        MAP_MIN_X,
        MAP_MIN_Y,
        OMEGA_MAX,
        OMEGA_MIN,
        ROBOT_RADIUS,
        V_MAX,
        V_MIN,
    )

    filter_config = CBFFilterConfig(
        robot_radius=ROBOT_RADIUS,
        dynamic_obs_radius=DYNAMIC_OBS_RADIUS,
        safety_margin=CBF_SAFETY_MARGIN,
        gamma=CBF_GAMMA,
        v_bounds=(V_MIN, V_MAX),
        omega_bounds=(OMEGA_MIN, OMEGA_MAX),
        backend="scipy",
    )
    return CBFSafetyFilter(filter_config, map_bounds=(MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y))