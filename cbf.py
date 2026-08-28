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
      time rather than failing silently or crashing training mid-run. The formulation strictly bounds worst-case erosion of the safety margin.

A bounded slack variable is implemented in both the `scipy` and `cvxpy` backends 
(when `enable_slack=True`). When slack is exhausted or the QP is irrecoverably 
infeasible (e.g. robot completely boxed in), the filter correctly falls back to a 
hard-stop (0.0, 0.0).
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


def recommended_h_activation_threshold(
    v_max: float, safety_margin: float, safe_radius: float, reaction_horizons: float = 3.0
) -> float:
    """Derives a physically-grounded ``h_activation_threshold`` instead of
    an arbitrary magic number.

    ``h`` in this project is defined as squared-distance-minus-squared-
    safe-radius (see ``compute_barrier_value``), so it has units of m^2,
    not m -- the raw value is not directly interpretable as "meters of
    clearance". This helper instead reasons in physical clearance and
    converts.

    A constraint only needs to be handed to the QP once the robot could
    plausibly reach the barrier within the next few control steps at
    ``v_max``; there is no benefit to activating it further out, only
    solver overhead and (if h's fall-off is steep near the boundary,
    which it isn't here since it's exactly quadratic) no accuracy cost
    either way. We pick the activation *clearance* (in meters, beyond
    ``safe_radius``) as ``reaction_horizons`` control steps of travel at
    ``v_max``, then convert that clearance to the h-threshold via
    ``h = (safe_radius + clearance)^2 - safe_radius^2``.

    Args:
        v_max: Maximum linear velocity (m/s).
        safety_margin: CBF safety margin (m) -- unused directly here but
            accepted for call-site symmetry with ``recommended_slack_max``
            and to make the physical reasoning explicit at call sites.
        safe_radius: Combined collision radius already used to compute
            ``h`` (e.g. robot_radius + obstacle_radius + safety_margin).
        reaction_horizons: Number of ``config.DT``-length control steps of
            travel-at-``v_max`` worth of clearance to activate on.
            Default 3 is a conservative but not excessive lookout window
            for a 10 Hz loop (0.3 s of travel).

    Returns:
        A threshold in the same (m^2) units as ``compute_barrier_value``,
        traceable to a concrete "how many control steps of warning" choice
        instead of an unexplained constant.
    """
    from config import DT

    clearance = reaction_horizons * DT * v_max
    return (safe_radius + clearance) ** 2 - safe_radius ** 2


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
    enable_circulation: bool = False
    circulation_threshold: float = 0.3
    circulation_gain: float = 0.6
    slack_penalty_weight: float = 1e5

    # --- Selective hard escape for feasible CBF deadlocks -----------------
    # Activates only when the strict QP succeeds but returns an almost-zero
    # control even though the nominal policy still wants to move and the robot
    # is close to an active threat.  The escape is a HARD tangential velocity
    # inequality, tested on both circulation directions; zero input therefore
    # cannot remain the optimizer while the escape row is active.
    enable_hard_escape: bool = True
    deadlock_v_threshold: float = 0.03
    deadlock_omega_threshold: float = 0.08
    deadlock_actual_v_threshold: float = 0.05
    deadlock_actual_omega_threshold: float = 0.10
    deadlock_nominal_effort_threshold: float = 0.20
    escape_activation_clearance: float = 0.20
    escape_tangent_speed: float = 0.12

    # --- Lookahead-point offset (m). See compute_lie_derivatives() in
    # utils.py and the "Underactuation fix" section of this module's
    # docstring. THIS FIELD WAS PREVIOUSLY UNWIRED: config.py has carried
    # CBFConfig.LOOKAHEAD_DISTANCE=0.25 since the lookahead-point math was
    # added, but no call site ever read it, so every compute_lie_derivatives
    # call ran with the implicit default lookahead_distance=0.0. At L=0 the
    # barrier is evaluated at the robot's rotation centre, whose velocity
    # Jacobian is  d(x,y)/d(v,omega) = [[cos theta, 0], [sin theta, 0]]  --
    # the omega column is IDENTICALLY ZERO, because spinning in place does
    # not move a point located at the centre of rotation. Consequently
    # Lg_h[1] == 0 for every constraint the filter has ever built, and the
    # QP objective 0.5*||u - u_nom||^2 subject to A@u<=b degenerates: the
    # only column of A that can ever be nonzero is the v-column, so the
    # solver's only lever for satisfying any barrier is to brake (reduce or
    # zero v). It is structurally unable to steer around an obstacle. This
    # is consistent with, and is the most likely root cause of, the
    # observed 87% flat CBF intervention rate that never decreases over
    # 120k training steps: no matter how good the policy's steering
    # becomes, every close approach gets braked identically rather than
    # routed around.
    #
    # Setting lookahead_distance = L > 0 evaluates h at a point L ahead of
    # the robot along its heading, p_L = (x + L cos theta, y + L sin theta).
    # Its velocity Jacobian is [[cos theta, -L sin theta], [sin theta,
    # L cos theta]], which is invertible for any L > 0 -- omega now has
    # first-order authority over the barrier, so the QP can choose to turn
    # instead of only slowing down. This is the standard "unicycle -> fully
    # actuated via output offset" trick (see e.g. Wang & Ames, "Safety
    # Barrier Certificates for Collision-Free Multirobot Systems", 2017;
    # Notomista et al. and follow-on unicycle-CBF literature use the same
    # device). L should be a modest fraction of robot_radius: too small
    # reintroduces the near-singular, v-only behavior; too large evaluates
    # safety at a point far enough from the robot's own body that the QP
    # effectively protects a phantom robot of a different shape/position.
    # Default below is 0.5 * a typical robot_radius (0.30 m) = 0.15 m; the
    # project's own config.py value (0.25 m, ~0.83 * robot_radius) is also
    # reasonable and is what build_filter_from_config() wires in by default
    # now that this field exists. A value of exactly 0.0 is still accepted
    # (recovers the historical, position-only, brake-but-cannot-steer
    # behavior) for anyone who needs to A/B against the old filter.
    lookahead_distance: float = 0.0

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

    def _lookahead_point(self, robot_state: Sequence[float]) -> Tuple[float, float]:
        """The point p_L = (rx + L*cos(theta), ry + L*sin(theta)) that h and
        its Lie derivatives must BOTH be evaluated at. Previously h was
        computed at the robot center while compute_lie_derivatives computed
        the gradient at p_L -- two different points, which is not the
        derivative of the function actually being constrained. See
        README/handoff notes on the lookahead-point CBF design."""
        L = self.config.lookahead_distance
        rx, ry, theta = robot_state[0], robot_state[1], robot_state[2]
        return rx + L * math.cos(theta), ry + L * math.sin(theta)

    def _dynamic_obstacle_constraints(
        self, robot_state: Sequence[float], dynamic_obstacles: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """Builds one linear CBF constraint per active dynamic obstacle.

        This is a genuine time-varying CBF: h and Lf_h/Lg_h are evaluated
        at the obstacle's CURRENT position, with the obstacle's velocity
        entering only through the Lf_h drift term (as compute_lie_derivatives
        already does). Earlier this function instead evaluated h at a
        constant-velocity-PREDICTED future obstacle position
        (pred_horizon seconds ahead) -- which is not an interval collision
        guarantee, and empirically made things worse as pred_horizon grew
        (a longer horizon can put the "ghost" prediction on the far side of
        the robot, so the CBF reacts to a position the obstacle isn't
        actually at yet, rather than the one it must pass through first).
        pred_horizon is no longer used to move the obstacle; only the
        current, true relative geometry defines the barrier.

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
        pLx, pLy = self._lookahead_point(robot_state)

        for obs in dynamic_obstacles:
            safe_radius = (
                cfg.robot_radius + cfg.dynamic_obs_radius + cfg.safety_margin
                + cfg.tracking_uncertainty + cfg.lookahead_distance
            )
            h = compute_barrier_value((pLx, pLy), obs[:2], safe_radius)
            if h > cfg.h_activation_threshold:
                continue

            Lf_h, Lg_h = compute_lie_derivatives(robot_state, obs[:4], cfg.lookahead_distance)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            constraints.append((A_i[0], b_i))

        return constraints

    def _static_shelf_constraints(
        self, robot_state: Sequence[float], shelves: Sequence[RectBounds]
    ) -> List[Tuple[np.ndarray, float]]:
        """Builds one linear CBF constraint per nearby static shelf.
        (ABLATION: reverted to original robot-center h, no L inflation --
        testing whether the p_L-consistency fix or the dynamic-obstacle
        fix is responsible for the timeout blowup.)
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
            Lf_h, Lg_h = compute_lie_derivatives(robot_state, dummy_static_obs, cfg.lookahead_distance)
            A_i, b_i = control_barrier_constraint(Lf_h, Lg_h, h, cfg.gamma)
            constraints.append((A_i[0], b_i))

        return constraints

    def _wall_constraints(self, robot_state: Sequence[float]) -> List[Tuple[np.ndarray, float]]:
        """Builds one linear CBF constraint per nearby map boundary wall.
        (ABLATION: reverted, see _static_shelf_constraints note above.)
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
            Lf_h, Lg_h = compute_lie_derivatives(robot_state, dummy_static_obs, cfg.lookahead_distance)
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
    # Actor-consistency regularizer (differentiable, solver-free)
    # ------------------------------------------------------------------ #

    def actor_consistency_loss(
        self,
        robot_state_batch: np.ndarray,
        dynamic_obstacles_batch: np.ndarray,
        shelves: Sequence[RectBounds],
        action_batch: "torch.Tensor",
        action_low: Sequence[float],
        action_high: Sequence[float],
    ) -> "torch.Tensor":
        """Vectorized, solver-free CBF actor-consistency regularizer.

        See the module-level discussion in the FIXES_README (or the
        previous, per-sample-Python-loop version of this method in git
        history) for the full derivation of *why* this term exists: it
        pulls the actor's raw action samples toward the CBF-feasible
        half-spaces using the exact same constraint algebra as the QP
        (:meth:`active_barrier_constraints`), without ever differentiating
        the solver.

        This version replaces that method's ``for i in range(batch_size)``
        Python loop -- which was correct but became the dominant cost in a
        3000-step stress test with continuous updates -- with a single
        batched computation. Every constraint *type* here (dynamic
        obstacles, shelves, walls) has a **fixed** count per environment
        (``N`` obstacles, ``len(shelves)`` shelves, 4 walls), so instead of
        building a ragged, variably-sized constraint list per sample and
        looping over it, we compute a dense
        ``(B, N + len(shelves) + 4)`` constraint tensor for the WHOLE
        batch via NumPy broadcasting, and turn "this constraint wasn't
        active for this sample" into a boolean mask instead of a Python
        ``continue``. The masked-out entries still cost a few flops (no
        early exit), which is exactly the right trade on this hardware:
        NumPy/Torch vector ops on a fixed-size dense array are far cheaper
        than a Python-level loop of the same total operation count.

        Numerically **identical** to the per-sample version for every
        constraint that WAS active in both (same formulas, just batched);
        the only behavioral difference is that inactive-constraint slots
        contribute exactly 0 to the loss via the mask rather than being
        skipped, which is mathematically the same thing.

        Args / Returns: identical contract to the previous
        ``actor_consistency_loss`` (see git history / FIXES_README) --
        this is a drop-in replacement, not a new API.
        """
        import torch

        cfg = self.config
        device = action_batch.device
        dtype = action_batch.dtype

        robot_state_batch = np.asarray(robot_state_batch, dtype=np.float64)
        dynamic_obstacles_batch = np.asarray(dynamic_obstacles_batch, dtype=np.float64)
        B = robot_state_batch.shape[0]

        rx = robot_state_batch[:, 0]            # (B,)
        ry = robot_state_batch[:, 1]             # (B,)
        theta = robot_state_batch[:, 2]          # (B,)
        cos_t = np.cos(theta)                    # (B,)
        sin_t = np.sin(theta)                    # (B,)
        L = float(cfg.lookahead_distance)
        pLx = rx + L * cos_t                     # (B,)
        pLy = ry + L * sin_t                     # (B,)

        A_parts: List[np.ndarray] = []   # each (B, k_i, 2)
        b_parts: List[np.ndarray] = []   # each (B, k_i)
        mask_parts: List[np.ndarray] = []  # each (B, k_i)

        def _finish(dh_dx: np.ndarray, dh_dy: np.ndarray, h: np.ndarray, ox: np.ndarray, oy: np.ndarray,
                    vox: np.ndarray, voy: np.ndarray) -> None:
            """Shared tail: turns (dh_dx, dh_dy, h, obstacle velocity) of
            shape (B, k) into (A, b, mask) of shape (B, k, 2)/(B, k)/(B, k),
            exactly mirroring compute_lie_derivatives + control_barrier_constraint
            + the h_activation_threshold check, batched.

            NOTE on a subtlety this must reproduce exactly: for STATIC
            shelves/walls (not dynamic obstacles, see below), the per-sample
            code (``_static_shelf_constraints``, ``_wall_constraints``)
            computes the barrier value ``h`` from the robot's *actual
            center* (``robot_xy``), while ``compute_lie_derivatives``
            internally evaluates the *gradient* at the lookahead point
            ``p_L``. So for those two constraint types, ``h`` and
            ``(dh_dx, dh_dy)`` are NOT evaluated at the same point when
            ``lookahead_distance > 0``. This was tried as a bug and fixed
            once already (making h and its gradient consistent at p_L) --
            but that fix, combined with inflating the safe radius by L,
            empirically made timeouts far worse (see the ABLATION notes on
            ``_static_shelf_constraints``/``_wall_constraints``), so it was
            reverted for shelves/walls specifically and kept only for
            dynamic obstacles (see below). Callers of ``_finish`` for
            shelves/walls therefore still pass ``h`` computed at
            ``(rx, ry)`` but ``dh_dx``/``dh_dy`` computed at ``(pLx, pLy)``
            -- do not "fix" this for shelves/walls without also changing
            the per-sample QP path and re-validating against eval episodes,
            or the two will silently diverge again.
            """
            Lf_h = dh_dx * (-vox) + dh_dy * (-voy)                       # (B, k)
            Lg_h_v = dh_dx * cos_t[:, None] + dh_dy * sin_t[:, None]      # (B, k)
            Lg_h_w = dh_dx * (-L * sin_t[:, None]) + dh_dy * (L * cos_t[:, None])  # (B, k)

            # Singular guard, vectorized version of compute_lie_derivatives's
            # "gradient vanished, force a v-stop" branch.
            grad_norm = np.hypot(Lg_h_v, Lg_h_w)
            singular = grad_norm < 1e-8
            Lg_h_v = np.where(singular, 1.0, Lg_h_v)
            Lg_h_w = np.where(singular, 0.0, Lg_h_w)
            Lf_h = np.where(singular, -1e3, Lf_h)

            b = Lf_h + cfg.gamma * h                                     # A@u <= b, A = -Lg_h
            A = -np.stack([Lg_h_v, Lg_h_w], axis=-1)                     # (B, k, 2)
            active = h <= cfg.h_activation_threshold                     # (B, k)

            A_parts.append(A)
            b_parts.append(b)
            mask_parts.append(active)

        # ---- Dynamic obstacles ------------------------------------------- #
        if dynamic_obstacles_batch.shape[1] > 0:
            ox = dynamic_obstacles_batch[:, :, 0]      # (B, N) -- current position,
            oy = dynamic_obstacles_batch[:, :, 1]      # no longer translated by pred_horizon
            o_theta = dynamic_obstacles_batch[:, :, 2]
            o_speed = dynamic_obstacles_batch[:, :, 3]

            vox = o_speed * np.cos(o_theta)
            voy = o_speed * np.sin(o_theta)
            # NOTE (fix applied, matches _dynamic_obstacle_constraints):
            # previously this translated the obstacle position forward by
            # pred_horizon before computing h -- a "ghost obstacle" position
            # that is not an interval collision guarantee and empirically
            # made the live filter WORSE as pred_horizon grew (a longer
            # horizon can put the predicted point on the far side of the
            # robot). Now h and its gradient are both evaluated at the
            # obstacle's true CURRENT position, with velocity entering only
            # through Lf_h (a genuine time-varying CBF), and h is evaluated
            # at the SAME lookahead point p_L that the gradient uses (see
            # _lookahead_point / _dynamic_obstacle_constraints) instead of
            # the robot center -- fixing the point-mismatch this function's
            # docstring used to warn about, for the dynamic-obstacle case
            # specifically. (Static shelves/walls below are intentionally
            # left as-is: the same point-consistency fix was tried there
            # too and empirically made timeouts much worse -- see cbf.py's
            # _static_shelf_constraints/_wall_constraints ABLATION notes --
            # so this function must keep matching THAT decision, not a
            # theoretically "more correct" one, per this function's own
            # parity requirement with the per-sample QP path.)
            safe_radius = (
                cfg.robot_radius + cfg.dynamic_obs_radius + cfg.safety_margin
                + cfg.tracking_uncertainty + cfg.lookahead_distance
            )
            dh_dx = 2.0 * (pLx[:, None] - ox)
            dh_dy = 2.0 * (pLy[:, None] - oy)
            h = (pLx[:, None] - ox) ** 2 + (pLy[:, None] - oy) ** 2 - safe_radius ** 2
            _finish(dh_dx, dh_dy, h, ox, oy, vox, voy)

        # ---- Static shelves (fixed set, same for every sample) ----------- #
        if len(shelves) > 0:
            rects = np.asarray(shelves, dtype=np.float64)   # (S, 4): xmin,ymin,xmax,ymax
            cx = np.clip(rx[:, None], rects[None, :, 0], rects[None, :, 2])  # (B, S)
            cy = np.clip(ry[:, None], rects[None, :, 1], rects[None, :, 3])
            safe_radius_static = cfg.robot_radius + cfg.safety_margin
            dh_dx = 2.0 * (pLx[:, None] - cx)
            dh_dy = 2.0 * (pLy[:, None] - cy)
            h = (rx[:, None] - cx) ** 2 + (ry[:, None] - cy) ** 2 - safe_radius_static ** 2
            zeros = np.zeros_like(cx)
            _finish(dh_dx, dh_dy, h, cx, cy, zeros, zeros)  # static: obstacle velocity = 0

        # ---- Walls (fixed 4 per sample: -x, +x, -y, +y map boundary) ----- #
        wx = np.stack([
            np.full(B, self.map_min_x), np.full(B, self.map_max_x), rx, rx,
        ], axis=-1)  # (B, 4)
        wy = np.stack([
            ry, ry, np.full(B, self.map_min_y), np.full(B, self.map_max_y),
        ], axis=-1)  # (B, 4)
        safe_radius_static = cfg.robot_radius + cfg.safety_margin
        dh_dx = 2.0 * (pLx[:, None] - wx)
        dh_dy = 2.0 * (pLy[:, None] - wy)
        h = (rx[:, None] - wx) ** 2 + (ry[:, None] - wy) ** 2 - safe_radius_static ** 2
        zeros = np.zeros_like(wx)
        _finish(dh_dx, dh_dy, h, wx, wy, zeros, zeros)

        A_all = np.concatenate(A_parts, axis=1)     # (B, C, 2)
        b_all = np.concatenate(b_parts, axis=1)     # (B, C)
        mask_all = np.concatenate(mask_parts, axis=1)  # (B, C) bool

        if not mask_all.any():
            return torch.zeros((), dtype=dtype, device=device)

        # ---- Affine map from normalized action to physical [v, omega] ---- #
        v_lo, v_hi = cfg.v_bounds
        w_lo, w_hi = cfg.omega_bounds
        a_lo = np.asarray(action_low, dtype=np.float64)
        a_hi = np.asarray(action_high, dtype=np.float64)
        scale = np.array(
            [(v_hi - v_lo) / (a_hi[0] - a_lo[0]), (w_hi - w_lo) / (a_hi[1] - a_lo[1])], dtype=np.float64
        )
        offset = np.array([v_lo - a_lo[0] * scale[0], w_lo - a_lo[1] * scale[1]], dtype=np.float64)
        scale_t = torch.as_tensor(scale, dtype=dtype, device=device)
        offset_t = torch.as_tensor(offset, dtype=dtype, device=device)
        u_phys = action_batch * scale_t + offset_t  # (B, 2), differentiable

        A_t = torch.as_tensor(A_all, dtype=dtype, device=device)         # (B, C, 2)
        b_t = torch.as_tensor(b_all, dtype=dtype, device=device)         # (B, C)
        mask_t = torch.as_tensor(mask_all, dtype=dtype, device=device)   # (B, C), 1.0/0.0

        # (B, C, 2) @ (B, 2, 1) -> (B, C, 1) -> (B, C)
        lhs = torch.bmm(A_t, u_phys.unsqueeze(-1)).squeeze(-1)
        violation = torch.relu(lhs - b_t) * mask_t
        # Normalize by batch size B (not total active-constraint count) to
        # exactly match actor_consistency_loss_looped_reference's
        # convention -- verified numerically identical to that reference
        # (which sums per-sample sum-of-squares over that sample's active
        # constraints, then divides by B), and consistent with how
        # actor_loss/critic_loss are already averaged per-sample elsewhere
        # in safe_sac.py. A per-sample sum (not mean) over that sample's
        # own active constraints is intentional: a state with 3 active
        # constraints should contribute proportionally more signal than a
        # state with 1, since satisfying all 3 is a harder requirement.
        per_sample_sum_sq = (violation ** 2).sum(dim=1)  # (B,)
        return per_sample_sum_sq.sum() / A_all.shape[0]

    def actor_consistency_loss_looped_reference(
        self,
        robot_state_batch: np.ndarray,
        dynamic_obstacles_batch: np.ndarray,
        shelves: Sequence[RectBounds],
        action_batch: "torch.Tensor",
        action_low: Sequence[float],
        action_high: Sequence[float],
    ) -> "torch.Tensor":
        """Reference (slow, per-sample Python loop) implementation kept
        ONLY as a numerically-independent cross-check for
        :meth:`actor_consistency_loss` -- it reuses
        :meth:`active_barrier_constraints` directly (the exact same code
        path :meth:`solve` uses), so agreement between this method and the
        vectorized one is strong evidence the vectorization is correct,
        not just fast. Not intended to be called in a training loop; see
        ``tests``/verification output for the comparison. See git history
        for this method's original docstring (identical logic, just
        renamed here).
        """
        import torch

        cfg = self.config
        device = action_batch.device
        dtype = action_batch.dtype
        batch_size = robot_state_batch.shape[0]

        v_lo, v_hi = cfg.v_bounds
        w_lo, w_hi = cfg.omega_bounds
        a_lo = np.asarray(action_low, dtype=np.float64)
        a_hi = np.asarray(action_high, dtype=np.float64)
        scale = np.array(
            [(v_hi - v_lo) / (a_hi[0] - a_lo[0]), (w_hi - w_lo) / (a_hi[1] - a_lo[1])], dtype=np.float64
        )
        offset = np.array([v_lo - a_lo[0] * scale[0], w_lo - a_lo[1] * scale[1]], dtype=np.float64)
        scale_t = torch.as_tensor(scale, dtype=dtype, device=device)
        offset_t = torch.as_tensor(offset, dtype=dtype, device=device)
        u_phys = action_batch * scale_t + offset_t

        losses = []
        for i in range(batch_size):
            A_list, b_list = self.active_barrier_constraints(
                robot_state_batch[i], dynamic_obstacles_batch[i], shelves
            )
            if len(A_list) == 0:
                continue
            A_i = torch.as_tensor(np.stack(A_list, axis=0), dtype=dtype, device=device)
            b_i = torch.as_tensor(np.asarray(b_list), dtype=dtype, device=device)
            violation = torch.relu(A_i @ u_phys[i] - b_i)
            losses.append((violation ** 2).sum())

        if not losses:
            return torch.zeros((), dtype=dtype, device=device)
        return torch.stack(losses).sum() / batch_size


    def _nominal_effort(self, v_nom: float, omega_nom: float) -> float:
        """Dimensionless command effort in the same scale as normalized action."""
        cfg = self.config
        v_scale = max(cfg.v_bounds[1] - cfg.v_bounds[0], 1e-9)
        w_scale = max(max(abs(cfg.omega_bounds[0]), abs(cfg.omega_bounds[1])), 1e-9)
        return float(math.hypot(v_nom / v_scale, omega_nom / w_scale))

    def _is_feasible_deadlock(
        self,
        v_safe: float,
        omega_safe: float,
        v_nom: float,
        omega_nom: float,
        robot_state: Sequence[float],
        num_active_constraints: int,
    ) -> bool:
        """Detect the instantaneous signature of the observed feasible deadlock.

        This deliberately does NOT require a history buffer, so the safety filter
        remains reusable and deterministic.  It triggers only when:
          * at least one CBF row is active,
          * the strict QP succeeded,
          * the returned command is almost zero,
          * the measured actuator state is also almost stationary, and
          * the policy is still asking for meaningful motion.
        """
        cfg = self.config
        if not cfg.enable_hard_escape or num_active_constraints <= 0:
            return False

        v_actual = float(robot_state[3]) if len(robot_state) > 3 else 0.0
        w_actual = float(robot_state[4]) if len(robot_state) > 4 else 0.0

        command_stopped = (
            abs(v_safe) <= cfg.deadlock_v_threshold
            and abs(omega_safe) <= cfg.deadlock_omega_threshold
        )
        plant_stopped = (
            abs(v_actual) <= cfg.deadlock_actual_v_threshold
            and abs(w_actual) <= cfg.deadlock_actual_omega_threshold
        )
        policy_wants_motion = (
            self._nominal_effort(v_nom, omega_nom)
            >= cfg.deadlock_nominal_effort_threshold
        )
        return bool(command_stopped and plant_stopped and policy_wants_motion)

    def _tightest_physical_threat(
        self,
        robot_state: Sequence[float],
        dynamic_obstacles: np.ndarray,
        shelves: Sequence[RectBounds],
    ) -> Optional[Dict[str, Any]]:
        """Return the nearest threat in BUFFERED physical-clearance coordinates.

        The result provides a Cartesian tangent and threat velocity so a hard
        circulation row can be written as

            t^T (J u - v_threat) >= v_tangent_min.

        For static objects v_threat = 0.  For dynamic obstacles the inequality
        is imposed on RELATIVE tangential velocity.
        """
        cfg = self.config
        rx, ry, theta = map(float, robot_state[:3])
        pL = np.asarray(self._lookahead_point(robot_state), dtype=np.float64)
        centre = np.array([rx, ry], dtype=np.float64)
        candidates: List[Dict[str, Any]] = []

        # Dynamic obstacles: match the protected point/radius used by the live
        # time-varying CBF.
        for j, obs in enumerate(np.asarray(dynamic_obstacles, dtype=np.float64)):
            p = np.array([float(obs[0]), float(obs[1])], dtype=np.float64)
            rel = pL - p
            d = float(np.linalg.norm(rel))
            if d < 1e-9:
                continue
            safe_r = (
                cfg.robot_radius + cfg.dynamic_obs_radius + cfg.safety_margin
                + cfg.tracking_uncertainty + cfg.lookahead_distance
            )
            radial = rel / d
            tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
            v_threat = float(obs[3]) * np.array(
                [math.cos(float(obs[2])), math.sin(float(obs[2]))],
                dtype=np.float64,
            )
            candidates.append({
                "kind": "dynamic",
                "index": j,
                "clearance": d - safe_r,
                "tangent": tangent,
                "velocity": v_threat,
            })

        # Shelves: use the empirically validated centre-based buffered geometry.
        safe_static = cfg.robot_radius + cfg.safety_margin
        for j, rect in enumerate(shelves):
            cpnt = np.asarray(closest_point_on_rectangle(centre, rect), dtype=np.float64)
            rel = centre - cpnt
            d = float(np.linalg.norm(rel))
            if d < 1e-9:
                continue
            radial = rel / d
            tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
            candidates.append({
                "kind": "shelf",
                "index": j,
                "clearance": d - safe_static,
                "tangent": tangent,
                "velocity": np.zeros(2, dtype=np.float64),
            })

        # Walls: tangent is exactly parallel to the wall.
        wall_specs = [
            ("wall_xmin", rx - self.map_min_x - safe_static, np.array([0.0, 1.0])),
            ("wall_xmax", self.map_max_x - rx - safe_static, np.array([0.0, 1.0])),
            ("wall_ymin", ry - self.map_min_y - safe_static, np.array([1.0, 0.0])),
            ("wall_ymax", self.map_max_y - ry - safe_static, np.array([1.0, 0.0])),
        ]
        for kind, clearance, tangent in wall_specs:
            candidates.append({
                "kind": kind,
                "index": -1,
                "clearance": float(clearance),
                "tangent": tangent.astype(np.float64),
                "velocity": np.zeros(2, dtype=np.float64),
            })

        if not candidates:
            return None
        return min(candidates, key=lambda x: x["clearance"])

    def _try_hard_escape(
        self,
        v_nom: float,
        omega_nom: float,
        robot_state: Sequence[float],
        dynamic_obstacles: np.ndarray,
        shelves: Sequence[RectBounds],
        A_list: List[np.ndarray],
        b_list: List[float],
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """Try both hard circulation directions and return the best feasible one.

        Unlike the existing soft circulation objective nudge, this appends a
        HARD linear inequality.  While active, u=(0,0) violates the escape row
        whenever escape_tangent_speed > 0, so the zero-input equilibrium is
        removed from the candidate feasible set.
        """
        cfg = self.config
        threat = self._tightest_physical_threat(robot_state, dynamic_obstacles, shelves)
        if threat is None or threat["clearance"] > cfg.escape_activation_clearance:
            return None

        theta = float(robot_state[2])
        L = float(cfg.lookahead_distance)
        c, s = math.cos(theta), math.sin(theta)
        J = np.array([[c, -L * s], [s, L * c]], dtype=np.float64)
        v_threat = np.asarray(threat["velocity"], dtype=np.float64)
        base_tangent = np.asarray(threat["tangent"], dtype=np.float64)

        best: Optional[Tuple[float, float, float, int]] = None
        for side in (+1, -1):
            t = float(side) * base_tangent

            # t^T (J u - v_threat) >= escape_tangent_speed
            # -> -(t^T J) u <= -(escape_tangent_speed + t^T v_threat)
            escape_A = -(t @ J)
            escape_b = -(cfg.escape_tangent_speed + float(t @ v_threat))

            A_try = list(A_list) + [np.asarray(escape_A, dtype=np.float64)]
            b_try = list(b_list) + [float(escape_b)]

            if cfg.backend == "cvxpy":
                v_e, w_e, ok = self._solve_cvxpy(v_nom, omega_nom, A_try, b_try)
            else:
                v_e, w_e, ok = self._solve_scipy(v_nom, omega_nom, A_try, b_try)
            if not ok:
                continue

            # Explicit feasibility check because SLSQP's success flag alone is
            # too permissive for a safety-critical row.
            A_arr = np.asarray(A_try, dtype=np.float64)
            b_arr = np.asarray(b_try, dtype=np.float64)
            u = np.array([v_e, w_e], dtype=np.float64)
            max_violation = float(np.max(A_arr @ u - b_arr))
            if max_violation > 1e-6:
                continue

            cost = float((v_e - v_nom) ** 2 + (w_e - omega_nom) ** 2)
            if best is None or cost < best[0]:
                best = (cost, float(v_e), float(w_e), int(side))

        if best is None:
            return None

        _, v_e, w_e, side = best
        diag = {
            "tier": "hard_escape",
            "deadlock_detected": True,
            "escape_active": True,
            "escape_side": side,
            "escape_threat_kind": threat["kind"],
            "escape_threat_index": threat["index"],
            "escape_buffered_clearance": float(threat["clearance"]),
            "escape_tangent_speed": float(cfg.escape_tangent_speed),
            "solver_success": True,
            "slack_used": 0.0,
            "intervened": abs(v_e-v_nom) > 1e-6 or abs(w_e-omega_nom) > 1e-6,
            "num_active_constraints": len(A_list) + 1,
        }
        return v_e, w_e, diag

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #

    def _sequence_emergency_action(
        self,
        v_nom: float,
        omega_nom: float,
        robot_state: Sequence[float],
        dynamic_obstacles: np.ndarray,
        shelves: Sequence[RectBounds],
        horizon_steps: int = 12,
    ) -> Tuple[float, float, float]:
        """Fast two-phase receding-horizon emergency search.

        Used only when the strict buffered CBF-QP is infeasible.  Unlike the
        older constant-command rollout, this evaluates command *sequences*:
        one control for the first few reaction steps and a second control for
        the rest of the horizon.  Only the first control is executed and the
        search is repeated next environment step.

        Safety is lexicographic: robust physical non-collision first, maximum
        minimum clearance second, nominal-action proximity only after safety.
        This is an engineering recovery layer, not part of the CBF proof.
        """
        from config import DT
        cfg = self.config
        tau = 0.15
        alpha = DT / (tau + DT)

        # Compact action lattice; the nominal components are included so the
        # emergency controller can remain minimally invasive when safe.
        v_vals = np.unique(np.clip(
            np.array([0.0, 0.30, 0.60, 1.0, v_nom], dtype=np.float64),
            cfg.v_bounds[0], cfg.v_bounds[1]
        ))
        w_vals = np.unique(np.clip(
            np.array([cfg.omega_bounds[0], -0.75, 0.0, 0.75, cfg.omega_bounds[1], omega_nom], dtype=np.float64),
            cfg.omega_bounds[0], cfg.omega_bounds[1]
        ))
        controls = np.array([(v, w) for v in v_vals for w in w_vals], dtype=np.float64)
        n = len(controls)

        # Every two-phase sequence (u1,u2).  u1 is executed now; u2 represents
        # the ability to change strategy after the immediate reaction phase.
        u1 = np.repeat(controls, n, axis=0)
        u2 = np.tile(controls, (n, 1))
        m = len(u1)
        horizon_steps = max(horizon_steps, 20)
        switch_step = min(7, max(2, horizon_steps // 3))

        x = np.full(m, float(robot_state[0]), dtype=np.float64)
        y = np.full(m, float(robot_state[1]), dtype=np.float64)
        th = np.full(m, float(robot_state[2]), dtype=np.float64)
        v = np.full(m, float(robot_state[3]), dtype=np.float64)
        w = np.full(m, float(robot_state[4]), dtype=np.float64)

        obs = np.asarray(dynamic_obstacles, dtype=np.float64).copy()

        # Predict the warehouse's social-force obstacle dynamics, not a ghost
        # constant-velocity path.  Obstacles do not react to the ego robot in
        # AMRWarehouseEnv, so this forecast is shared by every candidate.
        # When the simulator would require a random goal/bounce, we conservatively
        # hold the obstacle at its current location for that prediction step.
        obs_forecast = []
        obs_work = obs.copy()
        for _pred_k in range(horizon_steps):
            new_obs = obs_work.copy()
            for i in range(len(obs_work)):
                ox, oy, oth, osp = map(float, obs_work[i, :4])
                if obs_work.shape[1] >= 6:
                    gx, gy = map(float, obs_work[i, 4:6])
                else:
                    gx = ox + math.cos(oth); gy = oy + math.sin(oth)
                dgoal = math.hypot(gx-ox, gy-oy)
                if dgoal < 1.0:
                    # Future goal resampling is stochastic; holding position is
                    # safer than inventing a favorable direction.
                    new_obs[i, 0] = ox; new_obs[i, 1] = oy
                    continue
                dgoal = max(dgoal, 1e-6)
                dirx, diry = (gx-ox)/dgoal, (gy-oy)/dgoal
                desired_speed = 0.8
                relax_time = 0.5
                fgx = (dirx*desired_speed - osp*math.cos(oth))/relax_time
                fgy = (diry*desired_speed - osp*math.sin(oth))/relax_time
                frx = fry = 0.0
                for j in range(len(obs_work)):
                    if i == j: continue
                    dx = ox - float(obs_work[j,0]); dy = oy - float(obs_work[j,1])
                    dd = math.hypot(dx,dy)
                    if 1e-9 < dd < 2.5:
                        force = 2.0*math.exp(-dd/0.5)
                        frx += force*dx/dd; fry += force*dy/dd
                vx_o = osp*math.cos(oth) + (fgx+frx)*DT
                vy_o = osp*math.sin(oth) + (fgy+fry)*DT
                spn = min(0.8, max(0.2, math.hypot(vx_o,vy_o)))
                thn = math.atan2(vy_o,vx_o)
                oxn = ox + spn*math.cos(thn)*DT
                oyn = oy + spn*math.sin(thn)*DT
                # Conservative handling of a predicted shelf/map contact: keep
                # the obstacle at the old location rather than relying on the
                # simulator's random rebound angle.
                bad = (oxn < self.map_min_x + cfg.dynamic_obs_radius or
                       oxn > self.map_max_x - cfg.dynamic_obs_radius or
                       oyn < self.map_min_y + cfg.dynamic_obs_radius or
                       oyn > self.map_max_y - cfg.dynamic_obs_radius)
                if not bad:
                    for rect in shelves:
                        cp = closest_point_on_rectangle((oxn,oyn),rect)
                        if math.hypot(oxn-cp[0], oyn-cp[1]) <= cfg.dynamic_obs_radius:
                            bad=True; break
                if not bad:
                    new_obs[i,0]=oxn; new_obs[i,1]=oyn; new_obs[i,2]=thn; new_obs[i,3]=spn
            obs_work = new_obs
            obs_forecast.append(obs_work[:, :2].copy())

        robust = 0.035
        safe_guard = 0.02

        def clearance_vec(xv: np.ndarray, yv: np.ndarray, obs_positions: np.ndarray) -> np.ndarray:
            c = np.minimum.reduce([
                xv - self.map_min_x,
                self.map_max_x - xv,
                yv - self.map_min_y,
                self.map_max_y - yv,
            ]) - cfg.robot_radius - robust

            for rect in shelves:
                cx = np.clip(xv, rect[0], rect[2])
                cy = np.clip(yv, rect[1], rect[3])
                d = np.hypot(xv - cx, yv - cy) - cfg.robot_radius - robust
                c = np.minimum(c, d)

            for j in range(len(obs_positions)):
                d = np.hypot(xv - obs_positions[j, 0], yv - obs_positions[j, 1])
                d = d - cfg.robot_radius - cfg.dynamic_obs_radius - robust
                c = np.minimum(c, d)
            return c

        min_clear = clearance_vec(x, y, obs[:, :2] if len(obs) else np.empty((0,2), dtype=np.float64))
        for k in range(horizon_steps):
            uc = u1 if k < switch_step else u2
            v += alpha * (uc[:, 0] - v)
            w += alpha * (uc[:, 1] - w)
            x += v * np.cos(th) * DT
            y += v * np.sin(th) * DT
            th += w * DT
            obs_xy_k = obs_forecast[k] if len(obs_forecast) else np.empty((0,2), dtype=np.float64)
            min_clear = np.minimum(min_clear, clearance_vec(x, y, obs_xy_k))

        terminal_obs_xy = obs_forecast[-1] if len(obs_forecast) else np.empty((0,2), dtype=np.float64)
        terminal_clear = clearance_vec(x, y, terminal_obs_xy)
        nominal = np.array([v_nom, omega_nom], dtype=np.float64)
        dev1 = np.sum((u1 - nominal) ** 2, axis=1)
        dev2 = np.sum((u2 - nominal) ** 2, axis=1)
        dev = dev1 + 0.25 * dev2

        robust_safe = min_clear >= safe_guard
        physically_safe = min_clear >= 0.0

        # np.lexsort uses last key as primary.  Choose safety class first,
        # then worst-case clearance, terminal clearance, and finally minimal
        # deviation from nominal.
        safety_class = robust_safe.astype(np.int8) * 2 + (~robust_safe & physically_safe).astype(np.int8)
        order = np.lexsort((-dev, terminal_clear, min_clear, safety_class))
        idx = int(order[-1])
        return float(u1[idx, 0]), float(u1[idx, 1]), float(min_clear[idx])

    def _constant_emergency_candidate(
        self,
        v_nom: float,
        omega_nom: float,
        robot_state: Sequence[float],
        dynamic_obstacles: np.ndarray,
        shelves: Sequence[RectBounds],
        horizon_steps: int = 12,
    ) -> Tuple[float, float, float]:
        """Vectorized historical constant-command recovery candidate."""
        from config import DT
        cfg=self.config
        alpha=DT/(0.15+DT)
        v_vals=np.unique(np.clip(np.array([0.,.15,.30,.50,.70,.90,1.,v_nom]),cfg.v_bounds[0],cfg.v_bounds[1]))
        w_vals=np.unique(np.clip(np.array([cfg.omega_bounds[0],-1.,-.5,0.,.5,1.,cfg.omega_bounds[1],omega_nom]),cfg.omega_bounds[0],cfg.omega_bounds[1]))
        U=np.array([(v,w) for v in v_vals for w in w_vals],dtype=np.float64)
        m=len(U)
        x=np.full(m,float(robot_state[0])); y=np.full(m,float(robot_state[1])); th=np.full(m,float(robot_state[2]))
        v=np.full(m,float(robot_state[3])); w=np.full(m,float(robot_state[4]))
        obs=np.asarray(dynamic_obstacles,dtype=np.float64)
        obsxy=obs[:,:2].copy()
        obsv=np.zeros((len(obs),2),dtype=np.float64)
        if len(obs):
            obsv[:,0]=obs[:,3]*np.cos(obs[:,2]); obsv[:,1]=obs[:,3]*np.sin(obs[:,2])

        def clear_vec(xv,yv,oxy):
            c=np.minimum.reduce([xv-self.map_min_x,self.map_max_x-xv,yv-self.map_min_y,self.map_max_y-yv])-cfg.robot_radius
            for rect in shelves:
                cx=np.clip(xv,rect[0],rect[2]); cy=np.clip(yv,rect[1],rect[3])
                c=np.minimum(c,np.hypot(xv-cx,yv-cy)-cfg.robot_radius)
            for j in range(len(oxy)):
                c=np.minimum(c,np.hypot(oxy[j,0]-xv,oxy[j,1]-yv)-cfg.robot_radius-cfg.dynamic_obs_radius)
            return c

        mn=clear_vec(x,y,obsxy)
        for _ in range(horizon_steps):
            v += alpha*(U[:,0]-v); w += alpha*(U[:,1]-w)
            x += v*np.cos(th)*DT; y += v*np.sin(th)*DT; th += w*DT
            if len(obsxy): obsxy=obsxy+obsv*DT
            mn=np.minimum(mn,clear_vec(x,y,obsxy))
        nom=np.array([v_nom,omega_nom])
        dev=np.sum((U-nom)**2,axis=1)
        order=np.lexsort((dev,-mn))
        i=int(order[0])
        return float(U[i,0]),float(U[i,1]),float(mn[i])

    def _predictive_emergency_action(
        self,
        v_nom: float,
        omega_nom: float,
        robot_state: Sequence[float],
        dynamic_obstacles: np.ndarray,
        shelves: Sequence[RectBounds],
        horizon_steps: int = 12,
    ) -> Tuple[float, float, float]:
        """Risk-gated hybrid emergency recovery.

        The historical constant-action search has better liveness, but the
        200-seed audit showed that all five collisions occurred after its own
        predicted minimum clearance had collapsed to approximately zero or
        negative.  Therefore retain that behavior only while it predicts at
        least 0.12 m of physical headroom; otherwise switch to the safer
        sequence-aware social-force MPC recovery.
        """
        v_fast,w_fast,c_fast = self._constant_emergency_candidate(
            v_nom,omega_nom,robot_state,dynamic_obstacles,shelves,horizon_steps
        )
        if c_fast >= 0.27:
            return v_fast,w_fast,c_fast
        return self._sequence_emergency_action(
            v_nom,omega_nom,robot_state,dynamic_obstacles,shelves,horizon_steps
        )

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
            # The QP can be mathematically feasible yet converge to a useless
            # stationary equilibrium.  Detect exactly that signature and, only
            # then, try a hard tangential escape row on the nearest threat.
            if self._is_feasible_deadlock(
                v_safe, omega_safe, v_nom, omega_nom, robot_state, len(A_list)
            ):
                diagnostics["deadlock_detected"] = True
                escaped = self._try_hard_escape(
                    v_nom, omega_nom, robot_state, dynamic_obstacles, shelves,
                    A_list, b_list,
                )
                if escaped is not None:
                    return escaped

                # If both hard circulation branches are infeasible, use the
                # already-validated short-horizon physical recovery instead of
                # returning the feasible-but-stationary action.
                v_rec, w_rec, rec_clear = self._predictive_emergency_action(
                    v_nom, omega_nom, robot_state, dynamic_obstacles, shelves
                )
                diagnostics["tier"] = "deadlock_predictive_recovery"
                diagnostics["escape_active"] = False
                diagnostics["slack_used"] = None
                diagnostics["recovery_min_clearance"] = rec_clear
                diagnostics["recovery_physically_safe"] = rec_clear >= 0.0
                diagnostics["intervened"] = (
                    abs(v_rec-v_nom) > 1e-6 or abs(w_rec-omega_nom) > 1e-6
                )
                return v_rec, w_rec, diagnostics

            diagnostics["tier"] = "strict"
            diagnostics["slack_used"] = 0.0
            diagnostics["deadlock_detected"] = False
            diagnostics["escape_active"] = False
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

        # The strict buffered QP is genuinely infeasible.  A zero command is
        # not safe when a dynamic obstacle is still closing, so use a short-
        # horizon physical-clearance recovery action instead of blindly stopping.
        v_rec, w_rec, rec_clear = self._predictive_emergency_action(
            v_nom, omega_nom, robot_state, dynamic_obstacles, shelves
        )
        diagnostics["tier"] = "predictive_recovery"
        diagnostics["slack_used"] = None
        diagnostics["recovery_min_clearance"] = rec_clear
        diagnostics["recovery_physically_safe"] = rec_clear >= 0.0
        diagnostics["deadlock_detected"] = False
        diagnostics["escape_active"] = False
        diagnostics["intervened"] = abs(v_rec-v_nom)>1e-6 or abs(w_rec-omega_nom)>1e-6
        return v_rec, w_rec, diagnostics

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
        """SLSQP backend (default, validated). See module docstring.

        Includes a circulation bias against the "spurious equilibria" /
        deadlock failure mode documented for CBF-QPs (Notomista &
        Egerstedt; Grover et al., "The Before, During, and After of
        Multi-Robot Deadlock"; and the circulation-embedded CBF-QP
        literature): a pure min-||u-u_nom||^2 objective can settle at a
        feasible point where the robot has zero net motion (satisfies the
        instantaneous rate constraint by simply not moving) even though a
        different feasible point would actually increase clearance. This
        matters most against MOVING obstacles, where "not moving" does not
        keep the robot safe over time the way it does against a static
        one. When the tightest constraint is nearly binding, we nudge the
        reference point used by the objective (not the hard constraint
        itself -- safety is unaffected) along the tangent of that
        constraint's gradient, so the solver is no longer indifferent
        between "stop" and "slip past" and tends to prefer the latter.
        """
        cfg = self.config
        u_nom = np.array([v_nom, omega_nom])
        A_mat = np.array(A_list)
        b_vec = np.array(b_list)

        u_ref = u_nom
        if cfg.enable_circulation:
            slack_at_nom = b_vec - A_mat @ u_nom
            i_tight = int(np.argmin(slack_at_nom))
            if slack_at_nom[i_tight] < cfg.circulation_threshold:
                a_i = A_mat[i_tight]
                norm = np.linalg.norm(a_i)
                if norm > 1e-8:
                    tangent = np.array([-a_i[1], a_i[0]]) / norm  # rotate 90 deg
                    if np.dot(tangent, u_nom) < 0.0:
                        tangent = -tangent  # keep the side closer to what the policy wanted
                    u_ref = u_nom + cfg.circulation_gain * tangent

        def objective(u: np.ndarray) -> float:
            return 0.5 * np.sum((u - u_ref) ** 2)

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
        LOOKAHEAD_DISTANCE,
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
        # Previously this constant existed in config.py but nothing passed
        # it anywhere -- see the long comment on CBFFilterConfig.
        # lookahead_distance for why that silently disabled the filter's
        # authority over omega. Now actually wired through.
        lookahead_distance=LOOKAHEAD_DISTANCE,
    )
    return CBFSafetyFilter(filter_config, map_bounds=(MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y))