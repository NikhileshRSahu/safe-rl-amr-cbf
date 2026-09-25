from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from benchmark.beast_classical import AStarORCADD, BeastORCAConfig
from benchmark.orca_geometry import OrcaLine, build_orca_line
from benchmark.shared_warehouse_perception import LastSeenTracker, visible_with_shelves
from benchmark.train_multi_agent_research import DT, ROBOT_R, SHELVES, VMAX
from benchmark.warehouse_interaction_features import compute_ttc_cpa


@dataclass
class AdaptiveORCAConfig:
    horizon_min: float = 1.0
    horizon_max: float = 4.0
    uncertainty_gain: float = 0.35
    max_uncertainty_extra: float = 0.40
    base_uncertainty: float = 0.02
    acceleration_scale: float = 0.20
    heading_rate_scale: float = 0.12
    density_scale: float = 0.06
    yield_relief_ticks: float = 30.0
    min_human_responsibility: float = 0.75
    occlusion_memory_seconds: float = 1.5
    occlusion_uncertainty_rate: float = 0.25


def adaptive_time_horizon(
    *,
    ttc: float,
    uncertainty: float,
    density: int,
    yield_streak: int,
    config: AdaptiveORCAConfig,
) -> float:
    """Choose a bounded prediction horizon from causal interaction evidence."""
    lo = float(config.horizon_min)
    hi = float(config.horizon_max)
    if not (0.0 < lo <= hi):
        raise ValueError("invalid adaptive horizon bounds")
    uncertainty = max(0.0, float(uncertainty))
    ttc = max(0.0, float(ttc))
    density = max(0, int(density))
    yield_streak = max(0, int(yield_streak))

    confidence = math.exp(-uncertainty)
    urgency = math.exp(-ttc / 2.0)
    density_relief = 1.0 / (1.0 + config.density_scale * density)
    yield_relief = 1.0 / (1.0 + yield_streak / max(config.yield_relief_ticks, 1e-6))
    fraction = confidence * (0.35 + 0.65 * urgency) * density_relief * yield_relief
    return float(np.clip(lo + (hi - lo) * fraction, lo, hi))


def uncertainty_inflation(
    physical_radius: float,
    *,
    sigma: float,
    gain: float,
    max_extra: float,
) -> float:
    physical_radius = max(0.0, float(physical_radius))
    sigma = max(0.0, float(sigma))
    extra = min(max(0.0, float(max_extra)), max(0.0, float(gain)) * sigma)
    return physical_radius + extra


def human_responsibility(observed_yield_probability: float, min_robot_share: float = 0.75) -> float:
    """Robot share of avoidance responsibility for a human interaction."""
    p = float(np.clip(observed_yield_probability, 0.0, 1.0))
    lo = float(np.clip(min_robot_share, 0.5, 1.0))
    return float(np.clip(1.0 - (1.0 - lo) * p, lo, 1.0))


def _heading_rate(previous: np.ndarray, current: np.ndarray, dt: float) -> float:
    ps = float(np.linalg.norm(previous))
    cs = float(np.linalg.norm(current))
    if ps < 1e-8 or cs < 1e-8 or dt <= 0.0:
        return 0.0
    a = math.atan2(float(previous[1]), float(previous[0]))
    b = math.atan2(float(current[1]), float(current[0]))
    delta = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return delta / dt


class AStarAdaptivePredictiveORCADD(AStarORCADD):
    """Strong causal ORCA baseline with uncertainty-aware human prediction.

    AP-ORCA receives the same shelf-occluded human visibility contract as the
    RL navigator. During brief occlusion it may extrapolate only previously
    observed tracks for a bounded time window; hidden future state is never read.
    """

    def __init__(
        self,
        world,
        i: int,
        config: BeastORCAConfig | None = None,
        adaptive_config: AdaptiveORCAConfig | None = None,
    ):
        super().__init__(world, i, config)
        self.adaptive_cfg = adaptive_config or AdaptiveORCAConfig()
        self._prev_human_velocity: dict[int, np.ndarray] = {}
        self._human_yield_probability: dict[int, float] = {}
        self._human_tracks = LastSeenTracker(self.adaptive_cfg.occlusion_memory_seconds)
        self._diag.update(
            adaptive_horizon_sum=0.0,
            adaptive_horizon_count=0,
            uncertainty_inflation_sum=0.0,
            uncertainty_inflation_max=0.0,
            visible_human_constraints=0,
            occluded_track_constraints=0,
        )

    def _human_motion_estimate(self, j: int, velocity: np.ndarray):
        current = np.asarray(velocity, dtype=float)
        previous = self._prev_human_velocity.get(j)
        self._prev_human_velocity[j] = current.copy()
        if previous is None:
            acceleration = np.zeros(2, dtype=float)
            heading_rate = 0.0
        else:
            acceleration = (current - previous) / max(DT, 1e-9)
            heading_rate = _heading_rate(previous, current, DT)
        uncertainty = (
            self.adaptive_cfg.base_uncertainty
            + self.adaptive_cfg.acceleration_scale * float(np.linalg.norm(acceleration))
            + self.adaptive_cfg.heading_rate_scale * abs(float(heading_rate))
        )
        return acceleration, heading_rate, float(uncertainty)

    def _absolute_orca_lines(self, w, current_vel):
        i = self.i
        p = np.asarray(w.p[i], dtype=float)
        lines: list[OrcaLine] = []

        for j in range(w.n):
            if j == i or w.done[j]:
                continue
            q = np.asarray(w.p[j], dtype=float)
            delta = q - p
            if float(np.linalg.norm(delta)) > self.cfg.neighbor_distance:
                continue
            other = self._velocity(w, j)
            rel = other - current_vel
            responsibility = self._peer_responsibility(w, j)
            relative_line = build_orca_line(
                delta,
                rel,
                2 * ROBOT_R + self.cfg.peer_margin,
                self.cfg.time_horizon,
                responsibility,
            )
            lines.append(OrcaLine(point=other - relative_line.point, normal=-relative_line.normal))

        now = float(w.steps * DT)
        visible_ids = []
        for j in range(w.nppl):
            q = np.asarray(w.hp[j], dtype=float)
            if visible_with_shelves(
                p,
                q,
                SHELVES,
                max_range=self.cfg.neighbor_distance,
                shelf_padding=0.02,
            ):
                visible_ids.append(j)
        nearby_humans = len(visible_ids)

        for j in range(w.nppl):
            true_q = np.asarray(w.hp[j], dtype=float)
            visible = j in visible_ids
            if visible:
                observed_position = true_q
                observed_velocity = np.asarray(w.hv[j], dtype=float)
                self._human_tracks.observe(j, observed_position, observed_velocity, now)
                acceleration, _, uncertainty = self._human_motion_estimate(j, observed_velocity)
                occlusion_age = 0.0
                self._diag["visible_human_constraints"] += 1
            else:
                prediction = self._human_tracks.predict(j, now)
                if prediction is None:
                    continue
                observed_position, observed_velocity, occlusion_age = prediction
                observed_position = np.asarray(observed_position, dtype=float)
                observed_velocity = np.asarray(observed_velocity, dtype=float)
                acceleration = np.zeros(2, dtype=float)
                uncertainty = (
                    self.adaptive_cfg.base_uncertainty
                    + self.adaptive_cfg.occlusion_uncertainty_rate * float(occlusion_age)
                )
                self._diag["occluded_track_constraints"] += 1

            delta = observed_position - p
            if float(np.linalg.norm(delta)) > self.cfg.neighbor_distance:
                continue
            rel = observed_velocity - current_vel
            ttc, _ = compute_ttc_cpa(delta, rel, horizon=self.adaptive_cfg.horizon_max)
            horizon = adaptive_time_horizon(
                ttc=ttc,
                uncertainty=uncertainty,
                density=nearby_humans,
                yield_streak=self.stuck,
                config=self.adaptive_cfg,
            )

            predicted_velocity = observed_velocity + 0.5 * acceleration * min(horizon, 1.0)
            speed = float(np.linalg.norm(predicted_velocity))
            max_human_speed = 1.5
            if speed > max_human_speed:
                predicted_velocity *= max_human_speed / speed

            base_radius = 2 * ROBOT_R + self.cfg.human_margin
            effective_radius = uncertainty_inflation(
                base_radius,
                sigma=uncertainty,
                gain=self.adaptive_cfg.uncertainty_gain,
                max_extra=self.adaptive_cfg.max_uncertainty_extra,
            )
            extra = effective_radius - base_radius
            yield_p = self._human_yield_probability.get(j, 0.0)
            responsibility = human_responsibility(
                yield_p, self.adaptive_cfg.min_human_responsibility
            )
            relative_line = build_orca_line(
                delta,
                predicted_velocity - current_vel,
                effective_radius,
                horizon,
                responsibility,
            )
            lines.append(
                OrcaLine(
                    point=predicted_velocity - relative_line.point,
                    normal=-relative_line.normal,
                )
            )
            self._diag["adaptive_horizon_sum"] += horizon
            self._diag["adaptive_horizon_count"] += 1
            self._diag["uncertainty_inflation_sum"] += extra
            self._diag["uncertainty_inflation_max"] = max(
                self._diag["uncertainty_inflation_max"], extra
            )

        self._diag["orca_constraints_total"] += len(lines)
        return lines

    def diagnostics(self):
        out = super().diagnostics()
        n = max(1, int(out.get("adaptive_horizon_count", 0)))
        out["adaptive_horizon_mean"] = float(out.get("adaptive_horizon_sum", 0.0)) / n
        return out
