from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class ControllerAggregate:
    agents: int
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    fleet_success_rate: float
    throughput_per_min: float


@dataclass(frozen=True)
class ComparisonResult:
    rl_wins: bool
    safety_parity: bool
    success_delta: float
    collision_delta: float
    timeout_delta: float
    fleet_success_delta: float
    throughput_delta: float


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0,n]")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt((p * (1.0 - p) / n) + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def compare_best_vs_best(
    rl: ControllerAggregate,
    orca: ControllerAggregate,
    *,
    collision_tolerance: float = 0.01,
    min_liveness_delta: float = 1e-12,
) -> ComparisonResult:
    collision_delta = rl.collision_rate - orca.collision_rate
    success_delta = rl.success_rate - orca.success_rate
    timeout_delta = rl.timeout_rate - orca.timeout_rate
    fleet_delta = rl.fleet_success_rate - orca.fleet_success_rate
    throughput_delta = rl.throughput_per_min - orca.throughput_per_min
    safety_parity = collision_delta <= float(collision_tolerance) + 1e-12

    # A claim of RL superiority requires safety parity plus a real liveness
    # improvement. Success/fleet/timeout are primary; throughput can support
    # the claim but cannot compensate for worse completion metrics by itself.
    better_primary_liveness = (
        success_delta > min_liveness_delta
        or fleet_delta > min_liveness_delta
        or timeout_delta < -min_liveness_delta
    )
    no_primary_liveness_regression = (
        success_delta >= -min_liveness_delta
        and fleet_delta >= -min_liveness_delta
        and timeout_delta <= min_liveness_delta
    )
    rl_wins = bool(safety_parity and better_primary_liveness and no_primary_liveness_regression)
    return ComparisonResult(
        rl_wins=rl_wins,
        safety_parity=bool(safety_parity),
        success_delta=float(success_delta),
        collision_delta=float(collision_delta),
        timeout_delta=float(timeout_delta),
        fleet_success_delta=float(fleet_delta),
        throughput_delta=float(throughput_delta),
    )


def select_validation_candidate(candidates: Mapping[str, ControllerAggregate]) -> str:
    """Safety-first frozen checkpoint/config selection on validation only.

    Ordering is lexicographic and fixed before experiments:
      1. lower collision rate
      2. higher fleet success
      3. lower timeout rate
      4. higher individual success
      5. higher throughput
      6. stable candidate name for deterministic ties
    """
    if not candidates:
        raise ValueError("no candidates")
    return min(
        candidates,
        key=lambda name: (
            float(candidates[name].collision_rate),
            -float(candidates[name].fleet_success_rate),
            float(candidates[name].timeout_rate),
            -float(candidates[name].success_rate),
            -float(candidates[name].throughput_per_min),
            str(name),
        ),
    )
