from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from benchmark.warehouse_scenario_world import make_scenario_world


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    family: str
    humans: int
    n_amr: int = 4
    speed_scale: float = 1.0
    randomness_level: str = "baseline"


def scenario_catalog() -> tuple[ScenarioSpec, ...]:
    """Legacy broad stress catalog retained for ablations and regression tests."""
    return (
        ScenarioSpec("cross_intersection", "intent", 12, 4, 1.0, "medium"),
        ScenarioSpec("shelf_corner", "occlusion", 12, 4, 1.0, "medium"),
        ScenarioSpec("hesitation", "intent", 12, 4, 1.0, "high"),
        ScenarioSpec("mixed_behavior", "mixed", 12, 4, 1.0, "high"),
        ScenarioSpec("density_06", "density", 6, 4, 1.0, "medium"),
        ScenarioSpec("density_12", "density", 12, 4, 1.0, "medium"),
        ScenarioSpec("density_18", "density", 18, 4, 1.0, "medium"),
        ScenarioSpec("density_24", "density", 24, 4, 1.0, "medium"),
    )


def local_human_navigation_catalog() -> tuple[ScenarioSpec, ...]:
    """Primary warehouse local-navigation benchmark.

    Fleet scheduling/deadlock is intentionally removed from the decisive test.
    The route is assumed to be assigned upstream; the local controller must
    navigate human-driven uncertainty using one AMR in the core scenarios.
    A two-AMR case is retained only as a secondary mixed-traffic stress test.
    """
    return (
        ScenarioSpec("human_crossing", "human_local", 6, 1, 1.0, "medium"),
        ScenarioSpec("blind_shelf_corner", "occlusion", 4, 1, 1.0, "medium"),
        ScenarioSpec("human_hesitation", "human_local", 6, 1, 1.0, "high"),
        ScenarioSpec("human_reversal", "human_local", 6, 1, 1.0, "high"),
        ScenarioSpec("forklift_crossing", "vehicle_like", 2, 1, 1.15, "medium"),
        ScenarioSpec("dense_human_flow", "human_density", 12, 1, 1.0, "high"),
        ScenarioSpec("mixed_local_traffic", "mixed_local", 10, 2, 1.0, "high"),
    )


def development_evaluation_seeds() -> tuple[int, ...]:
    """Development-only screening seeds never used for gradient training."""
    return tuple(range(10300, 10320))


def split_seed_sets(*, frozen: bool):
    """Return disjoint training-development, validation and gated holdout seeds."""
    dev = tuple(range(10100, 10120))
    validation = tuple(range(11100, 11120))
    holdout = tuple(range(13100, 13130)) if frozen else tuple()
    return dev, validation, holdout


def _world_for_spec(spec: ScenarioSpec, seed: int):
    world = make_scenario_world(spec, int(seed))
    world.reset()
    return world


def paired_world_fingerprint(spec: ScenarioSpec, seed: int) -> str:
    """Stable digest of the initial physical realization for pairing tests."""
    world = _world_for_spec(spec, seed)
    payload = {
        "scenario": spec.__dict__,
        "seed": int(seed),
        "p": np.asarray(world.p, dtype=np.float32).round(6).tolist(),
        "g": np.asarray(world.g, dtype=np.float32).round(6).tolist(),
        "th": np.asarray(world.th, dtype=np.float32).round(6).tolist(),
        "hp": np.asarray(world.hp, dtype=np.float32).round(6).tolist(),
        "hv": np.asarray(world.hv, dtype=np.float32).round(6).tolist(),
        "priority": np.asarray(world.priority, dtype=np.float32).round(6).tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_paired_worlds(spec: ScenarioSpec, seed: int):
    """Independent but bitwise-equivalent initial worlds for two controllers."""
    return _world_for_spec(spec, seed), _world_for_spec(spec, seed)
