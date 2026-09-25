from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from benchmark.human_sweep_experiment import SweepHumanWorld


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    family: str
    humans: int
    n_amr: int = 4
    speed_scale: float = 1.0
    randomness_level: str = "baseline"


def scenario_catalog() -> tuple[ScenarioSpec, ...]:
    """Warehouse-relevant scenario families used before final holdout.

    Names describe the interaction condition. The current base world is used
    for deterministic pairing; scenario-specific motion generators can extend
    these specs without changing the seed/fairness contract.
    """
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


def split_seed_sets(*, frozen: bool):
    """Return disjoint development/validation seeds and gated final holdout.

    Holdout numbers are deliberately inaccessible through this API until the
    caller declares both controller configurations frozen. They must never be
    used by training, architecture selection or AP-ORCA tuning.
    """
    dev = tuple(range(10100, 10120))
    validation = tuple(range(11100, 11120))
    holdout = tuple(range(13100, 13130)) if frozen else tuple()
    return dev, validation, holdout


def _world_for_spec(spec: ScenarioSpec, seed: int):
    # Same constructor is used for both controllers; this function has no
    # controller argument by design, preventing controller-specific worlds.
    world = SweepHumanWorld(
        spec.n_amr,
        spec.humans,
        int(seed),
        speed_scale=spec.speed_scale,
        randomness_level=spec.randomness_level,
    )
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
