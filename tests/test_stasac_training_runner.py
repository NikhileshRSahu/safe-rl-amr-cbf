import numpy as np
import torch

from benchmark.best_vs_best_protocol import split_seed_sets
from benchmark.spatiotemporal_policy import STASACActor
from benchmark.stasac_training_runner import (
    collect_training_episode,
    training_curriculum,
)
from benchmark.stasac_warehouse import EGO_DIM


def test_training_curriculum_uses_only_development_scenarios_and_never_holdout_seeds():
    curriculum = training_curriculum()
    assert curriculum
    assert {s.family for s in curriculum} >= {"intent", "mixed", "density"}
    dev, validation, holdout = split_seed_sets(frozen=True)
    assert set(dev).isdisjoint(validation)
    assert set(dev).isdisjoint(holdout)


def test_short_real_episode_collects_bounded_actor_and_teacher_actions_with_variable_entities():
    torch.manual_seed(17)
    actor = STASACActor(ego_dim=EGO_DIM)
    spec = training_curriculum()[0]
    result = collect_training_episode(
        actor,
        spec,
        seed=10100,
        max_steps=8,
        teacher_mix=0.5,
        use_cbf=True,
    )
    assert result["agent_steps"] > 0
    assert len(result["trajectories"]) == spec.n_amr
    transitions = [x for trajectory in result["trajectories"] for x in trajectory]
    assert transitions
    assert all("teacher_action" in t for t in transitions)
    assert all(np.max(np.abs(t["action"])) <= 1.0 + 1e-6 for t in transitions)
    assert all(np.max(np.abs(t["teacher_action"])) <= 1.0 + 1e-6 for t in transitions)
    assert any(len(t["entities"]) > 4 for t in transitions)
    assert all(np.isfinite(t["ego"]).all() for t in transitions)
