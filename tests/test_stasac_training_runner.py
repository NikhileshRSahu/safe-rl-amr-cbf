import numpy as np
import torch

from benchmark.best_vs_best_protocol import split_seed_sets
from benchmark.spatiotemporal_policy import STASACActor
from benchmark.stasac_training_runner import (
    collect_training_episode,
    training_curriculum,
    training_seed_for_episode,
)
from benchmark.stasac_warehouse import EGO_DIM


def test_training_curriculum_is_human_focused_local_navigation():
    curriculum = training_curriculum()
    assert curriculum
    names = {s.name for s in curriculum}
    assert {
        "human_crossing",
        "blind_shelf_corner",
        "human_hesitation",
        "human_reversal",
        "forklift_crossing",
        "dense_human_flow",
    } <= names
    assert all(1 <= s.n_amr <= 2 for s in curriculum)
    assert sum(s.n_amr == 1 for s in curriculum) >= 5
    assert not any(s.family == "fleet_deadlock" for s in curriculum)


def test_training_seed_schedule_never_leaks_validation_or_holdout():
    dev, validation, holdout = split_seed_sets(frozen=True)
    generated = {training_seed_for_episode(i) for i in range(500)}
    assert generated <= set(dev)
    assert generated.isdisjoint(validation)
    assert generated.isdisjoint(holdout)


def test_short_real_episode_collects_bounded_actor_and_teacher_actions_with_variable_entities():
    torch.manual_seed(17)
    actor = STASACActor(ego_dim=EGO_DIM)
    spec = next(s for s in training_curriculum() if s.name == "dense_human_flow")
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
    assert max(len(t["entities"]) for t in transitions) > 4
    assert all(np.isfinite(t["ego"]).all() for t in transitions)
