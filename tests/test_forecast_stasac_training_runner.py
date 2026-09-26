import torch

from benchmark.best_vs_best_protocol import local_human_navigation_catalog
from benchmark.human_forecaster import GRUHumanForecaster
from benchmark.spatiotemporal_policy import STASACActor
from benchmark.stasac_training_runner import collect_training_episode, load_frozen_forecaster, training_seed_for_episode
from benchmark.stasac_warehouse import EGO_DIM, FORECAST_ENTITY_DIM
from benchmark.train_human_forecaster import save_forecaster_checkpoint


def test_forecaster_checkpoint_loads_frozen(tmp_path):
    model = GRUHumanForecaster(hidden_dim=16, steps=4)
    path = tmp_path / "forecaster.pt"
    save_forecaster_checkpoint(path, model, {"architecture":"gru_human_forecaster_v1","hidden_dim":16,"forecast_steps":4,"training_seeds":[10100]})
    loaded, meta, digest = load_frozen_forecaster(path)
    assert len(digest) == 64
    assert meta["architecture"] == "gru_human_forecaster_v1"
    assert all(not p.requires_grad for p in loaded.parameters())


def test_forecast_episode_collects_extended_entities_without_seed_leakage(tmp_path):
    model = GRUHumanForecaster(hidden_dim=16, steps=4)
    actor = STASACActor(EGO_DIM, hidden_dim=32, entity_dim=FORECAST_ENTITY_DIM)
    spec = next(s for s in local_human_navigation_catalog() if s.name == "human_crossing")
    result = collect_training_episode(actor, spec, seed=10100, max_steps=8, teacher_mix=0.0, use_cbf=True, use_forecast=True, forecaster=model)
    transitions = [x for traj in result["trajectories"] for x in traj]
    assert transitions
    assert all(t["entities"].shape[-1] == FORECAST_ENTITY_DIM for t in transitions)
    generated = {training_seed_for_episode(i) for i in range(200)}
    assert generated <= set(range(10100, 10116))
