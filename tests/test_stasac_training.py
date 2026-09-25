from pathlib import Path

import numpy as np
import torch

from benchmark.spatiotemporal_policy import STASACActor, TwinRecurrentQ
from benchmark.train_stasac_cbf import (
    SequenceReplay,
    load_stasac_checkpoint,
    recurrent_sac_update,
    save_stasac_checkpoint,
)
from benchmark.warehouse_interaction_features import FEATURE_DIM


def _step(t, ego_dim=12, n_entities=3, done=False):
    return {
        "ego": np.full((ego_dim,), t, np.float32),
        "entities": np.full((n_entities, FEATURE_DIM), t, np.float32),
        "entity_mask": np.ones((n_entities,), np.bool_),
        "action": np.array([0.1, -0.2], np.float32),
        "reward": float(t) * 0.01,
        "next_ego": np.full((ego_dim,), t + 1, np.float32),
        "next_entities": np.full((n_entities, FEATURE_DIM), t + 1, np.float32),
        "next_entity_mask": np.ones((n_entities,), np.bool_),
        "done": bool(done),
    }


def test_sequence_replay_never_crosses_episode_boundaries_and_marks_burn_in():
    replay = SequenceReplay(capacity_episodes=8, burn_in=2, train_len=3)
    replay.add_episode([_step(t, done=(t == 5)) for t in range(6)])
    replay.add_episode([_step(100 + t, done=(t == 5)) for t in range(6)])
    batch = replay.sample(batch_size=4, rng=np.random.default_rng(7))
    assert batch["ego"].shape[:2] == (4, 5)
    assert batch["train_mask"].shape == (4, 5)
    assert not batch["train_mask"][:, :2].any()
    assert batch["train_mask"][:, 2:].all()
    for row in batch["ego"][:, :, 0].numpy():
        assert not (row.min() < 50 < row.max())


def test_done_mask_is_present_and_terminal_only_at_valid_episode_position():
    replay = SequenceReplay(capacity_episodes=4, burn_in=1, train_len=2)
    replay.add_episode([_step(0), _step(1), _step(2, done=True)])
    batch = replay.sample(batch_size=1, rng=np.random.default_rng(1))
    done = batch["done"][0, :, 0]
    assert done.dtype == torch.float32
    assert torch.all((done == 0) | (done == 1))
    assert done[-1].item() == 1.0


def test_one_recurrent_sac_update_changes_actor_parameters_and_is_finite():
    torch.manual_seed(3)
    replay = SequenceReplay(capacity_episodes=4, burn_in=1, train_len=3)
    replay.add_episode([_step(t, n_entities=(t % 4)) for t in range(7)])
    batch = replay.sample(batch_size=2, rng=np.random.default_rng(4))
    actor = STASACActor(ego_dim=12)
    q = TwinRecurrentQ(hidden_dim=actor.hidden_dim)
    target_q = TwinRecurrentQ(hidden_dim=actor.hidden_dim)
    target_q.load_state_dict(q.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    q_opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in actor.parameters()]
    metrics = recurrent_sac_update(
        batch, actor, q, target_q, actor_opt, q_opt, alpha=0.08, gamma=0.99, tau=0.01
    )
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, actor.parameters()))


def test_checkpoint_round_trip_preserves_deterministic_action(tmp_path: Path):
    torch.manual_seed(9)
    actor = STASACActor(ego_dim=12)
    q = TwinRecurrentQ(hidden_dim=actor.hidden_dim)
    path = tmp_path / "stasac_cbf.pt"
    save_stasac_checkpoint(path, actor, q, metadata={"agent_steps": 123})
    loaded_actor, loaded_q, metadata = load_stasac_checkpoint(path, ego_dim=12)
    assert metadata["agent_steps"] == 123
    ego = torch.randn(2, 12)
    entities = torch.randn(2, 5, FEATURE_DIM)
    mask = torch.ones(2, 5, dtype=torch.bool)
    hidden = torch.zeros(2, actor.hidden_dim)
    with torch.no_grad():
        a1 = actor.sample(ego, entities, mask, hidden, deterministic=True)[0]
        a2 = loaded_actor.sample(ego, entities, mask, hidden, deterministic=True)[0]
    torch.testing.assert_close(a1, a2)
    for p1, p2 in zip(q.parameters(), loaded_q.parameters()):
        torch.testing.assert_close(p1, p2)
