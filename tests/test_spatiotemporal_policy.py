import torch

from benchmark.spatiotemporal_policy import (
    RiskAttentionSceneEncoder,
    STASACActor,
    TwinRecurrentQ,
    reset_hidden,
)
from benchmark.warehouse_interaction_features import FEATURE_DIM


def _inputs(batch=3, entities=7, ego_dim=12):
    torch.manual_seed(11)
    ego = torch.randn(batch, ego_dim)
    ent = torch.randn(batch, entities, FEATURE_DIM)
    mask = torch.ones(batch, entities, dtype=torch.bool)
    return ego, ent, mask


def test_scene_encoder_is_permutation_invariant_over_entities():
    ego, entities, mask = _inputs()
    encoder = RiskAttentionSceneEncoder(ego_dim=ego.shape[-1])
    encoder.eval()
    h0 = torch.zeros(ego.shape[0], encoder.hidden_dim)
    with torch.no_grad():
        z1, h1, _ = encoder(ego, entities, mask, h0)
        perm = torch.tensor([4, 0, 6, 1, 5, 2, 3])
        z2, h2, _ = encoder(ego, entities[:, perm], mask[:, perm], h0)
    torch.testing.assert_close(z1, z2, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(h1, h2, atol=1e-6, rtol=1e-6)


def test_encoder_handles_zero_entities_and_twenty_four_entities():
    ego = torch.randn(2, 12)
    encoder = RiskAttentionSceneEncoder(ego_dim=12)
    h0 = torch.zeros(2, encoder.hidden_dim)

    empty_entities = torch.zeros(2, 0, FEATURE_DIM)
    empty_mask = torch.zeros(2, 0, dtype=torch.bool)
    z0, h0n, w0 = encoder(ego, empty_entities, empty_mask, h0)
    assert z0.shape == (2, encoder.hidden_dim)
    assert h0n.shape == (2, encoder.hidden_dim)
    assert w0.shape == (2, 0)
    assert torch.isfinite(z0).all()

    many_entities = torch.randn(2, 24, FEATURE_DIM)
    many_mask = torch.ones(2, 24, dtype=torch.bool)
    z24, h24, w24 = encoder(ego, many_entities, many_mask, h0)
    assert z24.shape == (2, encoder.hidden_dim)
    assert h24.shape == (2, encoder.hidden_dim)
    assert w24.shape == (2, 24)
    torch.testing.assert_close(w24.sum(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_hidden_reset_is_per_agent_and_exact():
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    done = torch.tensor([False, True, False])
    reset = reset_hidden(hidden, done)
    torch.testing.assert_close(reset[0], hidden[0])
    torch.testing.assert_close(reset[1], torch.zeros_like(hidden[1]))
    torch.testing.assert_close(reset[2], hidden[2])


def test_actor_deterministic_mode_and_shapes_are_finite():
    ego, entities, mask = _inputs(batch=4, entities=9)
    actor = STASACActor(ego_dim=ego.shape[-1])
    actor.eval()
    hidden = torch.zeros(4, actor.hidden_dim)
    with torch.no_grad():
        a1, lp1, h1, _ = actor.sample(ego, entities, mask, hidden, deterministic=True)
        a2, lp2, h2, _ = actor.sample(ego, entities, mask, hidden, deterministic=True)
    assert a1.shape == (4, 2)
    assert lp1 is None and lp2 is None
    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(h1, h2)
    assert torch.isfinite(a1).all()
    assert torch.all(a1 <= 1.0) and torch.all(a1 >= -1.0)


def test_actor_and_twin_q_have_finite_gradients():
    ego, entities, mask = _inputs(batch=5, entities=6)
    actor = STASACActor(ego_dim=ego.shape[-1])
    q = TwinRecurrentQ(hidden_dim=actor.hidden_dim)
    hidden = torch.zeros(5, actor.hidden_dim)
    actions, logp, next_hidden, _ = actor.sample(ego, entities, mask, hidden, deterministic=False)
    q1, q2 = q(next_hidden, actions)
    loss = q1.mean() + q2.mean() + logp.mean()
    loss.backward()
    grads = [p.grad for p in list(actor.parameters()) + list(q.parameters()) if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
