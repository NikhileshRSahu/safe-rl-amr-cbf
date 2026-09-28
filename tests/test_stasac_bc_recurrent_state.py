import torch

from benchmark.spatiotemporal_policy import STASACActor
import benchmark.train_stasac_cbf as train_stasac_cbf


def test_bc_deterministic_action_uses_current_scene_without_reencoding_observation():
    """BC must supervise the same recurrent scene used by the rollout action.

    Re-encoding the current observation with an already-advanced GRU hidden state
    creates a state the online policy never sees and can make BC look converged
    while policy-only evaluation fails.
    """
    torch.manual_seed(5)
    actor = STASACActor(ego_dim=3, hidden_dim=16, entity_dim=9)
    ego = torch.randn(4, 3)
    entities = torch.randn(4, 2, 9)
    mask = torch.ones(4, 2, dtype=torch.bool)
    hidden_before = torch.randn(4, 16)

    _, _, current_scene, _ = actor.sample(
        ego, entities, mask, hidden_before, deterministic=False
    )
    expected = torch.tanh(actor.mu(current_scene))

    actual = train_stasac_cbf.deterministic_action_from_scene(actor, current_scene)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
