import torch

from benchmark.history_sac import HistoryActor, augment_observation, transfer_run11_actor
from benchmark.train_multi_agent_research import Actor, OBS_DIM


def test_history_observation_uses_current_and_visible_delta_only():
    cur = torch.arange(OBS_DIM, dtype=torch.float32).numpy()
    prev = cur - 1.0
    aug = augment_observation(cur, prev)
    assert aug.shape == (2 * OBS_DIM,)
    assert (aug[:OBS_DIM] == cur).all()
    assert (aug[OBS_DIM:] == 1.0).all()


def test_run11_transfer_is_exact_when_history_delta_is_zero():
    torch.manual_seed(7)
    base = Actor()
    hist = HistoryActor()
    transfer_run11_actor(base, hist)
    obs = torch.randn(11, OBS_DIM)
    aug = torch.cat([obs, torch.zeros_like(obs)], dim=-1)
    with torch.no_grad():
        base_action, _ = base.sample(obs, True)
        hist_action, _ = hist.sample(aug, True)
    assert torch.allclose(base_action, hist_action, atol=1e-6, rtol=1e-6)
