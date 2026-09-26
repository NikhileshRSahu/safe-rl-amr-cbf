import torch

from benchmark.human_forecaster import GRUHumanForecaster


def test_gru_forecaster_shapes_sigma_and_gradients():
    torch.manual_seed(3)
    model = GRUHumanForecaster(history_dim=5, hidden_dim=32, steps=6, horizon_seconds=1.8)
    history = torch.randn(4, 8, 5)
    mask = torch.ones(4, 8, dtype=torch.bool)
    out = model(history, mask)
    assert out.mean_xy.shape == (4, 6, 2)
    assert out.sigma_xy.shape == (4, 6, 2)
    assert out.mask.shape == (4, 6)
    assert torch.isfinite(out.mean_xy).all()
    assert torch.isfinite(out.sigma_xy).all()
    assert (out.sigma_xy > 0).all()
    loss = out.mean_xy.square().mean() + out.sigma_xy.mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_gru_forecaster_is_permutation_equivariant_and_handles_empty_tracks():
    torch.manual_seed(4)
    model = GRUHumanForecaster(hidden_dim=16, steps=4)
    h = torch.randn(3, 5, 5)
    m = torch.ones(3, 5, dtype=torch.bool)
    a = model(h, m)
    perm = torch.tensor([2, 0, 1])
    b = model(h[perm], m[perm])
    assert torch.allclose(a.mean_xy[perm], b.mean_xy, atol=1e-6)
    empty = model(torch.zeros(0, 5, 5), torch.zeros(0, 5, dtype=torch.bool))
    assert empty.mean_xy.shape == (0, 4, 2)


def test_fresh_residual_forecaster_starts_at_constant_velocity_baseline():
    model = GRUHumanForecaster(hidden_dim=8, steps=3, horizon_seconds=0.9)
    history = torch.zeros(1, 5, 5)
    mask = torch.tensor([[False, False, True, True, True]])
    history[0, 2, :4] = torch.tensor([1.0, 1.0, 0.5, -0.25])
    history[0, 3, :4] = torch.tensor([2.0, 1.5, 0.5, -0.25])
    history[0, 4, :4] = torch.tensor([3.0, 2.0, 0.5, -0.25])
    out = model(history, mask)
    dt = 0.3
    expected = torch.stack([
        torch.tensor([3.0, 2.0]) + (k + 1) * dt * torch.tensor([0.5, -0.25])
        for k in range(3)
    ])
    assert torch.allclose(out.mean_xy[0], expected, atol=1e-6)
