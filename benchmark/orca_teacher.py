from __future__ import annotations

import numpy as np
import torch


def bc_coefficient(agent_step: int, total_agent_steps: int, warm_fraction: float = 0.15) -> float:
    """Linear ORCA imitation weight used only at the start of training."""
    total = max(1, int(total_agent_steps))
    warm = max(1, int(round(total * float(warm_fraction))))
    step = max(0, int(agent_step))
    if step >= warm:
        return 0.0
    return float(1.0 - step / warm)


def normalize_teacher_action(v: float, omega: float, v_max: float, omega_max: float) -> np.ndarray:
    """Convert physical DD command to the actor's normalized [-1,1]^2 action."""
    v_max = max(float(v_max), 1e-9)
    omega_max = max(float(omega_max), 1e-9)
    v = float(np.clip(v, 0.0, v_max))
    omega = float(np.clip(omega, -omega_max, omega_max))
    return np.asarray([2.0 * v / v_max - 1.0, omega / omega_max], dtype=np.float32)


def behavior_cloning_loss(
    actor_action: torch.Tensor,
    teacher_action: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Weighted MSE teacher loss; coefficient=0 removes teacher influence exactly."""
    coefficient = max(0.0, float(coefficient))
    if coefficient == 0.0:
        # Preserve a differentiable zero with the actor's device/dtype.
        return actor_action.sum() * 0.0
    return coefficient * torch.mean((actor_action - teacher_action.detach()) ** 2)
