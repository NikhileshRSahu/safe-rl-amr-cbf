from __future__ import annotations

import numpy as np
import torch


def bc_coefficient(agent_step: int, total_agent_steps: int, warm_fraction: float = 0.15) -> float:
    """Legacy linear warm-start schedule retained for reproducibility/ablations."""
    total = max(1, int(total_agent_steps))
    warm = max(1, int(round(total * float(warm_fraction))))
    step = max(0, int(agent_step))
    if step >= warm:
        return 0.0
    return float(1.0 - step / warm)


def performance_gated_bc_coefficient(
    policy_success_rate: float,
    policy_collision_rate: float,
    teacher_success_rate: float,
    teacher_collision_rate: float,
    *,
    collision_tolerance: float = 0.01,
    safety_scale: float = 0.10,
) -> float:
    """Choose ORCA imitation weight from measured policy competence.

    The teacher remains influential while the policy trails its success rate or
    violates safety parity.  Once the policy reaches/exceeds teacher success
    without materially worse collision rate, the coefficient becomes exactly 0.

    This is intentionally based on *policy-only* evaluation statistics; callers
    must not estimate these rates from teacher-mixed rollout outcomes.
    """
    ps = float(np.clip(policy_success_rate, 0.0, 1.0))
    pc = float(np.clip(policy_collision_rate, 0.0, 1.0))
    ts = float(np.clip(teacher_success_rate, 0.0, 1.0))
    tc = float(np.clip(teacher_collision_rate, 0.0, 1.0))
    tol = max(0.0, float(collision_tolerance))
    scale = max(1e-6, float(safety_scale))

    success_deficit = max(0.0, ts - ps) / max(ts, 0.05)
    excess_collision = max(0.0, pc - (tc + tol)) / scale
    return float(np.clip(max(success_deficit, excess_collision), 0.0, 1.0))


def should_promote_curriculum_stage(
    *,
    episodes: int,
    success_rate: float,
    collision_rate: float,
    min_episodes: int = 10,
    min_success_rate: float = 0.85,
    max_collision_rate: float = 0.02,
) -> bool:
    """Safety-first gate for increasing warehouse interaction difficulty."""
    return bool(
        int(episodes) >= int(min_episodes)
        and float(success_rate) >= float(min_success_rate)
        and float(collision_rate) <= float(max_collision_rate)
    )


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
        return actor_action.sum() * 0.0
    return coefficient * torch.mean((actor_action - teacher_action.detach()) ** 2)
