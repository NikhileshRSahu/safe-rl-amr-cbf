"""safe_sac.py

Off-policy Soft Actor-Critic (SAC) trainer for :class:`policy.SafeRLPolicy`,
paired with ``AMRWarehouseEnv``'s native Control-Barrier-Function (CBF-QP)
safety filter.

Design notes (matching the rest of this codebase -- see ``README.md``)
------------------------------------------------------------------------
* **Safety is enforced in the environment, not the loss.** ``AMRWarehouseEnv``
  filters every action through its CBF-QP *before* the transition is
  observed, so :class:`SafeSACAgent` trains on the standard maximum-entropy
  SAC objective using the resulting (already-safe) transitions. It never
  backpropagates through the QP. ``cost`` and ``barrier_value`` are sampled
  from :class:`replay_buffer.SafeReplayBuffer` and logged every ``update()``
  call for diagnostics only -- they are not (yet) part of the loss, ready to
  be wired into a Lagrangian safety-critic extension later.
* **Shared actor/critic trunk.** ``SafeRLPolicy`` shares one
  ``feature_extractor`` + ``trunk`` between the actor and the twin-Q critic.
  The target network used for the SAC Bellman backup is therefore a full
  Polyak-averaged copy of the *entire* policy (``target_policy``), not just
  a target Q-head, so the target is computed with a consistent feature
  pipeline.
* **Action space convention.** ``SafeSACAgent.select_action`` returns an
  action expressed in whatever units ``policy_config.action_bounds`` was
  configured with. ``train.py`` sets ``action_bounds`` to normalized
  ``[-1, 1]`` on both dims to match ``AMRWarehouseEnv.action_space``
  directly, so the returned action can be passed straight to ``env.step()``
  with no additional rescaling.

Standard SAC update (per call to :meth:`SafeSACAgent.update`)
---------------------------------------------------------------
1. **Critic**: sample ``a' ~ pi(.|s')`` from the *current* policy, evaluate
   the *target* twin-Q on ``(s', a')``, form the entropy-regularized Bellman
   target ``y = r + gamma * (1 - done) * (min(Q1', Q2')(s', a') - alpha *
   log pi(a'|s'))``, and regress the current twin-Q on ``y``.
2. **Actor**: sample ``a ~ pi(.|s)`` (reparameterized), and minimize
   ``alpha * log pi(a|s) - min(Q1, Q2)(s, a)``.
3. **Alpha** (if ``policy_config.use_auto_entropy_tuning``): minimize the
   standard SAC temperature loss so the policy's entropy tracks
   ``target_entropy`` (default ``-action_dim``).
4. **Target network**: Polyak-average ``policy`` into ``target_policy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from policy import PolicyConfig, SafeRLPolicy
from replay_buffer import SafeReplayBuffer

logger = logging.getLogger(__name__)

ObsDict = Dict[str, np.ndarray]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class SafeSACConfig:
    """Hyperparameters for :class:`SafeSACAgent`.

    Attributes:
        gamma: Discount factor.
        tau: Polyak averaging coefficient for the target-network soft update
            (``target = (1 - tau) * target + tau * online``).
        actor_lr: Learning rate for the actor (+ shared trunk) optimizer.
        critic_lr: Learning rate for the twin-Q critic (+ shared trunk)
            optimizer.
        alpha_lr: Learning rate for the automatic entropy-temperature
            optimizer. Unused if ``policy_config.use_auto_entropy_tuning``
            is ``False``.
        fixed_alpha: Entropy coefficient used when automatic entropy tuning
            is disabled.
        max_grad_norm: Global gradient-clipping norm applied to each
            optimizer's parameter group before every ``step()``.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``) the agent's
            networks and replay buffer live on.
    """

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    fixed_alpha: float = 0.2
    max_grad_norm: float = 10.0
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class SafeSACAgent:
    """Off-policy SAC agent wrapping :class:`policy.SafeRLPolicy`.

    Example:
        >>> agent = SafeSACAgent(
        ...     policy_config=PolicyConfig(robot_state_dim=5, goal_dim=4, lidar_dim=192),
        ...     observation_space=env.observation_space,
        ...     action_dim=2,
        ...     sac_config=SafeSACConfig(device="cpu"),
        ... )
        >>> action = agent.select_action(obs, deterministic=False)
        >>> next_obs, reward, terminated, truncated, info = env.step(action)
        >>> agent.replay_buffer.add(obs, action, reward, next_obs, terminated)
        >>> stats = agent.update(batch_size=256)
    """

    def __init__(
        self,
        policy_config: PolicyConfig,
        observation_space: spaces.Dict,
        action_dim: int,
        sac_config: Optional[SafeSACConfig] = None,
        replay_buffer_size: int = 1_000_000,
    ) -> None:
        """Builds the online/target policies, optimizers, and replay buffer.

        Args:
            policy_config: Configuration for the underlying ``SafeRLPolicy``.
                Must have ``critic_type="q"`` for the twin-Q SAC critic to be
                instantiated.
            observation_space: The environment's (``spaces.Dict``)
                observation space, used to size the replay buffer.
            action_dim: Action dimensionality (must match
                ``policy_config.action_dim``).
            sac_config: SAC hyperparameters. Defaults to ``SafeSACConfig()``.
            replay_buffer_size: Ring-buffer capacity (number of transitions).
        """
        if policy_config.critic_type != "q":
            raise ValueError(
                f"SafeSACAgent requires policy_config.critic_type='q' (twin-Q "
                f"critic) for SAC; got {policy_config.critic_type!r}."
            )

        self.policy_config = policy_config
        self.sac_config = sac_config or SafeSACConfig()
        self.action_dim = int(action_dim)
        self.device = torch.device(self.sac_config.device)

        # -- Online policy (actor + twin-Q critic, shared trunk). -- #
        self.policy = SafeRLPolicy(policy_config).to(self.device)

        # -- Target policy: full Polyak-averaged copy, critic-only use. -- #
        self.target_policy = SafeRLPolicy(policy_config).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        for param in self.target_policy.parameters():
            param.requires_grad_(False)
        self.target_policy.eval()

        # -- Optimizers. Trunk + feature_extractor are shared, so both the
        #    actor and critic optimizers include them; each optimizer only
        #    *steps* its own parameters, so this is safe (see module
        #    docstring for the shared-trunk rationale). -- #
        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.feature_extractor.parameters())
            + list(self.policy.trunk.parameters())
            + list(self.policy.actor.parameters()),
            lr=self.sac_config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.policy.feature_extractor.parameters())
            + list(self.policy.trunk.parameters())
            + list(self.policy.critic.parameters()),
            lr=self.sac_config.critic_lr,
        )

        self.alpha_optimizer: Optional[torch.optim.Optimizer] = None
        if self.policy.entropy_temperature is not None:
            self.alpha_optimizer = torch.optim.Adam(
                self.policy.entropy_temperature.parameters(), lr=self.sac_config.alpha_lr
            )

        # -- Replay buffer. If the policy expects attention-based obstacle
        #    inputs but the provided observation_space doesn't include the
        #    required keys, extend it so the buffer pre-allocates storage for
        #    `obstacle_set` and `obstacle_set_mask`.
        obs_space_for_buffer = observation_space
        try:
            from gymnasium import spaces as _spaces
        except Exception:
            _spaces = None

        if getattr(self.policy_config, "use_attention_obstacles", False) and isinstance(observation_space, _spaces.Dict if _spaces is not None else dict):
            # Only add keys if missing
            if "obstacle_set" not in observation_space.spaces or "obstacle_set_mask" not in observation_space.spaces:
                logger.warning(
                    "policy_config.use_attention_obstacles=True but observation_space is missing "
                    "'obstacle_set'/'obstacle_set_mask'; auto-extending observation_space to add them. "
                    "This usually means the caller built observation_space from the base env instead "
                    "of using build_observation_space_with_obstacles(); replay buffer keys were "
                    "pre-allocated for the extended space."
                )
                new_spaces = dict(observation_space.spaces)
                max_obs = int(self.policy_config.max_obstacles)
                new_spaces["obstacle_set"] = _spaces.Box(low=-1.0, high=1.0, shape=(max_obs, int(self.policy_config.obstacle_feature_dim)), dtype=np.float32)
                new_spaces["obstacle_set_mask"] = _spaces.Box(low=0, high=1, shape=(max_obs,), dtype=bool)
                obs_space_for_buffer = _spaces.Dict(new_spaces)

        self.replay_buffer = SafeReplayBuffer(
            obs_space_for_buffer,
            action_dim=self.action_dim,
            max_size=replay_buffer_size,
            device=self.device,
        )

        self._update_count = 0

    # -- properties ------------------------------------------------------- #

    @property
    def alpha(self) -> torch.Tensor:
        """Current SAC entropy temperature (learned if enabled, else fixed)."""
        if self.policy.entropy_temperature is not None:
            return self.policy.alpha.detach()
        return torch.as_tensor(self.sac_config.fixed_alpha, device=self.device)

    # -- mode switching ----------------------------------------------------- #

    def train(self) -> None:
        """Puts the online policy into training mode (stochastic sampling)."""
        self.policy.train()

    def eval(self) -> None:
        """Puts the online policy into evaluation mode (deterministic by default)."""
        self.policy.eval()

    # -- inference ----------------------------------------------------------- #

    @torch.no_grad()
    def select_action(self, obs: ObsDict, deterministic: bool = False) -> np.ndarray:
        """Selects a single action for one (unbatched) environment observation.

        Args:
            obs: Structured observation dict of NumPy arrays, as returned by
                ``AMRWarehouseEnv.reset()`` / ``.step()``.
            deterministic: If ``True``, use the actor's distribution mean
                (evaluation/inference). If ``False``, sample stochastically
                (exploration during training).

        Returns:
            ``np.ndarray`` of shape ``(action_dim,)``, ``float32``, in the
            units defined by ``policy_config.action_bounds``.
        """
        obs_tensors = {}
        for key, value in obs.items():
            arr = np.asarray(value)
            if arr.dtype == bool:
                obs_tensors[key] = torch.as_tensor(arr, dtype=torch.bool, device=self.device).unsqueeze(0)
            else:
                obs_tensors[key] = torch.as_tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        action, _value, _log_prob = self.policy.forward(obs_tensors, deterministic=deterministic)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # -- training ------------------------------------------------------------ #

    def update(self, batch_size: int) -> Dict[str, float]:
        """Runs one SAC gradient step (critic, actor, alpha, target Polyak update).

        Args:
            batch_size: Number of transitions to sample from ``replay_buffer``.

        Returns:
            Dict of scalar diagnostics suitable for TensorBoard logging
            (``critic_loss``, ``actor_loss``, ``alpha_loss``, ``alpha``,
            ``q1_mean``, ``q2_mean``, ``policy_entropy``,
            ``mean_batch_reward``, ``mean_batch_cost``,
            ``mean_batch_barrier_value``).
        """
        obs, action, reward, next_obs, done, cost, barrier_value = self.replay_buffer.sample(batch_size)
        reward = reward.squeeze(-1)
        done = done.squeeze(-1)

        # ==================================================================== #
        # 1. Critic update.
        # ==================================================================== #
        with torch.no_grad():
            next_action, _next_value, next_log_prob = self.policy.forward(next_obs, deterministic=False)
            target_q1, target_q2 = self.target_policy.get_q_values(next_obs, next_action)
            target_min_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_y = reward + (1.0 - done) * self.sac_config.gamma * target_min_q

        current_q1, current_q2 = self.policy.get_q_values(obs, action)
        critic_loss = F.mse_loss(current_q1, target_y) + F.mse_loss(current_q2, target_y)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_optimizer.param_groups[0]["params"], self.sac_config.max_grad_norm
        )
        self.critic_optimizer.step()

        # ==================================================================== #
        # 2. Actor update.
        # ==================================================================== #
        new_action, new_q_value, log_prob = self.policy.forward(obs, deterministic=False)
        alpha = self.alpha
        actor_loss = (alpha * log_prob - new_q_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_optimizer.param_groups[0]["params"], self.sac_config.max_grad_norm
        )
        self.actor_optimizer.step()

        # ==================================================================== #
        # 3. Entropy temperature (alpha) update.
        # ==================================================================== #
        alpha_loss_value = 0.0
        if self.alpha_optimizer is not None:
            alpha_loss = self.policy.compute_alpha_loss(log_prob.detach())
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach().cpu())

        # ==================================================================== #
        # 4. Target network Polyak update.
        # ==================================================================== #
        self._polyak_update(self.sac_config.tau)
        self._update_count += 1

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().cpu()),
            "q1_mean": float(current_q1.mean().detach().cpu()),
            "q2_mean": float(current_q2.mean().detach().cpu()),
            "policy_entropy": float((-log_prob).mean().detach().cpu()),
            "mean_batch_reward": float(reward.mean().detach().cpu()),
            "mean_batch_cost": float(cost.mean().detach().cpu()),
            "mean_batch_barrier_value": float(barrier_value.mean().detach().cpu()),
        }

    @torch.no_grad()
    def _polyak_update(self, tau: float) -> None:
        """Soft-updates ``target_policy`` towards ``policy``: ``target = (1-tau)*target + tau*online``.

        Args:
            tau: Polyak averaging coefficient in ``(0, 1]``.
        """
        for target_param, online_param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)
        # Buffers (e.g. running-normalizer statistics) are hard-copied rather
        # than Polyak-averaged, since they are not learned weights.
        for target_buffer, online_buffer in zip(self.target_policy.buffers(), self.policy.buffers()):
            target_buffer.data.copy_(online_buffer.data)

    # -- checkpointing --------------------------------------------------------- #

    def save_models(self, filepath: Union[str, Path]) -> None:
        """Saves policy/target weights, optimizer states, and configs to a single file.

        Args:
            filepath: Destination ``.pt`` path (parent directories are
                created if needed).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "target_policy_state_dict": self.target_policy.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "alpha_optimizer_state_dict": (
                    self.alpha_optimizer.state_dict() if self.alpha_optimizer is not None else None
                ),
                "policy_config": self.policy_config,
                "sac_config": self.sac_config,
                "update_count": self._update_count,
            },
            filepath,
        )

    def load_models(self, filepath: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None) -> None:
        """Restores policy/target weights and optimizer states from a checkpoint.

        The agent must already have been constructed with the *same*
        ``policy_config`` the checkpoint was saved with (see
        ``evaluate.py``'s ``load_agent`` for the standard "read config from
        checkpoint, build agent, then load weights" pattern) so that every
        ``state_dict`` shape lines up.

        Args:
            filepath: Path to a ``.pt`` file written by :meth:`save_models`.
            map_location: Torch device to map tensors onto. Defaults to
                ``self.device``.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist.
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint: Dict[str, Any] = torch.load(
            filepath, map_location=map_location or self.device, weights_only=False
        )

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "target_policy_state_dict" in checkpoint:
            self.target_policy.load_state_dict(checkpoint["target_policy_state_dict"])
        else:
            self.target_policy.load_state_dict(self.policy.state_dict())

        if checkpoint.get("actor_optimizer_state_dict") is not None:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if checkpoint.get("critic_optimizer_state_dict") is not None:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if self.alpha_optimizer is not None and checkpoint.get("alpha_optimizer_state_dict") is not None:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

        self._update_count = int(checkpoint.get("update_count", 0))