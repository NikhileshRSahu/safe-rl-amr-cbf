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
from cbf import CBFSafetyFilter

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
    # Reward scaling applied to sampled batch rewards before the critic
    # target / actor objective are formed (standard SAC "reward_scale",
    # Haarnoja et al. 2018). NOT applied to env/eval logging -- episode
    # returns in episode_summary.csv, TensorBoard `rollout/episode_return`,
    # etc. remain in raw physical units. Only the internal SAC objective
    # sees the scaled reward.
    #
    # Why this exists: with this project's REWARD_CONFIG (SUCCESS_REWARD=
    # 150, COLLISION_PENALTY=100, dense per-step terms summing to roughly
    # 10-40/step) and gamma=0.99, the *unscaled* discounted return has a
    # naive geometric ceiling of mean_step_reward/(1-gamma) ~= 2000.
    # Empirically (see runs task-227/task-267/task-382 TensorBoard logs)
    # q1_mean/q2_mean swing from about -455 to +167 and critic_loss spikes
    # above 60,000 early in training and never fully settles. Since
    # actor_loss = alpha*log_prob - Q, and alpha is O(0.01-1.0) while log_prob
    # is O(1-10), a Q of that magnitude and noise level completely swamps
    # the entropy term: the actor's gradient is dominated by critic noise,
    # not by any state-dependent signal, and max_grad_norm=10 clipping then
    # collapses that noisy gradient into a fixed direction -- consistent
    # with the mean-network collapse observed identically across all three
    # isolation runs regardless of their (very different) entropy/CBF-reg
    # settings. Scaling rewards by ~0.05 brings the theoretical Q ceiling
    # down to ~100, a much more standard range for a bounded twin-Q critic
    # with Adam lr=3e-4, without changing what the policy is optimizing for
    # (scaling reward by a positive constant doesn't change the optimal
    # policy, only the objective's numerical conditioning).
    reward_scale: float = 0.05
    # Weight on the CBF actor-consistency regularizer (see
    # cbf.CBFSafetyFilter.actor_consistency_loss and the "Actor update"
    # section below). 0.0 (default) reproduces the previous, unregularized
    # behavior exactly -- this is opt-in via SafeSACAgent's `cbf_filter`
    # constructor argument, not a silent behavior change for existing
    # callers. See config.SAFE_SAC_EXTRA_CONFIG.CBF_ACTOR_REG_WEIGHT for
    # the project's chosen default when it IS enabled (train.py wires
    # that in).
    cbf_reg_weight: float = 0.0


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
        cbf_filter: Optional[CBFSafetyFilter] = None,
        robot_state_dim: int = 5,
        max_dynamic_obstacles: int = 0,
        dynamic_obstacle_dim: int = 6,
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
            cbf_filter: If provided (together with ``max_dynamic_obstacles
                > 0``), enables the CBF actor-consistency regularizer (see
                module docstring and ``cbf.CBFSafetyFilter.
                actor_consistency_loss``). This should be the *same*
                filter instance the training environment uses internally
                (e.g. ``env.cbf_filter``) so the regularizer's constraints
                are identical to the ones actually enforced at rollout
                time. ``None`` (default) disables the regularizer and
                reproduces the previous SAC-only behavior exactly.
            robot_state_dim: Dimensionality of the raw physical robot
                state stored alongside each transition when ``cbf_filter``
                is provided (unused otherwise).
            max_dynamic_obstacles: Row count of the raw dynamic obstacle
                array stored per transition when ``cbf_filter`` is
                provided. Must be > 0 together with ``cbf_filter`` for the
                regularizer to actually activate; 0 (default) disables it
                even if ``cbf_filter`` is passed, so the two knobs must
                both be set deliberately.
            dynamic_obstacle_dim: Column width of the raw dynamic obstacle
                array (6 in this project).
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

        self.cbf_filter = cbf_filter
        self.cbf_regularizer_enabled = cbf_filter is not None and max_dynamic_obstacles > 0
        if cbf_filter is not None and max_dynamic_obstacles <= 0:
            logger.warning(
                "SafeSACAgent received a cbf_filter but max_dynamic_obstacles<=0; the "
                "CBF actor-consistency regularizer is DISABLED because there is nowhere "
                "to store the raw obstacle state it needs. Pass max_dynamic_obstacles>0 "
                "(e.g. config.NUM_DYNAMIC_OBSTACLES) to actually enable it."
            )
        # Physical action bounds, used to map the actor's normalized
        # samples to [v, omega] for the regularizer's affine constraint
        # evaluation -- must match AMRWarehouseEnv.step()'s own mapping.
        ab = policy_config.action_bounds
        self._action_low = (ab.v_min, ab.omega_min)
        self._action_high = (ab.v_max, ab.omega_max)

        # -- Online policy (actor + twin-Q critic, shared trunk). -- #
        self.policy = SafeRLPolicy(policy_config).to(self.device)

        # -- Target policy: full Polyak-averaged copy, critic-only use. -- #
        self.target_policy = SafeRLPolicy(policy_config).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        for param in self.target_policy.parameters():
            param.requires_grad_(False)
        self.target_policy.eval()

        # -- Optimizers. --------------------------------------------------- #
        # IMPORTANT: the shared feature_extractor/trunk is trained ONLY by
        # the critic optimizer. Previously both the actor and critic
        # optimizers included the trunk parameters, meaning two INDEPENDENT
        # Adam trackers (separate momentum/variance state) each took a full
        # step on the same physical weight tensors every update() call.
        # That is not "safe" -- it's two uncoordinated optimizers fighting
        # over one representation, and it deterministically collapses the
        # trunk (trunk_out std -> ~0) within a few thousand updates,
        # independent of reward scale, activation function, or critic
        # warmup (all verified empirically; see diag_repro.py). The actor
        # head is now trained on top of a *detached* trunk feature tensor
        # (see `update()` below), so only the critic ever back-propagates
        # into feature_extractor/trunk -- the standard "encoder owned by
        # the critic" pattern used in shared-encoder actor-critics
        # (SAC-AE/DrQ) for exactly this reason.
        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.actor.parameters()),
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
            robot_state_dim=robot_state_dim if self.cbf_regularizer_enabled else 0,
            max_dynamic_obstacles=max_dynamic_obstacles if self.cbf_regularizer_enabled else 0,
            dynamic_obstacle_dim=dynamic_obstacle_dim,
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

    def update(self, batch_size: int, freeze_actor: bool = False) -> Dict[str, float]:
        """Runs one SAC gradient step (critic, actor, alpha, target Polyak update).

        Args:
            batch_size: Number of transitions to sample from ``replay_buffer``.
            freeze_actor: If True, skip the actor/alpha update this step and
                only update the critic. Intended for a short "critic warmup"
                phase right after loading a warm-started (e.g. behavior-cloned)
                actor: the freshly-initialized critic's early Q-estimates are
                noise relative to a pretrained actor, and letting the actor
                update against that noise immediately can erase the pretrained
                behavior before the critic catches up (observed empirically,
                2026-08: a BC-pretrained actor's eval collision rate swung
                20%->90%->10% and progress-per-episode collapsed toward zero
                within the first 40k steps of naive fine-tuning with no
                warmup). With freeze_actor=True the critic still gets real
                (obs, action) pairs to learn from -- the pretrained actor's
                own reasonable behavior -- so by the time the actor starts
                updating, the critic already has a meaningful Q-landscape to
                guide it instead of noise.

        Returns:
            Dict of scalar diagnostics suitable for TensorBoard logging
            (``critic_loss``, ``actor_loss``, ``alpha_loss``, ``alpha``,
            ``q1_mean``, ``q2_mean``, ``policy_entropy``,
            ``mean_batch_reward``, ``mean_batch_cost``,
            ``mean_batch_barrier_value``).
        """
        if self.cbf_regularizer_enabled:
            (obs, action, reward, next_obs, done, cost, barrier_value,
             robot_state_raw, dynamic_obstacles_raw) = self.replay_buffer.sample_with_cbf_state(batch_size)
        else:
            obs, action, reward, next_obs, done, cost, barrier_value = self.replay_buffer.sample(batch_size)
            robot_state_raw = dynamic_obstacles_raw = None
        # Scale only the internal SAC objective's reward (see
        # SafeSACConfig.reward_scale docstring). `reward` from here on is
        # in scaled units for the critic target below; nothing outside
        # this method (env, eval logs, HER buffer storage) is affected,
        # since the replay buffer stores the raw reward and this scaling
        # is applied fresh on every sampled batch. `raw_reward_mean` is
        # kept unscaled purely so the `mean_batch_reward` diagnostic stays
        # comparable to pre-existing TensorBoard runs (task-227/267/382).
        raw_reward_mean = float(reward.squeeze(-1).mean().detach().cpu())
        reward = reward.squeeze(-1) * self.sac_config.reward_scale
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
        if freeze_actor:
            with torch.no_grad():
                new_action, new_q_value, log_prob = self.policy.forward(obs, deterministic=False)
            actor_loss_value = float("nan")
            cbf_reg_loss_value = 0.0
        else:
            # Trunk features are DETACHED here: the actor head trains on
            # top of whatever representation the critic has already
            # learned, but its gradient never reaches feature_extractor/
            # trunk. This is what actually stops the collapse (see the
            # optimizer-construction comment above and diag_repro.py).
            trunk_features = self.policy.extract_features(obs).detach()
            dist = self.policy.actor.action_distribution(trunk_features)
            new_action, log_prob = dist.sample(deterministic=False)
            q1, q2 = self.policy.critic(trunk_features, new_action)
            new_q_value = torch.min(q1, q2)
            alpha = self.alpha
            actor_loss = (alpha * log_prob - new_q_value).mean()

            cbf_reg_loss_value = 0.0
            if self.cbf_regularizer_enabled:
                # See cbf.CBFSafetyFilter.actor_consistency_loss's docstring
                # for the full derivation. In one line: this pulls the actor's
                # raw, un-filtered action samples toward the exact half-spaces
                # the CBF-QP would itself enforce, using the SAME constraint-
                # assembly code the QP uses (no duplicated/driftable math), so
                # the actor's own Q-query point converges toward the region
                # the critic was actually trained on, rather than depending on
                # differentiating the QP solver itself.
                from config import SHELVES
                cbf_reg_loss = self.cbf_filter.actor_consistency_loss(
                    robot_state_raw, dynamic_obstacles_raw, SHELVES,
                    new_action, self._action_low, self._action_high,
                )
                actor_loss = actor_loss + self.sac_config.cbf_reg_weight * cbf_reg_loss
                cbf_reg_loss_value = float(cbf_reg_loss.detach().cpu())

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_optimizer.param_groups[0]["params"], self.sac_config.max_grad_norm
            )
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())

        # ==================================================================== #
        # 3. Entropy temperature (alpha) update.
        # ==================================================================== #
        alpha_loss_value = 0.0
        if self.alpha_optimizer is not None and not freeze_actor:
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
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().cpu()),
            "q1_mean": float(current_q1.mean().detach().cpu()),
            "q2_mean": float(current_q2.mean().detach().cpu()),
            "policy_entropy": float((-log_prob).mean().detach().cpu()),
            "mean_batch_reward": raw_reward_mean,
            "mean_batch_cost": float(cost.mean().detach().cpu()),
            "mean_batch_barrier_value": float(barrier_value.mean().detach().cpu()),
            "cbf_reg_loss": cbf_reg_loss_value,
            "cbf_reg_weighted": cbf_reg_loss_value * self.sac_config.cbf_reg_weight,
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