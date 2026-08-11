"""policy.py

Research-quality Safe Reinforcement Learning policy for CBF-filtered mobile
robot navigation.

This module implements an actor-critic policy network intended to sit
upstream of an external Control Barrier Function (CBF) safety filter. The
policy proposes an unfiltered action ``u_rl``; a separate safety layer
(zeroing CBF / HOCBF / soft-constrained QP / CLF, etc.) is responsible for
projecting ``u_rl`` onto a safe action ``u_safe`` before it is applied to the
robot. This module never assumes ``u_rl`` is the action that is ultimately
executed, and it does not train on the filtered action unless an external
training script explicitly feeds it back in.

Design goals:
    * Differentiable action squashing (tanh) instead of hard clipping.
    * Orthogonal weight initialization with ReLU-appropriate gain.
    * LayerNorm + residual MLP blocks for stable, deep training.
    * Modular per-modality encoders (robot / goal / lidar / obstacles /
      images / dynamic obstacle sets).
    * Independent actor and critic heads sharing only the encoder trunk.
    * Learnable, state-dependent Gaussian exploration noise (log_std), with
      optional generalized State-Dependent Exploration (gSDE).
    * A reusable action-distribution abstraction (Gaussian, squashed
      Gaussian, Beta) shared by PPO- and SAC-style heads.
    * Optional automatic SAC entropy-temperature (alpha) tuning.
    * Optional CNN encoder for image/depth/occupancy-map observations.
    * Optional attention-based encoder for variable-size dynamic obstacle
      sets.
    * Optional recurrent (LSTM/GRU) trunk, opt-in and fully backward
      compatible with the feed-forward default.
    * Optional auxiliary prediction heads (collision risk, barrier value,
      time-to-collision, goal distance) for auxiliary-loss training.
    * Optional running observation normalization (toggle-able).
    * Entropy coefficient scheduling for use in an external training loop.
    * Observation validation (shape / missing-key / NaN / Inf checks).
    * ONNX export and checkpoint save/load helpers.
    * Device-agnostic (CPU / CUDA / MPS) and autocast (AMP) friendly.
    * Leaf encoder/head modules are scriptable; the top-level policy is not. 
      Use ``export_onnx`` or ``torch.compile`` for deployment.

The public class names and call signatures below are intended to be stable
across revisions of this file. All additions in this revision are opt-in
via :class:`PolicyConfig` flags that default to the prior behavior, so
existing training scripts continue to work unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta, Normal

try:  # Optional dependency: keep this module usable stand-alone.
    from stable_baselines3.common.policies import BasePolicy as _SB3BasePolicy

    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent.
    _SB3BasePolicy = nn.Module  # type: ignore[assignment,misc]
    _SB3_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EPS: float = 1e-6
"""Small epsilon used throughout to avoid division-by-zero / NaN / Inf."""

_DEFAULT_HEAD_HIDDEN: Tuple[int, int] = (256, 256)


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class ActionBounds:
    """Physical bounds for a differential-drive action ``[v, omega]``.

    Attributes:
        v_min: Minimum linear velocity (m/s).
        v_max: Maximum linear velocity (m/s).
        omega_min: Minimum angular velocity (rad/s).
        omega_max: Maximum angular velocity (rad/s).
    """

    v_min: float = 0.0
    v_max: float = 1.0
    omega_min: float = -1.0
    omega_max: float = 1.0

    def as_tensors(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        """Return ``(low, high)`` tensors of shape ``[2]`` for ``[v, omega]``.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of ``(low, high)`` tensors.
        """
        low = torch.as_tensor([self.v_min, self.omega_min], device=device, dtype=dtype)
        high = torch.as_tensor([self.v_max, self.omega_max], device=device, dtype=dtype)
        return low, high


@dataclass
class ObservationKeys:
    """Dictionary keys expected in a structured observation.

    Any key may be absent from the observation dict; the corresponding
    encoder is simply skipped. Set the matching ``*_dim`` field in
    :class:`PolicyConfig` to ``0`` to disable an encoder explicitly.
    """

    robot: str = "robot_state"
    goal: str = "goal"
    lidar: str = "lidar"
    obstacles: str = "obstacles"
    image: str = "image"
    obstacle_set: str = "obstacle_set"
    obstacle_set_mask: str = "obstacle_set_mask"


@dataclass
class PolicyConfig:
    """Static configuration for :class:`SafeRLPolicy`.

    Attributes:
        robot_state_dim: Dimensionality of the robot proprioceptive state
            (e.g. linear/angular velocity, heading). ``0`` disables the
            robot encoder.
        goal_dim: Dimensionality of the raw goal vector (e.g. relative
            ``[dx, dy]``). ``0`` disables the goal encoder.
        lidar_dim: Dimensionality of the raw lidar/laser scan. ``0``
            disables the lidar encoder.
        obstacle_dim: Dimensionality of the raw obstacle-feature vector
            (e.g. flattened tracked-obstacle states). ``0`` disables the
            obstacle encoder.
        encoder_hidden: Hidden width used inside each modality encoder.
        encoder_out: Output width of each modality encoder.
        head_hidden: Hidden widths of the residual MLP trunk shared by the
            actor and critic heads (heads themselves remain independent).
        action_dim: Action dimensionality. Fixed at ``2`` for ``[v, omega]``
            differential-drive control.
        log_std_min: Lower clamp bound for the learnable log standard
            deviation.
        log_std_max: Upper clamp bound for the learnable log standard
            deviation.
        action_bounds: Physical ``[v, omega]`` bounds used for tanh scaling.
        normalize_obs: Whether running observation normalization is enabled
            by default (can be toggled at runtime).
        use_goal_conditioning: If ``True``, encode goal distance/heading
            instead of (or in addition to) raw goal coordinates.
        orthogonal_gain_hidden: Orthogonal-init gain for hidden layers.
        orthogonal_gain_output: Orthogonal-init gain for output layers.
        observation_keys: Dictionary key names for structured observations.
        critic_type: Type of critic to instantiate. ``"v"`` for standard state-value
            (PPO/A2C) or ``"q"`` for twin action-value (SAC). Default: ``"v"``.
        distribution_type: Action distribution family used by the actor
            head. One of ``"squashed_gaussian"`` (default; matches prior
            behavior), ``"gaussian"`` (no tanh squash; bounds enforced only
            through the affine output layer), or ``"beta"`` (bounded support
            without a tanh log-prob correction).
        use_sde: If ``True``, use generalized State-Dependent Exploration
            (gSDE) instead of state-independent-per-step Gaussian noise.
            Requires ``distribution_type in ("squashed_gaussian",
            "gaussian")``.
        sde_sample_freq: Number of environment steps between gSDE noise
            matrix resamples. ``-1`` means "resample only when
            :meth:`SafeRLPolicy.reset_noise` is called explicitly".
        use_auto_entropy_tuning: If ``True``, instantiate a learnable SAC
            entropy temperature (``log_alpha``) with an automatically
            derived target entropy (``-action_dim``) instead of a fixed
            entropy coefficient.
        initial_alpha: Initial value of the SAC entropy temperature
            ``alpha`` when ``use_auto_entropy_tuning`` is enabled.
        target_entropy: Explicit target entropy for automatic alpha tuning.
            If ``None``, defaults to ``-action_dim`` (standard SAC heuristic).
        image_shape: Optional ``(channels, height, width)`` for an
            image/depth/occupancy-map observation. ``None`` disables the
            CNN encoder.
        cnn_feature_dim: Output width of the CNN encoder, if enabled.
        use_attention_obstacles: If ``True``, encode the ``obstacle_set``
            observation (variable-count dynamic obstacles) with a
            self-attention encoder instead of (or in addition to) the flat
            ``obstacle_dim`` MLP encoder.
        max_obstacles: Maximum number of dynamic obstacles per time step
            when ``use_attention_obstacles`` is enabled.
        obstacle_feature_dim: Per-obstacle raw feature width (e.g.
            ``[rel_x, rel_y, rel_vx, rel_vy, radius]``) when
            ``use_attention_obstacles`` is enabled.
        attention_heads: Number of attention heads in the obstacle encoder.
        use_recurrent: If ``True``, replace the feed-forward trunk with an
            LSTM/GRU recurrent trunk. The default feed-forward call pattern
            (``forward(observations, deterministic)`` returning a 3-tuple)
            is preserved; callers that want the hidden state back must pass
            ``return_hidden_state=True`` explicitly (see
            :meth:`SafeRLPolicy.forward`).
        recurrent_type: One of ``"lstm"`` or ``"gru"``.
        recurrent_hidden_size: Hidden width of the recurrent trunk.
        recurrent_num_layers: Number of stacked recurrent layers.
        aux_heads: Sequence of auxiliary-head names to instantiate. Valid
            entries: ``"collision_risk"``, ``"barrier_value"``,
            ``"time_to_collision"``, ``"goal_distance"``. Empty by default
            (no auxiliary heads), matching prior behavior.
        aux_hidden_dim: Hidden width shared by all auxiliary heads.
        strict_observation_validation: If ``True``, :meth:`SafeRLPolicy.
            extract_features` validates observations (shape / missing-key /
            NaN / Inf) before use and raises on failure. Defaults to
            ``False`` to avoid changing prior runtime behavior/performance;
            NaN/Inf sanitization via ``nan_to_num`` still always applies.
    """

    robot_state_dim: int = 4
    goal_dim: int = 2
    lidar_dim: int = 0
    obstacle_dim: int = 0
    encoder_hidden: int = 128
    encoder_out: int = 64
    head_hidden: Sequence[int] = field(default_factory=lambda: _DEFAULT_HEAD_HIDDEN)
    action_dim: int = 2
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    action_bounds: ActionBounds = field(default_factory=ActionBounds)
    normalize_obs: bool = True
    use_goal_conditioning: bool = True
    orthogonal_gain_hidden: float = math.sqrt(2.0)
    orthogonal_gain_output: float = 0.01
    observation_keys: ObservationKeys = field(default_factory=ObservationKeys)
    critic_type: str = "v"

    # -- distribution / exploration (new, opt-in) -------------------------- #
    distribution_type: str = "squashed_gaussian"
    use_sde: bool = False
    sde_sample_freq: int = -1

    # -- SAC automatic entropy tuning (new, opt-in) ------------------------- #
    use_auto_entropy_tuning: bool = False
    initial_alpha: float = 1.0
    target_entropy: Optional[float] = None

    # -- CNN image/depth/map encoder (new, opt-in) -------------------------- #
    image_shape: Optional[Tuple[int, int, int]] = None
    cnn_feature_dim: int = 128

    # -- attention-based dynamic obstacle encoder (new, opt-in) -------------- #
    use_attention_obstacles: bool = False
    max_obstacles: int = 16
    obstacle_feature_dim: int = 5
    attention_heads: int = 4

    # -- recurrent trunk (new, opt-in) --------------------------------------- #
    use_recurrent: bool = False
    recurrent_type: str = "lstm"
    recurrent_hidden_size: int = 256
    recurrent_num_layers: int = 1

    # -- auxiliary prediction heads (new, opt-in) ---------------------------- #
    aux_heads: Sequence[str] = field(default_factory=tuple)
    aux_hidden_dim: int = 64

    # -- observation validation (new, opt-in) --------------------------------- #
    strict_observation_validation: bool = False


# --------------------------------------------------------------------------- #
# Initialization helpers
# --------------------------------------------------------------------------- #

def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Apply orthogonal initialization to ``nn.Linear`` layers in-place.

    Weights are initialized with :func:`torch.nn.init.orthogonal_` using the
    supplied ``gain``; biases are initialized to zero.

    Args:
        module: Module (or submodule) to initialize. Non-``Linear`` modules
            are ignored.
        gain: Orthogonal-init gain. Use ``sqrt(2)`` for ReLU hidden layers
            and a small value (e.g. ``0.01``) for policy output layers.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_orthogonal_init(net: nn.Module, hidden_gain: float, output_gain: float) -> None:
    """Recursively orthogonal-initialize a network's ``Linear`` layers.

    The last ``Linear`` layer encountered in module iteration order is
    treated as the output layer and receives ``output_gain``; all others
    receive ``hidden_gain``.

    Args:
        net: Network to initialize.
        hidden_gain: Gain applied to all hidden ``Linear`` layers.
        output_gain: Gain applied to the final ``Linear`` layer.
    """
    linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        gain = output_gain if i == len(linear_layers) - 1 else hidden_gain
        orthogonal_init(layer, gain=gain)


# --------------------------------------------------------------------------- #
# Action scaling utilities
# --------------------------------------------------------------------------- #

def scale_action(unscaled: Tensor, low: Tensor, high: Tensor, eps: float = EPS) -> Tensor:
    """Affinely maps a value in ``[-1, 1]`` into ``[low, high]``.

    Reusable, allocation-light utility shared by every actor head and by
    external code (e.g. a CBF safety filter) that needs to convert between
    the policy's normalized action space and physical units.

    Args:
        unscaled: Tensor of shape ``[..., action_dim]`` with values assumed
            to lie in ``[-1, 1]`` (e.g. the output of ``tanh``).
        low: Physical lower bound, broadcastable to ``unscaled``'s shape.
        high: Physical upper bound, broadcastable to ``unscaled``'s shape.
        eps: Unused numerical-stability placeholder kept for API symmetry
            with :func:`unscale_action`.

    Returns:
        Tensor of the same shape as ``unscaled``, in ``[low, high]``.
    """
    del eps  # No division here; kept for a symmetric function signature.
    center = (high + low) / 2.0
    half_range = (high - low) / 2.0
    return center + half_range * unscaled


def unscale_action(scaled: Tensor, low: Tensor, high: Tensor, eps: float = EPS) -> Tensor:
    """Inverse of :func:`scale_action`: maps ``[low, high]`` back to ``[-1, 1]``.

    Args:
        scaled: Tensor of shape ``[..., action_dim]`` in physical units.
        low: Physical lower bound, broadcastable to ``scaled``'s shape.
        high: Physical upper bound, broadcastable to ``scaled``'s shape.
        eps: Numerical-stability constant preventing division by zero when
            ``high == low`` for a degenerate action dimension.

    Returns:
        Tensor of the same shape as ``scaled``, clamped to
        ``[-1 + eps, 1 - eps]`` so a subsequent ``atanh`` stays finite.
    """
    center = (high + low) / 2.0
    half_range = (high - low) / 2.0
    unscaled = (scaled - center) / (half_range + eps)
    return unscaled.clamp(-1.0 + eps, 1.0 - eps)


# --------------------------------------------------------------------------- #
# Observation validation
# --------------------------------------------------------------------------- #

def validate_observations(
    observations: Dict[str, Tensor], config: "PolicyConfig"
) -> None:
    """Validates a structured observation dict against ``config``.

    Checks, for every modality enabled in ``config``, that the corresponding
    key is present, that its last dimension matches the configured
    dimensionality, and that it contains no ``NaN``/``Inf`` values.

    Args:
        observations: Structured observation dict to validate.
        config: Policy configuration describing the expected layout.

    Raises:
        KeyError: If an enabled modality's key is missing from
            ``observations``.
        ValueError: If a tensor's last dimension does not match the
            configured dimensionality, or if it contains ``NaN``/``Inf``.
    """
    keys = config.observation_keys
    expected: List[Tuple[str, int]] = []
    if config.robot_state_dim > 0:
        expected.append((keys.robot, config.robot_state_dim))
    if config.goal_dim > 0:
        expected.append((keys.goal, config.goal_dim))
    if config.lidar_dim > 0:
        expected.append((keys.lidar, config.lidar_dim))
    if config.obstacle_dim > 0:
        expected.append((keys.obstacles, config.obstacle_dim))
    if config.use_attention_obstacles:
        expected.append((keys.obstacle_set, config.obstacle_feature_dim))

    for key, expected_dim in expected:
        if key not in observations:
            raise KeyError(f"Missing required observation key: {key!r}")
        tensor = observations[key]
        if tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"Observation {key!r} has last dimension {tensor.shape[-1]}, "
                f"expected {expected_dim}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Observation {key!r} contains NaN or Inf values.")

    if config.image_shape is not None and keys.image in observations:
        image = observations[keys.image]
        if tuple(image.shape[-3:]) != tuple(config.image_shape):
            raise ValueError(
                f"Observation {keys.image!r} has trailing shape "
                f"{tuple(image.shape[-3:])}, expected {config.image_shape}."
            )
        if not torch.isfinite(image).all():
            raise ValueError(f"Observation {keys.image!r} contains NaN or Inf values.")


# --------------------------------------------------------------------------- #
# Observation normalization
# --------------------------------------------------------------------------- #

class RunningNormalizer(nn.Module):
    """Optional running mean/variance observation normalizer.

    Maintains streaming estimates of the mean and variance of incoming
    vectors (Welford-style) and normalizes new observations accordingly.
    Normalization can be disabled at any time via :attr:`enabled` without
    losing the running statistics.

    Attributes:
        enabled: Whether normalization is currently applied.
    """

    def __init__(self, dim: int, enabled: bool = True, epsilon: float = EPS) -> None:
        """Initializes the normalizer.

        Args:
            dim: Dimensionality of the vectors to normalize.
            enabled: Initial enabled/disabled state.
            epsilon: Numerical-stability constant added to the variance.
        """
        super().__init__()
        self.dim = dim
        self.enabled = enabled
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, batch: Tensor) -> None:
        """Updates running statistics with a batch of observations.

        Args:
            batch: Tensor of shape ``[..., dim]``.
        """
        flat = batch.reshape(-1, self.dim).to(device=self.running_mean.device, dtype=self.running_mean.dtype)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(flat.shape[0]), device=self.running_mean.device)

        total_count = self.count + batch_count
        delta = batch_mean - self.running_mean
        new_mean = self.running_mean + delta * (batch_count / total_count.clamp_min(self.epsilon))

        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        delta_sq = delta.pow(2) * self.count * batch_count / total_count.clamp_min(self.epsilon)
        new_var = (m_a + m_b + delta_sq) / total_count.clamp_min(self.epsilon)

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(total_count)

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes ``x`` if enabled, otherwise returns it unchanged.

        Args:
            x: Input tensor of shape ``[..., dim]``.

        Returns:
            Normalized (or passthrough) tensor, NaN/Inf-safe.
        """
        if not self.enabled:
            return x
        std = torch.sqrt(self.running_var.clamp_min(self.epsilon))
        normalized = (x - self.running_mean) / (std + self.epsilon)
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables normalization.

        Args:
            enabled: New enabled state.
        """
        self.enabled = enabled


# --------------------------------------------------------------------------- #
# Residual MLP block
# --------------------------------------------------------------------------- #

class ResidualMLPBlock(nn.Module):
    """A single residual block: ``Linear -> LayerNorm -> ReLU -> Linear -> (+skip) -> ReLU``.

    Used to build deep MLP trunks without vanishing gradients. Requires the
    input and output width to match so the skip connection is a plain
    element-wise addition.
    """

    def __init__(self, width: int) -> None:
        """Initializes the residual block.

        Args:
            width: Input/output feature width (must match for the skip add).
        """
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the residual block.

        Args:
            x: Input tensor of shape ``[..., width]``.

        Returns:
            Output tensor of shape ``[..., width]``.
        """
        residual = x
        out = F.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return F.relu(out + residual)


def build_mlp_trunk(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Module:
    """Builds an MLP trunk: a projection layer followed by residual blocks.

    If ``len(hidden_sizes) <= 2``, a plain ``Linear -> LayerNorm -> ReLU``
    stack is used per layer (no residual connections, per the shallow-network
    case). If more than two hidden layers are requested, residual skip
    connections are used to avoid vanishing gradients, with all residual
    blocks sharing the width of ``hidden_sizes[0]``.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_sizes: Sequence of hidden widths.

    Returns:
        An ``nn.Sequential`` (or ``nn.Module``) mapping ``[..., input_dim]``
        to ``[..., hidden_sizes[-1]]``.
    """
    if len(hidden_sizes) == 0:
        return nn.Identity()

    if len(hidden_sizes) <= 2:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            prev_dim = width
        return nn.Sequential(*layers)

    width = hidden_sizes[0]
    layers = [nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU()]
    for _ in hidden_sizes[1:]:
        layers.append(ResidualMLPBlock(width))
    # If the requested final hidden width differs from the residual block
    # width, append a projection down to the requested output width so the
    # trunk's output dimension matches ``hidden_sizes[-1]`` as callers
    # expect (actor/critic heads assume this). This preserves residual
    # structure while providing the correct final width.
    if hidden_sizes[-1] != width:
        layers.append(nn.Linear(width, hidden_sizes[-1]))
        layers.append(nn.LayerNorm(hidden_sizes[-1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Modality encoders
# --------------------------------------------------------------------------- #

class RobotStateEncoder(nn.Module):
    """Encodes raw proprioceptive robot state into a fixed-width embedding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initializes the encoder.

        Args:
            input_dim: Raw robot-state dimensionality.
            hidden_dim: Hidden width.
            output_dim: Output embedding width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encodes robot state.

        Args:
            x: Tensor of shape ``[..., input_dim]``.

        Returns:
            Embedding of shape ``[..., output_dim]``.
        """
        return self.net(x)


class GoalEncoder(nn.Module):
    """Encodes goal information, optionally as distance/heading features.

    When ``use_goal_conditioning`` is ``True`` and the raw goal vector is a
    planar ``[dx, dy]`` (i.e. ``goal_dim == 2``), the raw coordinates are
    augmented with the derived Euclidean distance and heading angle
    (encoded as ``sin``/``cos`` for continuity) before being passed through 
    the MLP. This tends to be an easier feature space for the policy to exploit 
    than raw coordinates alone.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_goal_conditioning: bool = True,
        epsilon: float = EPS,
    ) -> None:
        """Initializes the goal encoder.

        Args:
            input_dim: Raw goal-vector dimensionality.
            hidden_dim: Hidden width.
            output_dim: Output embedding width.
            use_goal_conditioning: Whether to derive distance/heading
                features (only applies when ``input_dim == 2``).
            epsilon: Numerical-stability constant.
        """
        super().__init__()
        self._derive_features = use_goal_conditioning and input_dim == 2
        self.epsilon = epsilon
        effective_dim = input_dim + 3 if self._derive_features else input_dim
        self.net = nn.Sequential(
            nn.Linear(effective_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def _augment(self, goal: Tensor) -> Tensor:
        """Appends distance and sin/cos-heading features to a planar goal.

        Args:
            goal: Tensor of shape ``[..., 2]`` holding ``[dx, dy]``.

        Returns:
            Tensor of shape ``[..., 5]``: ``[dx, dy, distance, sin(h), cos(h)]``.
        """
        dx = goal[..., 0]
        dy = goal[..., 1]
        distance = torch.sqrt(dx.pow(2) + dy.pow(2) + self.epsilon)
        # Note: exactly at the goal (dx=dy=0), sin_h and cos_h become 0 rather than a true unit vector.
        # This is safe and avoids NaNs, while maintaining the unit-circle invariant everywhere else.
        sin_h = dy / distance
        cos_h = dx / distance
        features = torch.stack([distance, sin_h, cos_h], dim=-1)
        return torch.cat([goal, features], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes the goal vector.

        Args:
            x: Tensor of shape ``[..., input_dim]``.

        Returns:
            Embedding of shape ``[..., output_dim]``.
        """
        if self._derive_features:
            x = self._augment(x)
        return self.net(x)


class ObstacleEncoder(nn.Module):
    """Encodes lidar/laser scans and/or tracked-obstacle feature vectors."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initializes the encoder.

        Args:
            input_dim: Raw obstacle/lidar feature dimensionality.
            hidden_dim: Hidden width.
            output_dim: Output embedding width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encodes obstacle/lidar features.

        Args:
            x: Tensor of shape ``[..., input_dim]``.

        Returns:
            Embedding of shape ``[..., output_dim]``.
        """
        return self.net(x)


class CNNEncoder(nn.Module):
    """Convolutional encoder for image/depth/occupancy-map observations.

    A compact 3-layer CNN (in the spirit of the Nature-DQN / SB3
    ``NatureCNN`` trunk) followed by a linear projection to
    ``output_dim``. Input is expected in channel-first ``[C, H, W]``
    format, with pixel-scale values (the encoder does not assume a
    specific normalization; feed already-normalized tensors for best
    results).
    """

    def __init__(self, image_shape: Tuple[int, int, int], output_dim: int) -> None:
        """Initializes the CNN encoder.

        Args:
            image_shape: ``(channels, height, width)`` of the input image.
            output_dim: Output embedding width.

        Raises:
            ValueError: If ``image_shape`` is too small for the fixed
                3-layer convolutional stack.
        """
        super().__init__()
        channels, height, width = image_shape
        if height < 16 or width < 16:
            raise ValueError(
                f"image_shape spatial dims must be >= 16x16, got {height}x{width}."
            )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flat_dim = self.conv(dummy).reshape(1, -1).shape[1]
        self.project = nn.Sequential(
            nn.Linear(flat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encodes a batch of images.

        Args:
            x: Tensor of shape ``[..., C, H, W]``.

        Returns:
            Embedding of shape ``[..., output_dim]``.
        """
        leading_shape = x.shape[:-3]
        flat = x.reshape(-1, *x.shape[-3:])
        conv_out = self.conv(flat).reshape(flat.shape[0], -1)
        embedding = self.project(conv_out)
        return embedding.reshape(*leading_shape, -1)


class AttentionObstacleEncoder(nn.Module):
    """Self-attention encoder for a variable-count set of dynamic obstacles.

    Encodes a padded set of per-obstacle feature vectors (e.g. relative
    position/velocity/radius from CTRV tracking) with a single
    multi-head self-attention layer followed by masked mean pooling, so the
    output is invariant to obstacle ordering and robust to a varying number
    of active obstacles per time step.
    """

    def __init__(
        self,
        obstacle_feature_dim: int,
        output_dim: int,
        num_heads: int = 4,
        epsilon: float = EPS,
    ) -> None:
        """Initializes the attention obstacle encoder.

        Args:
            obstacle_feature_dim: Per-obstacle raw feature width.
            output_dim: Output embedding width (also the attention
                embedding width).
            num_heads: Number of self-attention heads. Must divide
                ``output_dim`` evenly.
            epsilon: Numerical-stability constant for masked pooling.

        Raises:
            ValueError: If ``output_dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim ({output_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.epsilon = epsilon
        self.input_projection = nn.Linear(obstacle_feature_dim, output_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, obstacle_set: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encodes a padded set of obstacles into a single embedding.

        Args:
            obstacle_set: Tensor of shape ``[batch, max_obstacles,
                obstacle_feature_dim]``.
            mask: Optional boolean tensor of shape ``[batch, max_obstacles]``
                where ``True`` marks a *valid* (non-padding) obstacle slot.
                If ``None``, all slots are treated as valid.

        Returns:
            Embedding of shape ``[batch, output_dim]``.
        """
        batch_size, max_obstacles, _ = obstacle_set.shape
        if mask is None:
            mask = obstacle_set.new_ones((batch_size, max_obstacles), dtype=torch.bool)

        # Ensure mask is boolean: replay buffer may return float32 arrays.
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        key_padding_mask = ~mask  # nn.MultiheadAttention masks out True entries.
        # Guard against a fully-masked row, which would otherwise produce NaNs
        # inside softmax; such rows contribute nothing after pooling anyway.
        fully_masked = key_padding_mask.all(dim=-1)
        if fully_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_masked, 0] = False

        projected = self.input_projection(obstacle_set)
        attended, _ = self.attention(
            projected, projected, projected, key_padding_mask=key_padding_mask
        )
        attended = self.norm(attended + projected)

        mask_f = mask.to(attended.dtype).unsqueeze(-1)
        pooled = (attended * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + self.epsilon)
        return self.output_projection(pooled)


class MultiModalFeatureExtractor(nn.Module):
    """Combines per-modality encoders into a single feature vector.

    Enabled modalities are determined by the corresponding ``*_dim`` fields
    in :class:`PolicyConfig` being greater than zero. Disabled modalities are
    simply skipped (no zero-padding is inserted), so the concatenated
    feature width depends on which encoders are active.
    """

    def __init__(self, config: PolicyConfig) -> None:
        """Initializes the feature extractor.

        Args:
            config: Policy configuration describing observation layout.
        """
        super().__init__()
        self.config = config
        self.keys = config.observation_keys

        self.robot_encoder: Optional[nn.Module] = None
        self.goal_encoder: Optional[nn.Module] = None
        self.lidar_encoder: Optional[nn.Module] = None
        self.obstacle_encoder: Optional[nn.Module] = None
        self.image_encoder: Optional[nn.Module] = None
        self.attention_obstacle_encoder: Optional[nn.Module] = None

        out_dim = 0
        if config.robot_state_dim > 0:
            self.robot_encoder = RobotStateEncoder(
                config.robot_state_dim, config.encoder_hidden, config.encoder_out
            )
            out_dim += config.encoder_out
        if config.goal_dim > 0:
            self.goal_encoder = GoalEncoder(
                config.goal_dim,
                config.encoder_hidden,
                config.encoder_out,
                use_goal_conditioning=config.use_goal_conditioning,
            )
            out_dim += config.encoder_out
        if config.lidar_dim > 0:
            self.lidar_encoder = ObstacleEncoder(
                config.lidar_dim, config.encoder_hidden, config.encoder_out
            )
            out_dim += config.encoder_out
        if config.obstacle_dim > 0:
            self.obstacle_encoder = ObstacleEncoder(
                config.obstacle_dim, config.encoder_hidden, config.encoder_out
            )
            out_dim += config.encoder_out
        if config.image_shape is not None:
            self.image_encoder = CNNEncoder(config.image_shape, config.cnn_feature_dim)
            out_dim += config.cnn_feature_dim
        if config.use_attention_obstacles:
            self.attention_obstacle_encoder = AttentionObstacleEncoder(
                config.obstacle_feature_dim,
                config.encoder_out,
                num_heads=config.attention_heads,
            )
            out_dim += config.encoder_out

        if out_dim == 0:
            raise ValueError(
                "At least one of robot_state_dim/goal_dim/lidar_dim/"
                "obstacle_dim/image_shape/use_attention_obstacles must be "
                "enabled."
            )
        self.output_dim = out_dim

    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        """Encodes a structured observation dict into a single feature vector.

        Args:
            observations: Mapping from observation key (see
                :class:`ObservationKeys`) to tensor. Missing keys for
                disabled modalities are fine; missing keys for *enabled*
                modalities raise a ``KeyError``. When
                ``use_attention_obstacles`` is enabled, an optional
                ``obstacle_set_mask`` boolean tensor may be supplied
                alongside ``obstacle_set`` to mark padding slots.

        Returns:
            Concatenated feature tensor of shape ``[..., self.output_dim]``.
        """
        parts: List[Tensor] = []
        if self.robot_encoder is not None:
            parts.append(self.robot_encoder(observations[self.keys.robot]))
        if self.goal_encoder is not None:
            parts.append(self.goal_encoder(observations[self.keys.goal]))
        if self.lidar_encoder is not None:
            parts.append(self.lidar_encoder(observations[self.keys.lidar]))
        if self.obstacle_encoder is not None:
            parts.append(self.obstacle_encoder(observations[self.keys.obstacles]))
        if self.image_encoder is not None:
            parts.append(self.image_encoder(observations[self.keys.image]))
        if self.attention_obstacle_encoder is not None:
            mask = observations.get(self.keys.obstacle_set_mask)
            parts.append(
                self.attention_obstacle_encoder(observations[self.keys.obstacle_set], mask)
            )
        return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------- #
# Entropy coefficient scheduling
# --------------------------------------------------------------------------- #

class EntropyCoefficientScheduler:
    """Schedules the entropy bonus coefficient over training progress.

    Supports constant, linear-decay, and exponential-decay schedules. This
    object is intentionally decoupled from the network so an external
    training loop can advance it independently of ``forward`` calls.
    """

    def __init__(
        self,
        initial_value: float = 0.01,
        final_value: float = 0.0,
        total_steps: int = 1_000_000,
        schedule: str = "linear",
    ) -> None:
        """Initializes the scheduler.

        Args:
            initial_value: Entropy coefficient at ``step == 0``.
            final_value: Entropy coefficient at ``step >= total_steps``.
            total_steps: Number of steps over which to decay.
            schedule: One of ``"constant"``, ``"linear"``, ``"exponential"``.

        Raises:
            ValueError: If ``schedule`` is not a recognized option.
        """
        if schedule not in ("constant", "linear", "exponential"):
            raise ValueError(f"Unknown entropy schedule: {schedule!r}")
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = max(total_steps, 1)
        self.schedule = schedule

    def value(self, step: int) -> float:
        """Returns the entropy coefficient at a given training step.

        Args:
            step: Current training step (e.g. environment steps or updates).

        Returns:
            The scheduled entropy coefficient.
        """
        progress = min(max(step / self.total_steps, 0.0), 1.0)
        if self.schedule == "constant":
            return self.initial_value
        if self.schedule == "linear":
            return self.initial_value + progress * (self.final_value - self.initial_value)
        # exponential
        ratio = (self.final_value + EPS) / (self.initial_value + EPS)
        return self.initial_value * (ratio ** progress)


# --------------------------------------------------------------------------- #
# Reusable action-distribution abstraction
# --------------------------------------------------------------------------- #

class ActionDistribution:
    """Common interface shared by every action-distribution family.

    ``GaussianActorHead`` (and any future SAC/PPO-specific head) programs
    against this interface instead of a concrete ``torch.distributions``
    type, so swapping ``distribution_type`` in :class:`PolicyConfig` does
    not require touching the head or the training loop.
    """

    def sample(self, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
        """Draws (or deterministically computes) a physical-unit action.

        Args:
            deterministic: If ``True``, return the distribution mode/mean
                instead of a reparameterized sample.

        Returns:
            Tuple ``(action, log_prob)``.
        """
        raise NotImplementedError

    def log_prob(self, action: Tensor) -> Tensor:
        """Computes the log-probability of a physical-unit action.

        Args:
            action: Physical-unit action tensor.

        Returns:
            Log-probability tensor with the batch shape (last dim reduced).
        """
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Returns the (possibly approximate) entropy of the distribution."""
        raise NotImplementedError


class GaussianDistribution(ActionDistribution):
    """Unsquashed Gaussian action distribution.

    The raw Gaussian sample is used directly as the action (only clamped,
    never hard-clipped mid-graph via ``.detach()``, so gradients still flow
    through the clamp's active region). Suitable for tasks where the CBF
    safety filter -- not the policy -- is expected to enforce hard bounds.
    """

    def __init__(self, mean: Tensor, log_std: Tensor, low: Tensor, high: Tensor) -> None:
        """Initializes the distribution.

        Args:
            mean: Action mean, shape ``[..., action_dim]``.
            log_std: Log standard deviation, shape ``[..., action_dim]``.
            low: Physical lower bound, broadcastable to ``mean``.
            high: Physical upper bound, broadcastable to ``mean``.
        """
        self.dist = Normal(mean, log_std.exp())
        self.low = low
        self.high = high

    def sample(self, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
        frozen_noise = getattr(self, "frozen_noise", None)
        if frozen_noise is not None:
            raw = self.dist.mean if deterministic else self.dist.mean + frozen_noise
        else:
            raw = self.dist.mean if deterministic else self.dist.rsample()
        action = raw.clamp(self.low, self.high)
        log_prob = self.dist.log_prob(raw).sum(dim=-1)
        return action, log_prob

    def log_prob(self, action: Tensor) -> Tensor:
        return self.dist.log_prob(action).sum(dim=-1)

    def entropy(self) -> Tensor:
        return self.dist.entropy().sum(dim=-1)


class SquashedGaussianDistribution(ActionDistribution):
    """Tanh-squashed Gaussian action distribution (the prior default).

    Reparameterized samples are passed through ``tanh`` and affinely
    rescaled into ``[low, high]`` via :func:`scale_action`, with the
    standard SAC-style log-prob Jacobian correction. Fully differentiable;
    no hard clipping.
    """

    def __init__(
        self, mean: Tensor, log_std: Tensor, low: Tensor, high: Tensor, epsilon: float = EPS
    ) -> None:
        """Initializes the distribution.

        Args:
            mean: Pre-tanh action mean, shape ``[..., action_dim]``.
            log_std: Pre-tanh log standard deviation, same shape.
            low: Physical lower bound, broadcastable to ``mean``.
            high: Physical upper bound, broadcastable to ``mean``.
            epsilon: Numerical-stability constant.
        """
        self.dist = Normal(mean, log_std.exp())
        self.low = low
        self.high = high
        self.epsilon = epsilon

    def _scale_correction(self) -> Tensor:
        half_range = (self.high - self.low) / 2.0
        return torch.log(half_range + self.epsilon)

    def sample(self, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
        frozen_noise = getattr(self, "frozen_noise", None)
        if frozen_noise is not None:
            raw = self.dist.mean if deterministic else self.dist.mean + frozen_noise
        else:
            raw = self.dist.mean if deterministic else self.dist.rsample()
        squashed = torch.tanh(raw)
        action = scale_action(squashed, self.low, self.high)
        log_prob = self.dist.log_prob(raw) - torch.log(1.0 - squashed.pow(2) + self.epsilon) - self._scale_correction()
        return action, log_prob.sum(dim=-1)

    def log_prob(self, action: Tensor) -> Tensor:
        squashed = unscale_action(action, self.low, self.high, self.epsilon)
        raw = torch.atanh(squashed)
        log_prob = self.dist.log_prob(raw) - torch.log(1.0 - squashed.pow(2) + self.epsilon) - self._scale_correction()
        return log_prob.sum(dim=-1)

    def entropy(self) -> Tensor:
        # The exact entropy of a tanh-squashed Gaussian has no closed form;
        # the pre-squash Gaussian entropy is used as a standard, cheap
        # upper-bound approximation (matches the prior implementation).
        return self.dist.entropy().sum(dim=-1)


class BetaActionDistribution(ActionDistribution):
    """Beta-distributed action over a bounded support ``[low, high]``.

    Unlike the squashed Gaussian, the Beta distribution has *exact* bounded
    support with no tanh Jacobian correction needed, at the cost of a less
    standard reparameterization gradient (``torch.distributions.Beta`` still
    supports ``rsample`` via implicit reparameterization).
    """

    def __init__(
        self, concentration1: Tensor, concentration0: Tensor, low: Tensor, high: Tensor,
        epsilon: float = EPS,
    ) -> None:
        """Initializes the distribution.

        Args:
            concentration1: Beta ``alpha`` parameter (``> 0``), shape
                ``[..., action_dim]``.
            concentration0: Beta ``beta`` parameter (``> 0``), same shape.
            low: Physical lower bound, broadcastable to ``concentration1``.
            high: Physical upper bound, broadcastable to ``concentration1``.
            epsilon: Numerical-stability constant.
        """
        self.dist = Beta(concentration1, concentration0)
        self.low = low
        self.high = high
        self.epsilon = epsilon

    def sample(self, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
        unit = self.dist.mean if deterministic else self.dist.rsample()
        unit = unit.clamp(self.epsilon, 1.0 - self.epsilon)
        action = self.low + (self.high - self.low) * unit
        log_prob = self.dist.log_prob(unit) - torch.log(self.high - self.low + self.epsilon)
        return action, log_prob.sum(dim=-1)

    def log_prob(self, action: Tensor) -> Tensor:
        unit = ((action - self.low) / (self.high - self.low + self.epsilon)).clamp(
            self.epsilon, 1.0 - self.epsilon
        )
        log_prob = self.dist.log_prob(unit) - torch.log(self.high - self.low + self.epsilon)
        return log_prob.sum(dim=-1)

    def entropy(self) -> Tensor:
        return self.dist.entropy().sum(dim=-1)


# --------------------------------------------------------------------------- #
# Generalized State-Dependent Exploration (gSDE)
# --------------------------------------------------------------------------- #

class StateDependentNoiseMatrix(nn.Module):
    """Generalized State-Dependent Exploration (gSDE) noise generator.

    Instead of sampling a fresh, state-*independent* Gaussian perturbation
    at every step, gSDE samples a single ``[feature_dim, action_dim]`` noise
    matrix once per rollout (or on-demand via :meth:`sample_weights`) and
    derives per-step exploration noise as a linear function of the current
    latent features: ``noise = features @ exploration_matrix``. This
    produces temporally-correlated, state-dependent exploration that is
    smoother for physical robots than i.i.d. per-step noise (Raffin et al.,
    2021).
    """

    def __init__(self, feature_dim: int, action_dim: int, log_std_init: float = -0.5) -> None:
        """Initializes the noise generator.

        Args:
            feature_dim: Width of the latent features the noise matrix acts
                on.
            action_dim: Action dimensionality.
            log_std_init: Initial value of the learnable log standard
                deviation for each ``(feature, action)`` pair.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.full((feature_dim, action_dim), log_std_init))
        self.register_buffer("exploration_matrix", torch.zeros(feature_dim, action_dim))
        self.sample_weights()

    @torch.no_grad()
    def sample_weights(self) -> None:
        """Resamples the frozen exploration matrix from the current log_std."""
        std = self.log_std.exp()
        self.exploration_matrix = torch.randn_like(std) * std

    def get_noise(self, features: Tensor) -> Tensor:
        """Computes state-dependent noise for a batch of features.

        Args:
            features: Tensor of shape ``[..., feature_dim]``.

        Returns:
            Noise tensor of shape ``[..., action_dim]``.
        """
        return features @ self.exploration_matrix.to(features.dtype)

    def variance(self, features: Tensor) -> Tensor:
        """Computes the exact per-action-dim variance of the noise given features.

        Since ``noise = features @ M`` with ``M[i, j] ~ N(0, std[i, j]^2)``
        independently, ``Var(noise_j | features) = sum_i features_i^2 *
        std[i, j]^2``, computed here as a matrix product for efficiency.

        Args:
            features: Tensor of shape ``[..., feature_dim]``.

        Returns:
            Variance tensor of shape ``[..., action_dim]``.
        """
        std2 = self.log_std.exp().pow(2)
        return features.pow(2) @ std2.to(features.dtype)


# --------------------------------------------------------------------------- #
# Actor / Critic heads
# --------------------------------------------------------------------------- #

class GaussianActorHead(nn.Module):
    """Actor head producing a squashed Gaussian action distribution.

    Outputs a pre-tanh mean and a learnable, state-dependent log standard
    deviation. Sampling uses the reparameterization trick and a
    ``tanh``-squash into ``[-1, 1]`` per action dimension, which is then
    affinely rescaled into the physical ``[V_MIN, V_MAX]`` /
    ``[OMEGA_MIN, OMEGA_MAX]`` bounds. This keeps the action fully
    differentiable end-to-end (no hard clipping).

    Optionally supports generalized State-Dependent Exploration (gSDE, via
    ``use_sde=True``) and alternative distribution families (``"gaussian"``,
    ``"beta"``, via ``distribution_type``), both fully backward compatible
    with the default ``distribution_type="squashed_gaussian",
    use_sde=False`` configuration used by all prior call sites.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        action_bounds: ActionBounds,
        log_std_min: float,
        log_std_max: float,
        distribution_type: str = "squashed_gaussian",
        use_sde: bool = False,
    ) -> None:
        """Initializes the actor head.

        Args:
            feature_dim: Width of the shared trunk features.
            action_dim: Action dimensionality (``2`` for ``[v, omega]``).
            action_bounds: Physical action bounds used for tanh rescaling.
            log_std_min: Lower clamp for log standard deviation.
            log_std_max: Upper clamp for log standard deviation.
            distribution_type: One of ``"squashed_gaussian"`` (default),
                ``"gaussian"``, or ``"beta"``. See :class:`PolicyConfig`.
            use_sde: If ``True``, use gSDE noise instead of a per-step
                linear log_std head. Only valid for ``"squashed_gaussian"``
                and ``"gaussian"``.

        Raises:
            ValueError: If ``distribution_type`` is unrecognized, or if
                ``use_sde`` is combined with ``"beta"``.
        """
        super().__init__()
        if distribution_type not in ("squashed_gaussian", "gaussian", "beta"):
            raise ValueError(f"Unknown distribution_type: {distribution_type!r}")
        if use_sde and distribution_type == "beta":
            raise ValueError("use_sde is not supported with distribution_type='beta'.")

        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.distribution_type = distribution_type
        self.use_sde = use_sde

        self.mean_layer: Optional[nn.Linear] = None
        if distribution_type != "beta":
            self.mean_layer = nn.Linear(feature_dim, action_dim)
            apply_orthogonal_init(self.mean_layer, hidden_gain=0.01, output_gain=0.01)

        self.log_std_layer: Optional[nn.Linear] = None
        self.sde_noise: Optional[StateDependentNoiseMatrix] = None
        self.alpha_layer: Optional[nn.Linear] = None
        self.beta_layer: Optional[nn.Linear] = None

        if distribution_type == "beta":
            self.alpha_layer = nn.Linear(feature_dim, action_dim)
            self.beta_layer = nn.Linear(feature_dim, action_dim)
            apply_orthogonal_init(self.alpha_layer, hidden_gain=0.01, output_gain=0.01)
            apply_orthogonal_init(self.beta_layer, hidden_gain=0.01, output_gain=0.01)
        elif use_sde:
            self.sde_noise = StateDependentNoiseMatrix(feature_dim, action_dim)
        else:
            self.log_std_layer = nn.Linear(feature_dim, action_dim)
            apply_orthogonal_init(self.log_std_layer, hidden_gain=0.01, output_gain=0.01)

    def _bounds(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        """Returns the ``(low, high)`` action bound tensors on the right device."""
        return self.action_bounds.as_tensors(device=device, dtype=dtype)

    def _scale_from_tanh(self, squashed: Tensor) -> Tensor:
        """Affinely rescales a ``tanh``-squashed action in ``[-1, 1]``.

        Args:
            squashed: Tensor of shape ``[..., action_dim]`` in ``[-1, 1]``.

        Returns:
            Tensor of shape ``[..., action_dim]`` in physical action bounds.
        """
        low, high = self._bounds(squashed.device, squashed.dtype)
        return scale_action(squashed, low, high)

    def reset_noise(self) -> None:
        """Resamples the gSDE exploration matrix, if gSDE is enabled.

        A no-op when ``use_sde=False``. Intended to be called by the
        training loop at the start of each rollout (or every
        ``sde_sample_freq`` steps).
        """
        if self.sde_noise is not None:
            self.sde_noise.sample_weights()

    def distribution(self, features: Tensor) -> Normal:
        """Builds the pre-squash Gaussian distribution over raw actions.

        Preserved for backward compatibility with code that calls
        ``distribution()`` directly expecting a
        :class:`torch.distributions.Normal`. Only meaningful for
        ``distribution_type in ("squashed_gaussian", "gaussian")``; for
        ``"beta"`` configurations, use :meth:`action_distribution` instead.

        Args:
            features: Shared trunk features of shape ``[..., feature_dim]``.

        Returns:
            A :class:`torch.distributions.Normal` over pre-tanh actions.
        """
        if self.mean_layer is None:
            raise RuntimeError("distribution() is not supported for beta.")
        
        mean = self.mean_layer(features)
        if self.sde_noise is not None:
            variance = self.sde_noise.variance(features).clamp_min(EPS)
            std = variance.sqrt().clamp(
                math.exp(self.log_std_min), math.exp(self.log_std_max)
            )
        else:
            log_std = self.log_std_layer(features).clamp(self.log_std_min, self.log_std_max)
            std = log_std.exp()
        return Normal(mean, std)

    def action_distribution(self, features: Tensor) -> ActionDistribution:
        """Builds the configured :class:`ActionDistribution` for this head.

        This is the distribution-family-agnostic entry point used by
        :meth:`forward` and :meth:`log_prob_of`; use it directly when
        integrating with a custom SAC/PPO training loop that wants the
        richer ``ActionDistribution`` interface (e.g. to add gSDE noise).

        Args:
            features: Shared trunk features of shape ``[..., feature_dim]``.

        Returns:
            An :class:`ActionDistribution` instance.
        """
        low, high = self._bounds(features.device, features.dtype)

        if self.distribution_type == "beta":
            alpha = F.softplus(self.alpha_layer(features)) + 1.0
            beta = F.softplus(self.beta_layer(features)) + 1.0
            return BetaActionDistribution(alpha, beta, low, high)

        mean = self.mean_layer(features)
        frozen_noise: Optional[Tensor] = None
        
        if self.sde_noise is not None:
            frozen_noise = self.sde_noise.get_noise(features)
            variance = self.sde_noise.variance(features).clamp_min(EPS)
            log_std = 0.5 * torch.log(variance).clamp(self.log_std_min, self.log_std_max)
        else:
            log_std = self.log_std_layer(features).clamp(self.log_std_min, self.log_std_max)

        dist_cls = GaussianDistribution if self.distribution_type == "gaussian" else SquashedGaussianDistribution
        dist = dist_cls(mean, log_std, low, high)
        dist.frozen_noise = frozen_noise  # None unless gSDE is enabled
        return dist

    def forward(
        self, features: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Samples (or deterministically computes) an action.

        Args:
            features: Shared trunk features of shape ``[..., feature_dim]``.
            deterministic: If ``True``, use the distribution mean instead of
                sampling (used at evaluation/inference time).

        Returns:
            Tuple ``(action, log_prob, entropy, std)`` where ``action`` has shape
            ``[..., action_dim]`` and is expressed in physical units
            (``u_rl``, prior to any external CBF safety filtering);
            ``log_prob`` and ``entropy`` have shape ``[...]``.
        """
        dist = self.action_distribution(features)
        action, log_prob = dist.sample(deterministic=deterministic)
        entropy = dist.entropy()
        std = getattr(dist.dist, "stddev", None) if hasattr(dist, "dist") else None
        return action, log_prob, entropy, std

    def log_prob_of(self, features: Tensor, action: Tensor) -> Tensor:
        """Computes the log-probability of a previously taken physical action.

        Used by :meth:`SafeRLPolicy.evaluate_actions` for on-policy updates.

        Args:
            features: Shared trunk features of shape ``[..., feature_dim]``.
            action: Physical-unit action of shape ``[..., action_dim]``.

        Returns:
            Log-probability tensor of shape ``[...]``.
        """
        dist = self.action_distribution(features)
        return dist.log_prob(action)


class CriticHead(nn.Module):
    """State-value V(s) head for PPO / A2C."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        """Initializes the critic head.

        Args:
            feature_dim: Width of the shared trunk features.
            hidden_dim: Hidden width of the critic's private layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        apply_orthogonal_init(self.net, hidden_gain=math.sqrt(2.0), output_gain=1.0)

    def forward(self, features: Tensor) -> Tensor:
        """Computes the state-value estimate.

        Args:
            features: Shared trunk features of shape ``[..., feature_dim]``.

        Returns:
            Value tensor of shape ``[...]`` (last dim squeezed).
        """
        return self.net(features).squeeze(-1)


class DoubleQCriticHead(nn.Module):
    """Twin Action-Value Q1(s,a) and Q2(s,a) head for SAC."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        apply_orthogonal_init(self.q1, hidden_gain=math.sqrt(2.0), output_gain=1.0)
        apply_orthogonal_init(self.q2, hidden_gain=math.sqrt(2.0), output_gain=1.0)

    def forward(self, features: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        sa = torch.cat([features, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# --------------------------------------------------------------------------- #
# SAC automatic entropy temperature (alpha) tuning
# --------------------------------------------------------------------------- #

class EntropyTemperature(nn.Module):
    """Learnable SAC entropy temperature (``alpha``) with automatic tuning.

    Maintains ``log_alpha`` as a learnable scalar parameter (optimized in
    log-space for positivity and numerical stability) together with a
    target entropy, and exposes the standard SAC alpha loss:
    ``loss = -log_alpha * (log_prob + target_entropy).detach().mean()``.
    The caller is responsible for constructing an optimizer over
    :attr:`log_alpha` and stepping it using :meth:`loss`.
    """

    def __init__(
        self, action_dim: int, initial_alpha: float = 1.0, target_entropy: Optional[float] = None
    ) -> None:
        """Initializes the entropy temperature module.

        Args:
            action_dim: Action dimensionality, used to derive the default
                target entropy (``-action_dim``) if not given explicitly.
            initial_alpha: Initial value of ``alpha`` (must be ``> 0``).
            target_entropy: Explicit target entropy. Defaults to the
                standard SAC heuristic ``-action_dim``.
        """
        super().__init__()
        self.target_entropy = (
            float(target_entropy) if target_entropy is not None else -float(action_dim)
        )
        initial_log_alpha = math.log(max(initial_alpha, EPS))
        self.log_alpha = nn.Parameter(torch.tensor(initial_log_alpha))

    @property
    def alpha(self) -> Tensor:
        """Returns the current entropy coefficient ``alpha = exp(log_alpha)``."""
        return self.log_alpha.exp()

    def loss(self, log_prob: Tensor) -> Tensor:
        """Computes the SAC temperature loss for a batch of action log-probs.

        Args:
            log_prob: Log-probabilities of actions sampled from the current
                policy, shape ``[...]``. Must be detached from the actor's
                graph by the caller (temperature updates should not
                backpropagate into the actor).

        Returns:
            Scalar temperature loss to minimize.
        """
        return -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()


# --------------------------------------------------------------------------- #
# Recurrent trunk
# --------------------------------------------------------------------------- #

class RecurrentTrunk(nn.Module):
    """Optional LSTM/GRU trunk, drop-in alternative to the feed-forward trunk.

    Preserves the feed-forward calling convention: calling
    ``forward(features)`` with no hidden state behaves like a single-step
    feed-forward trunk (an internal zero initial state is used and
    discarded), so existing single-step call sites keep working unchanged.
    Passing an explicit hidden state (and requesting it back) enables
    proper sequence-aware rollouts.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        recurrent_type: str = "lstm",
    ) -> None:
        """Initializes the recurrent trunk.

        Args:
            input_dim: Width of the incoming (already-encoded) features.
            hidden_size: Recurrent hidden width (also the trunk output
                width).
            num_layers: Number of stacked recurrent layers.
            recurrent_type: One of ``"lstm"`` or ``"gru"``.

        Raises:
            ValueError: If ``recurrent_type`` is not ``"lstm"``/``"gru"``.
        """
        super().__init__()
        if recurrent_type not in ("lstm", "gru"):
            raise ValueError(f"Unknown recurrent_type: {recurrent_type!r}")
        self.recurrent_type = recurrent_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        rnn_cls = nn.LSTM if recurrent_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Builds a zeroed initial hidden state for a given batch size.

        Args:
            batch_size: Number of independent sequences in the batch.
            device: Target device.
            dtype: Target dtype.

        Returns:
            A zero tensor ``(num_layers, batch_size, hidden_size)`` for GRU,
            or a tuple of two such tensors ``(h_0, c_0)`` for LSTM.
        """
        shape = (self.num_layers, batch_size, self.hidden_size)
        h_0 = torch.zeros(shape, device=device, dtype=dtype)
        if self.recurrent_type == "gru":
            return h_0
        c_0 = torch.zeros(shape, device=device, dtype=dtype)
        return h_0, c_0

    def forward(
        self,
        features: Tensor,
        hidden_state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Runs the recurrent trunk over a (possibly single-step) sequence.

        Args:
            features: Tensor of shape ``[batch, seq_len, input_dim]`` or
                ``[batch, input_dim]`` (treated as ``seq_len == 1``).
            hidden_state: Optional previous hidden state as returned by a
                prior call, or :meth:`initial_state`. If ``None``, a zeroed
                initial state is used.

        Returns:
            Tuple ``(output, new_hidden_state)`` where ``output`` has the
            same leading shape as ``features`` with the last dim replaced
            by ``hidden_size``.
        """
        single_step = features.dim() == 2
        if single_step:
            features = features.unsqueeze(1)
        if hidden_state is None:
            hidden_state = self.initial_state(features.shape[0], features.device, features.dtype)
            
        device_type = "cuda" if features.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            features_f = features.float()
            if isinstance(hidden_state, tuple):
                hs_f = (hidden_state[0].float(), hidden_state[1].float())
            else:
                hs_f = hidden_state.float()
                
            output, new_hidden_state = self.rnn(features_f, hs_f)
            
            output = output.to(features.dtype)
            if isinstance(new_hidden_state, tuple):
                new_hidden_state = (new_hidden_state[0].to(features.dtype), new_hidden_state[1].to(features.dtype))
            else:
                new_hidden_state = new_hidden_state.to(features.dtype)
                
        if single_step:
            output = output.squeeze(1)
        return output, new_hidden_state


# --------------------------------------------------------------------------- #
# Auxiliary prediction heads
# --------------------------------------------------------------------------- #

_VALID_AUX_HEADS: Tuple[str, ...] = (
    "collision_risk",
    "barrier_value",
    "time_to_collision",
    "goal_distance",
)


class AuxiliaryHeads(nn.Module):
    """Optional auxiliary prediction heads trained alongside the actor-critic.

    Each enabled head is a small independent MLP mapping shared trunk
    features to a scalar prediction, useful as an auxiliary training signal
    that correlates with (and can regularize toward) the external CBF
    safety filter's own diagnostics -- e.g. supervising ``barrier_value``
    against the filter's live CBF value, or ``collision_risk`` /
    ``time_to_collision`` against logged near-miss events.

    ``collision_risk`` is passed through a sigmoid (probability in
    ``[0, 1]``); the remaining heads output raw (unbounded) scalars.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, enabled_heads: Sequence[str]) -> None:
        """Initializes the enabled auxiliary heads.

        Args:
            feature_dim: Width of the shared trunk features.
            hidden_dim: Hidden width shared by all auxiliary heads.
            enabled_heads: Names of heads to instantiate; see
                ``_VALID_AUX_HEADS`` for valid entries.

        Raises:
            ValueError: If an unrecognized head name is given.
        """
        super().__init__()
        unknown = set(enabled_heads) - set(_VALID_AUX_HEADS)
        if unknown:
            raise ValueError(f"Unknown auxiliary head(s): {sorted(unknown)}")

        self.heads = nn.ModuleDict()
        for name in enabled_heads:
            self.heads[name] = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            apply_orthogonal_init(self.heads[name], hidden_gain=math.sqrt(2.0), output_gain=1.0)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """Computes predictions for every enabled auxiliary head.

        Args:
            features: Shared trunk features of shape ``[..., feature_dim]``.

        Returns:
            Dict mapping head name to a prediction tensor of shape ``[...]``
            (last dim squeezed). Empty if no heads are enabled.
        """
        predictions: Dict[str, Tensor] = {}
        for name, head in self.heads.items():
            value = head(features).squeeze(-1)
            if name == "collision_risk":
                value = torch.sigmoid(value)
            predictions[name] = value
        return predictions


# --------------------------------------------------------------------------- #
# Training-diagnostics utilities
# --------------------------------------------------------------------------- #

def compute_explained_variance(values_pred: Tensor, values_true: Tensor, eps: float = EPS) -> float:
    """Computes the fraction of value-target variance explained by predictions.

    Standard PPO/A2C diagnostic: ``1 - Var[y_true - y_pred] / Var[y_true]``.
    A value near ``1`` indicates the critic tracks the returns well; a value
    near or below ``0`` indicates the critic is uninformative or worse.

    Args:
        values_pred: Predicted values, any shape.
        values_true: Target values (e.g. discounted returns), same shape.
        eps: Numerical-stability constant guarding against a zero-variance
            target batch.

    Returns:
        Explained variance as a Python float.
    """
    y_pred = values_pred.detach().reshape(-1)
    y_true = values_true.detach().reshape(-1)
    var_true = y_true.var(unbiased=False)
    explained = 1.0 - (y_true - y_pred).var(unbiased=False) / (var_true + eps)
    return float(explained.item())


# --------------------------------------------------------------------------- #
# Full policy
# --------------------------------------------------------------------------- #

@dataclass
class PolicyLogStats:
    """Scalar statistics captured on the most recent forward/evaluate call.

    Intended to be forwarded directly to a TensorBoard ``SummaryWriter`` by
    an external training loop, e.g. ``writer.add_scalar(k, v, step)`` for
    ``k, v in stats.as_dict().items()``.
    """

    policy_loss: Optional[float] = None
    entropy: Optional[float] = None
    approx_kl: Optional[float] = None
    action_std: Optional[float] = None
    mean_action: Optional[float] = None
    max_action: Optional[float] = None
    grad_norm: Optional[float] = None
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    explained_variance: Optional[float] = None
    clip_fraction: Optional[float] = None
    alpha: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        """Returns the populated (non-``None``) fields as a flat dict.

        Returns:
            Mapping of stat name to value for every field that has been set.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}


class SafeRLPolicy(_SB3BasePolicy):  # type: ignore[misc]
    """Actor-critic policy producing an unfiltered action ``u_rl``.

    This policy is designed to be paired with an external CBF-based safety
    filter (zeroing CBF / HOCBF / soft QP / CLF / adaptive alpha / CTRV
    prediction / warm-started QP, etc.). ``forward`` and ``predict`` return
    ``u_rl``: the raw policy proposal. The safety filter is responsible for
    computing ``u_safe = filter(u_rl, state, obstacles)`` and it is
    ``u_safe`` that should be applied to the robot. Unless an external
    training script is explicitly designed to do so, this policy is not
    trained on ``u_safe`` directly, since backpropagating through an
    external QP filter is a separate, deliberate architectural choice made
    outside this file.

    The actor and critic share only the multi-modal encoder trunk; final
    heads are fully independent, matching standard actor-critic best
    practice for decorrelating policy and value gradients.

    Example:
        >>> config = PolicyConfig(robot_state_dim=4, goal_dim=2)
        >>> policy = SafeRLPolicy(config)
        >>> obs = {
        ...     "robot_state": torch.zeros(1, 4),
        ...     "goal": torch.zeros(1, 2),
        ... }
        >>> u_rl, value, log_prob = policy(obs)
    """

    def __init__(self, config: Optional[PolicyConfig] = None, **kwargs) -> None:
        """Initializes the policy.

        Args:
            config: Policy configuration. If ``None``, defaults are used.
            **kwargs: Forwarded to ``BasePolicy.__init__`` when
                ``stable_baselines3`` is installed and this class is used
                through its ``ActorCriticPolicy``-style API; ignored
                otherwise.
        """
        has_obs_space = "observation_space" in kwargs
        has_act_space = "action_space" in kwargs
        if _SB3_AVAILABLE and has_obs_space and has_act_space:
            super().__init__(**kwargs)  # type: ignore[misc]
        elif _SB3_AVAILABLE and (has_obs_space or has_act_space):
            raise ValueError(
                "Partial SB3 kwargs given: both 'observation_space' and "
                "'action_space' are required together, or neither."
            )
        else:
            nn.Module.__init__(self)

        self.config = config or PolicyConfig()
        self._deterministic_eval = True

        self.feature_extractor = MultiModalFeatureExtractor(self.config)
        feature_dim = self.feature_extractor.output_dim

        self.obs_normalizers = nn.ModuleDict()
        if self.config.normalize_obs:
            keys = self.config.observation_keys
            if self.config.robot_state_dim > 0:
                self.obs_normalizers[keys.robot] = RunningNormalizer(self.config.robot_state_dim)
            if self.config.goal_dim > 0:
                self.obs_normalizers[keys.goal] = RunningNormalizer(self.config.goal_dim)
            if self.config.lidar_dim > 0:
                self.obs_normalizers[keys.lidar] = RunningNormalizer(self.config.lidar_dim)
            if self.config.obstacle_dim > 0:
                self.obs_normalizers[keys.obstacles] = RunningNormalizer(self.config.obstacle_dim)

        self.use_recurrent = self.config.use_recurrent
        if self.use_recurrent:
            self.trunk: nn.Module = RecurrentTrunk(
                feature_dim,
                self.config.recurrent_hidden_size,
                num_layers=self.config.recurrent_num_layers,
                recurrent_type=self.config.recurrent_type,
            )
            trunk_out_dim = self.config.recurrent_hidden_size
        else:
            self.trunk = build_mlp_trunk(feature_dim, list(self.config.head_hidden))
            trunk_out_dim = (
                self.config.head_hidden[-1] if len(self.config.head_hidden) > 0 else feature_dim
            )

        self.actor = GaussianActorHead(
            trunk_out_dim,
            self.config.action_dim,
            self.config.action_bounds,
            self.config.log_std_min,
            self.config.log_std_max,
            distribution_type=self.config.distribution_type,
            use_sde=self.config.use_sde,
        )
        
        if self.config.critic_type == "q":
            self.critic: nn.Module = DoubleQCriticHead(
                trunk_out_dim, self.config.action_dim, trunk_out_dim
            )
        else:
            self.critic = CriticHead(trunk_out_dim, trunk_out_dim)

        self.aux_heads: Optional[AuxiliaryHeads] = None
        if self.config.aux_heads:
            self.aux_heads = AuxiliaryHeads(
                trunk_out_dim, self.config.aux_hidden_dim, self.config.aux_heads
            )

        self.entropy_temperature: Optional[EntropyTemperature] = None
        if self.config.use_auto_entropy_tuning:
            self.entropy_temperature = EntropyTemperature(
                self.config.action_dim,
                initial_alpha=self.config.initial_alpha,
                target_entropy=self.config.target_entropy,
            )

        apply_orthogonal_init(
            self.feature_extractor,
            hidden_gain=self.config.orthogonal_gain_hidden,
            output_gain=self.config.orthogonal_gain_hidden,
        )
        if not self.use_recurrent:
            apply_orthogonal_init(
                self.trunk,
                hidden_gain=self.config.orthogonal_gain_hidden,
                output_gain=self.config.orthogonal_gain_hidden,
            )

        self.entropy_scheduler = EntropyCoefficientScheduler()
        self.last_stats = PolicyLogStats()

    # -- normalization / feature extraction ------------------------------- #

    def set_obs_normalization(self, enabled: bool) -> None:
        """Enables or disables observation normalization for all modalities.

        Args:
            enabled: New enabled state.
        """
        for normalizer in self.obs_normalizers.values():
            normalizer.set_enabled(enabled)

    @torch.no_grad()
    def update_obs_normalizers(self, observations: Dict[str, Tensor]) -> None:
        """Updates running normalization statistics from a batch of observations.

        Should be called by the training loop (e.g. once per rollout batch)
        rather than automatically on every forward pass, so that evaluation
        or CBF-side queries do not perturb the running statistics.

        Args:
            observations: Structured observation dict.
        """
        for key, normalizer in self.obs_normalizers.items():
            if key in observations:
                normalizer.update(observations[key])

    def _normalize(self, observations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Applies (optional) running normalization to each observation modality.

        Args:
            observations: Raw structured observation dict.

        Returns:
            Dict with the same keys, normalized where a normalizer exists.
        """
        if not self.obs_normalizers:
            return observations
        normalized = dict(observations)
        for key, normalizer in self.obs_normalizers.items():
            if key in normalized:
                normalized[key] = normalizer(normalized[key])
        return normalized

    def extract_features(
        self,
        observations: Dict[str, Tensor],
        hidden_state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        return_hidden_state: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]]:
        """Normalizes observations, encodes them, and passes them through the trunk.

        Args:
            observations: Structured observation dict. If
                ``config.strict_observation_validation`` is ``True``, this
                is validated (shape / missing-key / NaN / Inf) before use.
            hidden_state: Optional recurrent hidden state from a previous
                call. Ignored when ``config.use_recurrent`` is ``False``.
            return_hidden_state: If ``True`` (and ``config.use_recurrent``),
                also returns the updated hidden state. Defaults to ``False``
                so the default return type matches the prior, purely
                feed-forward behavior.

        Returns:
            Trunk feature tensor of shape ``[..., trunk_out_dim]``, or, when
            ``return_hidden_state=True`` and recurrence is enabled, a tuple
            ``(features, new_hidden_state)``.
        """
        if self.config.strict_observation_validation:
            validate_observations(observations, self.config)

        normalized = self._normalize(observations)
        features = self.feature_extractor(normalized)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.use_recurrent:
            features, new_hidden_state = self.trunk(features, hidden_state)
            if return_hidden_state:
                return features, new_hidden_state
            return features

        return self.trunk(features)

    # -- core actor-critic API --------------------------------------------- #

    def forward(
        self,
        observations: Dict[str, Tensor],
        deterministic: Optional[bool] = None,
        hidden_state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        return_hidden_state: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Union[Tensor, Tuple[Tensor, Tensor]]],
    ]:
        """Runs the actor and critic on a batch of observations.

        Args:
            observations: Structured observation dict (see
                :class:`ObservationKeys`).
            deterministic: If ``True``, use the distribution mean for the
                action (no sampling). If ``None``, uses deterministic
                behavior in ``eval()`` mode and stochastic behavior in
                ``train()`` mode, matching standard SB3 conventions.
            hidden_state: Optional recurrent hidden state (only used when
                ``config.use_recurrent`` is ``True``).
            return_hidden_state: If ``True`` (and recurrence is enabled),
                also returns the updated hidden state as a 4th tuple
                element. Defaults to ``False``, preserving the original
                3-tuple return signature for every existing call site.

        Returns:
            Tuple ``(u_rl, value, log_prob)`` (or, with
            ``return_hidden_state=True`` under recurrence, ``(u_rl, value,
            log_prob, new_hidden_state)``):
                * ``u_rl``: proposed action, shape ``[..., action_dim]``,
                  in physical units, **prior to CBF safety filtering**.
                * ``value``: critic value estimate, shape ``[...]``.
                * ``log_prob``: log-probability of ``u_rl`` under the
                  current policy, shape ``[...]``.
        """
        if deterministic is None:
            deterministic = (not self.training) and self._deterministic_eval

        new_hidden_state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None
        if self.use_recurrent:
            trunk_features, new_hidden_state = self.extract_features(
                observations, hidden_state=hidden_state, return_hidden_state=True
            )
        else:
            trunk_features = self.extract_features(observations)

        action, log_prob, entropy, std = self.actor(trunk_features, deterministic=deterministic)
        
        if self.config.critic_type == "q":
            q1, q2 = self.critic(trunk_features, action)
            value = torch.min(q1, q2)
        else:
            value = self.critic(trunk_features)

        self.last_stats.entropy = float(entropy.mean().detach().cpu())
        self.last_stats.mean_action = float(action.mean().detach().cpu())
        self.last_stats.max_action = float(action.abs().max().detach().cpu())
        
        if std is not None:
            self.last_stats.action_std = float(std.mean().detach().cpu())

        if self.use_recurrent and return_hidden_state:
            return action, value, log_prob, new_hidden_state
        return action, value, log_prob

    def evaluate_actions(
        self, observations: Dict[str, Tensor], actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluates previously taken actions for an on-policy update (e.g. PPO).

        Args:
            observations: Structured observation dict.
            actions: Physical-unit actions previously executed (``u_rl``,
                i.e. the action the policy actually proposed -- not
                ``u_safe`` -- unless the training script deliberately
                substitates the filtered action).

        Returns:
            Tuple ``(value, log_prob, entropy)`` matching the shapes
            returned by :meth:`forward`.
        """
        trunk_features = self.extract_features(observations)
        dist = self.actor.action_distribution(trunk_features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        if self.config.critic_type == "q":
            q1, q2 = self.critic(trunk_features, actions)
            value = torch.min(q1, q2)
        else:
            value = self.critic(trunk_features)

        self.last_stats.entropy = float(entropy.mean().detach().cpu())
        return value, log_prob, entropy

    def get_value(self, observations: Dict[str, Tensor]) -> Tensor:
        """Computes only the critic value estimate (no actor sampling).

        Args:
            observations: Structured observation dict.

        Returns:
            Value tensor of shape ``[...]``.
        """
        trunk_features = self.extract_features(observations)
        if self.config.critic_type == "q":
            raise RuntimeError("get_value() is invalid for SAC (critic_type='q') because it requires an action.")
        return self.critic(trunk_features)

    def get_q_values(self, observations: Dict[str, Tensor], action: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes twin Q-values for SAC.

        Args:
            observations: Structured observation dict.
            action: Physical-unit action of shape ``[..., action_dim]``.

        Returns:
            Tuple ``(q1, q2)`` tensors of shape ``[...]``.
        """
        trunk_features = self.extract_features(observations)
        if self.config.critic_type != "q":
            raise RuntimeError("get_q_values() requires critic_type='q'.")
        return self.critic(trunk_features, action)

    def predict_auxiliary(self, observations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Computes predictions from every enabled auxiliary head.

        Args:
            observations: Structured observation dict.

        Returns:
            Dict mapping auxiliary head name to prediction tensor of shape
            ``[...]``. Empty if ``config.aux_heads`` is empty.
        """
        if self.aux_heads is None:
            return {}
        trunk_features = self.extract_features(observations)
        return self.aux_heads(trunk_features)

    def reset_noise(self) -> None:
        """Resamples the gSDE exploration matrix, if ``config.use_sde`` is set.

        A no-op otherwise. Intended to be called once per rollout (PPO) or
        every ``config.sde_sample_freq`` environment steps.
        """
        self.actor.reset_noise()

    @property
    def alpha(self) -> Optional[Tensor]:
        """Returns the current SAC entropy temperature, or ``None`` if disabled."""
        if self.entropy_temperature is None:
            return None
        return self.entropy_temperature.alpha

    def compute_alpha_loss(self, log_prob: Tensor) -> Optional[Tensor]:
        """Computes the SAC automatic entropy-temperature loss, if enabled.

        Args:
            log_prob: Log-probabilities of actions sampled from the current
                policy, shape ``[...]``.

        Returns:
            Scalar temperature loss, or ``None`` if
            ``config.use_auto_entropy_tuning`` is ``False``.
        """
        if self.entropy_temperature is None:
            return None
        loss = self.entropy_temperature.loss(log_prob)
        self.last_stats.alpha = float(self.entropy_temperature.alpha.detach().cpu())
        return loss

    @torch.no_grad()
    def predict(
        self,
        observations: Dict[str, Tensor],
        deterministic: bool = True,
        hidden_state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Union[Tensor, Tuple[Tensor, Tensor]]]]:
        """Inference-time action prediction, mirroring SB3's ``predict`` signature.

        Args:
            observations: Structured observation dict. Tensors are moved to
                the policy's device automatically.
            deterministic: Whether to use the distribution mean (typical for
                evaluation/deployment) or to sample.
            hidden_state: Optional recurrent hidden state from a previous
                call. Ignored when ``config.use_recurrent`` is ``False``.

        Returns:
            Tuple ``(u_rl, state)`` where ``state`` is ``None`` unless
            ``config.use_recurrent`` is ``True`` (in which case it is the
            updated hidden state to pass into the next call); ``u_rl`` must
            still be passed through the external CBF safety filter to
            obtain ``u_safe`` before being sent to the robot.
        """
        device = self.device
        moved = {k: v.to(device) for k, v in observations.items()}
        if self.use_recurrent:
            action, _, _, new_hidden_state = self.forward(
                moved,
                deterministic=deterministic,
                hidden_state=hidden_state,
                return_hidden_state=True,
            )
            return action, new_hidden_state
        action, _, _ = self.forward(moved, deterministic=deterministic)
        return action, None

    def _predict(
        self,
        observation: Dict[str, Tensor],
        deterministic: bool = False,
    ) -> Tensor:
        """Satisfies stable_baselines3's BasePolicy abstract interface.
        
        SB3 calls this internally (e.g. from BasePolicy.predict's numpy/env
        wrapping) when SafeRLPolicy is used as an SB3-compatible policy. It
        simply delegates to this class's own predict(), discarding the
        optional recurrent hidden state SB3 doesn't expect here.
        
        Args:
            observation: Structured observation dict.
            deterministic: Whether to use the distribution mean or sample.
            
        Returns:
            Action tensor, shape ``[..., action_dim]``.
        """
        action, _ = self.predict(observation, deterministic=deterministic)
        return action

    # -- misc utilities ------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        """Returns the device of the policy's parameters (CPU/CUDA/MPS)."""
        return next(self.parameters()).device

    def set_deterministic_eval(self, deterministic: bool) -> None:
        """Configures whether ``eval()`` mode implies deterministic actions.

        Args:
            deterministic: If ``True`` (default), calling ``forward`` in
                ``eval()`` mode without an explicit ``deterministic`` flag
                returns the distribution mean rather than a sample.
        """
        self._deterministic_eval = deterministic

    def gradient_norm(self) -> float:
        """Computes the current global L2 gradient norm across all parameters.

        Intended for logging (e.g. ``writer.add_scalar("grad_norm", ...)``)
        after a ``.backward()`` call and before ``optimizer.step()``.

        Returns:
            The global gradient norm as a Python float, or ``0.0`` if no
            gradients have been computed yet.
        """
        total = torch.zeros(1, device=self.device)
        for param in self.parameters():
            if param.grad is not None:
                total += param.grad.detach().pow(2).sum()
        return float(torch.sqrt(total + EPS).item())

    def get_log_dict(self, step: int = 0) -> Dict[str, float]:
        """Returns the latest logging statistics as a flat dict for TensorBoard.

        Args:
            step: Current training step, used to query the entropy
                coefficient schedule.

        Returns:
            Dict mapping stat name to scalar value, including the current
            scheduled entropy coefficient and gradient norm.
        """
        self.last_stats.grad_norm = self.gradient_norm()
        stats = self.last_stats.as_dict()
        stats["entropy_coef"] = self.entropy_scheduler.value(step)
        return stats

    def record_training_stats(
        self,
        actor_loss: Optional[float] = None,
        critic_loss: Optional[float] = None,
        approx_kl: Optional[float] = None,
        clip_fraction: Optional[float] = None,
        values_pred: Optional[Tensor] = None,
        values_true: Optional[Tensor] = None,
    ) -> None:
        """Records external training-loop diagnostics for :meth:`get_log_dict`.

        The policy itself does not compute an algorithm-specific loss (that
        depends on PPO vs. SAC vs. another algorithm, which lives in the
        training script), so this method lets that external loop attach its
        own scalars to the same logging dict returned by
        :meth:`get_log_dict`.

        Args:
            actor_loss: Actor/policy loss for this update, if available.
            critic_loss: Critic/value loss for this update, if available.
            approx_kl: Approximate KL divergence between old and new
                policies (PPO-style), if available.
            clip_fraction: Fraction of PPO probability ratios that were
                clipped, if available.
            values_pred: Predicted values for explained-variance
                computation. Requires ``values_true`` to also be given.
            values_true: Target values (e.g. returns) for explained-variance
                computation. Requires ``values_pred`` to also be given.
        """
        if actor_loss is not None:
            self.last_stats.actor_loss = float(actor_loss)
        if critic_loss is not None:
            self.last_stats.critic_loss = float(critic_loss)
        if approx_kl is not None:
            self.last_stats.approx_kl = float(approx_kl)
        if clip_fraction is not None:
            self.last_stats.clip_fraction = float(clip_fraction)
        if values_pred is not None and values_true is not None:
            self.last_stats.explained_variance = compute_explained_variance(
                values_pred, values_true
            )

    def save_checkpoint(self, path: str) -> None:
        """Saves the policy's state dict and configuration to a checkpoint file.

        Args:
            path: Destination file path (e.g. ``"checkpoint.pt"``).
        """
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, map_location: Optional[Union[str, torch.device]] = None
    ) -> "SafeRLPolicy":
        """Loads a policy from a checkpoint written by :meth:`save_checkpoint`.

        Args:
            path: Path to the checkpoint file.
            map_location: Optional device remapping, forwarded to
                ``torch.load`` (e.g. ``"cpu"`` to force loading onto CPU).

        Returns:
            A reconstructed :class:`SafeRLPolicy` with weights loaded.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        policy = cls(checkpoint["config"])
        policy.load_state_dict(checkpoint["state_dict"])
        return policy

    def export_onnx(
        self,
        path: str,
        example_observations: Dict[str, Tensor],
        opset_version: int = 17,
    ) -> None:
        """Exports a deterministic-action forward pass to ONNX.

        Wraps the policy in a thin, ONNX-friendly module that always takes
        ``deterministic=True`` and returns only the action tensor (ONNX
        graphs require a fixed, tensor-only input/output signature), since
        ``forward``'s dict input and multi-type output are not directly
        exportable.

        Args:
            path: Destination ``.onnx`` file path.
            example_observations: Example structured observation dict (with
                a representative batch size) used to trace the graph. Keys
                must match ``config.observation_keys`` for every enabled
                modality.
            opset_version: ONNX opset version to target.
        """
        ordered_keys = list(example_observations.keys())
        example_inputs = tuple(example_observations[k] for k in ordered_keys)

        class _ONNXWrapper(nn.Module):
            """Adapts the dict-input policy to a positional-tensor ONNX graph."""

            def __init__(self, policy: "SafeRLPolicy", keys: List[str]) -> None:
                super().__init__()
                self.policy = policy
                self.keys = keys

            def forward(self, *tensors: Tensor) -> Tensor:
                observations = dict(zip(self.keys, tensors))
                action, _, _ = self.policy(observations, deterministic=True)
                return action

        wrapper = _ONNXWrapper(self, ordered_keys).to(self.device).eval()
        torch.onnx.export(
            wrapper,
            example_inputs,
            path,
            input_names=ordered_keys,
            output_names=["action"],
            opset_version=opset_version,
            dynamic_axes={k: {0: "batch"} for k in ordered_keys + ["action"]},
        )

    def compile(self, **compile_kwargs: Any) -> nn.Module:
        """Returns a ``torch.compile``-wrapped version of this policy, if available.

        Falls back to returning ``self`` unmodified on PyTorch versions
        without ``torch.compile`` (pre-2.0), so calling this is always
        safe regardless of the installed PyTorch version.

        Args:
            **compile_kwargs: Forwarded to ``torch.compile`` (e.g.
                ``mode="reduce-overhead"``).

        Returns:
            The compiled module, or ``self`` if ``torch.compile`` is
            unavailable.
        """
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:  # pragma: no cover - depends on torch version.
            return self
        return compile_fn(self, **compile_kwargs)