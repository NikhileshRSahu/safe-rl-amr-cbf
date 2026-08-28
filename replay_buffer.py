"""replay_buffer.py

High-performance, pre-allocated ring-buffer replay memory for the Hybrid
Soft Actor-Critic (SAC) + Control Barrier Function (CBF) safe RL agent.

Design & Integration
------------------
* **Dict Observation Native**: Perfectly matches `AMRWarehouseEnv` and `SafeRLPolicy`. 
  Instead of a single flat array, it pre-allocates a dictionary of NumPy arrays 
  based on the Gymnasium `spaces.Dict` (e.g., `robot_state`, `goal`, `lidar`).
* **Zero Allocation**: Storage is pre-allocated `numpy.ndarray` blocks sized to 
  `max_size` up front. Insertion and sampling are O(batch_size) vector operations.
* **Safety Signals**: Explicit columns for `cost` and `barrier_value`. This allows
  `safe_sac.py` to seamlessly update a Safety Critic or perform Lagrangian 
  multiplier updates without hacking values into the reward signal.
* **Direct PyTorch Outputs**: `sample()` returns nested dictionaries of `torch.Tensor`s 
  already moved to `self.device`. You can directly pass the sampled `obs` dict 
  into `policy(obs)` with zero additional boilerplate in your training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import numpy as np
import torch
from gymnasium import spaces


class SafeReplayBuffer:
    """Pre-allocated, ring-buffer replay memory for continuous-control SAC+CBF.
    
    Natively supports structured dictionary observations to bridge the gap between
    `AMRWarehouseEnv` (which outputs dicts) and `SafeRLPolicy` (which ingests dicts),
    ensuring no shape mismatches or flattening bottlenecks occur during training.
    """

    def __init__(
        self,
        observation_space: Union[spaces.Dict, spaces.Box],
        action_dim: int,
        max_size: int = 1_000_000,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
        robot_state_dim: int = 0,
        max_dynamic_obstacles: int = 0,
        dynamic_obstacle_dim: int = 6,
    ) -> None:
        """Initializes the buffer and pre-allocates all storage arrays.

        Args:
            observation_space: The Gymnasium observation space from `AMRWarehouseEnv`.
                Expected to be a `spaces.Dict`, but falls back to `spaces.Box` if flattened.
            action_dim: Dimensionality of the action vector (e.g., 2 for [v, omega]).
            max_size: Ring-buffer capacity (number of transitions).
            device: Torch device (e.g., "cpu" or "cuda") that `sample()` returns tensors on.
            seed: Optional RNG seed for deterministic sampling.
            robot_state_dim: If > 0, additionally pre-allocates storage for
                the raw, physical-units robot state (x, y, theta, v, omega,
                ...) paired with each `obs` -- i.e. exactly what
                `AMRWarehouseEnv.step()` passed into the CBF filter that
                step (see `info["cbf_robot_state"]`). 0 (default) disables
                this and keeps the buffer's memory footprint identical to
                before for any caller that doesn't need it. This is
                *separate* from the normalized `obs["robot_state"]` already
                stored -- CBF constraint reconstruction needs unnormalized
                physical units and raw obstacle positions the network's
                observation encoding doesn't preserve losslessly.
            max_dynamic_obstacles: Companion to `robot_state_dim`; if both
                are > 0, also pre-allocates storage for the raw dynamic
                obstacle array (see `info["cbf_dynamic_obstacles"]").
            dynamic_obstacle_dim: Column width of the raw dynamic obstacle
                array (6 in this project: x, y, theta, speed, goal_x,
                goal_y -- see `AMRWarehouseEnv._sample_dynamic_obstacles`).

        Raises:
            ValueError: If `action_dim` or `max_size` are not positive integers.
        """
        if action_dim <= 0 or max_size <= 0:
            raise ValueError(f"action_dim and max_size must be positive. Got {action_dim}, {max_size}.")

        self.action_dim = action_dim
        self.max_size = int(max_size)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.rng = np.random.default_rng(seed)

        self.ptr: int = 0
        self.size: int = 0

        # Pre-allocate dictionary arrays if observation_space is a Dict space
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        
        if self.is_dict_obs:
            self.obs_keys = list(observation_space.spaces.keys())
            self.obs = {}
            self.next_obs = {}
            self.obs_dtypes = {}
            for key, space in observation_space.spaces.items():
                # Use the Gym space's declared dtype when pre-allocating storage.
                dtype = getattr(space, "dtype", np.float32)
                np_dtype = np.dtype(dtype)
                self.obs_dtypes[key] = np_dtype
                self.obs[key] = np.zeros((self.max_size, *space.shape), dtype=np_dtype)
                self.next_obs[key] = np.zeros((self.max_size, *space.shape), dtype=np_dtype)
        else:
            # Fallback for flat Box spaces
            self.obs_keys = ["flat"]
            dtype = getattr(observation_space, "dtype", np.float32)
            np_dtype = np.dtype(dtype)
            self.obs_dtypes = {"flat": np_dtype}
            self.obs = {"flat": np.zeros((self.max_size, *observation_space.shape), dtype=np_dtype)}
            self.next_obs = {"flat": np.zeros((self.max_size, *observation_space.shape), dtype=np_dtype)}

        # Pre-allocated standard numpy storage
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

        # Safety-aware extensions
        self.cost = np.zeros((self.max_size, 1), dtype=np.float32)
        self.barrier_value = np.zeros((self.max_size, 1), dtype=np.float32)

        # Raw physical state for reconstructing CBF constraints at actor-
        # update time (see cbf.actor_consistency_loss). Disabled (arrays
        # not allocated) unless both dims are explicitly requested.
        self.store_cbf_state = robot_state_dim > 0 and max_dynamic_obstacles > 0
        self.robot_state_dim = robot_state_dim
        self.max_dynamic_obstacles = max_dynamic_obstacles
        self.dynamic_obstacle_dim = dynamic_obstacle_dim
        if self.store_cbf_state:
            self.robot_state_raw = np.zeros((self.max_size, robot_state_dim), dtype=np.float64)
            self.dynamic_obstacles_raw = np.zeros(
                (self.max_size, max_dynamic_obstacles, dynamic_obstacle_dim), dtype=np.float64
            )

    def __len__(self) -> int:
        """Returns the current number of valid (filled) transitions."""
        return self.size

    def add(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Union[Dict[str, np.ndarray], np.ndarray],
        done: bool,
        cost: float = 0.0,
        barrier_value: float = 0.0,
        robot_state_raw: Optional[np.ndarray] = None,
        dynamic_obstacles_raw: Optional[np.ndarray] = None,
    ) -> None:
        """Stores a single transition, overwriting the oldest one if full.

        Args:
            obs: Current state dict (e.g., {"robot_state": ..., "goal": ..., "lidar": ...}).
            action: Action taken, shape ``(action_dim,)``.
            reward: Scalar reward received.
            next_obs: Resulting next state dict.
            done: Terminal flag (1.0 if episode ended natively, 0.0 otherwise).
            cost: Optional scalar safety cost (e.g., 1.0 if constraint violated, else 0.0).
            barrier_value: Optional Control Barrier Function value ``h(x)`` from the env.
            robot_state_raw: Optional raw physical robot state paired with
                ``obs`` (see ``store_cbf_state``). Required if this buffer
                was constructed with ``robot_state_dim > 0``; silently
                ignored (with zeros stored) otherwise.
            dynamic_obstacles_raw: Optional raw dynamic obstacle array
                paired with ``obs``. Same conditions as above.
        """
        idx = self.ptr

        # Insert structured observations
        if self.is_dict_obs:
            for key in self.obs_keys:
                self.obs[key][idx] = obs[key]
                self.next_obs[key][idx] = next_obs[key]
        else:
            self.obs["flat"][idx] = obs
            self.next_obs["flat"][idx] = next_obs

        # Insert transition data
        self.action[idx] = action
        self.reward[idx] = float(reward)
        self.done[idx] = float(done)
        
        # Insert safety data
        self.cost[idx] = float(cost)
        self.barrier_value[idx] = float(barrier_value)

        if self.store_cbf_state:
            if robot_state_raw is not None:
                self.robot_state_raw[idx] = robot_state_raw
            if dynamic_obstacles_raw is not None:
                self.dynamic_obstacles_raw[idx] = dynamic_obstacles_raw

        # Advance ring buffer pointers
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[
        Dict[str, torch.Tensor],  # obs_batch
        torch.Tensor,             # action_batch
        torch.Tensor,             # reward_batch
        Dict[str, torch.Tensor],  # next_obs_batch
        torch.Tensor,             # done_batch
        torch.Tensor,             # cost_batch
        torch.Tensor              # barrier_value_batch
    ]:
        """Samples a random batch of transitions, returning Tensors ready for the policy.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A tuple containing batched PyTorch tensors explicitly moved to `self.device`.
            `obs` and `next_obs` are returned as Dictionaries of Tensors, which can be 
            passed directly to `policy(obs)`.
        """
        idx = self._sample_indices(batch_size)
        return self._batch_from_indices(idx)

    def sample_with_cbf_state(self, batch_size: int) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor],
        torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray,
    ]:
        """Same as :meth:`sample`, plus the raw physical state needed to
        reconstruct CBF constraints for the actor-consistency regularizer.

        Only usable when the buffer was constructed with
        ``robot_state_dim > 0`` and ``max_dynamic_obstacles > 0``.

        Returns:
            The same 7-tuple as :meth:`sample`, followed by
            ``(robot_state_raw, dynamic_obstacles_raw)`` as plain NumPy
            arrays (not torch tensors -- ``cbf.py``'s constraint assembly
            is NumPy/SciPy code, not a torch graph; only the action tensor
            passed into ``actor_consistency_loss`` needs to carry
            gradients).

        Raises:
            RuntimeError: If this buffer was not constructed with CBF-state
                storage enabled.
        """
        if not self.store_cbf_state:
            raise RuntimeError(
                "sample_with_cbf_state() requires the buffer to be constructed with "
                "robot_state_dim > 0 and max_dynamic_obstacles > 0."
            )
        idx = self._sample_indices(batch_size)
        batch = self._batch_from_indices(idx)
        return batch + (self.robot_state_raw[idx].copy(), self.dynamic_obstacles_raw[idx].copy())

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if batch_size > self.size:
            raise ValueError(f"Requested {batch_size} but buffer only holds {self.size}.")
        return self.rng.integers(0, self.size, size=batch_size)

    def _batch_from_indices(self, idx: np.ndarray) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor],
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        # Batch structural dicts directly to PyTorch tensors on the correct device
        obs_batch = {
            key: (
                torch.as_tensor(self.obs[key][idx], dtype=torch.bool, device=self.device)
                if np.issubdtype(self.obs_dtypes[key], np.bool_)
                else torch.as_tensor(self.obs[key][idx], dtype=torch.float32, device=self.device)
            )
            for key in self.obs_keys
        }
        next_obs_batch = {
            key: (
                torch.as_tensor(self.next_obs[key][idx], dtype=torch.bool, device=self.device)
                if np.issubdtype(self.obs_dtypes[key], np.bool_)
                else torch.as_tensor(self.next_obs[key][idx], dtype=torch.float32, device=self.device)
            )
            for key in self.obs_keys
        }

        # If it was a flat space fallback, unwrap the dict so the caller gets a raw Tensor
        if not self.is_dict_obs:
            obs_batch = obs_batch["flat"]
            next_obs_batch = next_obs_batch["flat"]

        # Batch flat metadata to PyTorch tensors
        action_batch = torch.as_tensor(self.action[idx], dtype=torch.float32, device=self.device)
        reward_batch = torch.as_tensor(self.reward[idx], dtype=torch.float32, device=self.device)
        done_batch = torch.as_tensor(self.done[idx], dtype=torch.float32, device=self.device)
        cost_batch = torch.as_tensor(self.cost[idx], dtype=torch.float32, device=self.device)
        barrier_batch = torch.as_tensor(self.barrier_value[idx], dtype=torch.float32, device=self.device)

        return (
            obs_batch, 
            action_batch, 
            reward_batch, 
            next_obs_batch, 
            done_batch, 
            cost_batch, 
            barrier_batch
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves current buffer state to a compressed NumPy file (.npz)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "ptr": np.array(self.ptr),
            "size": np.array(self.size),
            "action": self.action[:self.size],
            "reward": self.reward[:self.size],
            "done": self.done[:self.size],
            "cost": self.cost[:self.size],
            "barrier_value": self.barrier_value[:self.size],
        }

        for key in self.obs_keys:
            save_dict[f"obs_{key}"] = self.obs[key][:self.size]
            save_dict[f"next_obs_{key}"] = self.next_obs[key][:self.size]

        if self.store_cbf_state:
            save_dict["robot_state_raw"] = self.robot_state_raw[:self.size]
            save_dict["dynamic_obstacles_raw"] = self.dynamic_obstacles_raw[:self.size]

        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads buffer arrays from a saved .npz file."""
        data = np.load(filepath)
        
        loaded_size = int(data["size"])
        assert loaded_size <= self.max_size, "Loaded size exceeds allocated max_size."

        self.ptr = int(data["ptr"])
        self.size = loaded_size

        self.action[:self.size] = data["action"]
        self.reward[:self.size] = data["reward"]
        self.done[:self.size] = data["done"]
        self.cost[:self.size] = data["cost"]
        self.barrier_value[:self.size] = data["barrier_value"]

        for key in self.obs_keys:
            self.obs[key][:self.size] = data[f"obs_{key}"]
            self.next_obs[key][:self.size] = data[f"next_obs_{key}"]

        if self.store_cbf_state and "robot_state_raw" in data:
            self.robot_state_raw[:self.size] = data["robot_state_raw"]
            self.dynamic_obstacles_raw[:self.size] = data["dynamic_obstacles_raw"]

# Alias for backwards compatibility if needed
ReplayBuffer = SafeReplayBuffer