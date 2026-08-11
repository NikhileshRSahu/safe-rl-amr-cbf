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
    ) -> None:
        """Initializes the buffer and pre-allocates all storage arrays.

        Args:
            observation_space: The Gymnasium observation space from `AMRWarehouseEnv`.
                Expected to be a `spaces.Dict`, but falls back to `spaces.Box` if flattened.
            action_dim: Dimensionality of the action vector (e.g., 2 for [v, omega]).
            max_size: Ring-buffer capacity (number of transitions).
            device: Torch device (e.g., "cpu" or "cuda") that `sample()` returns tensors on.

        Raises:
            ValueError: If `action_dim` or `max_size` are not positive integers.
        """
        if action_dim <= 0 or max_size <= 0:
            raise ValueError(f"action_dim and max_size must be positive. Got {action_dim}, {max_size}.")

        self.action_dim = action_dim
        self.max_size = int(max_size)
        self.device = torch.device(device) if isinstance(device, str) else device

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
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if batch_size > self.size:
            raise ValueError(f"Requested {batch_size} but buffer only holds {self.size}.")

        idx = np.random.randint(0, self.size, size=batch_size)

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

# Alias for backwards compatibility if needed
ReplayBuffer = SafeReplayBuffer