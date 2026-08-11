import numpy as np
import torch
from environment import AMRWarehouseEnv
from train_pure_rl import build_policy_config
from safe_sac import SafeSACAgent, SafeSACConfig


def test_replay_buffer_includes_obstacle_keys():
    env = AMRWarehouseEnv(use_cbf_filter=True)
    policy_config = build_policy_config(env)
    # Build agent with base env.observation_space (no manual expansion)
    agent = SafeSACAgent(
        policy_config=policy_config,
        observation_space=env.observation_space,
        action_dim=int(env.action_space.shape[0]),
        sac_config=SafeSACConfig(device="cpu"),
        replay_buffer_size=10,
    )

    # Create an augmented observation and add to buffer
    obs, _ = env.reset(seed=0)
    # Build obstacle_set and mask to match policy_config
    max_obs = int(policy_config.max_obstacles)
    obstacle_set = np.zeros((max_obs, policy_config.obstacle_feature_dim), dtype=np.float32)
    obstacle_set_mask = np.zeros((max_obs,), dtype=bool)
    obs_aug = dict(obs)
    obs_aug[policy_config.observation_keys.obstacle_set] = obstacle_set
    obs_aug[policy_config.observation_keys.obstacle_set_mask] = obstacle_set_mask

    action = env.action_space.sample()
    agent.replay_buffer.add(obs=obs_aug, action=action, reward=0.0, next_obs=obs_aug, done=False)

    # Sample from buffer and ensure keys are present and types preserved
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, cost_batch, barrier_batch = agent.replay_buffer.sample(1)

    assert policy_config.observation_keys.obstacle_set in obs_batch
    assert policy_config.observation_keys.obstacle_set_mask in obs_batch
    mask_tensor = obs_batch[policy_config.observation_keys.obstacle_set_mask]
    assert mask_tensor.dtype == torch.bool or mask_tensor.dtype == torch.uint8

    print('test_replay_buffer_includes_obstacle_keys passed')
