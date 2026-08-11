"""Smoke test for improved RL policy setup."""
import torch
from environment import AMRWarehouseEnv
from policy import PolicyConfig, SafeRLPolicy
from safe_sac import SafeSACAgent, SafeSACConfig

env = AMRWarehouseEnv(use_cbf_filter=True)
obs, info = env.reset(seed=42)

# Build improved policy config
from train_improved import build_policy_config, augment_observation

policy_config = build_policy_config(env)
print(f"Policy config: max_obstacles={policy_config.max_obstacles}, "
      f"use_attention={policy_config.use_attention_obstacles}, "
      f"aux_heads={policy_config.aux_heads}")

agent = SafeSACAgent(
    policy_config=policy_config,
    observation_space=env.observation_space,
    action_dim=int(env.action_space.shape[0]),
    sac_config=SafeSACConfig(device="cpu"),
    replay_buffer_size=1000,
)

# Test action selection with augmented obs
obs_aug = augment_observation(obs, env, policy_config.max_obstacles)
action = agent.select_action(obs_aug, deterministic=True)
print(f"Action shape: {action.shape}, values: {action}")

# Test one step
obs, reward, terminated, truncated, info = env.step(action)
obs_aug = augment_observation(obs, env, policy_config.max_obstacles)
print(f"Step reward: {reward:.2f}, done: {terminated or truncated}")

# Test SAC update
agent.replay_buffer.add(
    obs=obs_aug, action=action, reward=1.0,
    next_obs=obs_aug, done=0.0, cost=0.0, barrier_value=1.0,
)
for _ in range(5):
    obs_tmp, info_tmp = env.reset()
    obs_tmp = augment_observation(obs_tmp, env, policy_config.max_obstacles)
    action_tmp = env.action_space.sample()
    next_obs_tmp, reward_tmp, terminated_tmp, truncated_tmp, info_tmp = env.step(action_tmp)
    next_obs_tmp = augment_observation(next_obs_tmp, env, policy_config.max_obstacles)
    agent.replay_buffer.add(
        obs=obs_tmp, action=action_tmp, reward=reward_tmp,
        next_obs=next_obs_tmp, done=float(terminated_tmp), cost=0.0, barrier_value=1.0,
    )

stats = agent.update(batch_size=4)
print(f"Update stats: {stats}")
print("\nImproved RL+CBF smoke test passed!")
env.close()
