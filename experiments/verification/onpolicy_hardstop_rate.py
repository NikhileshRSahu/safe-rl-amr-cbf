import os
import sys
from pathlib import Path
import numpy as np
import torch
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import AMRWarehouseEnv
from evaluate import load_agent, augment_observation
from cbf import recommended_slack_max

def run_onpolicy_evaluation(checkpoint_path: str, num_episodes: int = 50):
    env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
    
    # Enable slack
    cfg = env.cbf_filter.config
    cfg.enable_slack = True
    cfg.slack_max = recommended_slack_max(cfg.gamma, cfg.safety_margin)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent = load_agent(checkpoint_path, env, device=device)
    
    total_timesteps = 0
    total_hard_stops = 0
    episodes_with_hard_stop = 0
    
    print(f"Running {num_episodes} on-policy episodes using SLACK-RELAXED CBF...")
    print(f"slack_max={cfg.slack_max}")
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        if agent.policy_config.use_attention_obstacles:
            obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
        
        done = False
        ep_hard_stops = 0
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if agent.policy_config.use_attention_obstacles:
                obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
            
            done = terminated or truncated
            
            total_timesteps += 1
            # cbf_solver_success is False if both strict and slack QPs fail (tier="hard_stop")
            if not info.get("cbf_solver_success", True):
                ep_hard_stops += 1
                total_hard_stops += 1
                
        if ep_hard_stops > 0:
            episodes_with_hard_stop += 1
            
        print(f"Episode {ep + 1}/{num_episodes} finished. Hard stops in ep: {ep_hard_stops}")
            
    print("\n--- Final Results (Slack Enabled) ---")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Total Hard Stops: {total_hard_stops} ({(total_hard_stops/total_timesteps)*100:.2f}% of timesteps)")
    print(f"Episodes with at least one hard stop: {episodes_with_hard_stop}/{num_episodes} ({(episodes_with_hard_stop/num_episodes)*100:.1f}%)")

if __name__ == "__main__":
    ckpt = "checkpoints/safe_sac_20260802_113318/best_model.pt"
    run_onpolicy_evaluation(ckpt, num_episodes=50)
