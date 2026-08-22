import os
import sys
from pathlib import Path
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import AMRWarehouseEnv
from evaluate import load_agent, augment_observation
from cbf import recommended_slack_max

def run_slack_distribution_evaluation(checkpoint_path: str, num_episodes: int = 50):
    env = AMRWarehouseEnv(render_mode=None, use_cbf_filter=True)
    
    cfg = env.cbf_filter.config
    cfg.enable_slack = True
    cfg.slack_max = recommended_slack_max(cfg.gamma, cfg.safety_margin)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_agent(checkpoint_path, env, device=device)
    
    deltas_used = []
    hard_stops = 0
    total_timesteps = 0
    
    print(f"Running {num_episodes} on-policy episodes to collect slack distribution...")
    print(f"slack_max={cfg.slack_max}")
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        if agent.policy_config.use_attention_obstacles:
            obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
        
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if agent.policy_config.use_attention_obstacles:
                obs = augment_observation(obs, env, agent.policy_config.max_obstacles)
            
            done = terminated or truncated
            total_timesteps += 1
            
            tier = env.last_cbf_diagnostics.get("tier")
            slack_used = env.last_cbf_diagnostics.get("slack_used")
            
            if tier == "slack" and slack_used is not None:
                deltas_used.append(slack_used)
            elif tier == "hard_stop":
                hard_stops += 1
                
        print(f"Episode {ep + 1}/{num_episodes} finished.")
            
    print("\n=== Slack Usage Distribution ===")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Hard Stops (fallback due to exceeding slack_max): {hard_stops}")
    print(f"Total interventions using slack: {len(deltas_used)}")
    if deltas_used:
        print(f"Average delta used: {np.mean(deltas_used):.6f}")
        print(f"Median delta used: {np.median(deltas_used):.6f}")
        print(f"Max delta used: {np.max(deltas_used):.6f}")
        print(f"95th percentile delta used: {np.percentile(deltas_used, 95):.6f}")
        
        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(deltas_used, bins=50, color='blue', alpha=0.7)
        plt.axvline(cfg.slack_max, color='red', linestyle='dashed', linewidth=2, label=f'slack_max ({cfg.slack_max:.2f})')
        plt.xlabel('Slack Used ($\delta$)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Slack Used During On-Policy Interventions')
        plt.legend()
        out_path = Path(__file__).parent / 'slack_distribution.png'
        plt.savefig(out_path)
        print(f"Histogram saved to {out_path}")

if __name__ == "__main__":
    ckpt = "checkpoints/safe_sac_20260802_113318/best_model.pt"
    run_slack_distribution_evaluation(ckpt, num_episodes=50)
