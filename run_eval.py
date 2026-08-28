import torch
from pathlib import Path
from environment import AMRWarehouseEnv
from safe_sac import SafeSACAgent, SafeSACConfig
from train import evaluate_trajectories
from policy import PolicyConfig
import sys

def main():
    ckpt_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    print(f"Evaluating {ckpt_path} -> {output_dir}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    env = AMRWarehouseEnv()
    
    # We just need to reconstruct the agent enough to call select_action.
    sac_cfg = SafeSACConfig(device="cpu")
    agent = SafeSACAgent(
        policy_config=ckpt['policy_config'],
        observation_space=env.observation_space,
        action_dim=2,
        sac_config=sac_cfg,
        replay_buffer_size=10, # small buffer
    )
    
    agent.policy.load_state_dict(ckpt['policy_state_dict'])
    
    evaluate_trajectories(
        agent=agent,
        eval_env=env,
        num_episodes=40,
        seed=42,
        output_dir=output_dir
    )
    print("Done evaluation.")

if __name__ == "__main__":
    main()
