"""Quick smoke test for baselines."""
from environment import AMRWarehouseEnv
from baselines import PIDController, PotentialField, AStarPlanner, DWA, RRTPlanner, MPCBaseline, PureCBF

baselines = [
    ("PID", PIDController()),
    ("PotentialField", PotentialField()),
    ("AStar", AStarPlanner()),
    ("DWA", DWA()),
    ("RRT", RRTPlanner()),
    ("MPC", MPCBaseline()),
    ("PureCBF", PureCBF()),
]

for name, ctrl in baselines:
    env = AMRWarehouseEnv(use_cbf_filter=False)
    obs, info = env.reset(seed=42)
    ctrl.reset() if hasattr(ctrl, "reset") else None
    done = False
    steps = 0
    total_reward = 0.0
    while not done and steps < 50:
        action = ctrl.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        total_reward += reward
    print(f"{name:15s} steps={steps:3d} total_reward={total_reward:8.2f} success={info.get('goal_reached', False)} collision={info.get('collision', False)}")
    env.close()

print("\nAll baselines smoke test passed!")
