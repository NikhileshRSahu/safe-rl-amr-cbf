import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment import AMRWarehouseEnv
env = AMRWarehouseEnv()
env.reset()
dyn_obs = env.dynamic_obstacles
print('Type:', type(dyn_obs))
curr_dyn_obs = env.dynamic_obstacles.copy() if env.dynamic_obstacles is not None else []
snapshot = [tuple(o) for o in curr_dyn_obs]
env.step((0.0, 0.0))
after = [tuple(o) for o in curr_dyn_obs]
print('Matches?', snapshot == after)
