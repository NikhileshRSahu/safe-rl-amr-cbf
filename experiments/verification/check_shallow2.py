import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment import AMRWarehouseEnv
import numpy as np
env = AMRWarehouseEnv()
env.reset()
curr_dyn_obs = env.dynamic_obstacles.copy()
env.step((1.0, 1.0))
print('Arrays equal?', np.array_equal(curr_dyn_obs, env.dynamic_obstacles))
