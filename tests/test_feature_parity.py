import numpy as np
import pytest
from collections import deque
import math
import sys
from unittest.mock import MagicMock

class DummyNode:
    def __init__(self, name):
        pass
    def create_subscription(self, *args, **kwargs):
        pass
    def create_publisher(self, *args, **kwargs):
        pass
    def create_timer(self, *args, **kwargs):
        pass
    def get_logger(self):
        return MagicMock()

# Mock rclpy and ROS messages before importing the node
sys.modules['rclpy'] = MagicMock()
mock_node_module = MagicMock()
mock_node_module.Node = DummyNode
sys.modules['rclpy.node'] = mock_node_module
sys.modules['nav_msgs'] = MagicMock()
sys.modules['nav_msgs.msg'] = MagicMock()
sys.modules['sensor_msgs'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['std_msgs'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()

from environment import AMRWarehouseEnv
from ros2_ws.src.amr_safe_rl.amr_safe_rl.feature_extractor_node import FeatureExtractorNode

def test_feature_extraction_parity():
    """Verify feature_extractor_node normalizes identically to environment.py."""
    env = AMRWarehouseEnv(use_cbf_filter=False)
    env.reset(seed=42)
    
    # Take a few steps to populate the LiDAR history and state
    for _ in range(5):
        action = env.action_space.sample()
        _ = env.step(action)
    
    # 1. Grab the current synchronous observation
    obs = env._get_observation()
    env_flat = np.concatenate([
        obs["robot_state"],
        obs["goal"],
        obs["lidar"]
    ])
    
    # 2. Re-create the inputs and pass to the ROS node logic
    node = FeatureExtractorNode()
    robot_state = env.robot_state
    goal_pos = env.goal_pos
    
    node_flat = node.extract_features(robot_state, goal_pos, env.lidar_history)
    
    # 3. Assert they are floating-point equal
    np.testing.assert_allclose(env_flat, node_flat, rtol=1e-5, atol=1e-5,
                               err_msg="Node normalization diverges from environment normalization!")
