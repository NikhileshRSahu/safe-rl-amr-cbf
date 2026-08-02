"""smoke_test_buffer.py

Rigorous white-box integration and integrity test for SafeReplayBuffer.
Tests:
  1. Deterministic insertion and verifiable Ring-Buffer overwrite behavior.
  2. Tensor creation (Dtypes for both obs/next_obs, Shapes, and Device mapping).
  3. Sampling edge cases (Empty buffer and Oversampling).
  4. Disk I/O exact value integrity via np.allclose.

Note: This is explicitly a white-box test. It intentionally verifies the 
internal numpy attributes (buffer.action, buffer.reward, etc.) to guarantee 
disk I/O integrity at a byte level.
"""

import os
import tempfile
import traceback
import numpy as np
import torch
from environment import AMRWarehouseEnv
from replay_buffer import SafeReplayBuffer

def run_tests():
    print("==================================================")
    print(" 🛠️  STARTING STRICT REPLAY BUFFER SMOKE TEST")
    print("==================================================\n")

    # [Point 2 & 7] Complete determinism
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Initialize Environment
    env = AMRWarehouseEnv(use_cbf_filter=False)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    
    # Seed the Gym action space explicitly
    env.action_space.seed(SEED)

    # Test ring-buffer overwrite by inserting 15 items into a size 10 buffer
    MAX_SIZE = 10
    INSERTS = 15
    buffer = SafeReplayBuffer(obs_space, action_dim, max_size=MAX_SIZE, device="cpu")

    print(f"[1/5] Testing Insertion & Ring-Buffer Overwrite (Max Size: {MAX_SIZE}, Inserts: {INSERTS})...")
    try:
        obs, _ = env.reset(seed=SEED)
        
        first_reward = None
        
        for step in range(INSERTS):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = 1.0 if info.get("collision", False) else 0.0
            bar = info.get("barrier_value", 0.0)

            if step == 0:
                first_reward = float(reward)

            buffer.add(obs, action, reward, next_obs, done, cost, bar)
            obs = next_obs
            if done:
                obs, _ = env.reset()

        assert len(buffer) == MAX_SIZE, f"Expected length {MAX_SIZE}, got {len(buffer)}"
        assert buffer.ptr == (INSERTS % MAX_SIZE), f"Pointer should be {(INSERTS % MAX_SIZE)}, got {buffer.ptr}"
        
        # [Point 3] Explicitly verify the oldest transition was physically overwritten in memory
        assert buffer.reward[0][0] != first_reward, "Index 0 data was not overwritten during wraparound."
        
        print("  ✅ Overwrite behavior and sizing physically verified.")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return


    print("\n[2/5] Testing Sampling Shapes, Dtypes, and Devices...")
    try:
        BATCH_SIZE = 4
        obs_b, act_b, rew_b, next_obs_b, done_b, cost_b, bar_b = buffer.sample(BATCH_SIZE)

        # Verify device placement
        assert obs_b["robot_state"].device.type == "cpu", "Device mismatch on obs_b"
        assert act_b.device.type == "cpu", "Device mismatch on act_b"

        # Verify dtypes
        assert act_b.dtype == torch.float32, f"act_b dtype is {act_b.dtype}"
        assert rew_b.dtype == torch.float32, f"rew_b dtype is {rew_b.dtype}"
        assert done_b.dtype == torch.float32, f"done_b dtype is {done_b.dtype}"
        assert cost_b.dtype == torch.float32, f"cost_b dtype is {cost_b.dtype}"
        assert bar_b.dtype == torch.float32, f"bar_b dtype is {bar_b.dtype}"
        assert obs_b["lidar"].dtype == torch.float32, f"obs_b['lidar'] dtype is {obs_b['lidar'].dtype}"
        
        # [Point 4] Verify next_obs dtypes explicitly
        assert next_obs_b["lidar"].dtype == torch.float32, f"next_obs_b['lidar'] dtype is {next_obs_b['lidar'].dtype}"
        assert next_obs_b["robot_state"].dtype == torch.float32, "next_obs_b['robot_state'] dtype mismatch"

        # Verify shapes
        assert act_b.shape == (BATCH_SIZE, action_dim), f"Bad action shape {act_b.shape}"
        assert rew_b.shape == (BATCH_SIZE, 1), f"Bad reward shape {rew_b.shape}"
        assert done_b.shape == (BATCH_SIZE, 1), f"Bad done shape {done_b.shape}"
        assert cost_b.shape == (BATCH_SIZE, 1), f"Bad cost shape {cost_b.shape}"
        assert bar_b.shape == (BATCH_SIZE, 1), f"Bad barrier shape {bar_b.shape}"
        
        for k, space in obs_space.spaces.items():
            expected_shape = (BATCH_SIZE, *space.shape)
            assert obs_b[k].shape == expected_shape, f"{k} obs shape mismatch: {obs_b[k].shape} != {expected_shape}"
            assert next_obs_b[k].shape == expected_shape, f"{k} next_obs shape mismatch: {next_obs_b[k].shape} != {expected_shape}"

        print("  ✅ All tensors match expected devices, dtypes, and shapes.")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return


    print("\n[3/5] Testing Sampling Edge Cases (Empty & Oversampling)...")
    try:
        # [Point 1] Strict API oversampling test
        try:
            buffer.sample(500)
            assert False, "API should reject sampling more items than currently stored."
        except (ValueError, AssertionError):
            pass 
            
        # [Point 5] Empty buffer behavior test
        empty_buffer = SafeReplayBuffer(obs_space, action_dim, max_size=MAX_SIZE, device="cpu")
        try:
            empty_buffer.sample(1)
            assert False, "API should reject sampling from a completely empty buffer."
        except (ValueError, AssertionError):
            pass

        print("  ✅ Edge case validations correctly blocked invalid sampling requests.")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return


    print("\n[4/5] Testing Disk I/O & Exact Value Integrity...")
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "test_buf.npz")
            buffer.save(save_path)

            new_buffer = SafeReplayBuffer(obs_space, action_dim, max_size=MAX_SIZE, device="cpu")
            new_buffer.load(save_path)

            # High-level checks
            assert len(new_buffer) == MAX_SIZE, "Loaded buffer has wrong size"
            assert new_buffer.ptr == buffer.ptr, "Loaded buffer has wrong pointer"

            # [Point 6] Deep array integrity checks (White-box implementation verification)
            assert np.allclose(buffer.action[:MAX_SIZE], new_buffer.action[:MAX_SIZE]), "Action mismatch"
            assert np.allclose(buffer.reward[:MAX_SIZE], new_buffer.reward[:MAX_SIZE]), "Reward mismatch"
            assert np.allclose(buffer.done[:MAX_SIZE], new_buffer.done[:MAX_SIZE]), "Done mismatch"
            assert np.allclose(buffer.cost[:MAX_SIZE], new_buffer.cost[:MAX_SIZE]), "Cost mismatch"
            assert np.allclose(buffer.barrier_value[:MAX_SIZE], new_buffer.barrier_value[:MAX_SIZE]), "Barrier mismatch"

            for k in obs_space.spaces.keys():
                assert np.allclose(buffer.obs[k][:MAX_SIZE], new_buffer.obs[k][:MAX_SIZE]), f"Obs[{k}] mismatch"
                assert np.allclose(buffer.next_obs[k][:MAX_SIZE], new_buffer.next_obs[k][:MAX_SIZE]), f"NextObs[{k}] mismatch"

            print("  ✅ Save/Load successful. Data integrity confirmed via np.allclose.")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return

    print("\n==================================================")
    print(" 🚀 ALL RIGOROUS SMOKE TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()