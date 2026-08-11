Changelog — fixes applied on 2026-08-11

Fixed issues:
- safe_sac.py: Remove stray dict-comprehension in `select_action` causing syntax error. Kept proper obs->tensor conversion loop.
- train_improved.py: Use boolean dtype for `obstacle_set_mask` in observation space and runtime mask.
- policy.py: Ensure `build_mlp_trunk()` projects final trunk output to `hidden_sizes[-1]` when it differs from `hidden_sizes[0]` to avoid head shape mismatches.
- policy.py: Make `AttentionObstacleEncoder.forward()` defensively cast masks to `torch.bool` before bitwise inversion to handle float32 masks returned from replay buffer.
- train_pure_rl.py: Build an expanded observation space including `obstacle_set` and `obstacle_set_mask` when constructing `SafeSACAgent`, so the replay buffer pre-allocates storage for augmented observations.
- baselines.py: Fix LiDAR front-sector min computation to use `np.concatenate` and `np.min`.
- safe_sac.py: Added defensive logic to auto-extend observation space when `policy_config.use_attention_obstacles` is enabled but keys are missing, so the replay buffer won't KeyError on augmented observations.

Added tests:
- tests/test_replay_obstacle_keys.py: Unit test ensuring the replay buffer stores and samples `obstacle_set` and `obstacle_set_mask` correctly when agent is constructed with the base env observation space but policy expects attention obstacles.

Notes:
- All smoke tests and short training/eval/benchmark scripts were executed locally and completed without errors in quick runs.
- If you want, I can create a git branch and open a PR with these changes; I can also run longer training schedules next.
