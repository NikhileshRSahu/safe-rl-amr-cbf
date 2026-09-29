# Hierarchical Safe Reinforcement Learning for AMR Navigation

Safe autonomous mobile robot navigation in dynamic warehouse environments using:

- Soft Actor-Critic (SAC)
- Hindsight Experience Replay (HER)
- Time-Varying Control Barrier Functions (CBF)
- Circulation-Based Deadlock Mitigation
- Hard Tangential Escape Constraints
- Selective Predictive Recovery
- Curriculum Learning

## Final Simulation Result

Evaluation on 200 unseen warehouse episodes:

- Success Rate: 93.0%
- Collision Rate: 0.0%
- Timeout Rate: 7.0%
- Successful Episodes: 186 / 200
- Collisions: 0 / 200

## Architecture

SAC Policy
    |
    v
Time-Varying CBF-QP
    |
    +-- Circulation
    |
    +-- Hard Deadlock Escape
    |
    +-- Selective Predictive Recovery
    |
    v
Safe Robot Action

## Installation

```bash
pip install -r requirements.txt
```

## Verify Architecture

```bash
python verify_architecture_fixes.py
```

## Training

```bash
python train.py --timesteps 90000 --learning-starts 5000 --batch-size 256 --buffer-size 1000000 --eval-freq 10000 --eval-episodes 30 --warm-start-checkpoint PATH_TO_CHECKPOINT --critic-warmup-steps 0 --target-entropy -1.0 --curriculum-start-level 0.15 --curriculum-steps 48000 --run-name architecture_fixed_final
```

## Evaluation

```bash
python run_eval.py PATH_TO_MODEL output_folder
```

## Aegis-RL vs A* + DWA comparison

The comparison runner uses the same warehouse environment for both controllers:

- **Aegis-RL:** deterministic SAC policy + the existing CBF-QP safety stack.
- **Classical baseline:** A* global planning + Dynamic Window Approach local velocity control with the environment CBF disabled.

Both controllers start from an exact copied environment snapshot (robot pose, goal, dynamic obstacles, actuator state, observation history, and RNG state). The script benchmarks a seed range, saves aggregate results, ranks reproducible scenarios where Aegis has a measured advantage, then renders only the selected representative scenarios.

```bash
python compare_controllers.py PATH_TO_MODEL.pt \
  --seed-start 42 \
  --num-seeds 200 \
  --output-dir comparison_results \
  --render-top-k 3
```

Outputs:

- `episode_results.csv` — every controller/seed outcome and metric.
- `aggregate_results.json` — success, collision, timeout, time, path, clearance, and smoothness aggregates.
- `representative_aegis_wins.csv` — selected seeds and the exact reason each was selected.
- `selected_XX_seed_YYY.mp4` — real side-by-side simulation clips.

Selected videos are explicitly labelled **"Selected representative benchmark scenario"**. They are examples from the completed benchmark, not substitutes for the aggregate results.

Run the focused comparison tests with:

```bash
python -m pytest -q tests/test_classical_baseline.py tests/test_compare_controllers.py
```

## Next Stage

High-fidelity validation using NVIDIA Isaac Sim and ROS 2.

## Research Goal

The project investigates both safety and liveness in reinforcement-learning-based AMR navigation, particularly CBF-induced deadlocks and infeasible safety-filter states.
