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

`ash
pip install -r requirements.txt
`

## Verify Architecture

`ash
python verify_architecture_fixes.py
`

## Training

`ash
python train.py --timesteps 90000 --learning-starts 5000 --batch-size 256 --buffer-size 1000000 --eval-freq 10000 --eval-episodes 30 --warm-start-checkpoint PATH_TO_CHECKPOINT --critic-warmup-steps 0 --target-entropy -1.0 --curriculum-start-level 0.15 --curriculum-steps 48000 --run-name architecture_fixed_final
`

## Evaluation

`ash
python run_eval.py PATH_TO_MODEL output_folder
`

## Next Stage

High-fidelity validation using NVIDIA Isaac Sim and ROS 2.

## Research Goal

The project investigates both safety and liveness in reinforcement-learning-based AMR navigation, particularly CBF-induced deadlocks and infeasible safety-filter states.
