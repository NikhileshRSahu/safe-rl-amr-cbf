# Safe RL Autonomous Mobile Robot (AMR) — Warehouse Navigation

A research codebase for **Safe Reinforcement Learning** on a differential-drive
Autonomous Mobile Robot (AMR) navigating a 20×20 m warehouse full of static
shelf racks and moving human/robot obstacles. The agent is trained with
**Soft Actor-Critic (SAC)**; every action the policy proposes is passed
through a **Control Barrier Function Quadratic Program (CBF-QP)** safety
filter *inside the environment step*, so the physical robot (in simulation)
can never execute an action that would violate a hand-verified safety
constraint, regardless of what the policy has learned so far.

```
                 ┌─────────────────────┐
   observation   │                     │   u_rl (unfiltered)
  ──────────────►│   SafeRLPolicy      ├───────────────┐
                 │  (actor + critic)   │                │
                 └─────────────────────┘                ▼
                                              ┌─────────────────────┐
                                              │   CBFSafetyFilter    │
                                              │  (QP: SLSQP / OSQP)  │
                                              └──────────┬───────────┘
                                                          │ u_safe
                                                          ▼
                                              ┌─────────────────────┐
                                              │   AMRWarehouseEnv    │
                                              │ (unicycle dynamics,  │
                                              │  LiDAR, social-force │
                                              │  dynamic obstacles)  │
                                              └──────────┬───────────┘
                                                          │ obs, reward, done, info
                                                          ▼
                                              ┌─────────────────────┐
                                              │  SafeReplayBuffer    │
                                              └─────────────────────┘
```

---

## Project layout

| File               | Purpose                                                                                   |
|--------------------|---------------------------------------------------------------------------------------------|
| `config.py`        | All physical, safety, reward, PPO/SAC, and visualization hyperparameters (frozen dataclasses). |
| `utils.py`         | Stateless geometry, kinematics, CBF, and sampling helper functions.                        |
| `cbf.py`           | Standalone `CBFSafetyFilter` (SLSQP or CVXPY/OSQP backend) used by both the env and `safe_sac.py`. |
| `environment.py`   | `AMRWarehouseEnv`: Gymnasium environment with native CBF-QP filtering, LiDAR, actuator noise, and social-force dynamic obstacles. |
| `policy.py`        | `SafeRLPolicy` and `PolicyConfig`: multi-modal actor-critic network (robot state / goal / LiDAR / obstacles / image encoders, squashed-Gaussian or Beta action heads, optional gSDE, recurrence, auxiliary heads). |
| `replay_buffer.py` | `SafeReplayBuffer`: pre-allocated ring-buffer replay memory, native `spaces.Dict` support, extra `cost`/`barrier_value` columns. |
| `safe_sac.py`      | `SafeSACAgent`: off-policy SAC training loop (twin-Q critic, clipped double-Q target, learned entropy temperature, Polyak target updates). |
| `train.py`         | Main training entry point: rollout collection, SAC updates, periodic deterministic evaluation, TensorBoard logging, best-model checkpointing. |
| `evaluate.py`      | Loads a saved checkpoint and runs it with live (or off-screen) rendering; prints per-episode and summary statistics. |
| `visualizer.py`    | Publication-quality trajectory plots and training-curve plots (Success Rate vs. CBF interventions). |
| `logger.py`        | Reusable console/file logger (`get_logger`) and a combined TensorBoard+CSV metric logger (`MetricLogger`). |

---

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU users:** install the CUDA build of PyTorch that matches your driver
> *before* running `pip install -r requirements.txt` (see
> https://pytorch.org/get-started/locally/); the pinned `torch` line above
> will otherwise pull the default CPU/CUDA wheel for your platform.

The CBF-QP filter defaults to the `scipy` (SLSQP) backend, which needs
nothing beyond `scipy`. The `cvxpy` + `osqp` backend (`CBFFilterConfig(backend="cvxpy")`)
matches `config.CBFConfig.QP_SOLVER = "OSQP"` and is optional — it's only
required if you explicitly construct a filter with `backend="cvxpy"`.

---

## Quickstart

### 1. Train an agent

```bash
python train.py \
    --timesteps 500000 \
    --batch-size 256 \
    --eval-freq 10000 \
    --eval-episodes 10 \
    --seed 42
```

This creates:
- `runs/safe_sac_<timestamp>/` — TensorBoard event files (`tensorboard --logdir runs`).
- `checkpoints/safe_sac_<timestamp>/best_model.pt` — best checkpoint by eval success rate.
- `checkpoints/safe_sac_<timestamp>/final_model.pt` — checkpoint at the end of training.

Run `python train.py --help` for the full list of overridable hyperparameters
(learning rates, `gamma`, `tau`, buffer size, `--no-cbf-filter` for an
unfiltered baseline comparison, etc.).

### 2. Watch a trained agent

```bash
python evaluate.py \
    --checkpoint checkpoints/safe_sac_<timestamp>/best_model.pt \
    --episodes 5 \
    --render-mode human
```

For headless / server use, render off-screen and optionally save GIFs per
episode:

```bash
python evaluate.py \
    --checkpoint checkpoints/safe_sac_<timestamp>/best_model.pt \
    --render-mode rgb_array \
    --save-video-dir eval_videos/
```

### 3. Plot results

```bash
# Success rate vs. CBF interventions over training, from the TensorBoard logs:
python visualizer.py \
    --log-dir runs/safe_sac_<timestamp> \
    --save-path figures/training_curves.png
```

`visualizer.plot_trajectory(...)` is a library function (not a CLI) intended
to be called from `evaluate.py` or a notebook with an episode's
`env.trajectory_history`, `env.dynamic_obstacles`, `config.SHELVES`, and
`env.goal_pos`:

```python
from visualizer import plot_trajectory
from config import SHELVES

plot_trajectory(
    trajectory_history=env.trajectory_history,
    obstacles=env.dynamic_obstacles,
    shelves=SHELVES,
    goal=tuple(env.goal_pos),
    save_path="figures/episode_042.pdf",
)
```

---

## Design notes

- **Safety is enforced in the environment, not the loss.** `AMRWarehouseEnv.step`
  calls `CBFSafetyFilter.solve(...)` on every step before applying the
  action to the unicycle dynamics. `SafeSACAgent` trains on the standard
  maximum-entropy SAC objective using the *resulting* (already-safe)
  transitions; it does not backpropagate through the QP. `cost` and
  `barrier_value` are stored in `SafeReplayBuffer` and logged during
  `SafeSACAgent.update` for diagnostics, ready to be wired into a
  Lagrangian safety-critic extension later.
- **Shared actor/critic trunk.** `SafeRLPolicy` shares one
  `feature_extractor` + `trunk` between the actor and the twin-Q critic.
  `SafeSACAgent`'s target network is therefore a full Polyak-averaged copy
  of the policy (`target_policy`), not just a target Q-head, so that the
  SAC Bellman target is computed with a consistent feature pipeline.
- **Action space convention.** `AMRWarehouseEnv.action_space` is a
  normalized `Box(-1, 1)` that the environment internally rescales to
  physical `[V_MIN, V_MAX]` / `[-OMEGA_MAX, OMEGA_MAX]`. `train.py`
  configures `PolicyConfig.action_bounds = ActionBounds(v_min=-1.0,
  v_max=1.0, omega_min=-1.0, omega_max=1.0)` so the actor's tanh-squashed
  output is directly compatible with the environment's expected input —
  keep this in mind if you construct a `PolicyConfig` by hand elsewhere.

---

## License

Add your license of choice here (e.g. MIT, Apache-2.0) before publishing or
submitting alongside a thesis/paper.