# Spatio-Temporal Risk-Attention SAC + CBF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a warehouse navigation policy in which RL remains the actual local navigator, directly outputs `(v, omega)`, and is upgraded with variable-neighbor interaction encoding, temporal memory, risk attention, and CBF safety so it can be fairly compared against the strongest ORCA baseline.

**Architecture:** Keep the existing warehouse dynamics, A* global route context, and CBF safety layer. Replace the current fixed-slot feed-forward SAC actor/critics with a spatio-temporal encoder: shared entity embedding -> risk-aware attention over visible humans/AMRs -> GRU temporal memory -> SAC actor/critics. Use ORCA only for imitation warm-start and as a baseline, never as runtime navigation for the RL arm.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy, cvxpy/OSQP, pytest, existing benchmark environment and GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-25-best-vs-best-orca-safe-rl-design.md`

## Global Constraints

- RL must remain the runtime local navigator and directly produce normalized `(v, omega)` actions.
- CBF remains the execution-level safety filter.
- ORCA may be used only as teacher/baseline, not as runtime controller in the RL arm.
- Same map, robot radius, speed/omega limits, collision definition, sensing budget, seeds, and episode horizon for both methods.
- No future human trajectories, hidden intent labels, simulator RNG state, or privileged future state may reach the deployed actor.
- Development and validation seeds may be used for architecture/tuning; final holdout is untouched until both systems are frozen.
- Do not reward arbitrary large clearance; optimize mission completion, liveness, smoothness, and safety.
- A claim that RL is better requires collision rate no worse than the classical baseline within the declared tolerance and better success/fleet success or timeout/throughput.

## Review Focus

1. Variable crowd size from 0 to 24 humans must not change tensor dimensions or crash attention masking.
2. Episodes with no visible humans or no visible peer AMRs must produce finite deterministic actions.
3. Occluded/stale detections must not leak future state; observation age/mask must be handled causally.
4. GRU hidden state must reset exactly at episode boundaries and per-agent termination.
5. CBF must receive only currently observable/estimated dynamic states in the new benchmark path.

---

### Task 1: Warehouse interaction observation and risk features

**Files:**
- Create: `benchmark/warehouse_interaction_features.py`
- Modify: `benchmark/human_sweep_experiment.py`
- Test: `tests/test_warehouse_interaction_features.py`

**Interfaces:**
- Consumes: current robot state, visible human/AMR positions and velocities, short observation history.
- Produces: `EntityBatch`, `compute_ttc_cpa(...)`, `compute_risk_features(...)`, and fixed-size ego/static context plus variable-length entity features and masks.

- [ ] **Step 1: Write failing tests** covering: stationary entities, closing trajectories, diverging trajectories, 0 humans, 24 humans, finite TTC/CPA values, deterministic ordering, and no future-state access.
- [ ] **Step 2: Run** `pytest -q tests/test_warehouse_interaction_features.py` and verify failure because the module does not exist.
- [ ] **Step 3: Implement** analytic features:
  - relative position `(dx, dy)`;
  - relative velocity `(dvx, dvy)`;
  - type bit;
  - distance;
  - `t_cpa = clip(-p·v/(||v||^2+eps), 0, T)`;
  - `d_cpa = ||p + v*t_cpa||`;
  - `ttc_risk = exp(-t_cpa/tau) * sigmoid((d_safe-d_cpa)/s)`;
  - observed acceleration and heading-rate from causal history;
  - observation age/visibility mask.
- [ ] **Step 4: Run** the feature tests and verify all pass.
- [ ] **Step 5: Commit** `feat: add warehouse interaction risk features`.

### Task 2: Spatio-temporal set encoder

**Files:**
- Create: `benchmark/spatiotemporal_policy.py`
- Test: `tests/test_spatiotemporal_policy.py`

**Interfaces:**
- Consumes: ego/static tensor `[B, E]`, entity tensor `[B, N, F]`, entity mask `[B, N]`, recurrent hidden state `[B, H]`.
- Produces: `scene_embedding`, next GRU state, SAC action distribution parameters, and Q inputs.

- [ ] **Step 1: Write failing tests** for permutation invariance, 0-entity masking, 24-entity batching, hidden-state reset, finite gradients, deterministic action in evaluation mode, and shape consistency.
- [ ] **Step 2: Run** `pytest -q tests/test_spatiotemporal_policy.py` and verify RED.
- [ ] **Step 3: Implement**:
  - shared entity MLP `F -> 64 -> 64`;
  - risk-aware attention score using embedded entity plus TTC/CPA risk channels;
  - masked softmax pooling;
  - ego/static encoder `E -> 96`;
  - fused scene vector -> GRU hidden size 128;
  - actor heads for `mu/log_std` of 2-D SAC action;
  - twin Q networks consuming recurrent scene embedding and action.
- [ ] **Step 4: Run** unit tests and verify GREEN.
- [ ] **Step 5: Commit** `feat: add spatiotemporal risk-attention SAC policy`.

### Task 3: Sequence replay and recurrent SAC training

**Files:**
- Create: `benchmark/train_stasac_cbf.py`
- Test: `tests/test_stasac_training.py`

**Interfaces:**
- Consumes: sequence observations from Task 1, recurrent policy from Task 2, current CBF filter and warehouse environment.
- Produces: trainable recurrent SAC loop, sequence replay samples, checkpoint `stasac_cbf.pt`, training summary JSON.

- [ ] **Step 1: Write failing tests** for sequence replay boundaries, hidden reset, burn-in handling, done masking, one optimizer update changing parameters, and checkpoint round-trip.
- [ ] **Step 2: Run** `pytest -q tests/test_stasac_training.py` and verify RED.
- [ ] **Step 3: Implement** sequence replay with `burn_in=8`, `train_len=16`, episode-boundary-safe sampling, recurrent SAC targets, gradient clipping, target-network Polyak updates, and current CBF execution.
- [ ] **Step 4: Add curriculum** over 1/2/4/6 AMRs, 3-24 humans, structured motion, stop/restart, bounded 60-120° turns, reversals, mixed yielding, and shelf-corner emergence. All motion must obey acceleration/turn-rate bounds and causal visibility.
- [ ] **Step 5: Run** smoke training for >=2,000 agent-steps and verify finite losses, nonzero replay, valid checkpoint, and no NaNs.
- [ ] **Step 6: Commit** `feat: train recurrent risk-attention SAC with CBF`.

### Task 4: ORCA imitation warm-start without runtime dependence

**Files:**
- Create: `benchmark/orca_teacher.py`
- Modify: `benchmark/train_stasac_cbf.py`
- Test: `tests/test_orca_teacher_warmstart.py`

**Interfaces:**
- Consumes: frozen Peak/AP-ORCA action for the same causal observation state.
- Produces: behavior-cloning batches and decaying BC coefficient used only in early training.

- [ ] **Step 1: Write failing tests** ensuring ORCA teacher actions never enter deployment inference, BC coefficient decays to zero, and teacher collection respects identical physical limits.
- [ ] **Step 2: Run** `pytest -q tests/test_orca_teacher_warmstart.py` and verify RED.
- [ ] **Step 3: Implement** `L_total = L_SAC + lambda_bc(t) * ||a_actor-a_orca||^2`, with `lambda_bc` linearly decaying from `1.0` to `0.0` over the first 15% of training agent-steps and remaining zero thereafter.
- [ ] **Step 4: Run** tests and a short warm-start smoke run proving `lambda_bc==0` after the scheduled point.
- [ ] **Step 5: Commit** `feat: add decaying ORCA teacher warm-start`.

### Task 5: Strong classical contender and identical benchmark protocol

**Files:**
- Create: `benchmark/adaptive_predictive_orca.py`
- Create: `benchmark/best_vs_best_protocol.py`
- Test: `tests/test_best_vs_best_protocol.py`

**Interfaces:**
- Consumes: same causal detection history as RL.
- Produces: AP-ORCA controller and paired-seed benchmark scenarios with identical sensing/dynamics.

- [ ] **Step 1: Write failing tests** for adaptive horizon bounds, uncertainty inflation monotonicity, no future-state use, non-reciprocal human responsibility, and identical seed/world initialization between AP-ORCA and RL.
- [ ] **Step 2: Run** `pytest -q tests/test_best_vs_best_protocol.py` and verify RED.
- [ ] **Step 3: Implement AP-ORCA** with filtered velocity/acceleration estimate, uncertainty radius `r_eff = r_phys + k_sigma*sigma_pred`, horizon clamped to `[1.0, 4.0] s`, human responsibility >50% when non-yielding evidence is detected, and hysteretic pass-side memory.
- [ ] **Step 4: Implement benchmark families**:
  - cross-intersection;
  - shelf-end occlusion;
  - stop/restart hesitation;
  - mixed yielding/non-yielding crowd;
  - density escalation 6/12/18/24 humans.
- [ ] **Step 5: Run** protocol tests and verify same seeds produce identical initial states and visibility for both controllers.
- [ ] **Step 6: Commit** `feat: add adaptive predictive ORCA and paired benchmark`.

### Task 6: Validation search, freeze, and fresh holdout

**Files:**
- Create: `benchmark/run_best_vs_best_validation.py`
- Create: `benchmark/run_best_vs_best_holdout.py`
- Create: `.github/workflows/best-vs-best-validation.yml`
- Create: `.github/workflows/best-vs-best-holdout.yml`
- Test: `tests/test_freeze_protocol.py`

**Interfaces:**
- Consumes: trained STASAC checkpoints and AP-ORCA configs.
- Produces: frozen winning RL checkpoint, frozen AP-ORCA config, validation report, untouched final holdout report.

- [ ] **Step 1: Write failing tests** proving holdout seed ranges are absent from training/validation code paths and frozen artifact hashes are checked before holdout execution.
- [ ] **Step 2: Run** `pytest -q tests/test_freeze_protocol.py` and verify RED.
- [ ] **Step 3: Implement validation** with equal candidate budget for both systems and select using lexicographic objective: minimize collision rate, maximize fleet success, minimize timeout, maximize throughput.
- [ ] **Step 4: Freeze** checkpoint/config plus SHA256 hashes and generate new holdout seeds only after freeze.
- [ ] **Step 5: Run** >=30 paired episodes per major scenario on holdout and compute Wilson 95% CI for binary rates plus paired bootstrap CI for throughput/traversal differences.
- [ ] **Step 6: Enforce claim rule**: RL can be labeled better only if collision rate is not worse within the predeclared tolerance and at least one primary liveness metric is significantly better without another primary metric significantly worse.
- [ ] **Step 7: Commit** `exp: add frozen best-vs-best holdout evaluation`.

### Task 7: Reproducible video proof and final report

**Files:**
- Create: `benchmark/render_best_vs_best_proof.py`
- Create: `.github/workflows/render-best-vs-best-proof.yml`
- Create: `docs/results/best-vs-best-summary.md`
- Test: `tests/test_best_vs_best_renderer.py`

**Interfaces:**
- Consumes: frozen holdout artifacts and selected representative seeds after aggregate results are finalized.
- Produces: side-by-side MP4, metadata JSON, aggregate table, and exact reproduction command.

- [ ] **Step 1: Write failing tests** ensuring renderer uses frozen hashes, same seed/world realization, and never selects a seed before aggregate evaluation finishes.
- [ ] **Step 2: Run** renderer tests and verify RED.
- [ ] **Step 3: Implement** real-simulator frame capture for RL and AP-ORCA with identical overlays: AMR success, collisions, timeout, sim time, human count, and scenario name.
- [ ] **Step 4: Run** `ffprobe` and metadata assertions on generated MP4s.
- [ ] **Step 5: Write** final results table with VO -> RVO -> ORCA progression plus AP-ORCA vs STASAC-CBF, including unfavorable results.
- [ ] **Step 6: Commit** `docs: add verified best-vs-best warehouse navigation proof`.

## Self-Review

- Spec coverage: variable-neighbor RL, temporal memory, risk awareness, CBF safety, ORCA strengthening, equal sensing, validation/freeze/holdout, confidence intervals, and video proof are each assigned to an implementation task.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: Tasks 1-3 use variable entity tensors and masks; Task 5 uses the same causal observation history; Tasks 6-7 consume frozen artifacts only.
- Review-focus coverage: variable crowd size -> Tasks 1/2; empty entity sets -> Task 2; occlusion causality -> Tasks 1/5; GRU reset -> Tasks 2/3; CBF causal dynamic state -> Tasks 3/5.
