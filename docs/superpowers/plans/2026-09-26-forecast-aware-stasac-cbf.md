# Forecast-Aware ST-SAC + CBF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a causal short-horizon human-motion forecaster and forecast-risk features to the existing ST-SAC+CBF warehouse navigator, then evaluate the resulting forecast-aware policy fairly against non-forecast ST-SAC+CBF and AP-ORCA+CBF.

**Architecture:** Keep the current A* route, recurrent ST-SAC actor, and shared CBF safety layer. Add a causal per-track history buffer, a compact GRU forecaster, bounded forecast-risk features, and fuse those features into the existing variable-entity representation. Train the forecaster independently first, freeze it for the first RL integration, and preserve a switchable non-forecast ablation path.

**Tech Stack:** Python 3.11, NumPy, PyTorch, existing warehouse simulator, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-26-forecast-aware-stasac-cbf-design.md`

## Global Constraints

- RL remains the runtime local navigator and directly outputs normalized `(v, omega)`.
- The forecaster may consume only observations at or before the current timestamp.
- AP-ORCA and RL retain the same raw sensing/range/LOS budget and the same primary CBF safety layer.
- Initial forecast horizon is 1.5-2.0 s with 5-10 deterministic forecast points.
- The first RL integration uses a frozen forecaster checkpoint; joint fine-tuning is out of scope until the frozen version is evaluated.
- Gradient-training seeds remain `10100-10115`; competence-gate seeds `10116-10119`; development screen `10300-10319`; validation `11100-11119`; final holdout `13100-13129`.
- Validation and holdout future trajectories must never be used to train the forecaster.
- Easy AP-ORCA-near-perfect scenarios remain sanity checks, not headline superiority evidence.

## Review Focus

- Missing/occluded humans must not create fabricated fresh observations or future leakage.
- Track order permutations must not alter scene-level policy behavior.
- Zero-human scenes and large variable crowds must preserve valid tensor shapes and finite outputs.
- Forecast uncertainty must remain bounded/finite even for stale or short histories.
- Disabling forecast features must reproduce the existing non-forecast ST-SAC interface and checkpoint path.

---

### Task 1: Causal Human Track History

**Files:**
- Create: `benchmark/human_history.py`
- Modify: `benchmark/stasac_warehouse.py`
- Test: `tests/test_human_history.py`

**Interfaces:**
- Consumes: visible human observations already produced by `WarehouseObservationBuilder._entity_observations(...)`.
- Produces: `HumanTrackHistory.update(entity_id, timestamp, position, velocity, visible=True)` and `HumanTrackHistory.sequence(entity_id, now) -> HistorySequence`.

- [ ] **Step 1: Write failing causal-history tests**

Cover monotonic timestamps, duplicate-timestamp suppression, stale-age reporting, short-history padding/masking, zero-human behavior, and explicit rejection of future timestamps.

- [ ] **Step 2: Run the new tests and confirm RED**

Run: `pytest tests/test_human_history.py -q`

Expected: failures because `benchmark.human_history` does not exist.

- [ ] **Step 3: Implement `HumanTrackHistory` and `HistorySequence`**

Use bounded deques per track. Store only observed timestamp, position, velocity, and visibility metadata. `sequence(...)` returns fixed-length numeric arrays plus a boolean validity mask and observation age.

- [ ] **Step 4: Integrate history updates into `WarehouseObservationBuilder` without changing current entity features yet**

History updates occur only when an entity is actually visible under the shared shelf/range perception model.

- [ ] **Step 5: Verify GREEN**

Run: `pytest tests/test_human_history.py tests/test_stasac_training_runner.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

`git commit -am "feat: add causal human track history"`

---

### Task 2: Analytical Forecast Baseline and Dataset Contract

**Files:**
- Create: `benchmark/human_forecast_dataset.py`
- Create: `benchmark/constant_velocity_forecaster.py`
- Test: `tests/test_human_forecast_dataset.py`
- Test: `tests/test_constant_velocity_forecaster.py`

**Interfaces:**
- Consumes: `HistorySequence` from Task 1 and simulator trajectories from development seeds only.
- Produces: `ForecastSample(history, history_mask, future_xy, future_mask, scenario, seed)` and `ConstantVelocityForecaster.predict(...) -> ForecastOutput`.

- [ ] **Step 1: Write failing tests for sample construction**

Assert every input timestamp is `<= t0`, every target timestamp is `> t0`, training sample seeds are development-training only, and validation/holdout seeds are rejected by the training-data builder.

- [ ] **Step 2: Write failing constant-velocity tests**

Use synthetic straight-line motion and require near-zero ADE/FDE; include masked short histories and stationary targets.

- [ ] **Step 3: Run RED**

Run: `pytest tests/test_human_forecast_dataset.py tests/test_constant_velocity_forecaster.py -q`

- [ ] **Step 4: Implement dataset builder and `ForecastOutput` contract**

`ForecastOutput` contains `mean_xy: [N,H,2]`, `sigma_xy: [N,H,2]`, `mask: [N,H]`.

- [ ] **Step 5: Implement constant-velocity baseline**

Estimate velocity only from valid past/current samples; never inspect future targets.

- [ ] **Step 6: Verify GREEN**

Run the two new test files plus `tests/test_best_vs_best_protocol.py`.

- [ ] **Step 7: Commit**

`git commit -am "feat: add causal forecast dataset and CV baseline"`

---

### Task 3: Compact GRU Human Forecaster

**Files:**
- Create: `benchmark/human_forecaster.py`
- Test: `tests/test_human_forecaster.py`

**Interfaces:**
- Consumes: fixed-length history tensors and masks from Task 2.
- Produces: `GRUHumanForecaster.forward(history, history_mask) -> ForecastOutput` compatible with the analytical baseline.

- [ ] **Step 1: Write failing model-contract tests**

Assert deterministic shapes, finite outputs, positive bounded sigma, batch size 1 and many tracks, permutation equivariance across independent tracks, and backward-pass finite gradients.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_human_forecaster.py -q`

- [ ] **Step 3: Implement the minimal GRU forecaster**

Use shared weights across tracks. Encode history with a GRU and decode 5-10 future points for a default 2.0 s horizon. Parameterize uncertainty with a positive transform and clamp to a documented finite range.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_human_forecaster.py tests/test_constant_velocity_forecaster.py -q`

- [ ] **Step 5: Commit**

`git commit -am "feat: add GRU human motion forecaster"`

---

### Task 4: Forecaster Training and Independent Evaluation

**Files:**
- Create: `benchmark/train_human_forecaster.py`
- Create: `benchmark/evaluate_human_forecaster.py`
- Test: `tests/test_train_human_forecaster.py`
- Add workflow: `.github/workflows/human-forecaster-smoke.yml`

**Interfaces:**
- Consumes: dataset and model from Tasks 2-3.
- Produces: checkpoint containing architecture/version, horizon, history length, seed split metadata, state dict, ADE/FDE summary, and calibration statistics.

- [ ] **Step 1: Write failing checkpoint/evaluation tests**

Require finite loss, checkpoint round-trip, development-only training metadata, ADE/FDE computation, and no validation/holdout seed usage.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_train_human_forecaster.py -q`

- [ ] **Step 3: Implement training loop**

Use supervised future-position loss plus uncertainty-aware negative-log-likelihood or equivalent bounded probabilistic loss. Keep optimizer/model small enough for CPU CI smoke tests.

- [ ] **Step 4: Implement evaluator**

Report overall and per-scenario ADE/FDE and uncertainty coverage. Always evaluate the constant-velocity baseline on the same samples.

- [ ] **Step 5: Add smoke workflow**

Train a tiny checkpoint on development-training data, verify finite parameters and seed metadata, and upload checkpoint + JSON report.

- [ ] **Step 6: Verify GREEN locally/CI**

Run new tests and workflow contract checks.

- [ ] **Step 7: Commit**

`git commit -am "feat: train and evaluate human forecaster"`

---

### Task 5: Forecast Risk Features

**Files:**
- Create: `benchmark/forecast_risk_features.py`
- Test: `tests/test_forecast_risk_features.py`

**Interfaces:**
- Consumes: robot pose/velocity, current entity state, `ForecastOutput`, forecast dt, route/waypoint direction.
- Produces: `ForecastRiskBatch(features: [N,F], mask: [N])` with bounded normalized features.

- [ ] **Step 1: Write failing risk-feature tests**

Cover approaching vs diverging trajectories, higher uncertainty increasing risk, crossing corridor detection, stationary human, empty batch, finite bounds, and permutation equivariance.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_forecast_risk_features.py -q`

- [ ] **Step 3: Implement risk features**

Include minimum predicted separation, time to minimum separation, discounted risk score, uncertainty summary, corridor-entry score, and observation age. Do not use ground-truth future robot or human state.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_forecast_risk_features.py -q`

- [ ] **Step 5: Commit**

`git commit -am "feat: derive bounded forecast risk features"`

---

### Task 6: Forecast-Aware Warehouse Observation Builder

**Files:**
- Modify: `benchmark/stasac_warehouse.py`
- Modify: `benchmark/warehouse_interaction_features.py`
- Test: `tests/test_forecast_warehouse_observation.py`

**Interfaces:**
- Consumes: frozen forecaster + Task 1 history + Task 5 risk features.
- Produces: entity feature batches with optional forecast extension; `use_forecast=False` retains the original entity contract.

- [ ] **Step 1: Write failing observation tests**

Assert zero-human validity, occluded humans are not refreshed, stale history remains marked stale, forecast-enabled features are finite/bounded, track permutation does not change pooled scene representation, and `use_forecast=False` matches the existing baseline feature dimension and values.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_forecast_warehouse_observation.py -q`

- [ ] **Step 3: Integrate forecast context into entity observations**

Prefer appending forecast summaries per entity rather than changing ego features. Add explicit constants/version names for base and forecast entity dimensions.

- [ ] **Step 4: Verify GREEN plus regression suite**

Run: `pytest tests/test_forecast_warehouse_observation.py tests/test_warehouse_interaction_features.py tests/test_stasac_warehouse.py -q`

- [ ] **Step 5: Commit**

`git commit -am "feat: add forecast-aware warehouse observations"`

---

### Task 7: Forecast-Aware ST-SAC Policy Compatibility

**Files:**
- Modify: `benchmark/spatiotemporal_policy.py`
- Modify: `benchmark/train_stasac_cbf.py`
- Test: `tests/test_forecast_stasac_policy.py`

**Interfaces:**
- Consumes: base or forecast-extended entity batches from Task 6.
- Produces: actor/Q networks parameterized by entity feature dimension and checkpoint metadata that records observation version + forecaster checkpoint hash/version.

- [ ] **Step 1: Write failing policy tests**

Assert bounded deterministic actions, 0-24 entity support, finite gradients, recurrent reset, forecast-enabled dimension support, and unchanged behavior contract for the base non-forecast path.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_forecast_stasac_policy.py -q`

- [ ] **Step 3: Generalize policy constructors to explicit entity feature dimension**

Do not duplicate the actor architecture. Preserve permutation-invariant attention and GRU memory.

- [ ] **Step 4: Extend checkpoint schema**

Store `observation_version`, `entity_dim`, forecast-enabled flag, forecaster metadata/hash, and all existing actor/Q state.

- [ ] **Step 5: Verify GREEN and checkpoint round-trip**

Run the new test plus existing `test_spatiotemporal_policy.py` and `test_train_stasac_cbf.py`.

- [ ] **Step 6: Commit**

`git commit -am "feat: support forecast-aware STASAC inputs"`

---

### Task 8: Forecast-Aware RL Training Runner

**Files:**
- Modify: `benchmark/stasac_training_runner.py`
- Test: `tests/test_forecast_stasac_training_runner.py`
- Add workflow: `.github/workflows/forecast-stasac-smoke.yml`

**Interfaces:**
- Consumes: frozen forecaster checkpoint, forecast-aware observation builder, existing performance-gated AP-ORCA teacher, staged curriculum.
- Produces: Forecast-ST-SAC+CBF checkpoint and training summary with forecaster provenance and unchanged seed-partition guarantees.

- [ ] **Step 1: Write failing runner tests**

Assert forecaster is frozen (`requires_grad=False` and unchanged weights), training seeds never leak, competence probes remain policy-only, AP-ORCA is training-only teacher, and deployment actor inference never calls ORCA.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_forecast_stasac_training_runner.py -q`

- [ ] **Step 3: Add `--forecaster-checkpoint` and forecast-enabled training path**

Reuse existing performance-gated BC and stage promotion. Keep `use_forecast=False` as an ablation path.

- [ ] **Step 4: Add smoke workflow**

Use a small frozen forecaster and short RL run, but retain the existing independent realistic competence-gate horizon contract.

- [ ] **Step 5: Verify GREEN**

Run forecast runner tests plus existing STASAC runner tests.

- [ ] **Step 6: Commit**

`git commit -am "feat: train forecast-aware STASAC with frozen forecaster"`

---

### Task 9: Fair Three-Way Evaluation

**Files:**
- Modify: `benchmark/best_vs_best_runner.py`
- Modify: `benchmark/best_vs_best_evaluation.py`
- Test: `tests/test_forecast_best_vs_best.py`
- Add workflow: `.github/workflows/forecast-best-vs-best-dev-screen.yml`

**Interfaces:**
- Consumes: frozen AP-ORCA config, base ST-SAC checkpoint, forecast ST-SAC checkpoint, frozen forecaster, same scenario/seed catalog.
- Produces: paired episode rows and aggregate JSON for AP-ORCA+CBF, ST-SAC+CBF, and Forecast-ST-SAC+CBF.

- [ ] **Step 1: Write failing fairness tests**

Assert same seeds/start/goal/scenario, same max steps, same collision definition, same CBF use, same raw perception range/LOS, and no controller receives simulator future state.

- [ ] **Step 2: Add interaction-efficiency metrics**

Track traversal time, path length, CBF intervention rate, stop/yield duration, angular/action oscillation, and commitment reversals in addition to existing success/collision/timeout/throughput.

- [ ] **Step 3: Run RED**

Run: `pytest tests/test_forecast_best_vs_best.py -q`

- [ ] **Step 4: Implement three-way paired evaluator**

Keep development screen restricted to `10300-10319`. Do not touch formal validation or holdout.

- [ ] **Step 5: Verify GREEN**

Run evaluator tests and existing protocol/evaluation tests.

- [ ] **Step 6: Commit**

`git commit -am "feat: compare AP-ORCA base and forecast STASAC fairly"`

---

### Task 10: Development Screening and Freeze Decision

**Files:**
- Add: `docs/superpowers/reports/2026-09-26-forecast-aware-dev-screen.md`
- No production code unless a failing test reveals a defect.

**Interfaces:**
- Consumes: development-screen artifacts from Task 9.
- Produces: explicit go/no-go decision for formal validation.

- [ ] **Step 1: Run the full development screen on `10300-10319`**

Evaluate all three controllers on the same discriminative scenarios and sanity-check scenarios.

- [ ] **Step 2: Check safety-first win criterion**

Forecast-aware RL cannot be declared better if collision rate is materially worse than AP-ORCA.

- [ ] **Step 3: Compare forecast usefulness**

Require that Forecast-ST-SAC improves at least one meaningful navigation metric over non-forecast ST-SAC on the hard scenarios without a meaningful safety regression.

- [ ] **Step 4: Write the development report**

Include forecaster ADE/FDE vs constant-velocity baseline, controller aggregates, scenario-level breakdowns, confidence intervals where available, and failure analysis.

- [ ] **Step 5: Freeze only if justified**

If criteria are met, freeze architecture/checkpoints/config/metric definitions before using `11100-11119`. Otherwise remain in development and do not consume validation/holdout.

- [ ] **Step 6: Commit**

`git commit -am "docs: report forecast-aware development screen"`

---

## Final Verification Before Any Superiority Claim

Run the complete development test suite, forecaster smoke workflow, forecast-STASAC smoke workflow, and three-way development screen. A superiority statement is prohibited until the paired frozen evaluation supports it. Formal validation seeds `11100-11119` and final holdout `13100-13129` remain untouched until the development freeze decision is documented.
