#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARTIFACT_ID="10613786048"
REPO="NikhileshRSahu/safe-rl-amr-cbf"
CHECKPOINT="results/run11/shared_sac_cbf.pt"
TUNE_DIR="results/peak_orca_tuning"
FINAL_DIR="results/peak_orca_final"

printf '\n=== Peak ORCA-DD vs SAC+CBF local benchmark ===\n'
printf 'Repository: %s\n' "$ROOT"
printf 'Python: '
python3 --version

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required." >&2
  exit 2
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "ERROR: unzip is required. In WSL Ubuntu run: sudo apt update && sudo apt install -y unzip" >&2
  exit 2
fi

python3 -m venv .venv-benchmark
# shellcheck disable=SC1091
source .venv-benchmark/bin/activate
python -m pip install --upgrade pip
pip install "numpy<2.0" scipy cvxpy osqp pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu

printf '\n=== 1/5 Verify implementation ===\n'
pytest -q \
  tests/test_orca_geometry.py \
  tests/test_dd_motion.py \
  tests/test_beast_classical.py \
  tests/test_beast_benchmark.py \
  tests/test_beast_scenarios.py

printf '\n=== 2/5 Verify untouched final seed ranges ===\n'
python - <<'PY'
from benchmark.beast_config import TEST_SEEDS_BY_N
expected = {
    2: tuple(range(6200, 6230)),
    4: tuple(range(6400, 6430)),
    6: tuple(range(6600, 6630)),
}
assert TEST_SEEDS_BY_N == expected, (TEST_SEEDS_BY_N, expected)
print(TEST_SEEDS_BY_N)
PY

printf '\n=== 3/5 Tune Peak ORCA-DD on validation seeds only ===\n'
rm -rf "$TUNE_DIR"
python benchmark/tune_beast_classical.py --out "$TUNE_DIR"
test -s "$TUNE_DIR/best_config.json"

printf '\n=== 4/5 Resolve exact Run-11 SAC+CBF checkpoint ===\n'
if [[ -s "$CHECKPOINT" ]]; then
  echo "Using existing $CHECKPOINT"
else
  mkdir -p "$(dirname "$CHECKPOINT")"
  TMP_ZIP="/tmp/run11-${ARTIFACT_ID}.zip"
  rm -f "$TMP_ZIP"

  if command -v gh >/dev/null 2>&1; then
    if ! gh auth status >/dev/null 2>&1; then
      echo "ERROR: GitHub CLI is installed but not authenticated." >&2
      echo "Run once: gh auth login" >&2
      exit 3
    fi
    gh api \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "/repos/${REPO}/actions/artifacts/${ARTIFACT_ID}/zip" > "$TMP_ZIP"
  elif [[ -n "${GH_TOKEN:-}" ]]; then
    curl -fL \
      -H "Authorization: Bearer ${GH_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${REPO}/actions/artifacts/${ARTIFACT_ID}/zip" \
      -o "$TMP_ZIP"
  else
    cat >&2 <<'EOF'
ERROR: The exact Run-11 checkpoint is not present locally.
Best option in WSL:
  1) Install GitHub CLI: sudo apt update && sudo apt install -y gh
  2) Authenticate once: gh auth login
  3) Re-run this script.

Alternative: export a GitHub token as GH_TOKEN before running this script.
EOF
    exit 3
  fi

  unzip -oq "$TMP_ZIP" -d results/run11
  test -s "$CHECKPOINT"
fi

python - <<'PY'
import torch
p='results/run11/shared_sac_cbf.pt'
ck=torch.load(p,map_location='cpu',weights_only=False)
print('checkpoint:', p)
print('obs_dim:', ck.get('obs_dim'))
print('observation_version:', ck.get('observation_version'))
print('seed:', ck.get('seed'))
PY

printf '\n=== 5/5 Run fresh untouched final comparison ===\n'
rm -rf "$FINAL_DIR"
python benchmark/evaluate_beast_controllers.py \
  --checkpoint "$CHECKPOINT" \
  --config "$TUNE_DIR/best_config.json" \
  --out "$FINAL_DIR"

printf '\n=== FINAL TABLE ===\n'
python - <<'PY'
import json
from pathlib import Path
p=Path('results/peak_orca_final/beast_eval_summary.json')
s=json.loads(p.read_text())
print('| Fleet | Controller | Success | Collision | Timeout | Fleet success | Path m* | Time s* | Throughput/min | Compute ms/agent-step |')
print('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for n in ('2','4','6'):
    for key,label in [('sac_cbf','SAC + CBF'),('astar_orca_dd','Peak A* + ORCA-DD')]:
        r=s[n][key]
        def pct(x): return f'{100*x:.1f}%'
        def num(x): return '—' if x is None else f'{x:.2f}'
        print(f"| {n} | {label} | {pct(r['success_rate'])} | {pct(r['collision_rate'])} | {pct(r['timeout_rate'])} | {pct(r['fleet_success_rate'])} | {num(r['path_length_success_mean'])} | {num(r['traversal_time_success_mean'])} | {r['mean_throughput_per_min']:.2f} | {r['mean_controller_ms_per_agent_step']:.3f} |")
print('\n*Path/time are means over successful agents.')
print('\nProvenance:')
print(json.dumps(s['_provenance'], indent=2))
PY

printf '\nSaved results:\n  %s/beast_eval_summary.json\n  %s/beast_eval_rows.json\n  %s/beast_eval_config.json\n' "$FINAL_DIR" "$FINAL_DIR" "$FINAL_DIR"
