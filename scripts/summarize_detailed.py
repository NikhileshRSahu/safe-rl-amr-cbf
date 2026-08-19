import json
from pathlib import Path

p = Path('benchmark_results/detailed_cbf_only_seed999.json')
if not p.exists():
    print('File not found:', p)
    raise SystemExit(1)

data = json.loads(p.read_text())

for ep in data:
    idx = ep['episode_index']
    seed = ep['seed']
    coll = ep['collision']
    ret = ep['return']
    length = ep['length']
    print(f"Episode {idx} seed={seed} collision={coll} len={length} return={ret:.2f}")
    if coll:
        steps = ep['steps']
        # Find collision step: first step where info collision likely occurred at end of step; we don't have explicit per-step collision flag, so use episode length as indicator
        # Show last 8 recorded steps before episode end or collision
        last = steps[max(0, len(steps)-8):]
        print('  Last steps (step, solver_success, tier, slack, intervened, barrier):')
        for s in last:
            print(f"    {s['step']:3d}: solver_success={s['cbf_solver_success']} tier={s['tier']} slack={s['slack']} intervened={s['cbf_intervened']} barrier={s['barrier_value']}")
        print()
