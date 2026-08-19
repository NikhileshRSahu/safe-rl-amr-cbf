import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
pre_path = repo_root / "benchmark_results" / "detailed_cbf_only_seed999.json"
post_path = repo_root / "benchmark_results" / "detailed_cbf_only_seed999_with_diags.json"
out_path = repo_root / "benchmark_results" / "episode_comparison_seed999.txt"

def outcome(ep):
    if ep.get("success"):
        return "success"
    if ep.get("collision"):
        return "collision"
    return "other"

def last_steps_info(ep, n=5):
    steps = ep.get("steps", [])
    last = steps[-n:]
    lines = []
    for s in last:
        lines.append(f" step={s.get('step')}, tier={s.get('tier')}, slack={s.get('slack')}")
    return "\n".join(lines)

def main():
    pre = json.loads(pre_path.read_text())
    post = json.loads(post_path.read_text())

    pre_by_seed = {e['seed']: e for e in pre}
    post_by_seed = {e['seed']: e for e in post}

    lines = []
    lines.append("episode_index,seed,pre_outcome,post_outcome,flipped")
    for ep in sorted(post, key=lambda e: e['episode_index']):
        seed = ep['seed']
        pre_ep = pre_by_seed.get(seed)
        pre_out = outcome(pre_ep) if pre_ep else 'missing'
        post_out = outcome(ep)
        flipped = pre_out != post_out
        lines.append(f"{ep['episode_index']},{seed},{pre_out},{post_out},{flipped}")
        if post_out == 'collision':
            lines.append("Final 5 steps (post) for collision episode:")
            lines.append(last_steps_info(ep, n=5))
            lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote episode comparison to: {out_path}")

if __name__ == '__main__':
    main()
