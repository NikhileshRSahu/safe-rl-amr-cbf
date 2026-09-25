from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmark.beast_classical import AStarORCADD
from benchmark.beast_config import load_beast_config
from benchmark.progressive_velocity_baselines import AStarRVODD, AStarVODD
from benchmark.train_multi_agent_research import DT, SHELVES, WORLD, World

CONTROLLERS = {"vo": AStarVODD, "rvo": AStarRVODD, "orca": AStarORCADD}


def snapshot(w):
    return {
        "p": np.asarray(w.p, np.float32).copy(),
        "th": np.asarray(w.th, np.float32).copy(),
        "done": np.asarray(w.done, bool).copy(),
        "hit": np.asarray(w.hit, bool).copy(),
    }


def run(controller_name, config, seed):
    w = World(6, 0, seed)
    w.reset()
    cls = CONTROLLERS[controller_name]
    ctrls = [cls(w, i, config) for i in range(6)]
    frames = [snapshot(w)]
    for _ in range(600):
        actions = np.asarray([
            ctrls[i].action(w) if not w.done[i] else [-1.0, 0.0]
            for i in range(6)
        ], dtype=np.float32)
        _, _, done = w.step(actions, False)
        frames.append(snapshot(w))
        if np.all(done):
            break
    success = w.done & ~w.hit
    metrics = {
        "success": int(success.sum()),
        "collision": int(w.hit.sum()),
        "timeout": int((~w.done).sum()),
        "fleet_success": bool(success.all()),
        "steps": int(w.steps),
        "stop_yield_ticks": int(sum(c.diagnostics().get("stop_yield_ticks", 0) for c in ctrls)),
        "recovery_count": int(sum(c.diagnostics().get("recovery_count", 0) for c in ctrls)),
    }
    return w, frames, metrics


def render(frames, goals, title, outfile, stride=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.set_xlim(-WORLD, WORLD); ax.set_ylim(-WORLD, WORLD)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)
    for x0,y0,x1,y1 in SHELVES:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor="0.35",edgecolor="0.15",alpha=.85))
    for i,g in enumerate(goals):
        ax.scatter([g[0]],[g[1]],marker="*",s=110,label=f"Goal {i+1}")
    markers=["o","s","^","D","P","X"]
    pts=[ax.plot([],[],markers[i],markersize=9,label=f"AMR {i+1}")[0] for i in range(6)]
    heads=[ax.plot([],[],linewidth=1.8)[0] for _ in range(6)]
    trails=[ax.plot([],[],linewidth=1.1,alpha=.65)[0] for _ in range(6)]
    time_text=ax.text(.02,.98,"",transform=ax.transAxes,va="top")
    status=ax.text(.02,.94,"",transform=ax.transAxes,va="top")
    ax.legend(loc="lower right",fontsize=6,ncol=2)
    hist=[[] for _ in range(6)]
    chosen=list(range(0,len(frames),max(1,stride)))
    if chosen[-1] != len(frames)-1: chosen.append(len(frames)-1)
    writer=FFMpegWriter(fps=10,metadata={"title":title},bitrate=2600)
    outfile=Path(outfile); outfile.parent.mkdir(parents=True,exist_ok=True)
    with writer.saving(fig,str(outfile),dpi=120):
        for k in chosen:
            f=frames[k]
            for i in range(6):
                p=f["p"][i]; pts[i].set_data([p[0]],[p[1]])
                hx=p[0]+.5*math.cos(float(f["th"][i])); hy=p[1]+.5*math.sin(float(f["th"][i]))
                heads[i].set_data([p[0],hx],[p[1],hy])
                hist[i].append((float(p[0]),float(p[1])))
                a=np.asarray(hist[i]); trails[i].set_data(a[:,0],a[:,1])
            time_text.set_text(f"sim time: {k*DT:5.1f} s")
            status.set_text(f"successful AMRs: {int(np.sum(f['done'] & ~f['hit']))}/6")
            writer.grab_frame()
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=9103); ap.add_argument("--config",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    config=load_beast_config(args.config)
    meta={"proof_type":"same_seed_real_simulator_rollout","seed":args.seed,"n_amr":6,"humans":0,"controllers":{}}
    for name in ("vo","rvo","orca"):
        w,frames,m=run(name,config,args.seed); meta["controllers"][name]=m
        render(frames,w.g,f"PROOF: {name.upper()} | 6 AMRs | dense reciprocal | seed {args.seed}",out/f"{name}_proof.mp4")
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2),flush=True)

if __name__=="__main__": main()
