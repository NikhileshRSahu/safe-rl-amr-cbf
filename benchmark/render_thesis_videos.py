import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches

from benchmark.train_multi_agent_research import Actor, World, SHELVES, WORLD, ROBOT_R, DT
from benchmark.evaluate_multi_agent_thesis import AStarVO

def snapshot(w):
    return {
        "p": w.p.copy(), "g": w.g.copy(), "hp": w.hp.copy(),
        "done": w.done.copy(), "hit": w.hit.copy(),
        "interventions": w.interventions.copy(),
        "collision_type": w.collision_type.copy(),
        "steps": int(w.steps),
    }

def rollout_sac_cbf(actor, n, seed):
    w = World(n, max(3, n+2), seed)
    obs = w.reset()
    frames = [snapshot(w)]
    for _ in range(600):
        with torch.no_grad():
            a, _ = actor.sample(torch.tensor(np.asarray(obs), dtype=torch.float32), True)
        obs, _, done = w.step(a.numpy(), True)
        frames.append(snapshot(w))
        if np.all(done):
            break
    return w, frames

def rollout_astar_vo(n, seed):
    w = World(n, max(3, n+2), seed)
    w.reset()
    ctrls = [AStarVO(w, i) for i in range(n)]
    frames = [snapshot(w)]
    for _ in range(600):
        acts = np.asarray([
            ctrls[i].action(w, i) if not w.done[i] else [-1.0, 0.0]
            for i in range(n)
        ], dtype=np.float32)
        _, _, done = w.step(acts, False)
        frames.append(snapshot(w))
        if np.all(done):
            break
    return w, frames

def result(w):
    success = int(np.sum(w.done & ~w.hit))
    collision = int(np.sum(w.hit))
    timeout = int(w.n - success - collision)
    return {
        "success": success, "collision": collision, "timeout": timeout,
        "steps": int(w.steps), "interventions": int(w.interventions.sum()),
        "collision_types": {
            "amr_amr": int(np.sum(w.collision_type == "amr_amr")),
            "human": int(np.sum(w.collision_type == "human")),
            "static": int(np.sum(w.collision_type == "static")),
        },
    }

def draw_panel(ax, frame, title, trails, show_interventions=False):
    ax.clear()
    ax.set_xlim(-WORLD, WORLD); ax.set_ylim(-WORLD, WORLD)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    for r in SHELVES:
        ax.add_patch(patches.Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1],
                                       facecolor="0.82", edgecolor="0.35"))
    ax.add_patch(patches.Rectangle((-WORLD,-WORLD),2*WORLD,2*WORLD,
                                   fill=False, edgecolor="black", lw=1.2))
    colors = plt.cm.tab10(np.linspace(0,1,max(6,len(frame["p"]))))
    for h in frame["hp"]:
        ax.add_patch(patches.Circle((h[0],h[1]), ROBOT_R,
                                    facecolor="lightcoral", edgecolor="maroon", alpha=0.8))
    for i,g in enumerate(frame["g"]):
        ax.plot(g[0], g[1], marker="*", ms=10, color=colors[i], mec="k", mew=.4)
    for i,p in enumerate(frame["p"]):
        if len(trails[i]) > 1:
            arr=np.asarray(trails[i])
            ax.plot(arr[:,0], arr[:,1], lw=1.1, color=colors[i], alpha=.55)
        face = "black" if frame["hit"][i] else colors[i]
        alpha = .32 if frame["done"][i] and not frame["hit"][i] else 1.0
        ax.add_patch(patches.Circle((p[0],p[1]), ROBOT_R,
                                    facecolor=face, edgecolor="k", alpha=alpha))
        ax.text(p[0],p[1],str(i+1),ha="center",va="center",fontsize=7,color="white",fontweight="bold")
    success=int(np.sum(frame["done"] & ~frame["hit"]))
    collision=int(np.sum(frame["hit"]))
    timeout=len(frame["done"])-success-collision
    status=f't={frame["steps"]*DT:4.1f}s  success={success}/{len(frame["done"])}  collision={collision}  active={timeout}'
    if show_interventions:
        status += f'  CBF interventions={int(frame["interventions"].sum())}'
    ax.text(.01,.99,status,transform=ax.transAxes,ha="left",va="top",fontsize=8,
            bbox=dict(boxstyle="round,pad=.25",facecolor="white",alpha=.85,edgecolor="0.8"))

def render_pair(actor, n, seed, outfile):
    wc, fc = rollout_sac_cbf(actor,n,seed)
    wa, fa = rollout_astar_vo(n,seed)
    m=max(len(fc),len(fa))
    fc += [fc[-1]]*(m-len(fc)); fa += [fa[-1]]*(m-len(fa))
    ids=list(range(0,m,3))
    fig,axs=plt.subplots(1,2,figsize=(13,6.4))
    fig.suptitle(f"Same scenario / same seed — {n} AMRs — seed {seed}",fontsize=14,fontweight="bold")
    tc=[[] for _ in range(n)]; ta=[[] for _ in range(n)]
    def upd(ii):
        k=ids[ii]
        for i in range(n):
            tc[i].append(fc[k]["p"][i].copy())
            ta[i].append(fa[k]["p"][i].copy())
        draw_panel(axs[0],fa[k],"A* + VO-style predictive avoidance",ta,False)
        draw_panel(axs[1],fc[k],"Trained SAC + CBF-QP",tc,True)
        return []
    ani=animation.FuncAnimation(fig,upd,frames=len(ids),interval=80,blit=False)
    ani.save(outfile,writer=animation.FFMpegWriter(fps=10,bitrate=1800),dpi=105)
    plt.close(fig)
    return {"n":n,"seed":seed,"astar_vo_style":result(wa),"sac_cbf":result(wc)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--out",default="results/thesis_videos")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    actor=Actor(); actor.load_state_dict(ck["actor"]); actor.eval()
    cases=[(2,5214),(4,5414),(6,5614)]
    summaries=[]
    for n,seed in cases:
        target=out/f"same_seed_{n}amr_seed{seed}.mp4"
        s=render_pair(actor,n,seed,str(target))
        summaries.append(s)
        print(json.dumps(s),flush=True)
    (out/"video_results.json").write_text(json.dumps(summaries,indent=2))

if __name__=="__main__":
    main()
