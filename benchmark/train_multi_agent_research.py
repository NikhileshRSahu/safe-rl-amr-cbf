import argparse, json, math, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from cbf import CBFFilterConfig, CBFSafetyFilter

DT=0.1; WORLD=10.0; ROBOT_R=0.30; VMAX=1.0; WMAX=1.5
SAFE_MARGIN=0.35; N_NEAR=8; GOAL_TOL=0.45
OBS_DIM=5 + N_NEAR*6 + 8
ACT_DIM=2
SHELVES=[
(-8,-8,-6,-2),(-8,2,-6,8),(-3,-8,-1,-2),(-3,2,-1,8),
(1,-8,3,-2),(1,2,3,8),(6,-8,8,-2),(6,2,8,8),
(-0.8,-0.8,0.8,0.8)
]
STARTS=np.array([[-9,-9],[-9,0],[-9,9],[9,-9],[9,0],[9,9]],np.float32)
GOALS =np.array([[9,9],[9,0],[9,-9],[-9,9],[-9,0],[-9,-9]],np.float32)

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
def clip(x,a,b): return max(a,min(b,x))

class World:
    def __init__(self,n_agents=1,n_people=3,seed=0):
        self.n=n_agents; self.nppl=n_people; self.rng=np.random.default_rng(seed)
        cfg=CBFFilterConfig(robot_radius=ROBOT_R,dynamic_obs_radius=ROBOT_R,safety_margin=SAFE_MARGIN,
            gamma=2.0,pred_horizon=0.5,tracking_uncertainty=0.0,h_activation_threshold=2.0,
            v_bounds=(0.0,VMAX),omega_bounds=(-WMAX,WMAX),backend="scipy",max_iter=40,
            enable_slack=False,lookahead_distance=0.25,enable_hard_escape=True)
        self.cbf=CBFSafetyFilter(cfg,(-WORLD,-WORLD,WORLD,WORLD))
        self.reset()
    def reset(self):
        ids=np.arange(self.n)
        self.p=STARTS[ids].copy()+self.rng.normal(0,.10,(self.n,2))
        self.g=GOALS[ids].copy()+self.rng.normal(0,.10,(self.n,2))
        self.th=np.array([math.atan2(self.g[i,1]-self.p[i,1],self.g[i,0]-self.p[i,0]) for i in range(self.n)],np.float32)
        self.v=np.zeros(self.n,np.float32); self.w=np.zeros(self.n,np.float32)
        self.done=np.zeros(self.n,bool); self.hit=np.zeros(self.n,bool)
        # Episode-specific right-of-way ordering breaks shared-policy symmetry
        # without giving any agent a permanently privileged identity.
        order=self.rng.permutation(self.n)
        self.priority=np.zeros(self.n,np.float32)
        for rank,idx in enumerate(order):
            self.priority[idx]=1.0 if self.n==1 else rank/(self.n-1)
        self.stall=np.zeros(self.n,np.int32)
        self.min_clearance=np.full(self.n,np.inf,np.float32)
        self.intervention_delta=np.zeros(self.n,np.float32)
        self.steps=0; self.interventions=np.zeros(self.n,np.int32); self.deadlock=np.zeros(self.n,np.int32)
        self.hp=self.rng.uniform(-7.5,7.5,(self.nppl,2)).astype(np.float32)
        ang=self.rng.uniform(-math.pi,math.pi,self.nppl); sp=self.rng.uniform(.25,.6,self.nppl)
        self.hv=np.c_[np.cos(ang)*sp,np.sin(ang)*sp].astype(np.float32)
        return [self.obs(i) for i in range(self.n)]
    def _rect_dist(self,p,r):
        cx=min(max(float(p[0]),r[0]),r[2]); cy=min(max(float(p[1]),r[1]),r[3])
        return math.hypot(float(p[0])-cx,float(p[1])-cy)
    def static_collision(self,p):
        if abs(float(p[0]))>WORLD-ROBOT_R or abs(float(p[1]))>WORLD-ROBOT_R:return True
        return any(self._rect_dist(p,r)<ROBOT_R for r in SHELVES)
    def ray(self,i,a,maxd=4.0):
        for d in np.linspace(.2,maxd,20):
            q=self.p[i]+np.array([math.cos(a),math.sin(a)],np.float32)*d
            if self.static_collision(q): return d/maxd
        return 1.0
    def obs(self,i):
        d=self.g[i]-self.p[i]; c=math.cos(-float(self.th[i])); s=math.sin(-float(self.th[i]))
        bx=c*d[0]-s*d[1]; by=s*d[0]+c*d[1]
        base=[clip(float(bx)/20,-1,1),clip(float(by)/20,-1,1),clip(float(np.linalg.norm(d))/20,0,1),
              wrap(math.atan2(float(d[1]),float(d[0]))-float(self.th[i]))/math.pi,
              float(self.priority[i])]
        ego_vel=np.array([math.cos(float(self.th[i]))*self.v[i],math.sin(float(self.th[i]))*self.v[i]],np.float32)

        # Reserve slots by entity type so pedestrians can never hide another
        # controlled AMR from the policy at high fleet density.
        amrs=[]
        for j in range(self.n):
            if j==i or self.done[j]: continue
            vel=np.array([math.cos(float(self.th[j]))*self.v[j],math.sin(float(self.th[j]))*self.v[j]],np.float32)
            amrs.append((self.p[j],vel,1.0))
        humans=[(self.hp[j],self.hv[j],0.0) for j in range(self.nppl)]
        amrs.sort(key=lambda z: float(np.linalg.norm(z[0]-self.p[i])))
        humans.sort(key=lambda z: float(np.linalg.norm(z[0]-self.p[i])))

        def encode_group(group,nslots):
            out=[]
            for pos,vel,typ in group[:nslots]:
                r=pos-self.p[i]; rx=c*r[0]-s*r[1]; ry=s*r[0]+c*r[1]
                relv=vel-ego_vel
                rvx=c*relv[0]-s*relv[1]; rvy=s*relv[0]+c*relv[1]
                out += [clip(float(rx)/6,-1,1),clip(float(ry)/6,-1,1),
                        clip(float(rvx)/2,-1,1),clip(float(rvy)/2,-1,1),
                        clip(float(np.linalg.norm(r))/6,0,1),typ]
            out += [0.0]*(nslots*6-len(out))
            return out

        ef=encode_group(amrs,N_AMR_SLOTS)+encode_group(humans,N_HUMAN_SLOTS)
        rays=[self.ray(i,float(self.th[i])+k*math.pi/4) for k in range(8)]
        return np.asarray(base+ef+rays,np.float32)
    def expert(self,i):
        d=self.g[i]-self.p[i]; desired=math.atan2(float(d[1]),float(d[0]))
        for j in range(self.n):
            if j==i or self.done[j]: continue
            q=self.p[i]-self.p[j]; dist=float(np.linalg.norm(q))
            if .01<dist<1.6:
                side=math.copysign(1.0, wrap(desired-math.atan2(float(q[1]),float(q[0]))))
                yield_gain=.55 + .55*(1.0-float(self.priority[i]))
                desired += yield_gain*side
        e=wrap(desired-float(self.th[i])); w=clip(2.2*e,-WMAX,WMAX); v=VMAX*max(.08,1-abs(e)/1.3)
        return np.array([2*v/VMAX-1,w/WMAX],np.float32)
    def dyn_array(self,i):
        rows=[]
        for j in range(self.n):
            if j==i or self.done[j]: continue
            rows.append([self.p[j,0],self.p[j,1],self.th[j],self.v[j],0,0])
        for j in range(self.nppl):
            sp=float(np.linalg.norm(self.hv[j])); th=math.atan2(float(self.hv[j,1]),float(self.hv[j,0]))
            rows.append([self.hp[j,0],self.hp[j,1],th,sp,0,0])
        return np.asarray(rows,np.float64) if rows else np.zeros((0,6),np.float64)
    def step(self,actions,use_cbf=True):
        prev=np.linalg.norm(self.g-self.p,axis=1)

        # Move pedestrians once per world tick.
        self.hp += self.hv*DT
        for j in range(self.nppl):
            for k in (0,1):
                if abs(float(self.hp[j,k]))>8.8:
                    self.hp[j,k]=np.sign(self.hp[j,k])*8.8
                    self.hv[j,k]*=-1

        # IMPORTANT: all AMRs see the same pre-step snapshot.  The old
        # implementation updated agent 0 before filtering agent 1, creating
        # order-dependent dynamics and an unfair source of multi-agent
        # collisions/deadlocks.
        p0=self.p.copy(); th0=self.th.copy(); v0=self.v.copy(); w0=self.w.copy()
        cand=p0.copy(); new_th=th0.copy(); new_v=v0.copy(); new_w=w0.copy()
        intervened=np.zeros(self.n,bool); deadlock_now=np.zeros(self.n,bool)
        intervention_delta_now=np.zeros(self.n,np.float32)

        def dyn_snapshot(i):
            rows=[]
            for j in range(self.n):
                if j==i or self.done[j]: continue
                rows.append([p0[j,0],p0[j,1],th0[j],v0[j],0,0])
            for j in range(self.nppl):
                sp=float(np.linalg.norm(self.hv[j]))
                oth=math.atan2(float(self.hv[j,1]),float(self.hv[j,0]))
                rows.append([self.hp[j,0],self.hp[j,1],oth,sp,0,0])
            return np.asarray(rows,np.float64) if rows else np.zeros((0,6),np.float64)

        for i in range(self.n):
            if self.done[i]: continue
            a=np.asarray(actions[i],np.float32)
            vnom=(float(a[0])+1)*.5*VMAX
            wnom=float(a[1])*WMAX
            vs,ws=vnom,wnom
            if use_cbf:
                rs=[float(p0[i,0]),float(p0[i,1]),float(th0[i]),float(v0[i]),float(w0[i])]
                vs,ws,diag=self.cbf.solve(vnom,wnom,rs,dyn_snapshot(i),SHELVES)
                intervened[i]=bool(diag.get("intervened"))
                deadlock_now[i]=bool(diag.get("deadlock_detected"))
                if intervened[i]: self.interventions[i]+=1
                if deadlock_now[i]: self.deadlock[i]+=1
                intervention_delta_now[i]=math.sqrt(((vs-vnom)/max(VMAX,1e-6))**2 + ((ws-wnom)/max(WMAX,1e-6))**2)
                self.intervention_delta[i]+=intervention_delta_now[i]
            new_th[i]=wrap(float(th0[i])+ws*DT)
            new_v[i]=vs; new_w[i]=ws
            cand[i]=p0[i]+np.array([math.cos(float(new_th[i])),math.sin(float(new_th[i]))],np.float32)*vs*DT

        # Determine collisions on the simultaneously proposed state.
        collision=np.zeros(self.n,bool)
        for i in range(self.n):
            if self.done[i]: continue
            if self.static_collision(cand[i]): collision[i]=True
            if not collision[i]:
                for q in self.hp:
                    if np.linalg.norm(cand[i]-q)<2*ROBOT_R:
                        collision[i]=True; break
        for i in range(self.n):
            if self.done[i]: continue
            for j in range(i+1,self.n):
                if self.done[j]: continue
                if np.linalg.norm(cand[i]-cand[j])<2*ROBOT_R:
                    collision[i]=True; collision[j]=True

        # Physical surface clearance (meters), not CBF h.  Positive means
        # separation between bodies; zero is contact.  Reward shaping below
        # only activates inside a short near-risk band so the agent is not
        # rewarded for simply staying far away or freezing.
        clearance=np.full(self.n,np.inf,np.float32)
        for i in range(self.n):
            if self.done[i]: continue
            ci=min(
                float(cand[i,0]-(-WORLD)-ROBOT_R),
                float(WORLD-cand[i,0]-ROBOT_R),
                float(cand[i,1]-(-WORLD)-ROBOT_R),
                float(WORLD-cand[i,1]-ROBOT_R),
            )
            for rect in SHELVES:
                ci=min(ci, self._rect_dist(cand[i],rect)-ROBOT_R)
            for q in self.hp:
                ci=min(ci, float(np.linalg.norm(cand[i]-q)-2*ROBOT_R))
            for j in range(self.n):
                if j==i or self.done[j]: continue
                ci=min(ci, float(np.linalg.norm(cand[i]-cand[j])-2*ROBOT_R))
            clearance[i]=ci
            self.min_clearance[i]=min(float(self.min_clearance[i]),ci)

        rewards=np.zeros(self.n,np.float32); done=np.zeros(self.n,bool)
        for i in range(self.n):
            if self.done[i]:
                done[i]=True; continue
            self.th[i]=new_th[i]; self.v[i]=new_v[i]; self.w[i]=new_w[i]
            if collision[i]:
                self.hit[i]=True; self.done[i]=True
            else:
                self.p[i]=cand[i]
                if np.linalg.norm(self.g[i]-self.p[i])<=GOAL_TOL:
                    self.done[i]=True

            now=float(np.linalg.norm(self.g[i]-self.p[i]))
            prog=float(prev[i]-now)
            if prog < 0.002 and now>GOAL_TOL:
                self.stall[i]+=1
            else:
                self.stall[i]=0

            rewards[i]=12*prog-.02-.002*abs(float(new_w[i]))
            if use_cbf and intervened[i]:
                rewards[i]-=.015
            if use_cbf and deadlock_now[i]:
                rewards[i]-=.10

            # Liveness shaping only: brief yielding is allowed, prolonged
            # no-progress is penalized.  We intentionally removed the
            # clearance and intervention-magnitude rewards because the
            # previous ablation increased 6-AMR collisions.
            if self.stall[i]>20:
                rewards[i]-=min(.25,.005*(self.stall[i]-20))

            if collision[i]:
                rewards[i]-=35
            elif self.done[i]:
                rewards[i]+=30
            done[i]=self.done[i]

        self.steps+=1
        if self.steps>=600: done[:]=True
        return [self.obs(i) for i in range(self.n)],rewards,done
    def stats(self): return int(np.sum(self.done & ~self.hit)),int(np.sum(self.hit))

class Replay:
    def __init__(self,cap=250000):
        self.cap=cap; self.n=0; self.ptr=0
        self.s=np.zeros((cap,OBS_DIM),np.float32); self.a=np.zeros((cap,2),np.float32)
        self.r=np.zeros((cap,1),np.float32); self.ns=np.zeros((cap,OBS_DIM),np.float32); self.d=np.zeros((cap,1),np.float32)
    def add(self,s,a,r,ns,d):
        j=self.ptr; self.s[j]=s;self.a[j]=a;self.r[j]=r;self.ns[j]=ns;self.d[j]=d
        self.ptr=(j+1)%self.cap;self.n=min(self.n+1,self.cap)
    def sample(self,b):
        ii=np.random.randint(0,self.n,b)
        return [torch.tensor(x[ii]) for x in (self.s,self.a,self.r,self.ns,self.d)]

class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.h=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.mu=nn.Linear(128,2); self.ls=nn.Linear(128,2)
    def sample(self,s,det=False):
        h=self.h(s); mu=self.mu(h); ls=torch.clamp(self.ls(h),-5,1); z=mu if det else mu+torch.exp(ls)*torch.randn_like(mu)
        a=torch.tanh(z)
        if det:return a,None
        lp=(-.5*((z-mu)/torch.exp(ls))**2-ls-.5*math.log(2*math.pi)).sum(-1,keepdim=True)-torch.log(1-a*a+1e-6).sum(-1,keepdim=True)
        return a,lp
class Q(nn.Module):
    def __init__(self):
        super().__init__(); self.n=nn.Sequential(nn.Linear(OBS_DIM+2,160),nn.ReLU(),nn.Linear(160,160),nn.ReLU(),nn.Linear(160,1))
    def forward(self,s,a):return self.n(torch.cat([s,a],-1))

def eval_actor(actor,n,seeds,use_cbf=True):
    rows=[]
    for sd in seeds:
        e=World(n,max(3,n+2),sd); s=e.reset()
        for _ in range(600):
            with torch.no_grad(): a,_=actor.sample(torch.tensor(np.asarray(s),dtype=torch.float32),True)
            s,_,d=e.step(a.numpy(),use_cbf)
            if np.all(d):break
        suc,col=e.stats(); rows.append(dict(seed=sd,success=suc,collision=col,steps=e.steps,
            interventions=int(e.interventions.sum()),deadlock=int(e.deadlock.sum()),
            min_clearance=float(np.min(e.min_clearance[np.isfinite(e.min_clearance)])) if np.any(np.isfinite(e.min_clearance)) else None,
            intervention_delta=float(e.intervention_delta.sum())))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--agent-steps",type=int,default=250000);ap.add_argument("--seed",type=int,default=52)
    ap.add_argument("--out",default="results/multi_agent_run");args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    actor=Actor();q1=Q();q2=Q();tq1=Q();tq2=Q();tq1.load_state_dict(q1.state_dict());tq2.load_state_dict(q2.state_dict())
    ao=Adam(actor.parameters(),2e-4);qo=Adam(list(q1.parameters())+list(q2.parameters()),3e-4);buf=Replay()
    gamma=.99;alpha=.08;tau=.01;global_steps=0
    stages=[1,2,4,6]
    per=max(1,args.agent_steps//sum(stages))
    logs=[]
    for n in stages:
        e=World(n,max(3,n+2),args.seed+100+n);s=e.reset();stage_agent_steps=0
        while stage_agent_steps<per*n:
            if buf.n<1000: acts=np.asarray([e.expert(i) if not e.done[i] else [-1,0] for i in range(n)],np.float32)
            else:
                with torch.no_grad(): acts=actor.sample(torch.tensor(np.asarray(s),dtype=torch.float32))[0].numpy()
            ns,r,d=e.step(acts,True)
            for i in range(n):buf.add(s[i],acts[i],r[i],ns[i],float(d[i]))
            s=ns;stage_agent_steps+=n;global_steps+=n
            if np.all(d):e=World(n,max(3,n+2),args.seed+1000+global_steps);s=e.reset()
            if buf.n>=1000 and global_steps%2==0:
                bs,ba,br,bns,bd=buf.sample(192)
                with torch.no_grad():
                    na,nlp=actor.sample(bns); y=br+gamma*(1-bd)*(torch.min(tq1(bns,na),tq2(bns,na))-alpha*nlp)
                ql=F.mse_loss(q1(bs,ba),y)+F.mse_loss(q2(bs,ba),y);qo.zero_grad();ql.backward();torch.nn.utils.clip_grad_norm_(list(q1.parameters())+list(q2.parameters()),5);qo.step()
                pa,lp=actor.sample(bs);al=(alpha*lp-torch.min(q1(bs,pa),q2(bs,pa))).mean();ao.zero_grad();al.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),5);ao.step()
                with torch.no_grad():
                    for p,tp in zip(q1.parameters(),tq1.parameters()):tp.mul_(1-tau).add_(tau*p)
                    for p,tp in zip(q2.parameters(),tq2.parameters()):tp.mul_(1-tau).add_(tau*p)
        ev=eval_actor(actor,n,range(900+n*10,905+n*10),True);logs.append(dict(stage=n,eval=ev))
        print("stage",n,ev,flush=True)
    torch.save({"actor":actor.state_dict(),"obs_dim":OBS_DIM,"observation_version":"typed_slots_relvel_v1","seed":args.seed},out/"shared_sac_cbf.pt")
    final_cbf={str(n):eval_actor(actor,n,range(1200+n*100,1210+n*100),True) for n in stages}
    final_sac={str(n):eval_actor(actor,n,range(1200+n*100,1210+n*100),False) for n in stages}
    agg={}
    for name,data in [("sac_cbf",final_cbf),("sac",final_sac)]:
        agg[name]={}
        for n,rows in data.items():
            N=int(n)*len(rows);agg[name][n]={
                "success_rate":sum(r["success"] for r in rows)/N,
                "collision_rate":sum(r["collision"] for r in rows)/N,
                "mean_interventions":float(np.mean([r["interventions"] for r in rows])),
                "mean_deadlock":float(np.mean([r["deadlock"] for r in rows])),
                "mean_intervention_delta":float(np.mean([r["intervention_delta"] for r in rows])),
                "min_clearance":float(np.min([r["min_clearance"] for r in rows if r["min_clearance"] is not None])) if any(r["min_clearance"] is not None for r in rows) else None}
    result={"agent_steps":global_steps,"reward_version":"liveness_v1","observation_version":"typed_slots_relvel_v1","stages":logs,"aggregate":agg}
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(agg,indent=2),flush=True)

if __name__=="__main__":main()
