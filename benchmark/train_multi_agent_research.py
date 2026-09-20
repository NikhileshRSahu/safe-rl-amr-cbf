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
OBS_DIM=4 + N_NEAR*6 + 8
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
              wrap(math.atan2(float(d[1]),float(d[0]))-float(self.th[i]))/math.pi]
        ents=[]
        for j in range(self.n):
            if j==i or self.done[j]: continue
            vel=np.array([math.cos(float(self.th[j]))*self.v[j],math.sin(float(self.th[j]))*self.v[j]],np.float32)
            ents.append((self.p[j],vel,1.0))
        for j in range(self.nppl): ents.append((self.hp[j],self.hv[j],0.0))
        ents.sort(key=lambda z: float(np.linalg.norm(z[0]-self.p[i])))
        ef=[]
        for pos,vel,typ in ents[:N_NEAR]:
            r=pos-self.p[i]; rx=c*r[0]-s*r[1]; ry=s*r[0]+c*r[1]
            rvx=c*vel[0]-s*vel[1]; rvy=s*vel[0]+c*vel[1]
            ef += [clip(float(rx)/6,-1,1),clip(float(ry)/6,-1,1),clip(float(rvx),-1,1),clip(float(rvy),-1,1),
                   clip(float(np.linalg.norm(r))/6,0,1),typ]
        ef += [0.0]*(N_NEAR*6-len(ef))
        rays=[self.ray(i,float(self.th[i])+k*math.pi/4) for k in range(8)]
        return np.asarray(base+ef+rays,np.float32)
    def expert(self,i):
        d=self.g[i]-self.p[i]; desired=math.atan2(float(d[1]),float(d[0]))
        for j in range(self.n):
            if j==i or self.done[j]: continue
            q=self.p[i]-self.p[j]; dist=float(np.linalg.norm(q))
            if .01<dist<1.6: desired += .9*math.copysign(1.0, wrap(desired-math.atan2(float(q[1]),float(q[0]))))
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
        self.hp += self.hv*DT
        for j in range(self.nppl):
            for k in (0,1):
                if abs(float(self.hp[j,k]))>8.8: self.hp[j,k]=np.sign(self.hp[j,k])*8.8; self.hv[j,k]*=-1
        rewards=np.zeros(self.n,np.float32); done=np.zeros(self.n,bool)
        for i in range(self.n):
            if self.done[i]: done[i]=True; continue
            a=np.asarray(actions[i],np.float32); vnom=(float(a[0])+1)*.5*VMAX; wnom=float(a[1])*WMAX
            vs,ws=vnom,wnom
            if use_cbf:
                rs=[float(self.p[i,0]),float(self.p[i,1]),float(self.th[i]),float(self.v[i]),float(self.w[i])]
                vs,ws,diag=self.cbf.solve(vnom,wnom,rs,self.dyn_array(i),SHELVES)
                if diag.get("intervened"): self.interventions[i]+=1
                if diag.get("deadlock_detected"): self.deadlock[i]+=1
            self.th[i]=wrap(float(self.th[i])+ws*DT); self.v[i]=vs; self.w[i]=ws
            cand=self.p[i]+np.array([math.cos(float(self.th[i])),math.sin(float(self.th[i]))],np.float32)*vs*DT
            coll=self.static_collision(cand)
            if not coll:
                for j in range(self.n):
                    if j!=i and not self.done[j] and np.linalg.norm(cand-self.p[j])<2*ROBOT_R: coll=True; break
            if not coll:
                for q in self.hp:
                    if np.linalg.norm(cand-q)<ROBOT_R+ROBOT_R: coll=True; break
            if coll:self.hit[i]=True; self.done[i]=True
            else:
                self.p[i]=cand
                if np.linalg.norm(self.g[i]-self.p[i])<=GOAL_TOL:self.done[i]=True
            now=float(np.linalg.norm(self.g[i]-self.p[i])); prog=float(prev[i]-now)
            rewards[i]=10*prog-.02-.002*abs(ws)
            if coll: rewards[i]-=20
            elif self.done[i]:rewards[i]+=25
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
        suc,col=e.stats(); rows.append(dict(seed=sd,success=suc,collision=col,steps=e.steps,interventions=int(e.interventions.sum()),deadlock=int(e.deadlock.sum())))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--agent-steps",type=int,default=50000);ap.add_argument("--seed",type=int,default=52)
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
    torch.save({"actor":actor.state_dict(),"obs_dim":OBS_DIM,"seed":args.seed},out/"shared_sac_cbf.pt")
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
                "mean_deadlock":float(np.mean([r["deadlock"] for r in rows]))}
    result={"agent_steps":global_steps,"stages":logs,"aggregate":agg}
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(agg,indent=2),flush=True)

if __name__=="__main__":main()
