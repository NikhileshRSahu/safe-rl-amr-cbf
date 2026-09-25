from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.optim import Adam
from benchmark.intent_shift_benchmark import IntentShiftWorld, TRAIN_SEEDS
from benchmark.train_multi_agent_research import Actor,Q,Replay


def evaluate(actor,seeds):
    rows=[]; actor.eval()
    for seed in seeds:
        w=IntentShiftWorld(4,12,int(seed));obs=w.reset()
        for _ in range(600):
            with torch.no_grad():a,_=actor.sample(torch.tensor(np.asarray(obs),dtype=torch.float32),True)
            obs,_,done=w.step(a.numpy(),True)  # deployment always uses CBF
            if np.all(done):break
        success=w.done & ~w.hit
        rows.append({'seed':int(seed),'success':int(success.sum()),'collision':int(w.hit.sum()),'timeout':int((~w.done).sum()),'fleet_success':bool(success.all()),'steps':int(w.steps)})
    n=4*len(rows)
    return {'rows':rows,'success_rate':sum(r['success'] for r in rows)/n,'collision_rate':sum(r['collision'] for r in rows)/n,'timeout_rate':sum(r['timeout'] for r in rows)/n,'fleet_success_rate':sum(r['fleet_success'] for r in rows)/len(rows)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--agent-steps',type=int,default=80000);ap.add_argument('--seed',type=int,default=81);ap.add_argument('--out',required=True);args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    actor=Actor();actor.load_state_dict(ck['actor']);q1=Q();q2=Q();tq1=Q();tq2=Q();tq1.load_state_dict(q1.state_dict());tq2.load_state_dict(q2.state_dict())
    ao=Adam(actor.parameters(),1.2e-4);qo=Adam(list(q1.parameters())+list(q2.parameters()),3e-4);buf=Replay(cap=300000)
    gamma=.99;alpha=.08;tau=.01;rng=np.random.default_rng(args.seed+7000);steps=0;episodes=0;w=None;obs=None;history=[];val=tuple(range(13005,13010))
    while steps<args.agent_steps:
        if w is None or np.all(w.done) or w.steps>=600:
            sd=int(TRAIN_SEEDS[int(rng.integers(0,len(TRAIN_SEEDS)))]);w=IntentShiftWorld(4,12,sd);obs=w.reset();episodes+=1
        s=np.asarray(obs,dtype=np.float32)
        with torch.no_grad():
            acts=actor.sample(torch.tensor(s),False)[0].numpy() if steps>=1000 else np.clip(actor.sample(torch.tensor(s),True)[0].numpy()+rng.normal(0,.10,(4,2)),-1,1)
        ns,r,d=w.step(acts,False)  # nominal learning; collisions remain strongly penalized by environment
        for i in range(4):buf.add(s[i],acts[i],r[i],ns[i],float(d[i]))
        obs=ns;steps+=4
        if buf.n>=1500 and steps%4==0:
            bs,ba,br,bns,bd=buf.sample(192)
            with torch.no_grad():na,nlp=actor.sample(bns);y=br+gamma*(1-bd)*(torch.min(tq1(bns,na),tq2(bns,na))-alpha*nlp)
            ql=F.mse_loss(q1(bs,ba),y)+F.mse_loss(q2(bs,ba),y);qo.zero_grad();ql.backward();torch.nn.utils.clip_grad_norm_(list(q1.parameters())+list(q2.parameters()),5);qo.step()
            pa,lp=actor.sample(bs);al=(alpha*lp-torch.min(q1(bs,pa),q2(bs,pa))).mean();ao.zero_grad();al.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),5);ao.step()
            with torch.no_grad():
                for p,tp in zip(q1.parameters(),tq1.parameters()):tp.mul_(1-tau).add_(tau*p)
                for p,tp in zip(q2.parameters(),tq2.parameters()):tp.mul_(1-tau).add_(tau*p)
        if steps%20000==0 or steps>=args.agent_steps:
            ev=evaluate(actor,val);rec={'agent_steps':steps,'episodes':episodes,'validation':ev};history.append(rec);print(json.dumps(rec),flush=True)
            torch.save({'actor':actor.state_dict(),'base_checkpoint':'Run11','training':'intent_nominal_runtime_cbf','seed':args.seed,'agent_steps':steps},out/f'intent_nominal_{steps}.pt')
    def key(x):
        v=x['validation'];return(v['collision_rate'],v['timeout_rate'],-v['fleet_success_rate'],-v['success_rate'])
    best=min(history,key=key);name=f"intent_nominal_{best['agent_steps']}.pt";(out/'summary.json').write_text(json.dumps({'seed':args.seed,'history':history,'best':best,'best_checkpoint':name},indent=2));print(json.dumps({'best':best,'best_checkpoint':name},indent=2))
if __name__=='__main__':main()
