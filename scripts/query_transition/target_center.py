"""Test an unlabeled template offset on a prefix-disjoint calibration/evaluation split."""
import argparse
from pathlib import Path

import numpy as np
from materials import infer, save_json
from score import read


def analyze(food, materials, scale_file):
    train=np.load(food/"inputs.npz")["clean_query"][:,:,0].astype(float)
    labels=np.array([[r["edited"],1-r["edited"]] for r in read(food/"records.json")])
    zero,one=(train[labels==q].mean(0) for q in (0,1))
    inputs=np.load(materials/"inputs.npz")
    all_queries=inputs["clean_query"].astype(float)
    center=all_queries[:16,:,0].mean((0,1))
    query=all_queries[16:]
    n=len(query)
    delta=inputs["delta"][16:].astype(float)
    dic=np.load(materials/"dictionary.npz");bank=dic["query"].astype(float)
    source=dic["source"].astype(float)
    # One state per held-out question is passed to the classifier. The 16
    # prefix pairs determine a single fixed offset, independent of held-out states.
    perm=np.random.default_rng(31038).permutation(2*n); undo=np.argsort(perm)
    q=query.reshape(2*n,*query.shape[2:])[perm]
    x=np.repeat(delta,2,axis=0)[perm]
    old,new,answer=infer(source,bank,x,q)
    gain=bank[new,-1]-bank[old,-1]
    frozen=((q[:,0]-(zero+one)/2)@(one-zero)>0)
    logits=(q[:,0]-center)@(one-zero)
    centered=logits>0
    rows=read(materials/"records.json")[16:]; sites=read(materials/"sites.json")
    site=np.repeat([sites.index(r["site"]) for r in rows],2)[perm]
    amp=np.asarray(read(scale_file)["scales"])
    def scaled(relevant):
        return (gain*np.where(relevant,amp[0],amp[1])[:,None])[undo].reshape(n,2,-1)
    pred={"frozen":scaled(frozen==site),"centered":scaled(centered==site),
          "late":scaled(answer[:,1]==old),"mean":(gain*amp.mean())[undo].reshape(n,2,-1)}
    # Only scoring uses word labels, slot labels and the finite-response targets.
    words=read(materials/"words.json")
    true=np.array([[r["edited"],1-r["edited"]] for r in rows])
    orig=np.array([words.index(r["values"][r["edited"]]) for r in rows])
    dest=np.array([words.index(r["replacement"]) for r in rows])
    pred["oracle"]=(bank[dest,-1]-bank[orig,-1])[:,None,:]*amp[None,:,None]
    y=np.load(materials/"targets.npz")["y"][16:].astype(float)
    err={k:np.square(v-y).sum((1,2)) for k,v in pred.items()}
    ss=np.square(y-y.mean((0,1))).sum()
    draws=np.random.default_rng(31016).integers(n,size=(10000,n))
    counts=np.stack([np.bincount(s,minlength=n) for s in draws])
    gram=y.sum(1)@y.sum(1).T
    den=counts@np.square(y).sum((1,2))-((counts@gram)*counts).sum(1)/(2*n)
    def stat(value,samples):
        return dict(mean=float(value),ci95=np.quantile(samples,[.025,.975]).tolist())
    d=err["mean"]-err["centered"]
    scr=logits[undo].reshape(n,2)
    pair_order=(scr[np.arange(n),1-true[:,0]]>scr[np.arange(n),true[:,0]])
    return dict(calibration_pairs=16,test_pairs=n,
                frozen_slot_correct=int((frozen[undo].reshape(n,2)==true).sum()),
                centered_slot_correct=int((centered[undo].reshape(n,2)==true).sum()),
                correctly_ordered_pairs=int(pair_order.sum()),
                r2={k:stat(1-v.sum()/ss,1-counts@v/den) for k,v in err.items()},
                centered_minus_mean=stat(d.sum()/ss,(counts@d)/den))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("food","materials","scales","output"):p.add_argument(arg,type=Path)
    a=p.parse_args();save_json(a.output,analyze(a.food,a.materials,a.scales))
