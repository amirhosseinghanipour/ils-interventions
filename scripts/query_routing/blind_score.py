"""Decode endpoints and relevance from model states with annotations withheld."""
import argparse
import json
from pathlib import Path

import numpy as np
from run import save_json


def unit(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)


def infer(source,query,delta,clean):
    # The prediction path receives numeric dictionaries and activation vectors only.
    start,end=np.where(~np.eye(len(source),dtype=bool))
    direction=unit(source[end]-source[start])
    best=(unit(delta)@direction.T).argmax(1)
    old,new=start[best],end[best]
    answers=[]
    for j in range(query.shape[1]):
        center=query[:,j].mean(0)
        answers.append((unit(clean[:,j]-center)@unit(query[:,j]-center).T).argmax(1))
    return old,new,np.stack(answers,axis=1)


def analyze(root, scale_file):
    scales=np.array(json.loads(scale_file.read_text())["scales"])
    dic=np.load(root/"dictionary.npz")
    source=dic["source"].astype(float);query=dic["query"].astype(float)
    inputs=np.load(root/"inputs.npz")
    # Scramble query rows before decoding so that array order cannot encode relevance.
    x=np.repeat(inputs["delta"].astype(float),2,axis=0)
    clean=inputs["clean_query"].reshape(len(x),4,-1).astype(float)
    order=np.random.default_rng(32501).permutation(len(x))
    old,new,answer=infer(source,query,x[order],clean[order])
    undo=np.argsort(order)
    old,new,answer=old[undo],new[undo],answer[undo]
    gain=query[new,-1]-query[old,-1]
    preds={"mean_scale":scales.mean()*gain}
    layers=("15","23","27","31")
    for j,layer in enumerate(layers):
        preds[layer]=gain*np.where(answer[:,j]==old,scales[0],scales[1])[:,None]

    # Labels and actual finite responses enter only after blind predictions exist.
    rows=json.loads((root/"records.json").read_text())
    words=json.loads((root/"words.json").read_text())
    true_old=np.repeat([words.index(r["values"][r["edited"]]) for r in rows],2)
    true_new=np.repeat([words.index(r["replacement"]) for r in rows],2)
    correct=np.array([words.index(r["values"][slot]) for r in rows for slot in (r["edited"],1-r["edited"])])
    relevant=(correct==true_old)
    oracle=(query[true_new,-1]-query[true_old,-1])*np.where(relevant,scales[0],scales[1])[:,None]
    preds["oracle"]=oracle
    y=np.load(root/"targets.npz")["y"].astype(float)
    n=len(y)
    assert (n,len(rows),len(set(tuple(r["tokens"][0][:r["cue"]]) for r in rows)))==(64,64,64)
    for r in rows:
        a,b,d=np.array(r["tokens"])
        assert np.flatnonzero(a!=b).tolist()==[r["cue"]] and np.flatnonzero(a!=d).tolist()==[r["site"]]
    draws=np.random.default_rng(31016).integers(n,size=(10000,n))
    counts=np.array([np.bincount(d,minlength=n) for d in draws])
    sum_vec=y.sum(1)
    energy=np.square(y).sum((1,2))
    denominator=np.square(y-y.mean((0,1))).sum()
    bootden=counts@energy-((counts@(sum_vec@sum_vec.T))*counts).sum(1)/(2*n)
    err={k:np.square(v.reshape(y.shape)-y).sum((1,2)) for k,v in preds.items()}
    def stat(value,boot):
        return dict(estimate=float(value),interval=np.quantile(boot,[.025,.975]).tolist())
    scores={k:stat(1-e.sum()/denominator,1-counts@e/bootden) for k,e in err.items()}
    diff=err["mean_scale"]-err["15"]
    effect=stat(diff.sum()/denominator,counts@diff/bootden)
    return dict(pairs=n,candidates=len(words),source_pair_correct=int(np.sum((old[::2]==true_old[::2])&(new[::2]==true_new[::2]))),
                clean_top_correct=sum(r["correct"][i]==r["clean_top"][i] for r in rows for i in range(2)),
                answer_correct={l:int(np.sum(answer[:,j]==correct)) for j,l in enumerate(layers)},
                relevance_correct={l:int(np.sum((answer[:,j]==old)==relevant)) for j,l in enumerate(layers)},
                r2=scores,primary_gain_over_mean_scale=effect)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("root","scale_file","out"):p.add_argument(arg,type=Path)
    args=p.parse_args();save_json(args.out,analyze(args.root,args.scale_file))
