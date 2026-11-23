"""Test the frozen output-prototype specification on each complete prefix group."""
import argparse
import json
from pathlib import Path
import numpy as np

from run import save_json
from scalar_prior import fit


def score(train_dir,test_dir,carrier_dir,results):
    words=json.loads((carrier_dir/"words.json").read_text())
    bank=dict(zip([w["word"] for w in words],np.load(carrier_dir/"vectors.npz")["vectors"].astype(float),strict=True))
    features=[]
    for root,split in ((train_dir,"train"),(test_dir,"confirm")):
        rows=json.loads((root/(split+".json")).read_text())
        features.append(np.stack([bank[r["replacement"]]-bank[r["values"][r["edited"]]] for r in rows]))
    train,test=[np.load(d/(k+".npz")) for d,k in ((train_dir,"train"),(test_dir,"confirm"))]
    x,y,xt,yt=[v.astype(float) for v in (train["x"],train["y"],test["x"],test["y"])]
    scalar,corrected,penalty,scale=fit(x,y,xt,*features)
    scores=np.load(results/"full-sufficient.npz")
    n=len(yt)
    gram=scores["gram"]
    draw=np.random.default_rng(48221).integers(n,size=(10000,n))
    freq=np.array([np.bincount(row,minlength=n) for row in draw])
    ss=scores["total"].sum()-gram.sum()/(2*n)
    den=freq@scores["total"]-((freq@gram)*freq).sum(1)/(2*n)
    predictions={"shared_prototype":scale.mean()*features[1][:,None],"prototype":scalar,"corrected":corrected}
    errors={k:np.square(pred-yt).sum((1,2)) for k,pred in predictions.items()}
    def stat(estimate,boot):
        return dict(estimate=float(estimate),interval=np.quantile(boot,[.025,.975]).tolist())
    out={k:stat(1-e.sum()/ss,1-freq@e/den) for k,e in errors.items()}
    delta=scores["sse_conditional"]-errors["corrected"]
    out["corrected minus conditional"]=stat(delta.sum()/ss,freq@delta/den)
    save_json(results/"prototypes.json",dict(pairs=n,carrier_words=len(words),carrier_top1=sum(w["top"]==w["expected"] for w in words),
                                           scales=scale.tolist(),penalty=penalty,scores=out))
    np.savez_compressed(results/"prototypes-sufficient.npz",**errors)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("train","test","carriers","results"): p.add_argument(arg,type=Path)
    a=p.parse_args()
    score(a.train,a.test,a.carriers,a.results)
