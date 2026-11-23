"""Fit a relevance-specific scalar copy and a ridge correction using color training only."""
import argparse
import json
from pathlib import Path
import numpy as np

from score_routing import PENALTIES, predict_paths
from run import save_json


def copies(x,y,xt):
    # One scalar per question relation, fitted through the origin.
    scale=np.einsum("nd,nqd->q",x,y) / np.square(x).sum()
    return x[:,None]*scale[None,:,None],xt[:,None]*scale[None,:,None],scale


def fit(x,y,xt,anchor=None,target_anchor=None):
    if anchor is None:
        anchor,target_anchor=x,xt
    cv=np.zeros(len(PENALTIES))
    folds=np.array_split(np.random.default_rng(44017).permutation(len(x)),5)
    for test in folds:
        train=np.setdiff1d(np.arange(len(x)),test)
        source,dest,_=copies(anchor[train],y[train],anchor[test])
        residual=(y[train]-source).reshape(len(train),-1)
        for k,pred in enumerate(predict_paths(x[train],residual,x[test])):
            cv[k]+=np.square(y[test]-dest-pred.reshape(dest.shape)).sum()
    source,dest,scale=copies(anchor,y,target_anchor)
    best=int(cv.argmin())
    residual=(y-source).reshape(len(y),-1)
    prediction=list(predict_paths(x,residual,xt))[best].reshape(dest.shape)
    return dest,dest+prediction,PENALTIES[best],scale


def score(train_dir,test_dir,results):
    train,test=[np.load(d/(k+".npz")) for d,k in ((train_dir,"train"),(test_dir,"confirm"))]
    x,y,xt,yt=[v.astype(float) for v in (train["x"],train["y"],test["x"],test["y"])]
    scalar,corrected,penalty,scale=fit(x,y,xt)
    original=np.load(results/"full-sufficient.npz")
    n=len(yt)
    draw=np.random.default_rng(48221).integers(n,size=(10000,n))
    freq=np.array([np.bincount(row,minlength=n) for row in draw])
    gram=original["gram"]
    ss=original["total"].sum()-gram.sum()/(2*n)
    den=freq@original["total"]-((freq@gram)*freq).sum(1)/(2*n)
    err={name:np.square(pred-yt).sum((1,2)) for name,pred in (("scalar",scalar),("corrected",corrected))}
    def interval(value,samples):
        return dict(estimate=float(value),interval=np.quantile(samples,[.025,.975]).tolist())
    scores={k:interval(1-e.sum()/ss,1-freq@e/den) for k,e in err.items()}
    for k in ("conditional","lexical"):
        diff=original["sse_"+k]-err["corrected"]
        scores["corrected minus "+k]=interval(diff.sum()/ss,freq@diff/den)
    save_json(results/"scalar-prior.json",dict(pairs=n,penalty=penalty,scales=scale.tolist(),scores=scores))
    np.savez_compressed(results/"scalar-prior-sufficient.npz",**err)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("train","test","results"): p.add_argument(arg,type=Path)
    args=p.parse_args()
    score(args.train,args.test,args.results)
