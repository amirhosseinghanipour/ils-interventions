"""Replay selected updates and fit out-of-sample finite-effect predictors."""
import argparse
from pathlib import Path

import numpy as np
from collect import Tracer, make_pairs, save_json
from score import interval, read


def collect(model,rows,layer,recorded):
    x=[];y=[]
    for idx,row in enumerate(rows):
        tokens=[np.array(t+[row["first"]],np.int32) for t in row["tokens"]]
        clean=[];final=[]
        for q,t in enumerate(tokens):
            trace,_=model.clean(t,row["second"][1-q],row["second"][q])
            clean.append(trace[:,1])
            final.append(model.evaluate(t,len(t)-1,None,0,0)[1][-1])
        diff=clean[1]-clean[0]
        edit=diff[layer-15]-diff[layer-16]
        inputs=[];outputs=[]
        for q,t in enumerate(tokens):
            signed=(1-2*q)*edit
            _,patched,obs=model.evaluate(t,len(t)-1,signed,row["second"][1-q],row["second"][q],layer=layer)
            if recorded:
                assert obs==recorded[idx]["scan"][str(layer)][q]["update"]
            inputs.append(signed);outputs.append(patched[-1]-final[q])
        x.append(inputs);y.append(outputs)
    return np.array(x),np.array(y)


def paths(x,y,xt,prior):
    x=x.reshape(-1,x.shape[-1]); y=y.reshape(x.shape)
    alpha=np.sum(x*y)/np.sum(x*x) if prior else 0.
    gram=x@x.T
    val,v=np.linalg.eigh(gram)
    keep=val>val.max()*max(x.shape)*np.finfo(float).eps
    val=val[keep];v=v[:,keep]
    weights=(xt@x.T)@v
    target=v.T@(y-alpha*x)
    for k in (0.,.001,.1,1.,10.):
        yield alpha*xt+(weights/(val+k*gram.diagonal().mean()))@target


def predict(x,y,xt,prior):
    cv=np.zeros(5);folds=np.array_split(np.arange(len(x)),4)
    for test in folds:
        train=np.setdiff1d(np.arange(len(x)),test)
        for i,p in enumerate(paths(x[train],y[train],x[test],prior)):
            cv[i]+=np.square(y[test]-p).sum()
    best=int(cv.argmin())
    return list(paths(x,y,xt,prior))[best],(0.,.001,.1,1.,10.)[best]


def score(x,y,xt,yt):
    x,y,xt,yt=(a.astype(float) for a in (x,y,xt,yt))
    alpha=np.sum(x*y)/np.square(x).sum()
    ridge,penalty=predict(x,y,xt,False);corrected,second=predict(x,y,xt,True)
    predictions=dict(direct=xt,scalar=alpha*xt,ridge=ridge,corrected=corrected)
    err={k:np.square(v-yt).sum((1,2)) for k,v in predictions.items()}
    # Gram form permits exact re-centering in every bootstrap sample.
    sums=yt.sum(1);gram=sums@sums.T;n=len(xt);energy=np.square(yt).sum((1,2))
    ss=np.square(yt-yt.mean((0,1))).sum()
    draws=np.random.default_rng(91062).integers(n,size=(10000,n))
    counts=np.array([np.bincount(row,minlength=n) for row in draws])
    den=counts@energy-((counts@gram)*counts).sum(1)/(2*n)
    def stat(mean,z):return dict(estimate=float(mean),ci95=np.quantile(z,[.025,.975]).tolist())
    out={k:stat(1-e.sum()/ss,1-counts@e/den) for k,e in err.items()}
    diff=err["scalar"]-err["corrected"]
    return dict(r2=out,corrected_minus_scalar=stat(diff.sum()/ss,counts@diff/den),
                fraction_scalar_error_removed=stat(diff.sum()/err["scalar"].sum(),(counts@diff)/(counts@err["scalar"])),
                scalar=float(alpha),penalties=[penalty,second])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("library","model","blocks","selection","out"):p.add_argument(arg,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    model=Tracer(a.library,a.model);layer=read(a.selection)["layer"]
    arrays=[]
    for split in ("dev","test"):
        rows=list(make_pairs(model,split));path=a.blocks/(split+".json")
        observed=read(path) if path.exists() else None
        arrays.extend(collect(model,rows,layer,observed))
    np.savez(a.out/"responses.npz",**dict(zip(("x","y","xt","yt"),arrays)))
    save_json(a.out/"result.json",score(*arrays))
