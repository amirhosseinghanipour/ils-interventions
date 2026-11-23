"""Fit paired residual predictors on development prefixes and evaluate the fixed holdout."""
import argparse
import json
from pathlib import Path

import numpy as np

from run import COLORS, save_json

PENALTIES = [0, 1e-6, 1e-3, .1, 1.]
DRAWS = 10000


def paired_ci(values, draw):
    values = np.asarray(values)
    return {"mean":float(values.mean()),"interval":np.quantile(values[draw].mean(1),[.025,.975]).tolist()}


def predict_paths(x,y,xt):
    # Rows of y concatenate the two flattened response vectors or their shared mean.
    gram = x @ x.T
    vals,vecs = np.linalg.eigh(gram)
    keep=vals > vals.max()*max(x.shape)*np.finfo(float).eps
    test=(xt @ x.T) @ vecs[:,keep]
    labels=vecs[:,keep].T @ y
    scale=gram.diagonal().mean()
    for penalty in PENALTIES:
        yield (test/(vals[keep]+penalty*scale)) @ labels


def fit(x,y,xt):
    folds=np.array_split(np.random.default_rng(44017).permutation(len(x)),5)
    flat=y.reshape(len(y),-1)
    cv=np.zeros(len(PENALTIES))
    for take in folds:
        train=np.setdiff1d(np.arange(len(x)),take)
        for k,p in enumerate(predict_paths(x[train],flat[train],x[take])):
            cv[k] += np.square(flat[take]-p).sum()
    best=int(cv.argmin())
    return list(predict_paths(x,flat,xt))[best].reshape((len(xt),)+y.shape[1:]), PENALTIES[best]


def colors(records):
    x=np.zeros((len(records),len(COLORS)))
    for i,row in enumerate(records):
        for word, sign in [(row["replacement"], 1), (row["values"][row["edited"]], -1)]:
            if word in COLORS:
                x[i,COLORS.index(word)]=sign
    return x


def compare(root,small=False,training_root=None):
    roots = [training_root or root,root]
    records=[json.loads((r/(s+".json")).read_text()) for r,s in zip(roots,("train","confirm"))]
    train,test=[np.load(r/(s+".npz")) for r,s in zip(roots,("train","confirm"))]
    x,xt=(v["x"].astype(float) for v in (train,test))
    field="small" if small else "y"
    y,yt=(v[field].astype(float) for v in (train,test))
    assert y.shape[:2]==(128,2) and yt.shape[:2]==(48,2)
    # IDs, one-token counterfactual, split integrity and paired input equality.
    sets=[]
    for items in records:
        prefixes=[]
        for row in items:
            a,b,d=np.array(row["tokens"])
            assert np.flatnonzero(a!=b).tolist()==[row["cue"]]
            assert np.flatnonzero(a!=d).tolist()==[row["site"]]
            assert row["site"]<row["cue"] and len(a)==row["length"]
            prefixes.append(tuple(a[:row["cue"]]))
        assert len(set(prefixes))==len(items)
        sets.append(set(prefixes))
    assert sets[0].isdisjoint(sets[1])
    n=len(xt)
    preds={}; lambdas={}
    preds["shared"],lambdas["shared"]=fit(x,y.mean(1)[:,None],xt)
    preds["conditional"],lambdas["conditional"]=fit(x,y,xt)
    permuted=y.copy()
    swaps=np.random.default_rng(112316).permutation(len(x))[:len(x)//2]
    permuted[swaps]=permuted[swaps,::-1]
    preds["permuted"],lambdas["permuted"]=fit(x,permuted,xt)
    preds["lexical"],lambdas["lexical"]=fit(colors(records[0]),y,colors(records[1]))
    preds["class_mean"]=y.mean(0)[None]
    preds["zero"]=np.zeros((1,1,y.shape[-1]))
    if "local" in test:
        scale=.05 if small else 1.
        preds["local_J"]=scale*test["local"]
        preds["mean_J"]=scale*test["mean_j"][:,None]
    err={k:np.square(value-yt).sum((1,2)) for k,value in preds.items()}
    sums=yt.sum(1)
    gram=sums @ sums.T
    total=np.square(yt).sum((1,2))
    draw=np.random.default_rng(48221).integers(n,size=(DRAWS,n))
    counts=np.array([np.bincount(d,minlength=n) for d in draw])
    ss=total.sum()-gram.sum()/(2*n)
    ss_boot=counts @ total - ((counts@gram)*counts).sum(1)/(2*n)
    def stat(est,dist):
        return dict(estimate=float(est),interval=np.quantile(dist,[.025,.975]).tolist())
    out={k:stat(1-e.sum()/ss,1-counts@e/ss_boot) for k,e in err.items()}
    contrasts={}
    for a,b in [("conditional","shared"),("conditional","lexical"),("conditional","permuted")]+(
            [("local_J","mean_J"),("local_J","conditional")] if "local_J" in err else []):
        diff=err[b]-err[a]; contrasts[a+" minus "+b]=stat(diff.sum()/ss,counts@diff/ss_boot)
    ratios=[np.square(yt[:,0]-yt[:,1]).sum(1)/4,np.square(yt).sum((1,2))/2]
    bound=stat(ratios[0].sum()/ratios[1].sum(),(counts@ratios[0])/(counts@ratios[1]))
    margins=np.array([r["shifts"] for r in records[1]])
    m=margins[:,0]-margins[:,1]
    clean=np.array([[r["top"][i][0]==r["correct"][i] for i in range(2)] for r in records[1]])
    top_changes=np.array([[r["top"][i][1]==r["donor_token"] for i in range(2)] for r in records[1]])
    behavioral=dict(clean_correct=int(clean.sum()),both=int(clean.all(1).sum()),difference=paired_ci(m,draw),
                    relevant=paired_ci(margins[:,0],draw),irrelevant=paired_ci(margins[:,1],draw),
                    edited_donor_top1_counts=top_changes.sum(0).tolist())
    if "random_shifts" in records[1][0]:
        ran=np.array([r["random_shifts"] for r in records[1]])
        behavioral["random_difference"]=paired_ci(ran[:,0]-ran[:,1],draw)
    suff=dict(total=total,gram=gram,lower_numerator=ratios[0],lower_denominator=ratios[1],**{"sse_"+k:v for k,v in err.items()})
    return dict(pairs=n,r2=out,paired_r2=contrasts,penalties=lambdas,source_only_error_floor=bound,
                full_edit_behavior=behavioral),suff


def head_results(root):
    screen=json.loads((root/"screen.json").read_text()) if (root/"screen.json").exists() else None
    data=json.loads((root/"head-confirm.json").read_text())
    winner=screen["winner"] if screen else "23/0"
    block, head=map(int,winner.split("/"))
    keys=list(data[0])
    assert keys==[winner]+[f"{block}/{(head+i)%16}" for i in (1,5,9)]
    draw=np.random.default_rng(48221).integers(len(data),size=(DRAWS,len(data)))
    result={"winner":winner,"controls":keys[1:]}
    if screen: result["dev_reduction"]=screen["means"][winner]
    for field in ("reduction","sufficiency","base"):
        arr=np.array([[r[k][0][field]-r[k][1][field] for k in keys] for r in data])
        result[field]={k:paired_ci(arr[:,i],draw) for i,k in enumerate(keys)}
        result[field]["selected_minus_controls"]=paired_ci(arr[:,0]-arr[:,1:].mean(1),draw)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--training-root",type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for field in (["full","small"] if (args.root/"numerical.json").exists() else ["full"]):
        stats,suff=compare(args.root,field=="small",args.training_root)
        save_json(args.out/(field+".json"),stats)
        np.savez_compressed(args.out/(field+"-sufficient.npz"),**suff)
    if (args.root/"head-confirm.json").exists():
        save_json(args.out/"heads.json",head_results(args.root))
