"""Independently replay the query-routing estimates from published raw measurements."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    np.testing.assert_allclose(a, b, atol=3e-6, rtol=2e-6)


def primary(data, results):
    stored = read(results / "v1/outcome.json")["results"]["test"]
    rows = [read(p) for p in sorted((data/"v1").glob("test-[0-9]*.json"))]
    assert len(rows) == 32
    effects, lower, total = [], [], []
    for row in rows:
        contrasts = []
        for label in ("donor1", "random1"):
            changes = []
            for side in ("relevant", "irrelevant"):
                obs = row["measurements"][side]
                changes.append(obs[label+"_plus"]["donor_minus_original"] - obs["clean"]["donor_minus_original"])
            contrasts.append(changes[0]-changes[1])
        effects.append(contrasts)
        with np.load(data/"v1"/f'test-{row["index"]:03}.npz') as raw:
            close(np.linalg.norm(raw["direction"]), np.linalg.norm(raw["random"]))
            a, b = (raw[k+"_donor1_plus"].astype(float) for k in ("relevant","irrelevant"))
            close(a[:row["cue"]], b[:row["cue"]])
            lower.append(np.square(a[-1]-b[-1]).sum()/2)
            total.append(np.square(a[-1]).sum()+np.square(b[-1]).sum())
    close(np.mean(effects,axis=0), [stored["behavior"][k]["mean"] for k in ("primary","random")])
    close(sum(lower)/sum(total),stored["destination_residual"]["donor1"]["source_only_mse_lower_bound_over_zero_predictor_mse"])


def kernel_predict(x, xt, y, penalty):
    k = x @ x.T
    penalty *= np.trace(k)/len(x)
    if penalty:
        coeff = np.linalg.solve(k + penalty*np.eye(len(k)), x @ xt.T)
    else:
        coeff = np.linalg.pinv(k, rcond=max(x.shape)*np.finfo(float).eps) @ (x @ xt.T)
    return (coeff.T @ y.reshape(len(y),-1)).reshape(len(xt),*y.shape[1:])


def lexical(rows, vocabulary):
    features = np.zeros((len(rows),len(vocabulary)))
    for i,r in enumerate(rows):
        for word,sign in ((r["replacement"],1),(r["values"][r["edited"]],-1)):
            if word in vocabulary:
                features[i,vocabulary.index(word)] += sign
    return features


def audit_set(train_dir, test_dir, result_dir, field, carrier_dir=None):
    rows, test_rows = [read(folder/(key+".json")) for folder,key in ((train_dir,"train"),(test_dir,"confirm"))]
    train, test = np.load(train_dir/"train.npz"),np.load(test_dir/"confirm.npz")
    outcome = read(result_dir/(field+".json"))
    key = "small" if field=="small" else "y"
    y,yt,x,xt=(np.asarray(a,dtype=float) for a in (train[key],test[key],train["x"],test["x"]))
    assert (len(y),len(yt))==(128,48)
    value_sets = [{w for r in group for w in r["values"]+[r["replacement"]]} for group in (rows,test_rows)]
    if test_dir!=train_dir:
        assert value_sets[0].isdisjoint(value_sets[1])
    # Center each residual coordinate across all 96 query rows for the point estimator.
    # The bootstrap preserves their pairings.
    centered = yt-yt.mean(axis=(0,1))
    denominator = np.square(centered).sum()
    vocab = sorted(value_sets[0])
    predictions = {}
    for model, labels in (("shared",y.mean(1,keepdims=True)),("conditional",y),("lexical",y)):
        a,b = (lexical(rows,vocab),lexical(test_rows,vocab)) if model=="lexical" else (x,xt)
        predictions[model] = kernel_predict(a,b,labels,outcome["penalties"][model])
    shuffle=y.copy()
    flips=np.random.default_rng(112316).permutation(len(y))[:len(y)//2]
    shuffle[flips]=shuffle[flips,::-1]
    predictions["permuted"]=kernel_predict(x,xt,shuffle,outcome["penalties"]["permuted"])
    predictions["class_mean"] = y.mean(0)
    predictions["zero"] = np.zeros(yt.shape[-1])
    if "local" in test:
        eps = .05 if field=="small" else 1.
        predictions.update(local_J=eps*test["local"].astype(float),mean_J=eps*test["mean_j"].astype(float)[:,None])
    saved = np.load(result_dir/(field+"-sufficient.npz"))
    for model, estimate in predictions.items():
        sse = np.square(yt-estimate).sum(axis=(1,2))
        close(sse,saved["sse_"+model])
        close(1-sse.sum()/denominator,outcome["r2"][model]["estimate"])
    midpoint=yt.mean(axis=1,keepdims=True)
    close(np.square(yt-midpoint).sum()/np.square(yt).sum(),outcome["source_only_error_floor"]["estimate"])
    real = outcome["full_edit_behavior"]
    shift = np.array([r["shifts"] for r in test_rows])
    close((shift[:,0]-shift[:,1]).mean(),real["difference"]["mean"])
    assert sum(r["top"][i][0]==r["correct"][i] for r in test_rows for i in range(2))==real["clean_correct"]
    # Recreate the published bootstrap numerically by direct centering on a subset of draws.
    draws=np.random.default_rng(48221).integers(48,size=(10000,48))
    counts=np.array([np.bincount(sample,minlength=48) for sample in draws])
    boot_den=counts@saved["total"]-((counts@saved["gram"])*counts).sum(1)/96
    for j in (0,1,13,9999):
        sample=yt[draws[j]]
        close(boot_den[j],np.square(sample-sample.mean(axis=(0,1))).sum())
    for model in outcome["r2"]:
        intervals=np.quantile(1-counts@saved["sse_"+model]/boot_den,[.025,.975])
        close(intervals,outcome["r2"][model]["interval"])
    if (result_dir/"scalar-prior.json").exists():
        prior=read(result_dir/"scalar-prior.json")
        scalars=np.array([np.vdot(x,y[:,q])/np.vdot(x,x) for q in (0,1)])
        close(scalars,prior["scales"])
        projected=xt[:,None]*scalars[:,None]
        resid=y-x[:,None]*scalars[:,None]
        predicted=projected+kernel_predict(x,xt,resid,prior["penalty"])
        published=np.load(result_dir/"scalar-prior-sufficient.npz")
        for label, pred in (("corrected",predicted),("scalar",projected)):
            err=np.square(yt-pred).sum((1,2))
            close(err,published[label])
            close(1-err.sum()/denominator,prior["scores"][label]["estimate"])
            close(np.quantile(1-counts@err/boot_den,[.025,.975]),prior["scores"][label]["interval"])
    if carrier_dir is not None:
        records=read(carrier_dir/"words.json")
        vectors=np.load(carrier_dir/"vectors.npz")["vectors"].astype(float)
        lookup={row["word"]:vec for row,vec in zip(records,vectors,strict=True)}
        def changes(group):
            return np.stack([lookup[r["replacement"]]-lookup[r["values"][r["edited"]]] for r in group])
        tr,te=changes(rows),changes(test_rows)
        coef=np.array([np.sum(tr*y[:,q])/np.square(tr).sum() for q in range(2)])
        published=read(result_dir/"prototypes.json")
        close(coef,published["scales"])
        assert sum(w["expected"]==w["top"] for w in records)==published["carrier_top1"]
        base=te[:,None]*coef[:,None]
        pred=base+kernel_predict(x,xt,y-tr[:,None]*coef[:,None],published["penalty"])
        errors=np.load(result_dir/"prototypes-sufficient.npz")
        for label, pr in (("shared_prototype",te[:,None]*coef.mean()),("prototype",base),("corrected",pred)):
            err=np.square(yt-pr).sum((1,2))
            close(err,errors[label])
            close(1-err.sum()/denominator,published["scores"][label]["estimate"])
            close(np.quantile(1-counts@err/boot_den,[.025,.975]),published["scores"][label]["interval"])
        diff=saved["sse_conditional"]-errors["corrected"]
        close(diff.sum()/denominator,published["scores"]["corrected minus conditional"]["estimate"])
        close(np.quantile(counts@diff/boot_den,[.025,.975]),published["scores"]["corrected minus conditional"]["interval"])
    if (test_dir/"head-confirm.json").exists():
        heads=read(result_dir/"heads.json")
        records=read(test_dir/"head-confirm.json")
        winner=heads["winner"]
        assert len(records)==48 and list(records[0])==[winner]+heads["controls"]
        for m in ("reduction","sufficiency"):
            values=[]
            for i,r in enumerate(records):
                measurements=[]
                for h,sides in r.items():
                    close([v["base"] for v in sides],shift[i])
                    measurements.append(sides[0][m]-sides[1][m])
                values.append(measurements[0]-np.mean(measurements[1:]))
            close(np.mean(values),heads[m]["selected_minus_controls"]["mean"])
            close(np.quantile(np.array(values)[draws].mean(1),[.025,.975]),heads[m]["selected_minus_controls"]["interval"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    p.add_argument("results",type=Path)
    args=p.parse_args()
    primary(args.data,args.results)
    for model in ("9b","4b","05b"):
        for field in (("full","small") if model=="05b" else ("full",)):
            audit_set(args.data/model,args.data/model,args.results/model,field,
                      args.data/"carriers"/model if model!="05b" else None)
        if model!="05b":
            for task in ("parcels","animals"):
                audit_set(args.data/model,args.data/task/model,args.results/task/model,"full",args.data/"carriers"/model)
    print("Primary, four color tables, four transfer tables, six prototype fits and both head confirmations verified.")
