"""Cross-check split integrity, measured logit contrasts, and frozen model comparisons."""
import argparse
from pathlib import Path

import numpy as np
from score import LAYERS, read

MARGIN="donor_minus_original"


def primary(pairs,stored):
    pairs=np.asarray(pairs).mean(1)
    draws=np.random.default_rng(91062).integers(64,size=(10000,64))
    np.testing.assert_allclose(stored["mean"],np.mean(pairs),atol=1e-10)
    np.testing.assert_allclose(stored["ci95"],np.percentile(pairs[draws].mean(1),[2.5,97.5]),atol=1e-10)


def audit(data,results):
    dev=read(data/"blocks/9b/dev.json")
    scores={b:np.mean([r["scan"][str(b)][q]["update"][MARGIN]-r["scan"][str(b)][q]["random"][MARGIN]
                       for r in dev for q in (0,1)]) for b in LAYERS}
    winner=max(scores,key=scores.get)
    attention=max((19,23),key=scores.get)
    for size in ("4b","9b"):
        summary=read(results/(size+".json"))
        assert winner==summary["blocks"]["block"]
        block=data/"blocks"/size;head=data/"heads"/size
        rows=read(block/"test.json");traces=np.load(block/"test-trace.npz")["states"]
        assert len(rows)==len(traces)==64
        assert {w for r in rows for w in r["tails"]}.isdisjoint(w for r in dev for w in r["tails"])
        for r,t in zip(rows,traces):
            assert t.shape[:3]==(2,9,2) and np.isfinite(t).all()
            # Check telescoping of query-specific block increments over all eight blocks.
            d=t[1].astype(float)-t[0].astype(float)
            np.testing.assert_allclose(np.diff(d,axis=0).sum(0),d[-1]-d[0],atol=1e-9)
            assert all(r["scan"][str(winner)][i]["sham"][MARGIN]==r["clean"][i][MARGIN] for i in (0,1))
        primary([[s["update"][MARGIN]-s["random"][MARGIN] for s in r["scan"][str(winner)]] for r in rows],
                summary["blocks"]["primary"])
        records=read(head/"test.json");dev_heads=read(head/"dev.json")
        best=max(range(16),key=lambda h:sum(s[str(h)][MARGIN] for r in dev_heads for s in r["patches"]))
        assert best==summary["heads"]["head"] and attention==summary["heads"]["block"]
        ctrl=[str((best+i)%16) for i in (3,7,11)]
        primary([[s[str(best)][MARGIN]-np.mean([s[h][MARGIN] for h in ctrl]) for s in r["patches"]] for r in records],
                summary["heads"]["selected_minus_controls"])
        for r in rows+records:
            a,b=np.asarray(r["tokens"]);assert len(np.flatnonzero(a!=b))==1 and len(set(r["second"]))==2
        assert len({r["prompts"][0] for r in records})==64
        for r in records:
            assert r["first_words"][0]==r["first_words"][1]!=r["first_words"][2]
            assert len(set(r["tails"]))==4
            for i in (0,1):
                assert r["patches"][i]["sham"]==r["clean"][i]
        # Check the full-width transport predictions using a dual solve independent of eigenvalue fitting.
        trans=data/"transport"/size;arr=np.load(trans/"responses.npz");res=read(trans/"result.json")
        x,y,xt,yt=(arr[k].astype(float).reshape(-1,arr[k].shape[-1]) for k in ("x","y","xt","yt"))
        ss=np.square(yt-yt.mean(0)).sum()
        alpha=np.sum(x*y)/np.sum(x*x)
        np.testing.assert_allclose(alpha,res["scalar"])
        gram=x@x.T
        for j,(label,base) in enumerate((("ridge",0.),("corrected",alpha))):
            k=res["penalties"][j]
            K=gram+np.eye(len(gram))*k*np.trace(gram)/len(gram)
            dual=np.linalg.solve(K,y-base*x) if k else np.linalg.pinv(K,rcond=max(x.shape)*np.finfo(float).eps)@(y-base*x)
            pred=base*xt+(xt@x.T)@dual
            np.testing.assert_allclose(1-np.square(yt-pred).sum()/ss,res["r2"][label]["estimate"],atol=1e-8)
    print("Both block screens, independent head selections, primary bootstrap intervals, shams, splits and transport fits verified.")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("data",type=Path);p.add_argument("results",type=Path)
    a=p.parse_args();audit(a.data,a.results)
