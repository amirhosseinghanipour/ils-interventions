"""Rebuild the early prediction from per-record vectors without the collection scorer."""
import argparse
from pathlib import Path

import numpy as np
from score import read


def verify(data, food, scales):
    train_rows=read(food/"records.json");train=np.load(food/"inputs.npz")["clean_query"][:,:,0].astype(float)
    labels=np.array([[r["edited"],1-r["edited"]] for r in train_rows])
    centroids=np.stack([train[labels==i].mean(0) for i in (0,1)])
    dic=np.load(data/"dictionary.npz")
    bank=dic["source"].astype(float)
    diffs=bank[None,:,:]-bank[:,None,:]
    norms=np.linalg.norm(diffs,axis=-1)
    directions=diffs/np.maximum(norms[:,:,None],1e-12)
    inputs=np.load(data/"inputs.npz"); y=np.load(data/"targets.npz")["y"].astype(float)
    sites=read(data/"sites.json"); rows=read(data/"records.json"); words=read(data/"words.json")
    food_names={n for r in train_rows for n in r["people"]}
    food_values={w for r in train_rows for w in r["values"]}
    assert food_names.isdisjoint(n for r in rows for n in r["people"])
    assert food_values.isdisjoint(w for r in rows for w in r["values"]+[r["replacement"]])
    amplitudes=np.array(read(scales)["scales"])
    carriers=dic["query"][:,-1].astype(float)
    gains=[]; pred=[]; null=[]; correct=pairs=0
    for i,r in enumerate(rows):
        sim=np.einsum("ijd,d->ij",directions,inputs["delta"][i])
        np.fill_diagonal(sim,-np.inf)
        original,donor=np.unravel_index(sim.argmax(),sim.shape)
        pairs+=words[original]==r["values"][r["edited"]] and words[donor]==r["replacement"]
        s=sites.index(r["site"])
        dist=np.square(centroids[:,None]-inputs["clean_query"][i,None,:,0]).sum(-1)
        choice=dist.argmin(0)
        correct+=int((choice==[r["edited"],1-r["edited"]]).sum())
        gain=carriers[donor]-carriers[original]
        gains.append(gain)
        pred.append(np.where(choice==s,amplitudes[0],amplitudes[1])[:,None]*gain)
        null.append(np.repeat((amplitudes.mean()*gain)[None],2,axis=0))
    score=read(data/"result.json")
    assert pairs==score["source_pairs"] and correct==score["slot_correct"]
    energy=np.square(y-y.mean((0,1))).sum()
    errors=[np.square(y-v).sum((1,2)) for v in (pred,null)]
    np.testing.assert_allclose(1-errors[0].sum()/energy,score["r2"]["slot"]["mean"])
    np.testing.assert_allclose((errors[1]-errors[0]).sum()/energy,score["primary"]["mean"])
    # Explicitly re-center all response vectors on a subset of the bootstrap draws.
    draws=np.random.default_rng(31016).integers(len(y),size=(10000,len(y)))
    counts=np.stack([np.bincount(idx,minlength=len(y)) for idx in draws])
    gram=y.sum(1)@y.sum(1).T
    energies=counts@np.square(y).sum((1,2))-np.einsum("bi,ij,bj->b",counts,gram,counts,optimize=True)/(2*len(y))
    for d in (0,23,105):
        batch=y[draws[d]]
        np.testing.assert_allclose(energies[d],np.square(batch-batch.mean((0,1))).sum(),rtol=1e-12)
    np.testing.assert_allclose(np.percentile(counts@(errors[1]-errors[0])/energies,[2.5,97.5]),score["primary"]["ci95"])
    if (data/"target-center.json").exists():
        res=read(data/"target-center.json")
        clean=inputs["clean_query"][:,:,0].astype(float)
        # Subtract the estimated shift, then compare Euclidean distances to
        # the original two training centroids, independently of the dot-product scorer.
        shift=clean[:16].mean((0,1))-centroids.mean(0)
        corrected=[];right=0
        for j,r in enumerate(rows[16:],16):
            z=clean[j]-shift
            cls=np.square(z[None]-centroids[:,None]).sum(-1).argmin(0)
            right+=int((cls==[r["edited"],1-r["edited"]]).sum())
            corrected.append(np.where(cls==sites.index(r["site"]),amplitudes[0],amplitudes[1])[:,None]*gains[j])
        actual=y[16:]
        ss=np.square(actual-actual.mean((0,1))).sum()
        err=np.square(corrected-actual).sum()
        assert right==res["centered_slot_correct"] and len(actual)==res["test_pairs"]
        np.testing.assert_allclose(1-err/ss,res["r2"]["centered"]["mean"])
        np.testing.assert_allclose((errors[1][16:].sum()-err)/ss,res["centered_minus_mean"]["mean"])
    print(f"Verified {len(rows)} material prefixes, disjoint words and names, recovered endpoints, early states, residual predictions and uncertainty.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("data","food","scales"):p.add_argument(key,type=Path)
    a=p.parse_args();verify(a.data,a.food,a.scales)
