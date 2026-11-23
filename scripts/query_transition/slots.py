"""Test mention-order information without train/test overlap in names or answer words."""
import argparse
from pathlib import Path

import numpy as np
from collect import save_json
from score import read


def measure(queries, records, swapped):
    # Reorder for constructing labels and scoring only. Classify each test question
    # independently using a train-only center and difference of class centroids.
    labels = np.array([[r["edited"], 1-r["edited"]] for r in records])
    train_slots = labels ^ swapped[:,None]
    depth = queries.shape[2]
    scores = np.empty((len(records),2,depth))
    counts = []
    for i,record in enumerate(records):
        idx = np.array([j for j,r in enumerate(records)
                        if set(r["people"]).isdisjoint(record["people"])
                        and set(r["values"]).isdisjoint(record["values"])])
        assert len(idx)>0 and i not in idx
        counts.append(len(idx))
        train = queries[idx]
        c0 = train[train_slots[idx]==0].mean(0)
        c1 = train[train_slots[idx]==1].mean(0)
        scores[i] = np.einsum("qld,ld->ql", queries[i]-(c0+c1)/2, c1-c0)
    return ((scores>0)==labels[:,:,None]).sum((0,1)), counts


def sign_null(queries, records, observed):
    labels = np.array([[r["edited"], 1-r["edited"]] for r in records])
    n,_,layers,_ = queries.shape
    kernels = np.zeros((n,2,layers,n))
    directions = queries[np.arange(n),1-labels[:,0]]-queries[np.arange(n),labels[:,0]]
    for i,r in enumerate(records):
        train = [j for j,t in enumerate(records) if set(t["people"]).isdisjoint(r["people"])
                 and set(t["values"]).isdisjoint(r["values"])]
        center = queries[train].mean((0,1))
        kernels[i][:,:,train] = np.einsum("qld,nld->qln",queries[i]-center,directions[train])/len(train)
    # Flip both the training labels and held-out truth of each affected pair.
    # Cache inner products so all permutations reuse the identical classifier.
    rng = np.random.default_rng(31036)
    reps = 4095
    null = []
    for start in range(0,reps,64):
        signs = 1-2*rng.integers(2,size=(min(64,reps-start),n))
        scores = (signs@kernels.reshape(-1,n).T).reshape(-1,n,2,layers)
        truth = labels[None,:,:,None] ^ (signs[:,:,None,None]<0)
        null.extend(((scores>0)==truth).sum((1,2)))
    identity = kernels.sum(-1)
    np.testing.assert_array_equal(observed, ((identity>0)==labels[:,:,None]).sum((0,1)))
    null = np.array(null)
    return dict(draws=reps,quantiles=np.percentile(null,[2.5,50,97.5],axis=0).tolist(),
                max_depth_adjusted_p=((1+(null.max(1)[:,None]>=observed).sum(0))/(1+reps)).tolist())


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("food","dense","out"): p.add_argument(name,type=Path)
    a=p.parse_args()
    with np.load(a.dense) as raw:
        x=raw["queries"].astype(float)
    records=read(a.food/"records.json")
    true, counts=measure(x,records,np.zeros(len(records),int))
    # A seeded per-prefix sign shuffle disrupts slot labels but preserves pairs.
    shuffled,_=measure(x,records,np.random.default_rng(31035).integers(2,size=len(records)))
    save_json(a.out,dict(questions=2*len(records),min_train=min(counts),max_train=max(counts),
                         correct=dict(zip(range(15,24),true.tolist())),
                         shuffled=dict(zip(range(15,24),shuffled.tolist())),sign_null=sign_null(x,records,true)))
