"""Audit the carrier readout under between-template offsets; keep held-out pairs together."""
import argparse
import json
from pathlib import Path

import numpy as np
from collect import save_json


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def evaluate(bank,states,target):
    bank=bank.astype(float);states=states.astype(float)
    n,q=target.shape
    assert q==2 and n==len(states)
    word_means=bank.mean(0)
    dictionary=unit(bank-word_means)
    # The held-out prefix contributes neither question to its calibration mean.
    loo=(states.sum(axis=(0,1))[None]-states.sum(1))/(2*(n-1))
    out={}
    for name,center in (("carrier",word_means),("crossfit",loo[:,None])):
        sims=np.einsum("nqd,wd->nqw",unit(states-center),dictionary)
        true=np.take_along_axis(sims,target[:,:,None],axis=2)[...,0]
        foil=np.take_along_axis(sims,target[:,::-1,None],axis=2)[...,0]
        pred=sims.argmax(2)
        rank=1+(sims>true[:,:,None]).sum(2)
        out[name]=dict(correct=int((pred==target).sum()),within_pair=int((true>foil).sum()),
                       rank_median=float(np.median(rank)),rank_mean=float(rank.mean()))
    return out


def main():
    p=argparse.ArgumentParser()
    for arg in ("blind","output"):p.add_argument(arg,type=Path)
    p.add_argument("--dense",type=Path)
    a=p.parse_args()
    records=json.loads((a.blind/"records.json").read_text())
    words=json.loads((a.blind/"words.json").read_text())
    targets=np.array([[words.index(r["values"][k]) for k in (r["edited"],1-r["edited"])] for r in records])
    if a.dense:
        arrays=np.load(a.dense)
        bank,queries=arrays["bank"],arrays["queries"]
        np.testing.assert_array_equal(targets,arrays["targets"])
        depths=range(15,24)
    else:
        bank=np.load(a.blind/"dictionary.npz")["query"]
        queries=np.load(a.blind/"inputs.npz")["clean_query"]
        depths=(15,23,27,31)
    out={str(d):evaluate(bank[:,j],queries[:,:,j],targets) for j,d in enumerate(depths)}
    save_json(a.output,out)


if __name__=="__main__":main()
