"""Paired causal effects, clean phrase accuracy, and depth-dependent readouts."""
import argparse
import json
from pathlib import Path

import numpy as np

from collect import LAYERS
from run import save_json


def read(path):
    return json.loads(path.read_text())


def interval(x):
    x=np.asarray(x,dtype=float)
    assert len(x)==64
    draws=np.random.default_rng(91062).integers(len(x),size=(10000,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[draws].mean(axis=1),[.025,.975]).tolist())


def count_flips(rows,patched):
    original=np.asarray([r["second"] for r in rows])
    alternate=original[:,::-1]
    clean=np.array([[s["top_token"] for s in r["clean"]] for r in rows])
    first=np.array([[c["first_top"] for c in r["clean"]] if "first_top" not in r else r["first_top"] for r in rows])
    correct_first=first==np.array([r["first"] for r in rows])[:,None]
    eligible=correct_first & (clean==original)
    return dict(clean_first=int(correct_first.sum()),clean_both_words=int(eligible.sum()),
                alternate_second_after_forced_first=int(np.sum(patched==alternate)),
                alternate_two_word_prefix=int(np.sum(correct_first & (patched==alternate))),
                flips_from_clean_correct=int(np.sum(eligible & (patched==alternate))),
                questions=int(clean.size))


def effects(rows,states,key):
    return np.mean([[side[key]["donor_minus_original"]-r["clean"][i]["donor_minus_original"]
                     for i,side in enumerate(sides)] for r,sides in zip(rows,states)],axis=1)


def patch_tops(states,key):
    return np.array([[s[key]["top_token"] for s in sides] for sides in states])


def block_results(path,selected):
    rows=read(path/"test.json")
    per_layer={};controls={}
    for layer in LAYERS:
        scan=[r["scan"][str(layer)] for r in rows]
        update=effects(rows,scan,"update");random=effects(rows,scan,"random")
        per_layer[layer]=update-random
        if layer==selected:
            for key in scan[0][0]:
                controls[key]=dict(gain=interval(effects(rows,scan,key)),top=count_flips(rows,patch_tops(scan,key)))
    return dict(block=selected,primary=interval(per_layer[selected]),controls=controls,
                selected_minus_other_seven=interval(per_layer[selected]-np.mean([v for k,v in per_layer.items() if k!=selected],axis=0)),
                depth={k:interval(v) for k,v in per_layer.items()})


def decode(path):
    with np.load(path/"food-depth.npz") as a:
        bank=a["bank"].astype(float);obs=a["queries"].astype(float);target=a["targets"]
    def unit(x):
        return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    output={}
    for j in range(len(LAYERS)+1):
        mean=bank[:,j].mean(0)
        pred=(unit(obs[:,:,j]-mean)@unit(bank[:,j]-mean).T).argmax(-1)
        relevance=pred==target[:,0,None]
        output[j+15]=dict(answer=int((pred==target).sum()),
                          relation=int((relevance==np.array([True,False])).sum()),
                          accuracy=interval((pred==target).mean(1)))
    assert output[15]["answer"]==0 and output[23]["answer"]==128
    return output


def head_results(path):
    cfg=read(path/"selection.json");head=cfg["head"]
    rows=read(path/"test.json");sides=[r["patches"] for r in rows]
    gain={k:effects(rows,sides,k) for k in sides[0][0]}
    controls=[str((head+i)%16) for i in (3,7,11)]
    return dict(block=cfg["block"],head=head,selected_minus_controls=interval(gain[str(head)]-np.mean([gain[k] for k in controls],axis=0)),
                marginal_given_others=interval(gain["all"]-gain["others"]),
                selected_minus_prior=interval(gain[str(head)]-gain["prior_23_0"]),
                interventions={k:dict(gain=interval(v),top=count_flips(rows,patch_tops(sides,k))) for k,v in gain.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("data","output"):p.add_argument(arg,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selected=read(a.data/"blocks/9b/selection.json")["layer"]
    for size in ("9b","4b"):
        blocks=a.data/"blocks"/size
        out=dict(model=size,blocks=block_results(blocks,selected),food=decode(blocks),heads=head_results(a.data/"heads"/size))
        save_json(a.output/(size+".json"),out)
