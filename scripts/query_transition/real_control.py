"""Compare the target query displacement with an equal-size displacement from another pair."""
import argparse
import json
from pathlib import Path

import numpy as np
from collect import Tracer, save_json
from natural import sample
from score import read


def crossed(edits, i):
    other = edits[(i+1) % len(edits)]
    assert np.linalg.norm(other) > 0
    out = other * (np.linalg.norm(edits[i])/np.linalg.norm(other))
    np.testing.assert_allclose(np.linalg.norm(out), np.linalg.norm(edits[i]), rtol=1e-6)
    return out


def inputs(model, store, dataset, task, layer):
    tokens, labels, edits, observed = [], [], [], []
    if task == "blocks":
        states = np.load(store/"test-trace.npz")["states"]
        for r,t in zip(read(store/"test.json"), states):
            d = t[1,:,1] - t[0,:,1]
            edits.append(d[layer-15] - d[layer-16])
            tokens.append([np.array(x+[r["first"]], np.int32) for x in r["tokens"]])
            labels.append(r["second"])
            observed.append([s["update"] for s in r["scan"][str(layer)]])
    else:
        rows = read(store/"rows.json")
        for i,(article, pi, c) in enumerate(sample(model, json.loads(dataset.read_text()))):
            r = rows[i]
            assert article == r["article"] and pi == r["paragraph"] and [x[0] for x in c] == r["qids"]
            ids = [x[2] for x in c]
            gold = [x[1] for x in c]
            assert gold == r["labels"]
            traces = []
            for q in (0,1):
                t, baseline = model.clean(ids[q], gold[1-q], gold[q])
                assert baseline == r["clean"][q]
                traces.append(t[:,1])
            d = traces[1] - traces[0]
            edits.append(d[layer-15] - d[layer-16])
            tokens.append(ids)
            labels.append(gold)
            observed.append([s["update"] for s in r["patches"]])
    return tokens, labels, edits, observed


def measure(model, site, arrays):
    tokens, labels, edits, observed = arrays
    recorded = []
    for i,ids in enumerate(tokens):
        j = (i + 1) % len(edits)
        side = []
        for q in (0,1):
            def apply(delta):
                return model.evaluate(ids[q], len(ids[q])-1, (1-2*q)*delta, labels[i][1-q], labels[i][q], layer=site)[2]
            assert apply(edits[i]) == observed[i][q]
            side.append(apply(crossed(edits,i)))
        recorded.append(dict(index=i,other=j,answers=labels[i],shared=sorted(set(labels[i])&set(labels[j])),
                             correct_donor=observed[i],other_donor=side))
    return recorded


def summarize(rows):
    vals=np.array([np.mean([s["donor_minus_original"]-t["donor_minus_original"]
                            for s,t in zip(r["correct_donor"],r["other_donor"])]) for r in rows])
    n=len(rows)
    draws=np.random.default_rng(91062).integers(n,size=(10000,n))
    return dict(n=n, overlapping_answer_pairs=sum(bool(r["shared"]) for r in rows),
                mean=float(vals.mean()),ci95=np.percentile(vals[draws].mean(1), [2.5,97.5]).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("library","model","blocks","natural","dataset","selection","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    model=Tracer(a.library,a.model)
    layer=read(a.selection)["layer"]
    for task in ("blocks","natural"):
        rows=measure(model,layer,inputs(model,getattr(a,task),a.dataset,task,layer))
        save_json(a.out/(task+".json"),rows)
        save_json(a.out/(task+"-result.json"),summarize(rows))
