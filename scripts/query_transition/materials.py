"""Use a food-trained early mention-order probe to predict finite effects on materials."""
import argparse
from pathlib import Path

import numpy as np
from collect import HERE, Model, save_json
from blind_collect import BLOCKS, collect_cases
from blind_score import infer
from run import chat, design
from score import read

CFG = read(HERE/"materials.json")


def collect(model, food, out):
    old = read(food/"words.json")
    words = [w for w in CFG["words"] if w not in old and len(model.ids(w))==len(model.ids(" "+w))==1]
    assert len(words)>=3
    with np.load(food/"dictionary.npz") as dic:
        sources, carriers = list(dic["source"]),list(dic["query"])
    front, back = chat("FACT","QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")
    for w in words:
        ids=model.ids(front+"Repeat exactly this lowercase word: "+w+"."+back)
        site, = np.flatnonzero(ids==model.ids(" "+w)[0])
        source,final,_=model.evaluate(ids,int(site),None,0,0)
        sources.append(source)
        carriers.append([model.evaluate(ids,len(ids)-1,None,0,0,layer=b)[0] for b in BLOCKS]+[final[-1]])
    def render(facts,question):
        return chat(facts,question).replace("lowercase color word","lowercase material word")
    def records(seed,count):
        return design(model,"materials",seed=seed,count=count,name_bank=CFG["names"],
                      template_pair=(CFG["facts"],CFG["question"]),value_bank=words,
                      unique_facts=True,render_chat=render)
    # Locate the two editable positions on two calibration prompts, using tokenization only.
    first,second = records(17,2)
    sites = [first["site"],second["site"]]
    assert sites[0]<sites[1]
    save_json(out/"sites.json",sites)
    save_json(out/"words.json",old+words)
    np.savez(out/"dictionary.npz",source=sources,query=carriers)
    rows=list(records(CFG["seed"],CFG["pairs"]))
    assert all(r["site"] in sites for r in rows)
    collect_cases(model, rows, out)


def score(food, test, scales_file):
    train=np.load(food/"inputs.npz")["clean_query"][:,:,0].astype(float)
    lab=np.array([[r["edited"],1-r["edited"]] for r in read(food/"records.json")])
    a,b=[train[lab==i].mean(0) for i in (0,1)]
    inputs=np.load(test/"inputs.npz")
    x=inputs["delta"].astype(float)
    query=inputs["clean_query"].astype(float)
    dic=np.load(test/"dictionary.npz")
    bank=dic["query"].astype(float); source=dic["source"].astype(float)
    scales=np.asarray(read(scales_file)["scales"])
    records=read(test/"records.json")
    sites=read(test/"sites.json")
    edited=np.repeat([sites.index(r["site"]) for r in records],2)
    n=len(records); perm=np.random.default_rng(31038).permutation(2*n)
    flat=query.reshape(-1,*query.shape[2:])[perm]
    old,new,answer=infer(source,bank,np.repeat(x,2,axis=0)[perm],flat)
    slot=((flat[:,0]-(a+b)/2)@(b-a)>0).astype(int)
    gain=bank[new,-1]-bank[old,-1]
    def pred(relation):
        p=gain*np.where(relation,scales[0],scales[1])[:,None]
        return p[np.argsort(perm)].reshape(n,2,-1)
    # No answer annotations, clean top tokens or finite-effect targets enter these predictions.
    out={"slot":pred(slot==edited[perm]), "early_word":pred(answer[:,0]==old), "later_word":pred(answer[:,1]==old)}
    out["mean"]=(scales.mean()*gain[np.argsort(perm)]).reshape(n,2,-1)
    words=read(test/"words.json")
    orig=np.array([words.index(r["values"][r["edited"]]) for r in records])
    repl=np.array([words.index(r["replacement"]) for r in records])
    oracle=(bank[repl,-1]-bank[orig,-1])[:,None,:]*scales[None,:,None]
    out["oracle"]=oracle
    y=np.load(test/"targets.npz")["y"].astype(float)
    ysum=y.sum(1); energy=np.square(y).sum((1,2)); ss=np.square(y-y.mean((0,1))).sum()
    draws=np.random.default_rng(31016).integers(n,size=(10000,n))
    counts=np.array([np.bincount(row,minlength=n) for row in draws])
    den=counts@energy-((counts@(ysum@ysum.T))*counts).sum(1)/(2*n)
    err={k:np.square(y-p).sum((1,2)) for k,p in out.items()}
    def stat(mu,boot):
        return dict(mean=float(mu),ci95=np.percentile(boot,[2.5,97.5]).tolist())
    d=err["mean"]-err["slot"]
    order=np.argsort(perm)
    target_slot=np.array([[r["edited"],1-r["edited"]] for r in records])
    return dict(n=n, candidates=len(words),sites=sites,slot_correct=int((slot[order].reshape(n,2)==target_slot).sum()),
                source_pairs=int(((old[order][::2]==orig)&(new[order][::2]==repl)).sum()),
                clean_top=sum(t==g for r in records for t,g in zip(r["clean_top"],r["correct"])),
                r2={k:stat(1-e.sum()/ss,1-counts@e/den) for k,e in err.items()},
                primary=stat(d.sum()/ss,(counts@d)/den))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("library","model","food","scales","out"):p.add_argument(name,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    collect(Model(a.library,a.model),a.food,a.out)
    save_json(a.out/"result.json",score(a.food,a.out,a.scales))
