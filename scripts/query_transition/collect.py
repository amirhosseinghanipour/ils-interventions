"""Paired two-token retrieval and causal block-update substitution on native GGUF models."""
import argparse
import ctypes as ct
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"query_routing"))
from run import Model, chat, save_json

CFG=json.loads((HERE/"protocol.json").read_text())
LAYERS=CFG["layers"]


class Tracer(Model):
    def __init__(self,library,model):
        super().__init__(library,model)
        self.lib.configure_trace.argtypes=[ct.c_void_p]
        self.lib.configure_trace.restype=None

    def clean(self,tokens,alternative,original):
        trace=np.empty((9,2,self.width),np.float32)
        self.lib.configure_trace(trace.ctypes.data)
        try:
            source,_,obs=self.evaluate(tokens,len(tokens)-1,None,alternative,original,layer=15)
        finally:
            self.lib.configure_trace(None)
        np.testing.assert_array_equal(source,trace[0,1])
        return trace,obs


def template(name,lead,tail,split):
    if split=="dev":
        return f"The code for {name} is {lead} {tail}."
    return f"{name} uses the phrase {lead} {tail}."


def make_pairs(model,split):
    rng=np.random.default_rng(CFG[split+"_seed"])
    def single(name):
        return [w for w in CFG[name+"_"+split] if len(model.ids(w))==len(model.ids(" "+w))==1]
    leads,tails,names=map(single,("lead","tail","names"))
    pre,post=chat("FACT","QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")
    seen=set()
    while len(seen)<CFG[split+"_pairs"]:
        people=rng.choice(names,2,replace=False).tolist()
        words=rng.choice(tails,2,replace=False).tolist()
        lead=str(rng.choice(leads))
        facts=" ".join(template(n,lead,w,split) for n,w in zip(people,words))
        if facts in seen: continue
        seen.add(facts)
        prompts=[]
        for name in people:
            question=f"What is the code for {name}?" if split=="dev" else f"Give the phrase assigned to {name}."
            prompts.append(pre+"Answer with the two lowercase words only.\n"+facts+"\n"+question+post)
        ids=[model.ids(t) for t in prompts]
        assert len(ids[0])==len(ids[1]) and len(np.flatnonzero(ids[0]!=ids[1]))==1
        first=int(model.ids(lead)[0])
        second=[int(model.ids(" "+w)[0]) for w in words]
        for text,base,word,token in zip(prompts,ids,words,second):
            np.testing.assert_array_equal(model.ids(text+lead+" "+word),[*base,first,token])
        yield dict(people=people,tails=words,lead=lead,prompts=prompts,tokens=[a.tolist() for a in ids],
                   first=first,second=second)


def study_pair(model,row,seed,selected):
    rng=np.random.default_rng(seed)
    tokens=[np.array(t+[row["first"]],np.int32) for t in row["tokens"]]
    traces=[]; clean=[]
    for side,t in enumerate(tokens):
        a,b=row["second"][1-side],row["second"][side]
        obs=model.evaluate(t[:-1],len(t)-2,None,a,b)[2]
        trace,baseline=model.clean(t,a,b)
        repeated=model.evaluate(t,len(t)-1,None,a,b)[2]
        assert repeated==baseline
        traces.append(trace)
        clean.append(dict(first_top=obs["top_token"],**baseline))
    scans={}
    for k,layer in enumerate(LAYERS,1):
        # Directed difference in increments between question 1 and question 0.
        diff=traces[1]-traces[0]
        update=diff[k]-diff[k-1]
        random=rng.normal(size=model.width).astype(np.float32)
        random*=np.linalg.norm(update[1])/np.linalg.norm(random)
        np.testing.assert_allclose(np.linalg.norm(random),np.linalg.norm(update[1]),rtol=1e-6)
        per_side=[]
        for side,t in enumerate(tokens):
            a,b=row["second"][1-side],row["second"][side]
            signed=np.float32(1-2*side)
            probes={"update":(-1,update[1]),"random":(-1,random)}
            if layer==selected:
                probes.update(previous=(-2,update[0]),whole=(-1,diff[k,1]),sham=(-1,np.zeros(model.width,np.float32)))
            measured={}
            for key,(position,edit) in probes.items():
                obs=model.evaluate(t,len(t)+position,signed*edit,a,b,layer=layer)[2]
                measured[key]=obs
                if key=="sham": assert obs=={n:clean[side][n] for n in obs}
            per_side.append(measured)
        scans[str(layer)]=per_side
    return dict(**row,clean=clean,scan=scans),np.stack(traces)


def gain(records,layer):
    return np.mean([side["update"]["donor_minus_original"]-side["random"]["donor_minus_original"]
                    for r in records for side in r["scan"][str(layer)]])


def food_depth(model,data,out):
    words=json.loads((data/"words.json").read_text())
    records=json.loads((data/"records.json").read_text())
    pre,post=chat("FACT","QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")
    def trace(t):
        return model.clean(np.array(t,np.int32),0,0)[0][:,-1]
    bank=np.stack([trace(model.ids(pre+"Repeat exactly this lowercase word: "+w+"."+post)) for w in words])
    queries=np.stack([[trace(t) for t in r["tokens"][:2]] for r in records])
    for i,prior in ((0,0),(8,1)):
        np.testing.assert_array_equal(bank[:,i],np.load(data/"dictionary.npz")["query"][:,prior])
        np.testing.assert_array_equal(queries[:,:,i],np.load(data/"inputs.npz")["clean_query"][:,:,prior])
    targets=np.array([[words.index(r["values"][j]) for j in (r["edited"],1-r["edited"])] for r in records])
    np.savez(out/"food-depth.npz",bank=bank,queries=queries,targets=targets)


def main():
    p=argparse.ArgumentParser()
    for key in ("library","model","food","out"): p.add_argument(key,type=Path)
    p.add_argument("--selection",type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    model=Tracer(a.library,a.model)
    selected=json.loads(a.selection.read_text())["layer"] if a.selection else None
    for split in (["test"] if a.selection else ["dev","test"]):
        records=[];traces=[]
        for i,row in enumerate(make_pairs(model,split)):
            record,x=study_pair(model,row,1000*CFG[split+"_seed"]+i,selected)
            records.append(record);traces.append(x)
            if (i+1)%16==0: print(split,i+1,flush=True)
        save_json(a.out/(split+".json"),records)
        np.savez(a.out/(split+"-trace.npz"),states=np.stack(traces))
        if split=="dev":
            scores={str(l):float(gain(records,l)) for l in LAYERS}
            selected=max(LAYERS,key=lambda l:scores[str(l)])
            save_json(a.out/"selection.json",dict(layer=selected,means=scores))
            print("frozen",selected,flush=True)
    food_depth(model,a.food,a.out)
    print("completed",a.out.name,flush=True)


if __name__=="__main__": main()
