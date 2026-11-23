"""Collect a source dictionary, clean query states and separately scored finite edits."""
import argparse
import json
from pathlib import Path

import numpy as np
from run import HERE, Model, chat, design, save_json

BLOCKS = (15,23,27)
CFG = json.loads((HERE/"blind.json").read_text())


def collect(model, prior_bank, out):
    prior = json.loads((prior_bank/"words.json").read_text())
    words = [row["word"] for row in prior]
    for word in CFG["foods"]:
        if len(model.ids(word)) == len(model.ids(" "+word)) == 1:
            assert word not in words
            words.append(word)
    sources, carriers = [], []
    for word in words:
        pre,post = chat("WORD","QUERY").split("Answer with only the lowercase color word.\nWORD\nQUERY")
        tokens = model.ids(pre+"Repeat exactly this lowercase word: "+word+"."+post)
        site, = np.flatnonzero(tokens==model.ids(" "+word)[0])
        source,final,_ = model.evaluate(tokens,int(site),None,0,0)
        sources.append(source)
        layers = [model.evaluate(tokens,len(tokens)-1,None,0,0,layer=b)[0] for b in BLOCKS]
        carriers.append(layers+[final[-1]])
    np.testing.assert_array_equal(np.stack(carriers)[:len(prior),-1],np.load(prior_bank/"vectors.npz")["vectors"])
    save_json(out/"words.json",words)
    np.savez(out/"dictionary.npz",source=sources,query=carriers)
    def render(facts,question):
        return chat(facts,question).replace("lowercase color word","lowercase food word")
    rows=list(design(model,"foods",seed=CFG["seed"],count=CFG["pairs"],name_bank=CFG["names"],
                     template_pair=(CFG["facts"],CFG["question"]),value_bank=CFG["foods"],
                     unique_facts=True,render_chat=render))
    collect_cases(model,rows,out)


def collect_cases(model,rows,out):
    deltas,states,targets=[],[],[]
    for row in rows:
        a,b,donor=(np.array(t,np.int32) for t in row["tokens"])
        site=row["site"]
        source,_,_=model.evaluate(a,site,None,0,0)
        replacement,_,_=model.evaluate(donor,site,None,0,0)
        displacement=replacement-source
        deltas.append(displacement)
        sides=[]; measured=[]; tops=[]
        for tokens in (a,b):
            baseline,final,obs=model.evaluate(tokens,site,None,0,0)
            np.testing.assert_array_equal(source,baseline)
            h=[model.evaluate(tokens,len(tokens)-1,None,0,0,layer=l)[0] for l in BLOCKS]
            sides.append(h+[final[-1]])
            _,edited,_=model.evaluate(tokens,site,displacement,0,0)
            np.testing.assert_allclose(edited[:site],final[:site],atol=1e-6,rtol=0)
            measured.append(edited[-1]-final[-1])
            tops.append(obs["top_token"])
        states.append(sides);targets.append(measured);row["clean_top"]=tops
    np.savez(out/"inputs.npz",delta=deltas,clean_query=states)
    np.savez(out/"targets.npz",y=targets)
    save_json(out/"records.json",rows)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("library","model","prior_bank","out"): p.add_argument(name,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    collect(Model(a.library,a.model),a.prior_bank,a.out)
