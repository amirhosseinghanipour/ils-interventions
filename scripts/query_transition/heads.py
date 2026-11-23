"""Head-level sufficiency and joint interventions on independently ordered word lists."""
import argparse
import json
from pathlib import Path

import numpy as np
from collect import CFG, make_pairs
from followup import HeadModel
from run import chat, save_json


def assembly(model):
    rng=np.random.default_rng(31033)
    eligible=lambda pool:[w for w in pool if len(model.ids(w))==len(model.ids(" "+w))==1]
    names=eligible(CFG["names_test"]); leads=eligible(CFG["lead_test"]); tails=eligible(CFG["tail_test"])
    pre,post=chat("FACT","QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")
    seen=set()
    for i in range(64):
        people=rng.choice(names,4,replace=False).tolist()
        left=rng.choice(leads,2,replace=False).tolist()
        first=[left[0],left[0],left[1],left[1]]
        last=rng.choice(tails,4,replace=False).tolist()
        def section(label,words):
            return label+" "+", ".join(f"{people[j]} = {words[j]}" for j in rng.permutation(4))+"."
        facts=section("First words:",first)+"\n"+section("Last words:",last)
        assert facts not in seen
        seen.add(facts)
        prompts=[pre+"Give only the matching first and last words, in that order.\n"+facts+
                 f"\nWhat are the two words for {n}?"+post for n in people[:2]]
        ids=[model.ids(t) for t in prompts]
        t0=int(model.ids(first[0])[0])
        second=[int(model.ids(" "+w)[0]) for w in last[:2]]
        assert len(ids[0])==len(ids[1]) and len(np.flatnonzero(ids[0]!=ids[1]))==1
        for q in range(2):
            np.testing.assert_array_equal(model.ids(prompts[q]+first[q]+" "+last[q]),[*ids[q],t0,second[q]])
        yield dict(index=i,people=people,first_words=first,tails=last,prompts=prompts,
                   tokens=[t.tolist() for t in ids],first=t0,second=second)


def test(model,rows,block,selected):
    output=[]; raw=[]
    for row in rows:
        model.head()
        tokens=[np.array(t+[row["first"]],np.int32) for t in row["tokens"]]
        def obs(side,first=False):
            ids=tokens[side][:-1] if first else tokens[side]
            return model.evaluate(ids,len(ids)-1,None,row["second"][1-side],row["second"][side])[2]
        first=[obs(i,True)["top_token"] for i in range(2)]
        base=[]; states=[]
        for i in range(2):
            model.head(block)
            base.append(obs(i))
            states.append(model.buffer.copy())
        prior=states
        if selected is not None and block!=23:
            prior=[]
            for i in range(2):
                model.head(23)
                assert obs(i)==base[i]
                prior.append(model.buffer.copy())
        result=[]
        for i in range(2):
            donor=states[1-i]; original=states[i]
            heads=range(16) if selected is None else [selected]+[(selected+k)%16 for k in (3,7,11)]
            side={}
            for h in heads:
                model.head(block,h,donor[h])
                side[str(h)]=obs(i)
            if selected is not None:
                others=donor.copy();others[selected]=original[selected]
                for key,replacement in (("all",donor),("others",others),("sham",original)):
                    model.head(block,-2,replacement);side[key]=obs(i)
                assert side["sham"]==base[i]
                model.head(23,0,prior[1-i][0])
                side["prior_23_0"]=obs(i)
                # Independent contiguous/single-slice APIs must agree exactly.
                one=original.copy();one[selected]=donor[selected]
                model.head(block,-2,one);assert obs(i)==side[str(selected)]
            result.append(side)
        output.append(dict(**row,clean=base,first_top=first,patches=result));raw.append(states)
    model.head()
    return output,np.stack(raw)


def main():
    p=argparse.ArgumentParser()
    for name in ("library","model","selection","out"):p.add_argument(name,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    model=HeadModel(a.library,a.model)
    scores=json.loads(a.selection.read_text())["means"]
    block=max((19,23),key=lambda b:scores[str(b)])
    dev,dev_states=test(model,make_pairs(model,"dev"),block,None)
    gains={h:float(np.mean([r["patches"][i][str(h)]["donor_minus_original"]-r["clean"][i]["donor_minus_original"]
                            for r in dev for i in range(2)])) for h in range(16)}
    winner=max(gains,key=gains.get)
    save_json(a.out/"selection.json",dict(block=block,head=winner,means=gains))
    save_json(a.out/"dev.json",dev)
    print("selected head",block,winner,flush=True)
    rows,states=test(model,assembly(model),block,winner)
    np.savez(a.out/"states.npz",dev=dev_states,test=states)
    save_json(a.out/"test.json",rows)
    print("completed",a.out.name,flush=True)


if __name__=="__main__":main()
