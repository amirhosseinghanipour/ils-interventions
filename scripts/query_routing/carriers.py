"""Collect one clean first-answer residual per vocabulary item on a neutral carrier prompt."""
import argparse
import json
from pathlib import Path

import numpy as np
from run import COLORS, HERE, Model, chat, save_json


def main():
    p=argparse.ArgumentParser()
    for arg in ("library","model","out"): p.add_argument(arg,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    model=Model(a.library,a.model)
    items=json.loads((HERE/"transfer.json").read_text())["items"]
    animals=json.loads((HERE/"prototypes.json").read_text())["animals"]
    words=[w for w in COLORS+items+animals if len(model.ids(w))==len(model.ids(" "+w))==1]
    assert len(words)==len(set(words))
    observations=[]; vectors=[]
    for word in words:
        pre,post=chat("WORD","QUERY").split("Answer with only the lowercase color word.\nWORD\nQUERY")
        prompt=pre+"Repeat exactly this lowercase word: "+word+"."+post
        tokens=model.ids(prompt)
        expected=int(model.ids(word)[0])
        _,final,meta=model.evaluate(tokens,0,None,expected,expected)
        observations.append(dict(word=word,text=prompt,tokens=tokens.tolist(),expected=expected,top=meta["top_token"]))
        vectors.append(final[-1])
    save_json(a.out/"words.json",observations)
    np.savez(a.out/"vectors.npz",vectors=np.stack(vectors))


if __name__=="__main__": main()
