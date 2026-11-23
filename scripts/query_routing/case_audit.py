"""Decode clean 0.5B predictions to distinguish formatting errors from different answers."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from run import save_json


def audit(root, tokenizer_file):
    meta=json.loads((root/"checkpoint.json").read_text())["files"]["tokenizer.json"]
    assert hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()==meta["sha256"]
    tok=Tokenizer.from_file(str(tokenizer_file))
    rows=json.loads((root/"confirm.json").read_text())
    items=[]
    for r in rows:
        result={"index":r["index"]}
        for i,side in enumerate(("relevant","irrelevant")):
            result[side]=[tok.decode([k],skip_special_tokens=False) for k in (r["correct"][i],r["top"][i][0])]
        items.append(result)
    matches=[[(a.casefold()==b.casefold()) for a,b in (v["relevant"],v["irrelevant"])] for v in items]
    return dict(exact=sum(a==b for v in items for a,b in (v["relevant"],v["irrelevant"])),
                case_insensitive=sum(sum(pair) for pair in matches),both_case_insensitive=sum(all(pair) for pair in matches),
                questions=2*len(items),items=items)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("root","tokenizer","output"): p.add_argument(arg,type=Path)
    args=p.parse_args()
    save_json(args.output,audit(args.root,args.tokenizer))
