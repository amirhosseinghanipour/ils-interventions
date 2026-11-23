"""Roll each fixed predictor out on its own greedy token history on the test articles."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect import WEIGHTS, greedy
from ilslens.finite_response import partial_response, prefix_secants, secant_transport
from ilslens.suffix import DecoderSuffix
from sample import SQUAD_SHA, digest, get_ids

METHODS = ("exact", "secant_transport", "prefix_secants", "partial_donor_mix", "midpoint_derivative")


def normalize(s):
    # Official SQuAD v1 normalization. Input answers in this task have one reference word.
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def rollout(model, ids, stops, weights):
    clean, donor, site = ids
    depth = len(model.model.layers)
    source, mid, last = depth//2-1, 3*depth//4-1, depth-1

    @lru_cache(maxsize=64)
    def features(history):
        # Only one clean recipient/donor pair is needed for a shared generation prefix.
        def replay(side):
            tokens = torch.tensor([side+list(history)],device=model.device)
            return DecoderSuffix(model,tokens)
        r, d = replay(clean), replay(donor)
        base = r.states[source]
        delta = torch.zeros_like(base)
        delta[:,site] = d.states[source][:,site] - base[:,site]
        p = partial_response(r,base+delta,source,mid,len(clean)+len(history)-1)
        z = (d.states[mid]-r.states[mid])[0,site:]
        y = (d.states[last]-r.states[last])[0,site:]
        def top(change):
            h = r.states[last][0,-1:] + change
            return int(model.lm_head(model.model.norm(h)).argmax())
        pred = secant_transport(p,z[-1:],y[-1:],weights["secant_transport"][0])
        offset, orthogonal, _ = prefix_secants(p,z,y,len(z)-1)
        mixture = weights["partial_donor_mix"]
        def suffix(x):
            return r.forward(source,x)[0,-1:]
        midpoint = torch.autograd.functional.jvp(suffix,base+.5*delta,delta)[1]
        # Evaluation oracle is separate from the candidate calculations above.
        gold = suffix(base+delta)-r.states[last][0,-1:]
        return {"exact":top(gold), "secant_transport":top(pred),
                "prefix_secants":top(offset+weights["prefix_secants"][0]*orthogonal),
                "partial_donor_mix":top(mixture[0]*p+mixture[1]*y[-1:]),
                "midpoint_derivative":top(midpoint)}

    generations = {}
    for method in METHODS:
        tokens = []
        for _ in range(8):
            token = features(tuple(tokens))[method]
            tokens.append(token)
            if token in stops:
                break
        generations[method] = tokens
    return generations


def main():
    parser = argparse.ArgumentParser()
    for key in ("inputs","manifest","out"):
        parser.add_argument(key,type=Path)
    args = parser.parse_args()
    weights = json.loads((args.out/"weights.json").read_text())
    assert digest(args.manifest) == weights["manifest_sha256"]
    assert digest(args.inputs/"train-v1.1.json") == SQUAD_SHA
    articles = {a["title"]:a for a in json.loads((args.inputs/"train-v1.1.json").read_text())["data"]}
    rows = [r for r in json.loads(args.manifest.read_text()) if r["split"]=="test"]
    torch.backends.cuda.matmul.allow_tf32=False
    for name in WEIGHTS:
        observed = {r["qid"]:r for r in json.loads((args.out/(name+"-rows.json")).read_text())}
        path = args.inputs/name
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,attn_implementation="eager",
                local_files_only=True).eval().requires_grad_(False).cuda()
        tok=AutoTokenizer.from_pretrained(path)
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [eos])
        results=[]
        for row in rows:
            para=articles[row["article"]]["paragraphs"][row["paragraph"]]
            question=next(q["question"] for q in para["qas"] if q["id"]==row["qid"])
            ids=get_ids(tok,row,para["context"],question)
            answers=rollout(model,ids,stops,weights["weights"])
            answers["no_change"]=greedy(model,ids[0],stops)
            ref=observed[row["qid"]]
            if (answers["exact"]!=answers["no_change"]) != ref["actual_changed"]:
                raise ValueError("Independent autoregressive switch check failed")
            for k,v in answers.items():
                if k=="exact":
                    assert v[0]==ref["first_token"]
                else:
                    assert v[0]==ref["methods"][k]["first_token"]
                    assert (v!=answers["no_change"])==ref["methods"][k]["changed"]
            texts={k:normalize(tok.decode(v,skip_special_tokens=True)) for k,v in answers.items()}
            results.append(dict(qid=row["qid"], article=row["article"],
                clean_correct=texts["no_change"]==normalize(row["original"]),
                exact_matches_replacement=texts["exact"]==normalize(row["replacement"]),
                exact_ended=answers["exact"][-1] in stops,
                actual_text_change=texts["exact"]!=texts["no_change"],
                text_agreement={k:v==texts["exact"] for k,v in texts.items() if k!="exact"},
                token_agreement={k:v==answers["exact"] for k,v in answers.items() if k!="exact"},
                tokens=answers))
            if len(results)%20==0:
                print(name, "full generations", len(results), flush=True)
        (args.out/(name+"-decode.json")).write_text(json.dumps(results,indent=2)+"\n")
        del model


if __name__=="__main__":
    main()
