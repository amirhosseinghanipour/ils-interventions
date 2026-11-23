"""Held-out retrieval response collection and causal patching of a gated head."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np
from run import Model, F32, HERE, chat, design, save_json

CFG = json.loads((HERE / "followup.json").read_text())
BLOCKS = [19, 23, 27, 31]


def rows(model, split, **more):
    extra = dict(name_bank="Ray May Sue Ian Roy Kim Jill Joe Leo Ted Amy Eve".split(),
                 template_pair=("In the ledger, {name} holds {color}.", "Give the color assigned to {name}."))
    if split == "train":
        extra = dict(name_bank="Albert George Emma Clara Henry Lucy James Diana Frank Grace Peter Anna".split(),
                     template_pair=("{name} owns a {color} card.", "Which color belongs to {name}?"))
    return list(design(model, split, seed=CFG[split+"_seed"], count=CFG[split+"_pairs"], unique_facts=True, **extra, **more))


def parcels(model):
    cfg = json.loads((HERE / "transfer.json").read_text())
    def parcel_chat(facts, question):
        return chat(facts, question).replace("lowercase color word", "lowercase item word")
    return design(model, "parcel", seed=cfg["seed"], count=cfg["count"],
                  template_pair=("{name} packed the {color}.", "What did {name} pack?"),
                  value_bank=cfg["items"], name_bank=cfg["names"], unique_facts=True,
                  render_chat=parcel_chat)


def animals(model):
    cfg = json.loads((HERE / "prototypes.json").read_text())
    def animal_chat(facts, question):
        return chat(facts, question).replace("lowercase color word", "lowercase animal name")
    return design(model, "animal", seed=cfg["seed"], count=cfg["count"], template_pair=cfg["template"],
                  value_bank=cfg["animals"], name_bank=cfg["names"], unique_facts=True,
                  render_chat=animal_chat)


class HeadModel(Model):
    def __init__(self, *args):
        super().__init__(*args)
        self.lib.configure_head.argtypes = [ct.c_int, ct.c_int, F32, ct.c_void_p]
        self.buffer = np.empty((16,256), np.float32)
        self.head()

    def head(self, layer=-1, head=-1, replacement=None):
        self.replacement = replacement
        self.lib.configure_head(layer, head, self.buffer, None if replacement is None else replacement.ctypes.data)


def example(model, row):
    queries = [np.array(t, np.int32) for t in row["tokens"]]
    def evaluate(i, edit=None):
        return model.evaluate(queries[i], row["site"], edit, row["donor_token"], row["correct"][0])
    model.head()
    donor = evaluate(2)[0]
    source = evaluate(0)[0]
    assert np.array_equal(source, evaluate(1)[0])
    return evaluate, donor-source


def measure_heads(model, row, layers_heads, rescue):
    evaluate, delta = example(model, row)
    records = {}
    for block, heads in layers_heads.items():
        for side in [0,1]:
            model.head(block)
            _, _, clean = evaluate(side)
            clean_heads = model.buffer.copy()
            _, _, edited = evaluate(side, delta)
            edited_heads = model.buffer.copy()
            base = edited["donor_minus_original"] - clean["donor_minus_original"]
            for head in heads:
                model.head(block,head,clean_heads[head])
                restored = evaluate(side,delta)[2]
                key = f"{block}/{head}"
                result = records.setdefault(key, [])
                obs = {"base":base, "reduction":edited["donor_minus_original"]-restored["donor_minus_original"]}
                if rescue:
                    model.head(block,head,edited_heads[head])
                    injected = evaluate(side)[2]
                    obs["sufficiency"] = injected["donor_minus_original"]-clean["donor_minus_original"]
                    model.head(block,head,clean_heads[head])
                    assert evaluate(side)[2] == clean
                result.append(obs)
    model.head()
    return records


def screen(model, output):
    scores = [measure_heads(model,r,{b:range(16) for b in BLOCKS},False) for r in design(model,"dev")]
    means = {k:float(np.mean([r[k][0]["reduction"]-r[k][1]["reduction"] for r in scores])) for k in scores[0]}
    winner = max(means, key=means.get)
    save_json(output/"screen.json",dict(pair_measurements=scores,means=means,winner=winner))
    return map(int,winner.split("/"))


def main():
    parser = argparse.ArgumentParser()
    for k in ("library","model","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    parser.add_argument("--heads",action="store_true")
    task=parser.add_mutually_exclusive_group()
    task.add_argument("--parcels", action="store_true")
    task.add_argument("--animals", action="store_true")
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    model=HeadModel(args.library,args.model)
    if args.heads:
        block,winner=(23,0) if args.parcels else screen(model,args.output)
        heads=[winner]+[(winner+i)%16 for i in (1,5,9)]
        print("selected",block,heads,flush=True)
    prefixes=set()
    transfer=args.parcels or args.animals
    for split in (["confirm"] if transfer else ["train","confirm"]):
        info=[]; x=[]; y=[]; control=[]
        batch = animals(model) if args.animals else parcels(model) if args.parcels else rows(model,split)
        for row in batch:
            prefix=row["texts"][0].split("\n")[2]
            assert prefix not in prefixes
            prefixes.add(prefix)
            evaluate,delta=example(model,row)
            x.append(delta)
            seed = 31014 if args.animals else 31013 if args.parcels else CFG[split+"_seed"]
            rng=np.random.default_rng(seed*1000+row["index"])
            random=rng.standard_normal(model.width).astype(np.float32)
            random *= np.linalg.norm(delta)/np.linalg.norm(random)
            targets=[]; margins=[]; rand=[]; tops=[]; energies=[]
            for side in [0,1]:
                _,clean,mc=evaluate(side)
                _,edited,me=evaluate(side,delta)
                response=edited-clean
                assert np.max(np.abs(response[:row["site"]]))<=1e-6
                targets.append(response[-1])
                energies.append(np.linalg.norm(response,axis=1).tolist())
                margins.append(me["donor_minus_original"]-mc["donor_minus_original"])
                tops.append([mc["top_token"],me["top_token"]])
                if split=="confirm":
                    mr=evaluate(side,random)[2]
                    rand.append(mr["donor_minus_original"]-mc["donor_minus_original"])
            y.append(targets)
            info.append(dict(**row,shifts=margins,random_shifts=rand,top=tops,response_norms=energies))
            if split=="confirm" and args.heads:
                control.append(measure_heads(model,row,{block:heads},True))
        save_json(args.output/(split+".json"),info)
        np.savez(args.output/(split+".npz"),x=np.array(x),y=np.array(y))
        if control: save_json(args.output/"head-confirm.json",control)
        print(split,len(info),flush=True)


if __name__=="__main__":
    main()
