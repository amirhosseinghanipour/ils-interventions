"""Measure fully charged patch scheduling on matched early and late answer edits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from natural import save_json
from qa_subset import ALL, SubsetProbe, fresh_cases, prompt
from qa_timing import divergent_context, generate, summarize


def numeric_edit(answer, early):
    if not re.fullmatch(r"[1-9][0-9]{1,3}", answer):
        return None
    if early:
        digit = (int(answer[0])-1+3)%9+1
        return str(digit)+answer[1:]
    return answer[:-1]+str((int(answer[-1])+3)%10)


def name_edit(answer, early):
    if not re.fullmatch(r"[A-Z][a-z]{2,14} [A-Z][a-z]{2,14}", answer):
        return None
    first,last = answer.split()
    word = "Samuel" if "Samuel" not in (first,last) else "Joseph"
    return f"{word} {last}" if early else f"{first} {word}"


def paired_cases(model, articles, excluded, n, seed):
    pairs = []
    for kind,edit in (("numeric",numeric_edit),("name",name_edit)):
        candidates = fresh_cases(articles,excluded,model,len(articles),seed=seed,
                                 edit_answer=lambda answer:edit(answer,True))
        accepted = 0
        for early in candidates:
            if early["common_prefix"] or early["differing_tokens"] is None:
                continue
            article = articles[early["article"]]
            for para in article["paragraphs"]:
                questions = {q["id"]:q["question"] for q in para["qas"]}
                if set(early["qids"]) <= questions.keys():
                    break
            else:
                raise ValueError("Source paragraph missing for existing QA IDs")
            source = early["source"]
            replacement = edit(source,False)
            context = para["context"]
            source_pattern = r"(?<!\w)" + re.escape(source) + r"(?!\w)"
            if len(re.findall(source_pattern,context)) != 1:
                continue
            roundtrip = r"(?<!\w)" + re.escape(replacement) + r"(?!\w)"
            protected = [context] + [questions[q] for q in early["qids"]] + early["gold"][1]
            if any(re.search(roundtrip,text,re.IGNORECASE) for text in protected):
                continue
            original = prompt(context,questions[early["qids"][0]])
            base,_ = early["ids"]
            edited = re.sub(source_pattern,lambda _:replacement,context)
            donor = [model.ids(prompt(edited,questions[q])) for q in early["qids"]]
            if max(map(len,donor)) > 236:
                continue
            target = model.ids(original+replacement)
            if len(target)<=len(base[0]) or not np.array_equal(target[:len(base[0])],base[0]):
                continue
            generated = target[len(base[0]):]
            original_tokens = model.ids(original+source)[len(base[0]):]
            common = 0
            for a,b in zip(original_tokens,generated):
                if a != b:
                    break
                common += 1
            if not 0<common<min(len(original_tokens),len(generated)):
                continue
            late = dict(early,replacement=replacement,ids=(base,donor),gold_step=common)
            pairs.append((dict(early,kind=kind,edit_position="early",gold_step=0),
                          dict(late,kind=kind,edit_position="late")))
            excluded.add(early["article"])
            accepted += 1
            if accepted==n:
                break
        if accepted!=n:
            raise ValueError(f"Found {accepted} matched {kind} pairs, needed {n}")
    return pairs


class Meter:
    """Count synchronized model forwards without changing the underlying native probe."""
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.model.forward(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model,name)


def online(model, base, edited):
    """Reuse observed tokens before the first disagreement and capture only as needed."""
    tokens = []
    active = "scan"
    patch_step = None
    count = 0
    for step in range(min(14,256-max(len(base),len(edited)))):
        prefix = np.asarray(tokens,np.int32)
        recipient = np.concatenate((base,prefix))
        if active == "scan" and step == 6:
            active = "none"
        if active == "scan":
            token = model.forward(recipient)
            donor = np.concatenate((edited,prefix))
            candidate = model.forward(donor,ALL,capture=True)
            if candidate!=token:
                patch_step = step
                active = "rolling" if step==0 else "fixed"
                token = model.forward(recipient,ALL)
                count += 1
        elif active == "fixed":
            token = model.forward(recipient,ALL,position=len(base)-1+patch_step)
        elif active == "rolling":
            model.forward(np.concatenate((edited,prefix)),ALL,capture=True)
            token = model.forward(recipient,ALL)
            count += 1
        else:
            token = model.forward(recipient)
        if model.lib.is_eog(token):
            break
        tokens.append(token)
        if "\n" in model.text(tokens):
            break
    return model.text(tokens).split("\n")[0].strip(),count


def record_case(model, case, rng):
    outputs, counts, effort, milliseconds, steps = {}, {}, {}, {}, []
    for base, donor in zip(*case["ids"]):
        def scanned():
            step = divergent_context(meter,base,donor)[-2]
            steps.append(step)
            timing = "fixed" if step else "rolling"
            return generate(meter,base,donor,ALL,timing,step)
        arms = {
            "clean": lambda:generate(meter,base,donor,(),"none"),
            "text": lambda:generate(meter,donor,donor,(),"none"),
            "full_prompt": lambda:generate(meter,base,donor,ALL,"fixed"),
            "full_rolling": lambda:generate(meter,base,donor,ALL,"rolling"),
            "scanned": scanned,
            "online": lambda:online(meter,base,donor),
        }
        values = {}
        for name in rng.permutation(list(arms)):
            meter = Meter(model)
            start = perf_counter()
            text,count = arms[name]()
            values[name] = (text,count,meter.calls,(perf_counter()-start)*1000)
        for name in arms:
            text,count,calls,ms = values[name]
            outputs.setdefault(name,[]).append(text)
            counts.setdefault(name,[]).append(count)
            effort.setdefault(name,[]).append(calls)
            milliseconds.setdefault(name,[]).append(ms)
    keys = ("article","kind","edit_position","qids","source","replacement","gold","gold_step")
    return dict(**{key:case[key] for key in keys},outputs=outputs,counts=counts,forwards=effort,
                milliseconds=milliseconds,interaction=[],steps=steps)


def report(pairs):
    comparisons = [("online","full_rolling"),("scanned","full_rolling"),("online","scanned"),
                   ("single_at_disagreement","full_prompt")]
    cells = {}
    for position,which in (("early",0),("late",1)):
        for kind in ("numeric","name"):
            rows = [pair[which] for pair in pairs if pair[which]["kind"]==kind]
            if not rows:
                continue
            # Scanned generation uses one position on delayed queries, and
            # at a zero step the single-position arm is the prompt arm.
            replayed = []
            for row in rows:
                selected = [row["outputs"]["scanned" if step else "full_prompt"][query]
                            for query,step in enumerate(row["steps"])]
                outputs = dict(row["outputs"],single_at_disagreement=selected)
                replayed.append(dict(row,outputs=outputs))
            scored = summarize(replayed,comparisons)
            scored["cost"] = {
                metric:{name:sum(sum(r[metric][name]) for r in rows)
                        for name in ("scanned","online","full_rolling")}
                for metric in ("counts","forwards","milliseconds")
            }
            cells[f"{kind}_{position}"] = scored
    rows = [row for pair in pairs for row in pair]
    baseline = "full_rolling"
    methods = ("scanned","online",baseline)
    sample = np.random.default_rng(57640).integers(len(pairs),size=(5000,len(pairs)))
    total_cost, comparisons = {}, {}
    for metric in ("counts","forwards","milliseconds"):
        cost = {name:np.array([sum(sum(r[metric][name]) for r in pair) for pair in pairs])
                for name in methods}
        total_cost[metric] = {name:float(values.sum()) for name,values in cost.items()}
        reference = cost[baseline]
        comparisons[metric] = {}
        for name in methods[:2]:
            values = cost[name]
            ratios = values[sample].sum(1)/reference[sample].sum(1)
            comparisons[metric][name] = dict(
                ratio=float(values.sum()/reference.sum()),
                article_bootstrap95=np.quantile(ratios,[.025,.975]).tolist(),
            )
    same_output = {
        name:sum(answer==reference
                 for row in rows
                 for answer,reference in zip(row["outputs"][name],row["outputs"][baseline]))
        for name in methods[:2]
    }
    return dict(articles=len(pairs),cells=cells,patched_position_forwards_wall_ms=total_cost,
                cost_relative_to_every_step=comparisons,
                exact_query_answer_agreement_with_every_step=same_output,
                query_generations=2*len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","dataset","original_dev","pilot_rows","out"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--exclude",action="append",type=Path,default=[])
    parser.add_argument("--per-kind",type=int,default=10)
    parser.add_argument("--seed",type=int,default=16611)
    args = parser.parse_args()
    data = args.dataset.read_bytes()
    articles = {a["title"]:a for a in json.loads(data)["data"]}
    excluded = {a["title"] for a in json.loads(args.original_dev.read_text())["data"]}
    excluded.update(row["article"] for row in json.loads(args.pilot_rows.read_text()))
    for path in args.exclude:
        previous = json.loads(path.read_text())
        excluded.update(previous["dev"]+previous["test"])
    args.out.mkdir(parents=True,exist_ok=True)
    model = SubsetProbe(args.library,args.model)
    pairs = paired_cases(model,articles,excluded,args.per_kind,args.seed)
    sample = [a["article"] for a,_ in pairs]
    save_json(args.out/"provenance.json",dict(dataset_sha256=hashlib.sha256(data).hexdigest(),
              seed=args.seed,dev=[],test=sample,model=args.model.name))
    output = args.out/"rows.json"
    records = json.loads(output.read_text()) if output.exists() else []
    assert [pair[0]["article"] for pair in records]==sample[:len(records)]
    # Exclude one resident-process warm-up from all arm costs.
    model.forward(pairs[0][0]["ids"][0][0])
    for i,pair in enumerate(pairs[len(records):],len(records)):
        rng = np.random.default_rng(args.seed+i)
        records.append([record_case(model,case,rng) for case in pair])
        save_json(output,records)
        print(i+1,pair[0]["kind"],pair[0]["article"],flush=True)
    save_json(args.out/"result.json",report(records))


if __name__=="__main__":
    main()
