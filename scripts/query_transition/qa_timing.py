"""Frozen comparison of donor-head timing and generation-aware subset selection."""
import argparse
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import re

import numpy as np

from natural import save_json
from qa_pool_control import fraction
from qa_subset import ALL, ALL_ATTN, SubsetProbe, fresh_cases, normalized


def later_digit(answer):
    if not re.fullmatch(r"[1-9][0-9]{1,3}", answer):
        return None
    digit = int(answer[-1])
    return answer[:-1] + str(digit + (2 if digit < 8 else -2))


def person_name(answer):
    # Limit automatic lexical edits to capitalized, two-word spans.
    if not re.fullmatch(r"[A-Z][a-z]{2,14} [A-Z][a-z]{2,14}", answer):
        return None
    first, last = answer.split()
    substitute = "Martin" if first != "Martin" else "Thomas"
    return f"{substitute} {last}"


def draw(articles, excluded, model, count, seed):
    rows = []
    for offset, (kind, edit) in enumerate((("numeric", later_digit), ("name", person_name))):
        subset = fresh_cases(articles, excluded, model, count, seed=seed+offset, edit_answer=edit)
        if len(subset) != count:
            raise ValueError(f"Only {len(subset)} usable {kind} articles")
        for row in subset:
            row["kind"] = kind
            excluded.add(row["article"])
        rows.extend(subset)
    return rows


def divergent_context(model, base, donor, max_steps=6):
    """Use the first donor/original top-choice disagreement along a clean greedy prefix."""
    prefix = np.empty(0, dtype=np.int32)
    fallback = None
    for _ in range(max_steps):
        recipient = np.concatenate((base, prefix))
        source = np.concatenate((donor, prefix))
        clean_token = model.forward(recipient)
        clean = model.distribution()
        donor_token = model.forward(source)
        donor_dist = model.distribution()
        gain = float(np.exp(donor_dist) @ (donor_dist-clean))
        record = (recipient, source, clean, donor_dist, len(prefix), gain)
        if fallback is None or gain > fallback[-1]:
            fallback = record
        if clean_token != donor_token:
            return record
        if model.lib.is_eog(clean_token):
            break
        prefix = np.append(prefix, np.int32(clean_token))
        if "\n" in model.text(prefix):
            break
    return fallback


def distribution_scores(model, context, panels):
    base, donor, clean, donor_dist, _, total = context
    clean_p, donor_p = np.exp(clean), np.exp(donor_dist)
    model.forward(donor, ALL, capture=True)
    out = []
    for sites in panels:
        model.forward(base, sites)
        patch = model.distribution()
        delta = patch-clean
        out.append((float(donor_p @ delta), float(-clean_p @ delta)))
    return np.asarray(out)


def discover(model, dev, rng):
    groups = []
    for layer in ALL_ATTN:
        heads = rng.permutation(16).tolist()
        groups.extend(tuple((layer,h) for h in heads[i:i+4]) for i in range(0,16,4))
    contexts = [[divergent_context(model,b,d) for b,d in zip(*row["ids"])] for row in dev]
    # Swap query order randomly, then infer which query changed more from distributions.
    for pair in contexts:
        rng.shuffle(pair)
    changed = np.array([np.argmax([q[-1] for q in pair]) for pair in contexts])
    index = np.arange(len(dev))

    def score(panels):
        values = np.array([[distribution_scores(model,query,panels) for query in pair] for pair in contexts])
        selective = (values[index,changed,:,0] - 2*values[index,1-changed,:,1]).mean(0)
        pooled = values[:,:,:,0].mean(axis=(0,1))
        return selective,pooled

    singles, pooled = score(groups)
    top = np.argsort(-singles)[:6].tolist()
    agnostic_top = np.argsort(-pooled)[:6].tolist()
    pairs = sorted({tuple(sorted(pair)) for ranking in (top,agnostic_top) for pair in combinations(ranking,2)})
    values, pooled_pairs = score([groups[i]+groups[j] for i,j in pairs])
    selected = int(np.argmax(values))
    i,j = pairs[selected]
    greedy = max((n for n,pair in enumerate(pairs) if top[0] in pair), key=values.__getitem__)
    agnostic = int(np.argmax(pooled_pairs))
    def joined(pair):
        a,b = pair
        return groups[a]+groups[b]
    random = [tuple(ALL[x] for x in rng.choice(len(ALL),8,replace=False)) for _ in range(3)]
    return dict(group_a=groups[i],group_b=groups[j],greedy=joined(pairs[greedy]),
                query_agnostic=joined(pairs[agnostic]),random=random,selected_score=float(values[selected]),
                articles=[row["article"] for row in dev])


def generate(model, base, donor, sites, timing, selected_step=0):
    """Generate greedily. Simulations share a donor conditioned on the generated prefix."""
    tokens, interventions = [], 0
    for t in range(min(14,256-max(len(base),len(donor)))):
        prefix = np.asarray(tokens,np.int32)
        current = np.concatenate((base,prefix))
        if timing == "rolling":
            patch_step = t
        elif timing == "none" or t < selected_step:
            patch_step = None
        else:
            patch_step = selected_step
        if patch_step is None:
            token = model.forward(current)
        else:
            if t == patch_step:
                model.forward(np.concatenate((donor,prefix)),ALL,capture=True)
                interventions += 1
            token = model.forward(current,sites,position=len(base)-1+patch_step)
        if model.lib.is_eog(token):
            break
        tokens.append(token)
        if "\n" in model.text(tokens):
            break
    return model.text(tokens).split("\n")[0].strip(),interventions


def measure(model, row, frozen):
    panels = {}
    if frozen is not None:
        a,b = (tuple(map(tuple,frozen[key])) for key in ("group_a","group_b"))
        panels = dict(subset=a+b, group_a=a, group_b=b,
                      greedy=tuple(map(tuple,frozen["greedy"])),
                      query_agnostic=tuple(map(tuple,frozen["query_agnostic"])))
        panels.update({f"random{i}":tuple(map(tuple,sites)) for i,sites in enumerate(frozen["random"])})
    base, donor = row.pop("ids")
    outputs, counts, steps, interactions = {}, {}, [], []
    for recipient, source in zip(base,donor):
        context = divergent_context(model,recipient,source)
        step = context[-2]
        steps.append(step)
        if frozen is not None:
            score = distribution_scores(model,context,[(),a,b,a+b])
            interactions.append((score[3] - score[1] - score[2] + score[0]).tolist())
        calls = dict(clean=(recipient,source,(),"none",0),text=(source,source,(),"none",0),
                     full_prompt=(recipient,source,ALL,"fixed",0),
                     full_delayed=(recipient,source,ALL,"fixed",step),
                     full_rolling=(recipient,source,ALL,"rolling",0))
        calls.update({name:(recipient,source,sites,"fixed",step) for name,sites in panels.items()})
        if frozen is not None:
            calls["subset_rolling"] = (recipient,source,a+b,"rolling",0)
        for name, call in calls.items():
            answer,n = generate(model,*call)
            outputs.setdefault(name,[]).append(answer)
            counts.setdefault(name,[]).append(n)
    return dict(article=row["article"], kind=row["kind"], qids=row["qids"], source=row["source"],
                replacement=row["replacement"], gold=row["gold"], outputs=outputs, counts=counts,
                steps=steps, interaction=interactions)


def summarize(rows):
    def match(text, gold):
        return normalized(text) in {normalized(a) for a in gold}

    eligible = [r for r in rows if all(match(a,g) for a,g in zip(r["outputs"]["clean"],r["gold"]))]
    result, outcomes = {}, {}
    for arm in rows[0]["outputs"]:
        moved = [match(r["outputs"][arm][0],[r["replacement"]]) for r in eligible]
        kept = [match(r["outputs"][arm][1],r["gold"][1]) for r in eligible]
        y = np.asarray(moved,dtype=bool)&np.asarray(kept,dtype=bool)
        outcomes[arm] = y.astype(float)
        result[arm] = dict(joint=fraction(int(y.sum()),len(eligible)), all_articles=fraction(int(y.sum()),len(rows)),
                           switch=fraction(sum(moved),len(eligible)), preservation=fraction(sum(kept),len(eligible)),
                           by_kind={kind:fraction(sum(bool(ok) for r,ok in zip(eligible,y) if r["kind"]==kind),
                                                  sum(r["kind"]==kind for r in eligible)) for kind in ("numeric","name")})
    rng = np.random.default_rng(71254)
    def interval(x):
        return dict(mean=float(np.mean(x)), bootstrap95=np.quantile(
            rng.choice(x,(5000,len(x))).mean(1),[.025,.975]).tolist())
    result["paired"] = {}
    if eligible:
        for arm, control in [("full_delayed","full_prompt"),("full_rolling","full_prompt"),
                             ("subset","greedy"),("subset","query_agnostic"),("subset","random_mean")]:
            if arm not in outcomes:
                continue
            baseline = (np.mean([y for name,y in outcomes.items() if name.startswith("random")],axis=0)
                        if control=="random_mean" else outcomes[control])
            delta = outcomes[arm]-baseline
            comparison = interval(delta)
            if control != "random_mean":
                wins, losses = int(sum(delta>0)), int(sum(delta<0))
                discordant = wins + losses
                tail = sum(comb(discordant,i) for i in range(min(wins,losses)+1))
                comparison.update(wins=wins, losses=losses,
                                  exact_two_sided_p=min(1.,2*tail/2**discordant))
            result["paired"][f"{arm}-minus-{control}"] = comparison
    if rows[0]["interaction"]:
        result["differential_interaction_donor_gain"] = interval(
            np.array([r["interaction"][0][0]-r["interaction"][1][0] for r in rows]))
    result["timing"] = dict(step_histogram=np.bincount([n for r in rows for n in r["steps"]]).tolist(),
                            relevant_delayed=sum(r["steps"][0]>0 for r in rows))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","dataset","original_dev","pilot_rows","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--timing-only",action="store_true")
    p.add_argument("--exclude-rows",type=Path)
    p.add_argument("--test-seed",type=int,default=981344)
    p.add_argument("--dev-per-kind",type=int,default=5)
    p.add_argument("--test-per-kind",type=int,default=16)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    data = args.dataset.read_bytes()
    articles = {r["title"]:r for r in json.loads(data)["data"]}
    excluded = {r["title"] for r in json.loads(args.original_dev.read_text())["data"]}
    excluded |= {r["article"] for r in json.loads(args.pilot_rows.read_text())}
    if args.exclude_rows:
        prior = json.loads(args.exclude_rows.read_text())
        excluded |= set(prior["dev"]) | set(prior["test"])
    model = SubsetProbe(args.library,args.model)
    frozen, development = None, []
    if not args.timing_only:
        dev = draw(articles,excluded,model,args.dev_per_kind,76113)
        development = [r["article"] for r in dev]
        frozen_file = args.out/"frozen.json"
        if frozen_file.exists():
            frozen = json.loads(frozen_file.read_text())
            assert frozen["articles"] == development
        else:
            frozen = discover(model,dev,np.random.default_rng(94842))
            save_json(frozen_file,frozen)
        print("Frozen",flush=True)
    test = draw(articles,excluded,model,args.test_per_kind,args.test_seed)
    save_json(args.out/"provenance.json", dict(dataset_sha256=hashlib.sha256(data).hexdigest(),
                  dev=development,test=[r["article"] for r in test]))
    output = args.out/"rows.json"
    rows = json.loads(output.read_text()) if output.exists() else []
    assert [r["qids"] for r in rows] == [r["qids"] for r in test[:len(rows)]]
    for row in test[len(rows):]:
        rows.append(measure(model,row,frozen))
        save_json(output,rows)
        print(len(rows),row["kind"],row["article"],flush=True)
    save_json(args.out/"result.json",summarize(rows))


if __name__ == "__main__":
    main()
