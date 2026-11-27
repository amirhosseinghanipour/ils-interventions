"""Freeze answer-free head subsets on development edits and test new SQuAD articles."""
import argparse
import csv
import ctypes as ct
import hashlib
import json
from itertools import combinations
from pathlib import Path
import re
import unicodedata

import numpy as np

from qa_pool_control import ALL_ATTN, CFG, HERE, Probe, build_case, fraction
from natural import chat, save_json


ALL = tuple((layer, head) for layer in ALL_ATTN for head in range(16))
FRONT, BACK = chat("FACT", "QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")


def prompt(context, question):
    return FRONT + "Answer the question using only the exact answer from the passage.\n" + context + "\n" + question + BACK


class SubsetProbe(Probe):
    def __init__(self, library, model):
        super().__init__(library, model)
        arr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        mask = np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
        self.lib.subset_forward.argtypes = [arr, ct.c_int, mask, ct.c_int, ct.c_int]
        self.lib.last_logits.restype = ct.POINTER(ct.c_float)
        self.lib.detokenize.argtypes = [arr, ct.c_int, ct.c_char_p, ct.c_int]
        self.lib.is_eog.argtypes = [ct.c_int]
        self.lib.is_eog.restype = ct.c_bool

    def forward(self, ids, sites=(), capture=False, position=None):
        mask = np.zeros(32, np.uint16)
        for layer, head in sites:
            mask[layer] |= 1 << head
        pos = len(ids)-1 if position is None else position
        return self.lib.subset_forward(ids, len(ids), mask, capture, pos)

    def distribution(self):
        n = self.lib.vocabulary_size()
        logits = np.ctypeslib.as_array(self.lib.last_logits(), (n,)).astype(np.float64)
        z = logits - logits.max()
        return z - np.log(np.exp(z).sum())

    def answer(self, ids, sites=(), new_tokens=18):
        generated = []
        for _ in range(min(new_tokens, 256-len(ids))):
            current = np.concatenate((ids, np.asarray(generated, np.int32)))
            token = self.forward(current, sites, position=len(ids)-1)
            if self.lib.is_eog(token):
                break
            generated.append(token)
            if "\n" in self.text(generated):
                break
        return self.text(generated).split("\n")[0].strip()

    def text(self, tokens):
        out = ct.create_string_buffer(1024)
        n = self.lib.detokenize(np.asarray(tokens, np.int32), len(tokens), out, len(out))
        if n < 0:
            raise ValueError("detokenization buffer exceeded")
        return out.raw[:n].decode(errors="replace")


def normalized(text):
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.casefold())
    return " ".join(x for x in text.split() if x not in ("a", "an", "the"))


def truth(question):
    return [answer["text"].strip() for answer in question["answers"]]


def annotated(archive, articles, model):
    edits = {r["article"]:r for r in csv.DictReader((HERE/"qa_pool_edits.tsv").open(), delimiter="\t")}
    for row in sorted(archive, key=lambda x:x["article"]):
        title = row["article"]
        edit = edits[title]
        base, donor, _, _ = build_case(model, row, articles[title], edit)
        # Intentionally assign query order without relevance labels during discovery.
        order = np.random.default_rng(int(hashlib.sha256(title.encode()).hexdigest()[:8], 16)).permutation(2)
        yield [base[i] for i in order], [donor[i] for i in order]


def score_masks(model, cases, panels):
    result = np.zeros((len(cases), len(panels), 2))
    for row, (base, donor) in enumerate(cases):
        for i in range(2):
            model.forward(base[i])
            logp = model.distribution()
            p = np.exp(logp)
            model.forward(donor[i], ALL, capture=True)
            for idx, panel in enumerate(panels):
                model.forward(base[i], panel)
                result[row, idx, i] = p @ (logp - model.distribution())
    return result


def discover(model, cases, rng):
    # Start with 4-head groups. Select a reproducible 8-head pair by selective
    # distribution change. Score the larger-divergence query in either order.
    groups = []
    for layer in ALL_ATTN:
        order = rng.permutation(16)
        groups.extend(tuple((layer, int(h)) for h in order[j:j+4]) for j in range(0,16,4))
    blocks = [tuple(site for site in ALL if site[0]==layer) for layer in ALL_ATTN]
    singles = score_masks(model, cases, groups + blocks)
    selective = lambda x: np.mean(x.max(axis=-1) - 2*x.min(axis=-1), axis=0)
    top = np.argsort(-selective(singles[:, :len(groups)]))[:8]
    best_block = blocks[int(np.argmax(selective(singles[:, len(groups):])))]
    pair_indices = list(combinations(top, 2))
    pairs = [groups[i] + groups[j] for i,j in pair_indices]
    measured = score_masks(model, cases, pairs)
    chosen = int(np.argmax(selective(measured)))
    control = int(np.argmax(measured.mean(axis=(0, 2))))
    greedy_choices = [i for i, pair in enumerate(pair_indices) if top[0] in pair]
    greedy = max(greedy_choices, key=selective(measured).__getitem__)
    left, right = pair_indices[chosen]
    return dict(group_a=groups[left], group_b=groups[right], query_agnostic=pairs[control], greedy=pairs[greedy],
                single_group=groups[int(top[0])], best_block=best_block, development_n=len(cases),
                selective_dev_score=float(selective(measured)[chosen]))


def fresh_cases(articles, excluded, model, limit, *, seed=242911, edit_answer=None):
    """Sample distinct articles with a unique literal answer span and a deterministic edit."""
    rng = np.random.default_rng(seed)
    candidates = sorted(set(articles) - excluded)
    kept = []
    for title in rng.permutation(candidates):
        found = False
        for para in articles[title]["paragraphs"]:
            source = para["context"]
            for relevant in para["qas"]:
                spans = set(truth(relevant))
                if len(spans) != 1:
                    continue
                span = next(iter(spans))
                if edit_answer is None:
                    if not re.fullmatch(r"[1-9][0-9]{0,3}", span):
                        continue
                    number = int(span)
                    altered = str(number + (1 if number % 10 < 9 else -1))
                else:
                    altered = edit_answer(span)
                    if altered is None:
                        continue
                match = list(re.finditer(r"(?<!\w)" + re.escape(span) + r"(?!\w)", source))
                if len(match) != 1:
                    continue
                if re.search(r"(?<!\w)"+re.escape(altered)+r"(?!\w)", source, re.IGNORECASE):
                    continue
                edited = source[:match[0].start()] + altered + source[match[0].end():]
                for irrelevant in para["qas"]:
                    if irrelevant["id"] == relevant["id"]:
                        continue
                    other = truth(irrelevant)
                    if any(span in answer or altered in answer for answer in other):
                        continue
                    if re.search(r"\b(?:"+re.escape(span)+"|"+re.escape(altered)+r")\b",
                                 relevant["question"] + " " + irrelevant["question"], re.IGNORECASE):
                        continue
                    # Avoid paired arithmetic/numeric questions that may share the edited fact.
                    if re.search(r"\b(?:how many|how much|when|year|percent|number)\b", irrelevant["question"], re.I):
                        continue
                    questions = [relevant, irrelevant]
                    texts = [[prompt(context,q["question"]) for q in questions] for context in (source,edited)]
                    ids = [[model.ids(t) for t in pair] for pair in texts]
                    if max(len(x) for pair in ids for x in pair) > 236:
                        continue
                    label_ids = [model.ids(texts[0][0] + ans) for ans in (span,altered)]
                    if any(len(x)<=len(ids[0][0]) or not np.array_equal(x[:len(ids[0][0])],ids[0][0]) for x in label_ids):
                        continue
                    endings = [x[len(ids[0][0]):] for x in label_ids]
                    tokens = [int(x[0]) for x in endings]
                    common = 0
                    for left,right in zip(*endings):
                        if left != right:
                            break
                        common += 1
                    differing = [int(x[common]) for x in endings] if all(len(x)>common for x in endings) else None
                    kept.append(dict(article=str(title), qids=[q["id"] for q in questions], source=span,
                                     replacement=altered, gold=[list(spans),other], ids=ids, tokens=tokens,
                                     common_prefix=endings[0][:common].tolist(), differing_tokens=differing))
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if len(kept) == limit:
            break
    return kept


def evaluate(model, cases, frozen):
    a = tuple(map(tuple, frozen["group_a"]))
    b = tuple(map(tuple, frozen["group_b"]))
    rng = np.random.default_rng(91389)
    panels = dict(subset=a+b, group_a=a, group_b=b,
                  query_agnostic=tuple(map(tuple, frozen["query_agnostic"])),
                  greedy=tuple(map(tuple, frozen["greedy"])),
                  single_group=tuple(map(tuple, frozen["single_group"])), block19=tuple(x for x in ALL if x[0]==19),
                  full=ALL, head19_11=((19,11),),
                  best_block=tuple(map(tuple, frozen["best_block"])))
    for i in range(5):
        panels[f"random{i+1}"] = tuple(ALL[x] for x in rng.choice(len(ALL),8,replace=False))
    for row in cases:
        base, donor = row.pop("ids")
        row["answers"] = {"clean":[],"edited":[], **{key:[] for key in panels}}
        row["interactions"] = []
        for i in range(2):
            args = row["answers"]
            args["clean"].append(model.answer(base[i]))
            args["edited"].append(model.answer(donor[i]))
            model.forward(donor[i], ALL, capture=True)
            margins = []
            for sites in ((),a,b,a+b):
                model.forward(base[i], sites)
                logp = model.distribution()
                old,new = row["tokens"]
                margins.append(float(logp[new]-logp[old]))
            row["interactions"].append(margins[3]-margins[2]-margins[1]+margins[0])
            for key, sites in panels.items():
                args[key].append(model.answer(base[i], sites))
        print(row["article"], flush=True)
        yield row


def rolling_answer(model, base, donor, sites):
    """At each greedy decode step, use the donor conditional on the same generated prefix."""
    generated = []
    for _ in range(min(18,256-max(len(base),len(donor)))):
        prefix = np.asarray(generated,np.int32)
        model.forward(np.concatenate((donor,prefix)),ALL,capture=True)
        token = model.forward(np.concatenate((base,prefix)),sites)
        if model.lib.is_eog(token):
            break
        generated.append(token)
        if "\n" in model.text(generated):
            break
    return model.text(generated).split("\n")[0].strip()


def rolling_control(model, cases, source):
    frozen = json.loads((source/"frozen.json").read_text())
    previous = json.loads((source/"rows.json").read_text())
    assert [r["qids"] for r in previous] == [r["qids"] for r in cases]
    sites = tuple(map(tuple,frozen["group_a"] + frozen["group_b"]))
    for row, reference in zip(cases,previous):
        base, donor = row["ids"]
        answers = reference["answers"]
        answers["rolling_full"] = [rolling_answer(model,b,d,ALL) for b,d in zip(base,donor)]
        answers["rolling_subset"] = [rolling_answer(model,b,d,sites) for b,d in zip(base,donor)]
        differing = row["differing_tokens"]
        if differing:
            prefix = np.asarray(row["common_prefix"],np.int32)
            context = np.concatenate((base[0],prefix))
            model.forward(np.concatenate((donor[0],prefix)),ALL,capture=True)
            margins = []
            group_a = tuple(map(tuple,frozen["group_a"]))
            group_b = tuple(map(tuple,frozen["group_b"]))
            for selected in ((),group_a,group_b,sites,ALL):
                model.forward(context,selected)
                logp = model.distribution()
                margins.append(float(logp[differing[1]]-logp[differing[0]]))
            reference["shared_prefix_diagnostic"] = dict(prefix_tokens=len(prefix), margin=margins,
                                                         interaction=margins[3]-margins[2]-margins[1]+margins[0])
        print(row["article"],flush=True)
        yield reference


def summary(rows):
    def match(pred, gold):
        return normalized(pred) in {normalized(x) for x in gold}
    correct = [r for r in rows if all(match(p,g) for p,g in zip(r["answers"]["clean"],r["gold"]))]
    result, outcomes = {}, {}
    for method in rows[0]["answers"]:
        moved = [match(r["answers"][method][0],[r["replacement"]]) for r in correct]
        preserved = [match(r["answers"][method][1],r["gold"][1]) for r in correct]
        outcomes[method] = np.array([a and b for a,b in zip(moved,preserved)], dtype=float)
        joint = int(outcomes[method].sum())
        result[method] = dict(joint=fraction(joint,len(correct)), joint_all=fraction(joint,len(rows)),
                              switch=fraction(sum(moved),len(correct)), preservation=fraction(sum(preserved),len(correct)))
    if correct:
        sample = np.random.default_rng(91775).integers(len(correct),size=(5000,len(correct)))
        matches = outcomes["subset"]
        controls = {key: val for key,val in outcomes.items()
                    if key in ("greedy","query_agnostic","single_group","best_block","head19_11")}
        controls["random_mean"] = np.mean([val for key,val in outcomes.items() if key.startswith("random")],axis=0)
        result["paired_subset_gain"] = {}
        for name, baseline in controls.items():
            delta = matches - baseline
            result["paired_subset_gain"][name] = dict(
                mean=float(delta.mean()), bootstrap95=np.quantile(delta[sample].mean(1),[.025,.975]).tolist())
    differences = np.array([r["interactions"][0]-r["interactions"][1]
                            for r in rows if len(set(r["tokens"]))==2])
    draws = np.random.default_rng(31819).choice(differences,(5000,len(differences))).mean(1) if len(differences) else []
    result["differential_interaction"] = dict(
        n=len(differences), mean=float(differences.mean()) if len(differences) else None,
        bootstrap95=np.quantile(draws,[.025,.975]).tolist() if len(differences) else None)
    shared = [row["shared_prefix_diagnostic"] for row in rows if "shared_prefix_diagnostic" in row]
    if shared:
        interaction = np.array([r["interaction"] for r in shared])
        sample = np.random.default_rng(10321).choice(interaction,(5000,len(interaction))).mean(1)
        result["teacher_forced_relevant_interaction"] = dict(
            n=len(shared), mean=float(interaction.mean()), bootstrap95=np.quantile(sample,[.025,.975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("library","model","dataset","archive","out"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--fresh-dataset",type=Path)
    parser.add_argument("--rolling-from",type=Path)
    parser.add_argument("--dev",type=int,default=8)
    parser.add_argument("--test",type=int,default=24)
    args = parser.parse_args()
    data = args.dataset.read_bytes()
    assert hashlib.sha256(data).hexdigest() == CFG["sha256"]
    articles = {art["title"]:art for art in json.loads(data)["data"]}
    archive = json.loads(args.archive.read_text())
    args.out.mkdir(exist_ok=True,parents=True)
    model = SubsetProbe(args.library,args.model)
    frozen_path = args.out/"frozen.json"
    if args.rolling_from is not None:
        frozen = None
    elif frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
    else:
        dev = list(annotated(archive,articles,model))[:args.dev]
        frozen = discover(model,dev,np.random.default_rng(73218))
        save_json(frozen_path,frozen)
    fresh_dataset = args.fresh_dataset or args.dataset.with_name("train-v1.1.json")
    fresh = fresh_dataset.read_bytes()
    fresh_articles = {art["title"]:art for art in json.loads(fresh)["data"]}
    cases = fresh_cases(fresh_articles,set(articles),model,args.test)
    save_json(args.out/"fresh-provenance.json",dict(sha256=hashlib.sha256(fresh).hexdigest(),
                                                   articles=[case["article"] for case in cases]))
    if len(cases) != args.test:
        raise RuntimeError(f"Expected {args.test} eligible fresh cases, found {len(cases)}")
    smoke = cases[0]["ids"][0][0]
    native = model.run(smoke)
    assert model.forward(smoke) == native
    model.forward(smoke, ALL, capture=True)
    assert model.forward(smoke, ALL) == native
    measured = []
    results = rolling_control(model,cases,args.rolling_from) if args.rolling_from else evaluate(model,cases,frozen)
    for row in results:
        measured.append(row)
        save_json(args.out/"rows.json",measured)
    save_json(args.out/"result.json",summary(measured))


if __name__ == "__main__":
    main()
