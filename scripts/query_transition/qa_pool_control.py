"""Evaluate donor-passage head/MLP substitutions on the archived paired SQuAD sample."""
import argparse
from collections import Counter
import csv
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from natural import CFG, chat, save_json

HERE = Path(__file__).resolve().parent
ALL_ATTN = tuple(range(3, 32, 4))
ALL_MLP = tuple(range(32))
# The two central blocks were selected in previous synthetic experiments.
POOLS = {
    "head19_11": ((19,), (), 11),
    "heads16": ((19,), (), -1),
    "heads32": ((19, 23), (), -1),
    "heads64": ((15, 19, 23, 27), (), -1),
    "heads128": (ALL_ATTN, (), -1),
    "heads128_mlp8": (ALL_ATTN, ALL_ATTN, -1),
    "heads128_mlp32": (ALL_ATTN, ALL_MLP, -1),
}


class Probe:
    def __init__(self, library, model, vocab_only=False):
        self.lib = ct.CDLL(str(library.resolve()))
        arr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.lib.initialize.argtypes = [ct.c_char_p, ct.c_int]
        self.lib.tokenize.argtypes = [ct.c_char_p, arr, ct.c_int]
        self.lib.pool_forward.argtypes = [arr, ct.c_int, ct.c_uint32, ct.c_uint32, ct.c_int, ct.c_int]
        self.lib.initialize(str(model).encode(), int(vocab_only))

    def ids(self, text):
        buf = np.empty(len(text.encode()) + 1, np.int32)
        n = self.lib.tokenize(text.encode(), buf, len(buf))
        if n <= 0:
            raise ValueError("tokenizer failed")
        return buf[:n].copy()

    def run(self, ids, attn=(), mlp=(), head=-1, *, capture=False):
        def mask(layers):
            return sum(1 << layer for layer in layers)
        return self.lib.pool_forward(ids, len(ids), mask(attn), mask(mlp), int(capture), head)


def build_case(model, row, article, edit):
    paragraph = article["paragraphs"][row["paragraph"]]
    questions = {q["id"]: q for q in paragraph["qas"]}
    front, back = chat("FACT", "QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")
    source = paragraph["context"]
    assert edit["source"] in source and edit["replacement"] not in source
    changed = source.replace(edit["source"], edit["replacement"])
    relevant = int(edit["relevant"])
    assert relevant in (0, 1) and edit["intended_answer"] in changed
    untouched = questions[row["qids"][1 - relevant]]
    assert edit["source"].casefold() not in untouched["question"].casefold()
    assert all(edit["source"].casefold() not in a["text"].casefold() for a in untouched["answers"])
    ids, donor, truths = [], [], []
    for i, qid in enumerate(row["qids"]):
        question = questions[qid]["question"]
        def prompt(context):
            return front + "Answer the question using only the exact answer from the passage.\n" + context + "\n" + question + back
        ids.append(model.ids(prompt(source)))
        donor.append(model.ids(prompt(changed)))
        assert len(ids[-1]) == row["lengths"][i]
        for ans in questions[qid]["answers"]:
            joint = model.ids(prompt(source) + ans["text"].strip())
            np.testing.assert_array_equal(joint[:len(ids[-1])], ids[-1])
            assert joint[len(ids[-1])] == row["labels"][i]
        answer = edit["intended_answer"] if i == relevant else questions[qid]["answers"][0]["text"].strip()
        joint = model.ids(prompt(changed) + answer)
        np.testing.assert_array_equal(joint[:len(donor[-1])], donor[-1])
        truths.append(int(joint[len(donor[-1])]))
    assert truths[1 - relevant] == row["labels"][1 - relevant]
    assert len(set(truths + [row["labels"][relevant]])) == 3 and max(map(len, donor)) <= 256
    return ids, donor, truths, source.count(edit["source"])


def measure(model, row, article, edit):
    ids, donor, truths, count = build_case(model, row, article, edit)
    observed = {"text_edit": []}
    observed.update({key: [] for key in POOLS})
    clean = []
    for i, (base, changed) in enumerate(zip(ids, donor)):
        clean.append(model.run(base, ALL_ATTN, ALL_MLP, capture=True))
        # Same donor storage and write path as widest substitution.
        assert model.run(base, ALL_ATTN, ALL_MLP) == clean[-1]
        assert clean[-1] == row["clean"][i]["top_token"]
        observed["text_edit"].append(model.run(changed, ALL_ATTN, ALL_MLP, capture=True))
        for key, (attn, mlp, head) in POOLS.items():
            observed[key].append(model.run(base, attn, mlp, head))
    return dict(article=row["article"], paragraph=row["paragraph"], qids=row["qids"],
                relevant=int(edit["relevant"]), original_tokens=row["labels"], target_tokens=truths,
                clean=clean, top_tokens=observed, source_occurrences=count, lengths=row["lengths"],
                donor_lengths=list(map(len, donor)))


def fraction(k, n):
    if not n:
        return dict(successes=0, n=0, rate=None, wilson95=None)
    z = 1.959963984540054
    p = k/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return dict(successes=k, n=n, rate=p, wilson95=[max(0,center-radius), min(1,center+radius)])


def summarize(rows):
    def outcomes(row, key):
        i = row["relevant"]
        j = 1-i
        out = row["top_tokens"][key]
        old = row["clean"]
        gold = row["original_tokens"]
        return (old[i] == gold[i] and old[j] == gold[j], out[i] == row["target_tokens"][i],
                out[j] == old[j], old[j] == gold[j], old[i] == gold[i])
    result = {}
    for key in rows[0]["top_tokens"]:
        counts = Counter()
        for row in rows:
            both, moved, preserved, irr_ok, rel_ok = outcomes(row, key)
            changed = row["clean"][row["relevant"]] != row["target_tokens"][row["relevant"]]
            counts.update(n=1, both=both, rel_ok=rel_ok, irr_ok=irr_ok, switch=rel_ok and moved,
                          preserve=irr_ok and preserved, joint=both and moved and preserved,
                          any_clean_joint=changed and moved and preserved,
                          joint_in_text_positive=both and moved and preserved and all(outcomes(row,"text_edit")[:3]),
                          text_positive=all(outcomes(row,"text_edit")[:3]))
        def get(val, denom):
            return fraction(counts[val], counts[denom])
        result[key] = dict(strict_joint_all=get("joint","n"),
            conditional_joint_both_clean_correct=get("joint","both"),
            relevant_switch_from_clean_correct=get("switch","rel_ok"),
            irrelevant_preservation_from_clean_correct=get("preserve","irr_ok"),
            top1_joint_any_clean_correctness=get("any_clean_joint","n"),
            conditional_joint_clean_and_text_positive=get("joint_in_text_positive","text_positive"))
    return result


def main():
    if not __debug__:
        raise RuntimeError("Use Python with assertions enabled to check the donor and sham controls.")
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("library", "model", "dataset", "archive_rows", "out"):
        ap.add_argument(key, type=Path)
    ap.add_argument("--check-design", action="store_true")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    if not 1 <= args.limit <= 40:
        ap.error("--limit must be in [1, 40]")
    assert hashlib.sha256(args.dataset.read_bytes()).hexdigest() == CFG["sha256"]
    articles = {a["title"]:a for a in json.loads(args.dataset.read_text())["data"]}
    archived = json.loads(args.archive_rows.read_text())
    with (HERE / "qa_pool_edits.tsv").open() as f:
        edits = {r["article"]:r for r in csv.DictReader(f, delimiter="\t")}
    assert len(archived) == len(edits) == 40 and set(edits) == {r["article"] for r in archived}
    model = Probe(args.library, args.model, args.check_design)
    if args.check_design:
        for row in archived:
            print(row["article"], flush=True)
            build_case(model, row, articles[row["article"]], edits[row["article"]])
        return
    rows = []
    args.out.mkdir(parents=True, exist_ok=False)
    for row in archived[:args.limit]:
        record = measure(model, row, articles[row["article"]], edits[row["article"]])
        rows.append(record)
        save_json(args.out/"rows.json", rows)
        print(f"{len(rows)}/{args.limit} {row['article']}", flush=True)
    save_json(args.out/"result.json", summarize(rows))


if __name__ == "__main__":
    main()
