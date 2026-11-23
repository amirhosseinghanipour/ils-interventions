"""Transfer the frozen synthetic-task interventions to human-authored question pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from collect import Tracer
from followup import HeadModel
from run import chat, save_json

CFG = json.loads(Path(__file__).with_name("natural.json").read_text())


class Probe(Tracer, HeadModel):
    """The trace and head hooks are active on the same native model."""


def tokenize(model, text):
    buf = np.empty(256, np.int32)
    n = model.lib.tokenize(text.encode(), buf, len(buf))
    return buf[:n].copy() if 0 < n <= len(buf) else None


def sample(model, dataset):
    rng = np.random.default_rng(CFG["seed"])
    front, back = chat("FACT", "QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")
    articles = dataset["data"]
    count = 0
    for ai in rng.permutation(len(articles)):
        article = articles[ai]
        for pi in rng.permutation(len(article["paragraphs"])):
            para = article["paragraphs"][pi]
            candidates = []
            for qi in rng.permutation(len(para["qas"])):
                q = para["qas"][qi]
                prompt = front + "Answer the question using only the exact answer from the passage.\n" + para["context"] + "\n" + q["question"] + back
                ids = tokenize(model, prompt)
                if ids is None:
                    continue
                # Use joint tokenization for each annotator so the first token is valid
                # at this precise answer boundary. Read enough buffer for the full span.
                initial = set()
                for answer in q["answers"]:
                    span = answer["text"].strip()
                    if not span:
                        break
                    buf = np.empty(len((prompt + span).encode()) + 1, np.int32)
                    n = model.lib.tokenize((prompt + span).encode(), buf, len(buf))
                    if n <= len(ids) or not np.array_equal(buf[:len(ids)], ids):
                        break
                    initial.add(int(buf[len(ids)]))
                else:
                    if len(initial) == 1 and (not candidates or next(iter(initial)) != candidates[0][1]):
                        candidates.append((q["id"], next(iter(initial)), ids))
                if len(candidates) == 2:
                    break
            if len(candidates) == 2:
                yield article["title"], int(pi), candidates
                count += 1
                break
        if count == CFG["pairs"]:
            return
    raise ValueError(f"Only {count} eligible articles")


def measure(model, pair, layer, selected, rng):
    title, pi, candidates = pair
    labels = [c[1] for c in candidates]
    ids = [c[2] for c in candidates]
    def observe(i, edit=None):
        return model.evaluate(ids[i], len(ids[i])-1, edit, labels[1-i], labels[i], layer=layer)[2]
    base, states, activations, prior = [], [], [], []
    for i in (0, 1):
        model.head(layer)
        trace, baseline = model.clean(ids[i], labels[1-i], labels[i])
        base.append(baseline)
        states.append(trace[:,1])
        activations.append(model.buffer.copy())
        model.head(23)
        assert observe(i) == baseline
        prior.append(model.buffer.copy())
    diff = states[1] - states[0]
    update = diff[layer-15] - diff[layer-16]
    random = rng.normal(size=model.width).astype(np.float32)
    random *= np.linalg.norm(update)/np.linalg.norm(random)
    np.testing.assert_allclose(np.linalg.norm(update), np.linalg.norm(random), rtol=1e-6)
    patches = []
    controls = [(selected + j) % 16 for j in (3,7,11)]
    for i in (0,1):
        model.head()
        side = {key:observe(i, (1-2*i)*edit) for key, edit in (
            ("update", update), ("random", random), ("whole", diff[layer-15]), ("sham", np.zeros_like(random)))}
        assert side["sham"] == base[i]
        for h in [selected, *controls]:
            model.head(layer, h, activations[1-i][h])
            side[str(h)] = observe(i)
        model.head(23, 0, prior[1-i][0])
        side["prior"] = observe(i)
        patches.append(side)
    model.head()
    return dict(article=title, paragraph=pi, qids=[c[0] for c in candidates], lengths=list(map(len,ids)),
                labels=labels, clean=base, patches=patches)


def summarize(rows, head):
    labels = np.array([r["labels"] for r in rows])
    top = np.array([[x["top_token"] for x in r["clean"]] for r in rows])
    draws = np.random.default_rng(91062).integers(len(rows), size=(10000, len(rows)))
    def stat(a):
        return dict(mean=float(a.mean()), ci95=np.quantile(a[draws].mean(1), [.025,.975]).tolist())
    effects, interventions = {}, {}
    for key in rows[0]["patches"][0]:
        gain = np.array([[s[key]["donor_minus_original"]-r["clean"][i]["donor_minus_original"]
                          for i,s in enumerate(r["patches"])] for r in rows]).mean(1)
        patched = np.array([[s[key]["top_token"] for s in r["patches"]] for r in rows])
        effects[key] = gain
        interventions[key] = dict(gain=stat(gain), donor_top=int((patched==labels[:,::-1]).sum()),
            flips_from_clean_correct=int(((top==labels)&(patched==labels[:,::-1])).sum()))
    controls = [effects[str((head+j)%16)] for j in (3,7,11)]
    return dict(articles=len(rows), questions=labels.size, clean_first=int((top==labels).sum()), unique_tokens=len(set(labels.ravel())),
                block_minus_random=stat(effects["update"]-effects["random"]),
                selected_minus_controls=stat(effects[str(head)]-np.mean(controls,axis=0)), interventions=interventions)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("library","model","dataset","selection","out"):
        p.add_argument(name,type=Path)
    a = p.parse_args()
    assert hashlib.sha256(a.dataset.read_bytes()).hexdigest() == CFG["sha256"]
    selection = json.loads(a.selection.read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    model = Probe(a.library,a.model)
    rng = np.random.default_rng(CFG["seed"]+1)
    rows = [measure(model,pair,selection["block"],selection["head"],rng)
            for pair in sample(model,json.loads(a.dataset.read_text()))]
    assert len(set(r["article"] for r in rows)) == len(rows) == CFG["pairs"]
    save_json(a.out/"rows.json",rows)
    save_json(a.out/"result.json",summarize(rows,selection["head"]))
