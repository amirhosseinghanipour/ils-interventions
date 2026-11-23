"""Audit external article sampling and rebuild its primary statistic from raw logits."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def verify(data, corpus):
    articles = {a["title"]: a["paragraphs"] for a in read(corpus)["data"]}
    qids = None
    for size in ("9b", "4b"):
        root = data/"natural"/size
        rows, result = read(root/"rows.json"), read(root/"result.json")
        assert len({r["article"] for r in rows}) == len(rows) == 40
        current = [r["qids"] for r in rows]
        if qids is not None:
            assert qids == current
        qids = current
        deltas = []
        clean = flips = 0
        for r in rows:
            ids = {qa["id"] for qa in articles[r["article"]][r["paragraph"]]["qas"]}
            assert len(set(r["qids"])) == len(set(r["labels"])) == 2
            assert set(r["qids"]).issubset(ids) and max(r["lengths"]) <= 256
            diffs = []
            for i, (x, p) in enumerate(zip(r["clean"], r["patches"])):
                assert x == p["sham"]
                clean += x["top_token"] == r["labels"][i]
                flips += x["top_token"] == r["labels"][i] and p["update"]["top_token"] == r["labels"][1-i]
                diffs.append(p["update"]["donor_minus_original"] - p["random"]["donor_minus_original"])
            deltas.append(sum(diffs)/2)
        draws = np.random.default_rng(91062).choice(deltas, size=(10000, 40))
        np.testing.assert_allclose(np.percentile(draws.mean(1), [2.5,97.5]), result["block_minus_random"]["ci95"])
        np.testing.assert_allclose(np.mean(deltas), result["block_minus_random"]["mean"])
        assert clean == result["clean_first"] and flips == result["interventions"]["update"]["flips_from_clean_correct"]
        crossed = data/"crossed"/size
        if crossed.exists():
            for task in ("blocks","natural"):
                reference = (read(data/"blocks"/size/"test.json") if task=="blocks" else rows)
                assigned = read(crossed/(task+".json"))
                stats = read(crossed/(task+"-result.json"))
                n = len(reference)
                assert len(assigned) == n
                margins = []
                for i, a in enumerate(assigned):
                    j = (i+1)%n
                    assert (a["index"], a["other"]) == (i,j)
                    overlap = set(a["answers"]) & set(assigned[j]["answers"])
                    assert set(a["shared"]) == overlap
                    ref = reference[i]
                    original = ref["scan"]["19"] if task=="blocks" else ref["patches"]
                    assert a["correct_donor"] == [z["update"] for z in original]
                    margins.append(sum(t["donor_minus_original"]-c["donor_minus_original"]
                                       for t,c in zip(a["correct_donor"],a["other_donor"]))/2)
                np.testing.assert_allclose(np.mean(margins), stats["mean"])
                samples = np.random.default_rng(91062).choice(margins, size=(10000,n))
                np.testing.assert_allclose(np.percentile(samples.mean(1),[2.5,97.5]),stats["ci95"])
    print("Independently verified 40 common articles, gold question IDs, paired logit gains, flips, shams and uncertainty.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("corpus", type=Path)
    a = p.parse_args()
    verify(a.data, a.corpus)
