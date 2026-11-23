"""Calibrate on IOI and run timed policies before collecting the exhaustive table."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .data import data, encode, model, read, save
from .oracle import Oracle
from .policies import PAIR_GRID, TRIPLES, best_verified, propose, propose_grid

METHODS = ("ils", "linear", "midpoint", "atp", "ig5", "guided", "scan8", "scan32", "exhaustive", "greedy")
FALLBACK = 7


def now():
    torch.cuda.synchronize()
    return perf_counter() * 1000


def budget(o):
    durations = []
    for rep in range(5):
        start = now()
        base = o.gradient("midpoint")
        p = propose("ils", {"midpoint": base}, o.exact, seed=52815, max_pairs=max(PAIR_GRID))
        o.exact(list(p.ranking[:8]))
        durations.append(now() - start)
    return dict(warm_ms=durations, cap_ms=float(np.median(durations[1:]) * 1.10))


def timed(o, name, seed, pairs, cap, fallback):
    start = now()
    paid = []

    def measure(masks, role="fit"):
        values = o.exact(masks)
        paid.append(dict(role=role, masks=list(masks), values=values, ms=now() - start))
        return values

    scan = name.startswith("scan") or name == "exhaustive"
    method = "scan" if scan else "atp" if name == "guided" else name
    grad_name = "midpoint" if method in ("ils", "linear") else method
    grads = {} if method in ("scan", "greedy") else {grad_name: o.gradient(grad_name)}
    p = propose(method, grads, measure, seed=seed, max_pairs=pairs)
    ranking = list(p.ranking)
    known = {FALLBACK: fallback}
    order = [m for m in ranking if m != FALLBACK]
    if name == "guided":
        rng = np.random.default_rng(seed)
        ranks = sorted(order, key=lambda m: -abs(p.scores[m]))
        pilot = [int(rng.choice(b)) for b in np.array_split(ranks, 4)]
        batches = [pilot]
        order = [m for m in order if m not in pilot]
    else:
        batches = []
    size = 56 if name == "exhaustive" else 32 if name == "scan32" else 8
    batches.extend(order[i : i + size] for i in range(0, len(order), size))
    initial = now() - start
    decision_ms = 0.0
    # The shared fallback is a paid native measurement available at time zero.
    winner = FALLBACK
    for batch in batches:
        if now() - start >= cap:
            break
        vals = measure(batch, "verify")
        known.update(zip(batch, vals, strict=True))
        candidate = int(best_verified(ranking, known))
        decided = now() - start
        paid[-1]["decision_ms"] = decided
        if decided <= cap:
            winner = candidate
            decision_ms = decided
    return dict(
        raw_mask=int(ranking[0]),
        winner=winner,
        cap_ms=cap,
        initial_ms=initial,
        decision_ms=decision_ms,
        elapsed_ms=now() - start,
        paid=paid,
        calibration_masks=list(p.calibration_masks),
        selected_pairs=p.selected_pairs,
        scores=None if scan else p.scores.tolist(),
    )


def row_oracle(ref, bank, tok, m, sites):
    ids, labels = encode(tok, bank[ref["file"]][ref["index"]])
    return Oracle(m, ids, labels, sites), labels


def main():
    p = argparse.ArgumentParser()
    for field in ("inputs", "artifacts", "out"):
        p.add_argument(field, type=Path)
    p.add_argument("stage", choices=["dev", "qwen", "large"])
    a = p.parse_args()
    name = "qwen" if a.stage == "dev" else a.stage
    m, tok = model(a.inputs, name)
    refs = read(a.artifacts / "manifest.json")
    sites = read(a.artifacts / ("panel-" + name + ".json"))["sites"]
    bank = {f: data(a.inputs, f) for f in ("ioi-test.json", "colors-validation.jsonl")}
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "dev":
        runs = []
        for n, r in enumerate(refs["development"]):
            o, labels = row_oracle(r, bank, tok, m, sites)
            predictions = propose_grid({"midpoint": o.gradient("midpoint")}, o.exact, seed=781023 + n)
            measured = o.measure()
            g = np.asarray(measured["gold"])
            rr = {}
            for k, v in predictions.items():
                ids = list(TRIPLES)
                rr[k] = dict(
                    raw_mask=v.ranking[0],
                    regret=float(g[ids].max() - g[v.ranking[0]]),
                    mae=float(abs(g[ids] - v.scores[ids]).mean()),
                )
            runs.append(dict(**r, labels=labels, **measured, grid=rr))
        losses = {
            k: [float(np.mean([r["grid"][k][s] for r in runs])) for s in ("regret", "mae")]
            for k in PAIR_GRID
        }
        save(a.out / "development.json", dict(rows=runs))
        save(
            a.out / "frozen.json",
            dict(max_pairs=min(PAIR_GRID, key=lambda k: (*losses[k], k)), development_regret_mae=losses),
        )
        print("frozen", losses, flush=True)
        return

    frozen = read(a.out / "frozen.json")
    path = a.out / ("budget-" + name + ".json")
    if not path.exists():
        # Separate fixed calibration prompt; the clock allowance uses no evaluation rows.
        bo, _ = row_oracle(refs["panel"][0], bank, tok, m, sites)
        save(path, budget(bo))
    cap = read(path)["cap_ms"]
    for task in ("ioi", "colors"):
        rows = []
        for n, r in enumerate(refs[task]):
            checkpoint = a.out / ("parts-" + name + "-" + task) / f"{n:03}.json"
            if checkpoint.is_file():
                rows.append(read(checkpoint))
                continue
            oracle, labels = row_oracle(r, bank, tok, m, sites)
            start = now()
            fb = oracle.exact([FALLBACK])[0]
            fb_ms = now() - start
            methods = {}
            for key in np.random.default_rng(34017 + n).permutation(METHODS).tolist():
                methods[key] = timed(oracle, key, 225177 + n, frozen["max_pairs"], cap, fb)
            measured = oracle.measure()
            row = dict(
                **r, labels=labels, **measured, methods=methods, fallback_value=fb, fallback_ms=fb_ms
            )
            save(checkpoint, row)
            rows.append(row)
            if (n + 1) % 12 == 0:
                print(name, task, n + 1, flush=True)
        save(a.out / ("primary-" + name + "-" + task + ".json"), dict(sites=sites, rows=rows))


if __name__ == "__main__":
    main()
