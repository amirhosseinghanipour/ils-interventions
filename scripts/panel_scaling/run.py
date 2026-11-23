"""Fit and time sixteen-head intervention predictors on independent prompt templates."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.component_routing.sampling import WEIGHTS
from .data import encode
from .fit import design, fit
from .oracle import Oracle

WEIGHT_HASHES = {**WEIGHTS, "gpt2": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"}
BUDGETS = (0, 8, 16, 24, 32, 48, 64, 96, 128, 256)
METHODS = ("midpoint", "linear", "dense", "ils")
TOL = 0.10
TASKS = ("gift", "route")


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    p.parent.mkdir(exist_ok=True, parents=True)
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(p)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def clock():
    torch.cuda.synchronize()
    return perf_counter() * 1000


def attribution(model, ids, labels, layers, gpt):
    modules = [
        model.transformer.h[i].attn.c_proj if gpt else model.model.layers[i].self_attn.o_proj
        for i in layers
    ]
    states = []
    grads = []
    for is_clean, seq in ((False, ids[1]), (True, ids[0])):
        seen = {}
        with ExitStack() as hooks:
            for i, mod in enumerate(modules):

                def capture(mod, args, i=i):
                    val = args[0]
                    if is_clean and not val.requires_grad:
                        val = val.detach().requires_grad_(True)
                    seen[i] = val
                    return (val, *args[1:])

                hooks.enter_context(mod.register_forward_pre_hook(capture))
            logits = model(torch.tensor([seq], device=model.device), use_cache=False).logits[0, -1]
        ordered = [seen[i] for i in range(2)]
        if is_clean:
            grads = torch.autograd.grad(logits[labels[1]] - logits[labels[0]], ordered)
        states.append([s.detach() for s in ordered])
    return np.stack(
        [
            ((a - b) * g)
            .reshape(1, len(ids[0]), model.config.num_attention_heads, -1)
            .sum((0, 1, 3))
            .abs()
            .cpu()
            .numpy()
            for a, b, g in zip(*states, grads)
        ]
    )


def choose_panel(model, tok, dev, name):
    levels = (5, 8) if name == "gpt2" else (11, 17)
    attrs = [attribution(model, *encode(tok, row), levels, name == "gpt2") for row in dev]
    mean = np.mean(attrs, axis=0)
    sites = [
        [l, int(h)]
        for l, values in zip(levels, mean)
        for h in sorted(np.argsort(-values, kind="stable")[:8])
    ]
    return sites, mean.tolist()


def masks(row):
    seed = int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4], "big")
    return design(seed, pairs=128)


def tune_chunks(o, train, target):
    rows = {}
    for n in (*BUDGETS[1:], 512):
        query = (target if n == 512 else train)[:n]
        times = {}
        for chunk in sorted({min(n, c) for c in (16, 32, 64, 128)}):
            if chunk > n:
                continue
            try:
                o.exact(query, chunk=chunk)
                obs = []
                for _ in range(2):
                    start = clock()
                    o.exact(query, chunk=chunk)
                    obs.append(clock() - start)
                times[str(chunk)] = obs
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
        if not times:
            raise RuntimeError(f"No exact implementation could evaluate {n} queries")
        rows[str(n)] = {"chunk": int(min(times, key=lambda k: median(times[k]))), "times_ms": times}
    return rows


def score_grid(g, tr, y, tar, gold):
    errors = {}
    for method in METHODS:
        errors[method] = {}
        for b in BUDGETS if method != "midpoint" else (0,):
            pred = fit(method if b else "midpoint", g, tr[:b], y[:b], tar)
            errors[method][str(b)] = float(np.abs(pred - gold).mean())
    return errors


def develop(model, tok, dev, name):
    sites, attr = choose_panel(model, tok, dev, name)
    tasks = {}
    for task in TASKS:
        records = []
        for row in [r for r in dev if r["task"] == task]:
            ids, labels = encode(tok, row)
            o = Oracle(model, ids, labels, sites)
            tr, tar = masks(row)
            if not records:
                choices = tune_chunks(o, tr, tar)
            g = np.asarray(o.gradient())
            y = np.asarray(o.exact(tr, chunk=choices["256"]["chunk"]))
            gold = np.asarray(o.exact(tar, chunk=choices["512"]["chunk"]))
            records.append(
                dict(
                    id=row["id"],
                    gradient=g.tolist(),
                    calibration=y.tolist(),
                    gold=gold.tolist(),
                    errors=score_grid(g, tr, y, tar, gold),
                )
            )
        operating = {}
        for m in METHODS:
            opts = [
                b
                for b in (BUDGETS if m != "midpoint" else (0,))
                if np.mean([r["errors"][m][str(b)] for r in records]) <= TOL
            ]
            operating[m] = min(opts) if opts else None
        tasks[task] = dict(rows=records, chunks=choices, operating=operating)
    return dict(model=name, sites=sites, mean_absolute_attribution=attr, tasks=tasks)


def evaluate(model, tok, row, sites, config):
    ids, labels = encode(tok, row)
    start = clock()
    o = Oracle(model, ids, labels, sites)
    common = clock() - start
    tr, tar = masks(row)
    choices = {int(n): v["chunk"] for n, v in config["chunks"].items()}
    operating = config["operating"]
    keys = {m: ("exact" if b is None else ("midpoint" if b == 0 else m)) for m, b in operating.items()}
    methods = sorted(set(keys.values()) | {"exact", "midpoint"})
    timings = {m: [] for m in methods}
    forecasts = {}
    rng = np.random.default_rng(int(tr[0]))
    for _ in range(3):
        for m in rng.permutation(methods):
            start = clock()
            if m == "exact":
                pred = o.exact(tar, chunk=choices[512])
            else:
                b = 0 if m == "midpoint" else operating[m]
                g = o.gradient()
                y = o.exact(tr[:b], chunk=choices[b]) if b else []
                pred = fit(m, g, tr[:b], y, tar)
            timings[m].append(clock() - start)
            a = np.asarray(pred, dtype=float)
            if m in forecasts:
                np.testing.assert_allclose(forecasts[m], a, rtol=2e-4, atol=3e-4)
            forecasts[m] = a
    # The following data audit every predictor at equal budgets after the timed loop.
    g = np.asarray(o.gradient())
    y = np.asarray(o.exact(tr, chunk=choices[256]))
    gold = forecasts["exact"]
    errs = score_grid(g, tr, y, tar, gold)
    for m, key in keys.items():
        if key != "exact":
            np.testing.assert_allclose(
                errs[m][str(operating[m])], abs(forecasts[key] - gold).mean(), atol=3e-4
            )
    audit = float(o.audit([0, 65535, tr[0], tr[1], tar[0], tar[-1]]))
    if audit > 0.001 or not np.isfinite(audit):
        raise ValueError(f"Native substitution discrepancy {audit}")
    return dict(
        id=row["id"],
        task=row["task"],
        template=row["template"],
        labels=labels,
        clean_top=int(o.clean_top),
        donor_top=int(o.donor_top),
        common_ms=common,
        timings_ms=timings,
        methods=keys,
        gradient=g.tolist(),
        calibration=y.tolist(),
        gold=gold.tolist(),
        native_max_error=audit,
        errors=errs,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "manifest", "out"):
        p.add_argument(key, type=Path)
    p.add_argument("model", choices=WEIGHT_HASHES)
    p.add_argument("stage", choices=("development", "heldout"))
    a = p.parse_args()
    rows = read(a.manifest)["rows"]
    name = a.model
    path = a.inputs / name
    if sha(path / "model.safetensors") != WEIGHT_HASHES[name]:
        raise ValueError("Unexpected checkpoint")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    devfile = a.out / f"development-{name}.json"
    signature = {f: sha(Path(__file__).with_name(f)) for f in ("run.py", "oracle.py", "data.py", "fit.py")}
    signature.update(manifest=sha(a.manifest), weights=WEIGHT_HASHES[name])
    if a.stage == "development":
        result = develop(model, tok, [r for r in rows if r["split"] == "development"], name)
        result.update(signature=signature, device=torch.cuda.get_device_name())
        save(devfile, result)
        print(name, {t: c["operating"] for t, c in result["tasks"].items()}, flush=True)
        return
    dev = read(devfile)
    assert dev["signature"] == signature
    stamp = sha(devfile)
    collected = []
    for row in [r for r in rows if r["split"] == "heldout"]:
        part = a.out / f"parts-{name}" / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
        if not part.exists():
            v = evaluate(model, tok, row, dev["sites"], dev["tasks"][row["task"]])
            save(part, dict(freeze=stamp, row=v))
        obj = read(part)
        assert obj["freeze"] == stamp and obj["row"]["id"] == row["id"]
        collected.append(obj["row"])
        if len(collected) % 8 == 0:
            print(name, len(collected), flush=True)
    save(a.out / f"heldout-{name}.json", dict(freeze=stamp, rows=collected))


if __name__ == "__main__":
    main()
