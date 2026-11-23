"""Calibrate on addition validation, freeze ILS, and benchmark online exact verification."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine import Circuit
from sampling import FILES, WEIGHTS, encode

METHODS = ("midpoint", "ils", "guided", "atp", "input_midpoint", "scan")
STRENGTHS = (0.25, 0.5, 1.0)


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(path)


def now():
    torch.cuda.synchronize()
    return time.perf_counter() * 1000


def gold(circuit):
    values, top = [], []
    for i in range(0, len(circuit.keys), 8):
        effects, tokens = circuit.exact(list(range(i, min(i + 8, len(circuit.keys)))), tops=True)
        values.append(effects)
        top.extend(tokens)
    return np.concatenate(values), top



def rank(x):
    return np.argsort(-np.abs(x), kind="stable")


def policy(circuit, method, strength, seed, cap):
    """Replayable online publication timeline. Late results earn no credit at a cutoff."""
    start = now()
    out = (
        np.zeros(len(circuit.keys))
        if method == "scan"
        else circuit.estimates("atp" if method == "guided" else method, strength)
    )
    raw_time = now() - start
    order = rank(out)
    rng = np.random.default_rng(seed)
    if method == "scan":
        order = rng.permutation(len(out))
    batch_size = 8 if method == "scan" else 4
    pilot = [int(rng.choice(group)) for group in np.array_split(order, 4)] if method == "guided" else []
    order = pilot + [int(i) for i in order if i not in pilot]
    history = []
    while now() - start < cap and len(history) * batch_size < len(order):
        indices = order[batch_size * len(history) : batch_size * (len(history) + 1)]
        values = circuit.exact(indices)
        correction = float(np.median(values - out[indices])) if pilot and not history else None
        item = dict(indices=indices, values=values.tolist(), correction=correction)
        item["ms"] = now() - start
        history.append(item)
    return dict(raw=out.tolist(), raw_ms=raw_time, history=history)


def budget(model, tok):
    tokens = [
        tok.encode(t, add_special_tokens=False)
        for t in ("Q: How much is 32 plus 13? A: ", "Q: How much is 34 plus 13? A: ")
    ]
    assert len(tokens[0]) == len(tokens[1])
    circuit = Circuit(model, tokens, [tok.encode(n, add_special_tokens=False)[-1] for n in ("4", "5")])
    circuit.audit()
    ils, batch = [], []
    for _ in range(5):
        start = now()
        circuit.estimates("ils", 1.0)
        ils.append(now() - start)
        start = now()
        circuit.exact([0, 1, 2, 3])
        batch.append(now() - start)
    # Identical hardware allowance for every policy on this checkpoint.
    # The tight cap covers the expensive base estimate plus one four-site batch.
    core, verify = float(np.median(ils[1:])), float(np.median(batch[1:]))
    return dict(
        tight=1.20 * core + verify, roomy=1.20 * core + 4 * verify, pilot_ils_ms=ils, pilot_batch_ms=batch
    )


def problem(model, tok, row, ref):
    ids, labels = encode(tok, row, ref["counter"])
    return Circuit(model, ids, labels)


def main():
    p = argparse.ArgumentParser()
    for k in ("inputs", "manifest", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("model", choices=WEIGHTS)
    p.add_argument("split", choices=FILES)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fit = a.split == "addition-validation"
    if fit and a.model != "qwen":
        raise ValueError("Only Qwen addition validation may select the correction strength")
    lock = a.out / "frozen.json"
    if not fit and not lock.exists():
        raise ValueError("Freeze development selection first")
    source = a.inputs / (a.split + ".jsonl")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == FILES[a.split]
    data = [json.loads(s) for s in source.read_text().splitlines()]
    manifest = json.loads(a.manifest.read_text())
    path = a.inputs / a.model
    with (path / "model.safetensors").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHTS[a.model]
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
    tok = AutoTokenizer.from_pretrained(path)
    prefix = a.model + "-" + a.split
    cache = a.out / (prefix + "-parts")
    cache.mkdir(exist_ok=True)
    if fit:
        caps = None
        strength = None
    else:
        frozen = json.loads(lock.read_text())
        assert frozen["manifest_sha256"] == hashlib.sha256(a.manifest.read_bytes()).hexdigest()
        strength = frozen["strength"]
        budget_path = a.out / (a.model + "-budget.json")
        if not budget_path.is_file():
            save(budget_path, budget(model, tok))
        caps = json.loads(budget_path.read_text())
    records = []
    for n, ref in enumerate(manifest[a.split]):
        dest = cache / f"{n:03}.json"
        if dest.exists():
            records.append(json.loads(dest.read_text()))
            continue
        t = now()
        c = problem(model, tok, data[ref["index"]], ref)
        shared = now() - t
        if n < 2:
            c.audit()
        record = dict(
            **ref,
            shared_ms=shared,
            clean_margin=c.outputs[0],
            donor_margin=c.outputs[1],
            clean_correct=c.top[0] == c.labels[0],
            donor_correct=c.top[1] == c.labels[1],
        )
        if fit:
            m = c.estimates("midpoint", 0)
            full = c.estimates("ils", 1)
            record.update(midpoint=m.tolist(), full=full.tolist())
        else:
            order = np.random.default_rng(614931 + n).permutation(METHODS)
            record["methods"] = {key: policy(c, key, strength, 94817 + n, caps["roomy"]) for key in order}
        # Never use exhaustive labels in the online policies.
        t = now()
        exact, top = gold(c)
        record.update(gold=exact.tolist(), patch_top=top, clean_top=c.top[0], donor_top=c.top[1], labels=c.labels, oracle_ms=now() - t)
        if not fit:
            for run in record["methods"].values():
                for batch in run["history"]:
                    np.testing.assert_allclose(
                        batch["values"], exact[batch["indices"]], rtol=3e-4, atol=6e-4
                    )
        save(dest, record)
        records.append(record)
        if (n + 1) % 8 == 0:
            print(prefix, n + 1, flush=True)
    save(
        a.out / (prefix + ".json"),
        dict(
            keys=c.keys
            if "c" in locals()
            else [
                (i, j)
                for i in [len(model.model.layers) * f // 4 - 1 for f in (1, 2, 3)]
                for j in range(model.config.num_attention_heads + 1)
            ],
            rows=records,
        ),
    )
    if fit:
        loss = {}
        for s in STRENGTHS:
            err = []
            for r in records:
                m = np.array(r["midpoint"])
                pred = m + s * (np.array(r["full"]) - m)
                err.append(np.mean(np.abs(pred - r["gold"])))
            loss[s] = float(np.mean(err))
        save(
            lock,
            dict(
                strength=min(STRENGTHS, key=lambda s: (loss[s], s)),
                dev_mae=loss,
                manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            ),
        )
    print("complete", prefix, flush=True)


if __name__ == "__main__":
    main()
