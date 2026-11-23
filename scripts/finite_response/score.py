"""Fit on Qwen articles and evaluate frozen predictors on both held-out checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from collect import WEIGHTS
from sample import TEST, TRAIN, digest

# For each method, a fixed offset and zero or more columns with fitted scalar weights.
def features(row):
    p, z, end = (row[k].astype("float64") for k in ("partial", "direction", "endpoint"))
    denom = np.maximum(np.sum(z*z, axis=-1, keepdims=True), 1e-18)
    alpha = np.sum(p*z, axis=-1, keepdims=True) / denom
    ratio = np.sqrt(np.sum(end*end, axis=-1, keepdims=True) / denom)
    ratio = np.where(np.sum(z*z, axis=-1, keepdims=True) > 1e-18, ratio, 1.)
    orthogonal = ratio * (p - alpha*z)
    parallel = alpha * end
    zero = np.zeros_like(p)
    return {
        "no_change": (zero, []),
        "clean_donor": (end, []),
        "fit_donor": (zero, [end]),
        "local_derivative": (row["tangent"], []),
        "midpoint_derivative": (row["midpoint"], []),
        "fit_local_derivative": (zero, [row["tangent"]]),
        "partial": (zero, [p]),
        "partial_donor_mix": (zero, [p, end]),
        "norm_transport": (zero, [np.linalg.norm(p, axis=-1, keepdims=True)/np.sqrt(denom)*end]),
        "projection_only": (parallel, []),
        "secant_transport": (parallel, [orthogonal]),
    }


def fit(rows):
    # Calibrate only on the first answer position. Held-out answer length has no influence.
    stats = {}
    for row in rows:
        for key, (offset, cols) in features(row).items():
            if not cols:
                stats[key] = None
                continue
            x = np.stack([col[0] for col in cols], axis=1).astype("float64")
            y = (row["target"][0] - offset[0]).astype("float64")
            if key not in stats:
                stats[key] = [np.zeros((len(cols),len(cols))), np.zeros(len(cols))]
            stats[key][0] += x.T @ x
            stats[key][1] += x.T @ y
    return {k:[] if v is None else np.linalg.lstsq(*v, rcond=1e-12)[0].tolist()
            for k,v in stats.items()}


def load(path, manifest_hash):
    with np.load(path) as row:
        if str(row["manifest"]) != manifest_hash:
            raise ValueError("Manifest differs from collection")
        return {k:row[k] for k in row.files}


def evaluate(model, row, weights):
    info = json.loads(str(row["info"]))
    base, target = row["base"], row["target"]
    def top(x):
        t = torch.as_tensor(x, device=model.device, dtype=torch.float32)
        return model.lm_head(model.model.norm(t)).argmax(-1).tolist()
    clean = info["clean_generation"]
    assert clean == top(base)
    actual = top(base + target)
    assert actual == info["gold_along_clean"]
    results = {}
    for key, (offset, cols) in features(row).items():
        pred = offset.copy()
        for weight, col in zip(weights[key], cols, strict=True):
            pred = pred + weight * col
        gen = top(base + pred)
        results[key] = dict(sse=float(np.sum((pred[0] - target[0]).astype("float64")**2)),
                            changed=(gen != clean), first_token=gen[0])
    return dict(actual_changed=clean != actual, first_token=actual[0], response_tokens=len(clean),
                ended=info["ended"], energy=float(np.sum(target[0].astype("float64")**2)),
                clean_first=clean[0], source_position=info["source"], prompt_length=info["tokens"],
                checks={k:info[k] for k in ("relative_jvp_check","patched_generation","median_ms") if k in info},
                methods=results)


def aggregate(records):
    energy = np.array([r["energy"] for r in records])
    truth = np.array([r["actual_changed"] for r in records])
    rng = np.random.default_rng(18037)
    draws = rng.integers(0, len(truth), (10000,len(truth)))
    def stat(v):
        boot = v[draws].sum(1)/energy[draws].sum(1)
        return dict(value=v.sum()/energy.sum(), ci95=np.quantile(boot,[.025,.975]).tolist())
    methods = {}
    candidate = np.array([r["methods"]["secant_transport"]["sse"] for r in records])
    for k in records[0]["methods"]:
        sq = np.array([r["methods"][k]["sse"] for r in records])
        preds = np.array([r["methods"][k]["changed"] for r in records])
        methods[k] = dict(normalized_squared_error=stat(sq), error_reduction_with_secant=stat(sq-candidate),
            switch_true_positive=int((truth & preds).sum()), switch_false_positive=int((~truth & preds).sum()),
            switch_true_negative=int((~truth & ~preds).sum()), switch_false_negative=int((truth & ~preds).sum()),
            first_token_agreement=sum(r["methods"][k]["first_token"] == r["first_token"] for r in records))
    return dict(articles=len(records), measured_answer_changes=int(truth.sum()),
                clean_ended_by_cap=sum(r["ended"] for r in records), methods=methods)


def main():
    parser = argparse.ArgumentParser()
    for key in ("inputs","manifest","vectors","out"):
        parser.add_argument(key,type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    train = [r for r in rows if r["split"] == "fit"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(train) == TRAIN and len(test) == TEST and len({r["article"] for r in rows}) == len(rows)
    sha = digest(args.manifest)
    def source(row, name):
        return load(args.vectors / name / (row["qid"] + ".npz"), sha)
    weights = fit(source(row,"qwen") for row in train)
    args.out.mkdir(parents=True, exist_ok=True)
    def save(name, obj):
        (args.out / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    save("weights.json", dict(fit_checkpoint="qwen",manifest_sha256=sha,weights=weights))
    for name in WEIGHTS:
        model = AutoModelForCausalLM.from_pretrained(args.inputs / name, dtype=torch.float32,
                   local_files_only=True).requires_grad_(False).eval().cuda()
        results = [dict(qid=row["qid"],article=row["article"],**evaluate(model,source(row,name),weights))
                   for row in test]
        save(name+"-rows.json", results)
        save(name+".json", aggregate(results))
        del model


if __name__ == "__main__":
    main()
