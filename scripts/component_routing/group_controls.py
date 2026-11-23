"""Check grouped interventions against integrated gradients and exact singleton additivity."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from amortize import masks_for, timed
from benchmark import save
from engine import Circuit
from groups import data, read
from sampling import WEIGHTS, encode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("groups", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("model", choices=WEIGHTS)
    a = p.parse_args()
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
    out = {}
    for task, refs in read(a.groups).items():
        rows = data(a.inputs, task + "-test")
        out[task] = []
        for n, r in enumerate(refs):
            ids, labels = encode(tok, rows[r["index"]], r["counter"])
            c = Circuit(model, ids, labels)
            mask = torch.tensor(
                np.argsort(r["order"])[None, :] < np.array(r["sizes"])[:, None], device=model.device
            )
            record = dict(index=r["index"], counter=r["counter"], predictions={})
            methods = ("midpoint", "exact_additive", "matched_ig5")
            for m in methods:
                _, pred = timed(c, mask, m)
                record["predictions"][m] = pred.tolist()
            np.testing.assert_allclose(
                record["predictions"]["midpoint"],
                r["predictions"]["matched_midpoint"],
                rtol=3e-4,
                atol=6e-4,
            )
            if n < 2:
                workload = torch.tensor(masks_for(c.heads, 731955 + n), device=model.device)
                times = {m: [] for m in methods}
                for rep in range(4):
                    for m in np.random.default_rng(18034 + n + rep).permutation(methods):
                        elapsed, _ = timed(c, workload, m)
                        if rep:
                            times[m].append(elapsed)
                record["timings_128_ms"] = times
            out[task].append(record)
        print(a.model, task, len(refs), flush=True)
    save(a.dest, out)


if __name__ == "__main__":
    main()
