"""Collect a new holdout and compare the fixed predictors with matched targets."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ilslens.data import wikitext_sequences
from ilslens.model import load_local
from ilslens.suffix import QwenSuffix
from ilslens.scalar_transport import scalar_transport
from ilslens.experiment import write_json
from scripts.predictive_coverage import cv_ridge, samples, summarize, root, manifest
from scripts.coverage_innovation import weights

plan_path = Path(__file__).with_name("fresh-config.json")
p = json.loads(plan_path.read_text())
model, tok = load_local("cuda")
n = p["source_prompts"]
layers = p["layers"]


def sequences(offset):
    return wikitext_sequences(
        "../data/wikitext-103-raw-v1",
        tok,
        p["split"],
        n,
        offset=offset,
        length=p["sequence_length"],
    )


source, src = sequences(p["source_offset"])
donor, dnr = sequences(p["donor_offset"])
previous = json.loads((root / "heldout-provenance.json").read_text())
faith = json.loads(Path("runs/faithfulness-v1/results.json").read_text())
old = (
    manifest["source_rows"]
    + manifest["donor_rows"]
    + previous["sources"]
    + previous["donors"]
    + faith["source_provenance"]
    + faith["donor_provenance"]
)
new = {r["sha256"] for r in src + dnr}
assert len(new) == 2 * n and new.isdisjoint({r["sha256"] for r in old})
rng = np.random.default_rng(p["intervention_seed"])
obs = {l: [] for l in layers}
positions = []
donor_positions = []
for i in range(n):
    a = QwenSuffix(model, source[i : i + 1])
    b = QwenSuffix(model, donor[i : i + 1])
    for _ in range(p["draws_per_source"]):
        t, u = map(int, rng.integers(16, p["sequence_length"] - 1, size=2))
        positions.append(t)
        donor_positions.append(u)
        for l in layers:
            delta = p["epsilon"] * (b.states[l][0, u] - a.states[l][0, t])
            response = a.effects(l, t, delta[None], gammas=p["gammas"])[0]
            obs[l].append((delta.cpu(), response.cpu()))
    del a, b
del model
test = torch.arange(n).repeat_interleave(p["draws_per_source"])
wt = weights(torch.tensor(positions, device="cuda"))
rows = []
archive = {}
for l in layers:
    xt, yt = [torch.stack(v).cuda() for v in zip(*obs[l])]
    x, y, t, groups = samples("train", l, "cuda")
    w = weights(t)
    residual = y - w[:, :, None] * x[:, None]
    corrections, lams = cv_ridge(x, residual, groups) if l != 23 else (None, [None] * 4)
    for i, g in enumerate(p["gammas"]):
        gg = float(g)
        fit = torch.load(
            root / f"fits/layer-{l:02}-epsilon-0.05-gamma-{gg}.pt", weights_only=True
        )
        assert fit["metadata"]["fingerprint"] == manifest["fingerprint"]
        skip = xt * wt[:, i, None]
        corrected = skip + xt @ corrections[i].T if l != 23 else skip
        scaled = scalar_transport(
            x.cpu(), y[:, i].cpu(), w[:, i].cpu(), xt.cpu(), wt[:, i].cpu()
        )
        preds = {
            "ILS": corrected,
            "scalar": torch.from_numpy(scaled),
            "direct": skip,
            "original_ILS": xt @ fit["matrix"].cuda().T,
        }
        summary = summarize(yt[:, i], preds, test, archive, f"{l}/{i}/total")
        if l == 23:
            assert summary["direct"]["estimate"] > 0.999999
        remainder = (
            None
            if l == 23
            else summarize(
                yt[:, i] - skip,
                {"ILS": corrected - skip},
                test,
                archive,
                f"{l}/{i}/remainder",
            )["ILS"]
        )
        rows.append(
            dict(
                layer=l,
                gamma=g,
                penalty=lams[i],
                corrected=summary["ILS"],
                scalar=summary["scalar"],
                direct=summary["direct"],
                original_ILS=summary["original_ILS"],
                gain_over_scalar=summary["ILS minus scalar"],
                gain_over_direct=summary["ILS minus direct"],
                gain_over_original=summary["ILS minus original_ILS"],
                remainder=remainder,
            )
        )
np.savez_compressed("artifacts/qwen-fresh-sufficient.npz", **archive)
write_json(
    "artifacts/qwen-fresh-predictive-coverage.json",
    dict(
        plan=p,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        primary_fingerprint=manifest["fingerprint"],
        sources=src,
        donors=dnr,
        positions=positions,
        donor_positions=donor_positions,
        rows=rows,
    ),
)
