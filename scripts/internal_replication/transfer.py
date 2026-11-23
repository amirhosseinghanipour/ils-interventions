"""Evaluate the WikiText penultimate-block predictor on distinct FineWeb documents."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ilslens.experiment import write_json
from ilslens.model import load_local
from ilslens.scalar_transport import scalar_transport
from ilslens.suffix import QwenSuffix
from scripts.coverage_innovation import weights
from scripts.predictive_coverage import (
    cv_ridge,
    gammas,
    manifest,
    root,
    samples,
    summarize,
)


def collect_documents(path, tokenizer, plan, excluded):
    documents, provenance = [], []
    seen_ids = set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["source"] != "FineWeb/sample-10BT"
                or row["split"] != "validation"
                or row["id"] in seen_ids
            ):
                continue
            tokens = tokenizer.encode(
                row["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=plan["length"],
            )
            fingerprint = hashlib.sha256(
                ",".join(map(str, tokens)).encode()
            ).hexdigest()
            if len(tokens) < plan["length"] or fingerprint in excluded:
                continue
            excluded.add(fingerprint)
            seen_ids.add(row["id"])
            documents.append(torch.tensor([tokens]))
            provenance.append(dict(document=row["id"], tokens_sha256=fingerprint))
            if len(documents) == 2 * plan["pairs"]:
                return documents, provenance
    raise ValueError("Insufficient eligible FineWeb documents")


def main():
    plan_path = Path(__file__).with_name("transfer-config.json")
    plan = json.loads(plan_path.read_text())
    layer, gamma = plan["layer"], plan["gamma"]
    index = gammas.index(gamma)
    corpus = Path("../data/transcoder-corpus-100m-clean.jsonl")
    with corpus.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert (
        digest == json.loads(corpus.with_suffix(".manifest.json").read_text())["sha256"]
    )
    old = json.loads((root / "heldout-provenance.json").read_text())
    faith = json.loads(Path("runs/faithfulness-v1/results.json").read_text())
    fresh = json.loads(
        Path("artifacts/qwen-fresh-predictive-coverage.json").read_text()
    )
    records = (
        manifest["source_rows"]
        + manifest["donor_rows"]
        + old["sources"]
        + old["donors"]
        + faith["source_provenance"]
        + faith["donor_provenance"]
        + fresh["sources"]
        + fresh["donors"]
    )
    model, tokenizer = load_local("cuda")
    docs, provenance = collect_documents(
        corpus, tokenizer, plan, {r["sha256"] for r in records}
    )
    rng = np.random.default_rng(plan["seed"])
    inputs, targets, sites = [], [], []
    for pair in range(plan["pairs"]):
        source, donor = (
            QwenSuffix(model, seq) for seq in docs[2 * pair : 2 * pair + 2]
        )
        for _ in range(plan["draws_per_pair"]):
            t, u = map(int, rng.integers(16, plan["length"] - 1, size=2))
            delta = plan["epsilon"] * (
                donor.states[layer][0, u] - source.states[layer][0, t]
            )
            inputs.append(delta.cpu())
            targets.append(source.effects(layer, t, delta, gammas=[gamma])[0, 0].cpu())
            sites.append((t, u))
    del model, source, donor
    xt, yt = torch.stack(inputs).cuda(), torch.stack(targets)
    wt = weights(torch.tensor([t for t, _ in sites], device="cuda"))[:, index]
    x, y, positions, groups = samples("train", layer, "cuda")
    w = weights(positions)[:, index]
    residual = y[:, index] - w[:, None] * x
    maps, penalties = cv_ridge(x, residual[:, None], groups)
    direct = wt[:, None] * xt
    correction = xt @ maps[0].T
    scalar = scalar_transport(x.cpu(), y[:, index].cpu(), w.cpu(), xt.cpu(), wt.cpu())
    fit = torch.load(
        root / f"fits/layer-{layer:02}-epsilon-{plan['epsilon']}-gamma-{gamma}.pt",
        weights_only=True,
    )
    assert fit["metadata"]["fingerprint"] == manifest["fingerprint"]
    group = torch.arange(plan["pairs"]).repeat_interleave(plan["draws_per_pair"])
    predictions = {
        "ILS": direct + correction,
        "scalar": torch.from_numpy(scalar),
        "direct": direct,
        "original_ILS": xt @ fit["matrix"].cuda().T,
    }
    archive = {}
    scores = summarize(yt, predictions, group, archive, "total")
    remainder = summarize(
        yt - direct.cpu(), {"ILS": correction}, group, archive, "remainder"
    )
    assert penalties[0] == next(
        r["penalty"]
        for r in fresh["rows"]
        if r["layer"] == layer and r["gamma"] == gamma
    )
    np.savez_compressed("artifacts/qwen-transfer-sufficient.npz", **archive)
    write_json(
        "artifacts/qwen-transfer.json",
        dict(
            plan=plan,
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            corpus_sha256=digest,
            primary_fingerprint=manifest["fingerprint"],
            source_donor_documents=provenance,
            sites=sites,
            penalty=penalties[0],
            metrics=scores,
            remainder=remainder,
        ),
    )


if __name__ == "__main__":
    main()
