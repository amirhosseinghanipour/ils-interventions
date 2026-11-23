"""Public GPT-2 IOI evaluation with the MIB graph, patching, metric and sparsity grid."""

import argparse
import gzip
import hashlib
import json
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph
from MIB_circuit_track.dataset import HFEAPDataset, collate_EAP
from MIB_circuit_track.metrics import logit_diff
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
METHODS = ("eap_ig", "mattr", "technion", "observed", "linear", "pair")
MODES = ("CPR", "CMD")
SIX_BITS = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)
PAIR_COLUMNS = np.stack([SIX_BITS[:, a] * SIX_BITS[:, b] for a, b in combinations(range(6), 2)], axis=1)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def local_ioi(path, tokenizer):
    # Supply the organizer's pinned JSON locally, then use their real filter and __getitem__.
    ds = HFEAPDataset.__new__(HFEAPDataset)
    ds.task, ds.tokenizer, ds.model_name = "ioi", tokenizer, "gpt2"
    ds.control, ds.counterfactual_type = False, None
    ds.dataset = Dataset.from_list(read(path))
    ds.dataset = ds.filter_dataset()
    return ds


def apply(graph, flat_indices):
    # Let the organizer construct and prune the selected graph using its native top-n code.
    ranking = graph.scores
    graph.scores = torch.zeros_like(ranking)
    try:
        graph.scores.view(-1)[flat_indices] = 1.0
        graph.apply_topn(len(flat_indices), absolute=False)
    finally:
        graph.scores = ranking


def rank_edges(graph, absolute):
    scores = graph.scores.abs() if absolute else graph.scores.clone()
    scores[~graph.real_edge_mask] = -torch.inf
    return torch.argsort(scores.flatten(), descending=True)[: len(graph.edges)]


def boundary_mask(order, size, code):
    # Each bit exchanges equal-sized disjoint bands on opposite sides of the cutoff.
    width = max(1, min(size // 16, (len(order) - size) // 6))
    if size < 6 * width or len(order) - size < 6 * width:
        raise ValueError("Not enough edges to construct six equal-cardinality exchanges")
    mask = order[:size].clone()
    for bit in range(6):
        if code & (1 << bit):
            a, b = size - (bit + 1) * width, size - bit * width
            mask[a:b] = order[size + bit * width : size + (bit + 1) * width]
    return mask


def fits(measured, values, mode):
    # Residual from the unmodified edge ranking. All fits share the same observations.
    measured = np.asarray(measured)
    y = np.asarray(values) - values[0]

    def score(x):
        return x if mode == "CPR" else -abs(x - 1)

    choices = {"observed": int(measured[np.argmax(score(np.asarray(values)))])}
    for method, features in (("linear", SIX_BITS), ("pair", np.column_stack((SIX_BITS, PAIR_COLUMNS)))):
        train = features[measured]
        weights = np.linalg.solve(train.T @ train + 2 * np.eye(train.shape[1]), train.T @ y)
        prediction = values[0] + features @ weights
        choices[method] = int(np.argmax(score(prediction)))
    return choices


def measure(model, graph, loader, cache, tag):
    key = hashlib.sha256(
        graph.in_graph.numpy().tobytes() + graph.nodes_in_graph.numpy().tobytes()
    ).hexdigest()
    path = cache / (tag + "-" + key + ".json")
    if not path.exists():
        values = (
            evaluate_graph(
                model,
                graph,
                loader,
                partial(logit_diff, mean=False),
                quiet=True,
                intervention="patching",
                skip_clean=True,
            )
            .double()
            .tolist()
        )
        save(path, values)
    return np.array(read(path))


def normalized(x, full, empty):
    return float((x.mean() - empty.mean()) / (full.mean() - empty.mean()))


def curve_scores(values):
    return {
        "CPR": float(np.trapezoid(values, GRID)),
        "CMD": float(np.trapezoid(abs(1 - np.array(values)), GRID)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    a.cache.mkdir(exist_ok=True, parents=True)
    raw = a.inputs
    a.out.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "weights": raw / "base-gpt2/model.safetensors",
        "dev": raw / "dev.json",
        "test": raw / "test.json",
        "eap": raw / "upstream/circuits/ioi-eap-ig.json",
        "mattr": raw / "published/mattr.pt",
    }
    for mode in MODES:
        for pct in GRID[:-1]:
            name = f"{mode}-{pct}"
            paths[name] = raw / "published" / f"{mode == 'CMD'}-{pct}_edges.pt"
    checksums = {k: digest(v) for k, v in paths.items()}
    assert checksums["weights"] == "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
    assert checksums["dev"] == "721d6e7a26962c9cc3008029c7327522cb72b9b03e1fa7a9c2da51ab86372c72"
    assert checksums["test"] == "030354654ea1b0b0147eb07f427be0f1ac71beed63a784a76d1581f52e722ebc"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(raw / "base-gpt2", local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(raw / "base-gpt2", local_files_only=True)
    model = (
        HookedTransformer.from_pretrained(
            "gpt2-small", hf_model=hf, tokenizer=tok, device="cuda", local_files_only=True
        )
        .eval()
        .requires_grad_(False)
    )
    del hf
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    graph = Graph.from_json(str(paths["eap"]))
    alt = Graph.from_pt(str(paths["mattr"]))
    assert torch.equal(alt.real_edge_mask, graph.real_edge_mask)
    dev = local_ioi(paths["dev"], tok)
    chosen = np.random.default_rng(867301).choice(len(dev), 16, replace=False).tolist()
    dev_loader = DataLoader([dev[i] for i in chosen], batch_size=a.batch, collate_fn=collate_EAP)
    # Cache reuse requires the same source data, checkpoint, scoring code and prompt order.
    stamp = {
        "files": checksums,
        "development_indices": chosen,
        "batch": a.batch,
        "script": digest(Path(__file__)),
    }
    stamp_path = a.cache / "fingerprint.json"
    if stamp_path.exists():
        assert read(stamp_path) == stamp, "Run changed; use a fresh cache directory"
    else:
        save(stamp_path, stamp)
    ranks = {mode: rank_edges(graph, mode == "CMD") for mode in MODES}
    sizes = {str(pct): int(len(graph.edges) * pct) for pct in GRID}

    def references(loader, tag):
        full = (
            evaluate_baseline(model, loader, partial(logit_diff, mean=False), quiet=True).double().numpy()
        )
        graph.apply_topn(0, True)
        empty = measure(model, graph, loader, a.cache, tag)
        graph.apply_topn(len(graph.edges), absolute=False)
        patched_full = measure(model, graph, loader, a.cache, tag)
        np.testing.assert_allclose(full, patched_full, atol=5e-4, rtol=1e-5)
        assert abs(full.mean() - empty.mean()) > 1e-5
        return full, empty, patched_full

    dev_full, dev_empty, _ = references(dev_loader, "dev")
    policies = {}
    for mode in MODES:
        policies[mode] = {}
        for i, pct in enumerate(GRID[:-1]):
            rank = ranks[mode]
            n = sizes[str(pct)]
            codes = [0] + np.random.default_rng(92144 + i).choice(
                np.arange(1, 64), 23, replace=False
            ).tolist()
            faith = []
            for code in codes:
                apply(graph, boundary_mask(rank, n, code))
                faith.append(
                    normalized(measure(model, graph, dev_loader, a.cache, "dev"), dev_full, dev_empty)
                )
            policies[mode][str(pct)] = {
                "measured": codes,
                "calibration_faithfulness": faith,
                "selected": fits(codes, faith, mode),
            }
            print("Frozen", mode, pct, flush=True)
    save(a.cache / "frozen.json", policies)
    ds = local_ioi(paths["test"], tok)
    templates = list(ds.dataset["template"])
    assert not set(templates).intersection(dev.dataset["template"])
    assert not set(ds.dataset["prompt"]).intersection(dev.dataset["prompt"])
    loader = DataLoader(ds, batch_size=a.batch, collate_fn=collate_EAP)
    full, empty, patched_full = references(loader, "test")
    out = {
        "model": "openai-community/gpt2",
        "split": "test",
        "filtered_public_prompts": len(ds),
        "development_indices": chosen,
        "eligible_development_prompts": len(dev),
        "development_templates": sorted(set(dev.dataset["template"])),
        "test_templates": templates,
        "test_full": full.tolist(),
        "test_empty": empty.tolist(),
        "sha256": checksums,
        "grid": GRID,
        "policies": policies,
        "curves": {},
    }
    for mode in MODES:
        out["curves"][mode] = {}
        for method in METHODS:
            row = []
            for i, pct in enumerate(GRID):
                size = sizes[str(pct)]
                if pct == 1.0:
                    vals = patched_full
                    n = len(graph.edges)
                else:
                    selected_graph = graph
                    if method == "eap_ig":
                        graph.apply_topn(size, absolute=mode == "CMD")
                    elif method == "mattr":
                        alt.apply_topn(size, absolute=mode == "CMD")
                        selected_graph = alt
                    elif method == "technion":
                        selected_graph = Graph.from_pt(str(paths[f"{mode}-{pct}"]))
                        assert torch.equal(selected_graph.real_edge_mask, graph.real_edge_mask)
                    else:
                        code = policies[mode][str(pct)]["selected"][method]
                        apply(graph, boundary_mask(ranks[mode], size, code))
                    n = selected_graph.count_included_edges()
                    vals = measure(model, selected_graph, loader, a.cache, "test")
                row.append(
                    {"edges": n, "faithfulness": normalized(vals, full, empty), "margins": vals.tolist()}
                )
            out["curves"][mode][method] = {
                "points": row,
                "score": curve_scores([r["faithfulness"] for r in row])[mode],
            }
            print(mode, method, out["curves"][mode][method]["score"], flush=True)
            save(a.cache / "progress.json", out)
    with a.out.open("wb") as handle:
        handle.write(
            gzip.compress(json.dumps(out, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
        )


if __name__ == "__main__":
    main()
