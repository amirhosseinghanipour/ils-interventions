"""Score the organizer's separate submission-example artifact after the frozen six-method run."""

import argparse
import gzip
import json
from functools import partial
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from eap.evaluate import evaluate_baseline
from eap.graph import Graph
from MIB_circuit_track.dataset import collate_EAP
from MIB_circuit_track.metrics import logit_diff
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.mib_official.audit_masks import mask_hash
from scripts.mib_official.public_ioi import (
    GRID,
    MODES,
    curve_scores,
    digest,
    local_ioi,
    measure,
    normalized,
    rank_edges,
    read,
    save,
)

REVISION = "2cc3324f6d3780f5d82fe741cd2fd3e20fc1643b"
SOURCE = f"https://huggingface.co/mib-bench/mib-circuits-example/blob/{REVISION}/importances/pt/ioi_gpt2/importances.pt"
SHA256 = "d3f552d5d2dfdeb73f5f4b2aeb7426e4e1fa02fd62711cb12c24342bf48b5c2c"
KEY = "organizer_hf_example"


def compare_files(github_path, hf_path):
    g = Graph.from_json(str(github_path))
    h = Graph.from_pt(str(hf_path))
    assert g.cfg == h.cfg and torch.equal(g.real_edge_mask, h.real_edge_mask)
    assert h.neurons_in_graph is None and h.count_included_edges() == h.count_included_nodes() == 0
    mask = g.real_edge_mask
    x, y = g.scores[mask].double().numpy(), h.scores[mask].double().numpy()
    assert np.isfinite(y).all()
    overlap = {}
    for mode in MODES:
        a = rank_edges(g, mode == "CMD").numpy()
        b = rank_edges(h, mode == "CMD").numpy()
        overlap[mode] = {}
        for fraction in GRID[:-1]:
            k = int(fraction * len(g.edges))
            overlap[mode][str(fraction)] = float(np.intersect1d(a[:k], b[:k]).size / k)
    return {
        "legal_edges": len(g.edges),
        "finite_hf_scores": len(y),
        "unique_hf_scores": int(np.unique(y).size),
        "preselected_hf_edges": h.count_included_edges(),
        "preselected_hf_nodes": h.count_included_nodes(),
        "signed_pearson": float(np.corrcoef(x, y)[0, 1]),
        "signed_spearman": float(spearmanr(x, y).statistic),
        "top_k_overlap_fraction_before_pruning": overlap,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path)
    p.add_argument("primary", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    hf_path = a.inputs / "published" / "organizer-example.pt"
    warm_path = a.inputs / "upstream" / "circuits" / "ioi-eap-ig.json"
    weights = a.inputs / "base-gpt2"
    test_path = a.inputs / "test.json"
    primary = json.loads(gzip.decompress(a.primary.read_bytes()))
    assert primary["grid"] == list(GRID) and digest(hf_path) == SHA256
    assert digest(test_path) == primary["sha256"]["test"]
    assert digest(weights / "model.safetensors") == primary["sha256"]["weights"]
    assert digest(warm_path) == primary["sha256"]["eap"]
    fingerprint = {
        "primary_sha256": digest(a.primary),
        "hf_scores_sha256": SHA256,
        "script_sha256": digest(Path(__file__)),
        "native_measure_module_sha256": digest(Path(measure.__code__.co_filename)),
        "batch": 32,
    }
    a.cache.mkdir(parents=True, exist_ok=True)
    marker = a.cache / "fingerprint.json"
    if marker.exists():
        assert read(marker) == fingerprint, "Inputs changed; use a new cache"
    else:
        save(marker, fingerprint)

    provenance = compare_files(warm_path, hf_path)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(weights, local_files_only=True)
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
    ds = local_ioi(test_path, tok)
    assert list(ds.dataset["template"]) == primary["test_templates"]
    data = DataLoader(ds, batch_size=32, collate_fn=collate_EAP)
    first = DataLoader([ds[i] for i in range(32)], batch_size=32, collate_fn=collate_EAP)
    g = Graph.from_pt(str(hf_path))
    full = np.array(primary["test_full"])
    empty = np.array(primary["test_empty"])
    clean_check = evaluate_baseline(model, first, partial(logit_diff, mean=False), quiet=True).numpy()
    g.apply_topn(0)
    empty_check = measure(model, g, first, a.cache, "empty32")
    g.apply_topn(len(g.edges))
    full_check = measure(model, g, first, a.cache, "full32")
    for check, expected in ((clean_check, full), (empty_check, empty), (full_check, full)):
        np.testing.assert_allclose(check, expected[:32], rtol=1e-5, atol=5e-4)

    out = {
        "scope": "post-primary organizer HF submission example, public GPT-2 IOI",
        "method": KEY,
        "source": SOURCE,
        "revision": REVISION,
        "fingerprint": fingerprint,
        "graph_comparison": provenance,
        "endpoint_check_prompts": 32,
        "grid": GRID,
        "curves": {},
    }
    mask_runtimes = {}
    native_full = np.array(primary["curves"]["CPR"]["eap_ig"]["points"][-1]["margins"])
    for mode in MODES:
        points = []
        for fraction in GRID:
            if fraction == 1.0:
                values, count = native_full, len(g.edges)
            else:
                g.apply_topn(int(fraction * len(g.edges)), absolute=mode == "CMD")
                count = g.count_included_edges()
                key = mask_hash(g)
                timing = a.cache / ("timing-" + key + ".json")
                already_measured = (a.cache / ("test-" + key + ".json")).exists()
                started = monotonic()
                values = measure(model, g, data, a.cache, "test")
                elapsed = monotonic() - started
                if not already_measured:
                    save(timing, elapsed)
                mask_runtimes[key] = read(timing) if timing.exists() else None
                print(mode, fraction, round(elapsed, 1), flush=True)
            points.append(
                {
                    "edges": count,
                    "faithfulness": normalized(values, full, empty),
                    "margins": values.tolist(),
                }
            )
        out["curves"][mode] = {
            KEY: {"points": points, "score": curve_scores([p["faithfulness"] for p in points])[mode]}
        }
        print(mode, KEY, out["curves"][mode][KEY]["score"], flush=True)
    recorded = [t for t in mask_runtimes.values() if t is not None]
    out["runtime"] = {
        "scope": "nontrivial native public test graphs; excludes setup and endpoint parity checks",
        "unique_graphs": len(mask_runtimes),
        "timed_unique_graphs": len(recorded),
        "seconds_by_graph_sha256": mask_runtimes,
        "median_seconds": float(np.median(recorded)) if recorded else None,
        "sum_seconds": sum(recorded),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(
        gzip.compress(json.dumps(out, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
    )


if __name__ == "__main__":
    main()
