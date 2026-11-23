"""Transfer frozen coefficients to larger GGUF models using central finite differences."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "query_routing"))
from run import Model, chat, save_json

HASHES = {
    "4b": "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983",
    "9b": "a686d88ec1e6881f9bf161526826cd6d6874b7f0e80e0f79acf6144a132c5d7e",
}


def tokenize(model, text):
    buf = np.empty(len(text.encode()) + 8, np.int32)
    n = model.lib.tokenize(text.encode(), buf, len(buf))
    if not 0 < n <= len(buf):
        raise ValueError("Tokenizer returned an invalid length")
    return buf[:n].copy()


def examples(model, rows, articles):
    prefix, suffix = chat("FACT", "QUERY").split("Answer with only the lowercase color word.\nFACT\nQUERY")
    for r in rows:
        para = articles[r["article"]]["paragraphs"][r["paragraph"]]
        ctx = para["context"]
        pos, orig, word = r["start"], r["original"], r["replacement"]
        assert ctx[pos : pos + len(orig)] == orig
        donor = ctx[:pos] + word + ctx[pos + len(orig) :]
        qa = {q["id"]: q for q in para["qas"]}
        pair = []
        for tag, qid in (("edited_answer", r["qid"]), ("other_answer", r["other_qid"])):
            q = qa[qid]
            prompts = [
                prefix
                + "Answer the question with just the answer from the passage.\n"
                + c
                + "\nQuestion: "
                + q["question"]
                + suffix
                for c in (ctx, donor)
            ]
            clean, alt = (tokenize(model, t) for t in prompts)
            if len(clean) != len(alt) or len(clean) > 256:
                break
            diff = np.flatnonzero(clean != alt)
            span = q["answers"][0]["text"].strip()
            expanded = tokenize(model, prompts[0] + span)
            if (
                len(diff) != 1
                or len(expanded) <= len(clean)
                or not np.array_equal(expanded[: len(clean)], clean)
            ):
                break
            pair.append(
                (
                    dict(
                        article=r["article"], qid=qid, kind=tag, reference_first=int(expanded[len(clean)])
                    ),
                    clean,
                    alt,
                    int(diff[0]),
                )
            )
        if len(pair) == 2:
            yield pair


def measure(model, ids, donor, site, weights, audit):
    def run(tokens, *, capture=False, replacement=None, patch=None):
        block = np.empty((len(tokens), model.width), np.float32) if capture else None
        landmark = np.empty(model.width, np.float32)
        model.lib.configure_dense(
            None if block is None else block.ctypes.data,
            None if replacement is None else replacement.ctypes.data,
            landmark.ctypes.data,
        )
        try:
            _, state, info = model.evaluate(tokens, site, patch, 0, 0, layer=15)
        finally:
            model.lib.configure_dense(None, None, None)
        return block, landmark, state[-1].copy(), info["top_token"]

    h, z0, f0, clean_top = run(ids, capture=True)
    k, z1, f1, donor_top = run(donor, capture=True)
    delta = np.zeros_like(h)
    delta[site] = k[site] - h[site]
    complement = h + delta - k
    _, inter, exact, patch_top = run(ids, patch=delta[site])
    end = f1 - f0
    target = exact - f0
    cols = {"end": end, "p": inter - z0, "gap": inter - z1}

    def tangent(tokens, state, move, eps):
        plus = run(tokens, replacement=state + eps * move)[2]
        minus = run(tokens, replacement=state - eps * move)[2]
        return (plus - minus) / (2 * eps)

    cols["clean_jvp"] = tangent(ids, h, delta, 0.125)
    cols["donor_jvp"] = tangent(donor, k, complement, 0.125)
    diag = {}
    if audit:
        swapped = run(ids, replacement=k)
        assert swapped[3] == donor_top
        np.testing.assert_allclose(swapped[2], f1, rtol=5e-4, atol=5e-4)
        for name, t, state, move in (("clean_jvp", ids, h, delta), ("donor_jvp", donor, k, complement)):
            short = tangent(t, state, move, 0.0625)
            diag[name] = float(np.linalg.norm(short - cols[name]) / max(np.linalg.norm(cols[name]), 1e-8))
    keys = {
        "fit_donor": ["end"],
        "donor_jvp": ["end", "donor_jvp"],
        "donor_landmark": ["end", "gap"],
        "two_anchor": ["end", "donor_jvp", "clean_jvp"],
        "partial_donor_mix": ["end", "p"],
        "fit_clean_jvp": ["clean_jvp"],
    }
    predictions = {
        name: sum(w * cols[k] for w, k in zip(weights[name], cs, strict=True)) for name, cs in keys.items()
    }
    predictions.update(raw_donor=end, donor_jvp_raw=end + cols["donor_jvp"], no_change=np.zeros_like(end))
    return dict(
        clean_top=clean_top,
        exact_top=patch_top,
        donor_top=donor_top,
        source=site,
        prompt_tokens=len(ids),
        energy=float(np.sum(target.astype(float) ** 2)),
        relative_step_disagreement=diag,
        sse={name: float(np.sum((pred - target).astype(float) ** 2)) for name, pred in predictions.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("library", "model", "dataset", "manifest", "weights", "out"):
        parser.add_argument(key, type=Path)
    parser.add_argument("size", choices=HASHES)
    a = parser.parse_args()
    with a.model.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == HASHES[a.size]
    assert (
        hashlib.sha256(a.dataset.read_bytes()).hexdigest()
        == "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955"
    )
    articles = {r["title"]: r for r in json.loads(a.dataset.read_text())["data"]}
    weights = json.loads(a.weights.read_text())
    rows = [r for r in json.loads(a.manifest.read_text()) if r["split"] == "test"]
    model = Model(a.library, a.model)
    model.lib.configure_dense.argtypes = [ct.c_void_p] * 3
    model.lib.configure_dense.restype = None
    results = []
    for i, pair in enumerate(examples(model, rows, articles)):
        for meta, ids, donor, site in pair:
            results.append(dict(**meta, **measure(model, ids, donor, site, weights, i < 4)))
        if i == 23:
            break
    assert len(results) == 48
    save_json(a.out, results)
