"""Collect a fixed paired retrieval experiment on Qwen3.5-9B, with raw residuals."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
COLORS = "red blue green black white yellow pink orange brown purple gray gold".split()
NAMES = {
    "dev": "Ben Tom Mary Sam Jack Jane Rose Bill".split(),
    "test": "John Kate Paul Ann Jim Max Luke Mark Beth Ruth Dan Bob".split(),
}
# Each clause describes one entity and has exactly one source color.
TEMPLATES = {
    "dev": ("{name} has the {color} card.", "Which color card does {name} have?"),
    "test": ("The card for {name} is {color}.", "What color is the card for {name}?"),
}


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Model:
    def __init__(self, library, model):
        self.lib = ct.CDLL(str(library.resolve()))
        self.lib.initialize.argtypes = [ct.c_char_p]
        self.lib.tokenize.argtypes = [ct.c_char_p, I32, ct.c_int]
        self.lib.forward.argtypes = [I32, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p,
                                     F32, F32, ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
        self.width = self.lib.initialize(str(model).encode())

    def ids(self, text):
        buf = np.empty(256, dtype=np.int32)
        n = self.lib.tokenize(text.encode(), buf, len(buf))
        if not 0 < n <= len(buf):
            raise ValueError(f"Unexpected token count {n} for {text!r}")
        return buf[:n].copy()

    def evaluate(self, ids, site, edit, donor, original, *, layer=None):
        src = np.empty(self.width, np.float32)
        final = np.empty((len(ids), self.width), np.float32)
        margin = ct.c_float()
        address = None if edit is None else edit.ctypes.data
        layer = PROTOCOL["source_block"] if layer is None else layer
        top = self.lib.forward(ids, len(ids), site, layer, address,
                               src, final, donor, original, ct.byref(margin))
        if not np.isfinite(final).all() or not np.isfinite(margin.value):
            raise ValueError("Nonfinite model output")
        return src, final, {"donor_minus_original": margin.value, "top_token": top}


def chat(facts, query):
    body = f"Answer with only the lowercase color word.\n{facts}\n{query}"
    return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def design(model, split, *, seed=None, count=None, name_bank=None, template_pair=None, render_chat=chat, unique_facts=False, value_bank=COLORS):
    seed = PROTOCOL["seeds"][split] if seed is None else seed
    count = PROTOCOL["counts"][split] if count is None else count
    name_bank = NAMES[split] if name_bank is None else name_bank
    rng = np.random.default_rng(seed)
    # Only single-token entries, determined entirely from the vocabulary.
    colors = [c for c in value_bank if len(model.ids(c)) == len(model.ids(" " + c)) == 1]
    names = [n for n in name_bank if len(model.ids(n)) == len(model.ids(" " + n)) == 1]
    template, question = template_pair or TEMPLATES[split]
    seen = set()
    i = 0
    while i < count:
        people = rng.choice(names, 2, replace=False).tolist()
        first, second, replacement = rng.choice(colors, 3, replace=False).tolist()
        values = [first, second]
        facts_key = (*people, *values)
        if unique_facts and facts_key in seen:
            continue
        seen.add(facts_key)
        edited = i % 2
        def render(who, swap):
            facts = " ".join(template.format(name=n, color=replacement if swap and j == edited else values[j])
                             for j, n in enumerate(people))
            return render_chat(facts, question.format(name=people[who]))
        texts = [render(edited, False), render(1-edited, False), render(edited, True)]
        relevant, other, donor = map(model.ids, texts)
        assert len(relevant) == len(other) == len(donor)
        changed = np.flatnonzero(relevant != donor)
        cue = np.flatnonzero(relevant != other)
        assert len(changed) == len(cue) == 1 and changed[0] < cue[0]
        site = int(changed[0])
        assert relevant[site] == model.ids(" " + values[edited])[0]
        assert donor[site] == model.ids(" " + replacement)[0]
        yield dict(index=i, people=people, values=values, edited=edited, replacement=replacement,
                   texts=texts, site=site, cue=int(cue[0]), length=len(relevant),
                   tokens=[a.tolist() for a in (relevant, other, donor)],
                   correct=[int(model.ids(values[j])[0]) for j in (edited, 1-edited)],
                   donor_token=int(model.ids(replacement)[0]))
        i += 1


def collect(model, row, split, root):
    a, b, donor_ids = (np.array(x, np.int32) for x in row["tokens"])
    site = row["site"]
    def evaluate(ids, edit=None):
        return model.evaluate(ids, site, edit, row["donor_token"], row["correct"][0])
    donor_source, _, _ = evaluate(donor_ids)
    source_a, clean_a, meta_a = evaluate(a)
    source_b, clean_b, meta_b = evaluate(b)
    np.testing.assert_array_equal(source_a, source_b)
    np.testing.assert_allclose(clean_a[:row["cue"]], clean_b[:row["cue"]], rtol=0, atol=1e-6)
    direction = donor_source - source_a
    norm = np.linalg.norm(direction)
    assert norm > 0
    rng = np.random.default_rng(PROTOCOL["seeds"][split] * 100 + row["index"])
    random = rng.standard_normal(model.width).astype(np.float32)
    random *= norm / np.linalg.norm(random)
    np.testing.assert_allclose(np.linalg.norm(random), norm, rtol=1e-6)
    arrays = dict(direction=direction, random=random)
    measures = {}
    for key, ids, clean, meta in (("relevant", a, clean_a, meta_a), ("irrelevant", b, clean_b, meta_b)):
        arrays[key + "_clean"] = clean
        _, repeated, repeat_meta = evaluate(ids)
        np.testing.assert_array_equal(clean, repeated)
        assert repeat_meta == meta
        measures[key] = {"clean": meta}
        for label, edit in (("donor1", direction), ("donor005", np.float32(.05)*direction),
                            ("random1", random)):
            for sign_name, signed in (("plus", edit), ("minus", -edit)):
                _, result, obs = evaluate(ids, signed)
                np.testing.assert_allclose(result[:site], clean[:site], atol=1e-6, rtol=0)
                arrays[f"{key}_{label}_{sign_name}"] = result - clean
                measures[key][label + "_" + sign_name] = obs
    primary = contrast(measures, "donor1")
    random_effect = contrast(measures, "random1")
    correct = [m["clean"]["top_token"] == target
               for m, target in zip(measures.values(), row["correct"])]
    # Write records last, so partial arrays are never counted.
    stem = f'{split}-{row["index"]:03}'
    np.savez(root / (stem + ".npz"), **arrays)
    record = dict(**row, measurements=measures, correct_both=all(correct), correct_each=correct,
                  primary=primary, random_control=random_effect)
    save_json(root / (stem + ".json"), record)
    return record


def contrast(measures, label):
    def shift(which):
        m = measures[which]
        return m[label + "_plus"]["donor_minus_original"] - m["clean"]["donor_minus_original"]
    return shift("relevant") - shift("irrelevant")


def summarize(records):
    rng = np.random.default_rng(110023)
    n = len(records)
    draw = rng.integers(n, size=(10000, n))
    def stats(x):
        a = np.asarray(x)
        return {"mean": float(a.mean()), "interval": np.quantile(a[draw].mean(1), [.025, .975]).tolist()}
    return {"pairs": n, "both_top1": sum(r["correct_both"] for r in records),
            "individual_top1": sum(sum(r["correct_each"]) for r in records),
            "primary": stats([r["primary"] for r in records]),
            "random": stats([r["random_control"] for r in records]),
            "donor_minus_random": stats([r["primary"]-r["random_control"] for r in records])}


def main():
    p = argparse.ArgumentParser()
    for k in ("model", "library", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    args = p.parse_args()
    model = Model(args.library, args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    for split in ("dev", "test"):
        rows = list(design(model, split))
        save_json(args.output / (split + "-design.json"), rows)
        records = []
        for row in rows:
            records.append(collect(model, row, split, args.output))
            print(split, len(records), flush=True)
        summary = summarize(records)
        save_json(args.output / (split + "-summary.json"), summary)
        print(json.dumps(summary), flush=True)
        if split == "dev" and summary["both_top1"] < 6:
            print("Development capability gate failed; holdout stayed sealed.", flush=True)
            break


if __name__ == "__main__":
    main()
