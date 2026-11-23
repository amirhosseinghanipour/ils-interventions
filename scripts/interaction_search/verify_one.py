"""Post-primary sensitivity: pay for one proposed triple and score its actual completion time."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .policies import TRIPLES, propose


FALLBACK = 7
METHODS = ("ils", "linear", "midpoint", "atp", "ig5", "greedy", "exhaustive", "fallback")
ALLOWANCES = (1.0, 2.0)


def read(path):
    return json.loads(path.read_text())


def subset(rows, task):
    if task == "colors":
        chosen = list(enumerate(rows[:12]))
    else:
        counts = Counter()
        chosen = []
        for n, r in enumerate(rows):
            if counts[r["family"]] < 2:
                chosen.append((n, r))
                counts[r["family"]] += 1
    if len(chosen) != 12:
        raise ValueError("expected twelve prompts per timing cell")
    return chosen


def run_method(oracle, name, seed, pairs, fallback, now):
    if name == "fallback":
        return dict(proposed=FALLBACK, unbounded_choice=FALLBACK, ms=0.0, paid=[])
    payments = []
    start = now()

    def measure(masks, role="fit"):
        ids = list(masks)
        values = oracle.exact(ids)
        payments.append(dict(role=role, masks=ids, values=values))
        return values

    if name == "exhaustive":
        ids = list(TRIPLES)
        values = measure(ids, "verify")
        j = int(np.argmax(values))
        proposed, value = ids[j], values[j]
    else:
        grad = "midpoint" if name in ("ils", "linear") else name
        gradients = {} if name == "greedy" else {grad: oracle.gradient(grad)}
        proposal = propose(name, gradients, measure, seed=seed, max_pairs=pairs)
        proposed = int(proposal.ranking[0])
        value = measure([proposed], "verify")[0]
    # The same previously paid fallback is available even when this procedure overruns.
    choice = proposed if value >= fallback else FALLBACK
    return dict(proposed=proposed, unbounded_choice=choice, ms=now() - start, paid=payments)


def collect(inputs, artifacts, records, name):
    from .data import data, model, save
    from .run import now, row_oracle

    m, tok = model(inputs, name)
    refs = read(artifacts / "manifest.json")
    sites = read(artifacts / ("panel-" + name + ".json"))["sites"]
    pairs = read(records / "frozen.json")["max_pairs"]
    cap = read(records / ("budget-" + name + ".json"))["cap_ms"]
    bank = {f: data(inputs, f) for f in ("ioi-test.json", "colors-validation.jsonl")}
    for task in ("ioi", "colors"):
        rows = []
        for n, r in subset(refs[task], task):
            oracle, labels = row_oracle(r, bank, tok, m, sites)
            start = now()
            fallback = oracle.exact([FALLBACK])[0]
            fb_ms = now() - start
            order = np.random.default_rng(881350 + n).permutation(METHODS).tolist()
            methods = {k: run_method(oracle, k, 225177 + n, pairs, fallback, now) for k in order}
            rows.append(
                dict(
                    **r,
                    labels=labels,
                    original_row=n,
                    fallback_value=fallback,
                    common_ms=oracle.common_ms,
                    fallback_ms=fb_ms,
                    methods=methods,
                )
            )
        save(
            records / f"one-check-{name}-{task}.json",
            dict(cap_ms=cap, sites=sites, max_pairs=pairs, rows=rows),
        )
        print("single verification", name, task, len(rows), flush=True)


def score_cell(timing, primary, task, expected_cap, expected_pairs):
    cap = timing["cap_ms"]
    if timing["sites"] != primary["sites"] or cap != expected_cap or timing["max_pairs"] != expected_pairs:
        raise ValueError("timing configuration differs from the primary run")
    targets = list(TRIPLES)
    per = {m: [] for m in METHODS}
    for row, (n, ref) in zip(timing["rows"], subset(primary["rows"], task), strict=True):
        if row["original_row"] != n or any(row[k] != ref[k] for k in ("file", "index", "family", "labels")):
            raise ValueError("timing prompt differs from manifest-based subsample")
        gold = np.asarray(ref["gold"])
        top = np.asarray(ref["top"])
        fallback = row["fallback_value"]
        np.testing.assert_allclose(fallback, gold[FALLBACK], atol=8e-4, rtol=3e-4)
        if set(row["methods"]) != set(METHODS):
            raise ValueError("incomplete timing methods")
        for m, v in row["methods"].items():
            paid = v["paid"]
            proposed = v["proposed"]
            elapsed = v["ms"]
            if not np.isfinite(elapsed) or elapsed < 0 or proposed not in TRIPLES:
                raise ValueError("invalid timing or triple")
            if m == "fallback":
                if paid or proposed != FALLBACK or elapsed != 0 or v["unbounded_choice"] != FALLBACK:
                    raise ValueError("invalid shared fallback")
            else:
                expected_verify = targets if m == "exhaustive" else [proposed]
                if paid[-1]["role"] != "verify" or paid[-1]["masks"] != expected_verify:
                    raise ValueError("verification masks differ from proposed policy")
                for batch in paid[:-1]:
                    if batch["role"] != "fit" or any(k in targets for k in batch["masks"]):
                        raise ValueError("a target triple was used for fitting")
                for b in paid:
                    np.testing.assert_allclose(b["values"], gold[b["masks"]], atol=8e-4, rtol=3e-4)
                verification = paid[-1]
                if m == "exhaustive" and proposed != targets[int(np.argmax(verification["values"]))]:
                    raise ValueError("exhaustive policy did not select its measured best")
                val = verification["values"][expected_verify.index(proposed)]
                if v["unbounded_choice"] != (proposed if val >= fallback else FALLBACK):
                    raise ValueError("final choice disagrees with the timely measurements")
                if (
                    m != "exhaustive"
                    and [i for b in paid[:-1] for i in b["masks"]] != ref["methods"][m]["calibration_masks"]
                ):
                    raise ValueError("diagnostic used different calibration masks from the primary policy")
            optimum = float(gold[targets].max())
            best = v["unbounded_choice"]
            original, donor = ref["labels"]

            def measure_choice(choice):
                return dict(
                    regret=optimum - float(gold[choice]),
                    optimum=float(gold[choice] == optimum),
                    switch=float(ref["clean_top"] == original and top[choice] == donor),
                )

            per[m].append(
                dict(
                    ms=elapsed,
                    end_to_end_ms=elapsed + row["common_ms"] + row["fallback_ms"],
                    uncapped_choice=measure_choice(best),
                    raw_proposal=measure_choice(proposed),
                    deadlines={
                        str(int(scale)): dict(
                            completed=float(elapsed <= cap * scale),
                            **measure_choice(best if elapsed <= scale * cap else FALLBACK),
                        )
                        for scale in ALLOWANCES
                    },
                )
            )
    result = dict(prompts=len(timing["rows"]), cap_ms=cap, methods={})
    for m, rows in per.items():

        def averages(values):
            return {k: float(np.mean([v[k] for v in values])) for k in values[0]}

        times = np.array([r["ms"] for r in rows])
        result["methods"][m] = dict(
            mean_ms=float(times.mean()),
            median_ms=float(np.median(times)),
            mean_end_to_end_ms=float(np.mean([r["end_to_end_ms"] for r in rows])),
            uncapped_choice=averages([r["uncapped_choice"] for r in rows]),
            raw_proposal=averages([r["raw_proposal"] for r in rows]),
            deadlines={
                str(int(scale)): averages([r["deadlines"][str(int(scale))] for r in rows])
                for scale in ALLOWANCES
            },
        )
    return result


def summarize(directory):
    report = {}
    pairs = read(directory / "frozen.json")["max_pairs"]
    for name in ("qwen", "large"):
        cap = read(directory / f"budget-{name}.json")["cap_ms"]
        for task in ("ioi", "colors"):
            measured = read(directory / f"one-check-{name}-{task}.json")
            primary = read(directory / f"primary-{name}-{task}.json")
            report[f"{name}-{task}"] = score_cell(measured, primary, task, cap, pairs)
    return dict(allowance_multipliers=ALLOWANCES, cells=report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    collect_args = sub.add_parser("collect")
    for name in ("inputs", "artifacts", "records"):
        collect_args.add_argument(name, type=Path)
    collect_args.add_argument("model", choices=("qwen", "large"))
    score_args = sub.add_parser("summarize")
    score_args.add_argument("directory", type=Path)
    score_args.add_argument("out", type=Path)
    args = p.parse_args()
    if args.action == "collect":
        collect(args.inputs, args.artifacts, args.records, args.model)
    else:
        args.out.write_text(json.dumps(summarize(args.directory), indent=2, allow_nan=False) + "\n")
