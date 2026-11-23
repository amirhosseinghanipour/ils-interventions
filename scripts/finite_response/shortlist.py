"""Measure recovery of the strongest observed edit under a fixed site-verification budget."""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(rows):
    results = {}
    for budget in (1, 2, 3):
        methods = {}
        for row in rows:
            gold = np.abs(row["gold"])
            def ranked(name):
                return np.argsort(-np.abs(row["estimates"][name]), kind="stable")
            mid = ranked("global_midpoint")
            blend = ranked("endpoint_blend")
            count = min(len(gold), budget)
            mixed = list(mid[:count - 1])
            mixed.append(next(site for site in blend if site not in mixed))
            for method, sites in (("midpoint", mid[:count]), ("blend", blend[:count]), ("mixed", mixed)):
                strongest = gold[sites].max()
                methods.setdefault(method, []).append(float(strongest == gold.max()))
        results[str(budget)] = {key: int(sum(hits)) for key, hits in methods.items()}
    return {"prompts": len(rows), "hits": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = {}
    for model in ("qwen", "smol"):
        for task in ("ioi", "colors"):
            key = f"{model}-{task}"
            result[key] = evaluate(json.loads((args.directory / (key + ".json")).read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
