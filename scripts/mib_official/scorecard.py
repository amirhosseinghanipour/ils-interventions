"""Read organizer-displayed IOI GPT-2 scores and replay the older released baseline curve."""

import argparse
import hashlib
import json
from pathlib import Path

GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
DASHBOARD = "https://mib-bench-leaderboard.hf.space/config"
ARCHIVE = (
    "https://huggingface.co/spaces/mib-bench/leaderboard/raw/"
    "3f311010ee5af98bc80cadbee6b237021bd1f7dc/eval-results-mib-subgraph/baselines/EAP-IG-inputs_CF.json"
)


def extract(config, archive):
    dashboard = {}
    external_comparators = {}
    for mode, component_id in (("CPR", 10), ("CMD", 18)):
        component = next(c for c in config["components"] if c["id"] == component_id)
        assert component["type"] == "leaderboard"
        table = component["props"]["value"]
        index = table["headers"].index("IOI - GPT-2")
        external_comparators[mode] = {
            row[0]: {
                col: (row[i] if isinstance(row[i], (int, float)) else None)
                for i, col in enumerate(table["headers"])
                if col not in ("Method", "eval_name", "Average", "Score")
            }
            for row in table["data"]
            if row[0] in ("EAP-IG-inputs (CF)", "ILP+PNR+Bootstrapping", "MAttr (edge, 5k steps)")
        }
        dashboard[mode] = [
            {"release": row[0], "score": row[index]} for row in table["data"] if row[index] is not None
        ]
    scores = next(r for r in archive["results"] if r["model_id"] == "gpt2")["scores"]["ioi"]
    y = scores["faithfulness"]
    assert len(y) == len(GRID)
    auc = sum((b - a) * (c + d) / 2 for a, b, c, d in zip(GRID, GRID[1:], y, y[1:]))
    return {
        "task": "IOI - GPT-2",
        "dashboard_displayed_private_scores": dashboard,
        "selected_private_baselines_all_tasks": external_comparators,
        "archived_single_curve": {
            "release": archive["method_name"],
            "nominal_sparsities": GRID,
            "faithfulness": y,
            "released_edge_counts": scores["edge_counts"],
            "CPR_recomputed": auc,
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("archive", type=Path)
    p.add_argument("--retrieved", required=True, help="UTC snapshot date, YYYY-MM-DD")
    a = p.parse_args()
    raw = [path.read_bytes() for path in (a.config, a.archive)]
    result = extract(*(json.loads(b) for b in raw))
    result.update(
        retrieved_utc=a.retrieved,
        sources=[
            {"url": url, "sha256": hashlib.sha256(b).hexdigest()}
            for url, b in zip((DASHBOARD, ARCHIVE), raw)
        ],
    )
    print(json.dumps(result, indent=2))
