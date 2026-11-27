"""Replay a per-query timing policy against the two recorded QA policy outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from natural import save_json
from qa_subset import normalized
from qa_timing import summarize


ARMS = ("clean", "text", "full_prompt", "full_delayed", "full_rolling")
COMPARISONS = [("hybrid", "full_rolling"), ("hybrid", "full_delayed"),
               ("full_rolling", "full_delayed")]


def replay(rows):
    records, switches = [], Counter()
    for row in rows:
        outputs = {arm: row["outputs"][arm] for arm in ARMS}
        counts = {arm: row["counts"][arm] for arm in ARMS}
        # Query order and answer labels play no role in selecting the policy.
        selected = ["full_delayed" if step>0 else "full_rolling" for step in row["steps"]]
        outputs["hybrid"] = [outputs[arm][query] for query,arm in enumerate(selected)]
        counts["hybrid"] = [counts[arm][query] for query,arm in enumerate(selected)]
        switches.update(selected)
        records.append(dict(row, outputs=outputs, counts=counts, interaction=[]))
    return records, switches


def aggregate(rows):
    records, switches = replay(rows)
    result = summarize(records, COMPARISONS)
    eligible = [r for r in records if all(
        normalized(answer) in {normalized(label) for label in gold}
        for answer,gold in zip(r["outputs"]["clean"],r["gold"]))]
    result["patched_positions"] = {
        key: {scope:sum(sum(r["counts"][key]) for r in cohort)
              for scope,cohort in (("all_articles",records),("clean_correct",eligible))}
        for key in ("hybrid","full_rolling","full_delayed")
    }
    result["policy_counts_all_queries"] = dict(switches)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    cohorts = {name:json.loads((args.root/name/"rows.json").read_text()) for name in ("4b","9b")}
    source_articles = [row["article"] for rows in cohorts.values() for row in rows]
    if len(source_articles) != len(set(source_articles)):
        raise ValueError("Replay requires disjoint article sets")
    summary = {name:aggregate(rows) for name,rows in cohorts.items()}
    summary["pooled"] = aggregate([row for rows in cohorts.values() for row in rows])
    save_json(args.out,summary)


if __name__ == "__main__":
    main()
