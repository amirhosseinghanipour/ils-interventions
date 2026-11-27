"""Compare one-position timing and self-donor controls on fresh numeric QA."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from natural import save_json
from qa_subset import ALL, SubsetProbe, fresh_cases
from qa_timing import divergent_context, generate, summarize


COMPARISONS = [
    ("disagreement", "prompt"),
    ("disagreement", "previous"),
    ("disagreement", "next"),
    ("disagreement", "self"),
    ("gold_prefix", "disagreement"),
    ("hybrid", "full_rolling"),
]


def edit_answer(answer):
    if re.fullmatch(r"[1-9][0-9]{1,3}", answer):
        digit = (int(answer[-1])+3) % 10
        return answer[:-1] + str(digit)
    return None


def edit_name(answer):
    if re.fullmatch(r"[A-Z][a-z]{2,14} [A-Z][a-z]{2,14}", answer):
        first, last = answer.split()
        return f"{'Daniel' if first!='Daniel' else 'Robert'} {last}"
    return None


def measure(model, case):
    outputs, counts, positions = {}, {}, []
    gold_step = len(case["common_prefix"])
    for query, (base, donor) in enumerate(zip(*case["ids"])):
        context = divergent_context(model, base, donor)
        step = context[-2]
        positions.append(step)
        # The oracle uses answer labels on the affected query alone. All other policies
        # including the self-donor control share the same disagreement-step detector.
        arms = {
            "clean": ("none", 0, False),
            "prompt": ("fixed", 0, True),
            "previous": ("fixed", max(0, step-1), True),
            "disagreement": ("fixed", step, True),
            "next": ("fixed", step+1, True),
            "self": ("fixed", step, False),
            "gold_prefix": ("fixed", gold_step if query == 0 and case["kind"]=="numeric" else step, True),
            "full_rolling": ("rolling", 0, True),
            "hybrid": ("fixed", step, True) if step else ("rolling", 0, True),
        }
        cache = {}
        for arm, config in arms.items():
            mode, position, edited = config
            if config not in cache:
                cache[config] = generate(model, base, donor if edited else base, ALL, mode, position)
            answer, count = cache[config]
            outputs.setdefault(arm, []).append(answer)
            counts.setdefault(arm, []).append(count)
        assert outputs["clean"][-1] == outputs["self"][-1]
        edited_answer, _ = generate(model, donor, donor, (), "none")
        outputs.setdefault("text", []).append(edited_answer)
        counts.setdefault("text", []).append(0)

    return {**{key:case[key] for key in ("article","qids","source","replacement","gold")},
            "kind": case["kind"], "outputs": outputs, "counts": counts, "steps": positions,
            "gold_step": gold_step, "interaction": []}


def report(rows):
    # A next-position patch can only be delivered if the unpatched generation
    # reaches it. The primary comparison uses that shared support on both queries.
    numeric = [row for row in rows if row["kind"]=="numeric"]
    late_matched = [row for row in numeric if row["steps"][0]>0 and row["counts"]["next"]==[1,1]]
    return dict(
        sampled=len(rows), late_matched_articles=len(late_matched),
        numeric_articles=len(numeric),
        selected_matches_gold=sum(row["steps"][0]==row["gold_step"] for row in numeric),
        patched_positions={arm:sum(sum(row["counts"][arm]) for row in rows)
                           for arm in ("hybrid","full_rolling","disagreement")},
        overall=summarize(rows, COMPARISONS),
        late_matched=summarize(late_matched, COMPARISONS) if late_matched else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","dataset","original_dev","pilot_rows","out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--n", type=int, default=24, help="Number of numeric articles.")
    parser.add_argument("--names", type=int, default=12, help="Number of capitalized-span articles.")
    parser.add_argument("--seed", type=int, default=516633)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.read_bytes()
    articles = {a["title"]:a for a in json.loads(dataset)["data"]}
    excluded = {a["title"] for a in json.loads(args.original_dev.read_text())["data"]}
    excluded |= {a["article"] for a in json.loads(args.pilot_rows.read_text())}
    for path in args.exclude:
        split = json.loads(path.read_text())
        excluded.update(split["dev"] + split["test"])
    model = SubsetProbe(args.library, args.model)
    cases = fresh_cases(articles, excluded, model, args.n, seed=args.seed, edit_answer=edit_answer)
    if len(cases) != args.n:
        raise RuntimeError(f"Expected {args.n} eligible text cases, found {len(cases)}")
    for case in cases:
        case["kind"] = "numeric"
    excluded.update(case["article"] for case in cases)
    if args.names:
        names = fresh_cases(articles, excluded, model, args.names, seed=args.seed+1, edit_answer=edit_name)
        if len(names) != args.names:
            raise RuntimeError(f"Expected {args.names} eligible name cases, found {len(names)}")
        for case in names:
            case["kind"] = "name"
        cases.extend(names)
    provenance = dict(dataset_sha256=hashlib.sha256(dataset).hexdigest(), model=args.model.name,
                      seed=args.seed, dev=[], test=[case["article"] for case in cases])
    destination = args.out/"rows.json"
    rows = json.loads(destination.read_text()) if destination.exists() else []
    assert [r["qids"] for r in rows] == [case["qids"] for case in cases[:len(rows)]]
    save_json(args.out/"provenance.json", provenance)
    for case in cases[len(rows):]:
        rows.append(measure(model,case))
        save_json(destination,rows)
        print(len(rows), case["article"], flush=True)
    save_json(args.out/"result.json", report(rows))


if __name__ == "__main__":
    main()
