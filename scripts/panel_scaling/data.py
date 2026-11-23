"""Deterministic fresh gift-assignment and travel-direction prompt pairs for panel scaling."""

import argparse
import hashlib
import json
import random
from itertools import combinations
from pathlib import Path


PEOPLE = "Alice Bob Sam Rose Mark James Mary Anna David Lucy Paul Tony John Tom George Julia".split()
GIFTS = (
    "watch book scarf painting lamp hat camera candle ring coat vase mirror mug wallet pillow clock".split()
)
CITIES = "Paris London Rome Tokyo Berlin Madrid Athens Cairo Chicago Boston Dallas Toronto Dublin Vienna Milan Miami".split()

# The first two variants supply development; the next six supply held-out prompts.
GIFTS_TEXT = (
    "The {left} was given to {a}, and the {right} was given to {b}. The person who got the {item} was",
    "The present for {a} was the {left}. The present for {b} was the {right}. The {item} was a present for",
    "Gift labels: the {left} is for {a}; the {right} is for {b}. The label on the {item} names",
    "At the party, the host handed the {left} to {a} and the {right} to {b}. The {item} went to",
    "Someone packed a {left} for {a} and a {right} for {b}. In this parcel list the {item} belongs to",
    "The winners were {a}, who won the {left}, and {b}, who won the {right}. The winner of the {item} was",
    "On the wrapping paper she wrote {a} on the {left} and {b} on the {right}. The name on the {item} was",
    "The shop sent the {left} to {a}. It sent the {right} to {b}. According to the receipt it sent the {item} to",
)
# The two suffixes ask for the city in text position 0 or 1, respectively.
ROUTE_TEXT = (
    ("The trip ran from {a} to {b}.", "Its starting point was", "Its endpoint was"),
    (
        "The itinerary lists {a} as the origin and {b} as the destination.",
        "The journey began in",
        "The journey ended in",
    ),
    ("The plane took off in {a}, then landed in {b}.", "The departure city was", "The arrival city was"),
    (
        "The traveler left {a} before reaching {b}.",
        "The first city on the trip was",
        "The last city on the trip was",
    ),
    (
        "The ticket says: depart from {a}, arrive in {b}.",
        "The ticket holder will leave from",
        "The ticket holder will arrive in",
    ),
    ("The relay started in {a} and finished in {b}.", "The starting line was in", "The finish line was in"),
    (
        "After leaving {a}, the pilot flew straight to {b}.",
        "The city the pilot left was",
        "The city the pilot flew to was",
    ),
    (
        "The postcard reads: I traveled from {a} to {b}.",
        "The writer set out from",
        "The writer's destination was",
    ),
)


def randomized_pairs(items, seed):
    rng = random.Random(seed)
    pairs = list(combinations(items, 2))
    rng.shuffle(pairs)
    return [list(reversed(pair)) if rng.randrange(2) else list(pair) for pair in pairs]


def make_rows():
    rows = []
    for task, templates, bank, seed in (
        ("gift", GIFTS_TEXT, PEOPLE, 872151),
        ("route", ROUTE_TEXT, CITIES, 872152),
    ):
        answer_pairs = randomized_pairs(bank, seed)
        gift_pairs = randomized_pairs(GIFTS, 872153)
        for i, template in enumerate(templates):
            for j in range(4):
                k = 4 * i + j
                target = (i + j) % 2
                answers = answer_pairs[k]
                if task == "gift":
                    items = gift_pairs[k]
                    text = template.format(
                        left=items[0], right=items[1], item=items[target], a="{a}", b="{b}"
                    )
                else:
                    text = template[0] + " " + template[target + 1]
                row = {
                    "id": f"{task}-{i:02d}-{j}",
                    "task": task,
                    "template": f"{task}-{i:02d}",
                    "split": "development" if i < 2 else "heldout",
                    "clean": text.format(a=answers[0], b=answers[1]),
                    "donor": text.format(a=answers[1], b=answers[0]),
                    "original_answer": answers[target],
                    "donor_answer": answers[1 - target],
                    "entity_pair": answers,
                    "target_position": target,
                }
                if task == "gift":
                    row["gift_pair"] = items
                rows.append(row)
    return rows


def encode(tokenizer, row):
    """Check that swapping the two names/cities exchanges exactly two token IDs."""
    prompts = [row["clean"], row["donor"]]
    ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    if len(ids[0]) != len(ids[1]):
        raise ValueError(f"{row['id']}: unequal prefix lengths")
    labels = []
    for word in (row["original_answer"], row["donor_answer"]):
        continuations = []
        for p, pre in zip(prompts, ids):
            full = tokenizer.encode(p + " " + word, add_special_tokens=False)
            if len(full) != len(pre) + 1 or full[:-1] != pre:
                raise ValueError(f"{row['id']}: ambiguous or multitoken continuation {word}")
            continuations.append(full[-1])
        if continuations[0] != continuations[1]:
            raise ValueError(f"{row['id']}: context-dependent answer token")
        labels.append(continuations[0])
    changed = [(a, b) for a, b in zip(*ids) if a != b]
    if labels[0] == labels[1] or sorted(changed) != sorted([tuple(labels), tuple(reversed(labels))]):
        raise ValueError(f"{row['id']}: donor is not a two-token exchange of the answers")
    return ids, labels


def make_manifest(tokenizer_dirs):
    from transformers import AutoTokenizer

    rows = make_rows()
    tokenizers = {}
    for name, path in tokenizer_dirs.items():
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        lengths = [len(encode(tokenizer, row)[0][0]) for row in rows]
        tokenizers[name] = {
            "tokenizer_json_sha256": hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest(),
            "rows_validated": len(rows),
            "min_prefix_tokens": min(lengths),
            "max_prefix_tokens": max(lengths),
        }
    return {"tokenizers": tokenizers, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("qwen", "smol", "gpt2", "out"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    manifest = make_manifest({name: getattr(args, name) for name in ("qwen", "smol", "gpt2")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
