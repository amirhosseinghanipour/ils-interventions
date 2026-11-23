"""Select fresh articles with relevant and unrelated questions for the same edit."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from sample import SQUAD_SHA, digest, get_ids, make_rows


def main():
    p = argparse.ArgumentParser()
    for key in ("inputs", "previous", "out"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    source = args.inputs / "train-v1.1.json"
    assert digest(source) == SQUAD_SHA
    data = json.loads(source.read_text())
    exclude = {r["article"] for r in json.loads(args.previous.read_text())}
    tokenizers = [AutoTokenizer.from_pretrained(args.inputs / k) for k in ("qwen", "smol")]
    data["data"] = [a for a in data["data"] if a["title"] not in exclude]
    lookup = {a["title"]: a for a in data["data"]}
    rows = []
    for row in make_rows(data, tokenizers):
        para = lookup[row["article"]]["paragraphs"][row["paragraph"]]
        forbidden = {row["original"].casefold(), row["replacement"].casefold()}
        for q in sorted(para["qas"], key=lambda q: q["id"]):
            if q["id"] == row["qid"] or any(w in q["question"].casefold() for w in forbidden):
                continue
            # Require a different, nonoverlapping annotated answer; semantic
            # independence of the second question is not assumed.
            answers = q["answers"]
            if not answers or any(
                any(w in a["text"].casefold() for w in forbidden)
                or not (
                    a["answer_start"] + len(a["text"]) <= row["start"]
                    or a["answer_start"] >= row["start"] + len(row["original"])
                )
                for a in answers
            ):
                continue
            if all(get_ids(t, row, para["context"], q["question"]) for t in tokenizers):
                rows.append(dict(row, other_qid=q["id"], split="fit" if len(rows) < 32 else "test"))
                break
        if len(rows) == 96:
            break
    if len(rows) != 96:
        raise ValueError(f"Found {len(rows)} eligible articles")
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
