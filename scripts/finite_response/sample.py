"""Construct deterministic article-disjoint source edits in original SQuAD questions."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

SQUAD_SHA = "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955"
TRAIN = 48
TEST = 80


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def encode(tok, context, question):
    content = f"Answer the question with just the answer from the passage.\n{context}\nQuestion: {question}"
    text = tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    return tok.encode(text, add_special_tokens=False)


def get_ids(tok, row, context, question):
    start, original = row["start"], row["original"]
    assert context[start : start + len(original)] == original
    alternate = context[:start] + row["replacement"] + context[start + len(original) :]
    clean, donor = [encode(tok, part, question) for part in (context, alternate)]
    sites = [i for i, (a, b) in enumerate(zip(clean, donor)) if a != b]
    if len(clean) != len(donor) or len(clean) > 248 or len(sites) != 1:
        return None
    return clean, donor, sites[0]


def make_rows(dataset, tokenizers):
    rng = random.Random(816407)
    rows = []
    articles = sorted(dataset["data"], key=lambda a: a["title"])
    rng.shuffle(articles)
    for a in articles:
        paragraphs = list(enumerate(a["paragraphs"]))
        rng.shuffle(paragraphs)
        found = False
        for pi, paragraph in paragraphs:
            context = paragraph["context"]
            qas = sorted(paragraph["qas"], key=lambda q: q["id"])
            rng.shuffle(qas)
            for q in qas:
                ans = q["answers"][0]
                word, start = ans["text"], ans["answer_start"]
                if (
                    not re.fullmatch("[A-Za-z]{3,}", word)
                    or context.count(word) != 1
                    or word.casefold() in q["question"].casefold()
                ):
                    continue
                # A single-word alternative from another answer in the same paragraph.
                for other in qas:
                    alt = other["answers"][0]["text"]
                    if (
                        not re.fullmatch("[A-Za-z]{3,}", alt)
                        or alt.casefold() == word.casefold()
                        or alt.casefold() in q["question"].casefold()
                        or alt.istitle() != word.istitle()
                    ):
                        continue
                    row = dict(
                        article=a["title"],
                        paragraph=pi,
                        qid=q["id"],
                        start=start,
                        original=word,
                        replacement=alt,
                    )
                    if all(get_ids(tok, row, context, q["question"]) for tok in tokenizers):
                        rows.append(dict(row, split="fit" if len(rows) < TRAIN else "test"))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if len(rows) == TRAIN + TEST:
            return rows
    raise ValueError(f"Only {len(rows)} eligible articles")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    source = args.inputs / "train-v1.1.json"
    assert digest(source) == SQUAD_SHA
    tokenizers = [AutoTokenizer.from_pretrained(args.inputs / m) for m in ("qwen", "smol")]
    rows = make_rows(json.loads(source.read_text()), tokenizers)
    assert len({r["article"] for r in rows}) == len(rows)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
