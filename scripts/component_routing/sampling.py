"""Pinned MIB problem selection and common next-answer-token boundaries."""

import hashlib
import json

import numpy as np

FILES = {
    "addition-validation": "407448e75fad3daf20f6ae38313d356bb07e12926daf0952d76b93098bf003a1",
    "addition-test": "84373e6ee8d3eb440bb29ee365e20de58d8b700ba0869b763b64eb478752d954",
    "subtraction-test": "0826ad98e453ba4c1c75ec3f33bc3d3aafd55e9e11e55f3609b96374d279e1b5",
}
WEIGHTS = {
    "qwen": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "large": "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
}


def encode(tok, row, counter):
    cf = row[counter]
    prompts = [row["prompt"], cf["prompt"]]
    answers = [str(row["label"]), str(cf["label"])]
    inputs, choices = [], []
    for prompt in prompts:
        prefix = tok.encode(prompt.rstrip(), add_special_tokens=False)
        continuations = [tok.encode(prompt + s, add_special_tokens=False) for s in answers]
        if any(ids[: len(prefix)] != prefix for ids in continuations):
            return None
        pos = next((j for j, (a, b) in enumerate(zip(*continuations)) if a != b), None)
        if pos is None or pos < len(prefix):
            return None
        inputs.append(continuations[0][:pos])
        choices.append([ids[pos] for ids in continuations])
    if len(inputs[0]) != len(inputs[1]) or inputs[0] == inputs[1] or choices[0] != choices[1]:
        return None
    return inputs, choices[0]


def select(directory, toks):
    rng = np.random.default_rng(104307)
    manifest = {}
    occupied = set()
    for key, expected in FILES.items():
        file = directory / (key + ".jsonl")
        assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
        rows = [json.loads(s) for s in file.read_text().splitlines()]
        wanted = 20 if key.endswith("validation") else 48
        chosen = []
        for index in rng.permutation(len(rows)):
            row = rows[index]
            counter = ("ones_op1_counterfactual", "tens_op2_counterfactual")[len(chosen) % 2]
            cf = row.get(counter)
            if not cf:
                continue
            pair = {(key.split("-")[0], str(q["operand1"]), str(q["operand2"])) for q in (row, cf)}
            if occupied & pair or not all(encode(t, row, counter) for t in toks):
                continue
            chosen.append(dict(index=int(index), counter=counter, template=row["template"]))
            occupied.update(pair)
            if len(chosen) == wanted:
                break
        assert len(chosen) == wanted, (key, len(chosen))
        manifest[key] = chosen
    return manifest
