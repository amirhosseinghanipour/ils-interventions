"""Confirmatory study of the frozen online generation-time patch scheduler.

Subcommands
  draw      text-level cohort with tokenizer gates for every listed checkpoint
  screen    development-split clean competency and direct text-edit success
  validate  hook identity, head addressing, forward accounting and pilot equivalence
  run       all confirmatory arms on the frozen test split for one checkpoint
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter, time
import unicodedata

import numpy as np

from qa_confirm_backends import CONTEXT, MAX_NEW, MAX_PROMPT, HFModel, NativeQwen

HERE = Path(__file__).resolve().parent
NUMERIC = re.compile(r"[1-9][0-9]{1,3}")
CAPITAL = re.compile(r"[A-Z][a-z]{2,14} [A-Z][a-z]{2,14}")
FIRST = ("Samuel", "Joseph", "Martin", "Thomas", "Daniel", "Henry", "Arthur", "Walter",
         "Edward", "Oliver", "Victor", "Hugo")
LAST = ("Mitchell", "Harrison", "Coleman", "Fletcher", "Bennett", "Sullivan", "Porter",
        "Warren", "Graham", "Lawson", "Whitaker", "Holloway")
SKIP_IRRELEVANT = re.compile(r"\b(?:how many|how much|when|year|percent|number)\b", re.I)
SCAN_STEPS = 6
TIMED = ("clean", "text", "full_prompt", "full_rolling", "scanned", "online")
CONTROLS = ("single_disagreement", "self_prompt", "self_rolling")


def save_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def normalized(text):
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.casefold())
    return " ".join(x for x in text.split() if x not in ("a", "an", "the"))


def match(text, gold):
    return normalized(text) in {normalized(a) for a in gold}


def bounded(text):
    return r"(?<!\w)" + re.escape(text) + r"(?!\w)"


def truth(question):
    return [answer["text"].strip() for answer in question["answers"]]


# ----------------------------------------------------------------------------- policies

class Meter:
    """Count synchronized model forwards without changing the backend."""
    def __init__(self, model):
        self.model, self.calls = model, 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.model.forward(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _done(model, tokens, token):
    """Return a stop reason after a decode call, appending the token unless it ends generation."""
    if model.is_eog(token):
        return "eog"
    tokens.append(int(token))
    if "\n" in model.text(tokens):
        return "newline"
    return None


def _result(model, tokens, stop, patched, **events):
    return dict(text=model.text(tokens).split("\n")[0].strip(), tokens=tokens, stop=stop,
                patched=patched, events=events)


def generate(model, base, donor, sites, timing, selected_step=0):
    """Pilot schedule semantics (qa_timing.generate) with stop reasons and token ids."""
    tokens, patched, stop = [], 0, "cap"
    for t in range(min(MAX_NEW, CONTEXT - max(len(base), len(donor)))):
        prefix = np.asarray(tokens, np.int32)
        current = np.concatenate((base, prefix))
        if timing == "rolling":
            patch_step = t
        elif timing == "none" or t < selected_step:
            patch_step = None
        else:
            patch_step = selected_step
        if patch_step is None:
            token = model.forward(current)
        else:
            if t == patch_step:
                model.forward(np.concatenate((donor, prefix)), model.ALL, capture=True)
                patched += 1
            token = model.forward(current, sites, position=len(base) - 1 + patch_step)
        stop = _done(model, tokens, token) or stop
        if stop != "cap":
            break
    return _result(model, tokens, stop, patched)


def detect(model, base, donor, max_steps=SCAN_STEPS):
    """First donor/recipient top-token disagreement on the clean prefix, else max-KL step."""
    prefix = np.empty(0, dtype=np.int32)
    fallback = None
    for _ in range(max_steps):
        clean_token = model.forward(np.concatenate((base, prefix)))
        clean = model.distribution()
        donor_token = model.forward(np.concatenate((donor, prefix)))
        donor_dist = model.distribution()
        gain = float(np.exp(donor_dist) @ (donor_dist - clean))
        if fallback is None or gain > fallback[1]:
            fallback = (len(prefix), gain)
        if clean_token != donor_token:
            return len(prefix), True
        if model.is_eog(clean_token):
            break
        prefix = np.append(prefix, np.int32(clean_token))
        if "\n" in model.text(prefix):
            break
    return fallback[0], False


def scanned(model, base, donor):
    """Two-pass policy: estimate the position, then restart generation (pilot 'scanned')."""
    before = getattr(model, "calls", 0)
    step, disagreement = detect(model, base, donor)
    scan_forwards = getattr(model, "calls", 0) - before
    out = generate(model, base, donor, model.ALL, "fixed" if step else "rolling", step)
    out["events"] = dict(step=step, disagreement=disagreement, scan_forwards=scan_forwards)
    return out


def online(model, base, edited):
    """Frozen online scheduler (qa_timing_online.online) with events and stop reasons."""
    tokens, stop = [], "cap"
    active, patch_step, count = "scan", None, 0
    for step in range(min(MAX_NEW, CONTEXT - max(len(base), len(edited)))):
        prefix = np.asarray(tokens, np.int32)
        recipient = np.concatenate((base, prefix))
        if active == "scan" and step == SCAN_STEPS:
            active = "none"
        if active == "scan":
            token = model.forward(recipient)
            donor = np.concatenate((edited, prefix))
            candidate = model.forward(donor, model.ALL, capture=True)
            if candidate != token:
                patch_step = step
                active = "rolling" if step == 0 else "fixed"
                token = model.forward(recipient, model.ALL)
                count += 1
        elif active == "fixed":
            token = model.forward(recipient, model.ALL, position=len(base) - 1 + patch_step)
        elif active == "rolling":
            model.forward(np.concatenate((edited, prefix)), model.ALL, capture=True)
            token = model.forward(recipient, model.ALL)
            count += 1
        else:
            token = model.forward(recipient)
        stop = _done(model, tokens, token) or stop
        if stop != "cap":
            break
    mode = {"scan": "none", "none": "none"}.get(active, active)
    return _result(model, tokens, stop, count, trigger=patch_step, mode=mode)


def decode_calls(out):
    return len(out["tokens"]) + (out["stop"] == "eog")


def expected_forwards(arm, out):
    """Exact native forward count implied by each schedule (checked on every query)."""
    L = decode_calls(out)
    if arm in ("clean", "text"):
        return L
    if arm in ("full_prompt", "single_disagreement", "self_prompt"):
        return L + out["patched"]
    if arm in ("full_rolling", "self_rolling"):
        return 2 * L
    if arm == "scanned":
        return out["events"]["scan_forwards"] + L + out["patched"]
    if arm == "online":
        s = out["events"]["trigger"]
        if s is None:
            return L + min(L, SCAN_STEPS)
        return 2 * L + 1 if s == 0 else L + s + 2
    raise KeyError(arm)


# ----------------------------------------------------------------------------- cohort

def load_exclusions(paths, original_dev):
    excluded = {a["title"] for a in json.loads(Path(original_dev).read_text())["data"]}
    sources = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            titles = {row["article"] for row in data}
        else:
            titles = set(data.get("dev", [])) | set(data.get("test", [])) | set(data.get("articles", []))
        sources[str(path)] = len(titles)
        excluded |= titles
    return excluded, sources


def make_edits(kind, span, rng):
    if kind == "numeric":
        k, j = int(rng.integers(1, 9)), int(rng.integers(1, 10))
        early = str((int(span[0]) - 1 + k) % 9 + 1) + span[1:]
        late = span[:-1] + str((int(span[-1]) + j) % 10)
        return early, late, dict(first_digit_offset=k, last_digit_offset=j)
    first, last = span.split()
    a = str(rng.choice([x for x in FIRST if x != first]))
    b = str(rng.choice([x for x in LAST if x != last]))
    return f"{a} {last}", f"{first} {b}", dict(first_word=a, last_word=b)


def text_candidates(title, article, seed, wanted, reasons):
    """Eligible (paragraph, relevant, irrelevant) triples of the wanted families, in seeded order."""
    rng = np.random.default_rng([seed, int(hashlib.sha256(title.encode()).hexdigest()[:8], 16)])
    for pi in rng.permutation(len(article["paragraphs"])):
        para = article["paragraphs"][int(pi)]
        context = para["context"]
        qas = para["qas"]
        for ri in rng.permutation(len(qas)):
            relevant = qas[int(ri)]
            spans = set(truth(relevant))
            if len(spans) != 1:
                continue
            span = next(iter(spans))
            kind = "numeric" if NUMERIC.fullmatch(span) else "name" if CAPITAL.fullmatch(span) else None
            if kind is None or kind not in wanted:
                continue
            found = list(re.finditer(bounded(span), context))
            if len(found) != 1:
                reasons.append("span_not_unique")
                continue
            early, late, params = make_edits(kind, span, np.random.default_rng(
                [seed, int(hashlib.sha256((title + relevant["id"]).encode()).hexdigest()[:8], 16)]))
            if any(re.search(bounded(x), context, re.I) for x in (early, late)):
                reasons.append("replacement_in_passage")
                continue
            start, end = found[0].span()
            edited = {name: context[:start] + value + context[end:]
                      for name, value in (("early", early), ("late", late))}
            for ii in rng.permutation(len(qas)):
                irrelevant = qas[int(ii)]
                if irrelevant["id"] == relevant["id"]:
                    continue
                other = truth(irrelevant)
                if any(x in a for a in other for x in (span, early, late)):
                    continue
                joined = relevant["question"] + " " + irrelevant["question"]
                if re.search(r"\b(?:" + "|".join(map(re.escape, (span, early, late))) + r")\b",
                             joined, re.I):
                    continue
                if SKIP_IRRELEVANT.search(irrelevant["question"]):
                    continue
                if {normalized(a) for a in other} & {normalized(span)}:
                    continue
                # Answerability audit: every annotated irrelevant span lies outside the edit and
                # some irrelevant alias survives verbatim in both edited passages.
                overlaps = any(a["answer_start"] < end and start < a["answer_start"] + len(a["text"])
                               for a in irrelevant["answers"])
                if overlaps:
                    reasons.append("irrelevant_overlaps_edit")
                    continue
                if not all(any(a in passage for a in other) for passage in edited.values()):
                    reasons.append("irrelevant_not_answerable")
                    continue
                yield dict(article=title, paragraph=int(pi), kind=kind,
                           qids=[relevant["id"], irrelevant["id"]],
                           questions=[relevant["question"], irrelevant["question"]],
                           source=span, replacement=dict(early=early, late=late),
                           edit_params=params, char_span=[start, end],
                           gold=[sorted(spans), other])
                break  # one irrelevant partner per relevant question


def token_gate(model, case, article):
    context = article["paragraphs"][case["paragraph"]]["context"]
    start, end = case["char_span"]
    contexts = dict(original=context, **{
        v: context[:start] + case["replacement"][v] + context[end:] for v in ("early", "late")})
    prompts = {name: [model.prompt(text, q) for q in case["questions"]] for name, text in contexts.items()}
    ids = {name: [model.ids(p) for p in pair] for name, pair in prompts.items()}
    if max(len(x) for pair in ids.values() for x in pair) > MAX_PROMPT:
        return None, "prompt_too_long"
    base = ids["original"][0]
    answers = {}
    for name, text in (("original", case["source"]), ("early", case["replacement"]["early"]),
                       ("late", case["replacement"]["late"])):
        joint = model.ids(prompts["original"][0] + text)
        if len(joint) <= len(base) or not np.array_equal(joint[:len(base)], base):
            return None, "answer_boundary"
        answers[name] = joint[len(base):].tolist()
    if answers["original"][0] == answers["early"][0]:
        return None, "early_shares_first_token"
    common = 0
    for a, b in zip(answers["original"], answers["late"]):
        if a != b:
            break
        common += 1
    if not 0 < common < min(len(answers["original"]), len(answers["late"])):
        return None, "late_no_shared_prefix"
    digest = {name: [hashlib.sha256(np.asarray(x, np.int32).tobytes()).hexdigest()[:16] for x in pair]
              for name, pair in ids.items()}
    return dict(late_gold_step=common, answer_tokens=answers,
                prompt_lengths={k: [len(x) for x in v] for k, v in ids.items()},
                input_sha256_16=digest), None


def draw(args):
    data = args.dataset.read_bytes()
    articles = {a["title"]: a for a in json.loads(data)["data"]}
    excluded, sources = load_exclusions(args.exclude, args.original_dev)
    tokenizers = {}
    for spec in args.tokenizer:
        label, kind, path = spec.split("=", 2)[0], *spec.split("=", 2)[1].split(":", 1)
        tokenizers[label] = (NativeQwen(args.library, path, vocab_only=True) if kind == "native"
                             else _tokenizer_only(path))
    rng = np.random.default_rng(args.seed)
    order = [str(t) for t in rng.permutation(sorted(set(articles) - excluded))]
    quota = {"numeric": args.per_kind, "name": args.per_kind}
    cohort, log = [], []
    for title in order:
        wanted = {k for k, v in quota.items() if v > 0}
        if not wanted:
            break
        reasons, gate_failures, accepted = [], [], None
        for case in text_candidates(title, articles[title], args.seed, wanted, reasons):
            gates, failed = {}, None
            for label, model in tokenizers.items():
                gate, why = token_gate(model, case, articles[title])
                if gate is None:
                    failed = f"{label}:{why}"
                    break
                gates[label] = gate
            if failed:
                gate_failures.append(failed)
                continue
            case["tokenization"] = gates
            accepted = case
            break
        entry = dict(article=title, text_rejections=sorted(set(reasons)),
                     token_gate_failures=gate_failures)
        if accepted is None:
            entry["outcome"] = "no_text_candidate" if not gate_failures else "token_gate"
            log.append(entry)
            continue
        quota[accepted["kind"]] -= 1
        cohort.append(accepted)
        entry.update(outcome="accepted", kind=accepted["kind"])
        log.append(entry)
    dev_left = {"numeric": args.dev_per_kind, "name": args.dev_per_kind}
    for case in cohort:
        case["split"] = "dev" if dev_left[case["kind"]] > 0 else "test"
        dev_left[case["kind"]] -= case["split"] == "dev"
    counts = {split: {k: sum(c["split"] == split and c["kind"] == k for c in cohort)
                      for k in quota} for split in ("dev", "test")}
    save_json(args.out, dict(dataset_sha256=hashlib.sha256(data).hexdigest(), seed=args.seed,
                             excluded_articles=len(excluded), exclusion_sources=sources,
                             candidate_articles=len(order), counts=counts,
                             tokenizers=list(tokenizers), cases=cohort, draw_log=log))
    print(json.dumps(counts), len(log), "articles examined", flush=True)


class _tokenizer_only:
    def __init__(self, path):
        from transformers import AutoTokenizer
        path, *opts = path.split("@")
        self.style = opts[1] if len(opts) > 1 else "v0"
        self.tok = AutoTokenizer.from_pretrained(path)

    prompt = HFModel.prompt
    ids = HFModel.ids


# ----------------------------------------------------------------------------- measurement

def load_model(spec, library):
    kind, path = spec.split(":", 1)
    if kind == "native":
        return NativeQwen(library, path)
    path, *opts = path.split("@")
    dtype = opts[0] if opts else "bfloat16"
    style = opts[1] if len(opts) > 1 else "v0"
    return HFModel(path, dtype=dtype, style=style)


def case_inputs(model, case, article, variant):
    context = article["paragraphs"][case["paragraph"]]["context"]
    start, end = case["char_span"]
    edited = context[:start] + case["replacement"][variant] + context[end:]
    base = [model.ids(model.prompt(context, q)) for q in case["questions"]]
    donor = [model.ids(model.prompt(edited, q)) for q in case["questions"]]
    return base, donor


def run_arm(model, arm, base, donor, step=None):
    meter = Meter(model)
    calls = {
        "clean": lambda: generate(meter, base, donor, (), "none"),
        "text": lambda: generate(meter, donor, donor, (), "none"),
        "full_prompt": lambda: generate(meter, base, donor, model.ALL, "fixed", 0),
        "full_rolling": lambda: generate(meter, base, donor, model.ALL, "rolling"),
        "scanned": lambda: scanned(meter, base, donor),
        "online": lambda: online(meter, base, donor),
        "single_disagreement": lambda: generate(meter, base, donor, model.ALL, "fixed", step),
        "self_prompt": lambda: generate(meter, base, base, model.ALL, "fixed", 0),
        "self_rolling": lambda: generate(meter, base, base, model.ALL, "rolling"),
    }
    start = perf_counter()
    out = calls[arm]()
    out["ms"] = (perf_counter() - start) * 1000
    out["forwards"] = meter.calls
    expected = expected_forwards(arm, out)
    assert meter.calls == expected, (arm, meter.calls, expected, out)
    return out


def measure_query(model, base, donor, rng, repeats):
    step, _ = detect(model, base, donor)          # untimed, only to place the control patch
    arms, timing_repeats = {}, []
    for name in rng.permutation(TIMED + CONTROLS):
        arms[str(name)] = run_arm(model, str(name), base, donor, step)
    assert arms["scanned"]["events"]["step"] == step
    for _ in range(repeats):
        again = {}
        for name in rng.permutation(TIMED):
            out = run_arm(model, str(name), base, donor, step)
            again[str(name)] = dict(ms=out["ms"], same=out["tokens"] == arms[str(name)]["tokens"])
        timing_repeats.append(again)
    clean = arms["clean"]["tokens"]
    identity = {n: arms[n]["tokens"] == clean for n in ("self_prompt", "self_rolling")}
    return dict(arms=arms, timing_repeats=timing_repeats, self_identity=identity, scan_step=step)


def run(args):
    cohort = json.loads(args.cohort.read_text())
    protocol = json.loads(args.protocol.read_text())
    articles = {a["title"]: a for a in json.loads(args.dataset.read_bytes())["data"]}
    assert hashlib.sha256(args.dataset.read_bytes()).hexdigest() == cohort["dataset_sha256"]
    cases = [c for c in cohort["cases"] if c["split"] == args.split]
    if args.limit:
        cases = cases[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    model = load_model(args.model, args.library)
    load_ms = (perf_counter() - t0) * 1000
    label = args.label
    rows_path = args.out / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    assert [r["article"] for r in rows] == [c["article"] for c in cases[:len(rows)]]
    # Shared warm-up, excluded from every arm: one clean and one patched generation.
    t0 = perf_counter()
    b, d = case_inputs(model, cases[0], articles[cases[0]["article"]], "early")
    generate(model, b[0], d[0], (), "none")
    generate(model, b[0], d[0], model.ALL, "rolling")
    warm_ms = (perf_counter() - t0) * 1000
    every = protocol["timing_repeat_every"]
    provenance = dict(label=label, model=args.model, describe=model.describe(), split=args.split,
                      protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                      cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
                      dataset_sha256=cohort["dataset_sha256"], seed=args.seed,
                      load_ms=load_ms, warmup_ms=warm_ms, articles=[c["article"] for c in cases],
                      python=sys.version.split()[0], started_unix=time(),
                      host_load=os.getloadavg())
    if model.backend == "transformers":
        import torch, transformers
        provenance.update(torch=torch.__version__, transformers=transformers.__version__,
                          gpu=torch.cuda.get_device_name(0))
    save_json(args.out / f"provenance-{len(rows)}.json", provenance)
    for index in range(len(rows), len(cases)):
        case = cases[index]
        article = articles[case["article"]]
        repeats = protocol["timing_repeats"] if index % every == 0 else 0
        record = dict(article=case["article"], kind=case["kind"], qids=case["qids"],
                      source=case["source"], gold=case["gold"], index=index,
                      host_load=os.getloadavg(), variants={})
        for v_index, variant in enumerate(("early", "late")):
            base, donor = case_inputs(model, case, article, variant)
            queries = []
            for q in range(2):
                rng = np.random.default_rng([args.seed, index, v_index, q])
                queries.append(measure_query(model, base[q], donor[q], rng, repeats))
            record["variants"][variant] = dict(
                replacement=case["replacement"][variant],
                gold_step=0 if variant == "early" else case["tokenization"][label]["late_gold_step"],
                input_sha256_16=[hashlib.sha256(x.tobytes()).hexdigest()[:16] for x in base + donor],
                queries=queries)
        rows.append(record)
        save_json(rows_path, rows)
        print(index + 1, len(cases), case["kind"], case["article"], flush=True)


# ----------------------------------------------------------------------------- screening

def screen(args):
    cohort = json.loads(args.cohort.read_text())
    articles = {a["title"]: a for a in json.loads(args.dataset.read_bytes())["data"]}
    cases = [c for c in cohort["cases"] if c["split"] == "dev"]
    model = load_model(args.model, args.library)
    rows = []
    for case in cases:
        article = articles[case["article"]]
        row = dict(article=case["article"], kind=case["kind"], clean=[], text={})
        base, _ = case_inputs(model, case, article, "early")
        row["clean"] = [generate(model, b, b, (), "none")["text"] for b in base]
        for variant in ("early", "late"):
            _, donor = case_inputs(model, case, article, variant)
            row["text"][variant] = [generate(model, d, d, (), "none")["text"] for d in donor]
        clean_ok = [match(a, g) for a, g in zip(row["clean"], case["gold"])]
        row["clean_correct"] = all(clean_ok)
        row["clean_each"] = clean_ok
        row["text_success"] = {
            v: bool(match(t[0], [case["replacement"][v]]) and match(t[1], case["gold"][1]))
            for v, t in row["text"].items()}
        rows.append(row)
        print(case["article"], row["clean_correct"], row["text_success"], flush=True)
    n = len(rows)
    clean = sum(r["clean_correct"] for r in rows)
    usable = sum(r["clean_correct"] and all(r["text_success"].values()) for r in rows)
    summary = dict(model=args.model, articles=n, clean_correct=clean,
                   clean_query_accuracy=sum(sum(r["clean_each"]) for r in rows) / (2 * n),
                   irrelevant_clean_accuracy=sum(r["clean_each"][1] for r in rows) / n,
                   text_success_variants=sum(sum(r["text_success"].values()) for r in rows),
                   usable_articles=usable)
    save_json(args.out, dict(summary=summary, rows=rows))
    print(json.dumps(summary), flush=True)


# ----------------------------------------------------------------------------- validation

def validate(args):
    """Inference path, donor capture, self-donor identity, head addressing, accounting."""
    cohort = json.loads(args.cohort.read_text())
    articles = {a["title"]: a for a in json.loads(args.dataset.read_bytes())["data"]}
    cases = [c for c in cohort["cases"] if c["split"] == "dev"][:args.cases]
    model = load_model(args.model, args.library)
    report = dict(model=args.model, describe=model.describe(), checks=[])
    for case in cases:
        article = articles[case["article"]]
        for variant in ("early", "late"):
            base, donor = case_inputs(model, case, article, variant)
            for q in range(2):
                rng = np.random.default_rng([7, q])
                measured = measure_query(model, base[q], donor[q], rng, 1)
                arms = measured["arms"]
                # Logit-level identity of the self-donor at the prompt boundary.
                model.forward(base[q])
                clean = model.distribution()
                model.forward(base[q], model.ALL, capture=True)
                model.forward(base[q], model.ALL)
                same = model.distribution()
                # Capture determinism: two captures of the donor are bitwise equal.
                layer = model.ALL[len(model.ALL) // 2][0]
                model.forward(donor[q], model.ALL, capture=True)
                first = model.read(layer, 0)
                model.forward(donor[q], model.ALL, capture=True)
                second = model.read(layer, 0)
                report["checks"].append(dict(
                    article=case["article"], variant=variant, query=q,
                    self_identity=measured["self_identity"],
                    self_logit_max_abs=float(np.abs(np.exp(clean) - np.exp(same)).max()),
                    capture_bitwise=bool(np.array_equal(first, second)),
                    repeat_identity=all(v["same"] for r in measured["timing_repeats"] for v in r.values()),
                    forwards={k: v["forwards"] for k, v in arms.items()},
                    patched={k: v["patched"] for k, v in arms.items()},
                    stops={k: v["stop"] for k, v in arms.items()},
                    texts={k: v["text"] for k, v in arms.items()},
                    online=arms["online"]["events"], scanned=arms["scanned"]["events"]))
    report["addressing"] = addressing(model, cases, articles)
    report["passed"] = dict(
        self_identity=all(all(c["self_identity"].values()) for c in report["checks"]),
        self_logits_exact=all(c["self_logit_max_abs"] == 0 for c in report["checks"]),
        capture_bitwise=all(c["capture_bitwise"] for c in report["checks"]),
        repeat_identity=all(c["repeat_identity"] for c in report["checks"]),
        noop_identity=report["addressing"]["noop_identical"])
    if args.pilot_rows:
        report["pilot_equivalence"] = pilot_equivalence(model, args)
    save_json(args.out, report)
    print(json.dumps(report["passed"]), json.dumps(report["addressing"]["summary"]), flush=True)


def addressing(model, cases, articles):
    """Show that each (layer, head) slice is the per-head input of the output projection."""
    case = cases[0]
    base, donor = case_inputs(model, case, articles[case["article"]], "early")
    base, donor = base[0], donor[0]
    pos = len(base) - 1
    layers = sorted({layer for layer, _ in model.ALL})
    probe_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    results, noop = [], True
    # No-op: an empty mask with a capture in between leaves logits unchanged.
    model.forward(base)
    reference = model.distribution()
    model.forward(donor, model.ALL, capture=True)
    model.forward(base, ())
    noop = bool(np.array_equal(reference, model.distribution()))
    for layer in probe_layers:
        model.forward(base, [(layer, h) for h in range(model.heads)], capture=True,
                      out_layers=[layer])
        r, out0 = model.read(layer, 0), model.read(layer, 1)
        model.forward(donor, [(layer, h) for h in range(model.heads)], capture=True)
        d = model.read(layer, 0)
        full = [(layer, h) for h in range(model.heads)]
        D = model.head_dim
        for head in (0, model.heads // 2, model.heads - 1):
            deltas = {}
            for alpha in (1.0, 2.0):
                v = r.copy()
                sl = slice(head * D, (head + 1) * D)
                v[sl] = r[sl] + alpha * (d[sl] - r[sl])
                model.write(layer, v)
                model.forward(base, full, position=pos, out_layers=[layer])
                deltas[alpha] = model.read(layer, 1) - out0
            # Patch the same head through the mask path with the donor buffer.
            model.write(layer, d)
            model.forward(base, [(layer, head)], position=pos, out_layers=[layer])
            masked = model.read(layer, 1) - out0
            entry = dict(layer=layer, head=head,
                         delta_norm=float(np.linalg.norm(deltas[1.0])),
                         linearity_rel=float(np.linalg.norm(deltas[2.0] - 2 * deltas[1.0]) /
                                             max(np.linalg.norm(deltas[1.0]), 1e-12)),
                         mask_vs_write_rel=float(np.linalg.norm(masked - deltas[1.0]) /
                                                 max(np.linalg.norm(deltas[1.0]), 1e-12)))
            if hasattr(model, "projection"):
                W = model.projection(layer)
                predicted = W[:, sl] @ (d[sl] - r[sl]).astype(np.float64)
                wrong = W[:, slice(((head + 1) % model.heads) * D, ((head + 1) % model.heads + 1) * D)]
                entry["projection_rel"] = float(np.linalg.norm(deltas[1.0] - predicted) /
                                                max(np.linalg.norm(predicted), 1e-12))
                entry["wrong_head_rel"] = float(np.linalg.norm(
                    deltas[1.0] - wrong @ (d[sl] - r[sl])) / max(np.linalg.norm(predicted), 1e-12))
                entry["reconstruct_rel"] = float(np.linalg.norm(W @ r - out0) /
                                                 max(np.linalg.norm(out0), 1e-12))
            results.append(entry)
    summary = dict(max_linearity_rel=max(e["linearity_rel"] for e in results),
                   max_mask_vs_write_rel=max(e["mask_vs_write_rel"] for e in results),
                   min_delta_norm=min(e["delta_norm"] for e in results))
    if "projection_rel" in results[0]:
        summary.update(max_projection_rel=max(e["projection_rel"] for e in results),
                       min_wrong_head_rel=min(e["wrong_head_rel"] for e in results),
                       max_reconstruct_rel=max(e["reconstruct_rel"] for e in results))
    return dict(noop_identical=noop, layers=probe_layers, entries=results, summary=summary)


def pilot_equivalence(model, args):
    """Replay recorded pilot queries with the original functions and the instrumented ones."""
    import qa_timing
    import qa_timing_online
    from qa_subset import prompt as pilot_prompt
    rows = json.loads(Path(args.pilot_rows).read_text())
    articles = {a["title"]: a for a in json.loads(args.dataset.read_bytes())["data"]}
    out = []
    for pair in rows[:args.pilot_cases]:
        for row in pair:
            article = articles[row["article"]]
            for para in article["paragraphs"]:
                questions = {q["id"]: q["question"] for q in para["qas"]}
                if set(row["qids"]) <= questions.keys():
                    break
            context = para["context"]
            edited = re.sub(bounded(row["source"]), lambda _: row["replacement"], context)
            for q, qid in enumerate(row["qids"]):
                assert pilot_prompt(context, questions[qid]) == model.prompt(context, questions[qid])
                base = model.ids(model.prompt(context, questions[qid]))
                donor = model.ids(model.prompt(edited, questions[qid]))
                pm = qa_timing_online.Meter(model)
                old_online = qa_timing_online.online(pm, base, donor)
                old_online_calls = pm.calls
                pm = qa_timing_online.Meter(model)
                old_roll = qa_timing.generate(pm, base, donor, model.ALL, "rolling")
                old_roll_calls = pm.calls
                new_online = run_arm(model, "online", base, donor)
                new_roll = run_arm(model, "full_rolling", base, donor)
                out.append(dict(
                    article=row["article"], variant=row["edit_position"], query=q,
                    recorded_online=row["outputs"]["online"][q], recorded_rolling=row["outputs"]["full_rolling"][q],
                    old_online=list(old_online) + [old_online_calls], old_rolling=list(old_roll) + [old_roll_calls],
                    new_online=[new_online["text"], new_online["patched"], new_online["forwards"]],
                    new_rolling=[new_roll["text"], new_roll["patched"], new_roll["forwards"]],
                    recorded_forwards=[row["forwards"]["online"][q], row["forwards"]["full_rolling"][q]]))
    ok = all(e["old_online"] == e["new_online"] and e["old_rolling"] == e["new_rolling"] and
             e["new_online"][0] == e["recorded_online"] and e["new_rolling"][0] == e["recorded_rolling"] and
             [e["new_online"][2], e["new_rolling"][2]] == e["recorded_forwards"] for e in out)
    return dict(identical=ok, queries=len(out), entries=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("draw")
    p.add_argument("dataset", type=Path)
    p.add_argument("original_dev", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--tokenizer", action="append", required=True,
                   help="LABEL=native:GGUF or LABEL=hf:DIR (repeatable)")
    p.add_argument("--exclude", action="append", type=Path, default=[])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--per-kind", type=int, required=True)
    p.add_argument("--dev-per-kind", type=int, required=True)
    for name, fn in (("screen", screen), ("validate", validate), ("run", run)):
        q = sub.add_parser(name)
        q.add_argument("cohort", type=Path)
        q.add_argument("dataset", type=Path)
        q.add_argument("out", type=Path)
        q.add_argument("--model", required=True, help="native:GGUF or hf:DIR[@dtype]")
        q.add_argument("--library", type=Path)
        q.set_defaults(fn=fn)
        if name == "validate":
            q.add_argument("--cases", type=int, default=2)
            q.add_argument("--pilot-rows", type=Path)
            q.add_argument("--pilot-cases", type=int, default=2)
        if name == "run":
            q.add_argument("--protocol", type=Path, required=True)
            q.add_argument("--label", required=True)
            q.add_argument("--split", default="test")
            q.add_argument("--seed", type=int, required=True)
            q.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.command == "draw":
        draw(args)
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
