"""Download the exact public checkpoint, IOI splits and released graph files for public_ioi.py."""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

GPT2 = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
IOI = "e5f3468f3af4c0883be35cd3bced8c711c95d286"
MATTR = "3f88981654c117ef93fcab520d94f76fdc4ad765"
TECHNION = "de056ee7180bd3342a35e7fdf22dd3c1d3b54440"


def prepare(root):
    def download(repo_id, revision, name, destination, **kw):
        path = hf_hub_download(repo_id, filename=name, revision=revision, **kw)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(Path(path).resolve())

    for file in (
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model.safetensors",
    ):
        download("openai-community/gpt2", GPT2, file, root / "base-gpt2" / file)

    for split in ("dev", "test"):
        download("mib-bench/ioi", IOI, split + ".json", root / (split + ".json"), repo_type="dataset")
    dest = root / "published"
    download("aryaman/mattr-mib-circuits", MATTR, "edge/ioi_gpt2/importances.pt", dest / "mattr.pt")
    for mode in ("False", "True"):
        for frac in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            name = f"{frac}_edges.pt"
            download(
                "Technion-CS-NLP/ilp-pnr-bootstrapping-circuits",
                TECHNION,
                f"abs-{mode}/ioi_gpt2/{name}",
                dest / f"{mode}-{name}",
            )
    # TransformerLens resolves the "gpt2-small" alias to the legacy "gpt2" repository name.
    # The official checkpoint's local config is all it needs when we pass the model and tokenizer.
    hub = Path(os.environ["HF_HOME"]) / "hub" / "models--gpt2"
    refs = hub / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    ref = refs / "main"
    if not ref.exists():
        ref.write_text(GPT2)
    if ref.read_text().strip() != GPT2:
        raise ValueError("Use a dedicated HF_HOME for these pinned inputs")
    cfg = hub / "snapshots" / GPT2 / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        cfg.symlink_to((root / "base-gpt2" / "config.json").resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path)
    prepare(p.parse_args().inputs)
