"""Guard held-out task structure and the donor/recipient token exchange."""

import json
import re
import unittest
from collections import Counter
from pathlib import Path

from scripts.panel_scaling.data import encode, make_rows


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return re.findall(r" ?[a-zA-Z]+|[^\w\s]", text)


class PanelScalingDataTests(unittest.TestCase):
    def test_archived_rows_and_split_are_reproducible(self):
        manifest = Path(__file__).resolve().parents[1] / "artifacts/panel-scaling/manifest.json"
        rows = make_rows()
        data = json.loads(manifest.read_text())
        self.assertEqual(rows, data["rows"])
        self.assertTrue(all(x["rows_validated"] == len(rows) for x in data["tokenizers"].values()))
        self.assertEqual(set(data["tokenizers"]), {"qwen", "smol", "gpt2"})
        self.assertEqual(len(set(r["id"] for r in rows)), 64)
        self.assertEqual(len({r[c] for c in ("clean", "donor") for r in rows}), 128)
        for task in ("gift", "route"):
            subset = [r for r in rows if r["task"] == task]
            self.assertEqual(Counter(r["split"] for r in subset), {"development": 8, "heldout": 24})
            self.assertEqual(len({tuple(sorted(r["entity_pair"])) for r in subset}), 32)
            dev_templates = {r["template"] for r in subset if r["split"] == "development"}
            test_templates = {r["template"] for r in subset if r["split"] == "heldout"}
            self.assertEqual((len(dev_templates), len(test_templates)), (2, 6))
            self.assertTrue(dev_templates.isdisjoint(test_templates))
            for t in dev_templates | test_templates:
                self.assertEqual(
                    Counter(r["target_position"] for r in subset if r["template"] == t), {0: 2, 1: 2}
                )
        gifts = [r for r in rows if r["task"] == "gift"]
        self.assertEqual(len({tuple(sorted(r["gift_pair"])) for r in gifts}), 32)

    def test_encode_checks_both_contexts_and_exact_exchange(self):
        row = make_rows()[0]
        tokenizer = WordTokenizer()
        ids, labels = encode(tokenizer, row)
        self.assertEqual(len(ids[0]), len(ids[1]))
        self.assertEqual(labels, [" " + row["original_answer"], " " + row["donor_answer"]])
        with self.assertRaisesRegex(ValueError, "two-token exchange"):
            encode(tokenizer, {**row, "donor": row["clean"]})
        with self.assertRaisesRegex(ValueError, "unequal prefix lengths"):
            encode(tokenizer, {**row, "donor": row["donor"] + " today"})
        with self.assertRaisesRegex(ValueError, "multitoken continuation"):
            encode(tokenizer, {**row, "original_answer": "New York"})
