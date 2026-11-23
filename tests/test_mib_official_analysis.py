"""The public graph analysis must use paired, template-level ratios and opposite objective directions."""

import numpy as np
import pytest

from scripts.mib_official.analyze import summarize, summarize_additional_example


def test_mib_template_bootstrap_recomputes_shared_denominator_and_objective():
    # Two unequal template sizes, unequal full-minus-empty margins, and an overshooting circuit.
    labels = ["alpha", "alpha", "beta"]
    empty = np.array([-2.0, -1.0, -3.0])
    full = np.array([0.0, 3.0, 7.0])
    gap = full - empty
    grid = [0.1, 1.0]
    data = {
        "split": "test",
        "filtered_public_prompts": 3,
        "test_full": full.tolist(),
        "test_empty": empty.tolist(),
        "test_templates": labels,
        "eligible_development_prompts": 8,
        "development_indices": [0, 1],
        "development_templates": ["development"],
        "grid": grid,
        "curves": {},
    }
    for mode in ("CPR", "CMD"):
        data["curves"][mode] = {}
        for method, excess in (("pair", np.array([1.0, 3.0, 0.0])), ("linear", np.zeros(3))):
            faith = (gap.sum() + excess.sum()) / gap.sum()
            sparse, dense = (faith, 1.0) if mode == "CPR" else (abs(1 - faith), 0.0)
            data["curves"][mode][method] = {
                "points": [
                    {"edges": 1, "margins": (full + excess).tolist(), "faithfulness": faith},
                    {"edges": 10, "margins": full.tolist(), "faithfulness": 1.0},
                ],
                "score": 0.9 * (sparse + dense) / 2.0,
            }
    summary = summarize(data, draws=500, seed=111)
    assert summary["bootstrap"]["clusters"] == 2
    # Aggregate and single-template excess faithfulness are 4/16, 4/6 and 0 respectively.
    # The trapezoid gives the sparse endpoint weight 0.45.
    for mode in ("CPR", "CMD"):
        delta = summary["curves"][mode]["paired_differences"]["linear"]
        assert delta["pair_minus_comparator"] == pytest.approx(0.45 * 4 / 16)
        assert delta["interval"] == pytest.approx([0.0, 0.45 * 4 / 6])
        assert delta["leave_one_template_out_delta_range"] == pytest.approx([0.0, 0.45 * 4 / 6])
    key = "organizer_hf_example"
    supplement = {
        "method": key,
        "scope": "post-primary",
        "source": "fixture",
        "revision": "fixture",
        "grid": grid,
        "fingerprint": {"primary_sha256": "fixture"},
        "graph_comparison": {},
        "runtime": {},
        "curves": {m: {key: curves["linear"]} for m, curves in data["curves"].items()},
    }
    additional = summarize_additional_example(data, supplement)
    assert additional["bootstrap"]["draws"] == 4000
    for mode in ("CPR", "CMD"):
        new = additional["curves"][mode]["primary_pair_minus_organizer_hf_example"]
        assert new["pair_minus_comparator"] == pytest.approx(0.45 * 4 / 16)
        assert new["interval"] == pytest.approx([0.0, 0.45 * 4 / 6])
    # The same positive difference favors pair on CPR and linear on CMD.
    assert summary["curves"]["CPR"]["methods"]["linear"]["score"] == pytest.approx(0.9)
    assert summary["curves"]["CMD"]["methods"]["linear"]["score"] == pytest.approx(0.0)


def test_mib_rejects_shared_development_and_test_templates():
    data = {
        "test_full": [2.0, 3.0],
        "test_empty": [0.0, 0.0],
        "test_templates": ["shared", "other"],
        "filtered_public_prompts": 2,
        "development_templates": ["shared"],
        "grid": [0.1, 1.0],
    }
    with pytest.raises(AssertionError):
        summarize(data)
