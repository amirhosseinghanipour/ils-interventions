# Two-block head intervention measurements

| Evidence | Contents |
| --- | --- |
| [Manifest](manifest.json), [Qwen panel](panel-qwen.json), [SmolLM2 panel](panel-large.json) | Source indices, template/object split and four selected heads per block |
| [Development](development.json), [frozen pair count](frozen.json) | Exhaustive development outcomes and the grid selection made before test collection |
| [Qwen IOI](primary-qwen-ioi.json), [Qwen CopyColors](primary-qwen-colors.json), [SmolLM2 IOI](primary-large-ioi.json), [SmolLM2 CopyColors](primary-large-colors.json) | All 256 effects and output tokens per prompt, raw scores, exact query timelines and winning masks |
| [Qwen clock](budget-qwen.json), [SmolLM2 clock](budget-large.json), [primary scores](summary.json), [audit](audit.json) | Calibration, grouped bootstrap comparisons, model-query replay and data checks |
| [Conditional effects](mechanisms.json) | Within- and across-block pair interactions across all 64 contexts of the other six heads, plus IOI-to-CopyColors transfer |
| [Qwen repeat timings](amortized-qwen.json), [SmolLM2 repeat timings](amortized-large.json), [regression ablations and speed](amortized-summary.json) | Paired timings for 1 through 222 unmeasured subsets, with an independently paid calibration per fit |
| [Calibration budget](calibration-budget.json), [cross-prompt fit](transfer-prior.json) | Nested calibration sizes and an IOI-trained regression that uses only the new prompt's gradient |
| [Single-check sensitivity](one-check-summary.json) | One proposed triple verified natively, tested at the original deadline and twice that allowance; source records are the four `one-check-{model}-{task}.json` files |

Head indices and block indices in JSON are zero-based. Bit 0 denotes the first head in the stored site order. Higher signed effects favor the donor token. Regret is the exact best three-head effect minus the effect of the selected triple, in logits. Switches require an initially correct top-ranked vocabulary token to become the donor answer. Model and version pins are in the [earlier runtime record](../component-routing/runtime.json). [Collection protocol and input sources](../../scripts/interaction_search/README.md).

The nine alternatives in the primary score include optimized exact enumeration on the same hardware. A method receives credit for timely verified decisions, with a common, paid fallback if required. Raw estimators can be inspected separately. Fitting excludes all three-head groups. The subgroup masks, per-object replicates and two checkpoints that use the same prompts are dependent. Bootstrap intervals resample IOI templates or CopyColors object IDs. All four data tables are public held-out experiments.

Recompute the primary and supplementary values from the repository root:

    python -m scripts.interaction_search.score artifacts/interaction-search /tmp/interaction-primary.json
    python -m scripts.interaction_search.mechanisms artifacts/interaction-search /tmp/interaction-mechanisms.json
    python -m scripts.interaction_search.amortized score artifacts/interaction-search /tmp/interaction-speed.json
    python -m scripts.interaction_search.verify_one summarize artifacts/interaction-search /tmp/interaction-one.json
    python -m scripts.interaction_search.calibration_budget artifacts/interaction-search /tmp/interaction-budget.json
    python -m scripts.interaction_search.transfer_prior artifacts/interaction-search /tmp/interaction-transfer.json
    python -m scripts.interaction_search.audit artifacts/interaction-search /tmp/interaction-audit.json
    (cd artifacts/interaction-search && sha256sum -c SHA256SUMS)
