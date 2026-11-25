# Sixteen-head measurements

For a given prompt, all three models use the same 512 bitmasks. The head identities are checkpoint-specific. Held-out inputs comprise 24 gift and 24 route prompt pairs at each checkpoint. The sampled masks differ between prompts. All times below include recipient and donor capture. A zero budget uses the midpoint gradient alone. An exact fallback evaluates all 512 interventions and has zero approximation error.

## Fixed 0.10-logit tolerance

| Checkpoint | Task | Target (logits) | Sparse budget | Sparse MAE | Sparse ms | Dense budget | Dense MAE | Dense ms | Exact ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | gift | 0.10000 | 0 | 0.01374 | 21.0 | 0 | 0.01374 | 21.0 | 185.6 |
| GPT-2 small | route | 0.10000 | 0 | 0.01336 | 14.4 | 0 | 0.01336 | 14.4 | 134.8 |
| Qwen2.5-0.5B | gift | 0.10000 | 0 | 0.00372 | 64.1 | 0 | 0.00372 | 64.1 | 614.9 |
| Qwen2.5-0.5B | route | 0.10000 | 0 | 0.01289 | 42.4 | 0 | 0.01289 | 42.4 | 458.0 |
| SmolLM2-1.7B | gift | 0.10000 | 0 | 0.01685 | 116.8 | 0 | 0.01685 | 116.8 | 2335.5 |
| SmolLM2-1.7B | route | 0.10000 | 0 | 0.03350 | 105.1 | 0 | 0.03350 | 105.1 | 1695.5 |

## Half the development midpoint error

These tolerances were defined after the first two models selected zero calibration on development, before any held-out run. Each cell keeps its selected tolerance and budgets across all six held-out templates.

| Checkpoint | Task | Target (logits) | Sparse budget | Sparse MAE | Sparse ms | Dense budget | Dense MAE | Dense ms | Exact ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | gift | 0.00910 | 256 | 0.00531 | 147.4 | 256 | 0.00361 | 146.7 | 207.0 |
| GPT-2 small | route | 0.00454 | exact | 0.00000 | 149.9 | 256 | 0.00363 | 109.4 | 149.9 |
| Qwen2.5-0.5B | gift | 0.00170 | exact | 0.00000 | 623.7 | 256 | 0.00079 | 389.9 | 623.7 |
| Qwen2.5-0.5B | route | 0.00563 | 64 | 0.00648 | 110.4 | 64 | 0.00737 | 110.5 | 459.7 |
| SmolLM2-1.7B | gift | 0.01578 | 96 | 0.00974 | 658.6 | 96 | 0.01038 | 658.4 | 2350.2 |
| SmolLM2-1.7B | route | 0.02153 | 96 | 0.01868 | 493.2 | 256 | 0.00757 | 1030.9 | 1700.4 |

## Files

- [Frozen benchmark](benchmark.json), [raw prompt pairs and tokenizer identities](manifest.json) and [collection and scoring instructions](../../scripts/panel_scaling/README.md).
- [Primary score and uncertainty](summary.json), [tighter-target timing and accuracy](half-summary.json), [prompt-specific stopping](adaptive-summary.json) and [equal-budget errors](equal-budget.json).
- The three development files include the chosen panels and exact calibration and evaluation masks' effects on development. The heldout files contain exact outcomes, gradients and synchronized timings. The half files contain the additional tighter-target timing runs. The adaptive files store the validation measurements, stopping stages, returned predictions and exact controls.
- [SHA-256 checksums](SHA256SUMS) cover all JSON files.
