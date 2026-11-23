# Arithmetic intervention measurements

| Files | Contents |
| --- | --- |
| [Manifest](manifest.json), [frozen coefficient](frozen.json), [development measurements](qwen-addition-validation.json) | Selected source rows, donor pairing and coefficient calibration |
| [Qwen addition](qwen-addition-test.json), [Qwen subtraction](qwen-subtraction-test.json), [SmolLM2 addition](large-addition-test.json), [SmolLM2 subtraction](large-subtraction-test.json) | Every signed native patch effect, full-vocabulary top token, raw estimate, and timed verification batch |
| [Qwen allowances](qwen-budget.json), [SmolLM2 allowances](large-budget.json), [scores](summary.json) | Budget calibration, raw and budgeted outcomes, and joint holdout comparisons |
| [Group manifest](group-manifest.json), [Qwen groups](groups-qwen.json), [SmolLM2 groups](groups-large.json) | Disjoint prompt pairs, seeded head subsets, exact group effects, and nine prespecified predictors |
| [Qwen batch timings](amortize-qwen.json), [SmolLM2 batch timings](amortize-large.json), [diagnostics](diagnostics.json) | Actual answer switches, group errors by size, repeated-query predictions and timings |
| [Qwen cached exact timings](cached-qwen.json), [SmolLM2 cached exact timings](cached-large.json) | Second timing run allowing exact suffix reuse and up to 128 parallel masks |
| [Qwen integration and additivity](controls-qwen.json), [SmolLM2 integration and additivity](controls-large.json) | Five-point path integration, sum of exact singleton effects, and marginal timings |
| [Verification](verification.json), [runtime](runtime.json), [checksums](SHA256SUMS) | Counts, software and input provenance, and file integrity |

Each single-site row lists components by zero-based block and head index. At each block the last component index denotes its MLP. Effects are changes in the donor-minus-recipient answer-token margin. In the scoring tables, “increase” and “decrease” refer to the original correct-answer margin. Directional recovery averages both and gives uniform credit for ties. MAE is measured in logits. A lower MAE and a higher recovery are preferable. Timings are milliseconds. For full-vocabulary behavioral counts the source must initially predict its answer token; a switch requires the exact patched model's top vocabulary token to equal the donor answer token. Fractional switch counts represent expected credit for tied choices.

The group study scores five interior masks per prompt. Its simultaneous intervals compare the smooth two-endpoint predictor with eight comparators across four model-task cells, resampling whole prompts and preserving the shared arithmetic inputs across checkpoints. The throughput study separately evaluates four of those prompts per checkpoint. Speedups include the common capture cost and charge each method for its own setup. Oracle-generation time in the single-component files includes a full-vocabulary readout for behavioral scoring. Online single-component verification and both sides of the throughput comparison read only the two target logits.

Run the following commands from the repository root. Supply the original input files to repeat the operand-disjointness and source-hash checks. The public verifier also runs without local model weights or datasets.

    python scripts/component_routing/verify.py artifacts/component-routing --inputs INPUTS
    python scripts/component_routing/score.py artifacts/component-routing /tmp/component-summary.json
    python scripts/component_routing/diagnose.py artifacts/component-routing /tmp/component-diagnostics.json
    (cd artifacts/component-routing && sha256sum -c SHA256SUMS)

[Model revisions, dataset sources, estimators and collection commands](../../scripts/component_routing/README.md).
