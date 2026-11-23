# Finite-response benchmark outputs

| Evaluation | Public records | Recompute |
| --- | --- | --- |
| Single-position partial replay, 48 calibration and 80 test articles | [Fixed articles](articles.json), [weights](weights.json), [verification](verification.json), per-checkpoint rows and decoded token IDs in this directory | `scripts/finite_response/verify.py` |
| Donor-centered follow-up, 32 separate calibration articles and 64 test articles with two questions each | [Article pairs](paired-articles.json), [Qwen-fitted weights](paired/weights.json), [FP32 results](paired/summary.json), [marginal timings](paired/timing.json) | `scripts/finite_response/paired_score.py` |
| Quantized Qwen3.5-4B and 9B, 24 eligible test articles with two questions each | [First-position results](paired/native-summary.json) | `scripts/finite_response/native_score.py` |
| Released MIB IOI and four-choice CopyColors test counterfactuals, 50 prompts per task and model | [Site comparisons](sites/summary.json), per-position effects and estimates under [sites](sites/) | `scripts/finite_response/sites.py` |

Residual and logit-margin errors have different targets. Each normalized error divides squared prediction error by measured squared effect; smaller is better. Text agreement compares the predictor's own capped greedy generation to the exact patched model, using SQuAD normalization. SQuAD answer correctness, token agreement, and text agreement with the patched model are distinct quantities. MIB prompts here evaluate single-position residual patches at one depth.

[Runtime](runtime.json) and [reproduction commands, data sources, and licenses](../../scripts/finite_response/README.md).

The [native step-size sensitivity results](paired/native-sensitivity.json) repeat all native articles at three finite steps. [SHA256SUMS](SHA256SUMS) covers all JSON data files in this directory and its subdirectories.

[Shortlist reanalysis](sites/shortlist.json) counts prompts where the inspected sites include a site with maximal absolute measured effect. Each method inspects one, two, or three distinct positions; estimated ties use sequence order. The mixed rule fills the first slots from midpoint-gradient ranks and the remaining slot from the endpoint blend. These are retrospective hit counts on the four existing test cells. They exclude the time spent computing the rankings. Recompute with `python scripts/finite_response/shortlist.py artifacts/finite-response/sites artifacts/finite-response/sites/shortlist.json`.
