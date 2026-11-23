# Predicting finite edits from partial decoder replay

A source-token replacement at half depth induces a change `p` at three-quarter depth. Let `z` and `y` be the clean donor-minus-recipient differences at that answer position at three-quarter and final depth. Project `p` onto `z` with coefficient `a`. Predict the final residual change as

    a y + g (||y|| / ||z||) (p - a z).

The scalar g is least-squares fitted on first answer positions from 48 Qwen articles. The two clean trajectories and a partial intervened trajectory are required per example. When a donor landmark difference vanishes, the residual term uses unit scale. The estimator uses no backward pass. Clean donor answer positions after the first share the recipient's generated prefix.

The fixed manifest selects single-token, same-length source substitutions for both tokenizers in the original SQuAD v1.1 training split. Source words occur once in the passage. Replacements come from a different annotated answer in the same paragraph. The 80 test articles are disjoint from calibration and identical on both models. SmolLM supplies no training effects. Single-word selection, short context, English questions and the use of annotated passage edits constrain this benchmark. Clean answer correctness is not an eligibility criterion.

Report squared residual prediction error on the first answer token divided by the observed effect energy, article-bootstrap intervals, first-token agreement, and whether any token of the greedy answer changes through EOS or eight tokens. Scoring all clean prefixes detects the first divergence exactly for a deterministic decoder. A changed answer need not be a correctly answered question. The public rows include direct autoregressive checks for eight held-out articles per model. Timings on these articles exclude generation, the clean trajectories and the vocabulary head.

## Reproduction

Use CUDA, PyTorch, Transformers, and NumPy. From the repository root set `PYTHONPATH=.`. Place the original SQuAD `train-v1.1.json` in an input directory, with the Hugging Face files of Qwen/Qwen2.5-0.5B-Instruct (revision `7ae557604adf67be50417f59c2c2f167def9a775`) under `qwen/` and HuggingFaceTB/SmolLM2-360M-Instruct (revision `a10cc1512eabd3dde888204e902eca88bddb4951`) under `smol/`. Collection verifies the weight and dataset hashes.

    python scripts/finite_response/sample.py INPUTS articles.json
    cmp articles.json artifacts/finite-response/articles.json
    python scripts/finite_response/collect.py INPUTS articles.json CACHE qwen
    python scripts/finite_response/collect.py INPUTS articles.json CACHE smol
    python scripts/finite_response/score.py INPUTS articles.json CACHE RESULTS

Data from Rajpurkar et al., [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/), CC BY-SA 4.0. Article metadata and derived data retain [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The single-anchor formula is a rank-one Broyden secant update to a scaled identity. Controls include the classical identity initialization, unit norm-scaled gain, projection alone and the exact directional derivative at the clean and half-edited states. A secondary estimator uses all preceding clean token differences within the same prompt. It regularizes their projected landmark Gram matrix with a unit diagonal penalty while preserving the answer-position secant. A corresponding unconstrained prefix ridge predictor uses the same clean pairs. Both additions were fixed before inference on either benchmark model. No prediction sees a future answer token.

For end-to-end validation, run `python scripts/finite_response/decode.py INPUTS articles.json RESULTS` after scoring. This rolls each predictor out on its own prior tokens. The exact patched model, the unchanged recipient, the partial-plus-donor mixture, the midpoint derivative, and both candidate variants generate through EOS or the same eight-token cap. Public records separately report token sequence agreement, normalized text agreement, and whether the exact patch produces the replacement answer.

## Donor-centered follow-up

The follow-up draws 96 further SQuAD articles, all disjoint from the first experiment. Each pair shares the same substituted passage word. One question asks for the edited answer. The other has a different annotated answer at a nonoverlapping span. Both questions enter calibration on 32 Qwen articles and testing on the remaining 64 articles on each model. The same Qwen-fitted coefficients apply to SmolLM.

Let h and k be the clean recipient and donor sequence states at half depth, and x the hybrid state carrying a donor token at just the source position. F maps a sequence state to the final answer-position residual. The two-anchor estimator fits three scalar gains to the clean donor difference F(k)−F(h), the recipient-side tangent DF(h)(x−h), and the donor-side tangent DF(k)(x−k). Inference requires two clean runs and two directional derivatives. It does not require a forward run at x. A cheaper comparison corrects the donor estimate with the donor-relative discrepancy from partial replay. Midpoint controls query intermediate states and are reported separately in the results. The classical rank-one update in the first experiment follows [Broyden (1965)](https://doi.org/10.1090/S0025-5718-1965-0198670-6).

    python scripts/finite_response/paired.py INPUTS articles.json paired-articles.json
    cmp paired-articles.json artifacts/finite-response/paired-articles.json
    python scripts/finite_response/anchors.py INPUTS paired-articles.json PAIRED_RESULTS
    python scripts/finite_response/paired_score.py PAIRED_RESULTS paired-articles.json articles.json paired-summary.json

All checks compare predictions to exact patches. Residual error is pooled squared error divided by pooled squared effect size (zero prediction scores 1). Each of 10,000 bootstrap draws resamples articles with both questions kept together and the same draws across models. For the main pooled outcome the verifier constructs simultaneous 95% bands across two-anchor comparisons on both models using the bootstrap maximum absolute standardized deviation. These intervals condition on the fixed calibration and models.

The native replication evaluates first-answer-position residuals for the first 24 held-out article pairs that fit both questions in the 256-token GGUF context and have one differing source token. Both sizes must share question IDs. The finite-difference step is 0.125 in either direction. The first four article pairs also use 0.0625 to measure step sensitivity, and a whole-sequence state swap must reproduce the donor output. Report the standalone derivative results as finite-step approximations on these quantized models. This replication measures the actual emitted first tokens for the clean and patched runs. The larger models do not undergo the eight-token candidate rollout.

Use the native build command and pinned backend in `../query_transition/replicate.sh` to compile `../query_routing/probe.cpp` with the added dense-state callback. Invoke once for each GGUF, then summarize both:

    python scripts/finite_response/native.py PROBE.so MODEL.gguf INPUTS/train-v1.1.json artifacts/finite-response/paired-articles.json PAIRED_RESULTS/weights.json PAIRED_RESULTS/native-4b.json 4b
    python scripts/finite_response/native.py PROBE.so MODEL.gguf INPUTS/train-v1.1.json artifacts/finite-response/paired-articles.json PAIRED_RESULTS/weights.json PAIRED_RESULTS/native-9b.json 9b
    python scripts/finite_response/native_score.py PAIRED_RESULTS

## All-position external diagnostic

The external test uses original test prompts and supplied counterfactuals from two [MIB datasets](https://mib-bench.github.io/). It compares signed first-answer logit-margin changes for every nonzero single-position patch at half depth. This differs from MIB's official component/edge localization track. Run on the unchanged prompt text, without a chat wrapper. The clean and counterfactual answers must be different single continuation tokens under both tokenizers. Selection uses no model-output filter.

Let u be the fraction of squared whole-sequence displacement at a given position. The fixed estimator interpolates the recipient and donor first-order estimates with donor weight u²/(u²+(1−u)²). Two scalar reverse passes supply estimates for all positions. Controls include recipient-side attribution patching, a nearest-endpoint rule, a global midpoint gradient, trapezoidal averaging, and an energy-weighted secant correction. Exact batched single-position interventions supply the independent target.

Download the authors' [IOI test file](https://huggingface.co/datasets/mib-bench/ioi/blob/e5f3468f3af4c0883be35cd3bced8c711c95d286/test.json) as `ioi-test.json` and the [four-choice CopyColors test file](https://huggingface.co/datasets/mib-bench/copycolors_mcqa/blob/682676b0e80d3a80e847a21810157710c1f23e27/4_answer_choices/test.jsonl) as `colors-test.jsonl` under INPUTS, and run `python scripts/finite_response/sites.py INPUTS SITE_RESULTS`. Source licenses are MIT for IOI and Apache-2.0 for CopyColors. The raw records include source indices and measured/predicted scores.
