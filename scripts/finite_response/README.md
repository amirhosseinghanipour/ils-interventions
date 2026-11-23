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
