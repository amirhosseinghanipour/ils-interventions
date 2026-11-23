# Sixteen-head intervention prediction

The benchmark measures the cost of predicting 512 new head substitutions on a single prompt. It uses Qwen2.5-0.5B-Instruct, SmolLM2-1.7B-Instruct and GPT-2 small. Each prompt and its donor swap two single-token names or cities. The signed target is the change from the original prompt in the donor-minus-original answer logit margin. No initial-correctness filter is applied.

The two tasks use gift recipients and route directions. Each has eight development pairs from two templates and 24 independent held-out pairs from six other templates. The same prompt pairs are used across checkpoints. The two changed tokens and both candidate answers are validated under all three tokenizers. For each model, pool both development tasks and choose eight attention heads per block by mean absolute recipient-gradient attribution. The zero-based blocks are 11 and 17 on the two instruction checkpoints and 5 and 8 on GPT-2. Replacement affects every position. A later unpatched head retains its recipient activation conditional on the earlier edits.

The 16 bits specify 65,536 possible masks. Each prompt has a seeded pool of 128 complementary calibration pairs and 512 distinct unmeasured targets, sampled uniformly without replacement from the other interior masks. The gradient is taken at all gates equal to 0.5. The residual estimators fit additive effects, up to eight pair interactions selected by regularized loss reduction, or all 120 pairs in a dense ridge regression. All fit the same measured labels at any common budget, with ridge strength 0.03 after scaling columns on calibration data.

For each task, checkpoint and method, use the smallest calibration count reaching mean absolute error at most 0.10 logits on development. An unavailable operating point runs cached exact evaluation. At a budget of zero all three regressors use midpoint alone. The exact baseline reuses common early-block states, groups related queries and selects a chunk size on a development prompt for each native workload. Timings on every held-out pair include common capture, a fresh gradient and calibration for each fitted method, and all 512 predictions. Per-prompt compute is the median of three synchronized, interleaved method calls plus common setup. Model load, tokenization, query design and development are excluded for every method.

The primary assessment is on both GPT-2 tasks. The sparse method must actually fit calibration observations on both. Six simultaneous, template-clustered lower bounds must be positive. The margins are the tolerance less mean absolute error, and mean time saved over each of the dense and exact policies, for each task. Six clusters per task limit precision. The comparison concerns these development-selected operating points.

## Reproduce

Use a CUDA-enabled PyTorch environment and Transformers with eager attention. The runner verifies the local safetensors hashes. Under INPUTS supply model directories named `qwen`, `large`, and `gpt2`. For reference the earlier [model record](../../artifacts/component-routing/runtime.json) pins the instruction models. The [GPT-2 release](https://huggingface.co/openai-community/gpt2) supplies the base checkpoint. From the repository root use a fresh OUT directory, set `PYTHONPATH=.` and set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

    python -m scripts.panel_scaling.data INPUTS/qwen INPUTS/large INPUTS/gpt2 OUT/manifest.json
    for model in gpt2 qwen large; do
        python -m scripts.panel_scaling.run INPUTS OUT/manifest.json OUT "$model" development
    done
    for model in gpt2 qwen large; do
        python -m scripts.panel_scaling.run INPUTS OUT/manifest.json OUT "$model" heldout
    done
    python -m scripts.panel_scaling.score OUT OUT/summary.json

The study is a held-out test on new wording and entity pairs. Panel selection and budget selection use development data from each checkpoint, including GPT-2.

## Prompt-specific stopping

The supplementary policy reserves the last 32 calibration masks to validate a fit on each new prompt. It checks the midpoint and then fits on 16, 32, 64, 128 and 224 nested observations. The stopping statistic is validation MAE plus one standard error over the 16 complementary pairs. Both sparse and dense regressions use this heuristic. If no stage qualifies, the policy pays for all tried stages and returns 512 new exact evaluations. Each policy and an exact-only control have fresh synchronized timing on every held-out prompt, in randomized order, with one repetition per method. Paired differences resample whole templates. The secondary joint GPT-2 success rule uses six one-sided Bonferroni-adjusted template bootstrap bounds and requires at least one positive-budget accepted sparse fit per task.

    for model in gpt2 qwen large; do
        python -m scripts.panel_scaling.adaptive collect INPUTS OUT/manifest.json OUT "$model"
    done
    python -m scripts.panel_scaling.adaptive score OUT OUT/adaptive-summary.json

The 0.10-logit target required zero calibration on GPT-2 and Qwen development. Before testing any held-out prompt we added a separate target, set at half of the development midpoint MAE for each model and task. This requires an absolute tolerance specific to each cell. We keep the original outcome and run the same budget-selection and three-repetition native timing procedure at the stricter target, including exact fallback for infeasible fits. Each timed repetition preserves its actual gradient, calibration labels and forecast. The scorer reconstructs that forecast and verifies the native measurements against the independent table. A batched remeasurement can differ slightly numerically and change which pair a sparse fit selects.

    for model in gpt2 qwen large; do
        python -m scripts.panel_scaling.half_error collect INPUTS OUT "$model"
    done
    python -m scripts.panel_scaling.half_error score OUT OUT/half-summary.json
