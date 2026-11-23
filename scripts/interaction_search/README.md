# Conditional effects across two transformer blocks

Eight sites consist of four native heads in block 12 and four in block 18 (one-based). Select the four largest mean absolute first-order attributions per block on nine separate IOI prompts at each checkpoint. Preserve the panel across tasks. Every Boolean mask selects a subset of these sites. Zero is the unmodified recipient. At a later block, a selected head receives its full donor value and each unselected head keeps its value conditional on all earlier interventions. The target is the donor-minus-original answer logit difference, centered on the clean prompt. Top tokens use the entire vocabulary.

The public MIB [IOI test source](https://huggingface.co/datasets/mib-bench/ioi/tree/e5f3468f3af4c0883be35cd3bced8c711c95d286) has nine templates. On three templates use nine pairs for panel selection and 18 different pairs for model development. Evaluate on 48 pairs from the six remaining templates, eight per template. Exclude examples from the prior residual-site run. Task transfer uses 48 objects from the previously unused [four-choice CopyColors validation split](https://huggingface.co/datasets/mib-bench/copycolors_mcqa/tree/682676b0e80d3a80e847a21810157710c1f23e27/4_answer_choices). It shares no object identifiers with the earlier public test split. Both checkpoints receive the same pairs and there is no initial-correctness filter. Each example has two distinct single-token answer continuations under both tokenizers. Qwen2.5-0.5B-Instruct is the development checkpoint; the joint transfer target is SmolLM2-1.7B-Instruct on CopyColors.

Each prompt receives 32 exact calibration masks arranged as 16 complement pairs. These exclude size three. Linear and sparse pair fits use the identical observations and matched gradient. Ridge is fixed at 0.03 on unit-norm columns. The pair budget is selected from 0, 1, 3 and 6 by mean regret of the raw top-ranked triple on Qwen development, then held-out-triple MAE and smaller pair count for ties. A cluster of gates at 0.5 gives the matched gradient. Five-point integrated gradients average the gate derivatives at 0.1, 0.3, 0.5, 0.7 and 0.9. Serving a new prompt includes measuring its own calibration masks; the pair coefficients and support are per-prompt.

For the primary evaluation, find the best fixed-size three-head group at equal time. All 56 possible triples compete. Each method starts with the same exact fallback (mask 7), measured in common setup separately from model capture. For each checkpoint a panel-selection prompt determines the clock allowance using a full six-pair fit and a verification batch of eight, with 10% slack on the median of four warm runs. Online policy timings use fresh synchronized computation and include queries, ranking, and verifying proposed groups. Only a decision completed before the cutoff counts. Gradient and fitted policies verify in batches of eight, Guided-AP uses a four-triple stratified pilot then checks the highest remaining scores (its constant pilot offset leaves their ordering unchanged), greedy checks singles and pairs before proposing triples. Exact random checks use batches of eight or 32 or one batched exhaustive call. Common fallback and site capture are shared and charged equally. A separately collected exhaustive table of all 256 effects and top vocabulary tokens allows independent scoring, native substitution audits and interaction decomposition.

The primary endpoint is the actual donor-margin gain of the selected triple. Superiority requires positive simultaneous paired 95% lower confidence bounds against all competitors on the joint transfer cell. Also report the gap to the true best triple, full-vocabulary switches among initially correct recipients, raw prediction error, nonadditive effects and the held-out IOI template breakdown. The primary task models individual prompts and two blocks. Official graph-level scoring is a separate evaluation.

From the repository root, set `PYTHONPATH=.`, single-thread BLAS and OMP, and use the model revisions in [component routing](../component_routing/README.md). Under INPUTS provide the two checkpoints as `qwen` and `large`, plus `ioi-test.json`, `colors-test.jsonl` and `colors-validation.jsonl` at the pinned revisions above.

    python -m scripts.interaction_search.data INPUTS OUT select
    for model in qwen large; do python -m scripts.interaction_search.data INPUTS OUT "$model"; done
    for stage in dev qwen large; do python -m scripts.interaction_search.run INPUTS OUT OUT "$stage"; done
    python -m scripts.interaction_search.score OUT OUT/summary.json
    python -m scripts.interaction_search.mechanisms OUT OUT/mechanisms.json

Reproduce the calibration and timing diagnostics in the same output directory as the primary collection. These diagnostics were added after the primary protocol freeze. The single-verification run uses two examples from each held-out IOI template and the first twelve CopyColors objects. Each method pays for one proposed group and falls back to the shared measured triple when it misses the deadline. The repeated-query measurement uses the first two examples per task and checkpoint, three warm repetitions per workload and fresh calibration for each fitted predictor.

    for model in qwen large; do
        python -m scripts.interaction_search.amortized gpu INPUTS OUT OUT "$model"
        python -m scripts.interaction_search.verify_one collect INPUTS OUT OUT "$model"
    done
    python -m scripts.interaction_search.amortized score OUT OUT/amortized-summary.json
    python -m scripts.interaction_search.verify_one summarize OUT OUT/one-check-summary.json
    python -m scripts.interaction_search.audit OUT OUT/audit.json

The calibration-budget replay holds the three-pair support size fixed and takes nested, complement-balanced subsets of the original 32 observations. The chosen budget is the smallest tested budget retaining 90% of the 32-observation MAE improvement over the midpoint on development. The smaller-budget results measure raw prediction and raw top-choice accuracy.

    python -m scripts.interaction_search.calibration_budget OUT OUT/calibration-budget.json
