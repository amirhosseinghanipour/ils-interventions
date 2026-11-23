# Arithmetic component interventions

This benchmark predicts the signed effect of replacing a *whole component across all prompt positions* with its activation from a paired donor prompt. Three blocks are evaluated, at one-quarter, one-half and three-quarters of decoder depth (one-based). Every attention head and the MLP in each block is a candidate. A native head patch replaces the corresponding input slice of the attention output projection. A native MLP patch replaces its output. Exact evaluation reruns the model and measures the donor answer token's logit minus the original answer token's logit at the first answer token where the two answers differ. Counts of clean and donor correctness refer to the top vocabulary token at this position. This is a component intervention study on MIB texts; its metrics differ from the official MIB graph leaderboard.

## Model and data inputs

Use CUDA PyTorch and Hugging Face Transformers with FP32, eager attention, and TF32 disabled. Store these model revisions as Hugging Face directories under INPUTS. The runner checks the weight hashes.

- `qwen`: Qwen/Qwen2.5-0.5B-Instruct at `7ae557604adf67be50417f59c2c2f167def9a775`.
- `large`: HuggingFaceTB/SmolLM2-1.7B-Instruct at `31b70e2e869a7173562077fd711b654946d38674`.

Download `validation.jsonl` and `test.jsonl` from the [MIB addition release](https://huggingface.co/datasets/mib-bench/arithmetic_addition/tree/d56d68b7b1d8f9c7e9a262ec1f89331ed52c7516), and `test.jsonl` from the [subtraction release](https://huggingface.co/datasets/mib-bench/arithmetic_subtraction/tree/9eeb7f0f69c2adc131d81200f501a7f83c06fa8b). Name them `addition-validation.jsonl`, `addition-test.jsonl` and `subtraction-test.jsonl` under INPUTS. Frozen indices refer to each file's zero-based line number. Use unmodified prompt text. Alternate a changed ones digit of operand 1 and a changed tens digit of operand 2 using the released counterfactuals. Require compatible input lengths and distinct answer continuations in both tokenizers. Original and donor operand pairs are disjoint across examples of an operation. No model-output filter is applied.

## Estimators

For a head, the cut is the post-attention residual. Its edit is the change in that head's output projected through the corresponding columns of the output matrix. For an MLP, the cut is the block output and its edit is the change in MLP output. These isolated residual edits are checked against actual component substitutions. With recipient and donor cut states h and k, component displacement d, and suffix margin F, midpoint predicts the inner product of d with the gradient at (h+k)/2. ILS adds a fraction of the difference between this estimate and F(k)−F(h)+DF(k)(h+d−k). The fraction is the fitted scalar times ||d||⁴/(||d||⁴+||h+d−k||⁴). Six cut-specific suffix gradients supply midpoint predictions. The ILS donor gradients come from one shared backward pass through the clean donor. A cheaper midpoint control interpolates the input embeddings once. The correction strength is selected from 0.25, 0.5, and 1 by lowest signed-effect MAE on 20 Qwen addition validation examples, with smaller strengths breaking ties. The zero-strength control is midpoint itself.

Within-prompt Guided-AP follows [Zhang et al., Algorithm 1](https://arxiv.org/html/2610.27393v1). It uses one recipient gradient, a four-site pilot stratified across equal-sized absolute attribution rank bands, the pilot median residual as an additive correction, and direct verification of the largest remaining absolute attribution scores. The smaller pilot reflects this benchmark's 45 or 99 candidate components. An uncorrected attribution-patching comparator verifies the largest scores directly. A gradient-free random scan measures eight sites per batch. Gradient methods verify four at a time. Seeds and candidate ordering are fixed.

## Budgets and outcomes

An independent synthetic prompt sets the two allowances per checkpoint from four warm, synchronized timings. Tight allows 1.2 times the median full ILS estimate time plus one verification batch. Roomy adds three more batches. Clean recipient and donor capture and cut construction are common to all methods on each prompt; their measured time is reported separately and adds identically to the end-to-end allowance. The estimator, pilot, and verification passes are charged. Only predictions and measurements available before the cutoff count. An overrun of the initial estimate receives an all-zero fallback with uniform tie credit. Uncapped raw scores isolate estimator quality. Complete reference labels are collected separately after predictions; paid measurements are compared with the independent exact runs. Wall-clock verification counts can vary with hardware and scheduling.

The primary comparison is SmolLM on subtraction at the tight allowance. Evaluate both signed-effect MAE and the mean of two directional recovery fractions: actual increase and actual decrease in the original correct-answer margin at the predicted best sites, relative to the best possible effect in each direction. Opposite-sign choices recover zero. Absent directions are excluded, and tied predictions split credit uniformly. The success rule requires simultaneous positive advantages over every comparator on both outcomes, using a paired prompt bootstrap across the comparison family. Other cells, component types and allowances are diagnostic.

    python scripts/component_routing/benchmark.py INPUTS artifacts/component-routing/manifest.json OUT qwen addition-validation
    for model in qwen large; do
      for task in addition-test subtraction-test; do
        python scripts/component_routing/benchmark.py INPUTS artifacts/component-routing/manifest.json OUT "$model" "$task"
      done
    done
    python scripts/component_routing/score.py OUT OUT/summary.json
