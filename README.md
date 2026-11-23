# ILS Interventions

Interventional least squares (ILS) measures how accurately a change in a transformer's residual stream predicts its downstream effects on a held-out prompt. The depth study maps that accuracy across source blocks and token horizons in three Qwen checkpoints. It reports direct residual carry, learned changes beyond carry, and effects at strictly later positions.

On the 0.5B primary model, a fitted correction explained 62.2% of penultimate-block response variance on a fresh holdout at γ = 0.9. Its gain over a fitted scalar was 0.118 R² [0.100, 0.147]. The gains were smaller on the two larger checkpoints, and strictly later positions were much harder to predict. [Results and limits](REPORT.md) · [Reproduce and audit](scripts/internal_replication/README.md)

The follow-up study holds an edit fixed and changes which fact the later question asks about. On Qwen3.5-9B, reversing one attention head reduced the selective donor effect by 5.74 logits on new color prompts and 2.18 on packed objects. Color-trained source maps lost nearly all predictive coverage on unseen answer words. Calibrating each word with one clean repetition prompt and learning two relevance scales on colors explained 74% of animal-task residual variance on 9B and 70% on 4B. [Raw measurements and verification](scripts/query_routing/README.md)

The repository also retains the numerical validation, toy model, and historical readout and graph experiments. The original broader proposal remains incomplete. [Study history](REPORT.md) · [Initial pilot](RESULTS.md) · [Toy experiments](TOY_RESULTS.md)

## Query-conditioned interventions

At block 19, replacing a two-word query's residual update with the update from the other query increased the alternative answer margin by 7.73 logits over a norm-matched random edit on Qwen3.5-9B and 8.10 on 4B. The substitution switched 38 and 23 of 128 greedy second tokens, respectively. The complete block state switched all 128 on both models. Reproduction scripts, metrics and raw-data schemas accompany the held-out comparisons. [Experimental code](scripts/query_transition/README.md) · [Numerical results](artifacts/query-transition) · [Paired measurements](https://github.com/amirhosseinghanipour/ils-interventions/tree/data/query-transition)

On 64 fresh material prefixes, a food-trained block-15 mention-order classifier and independently calibrated word vectors gave held-out residual R² of 0.685 on 4B and 0.173 on 9B. Using 16 unlabeled material prefixes to re-center the classifier increased 9B residual R² from 0.167 to 0.555 on the other 48 prefixes. The 4B classifier identified all 96 mentions with either center. Across paired human questions from 40 SQuAD articles, the block-19 donor update shifted the alternative first-token margin by 6.41 logits over random on 4B and 4.17 on 9B. It switched the clean-correct first token to the other human answer on 1/58 and 0/66 questions, respectively. [New raw measurements](https://github.com/amirhosseinghanipour/ils-interventions/tree/data/query-transition) · [Earlier raw measurements](https://github.com/amirhosseinghanipour/ils-interventions/tree/data/query-routing)

## Arithmetic component interventions

The arithmetic benchmark measures how replacing attention heads and MLPs changes an answer-token margin. Its single-site experiment compares component selection under equal time budgets. Its group experiment measures the accuracy and speed of predicting many simultaneous head substitutions on one prompt. [Measurements and checksums](artifacts/component-routing/README.md) · [Reproduction commands](scripts/component_routing/README.md)

## Interactions between attention heads

The two-block benchmark measures every subset of an eight-head panel on 48 indirect object identification prompts and 48 CopyColors prompts at two checkpoints. Across 49,152 native interventions, a midpoint gradient with a sparse pair correction predicted three-head effects more accurately than the same fit with only additive terms in all four cells. On SmolLM2 CopyColors, mean absolute error was 0.0348 versus 0.0711 logits. By the frozen time cutoff, midpoint with exact verification had computed the best triple in all 48 traces, while full ILS calibration had produced only the common fallback. The synchronous verification loop could return later. On a separate timing sample with 222 queries per prompt, ILS was 2.47× faster than the best tested cached exact implementation on Qwen and 3.22× faster on SmolLM2. [Data, scores and checksums](artifacts/interaction-search/README.md) · [Reproduction commands](scripts/interaction_search/README.md)

## Install

Use Python 3.11 or newer in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,plots]'
sh scripts/test.sh
```

[requirements-observed.txt](requirements-observed.txt) records the package
versions used for the included experiments. It is an environment record, not a
tested cross-platform lock. No GPU is required for the tests or toy experiment.

## Reproduce the experiments

```sh
sh scripts/toy.sh
python scripts/plot_results.py
```

The toy run trains a four-layer transformer on a seeded HMM, fits local J/R and
ILS maps on separate prompts, and evaluates held-out interventions and true-state
readout. Configs, fitted matrices, samples, and results are included. Regeneration
overwrites the corresponding local artifact files; copy them first if preserving
a run. Set `ILS_PYTHON` to use a different interpreter.

The Qwen pilot requires an existing, unquantized local model directory:

```sh
export ILS_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
mkdir -p logs
python scripts/smoke.py
sh scripts/qwen_pilot.sh
```

Weights are loaded locally and cast to fp32. Qwen weights and external datasets
are not redistributed here. To download reference repositories and WikiText-103
in a network-enabled environment, run `bash scripts/download_inputs.sh`.
Inspect the downloaded code before using it. Reference commit IDs are recorded
in [configs/reference-revisions.json](configs/reference-revisions.json) and
[REFERENCES.md](REFERENCES.md); downloaded repositories remain untracked.

## Methods

For antithetic interventions δ at one block-output position, fit

```text
A = (mean(Δy δᵀ) + λ A₀) (mean(δ δᵀ) + λ I)⁻¹
```

The code supports normalized horizon weights, ridge/Huber/prompt-block MoM,
prompt-group cross-validation, and I/J/R priors without readout-label selection.
Additional experimental helpers cover calibration, concept residuals, contrasts,
random-direction controls, transcoders and phrase readout rows.

**Convention warning:** upstream `jlens.fit` averages source-position gradients
after summing over valid future targets. ILS uses a normalized future horizon
per source. These are different estimands. The completed isotropic-limit check
compares ILS with our own normalized-horizon J map, not the upstream summed map.
All 72 layer/epsilon cells passed its matrix/readout thresholds, while the
small-epsilon directional JVP error remains a documented limitation. See
[the numerical result](artifacts/qwen-numerical-validation.json) and
[REPORT.md](REPORT.md).

## Repository contents

- `ilslens/`: collection, fitting, readout and evaluation primitives.
- `tests/`: numerical, causality and tiny-model substitution checks.
- `configs/` and `scripts/`: reproducible experiment settings and commands.
- `artifacts/`: small toy weights, sample tensors, fitted lenses and measured results,
  including hashes for larger local Qwen artifacts.
- `REPORT.md`: current research findings, costs, deviations and missing evidence.
- `RESULTS.md`: initial pilot findings and limitations.

See [CONTRIBUTING.md](CONTRIBUTING.md) for research and contribution standards.
The original code is MIT licensed; third-party models, data and repositories
retain their own licenses. See [NOTICE](NOTICE).

## Primary Qwen run

The resumable J/R and donor-collection runner uses the preregistered settings in
`configs/qwen-main.yaml`: 16 source prompts, independent donor prompts, 7168
interventions per layer/epsilon, all 24 layers, six epsilons and four horizons.
The fp32 GPU suffix benchmark projects about 4.91 hours for suffix intervention
work at the registered budget; this excludes bank construction, baseline fitting,
cross-validation, checkpoint overhead and downstream evaluations. It is not a
study completion forecast. Primary J/R fitting completed in 1,339.6 seconds
including preparation; the full donor collection and ILS fits are still pending.
See the [resource projection](artifacts/primary-resource-projection.json) and
[baseline audit](artifacts/qwen-primary-baselines.json). Benchmark before running
on another machine and use a fresh output directory for a different protocol.

```sh
export ILS_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
python scripts/run_qwen_main.py --dataset data/wikitext-103-raw-v1 \
  --reference references/jacobian-lens --phase benchmark
python scripts/run_qwen_main.py --dataset data/wikitext-103-raw-v1 \
  --reference references/jacobian-lens --phase all
python scripts/fit_qwen_offline.py --run runs/qwen-main
```

The runner saves atomic per-prompt/per-layer checkpoints under ignored `runs/`.
The offline fitter requires complete training and held-out collection and uses
training-fold-only J/R priors. These commands cover only the primary fitting and
intervention work, not the entire planned research study.

Production model runners now default to guarded CUDA execution. They reject a
busy GPU or insufficient free VRAM and never silently fall back to CPU. Existing
checkpoints are reusable only when the required experiment provenance matches;
newly written checkpoints record their execution device. TF32 is disabled for
fp32 research. Completed numerical validation does not establish held-out
faithfulness or causal readout accuracy.

Additional evaluation entry points include `evaluate_readouts.py` (strict
capability/leakage gates), `fit_jpp_qwen.py` (labeled dev only),
`calibrate_qwen.py` (100k reference activations), and `evaluate_contrast.py`
(on/off-policy contrasts and prompt-group null ranks). Run each with `--help`.
Modules for causal swaps, phrase continuation effects, LoRA adapters and stress
test statistics have small-input tests; their presence is not a completed
large-model experiment.

`run_planted.py` saves an indirect-example LoRA adapter, measures paired base/tuned
behavior, and can evaluate absolute, contrastive, and model-difference readouts.
`run_subliminal.py` uses the two upstream number filters and requires a positive
paired effect at p < .01 for each of three student seeds. The configuration
predeclares up to five concepts; a failed transmission remains a result.

`train_transcoders.py` consumes a versioned local web/chat corpus, trains all-layer
TopK skip transcoders, and reports held-out variance explained, dead features and
all-MLP substitution CE recovery. Its configured budget is now 100M token
presentations per layer; distinct tokens and repeated presentations are separate.
That budget has not been run. `evaluate_graphs.py --exploratory-affine` validates
feature, error and embedding edges under an additional feature freeze, separately checks
interventions with only norms frozen in the original model, and emits itemwise
budget closure. `evaluate_causal.py` implements swap/ablation/answer controls and
100 covariance-matched random interventions at every patched site. These new
production experiments have not yet been run on Qwen.

The verified 100M distinct-token web/chat corpus has zero exact-window overlap
with its validation split. The all-layer transcoder throughput benchmark is
queued after successful primary fitting in a fixed checkout; its weights will
be discarded, and it will not launch training. Live-TopK graph propagation still
fails the required finite derivative/effect equality on the retained synthetic
fixture. That failure has not been waived or presented as a causal result.

Before the large transcoder run, measure the registered all-layer update workload:

```sh
PYTHONPATH=. python scripts/benchmark_transcoders.py --dataset data/wikitext-103-raw-v1 --device cuda
```

This discards benchmark weights and reports raw timings and peak CUDA memory.
Its update-time projection excludes corpus processing, checkpoint I/O and evaluation.

The default transcoder configuration requires distinct training windows. Run
`audit_transcoder_corpus.py --corpus PATH` before training; insufficient data or
validation overlap fails the audit. `download_transcoder_shards.py --output PATH`
downloads pinned, checksum-verified shards. `prepare_transcoder_corpus.py` accepts
`--web-parquet` and `--chat-parquet` with explicit dataset revisions to build a
larger local corpus without requiring streaming network support.

`run_primary_after_validation.py` can wait for an existing numerical validation
process, verify all 24 layers, and then run primary collection and fitting in
sequence. It never enables the full study or falls back to CPU. Use a fixed
checkout and distinct output paths; see `--help` for required inputs.

The current [research report](REPORT.md) separates completed measurements from
unrun production stages and answers all eight research questions with the
available evidence. `evaluate_workspacebench.py` uses upstream registered banks
and renderers; `run_template_lens.py`, `evaluate_concept.py`,
`evaluate_faithfulness.py`, `check_qwen_limit.py`, `collect_ablations.py`,
`fit_diagnostics.py` and `evaluate_procrustes.py` implement their respective
registered measurements. They remain pending on production Qwen artifacts.

The serial GPU runner checks pinned reference revisions, available disk, and the
small-input tests before starting. It waits for an idle GPU when requested,
records each command and elapsed time, and halts on any failed stage. Run it from
this checkout with the model environment variable set:

```sh
python -m ilslens.study --dataset data/wikitext-103-raw-v1 --primary-run runs/qwen-main \
  --references references --corpus data/transcoder-corpus.jsonl --dry-run
python -m ilslens.study --dataset data/wikitext-103-raw-v1 --primary-run runs/qwen-main \
  --references references --corpus data/transcoder-corpus.jsonl --wait-for-gpu
```

Use `--through gpu-benchmark` for an initial GPU timing run. Status, stage logs,
and completion records live in ignored `runs/study/`. Changed source code or
completed-stage provenance stops a resumed pipeline for investigation. The final
artifact audit distinguishes missing measurements from null findings; it does
not certify scientific validity or replace review of the research report.

Graph evaluations accept exact benchmark exports, preserve all their token read
positions, and optionally use fitted phrase rows. Task coverage uses an explicitly
documented excluded EOS sentinel, with skip0 on short task contexts; WikiText
fitting retains skip16. Unsupported phrase targets remain unknown. The planted
graph stages compare base/tuned models on identical base-completion token IDs.

`prepare_released_baseline.py --download` imports the pinned exact-model release
as `J-released-extra (200 prompts)`. Its different fitting budget, BF16 producer
and FP16 storage are recorded; conversion to fp32 does not undo precision loss.
It is never used as an ILS prior. `report_study.py` exports measured stage tables
and plots, including negative and undefined results, without declaring missing
stages complete. Feature graph exports keep reads as targets and include bounded,
individually validated upstream feature edges and explicit expansion limits.

The larger transcoder corpus initially failed its exact-window overlap audit.
To reproduce the correction while retaining the original corpus:

```sh
PYTHONPATH=. python scripts/deduplicate_transcoder_validation.py \
  --corpus data/transcoder-corpus-100m.jsonl \
  --output data/transcoder-corpus-100m-clean.jsonl
```

The script refuses to overwrite an existing output, leaves training records
unchanged, removes contaminated validation documents, and reruns the token-budget
audit. Pass the corrected path explicitly as `--corpus` to subsequent training or
study commands. The public clean-corpus manifest and budget audit record the
measured preprocessing result; they are not transcoder training results.
