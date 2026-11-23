# Reproducing the paired query-routing study

[Compact results](../../artifacts/query-routing) · [Raw data](https://github.com/amirhosseinghanipour/ils-interventions/tree/data/query-routing)

## Audit the released measurements

Start in the code repository root. Obtain a separate checkout of the data branch. Set `DATA` to the absolute checkout path. The data include synthetic prompts and tokens, per-pair margins and top tokens, residual changes, training displacements, the head screen, and the 0.5B directional derivatives. The initial 9B run also includes full signed residual trajectories. The archived WikiText binaries contain horizon summaries.

```sh
git clone --single-branch --branch data/query-routing https://github.com/amirhosseinghanipour/ils-interventions.git ../ils-query-routing-data
DATA="$(cd ../ils-query-routing-data && pwd)"
(cd "$DATA" && sha256sum --quiet -c SHA256SUMS)
export OPENBLAS_NUM_THREADS=1
p=scripts/query_routing
python "$p/verify.py" "$DATA" artifacts/query-routing
python "$p/check_bound.py"
tmp=$(mktemp -d)
python "$p/archived_magnitude.py" . "$tmp/magnitude.json" --raw-data "$DATA/archived/internal-replication"
python "$p/lag_dispersion.py" . "$tmp/dispersion.json" --raw-data "$DATA/archived/internal-replication"
cmp "$tmp/magnitude.json" artifacts/query-routing/v1/archive-magnitude.json
cmp "$tmp/dispersion.json" artifacts/query-routing/v1/archive-lag-dispersion.json
for model in 9b 4b 05b; do
  python "$p/score_routing.py" "$DATA/$model" "$tmp/$model"
done
for model in 9b 4b; do
  python "$p/score_routing.py" "$DATA/parcels/$model" "$tmp/parcels/$model" --training-root "$DATA/$model"
done
```

The independent verifier solves the ridge systems again, recalculates the component interventions, and checks the published bootstrap intervals. NumPy suffices. To plot the comparisons at a local destination, use `python scripts/query_routing/figure.py /path/to/figure.pdf` in an environment with Matplotlib.

## Collect new measurements

The [primary protocol](../../scripts/query_routing/protocol.json), [follow-up](../../scripts/query_routing/followup.json), and [transfer test](../../scripts/query_routing/transfer.json) define sample sizes, seeds, estimands and controls. The 9B primary code and original measurements predate the follow-up. The development screen chose block 23, head 0 before both component confirmations. A repeated training prefix triggered the deterministic, pre-inference rejection rule in the final sampler. The raw data contain the actual token sequences for every evaluated pair.

The native collector requires Linux, a C++17 compiler and compatible llama.cpp b12110 headers, CUDA shared libraries and GPU memory. Set `PYTHON` to the Python executable, `LLAMA_SRC` to the llama.cpp checkout, `LLAMA_BIN` to its shared-library directory, and `MODEL9`, `MODEL4` to the GGUF files. The launchers check whole-file model hashes. Allocate a new absolute output path for each run. Optional `HEAVY_LOCK` is an existing lock file shared with other jobs. The fp32 run uses a local Qwen2.5-0.5B-Instruct directory including its download manifest, PyTorch, Transformers and the installed ILS-lens package. The released checkpoint manifest identifies the files and revision.

```sh
MODEL="$MODEL9" OUT="$PWD/runs/primary" bash scripts/query_routing/launch.sh
OUT="$PWD/runs/ledger" bash scripts/query_routing/replicate.sh
"$PYTHON" scripts/query_routing/differential.py --model "$MODEL05" --output "$PWD/runs/diff/05b"
export LLAMA_BACKEND_DIR="$LLAMA_BIN" LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PYTHON" scripts/query_routing/followup.py --library runs/ledger/probe.so --model "$MODEL9" --output runs/parcels/9b --heads --parcels
"$PYTHON" scripts/query_routing/followup.py --library runs/ledger/probe.so --model "$MODEL4" --output runs/parcels/4b --parcels
```

Score these directories using the commands above, substituting the new locations for the data branch. For the primary run, `summary.py DATA test` audits the recorded raw effects and writes the test summary in the output directory. A single head record stores both query sides, ordered relevant then irrelevant. A positive restoration reduction means that restoring the clean head weakened the donor effect. The reverse injection is a separate intervention. All positions and blocks use zero-based indexing.

The scalar-copy refinement was specified after the 9B ledger result and before the parcel readout. To replay a fitted scalar plus ridge correction, first run the ordinary scorer, then run `python scripts/query_routing/scalar_prior.py TRAIN_DIR TEST_DIR RESULT_DIR`. Use the same color training directory for both the color and parcel evaluations. The supplementary 0.5B capitalization check uses `case_audit.py`, the published confirmation data and the exact tokenizer JSON pinned by the checkpoint manifest.

## Frozen answer-word prototype test

The [prototype specification](../../scripts/query_routing/prototypes.json) fixes an animal task held back from both earlier transfer experiments. On each larger model, `carriers.py` collects one neutral clean response per eligible word. `prototype_fit.py` learns the two scale factors and source correction entirely on color training data. The predictor has access to the intervention's word identities as well as the supplied relevance label. The raw carrier data permit scoring without running a model.

```sh
for model in 9b 4b; do
  python "$p/score_routing.py" "$DATA/animals/$model" "$tmp/animals/$model" --training-root "$DATA/$model"
  python "$p/prototype_fit.py" "$DATA/$model" "$DATA/$model" "$DATA/carriers/$model" "$tmp/$model"
  for task in parcels animals; do
    python "$p/prototype_fit.py" "$DATA/$model" "$DATA/$task/$model" "$DATA/carriers/$model" "$tmp/$task/$model"
  done
done
```

For a fresh native collection, run `carriers.py LIBRARY GGUF OUTPUT` and `followup.py --animals --library LIBRARY --model GGUF --output OUTPUT` once per checkpoint. The absolute paths in the earlier collection commands can be reused. These are separate output directories. Each carrier prompt and its exact tokenization are published.

## Withheld-annotation test

The [fixed food-task protocol](blind.json) scores single questions after shuffling their order. The deployment inputs are a known intervention displacement, a clean answer-position state, and a collection of neutral per-word states. Scoring withholds the prompt strings, food category, word identities, query relation and edited destination vector from the predictor. All eligible colors, objects, animals and foods compete in the same dictionary. Block 15 is the specified primary clean-state readout. Blocks 23, 27 and 31 are diagnostics for how query information becomes accessible with depth.

Use `blind_collect.py LIBRARY GGUF DATA/carriers/MODEL OUT` on each native checkpoint. It records the calibration dictionary, blinded inputs, finite-effect targets and tokens separately. To replay on CPU, run `OPENBLAS_NUM_THREADS=1 python scripts/query_routing/blind_score.py DATA/blind/MODEL artifacts/query-routing/MODEL/prototypes.json OUT.json`, replacing MODEL with 4b or 9b. The two color-trained coefficients remain frozen.
