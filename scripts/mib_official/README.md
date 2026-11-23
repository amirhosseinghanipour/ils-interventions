# MIB edge-circuit evaluation on public GPT-2 IOI

The evaluation measures a global circuit at ten edge budgets on the released GPT-2 small indirect object identification task. The model, IOI label metric, split filter, graph representation, pruning, counterfactual patching and sparsity grid come from the MIB organizers. Each circuit applies to every public test prompt. [Official evaluation code](https://github.com/hannamw/MIB-circuit-track/tree/b759df34433c9e31043ba9e02908ce0bf20e894f) · [Released IOI data](https://huggingface.co/datasets/mib-bench/ioi/tree/e5f3468f3af4c0883be35cd3bced8c711c95d286)

The measured response is the correct indirect object logit minus the subject logit. Edges outside each circuit receive activations from the matched subject-flipped counterfactual. Faithfulness is the difference between the circuit and empty-graph mean margins divided by the difference between the full and empty-graph mean margins. CPR integrates faithfulness over nominal edge fractions and favors larger values, including effects stronger than the full model's. CMD integrates absolute deviation from one and favors smaller values. The trapezoids span 0.1% to 100% of the 32,491 legal edges. [Organizer metric](https://github.com/hannamw/MIB-circuit-track/blob/b759df34433c9e31043ba9e02908ce0bf20e894f/MIB_circuit_track/evaluation.py)

## Compared graphs

| Output key | Construction |
| --- | --- |
| `eap_ig` | Signed or absolute ranking of the EAP-IG example shipped in the organizer repository. The repository labels this an example circuit and provides no training-split provenance for its importance file. |
| `mattr` | Released 5,000-step MAttr edge importances trained on MIB train. |
| `technion` | Released ILP+PNR+Bootstrapping retained edges and nodes for each sparsity and objective. |
| `observed` | Highest-scoring measured development circuit for each objective and budget. |
| `linear` | Main-effects ridge fit to the same measured development circuits. |
| `pair` | Main-effects and two-way interaction ridge fit to those circuits. |

The three development fits use 16 eligible prompts at fixed seed 867301. At each of the nine nontrivial budgets and for each objective, six switches can exchange disjoint, equal-sized bands of the organizer's edge ranking immediately inside and outside the cutoff. Exactly 24 of the 64 candidate settings are evaluated on the same development prompts, including the unmodified ranking. Organizer pruning can merge settings or change the realized edge count. At the smallest CPR budget the 24 settings produce 21 distinct retained circuits, and the 64 settings span 7 to 11 retained edges. The reference setting matches the organizer's retained edges and nodes at all 18 nontrivial objective/budget combinations. [Mask audit](../../artifacts/mib-official/mask-audit.json) The two regressions use an unpenalized fixed baseline, ridge penalty 2 and the same observations. Each selects the best predicted candidate among all 64. The policy at every sparsity is fixed before testing. This is a development-trained graph adaptation of interventional least squares. The per-prompt eight-head search in the main experiments answers a separate question.

The released graphs come from the original authors. Comparing their achieved scores uses these previously trained circuits. Training and calibration compute differs between methods. [MAttr graph release](https://huggingface.co/aryaman/mattr-mib-circuits/tree/3f88981654c117ef93fcab520d94f76fdc4ad765/edge/ioi_gpt2) · [ILP+PNR+Bootstrapping graph release](https://huggingface.co/Technion-CS-NLP/ilp-pnr-bootstrapping-circuits/tree/de056ee7180bd3342a35e7fdf22dd3c1d3b54440) · [Nikankin et al., BlackboxNLP 2025](https://aclanthology.org/2025.blackboxnlp-1.29/)

## Run

The recorded run used a 12 GB GeForce RTX 3060, Python 3.14.4, PyTorch 2.14.0, NumPy 2.5.3, SciPy 1.18.1, datasets 5.0.1, TransformerLens 2.15.4 and transformers 4.51.3. It used float32 with TF32 disabled. Use a CUDA-capable PyTorch environment with sufficient memory for transformer activations. Clone the two external repositories into a scratch directory. Run the following from this repository's root.

```sh
export INPUTS=/absolute/path/to/mib-inputs
export HF_HOME="$INPUTS/hf"
mkdir -p "$INPUTS"
git clone https://github.com/hannamw/MIB-circuit-track "$INPUTS/upstream"
git -C "$INPUTS/upstream" checkout b759df34433c9e31043ba9e02908ce0bf20e894f
git clone https://github.com/hannamw/EAP-IG "$INPUTS/eap-source"
git -C "$INPUTS/eap-source" checkout 7af394a5662de8b23ad6154716a0cd3993d447a3
python -m pip install 'transformer-lens==2.15.4' 'transformers==4.51.3' 'numpy>=2' tabulate
export PYTHONPATH=".:$INPUTS/upstream:$INPUTS/eap-source/src"
python scripts/mib_official/prepare_inputs.py "$INPUTS"
OMP_NUM_THREADS=1 python -m scripts.mib_official.audit_masks "$INPUTS/upstream/circuits/ioi-eap-ig.json"
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/mib_official/public_ioi.py "$INPUTS" "$INPUTS/measurements" \
  artifacts/mib-official/ioi-public-test.json.gz --batch 32
python scripts/mib_official/analyze.py artifacts/mib-official/ioi-public-test.json.gz \
  > artifacts/mib-official/public-summary.json
```

The GPU run saves prompt-level margins and reuses identical node and edge masks between methods. The result records the input checksums, development mask measurements, chosen masks and prompt-level test margins. The analysis recomputes the scores and draws 4,000 paired template bootstrap replicates, preserving every prompt within each sampled template. Intervals condition on the fitted circuits and carry uncertainty from nine test templates. The public test templates and development templates are disjoint.

## Additional organizer example

After observing the signed CPR of the frozen GitHub example, we identified a second example from the organizer's submission-format repository. This post-primary reference preserves all six original methods and their test results. Its GPT-2 file contains 32,491 finite edge importances and an empty initial selection. The model card gives no training history. Native signed and absolute top-N convert its scores into circuits for CPR and CMD. The run verifies the first 32 clean, empty-graph and complete-graph margins against the primary data and uses the same 1,000 test prompts and template bootstrap draws. [Organizer HF submission example](https://huggingface.co/mib-bench/mib-circuits-example/tree/2cc3324f6d3780f5d82fe741cd2fd3e20fc1643b/importances/pt/ioi_gpt2)

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.mib_official.organizer_example "$INPUTS" artifacts/mib-official/ioi-public-test.json.gz \
  "$INPUTS/organizer-hf-cache" artifacts/mib-official/organizer-hf-public-test.json.gz
python scripts/mib_official/analyze.py artifacts/mib-official/ioi-public-test.json.gz \
  --additional-example artifacts/mib-official/organizer-hf-public-test.json.gz \
  > artifacts/mib-official/organizer-hf-public-summary.json
```

## Organizer scorecard

The organizer dashboard evaluates submissions on a private test set. Its displayed scores and the public evaluation are separate results. A second check re-integrates the older signed EAP-IG-inputs curve from the organizer's archived JSON. The archived signed curve gives CPR 2.01188, while the live EAP-IG-inputs row displays 1.89. Its full-budget weighted edge count is 152,985, compared with the 32,491 edges in the organizer's bundled GPT-2 graph. The archived JSON supplies neither the original masks nor a split hash to resolve that discrepancy. Each source and input checksum is embedded in the scorecard. [Private leaderboard](https://huggingface.co/spaces/mib-bench/leaderboard) · [Archived baseline data](https://huggingface.co/spaces/mib-bench/leaderboard/blob/3f311010ee5af98bc80cadbee6b237021bd1f7dc/eval-results-mib-subgraph/baselines/EAP-IG-inputs_CF.json)

```sh
curl -fsSL https://mib-bench-leaderboard.hf.space/config -o "$INPUTS/dashboard.json"
curl -fsSL https://huggingface.co/spaces/mib-bench/leaderboard/raw/3f311010ee5af98bc80cadbee6b237021bd1f7dc/eval-results-mib-subgraph/baselines/EAP-IG-inputs_CF.json -o "$INPUTS/archive.json"
python scripts/mib_official/scorecard.py "$INPUTS/dashboard.json" "$INPUTS/archive.json" \
  --retrieved "$(date -u +%F)"
```
