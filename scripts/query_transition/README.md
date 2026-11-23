# Query-dependent retrieval across blocks

The two-person experiment compares two questions about the same facts. Both answers start with the same word. After that word, replace the recipient's residual update at one block with the update elicited by the other question. Record the change in the alternative-minus-original logit margin and the top token across the full vocabulary. Positive values favor the alternate answer. The four-person test asks for two words recorded in separate lists and intervenes on individual attention heads. A separate fit predicts the last residual change after each two-person block intervention.

## Recompute from raw measurements

With Python and NumPy installed, set `DATA` to the measurements checkout, `BLIND` to the previous study's directory containing the two food-task archives, and `SCORES` to a fresh summary directory. From the repository root run

```sh
(cd "$DATA" && sha256sum --quiet -c SHA256SUMS)
export OPENBLAS_NUM_THREADS=1
python scripts/query_transition/score.py "$DATA" "$SCORES"
python scripts/query_transition/verify.py "$DATA" "$SCORES"
for size in 9b 4b; do
    python scripts/query_transition/centering.py "$BLIND/$size" "$SCORES/dense-centering-$size.json" --dense "$DATA/blocks/$size/food-depth.npz"
done
```

Each `blocks/<size>/test.json` records 64 distinct prefixes, both tokenized questions, clean top tokens and paired interventions at every scanned block. Each `heads/<size>/test.json` records the same fields for 64 new four-person assemblies. Zero-based block and head indices match the native graph. `transport/<size>/responses.npz` contains the imposed edit and the resulting final-state change, grouped by prefix and sign, separately for training and test. `transport/<size>/result.json` contains the fitted scale, penalties, held-out scores and paired uncertainty. The verifier independently solves the ridge fits from these arrays.

## Recollect from the checkpoints

Use llama.cpp b12110, a CUDA GPU, a C++17 compiler and the exact GGUF files checked by the launcher. Set absolute paths for `MODEL9`, `MODEL4`, `LLAMA_SRC`, `LLAMA_BIN`, `BLIND`, `PYTHON` and a new output directory `OUT` under an existing parent. Set `HEAVY_LOCK` to an existing shared lock on multi-user machines. Run `bash scripts/query_transition/replicate.sh`. It selects the layer on 9B development pairs, screens the heads separately by checkpoint, repeats the block interventions for transport, and compares the food readout inputs byte-for-byte with the original traces at blocks 15 and 23.

## Transfer to human-authored questions

Set `SQUAD` to the authors' unmodified SQuAD v1.1 development JSON before collecting to include the external test. The launcher checks its SHA-256. Each record in `natural/<size>/rows.json` gives the original article, paragraph index, question IDs, token lengths, target first tokens and all raw margin and top-token observations. To verify existing measurements run `python scripts/query_transition/verify_natural.py "$DATA" "$SQUAD"`. The results describe the first token on 40 distinct articles under the model context and annotator-agreement filters. [SQuAD authors' dataset](https://rajpurkar.github.io/SQuAD-explorer/)

The supplemental real-donor control compares the targeted intervention to the next paired record's block update, rescaled to the same norm. It runs when `SQUAD` is provided. The paired intervals condition on this fixed cyclic assignment. The auxiliary affine fit and sign-symmetry analysis can be rebuilt using `python scripts/query_transition/affine.py "$DATA/transport/9b/responses.npz" "$SCORES/affine-9b.json"` and likewise for 4B. This diagnostic was added after observing the homogeneous fits.

The supervised mention-order comparison in `slots.py` trains a nearest-centroid classifier on both questions from prefixes that share neither participant nor food with the test prefix. It never trains on the test prefix. The label-permutation reference swaps both labels in each selected prefix, applies the same swaps to held-out truth, refits the classifier, and accounts for the maximum across nine depths. It tests an exploratory mention-order readout within the fixed food template.

The material-task test trains the early mention-order classifier on the earlier food questions. Its color-trained amplitudes are stored in `artifacts/query-routing/animals/<size>/prototypes.json`. The material predictor receives the clean block-15 query state, actual block-15 source displacement, known source token index, and independent word carriers. Two tokenization-only calibration prompts map editable indices to mention order. The article and four-person experiments use different endpoints. To verify the materials comparison directly, use `python scripts/query_transition/verify_materials.py "$DATA/materials/9b" "$BLIND/9b" artifacts/query-routing/animals/9b/prototypes.json` for 9B and substitute 4B paths for the replication.

For the target-template offset diagnostic, the first 16 material prefixes contribute only a clean-state mean at block 15. Score the remaining 48 with `target_center.py` using the food data, material data, color scales and output JSON as its four path arguments. The face-value early material result and its 64-prefix analysis are retained separately. The 9B material result informed this diagnostic before 4B material outcomes were available.
