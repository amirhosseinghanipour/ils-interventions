# Query-dependent retrieval across blocks

The two-person experiment compares two questions about the same facts. Both answers start with the same word. After that word, replace the recipient's residual update at one block with the update elicited by the other question. Record the change in the alternative-minus-original logit margin and the top token across the full vocabulary. Positive values favor the alternate answer. The four-person test asks for two words recorded in separate lists and intervenes on individual attention heads. A separate fit predicts the last residual change after each two-person block intervention.

## Get the raw measurements

The [transition archive](https://github.com/amirhosseinghanipour/ils-interventions/tree/data/query-transition) holds the new measurements, and the [query-routing archive](https://github.com/amirhosseinghanipour/ils-interventions/tree/data/query-routing) holds the earlier blind food controls.

```sh
git clone --depth 1 --single-branch --branch data/query-transition https://github.com/amirhosseinghanipour/ils-interventions.git ../ils-transition-data
git clone --depth 1 --single-branch --branch data/query-routing https://github.com/amirhosseinghanipour/ils-interventions.git ../ils-routing-data
DATA="$(cd ../ils-transition-data && pwd)"
BLIND="$(cd ../ils-routing-data/blind && pwd)"
```

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

## Fixed passage-edit positive control

The original 40 article pairs changed the question while keeping the passage fixed. This extension uses both questions from those same articles and one hand-specified counterfactual passage per article. `qa_pool_edits.tsv` specifies which question the edit addresses, the literal text replacement and the intended answer. Every occurrence of the case-sensitive source span is replaced in the passage, with both question strings kept fixed. Each query has its own donor forward on this edited passage. The donor activations replace the recipient's final prompt-position outputs on the original passage. The two original and relevant counterfactual first-answer tokens must be distinct. The annotations and the selection of the relevant question are available to this **positive control**. No subset discovery or answer-free inference is involved.

The shared pool contains every gated head in zero-indexed full-attention blocks 19 and 23 (32 heads), then adds blocks 15 and 27 (64). The 128-head pool covers all eight full-attention blocks. Additional arms include all 16 heads at block 19, the previously screened block 19 head 11, all 128 heads with the MLPs in those eight blocks, and all 128 heads with every MLP in the 32-block text trunk. Qwen3.5's other 24 blocks use recurrent linear attention; their attention updates are outside these head pools. All interventions occur at the last prompt position. Text-edit outputs test whether the checkpoint can follow the edited passage at all. Full-pool outcomes are positive controls; a subset can succeed where the full pool fails.

Each record contains question IDs, original and counterfactual first-answer token IDs, token lengths, the number of changed spans and every full-vocabulary top-1 observation. Self-transplants across all 128 heads and 32 MLPs must preserve the original top choice, and both clean top choices must match the corresponding archived experiment. Strict success requires both clean original first tokens to be correct, the intended relevant switch and unchanged irrelevant top-1. The results give this count on all 40 articles and among clean-correct pairs, plus separate marginal counts and a sensitivity analysis restricted to articles with successful text edits. Wilson intervals use one observation per article. The related questions can share premises, and the edits have no independent human re-annotation. Outcomes apply to first tokens, including for multi-token answers.

With the original SQuAD v1.1 file, public transition measurements and GGUFs identified above, rebuild the additional native hook and run:

```sh
g++ -O2 -fPIC -shared -std=c++17 -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  scripts/query_transition/qa_pool_probe.cpp -L"$LLAMA_BIN" -Wl,-rpath-link,"$LLAMA_BIN" \
  -lllama -lggml-base -lggml -o "$OUT/qa-pool.so"
export LLAMA_BACKEND_DIR="$LLAMA_BIN" LD_LIBRARY_PATH="$LLAMA_BIN:${LD_LIBRARY_PATH:-}"
for size in 4b 9b; do
    model="$MODEL4"
    if test "$size" = 9b; then model="$MODEL9"; fi
    "$PYTHON" scripts/query_transition/qa_pool_control.py "$OUT/qa-pool.so" "$model" \
        "$SQUAD" "$DATA/natural/$size/rows.json" "$OUT/pool-$size"
done
```

## Compact subset pilot

`qa_subset.py` uses the same native intervention site to test fixed eight-head subsets on article-disjoint, numeric-answer examples. Build the library as above. Run the script with the same five arguments as the positive control and an independent output directory. Discovery uses a fixed subset of the archived edits, ignores answer and relevance labels for scoring, and orders the two questions by a deterministic shuffle. It maximizes the larger KL divergence from the original distribution minus twice the smaller KL divergence. The frozen panel comprises two four-head groups. The candidate-pair control maximizes mean KL in the same pool. Additional controls include five fixed random eight-head panels, each constituent group, a development-selected full block, the historical single head and block, and the full head panel.

Fresh examples come from the official SQuAD v1.1 training partition, excluding all articles in its development partition. These articles are held out from subset discovery. Supply `--fresh-dataset` or put `train-v1.1.json` next to the original development file. The relevant answer must be a unique numeric string in the original paragraph. The source change substitutes a different number of the same length. Selection and ordering use text and tokenization, with no model predictions. These are automatically constructed edits and cover numeric answers only. Evaluation greedily generates up to 18 tokens while retaining the patch at the original final prompt position. Normalized exact answer match uses all original answer aliases. Group interactions use the four exact masks (neither, either group, both) and compare donor-minus-original first-token logit margin across the two queries. First-token ties are excluded from that contrast. This diagnostic remains distinct from full-answer evaluation.

The follow-up `--rolling-from PREVIOUS_RUN` reuses the frozen panel and article order. At every generated answer token it captures donor activations conditional on the generated prefix and patches the recipient at that decode position. It compares the selected subset with the full pool. Since this follow-up uses the same articles after the original results were inspected, it is exploratory. A second diagnostic teacher-forces the common numeric prefix on the relevant query and measures exact group interactions for the first divergent token. It does not use the forced prefix in the full-answer success metric.

## Matched patch timing and generation-aware head selection

Run `python scripts/query_transition/qa_timing.py LIBRARY MODEL SQUAD_TRAIN SQUAD_DEV PILOT_ROWS OUT`. Use the native library above, the SQuAD v1.1 train and development files, and `artifacts/qa-pool-control/fixed-subset/rows.json` as the pilot exclusion list. The program selects 10 development and 32 test articles, split equally between edits to a final numeric digit and edits to a two-word capitalized answer span. Test articles exclude every development and prior pilot article. Edits are generated from the text and tokenization, with no filtering based on predicted answers. Text-edit results measure the model's response to the constructed counterfactual.

The original and edited prompts receive the same clean greedy answer prefix when choosing a patch step. The policy chooses their first top-token disagreement within six steps, or the available step with greatest donor-to-recipient KL divergence if they agree. It receives no answer labels. The prompt-boundary and delayed conditions replace one position, retaining the replacement at that position in subsequent recomputations. The rolling condition replaces the current position on every decode call. It conditions the donor on the actual generated prefix. It discards earlier internal patches when recomputing later steps, while retaining the tokens they produced. The output saves the number of patched positions and the chosen steps.

The discovery procedure estimates donor-direction gain via the reduction in KL from the donor distribution to the patched distribution. It chooses which query to change from the larger observed donor effect, with a KL penalty for movement on the other query. It screens disjoint four-head groups across all full-attention blocks and verifies candidate pairs of groups on development. Exact greedy, query-agnostic, and three size-matched random panels share the same delayed timing rule. The freeze file records all selected sites before any held-out outcomes. A four-mask diagnostic uses donor-direction gain for each query and compares the group interaction across queries. These are model-local measurements of full-vocabulary distributions. Full-answer success separately requires normalized exact match to the new answer plus preservation of the other answer on the same article. The primary conditional denominator includes all articles where both clean answers were correct.

The 9B replication uses the same timing policy with `--timing-only --exclude-rows PROVENANCE_4B --test-seed 67219`. This skips subset discovery and excludes the complete 4B development and test split. Each checkpoint still uses both edit families. The experiment measures patch timing within the eight full-attention blocks of Qwen3.5. It does not intervene on recurrent linear-attention or MLP outputs. Greedy decoding stops at the first line break, end-of-generation token, or 14 generated tokens.

`qa_timing_replay.py artifacts/qa-timing artifacts/qa-timing/policy-replay.json` replays a per-query policy from the recorded answers. It uses one delayed patch if the disagreement detector chooses a position after the prompt, and patches every decode step otherwise. It uses neither answer labels nor edit type. Each query is generated independently, so the policy can replay the corresponding query output and intervention count exactly. The policy was derived after inspecting these traces. Patched position counts exclude detector overhead and measure interventions rather than wall time.

`qa_timing_specificity.py` accepts the same six positional arguments as `qa_timing.py`. Pass `--exclude` for each prior timing provenance file. It draws fresh numeric and capitalized-span edits, compares the disagreement patch position with its immediate neighbors and prompt position, and checks that self-donor replacement reproduces clean generation at the selected position. The primary matched-neighbor subset includes numeric articles with a delayed relevant position and with the next-position patch reachable on both queries. The late control sometimes patches after the model has already emitted a distinguishing token. The answer-prefix oracle uses the annotated common prefix on the relevant numeric query only. The program also measures the replay-derived per-query policy prospectively and stores the actual position counts.

## Fully charged online timing

`qa_timing_online.py` accepts the same six positionals as `qa_timing.py` and repeatable `--exclude PROVENANCE` arguments. For each new article it materializes two edits to the same numeric or capitalized answer span, keeping both question strings fixed. It checks from answer tokenization that the early edit changes the first answer token and the late edit has a nonempty shared prefix. The default sample is ten fresh articles per edit family. The synchronous CUDA probe records native forward-call counts and resident-model elapsed milliseconds for each query, with randomized method order. Clean and text-edit measurements are separate arms. A shared model forward warms the process before timing.

The scanned policy first estimates the position using donor and recipient forwards on the clean greedy prefix. It then runs the previously tested schedule from the beginning, including the KL fallback. The online policy compares top predictions during generation and reuses the donor forward at the triggering position. When a disagreement occurs at the prompt boundary, the policy patches every step. When the first disagreement occurs later, the intervention remains at that one position. If no disagreement occurs within six steps, it continues unpatched. This behavior differs from the scanned fallback, so compare generated answers as well as cost. For a continuation of fixed length (L) decode calls and first disagreement at step (s>0), the online policy uses (L+s+2) forwards and patching every step uses (2L). The benchmark counts actual forwards when output lengths differ.
