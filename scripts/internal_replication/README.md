# Internal-effect replication

Run from the repository root with the study environment and local WikiText-103 raw v1 at `../data/wikitext-103-raw-v1`. The primary reconstruction additionally needs the original `runs/qwen-proposal-v4` manifest, training/validation shards and fitted maps. The committed JSON summaries can regenerate the figures without those large artifacts.

The collection below uses llama.cpp b12110 with the matching CUDA 12.8 runtime, four host threads and one RTX 3060. Set SRC to the unpacked llama.cpp source, BIN to its unpacked CUDA binaries and CUDA runtime libraries, and MODEL4 and MODEL9 to the 4B Q4_K_M and 9B Q5_K_M GGUF files. Set PYTHONPATH to the repository root. The JSON results record the input digests. Use a fresh output directory for any changed protocol, input or executable. Keep C++ assertions enabled.

```sh
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
d=runs/internal-replication
mkdir -p "$d"
python scripts/internal_replication/prompts.py
g++ -O2 -I"$SRC/include" -I"$SRC/ggml/include" scripts/internal_replication/collect.cpp \
    -L"$BIN" -Wl,-rpath,"$BIN" -Wl,-rpath-link,"$BIN" -lllama -lggml-base -lggml -o "$d/collect"
sha256sum scripts/internal_replication/{collect.cpp,protocol.json,prompts.py} \
    "$d"/{inputs.txt,rows.json,collect} "$MODEL4" "$MODEL9" > "$d/provenance.sha256"
export LLAMA_BACKEND_DIR="$BIN" LD_LIBRARY_PATH="$BIN"
"$d/collect" "$MODEL4" "$d/inputs.txt" "$d/4b" 192
"$d/collect" "$MODEL9" "$d/inputs.txt" "$d/9b" 192
sha256sum -c "$d/provenance.sha256"
python scripts/internal_replication/score.py
```

Each file contains five consecutive blocks. Within each block are the displacement and four horizon effects, each a little-endian fp32 vector. The collector requires an all-position final residual and verifies causal prefix agreement on every signed intervention. Source and donor contexts are decoded separately with model memory reset. Scoring checks the analytic last-block identity for every horizon. The positive-penalty dual solution equals least squares with the specified matrix prior; at zero penalty it uses the minimum-norm map. Held-out data never enter penalty selection.

```sh
python scripts/predictive_coverage.py
python scripts/coverage_innovation.py
python scripts/internal_replication/figures.py
```

Run `python scripts/internal_replication/fresh.py` for the new 64-pair fp32 holdout and `python scripts/internal_replication/future.py` for the later-only analysis. Run CUDA jobs serially. The fresh and transfer runs use fresh-config.json and transfer-config.json. The primary and larger jobs save example counts, prompt sum-vector Gram matrices, target squared norms and per-predictor squared errors by prompt. These are sufficient to reconstruct both the pooled centered R² and the joint prompt-bootstrap denominator without releasing corpus text or checkpoint weights. From the repository root, `OPENBLAS_NUM_THREADS=2 python scripts/internal_replication/audit_tables.py` verifies all manuscript analysis grids using only the compact artifacts and NumPy.

For the frozen WikiText-to-FineWeb comparison, run `python scripts/internal_replication/transfer.py` with the study environment and clean local corpus named in the script. The script verifies the source corpus digest, token disjointness and fitted penalty. The published numerical summary and sufficient statistics reproduce the comparison on CPU without the corpus. For the simultaneous bands, run `OPENBLAS_NUM_THREADS=2 python scripts/internal_replication/depth_bands.py --verify`.
