# One-prompt Qwen3.5-4B suffix replay

The full reference runs fresh text-only CPU contexts for clean and signed perturbations at block 15. The suffix file contains trunk blocks 16–31, renumbered 0–15, plus the original shared embedding/output tensors. The four-block attention pattern and original quantized tensor bytes are preserved. The suffix runtime replaces its embedding lookup with the cached full-sequence boundary residual and applies the same perturbation. All tokens are output so the final pre-normalization residual is available at every position. The optional next-token-prediction module is unused.

Run from an empty working directory with the study Python environment (NumPy and the dependencies of gguf-py). Set SRC to the unpacked llama.cpp b12110 source directory, BIN to the matching unpacked Ubuntu x64 CPU binary directory, SCRIPTS to this directory, and MODEL to the Q4_K_M GGUF. Compile with assertions enabled.

```sh
PYTHONPATH="$SRC/gguf-py" python "$SCRIPTS/make_suffix.py" "$MODEL" suffix.gguf
g++ -O2 -I"$SRC/include" -I"$SRC/ggml/include" "$SCRIPTS/probe.cpp" \
    -L"$BIN" -Wl,-rpath,"$BIN" -Wl,-rpath-link,"$BIN" -lllama -lggml-base -lggml -o probe
LLAMA_BACKEND_DIR="$BIN" LD_LIBRARY_PATH="$BIN" ./probe "$MODEL" suffix.gguf > ids.txt
python "$SCRIPTS/summarize.py" "$MODEL" suffix.gguf > results.json
```

The result records SHA-256 hashes for both models and the six output arrays. This is a numerical intervention diagnostic on a quantized model. Held-out predictive coverage requires separate fitting and evaluation prompts.
