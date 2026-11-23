#!/usr/bin/env bash
set -euo pipefail
: "${MODEL9:?}" "${MODEL4:?}" "${LLAMA_SRC:?}" "${LLAMA_BIN:?}" "${PYTHON:?}" "${OUT:?}"
root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
cd "$root/scripts/query_routing"
if [[ -n "${HEAVY_LOCK:-}" ]]; then exec 9<"$HEAVY_LOCK"; flock -x 9; fi
export CUDA_CACHE_PATH="$OUT/cuda-cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LLAMA_BACKEND_DIR="$LLAMA_BIN" LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s  %s\n' a686d88ec1e6881f9bf161526826cd6d6874b7f0e80e0f79acf6144a132c5d7e "$MODEL9" 13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983 "$MODEL4" | sha256sum --check
g++ -std=c++17 -O2 -fPIC -shared -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" probe.cpp \
  -L"$LLAMA_BIN" -Wl,-rpath,"$LLAMA_BIN" -Wl,-rpath-link,"$LLAMA_BIN" -lllama -lggml-base -lggml -o "$OUT/probe.so"
for size in 9b 4b; do
    flags=()
    model="$MODEL4"
    if [[ $size == 9b ]]; then flags+=(--heads); model="$MODEL9"; fi
    "$PYTHON" followup.py --library "$OUT/probe.so" --model "$model" --output "$OUT/$size" "${flags[@]}"
done
