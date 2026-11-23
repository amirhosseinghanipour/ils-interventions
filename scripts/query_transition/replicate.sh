#!/usr/bin/env bash
set -euo pipefail
: "${MODEL9:?}" "${MODEL4:?}" "${LLAMA_SRC:?}" "${LLAMA_BIN:?}" "${PYTHON:?}" "${OUT:?}" "${BLIND:?}"
study=$(cd "$(dirname "$0")" && pwd)
if [[ -n "${HEAVY_LOCK:-}" ]]; then exec 9<"$HEAVY_LOCK"; flock -x 9; fi
mkdir "$OUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_CACHE_PATH="$OUT/cuda-cache"
export LLAMA_BACKEND_DIR="$LLAMA_BIN" LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s  %s\n' a686d88ec1e6881f9bf161526826cd6d6874b7f0e80e0f79acf6144a132c5d7e "$MODEL9" 13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983 "$MODEL4" | sha256sum --check
g++ -O2 -fPIC -shared -std=c++17 -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" "$study/../query_routing/probe.cpp" \
    -L"$LLAMA_BIN" -Wl,-rpath,"$LLAMA_BIN" -Wl,-rpath-link,"$LLAMA_BIN" -lllama -lggml-base -lggml -o "$OUT/probe.so"
selection="$OUT/blocks/9b/selection.json"
for size in 9b 4b; do
    args=()
    model="$MODEL9"
    if [[ "$size" == 4b ]]; then args+=(--selection "$selection"); model="$MODEL4"; fi
    "$PYTHON" "$study/collect.py" "$OUT/probe.so" "$model" "$BLIND/$size" "$OUT/blocks/$size" "${args[@]}"
    "$PYTHON" "$study/heads.py" "$OUT/probe.so" "$model" "$selection" "$OUT/heads/$size"
    scales="$study/../../artifacts/query-routing/animals/$size/prototypes.json"
    "$PYTHON" "$study/materials.py" "$OUT/probe.so" "$model" "$BLIND/$size" "$scales" "$OUT/materials/$size"
    "$PYTHON" "$study/target_center.py" "$BLIND/$size" "$OUT/materials/$size" "$scales" "$OUT/materials/$size/target-center.json"
    "$PYTHON" "$study/verify_materials.py" "$OUT/materials/$size" "$BLIND/$size" "$scales"
    "$PYTHON" "$study/transport.py" "$OUT/probe.so" "$model" "$OUT/blocks/$size" "$selection" "$OUT/transport/$size"
done
"$PYTHON" "$study/score.py" "$OUT" "$OUT/summary"
"$PYTHON" "$study/verify.py" "$OUT" "$OUT/summary"
for size in 9b 4b; do
    "$PYTHON" "$study/affine.py" "$OUT/transport/$size/responses.npz" "$OUT/summary/affine-$size.json"
    "$PYTHON" "$study/slots.py" "$BLIND/$size" "$OUT/blocks/$size/food-depth.npz" "$OUT/summary/slots-$size.json"
    "$PYTHON" "$study/centering.py" "$BLIND/$size" "$OUT/summary/dense-centering-$size.json" --dense "$OUT/blocks/$size/food-depth.npz"
done
if [[ -n "${SQUAD:-}" ]]; then
    for size in 9b 4b; do
        model="$MODEL9"
        if [[ "$size" == 4b ]]; then model="$MODEL4"; fi
        "$PYTHON" "$study/natural.py" "$OUT/probe.so" "$model" "$SQUAD" "$OUT/heads/$size/selection.json" "$OUT/natural/$size"
        "$PYTHON" "$study/real_control.py" "$OUT/probe.so" "$model" "$OUT/blocks/$size" "$OUT/natural/$size" "$SQUAD" "$selection" "$OUT/crossed/$size"
    done
    "$PYTHON" "$study/verify_natural.py" "$OUT" "$SQUAD"
fi
