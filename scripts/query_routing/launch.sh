#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set MODEL}" "${LLAMA_SRC:?set LLAMA_SRC}" "${LLAMA_BIN:?set LLAMA_BIN}" "${PYTHON:?set PYTHON}" "${OUT:?set OUT}"
root=$(cd "$(dirname "$0")/../.." && pwd)
study="$root/scripts/query_routing"
mkdir -p "$OUT"
cd "$study"
sha256sum protocol.json probe.cpp run.py launch.sh > "$OUT/source.sha256"
echo "Waiting for compute slot at $(date -u -Iseconds)"
if [[ -n "${HEAVY_LOCK:-}" ]]; then
    exec 9<"$HEAVY_LOCK"
    flock -x 9
    # Laptop jobs use separate lock files. Read these as the shared local gate does.
    if [[ -d "$HOME/ci/.jobs" ]]; then
        shopt -s nullglob
        busy() {
            local f id
            for f in "$HOME"/ci/.jobs/*.sh; do
                id=$(basename "$f" .sh)
                if [[ ! -e "$HOME/ci/.jobs/$id.rc" ]] && tmux has-session -t "=$id" 2>/dev/null; then return 0; fi
            done
            for f in "$HOME"/ci/.locks/*; do
                if [[ -f "$f" ]] && ! flock -n -s "$f" true; then return 0; fi
            done
            return 1
        }
        while busy; do sleep 20; done
    fi
fi
echo "Compute slot acquired at $(date -u -Iseconds)"
sha256sum --check "$OUT/source.sha256"
expected=$("$PYTHON" -c 'import json; print(json.load(open("protocol.json"))["sha256"])')
printf '%s  %s\n' "$expected" "$MODEL" | sha256sum --check
export LLAMA_BACKEND_DIR="$LLAMA_BIN" LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$OUT/cuda-cache" TMPDIR="$OUT/tmp"
mkdir -p "$CUDA_CACHE_PATH" "$TMPDIR"
g++ -std=c++17 -O2 -fPIC -shared -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" probe.cpp \
    -L"$LLAMA_BIN" -Wl,-rpath,"$LLAMA_BIN" -Wl,-rpath-link,"$LLAMA_BIN" \
    -lllama -lggml-base -lggml -o "$OUT/probe.so"
sha256sum "$OUT/probe.so" "$LLAMA_BIN"/libllama.so "$LLAMA_BIN"/libggml-cuda.so > "$OUT/backend.sha256"
"$PYTHON" run.py --model "$MODEL" --library "$OUT/probe.so" --output "$OUT/data"
