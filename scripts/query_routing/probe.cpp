// Qwen3.5 text trunk, llama.cpp b12110. The caller owns the arrays for one forward.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
llama_model *model;
llama_context *ctx;
int width, final_layer;
struct Probe {
    int n, site, layer;
    const float *edit;
    float *source, *final;
    int captured_source = 0, captured_final = 0;
};
Probe run{};
// Optional same-length sequence-state intervention and last-position landmark readout.
float *block_capture = nullptr, *landmark_capture = nullptr;
const float *block_replacement = nullptr;
int landmark_hits = 0;

// Optional clean trace: blocks 15..23 at the last two positions, in block order.
float *trace = nullptr;
unsigned trace_mask = 0;
// Optional intervention at the last position, after the attention gate, before W_O.
int head_layer = -1, head_index = -1, head_hits = 0;
float *head_capture = nullptr;
const float *head_replacement = nullptr;

bool observe(ggml_tensor *node, bool ask, void *) {
    const std::string name(node->name);
    if (head_layer >= 0 && name == "attn_gated-" + std::to_string(head_layer)) {
        if (ask) return true;
        constexpr size_t count = 16 * 256;
        assert(node->type == GGML_TYPE_F32 && ggml_is_contiguous(node));
        assert(node->ne[0] == count && ggml_nelements(node) == int64_t(count * run.n));
        const size_t last = (run.n - 1) * count * sizeof(float);
        ggml_backend_tensor_get(node, head_capture, last, count * sizeof(float));
        if (head_replacement && head_index == -2) {
            // One contiguous gated-output patch, with one or more changed head slices.
            ggml_backend_tensor_set(node, head_replacement, last, count * sizeof(float));
        } else if (head_replacement) {
            assert(head_index >= 0 && head_index < 16);
            ggml_backend_tensor_set(node, head_replacement, last + head_index * 256 * sizeof(float),
                                    256 * sizeof(float));
        }
        ++head_hits;
        return true;
    }
    if (landmark_capture && name == "l_out-23") {
        if (ask) return true;
        assert(run.layer == 15 && node->type == GGML_TYPE_F32 && ggml_is_contiguous(node));
        assert(node->ne[0] == width && ggml_nelements(node) == int64_t(width) * run.n);
        ggml_backend_tensor_get(node, landmark_capture, size_t(run.n-1)*width*sizeof(float), width*sizeof(float));
        ++landmark_hits;
        if (!trace) return true;
    }
    if (trace) {
        for (int layer = 15; layer <= 23; ++layer) {
            if (name != "l_out-" + std::to_string(layer)) continue;
            if (ask) return true;
            assert(!run.edit && run.n >= 2 && ggml_is_contiguous(node));
            assert(node->type == GGML_TYPE_F32 && node->ne[0] == width);
            const size_t offset = size_t(run.n - 2) * width * sizeof(float);
            ggml_backend_tensor_get(node, trace + (layer-15)*2*width, offset, 2*width*sizeof(float));
            assert(!(trace_mask & (1u << (layer-15))));
            trace_mask |= 1u << (layer-15);
            // Source/final callbacks below may also need to capture this tensor.
            if (layer != run.layer && layer != final_layer) return true;
            break;
        }
    }
    if (name != "l_out-" + std::to_string(run.layer) &&
        name != "l_out-" + std::to_string(final_layer)) return false;
    if (ask) return true;
    assert(ggml_is_contiguous(node) && node->type == GGML_TYPE_F32);
    assert(node->ne[0] == width && ggml_nelements(node) == int64_t(width) * run.n);
    if (name == "l_out-" + std::to_string(run.layer)) {
        const auto bytes = sizeof(float) * width;
        const auto offset = bytes * run.site;
        ggml_backend_tensor_get(node, run.source, offset, bytes);
        if (block_capture)
            ggml_backend_tensor_get(node, block_capture, 0, ggml_nbytes(node));
        if (block_replacement) {
            assert(!run.edit);
            ggml_backend_tensor_set(node, block_replacement, 0, ggml_nbytes(node));
        }
        if (run.edit) {
            std::vector<float> patched(width);
            for (int j = 0; j < width; ++j) patched[j] = run.source[j] + run.edit[j];
            ggml_backend_tensor_set(node, patched.data(), offset, bytes);
        }
        ++run.captured_source;
    }
    if (name == "l_out-" + std::to_string(final_layer)) {
        ggml_backend_tensor_get(node, run.final, 0, ggml_nbytes(node));
        ++run.captured_final;
    }
    return true;
}
}

extern "C" int initialize(const char *path) {
    ggml_backend_load_all_from_path(std::getenv("LLAMA_BACKEND_DIR"));
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    model = llama_model_load_from_file(path, mp);
    assert(model);
    width = llama_model_n_embd(model);
    final_layer = llama_model_n_layer(model) - 1;
    assert(final_layer == 31 && (width == 4096 || width == 2560));
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.cb_eval = observe;
    ctx = llama_init_from_model(model, cp);
    assert(ctx);
    return width;
}

extern "C" int tokenize(const char *text, int *out, int capacity) {
    return llama_tokenize(llama_model_get_vocab(model), text, std::strlen(text),
                          out, capacity, false, true);
}

extern "C" void configure_head(int layer, int head, float *capture, const float *replacement) {
    head_layer = layer;
    head_index = head;
    head_capture = capture;
    head_replacement = replacement;
}

extern "C" void configure_dense(float *state, const float *replacement, float *landmark) {
    block_capture = state;
    block_replacement = replacement;
    landmark_capture = landmark;
}

extern "C" void configure_trace(float *capture) {
    trace = capture;
}

extern "C" int forward(int *ids, int n, int site, int layer, const float *edit,
                       float *source, float *final, int answer, int foil, float *margin) {
    assert(site >= 0 && site < n && n <= 256 && layer < final_layer);
    head_hits = 0;
    landmark_hits = 0;
    trace_mask = 0;
    run = {n, site, layer, edit, source, final};
    llama_memory_clear(llama_get_memory(ctx), true);
    auto batch = llama_batch_get_one(ids, n);
    std::vector<int8_t> outputs(n, 1);
    batch.logits = outputs.data();
    assert(llama_decode(ctx, batch) == 0);
    llama_synchronize(ctx);
    assert(trace_mask == (trace ? 511u : 0u));
    assert(landmark_hits == (landmark_capture ? 1 : 0));
    assert(head_hits == (head_layer >= 0 ? 1 : 0));
    assert(run.captured_source == 1 && run.captured_final == 1);
    const auto *logits = llama_get_logits_ith(ctx, n - 1);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    assert(answer >= 0 && answer < vocab && foil >= 0 && foil < vocab);
    *margin = logits[answer] - logits[foil];
    return int(std::max_element(logits, logits + vocab) - logits);
}
