// Native Qwen3.5 b12110 probe for the confirmatory scheduler study.
// Same capture/replace semantics as qa_pool_probe.cpp (gated attention head outputs before
// the output projection, one sequence position). Additive validation hooks: read the
// post-projection attention output at the patch position and overwrite a captured buffer
// with an arbitrary vector. With an empty validation mask the observed graph is unchanged.
#ifdef NDEBUG
#error "The probe requires runtime tensor checks; compile with assertions enabled."
#endif
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
llama_model *model;
llama_context *ctx;
int width, length, patch_position;
constexpr int layers = 32, heads = 16, head_dim = 256;
std::array<std::vector<float>, layers> attn, out;
std::array<int, layers> seen_attn, seen_out;
bool capture;
uint32_t attn_mask, out_mask;
std::array<uint16_t, layers> head_masks;

int layer_of(const std::string &name) {
    int layer = std::stoi(name.substr(name.rfind('-')+1));
    assert(layer >= 0 && layer < layers);
    return layer;
}

bool observe(ggml_tensor *node, bool ask, void *) {
    const std::string name(node->name);
    const bool is_attn = name.rfind("attn_gated-", 0) == 0;
    const bool is_out = name.rfind("attn_output-", 0) == 0;
    if (!is_attn && !is_out) return false;
    const int layer = layer_of(name);
    if (!(((is_attn ? attn_mask : out_mask) >> layer) & 1u)) return false;
    if (ask) return true;
    const int count = is_attn ? heads * head_dim : width;
    assert(node->type == GGML_TYPE_F32 && ggml_is_contiguous(node) && node->ne[0] == count);
    assert(ggml_nelements(node) == int64_t(count) * length);
    const size_t offset = size_t(patch_position) * count * sizeof(float);
    if (is_out) {
        ++seen_out[layer];
        assert(seen_out[layer] == 1);
        out[layer].resize(count);
        ggml_backend_tensor_get(node, out[layer].data(), offset, count * sizeof(float));
        return true;
    }
    auto &buf = attn[layer];
    ++seen_attn[layer];
    assert(seen_attn[layer] == 1);
    if (capture) {
        buf.resize(count);
        ggml_backend_tensor_get(node, buf.data(), offset, count * sizeof(float));
    } else {
        assert(int(buf.size()) == count);
        if (head_masks[layer] == UINT16_MAX) {
            ggml_backend_tensor_set(node, buf.data(), offset, count * sizeof(float));
        } else {
            for (int h = 0; h < heads; ++h) {
                if ((head_masks[layer] >> h) & 1u) {
                    const int begin = h * head_dim;
                    ggml_backend_tensor_set(node, buf.data() + begin,
                        offset + begin * sizeof(float), head_dim * sizeof(float));
                }
            }
        }
    }
    return true;
}
}

extern "C" int initialize(const char *path, int vocab_only) {
    ggml_backend_load_all_from_path(std::getenv("LLAMA_BACKEND_DIR"));
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = vocab_only ? 0 : 999;
    mp.vocab_only = vocab_only;
    model = llama_model_load_from_file(path, mp);
    assert(model);
    if (vocab_only) return 0;
    assert(llama_model_n_layer(model) == layers);
    width = llama_model_n_embd(model);
    assert(width==2560 || width==4096);
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.cb_eval = observe;
    ctx = llama_init_from_model(model, cp);
    assert(ctx);
    return width;
}
extern "C" int tokenize(const char *s, int *ids, int capacity) {
    return llama_tokenize(llama_model_get_vocab(model), s, std::strlen(s), ids, capacity, false, true);
}
// Patch (or capture) the masked heads at one position; optionally read attn_output there.
extern "C" int probe_forward(int *ids, int n, const uint16_t *masks, int record, int position,
                             uint32_t output_layers) {
    assert(n > 0 && n <= 256 && position >= 0 && position < n);
    std::copy_n(masks, layers, head_masks.begin());
    attn_mask = 0;
    for (int i = 0; i < layers; ++i) {
        if (head_masks[i]) attn_mask |= 1u << i;
    }
    out_mask = output_layers; capture = record; length = n; patch_position = position;
    seen_attn.fill(0); seen_out.fill(0);
    llama_memory_clear(llama_get_memory(ctx), true);
    auto batch = llama_batch_get_one(ids, n);
    std::vector<int8_t> output(n, 0);
    output.back() = 1;
    batch.logits = output.data();
    const int status = llama_decode(ctx, batch);
    assert(status == 0);
    llama_synchronize(ctx);
    for (int i=0;i<layers;++i) {
        assert(seen_attn[i] == int((attn_mask>>i)&1u));
        assert(seen_out[i] == int((out_mask>>i)&1u));
    }
    const float *logits = llama_get_logits_ith(ctx, -1);
    return int(std::max_element(logits, logits+llama_vocab_n_tokens(llama_model_get_vocab(model))) - logits);
}
// kind 0: captured head-output buffer (heads*head_dim); kind 1: attn_output row (width).
extern "C" int read_buffer(int layer, int kind, float *dst, int capacity) {
    assert(layer >= 0 && layer < layers);
    const auto &buf = kind ? out[layer] : attn[layer];
    assert(int(buf.size()) <= capacity);
    std::copy(buf.begin(), buf.end(), dst);
    return int(buf.size());
}
extern "C" void write_buffer(int layer, const float *src, int n) {
    assert(layer >= 0 && layer < layers && n == heads * head_dim);
    attn[layer].assign(src, src + n);
}
extern "C" int vocabulary_size() {
    return llama_vocab_n_tokens(llama_model_get_vocab(model));
}
extern "C" const float *last_logits() {
    return llama_get_logits_ith(ctx, -1);
}
extern "C" int detokenize(const int *ids, int n, char *out_text, int capacity) {
    return llama_detokenize(llama_model_get_vocab(model), ids, n, out_text, capacity, true, false);
}
extern "C" bool is_eog(int id) {
    return llama_vocab_is_eog(llama_model_get_vocab(model), id);
}
