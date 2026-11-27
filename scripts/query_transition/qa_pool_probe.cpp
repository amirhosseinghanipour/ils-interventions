// Native Qwen3.5 b12110: capture or replace last-position head and MLP outputs.
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
std::array<std::vector<float>, layers> attn, mlp;
std::array<int, layers> seen_attn, seen_mlp;
bool capture;
uint32_t attn_mask, mlp_mask;
std::array<uint16_t, layers> head_masks;
bool observe(ggml_tensor *node, bool ask, void *) {
    const std::string name(node->name);
    const bool is_attn = name.rfind("attn_gated-", 0) == 0;
    if (!is_attn && name.rfind("ffn_out-", 0) != 0) return false;
    int layer = std::stoi(name.substr(name.find('-')+1));
    assert(layer >= 0 && layer < layers);
    if (!(((is_attn ? attn_mask : mlp_mask) >> layer) & 1u)) return false;
    if (ask) return true;
    auto &buf = (is_attn ? attn : mlp)[layer];
    auto &hits = (is_attn ? seen_attn : seen_mlp)[layer];
    ++hits;
    assert(hits == 1);
    int count = is_attn ? heads * head_dim : width;
    assert(node->type == GGML_TYPE_F32 && ggml_is_contiguous(node) && node->ne[0] == count);
    assert(ggml_nelements(node) == int64_t(count) * length);
    size_t offset = size_t(patch_position) * count * sizeof(float);
    if (capture) {
        buf.resize(count);
        ggml_backend_tensor_get(node, buf.data(), offset, count * sizeof(float));
    } else {
        assert(int(buf.size()) == count);
        if (!is_attn || head_masks[layer] == UINT16_MAX) {
            ggml_backend_tensor_set(node, buf.data(), offset, count * sizeof(float));
        } else {
            for (int h = 0; h < heads; ++h) {
                if ((head_masks[layer] >> h) & 1u) {
                    int begin = h * head_dim;
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
int forward(int *ids, int n, uint32_t a, uint32_t f, bool record, int position,
            bool all_outputs) {
    assert(n > 0 && n <= 256 && position >= 0 && position < n);
    attn_mask = a; mlp_mask = f; capture = record; length = n; patch_position = position;
    seen_attn.fill(0); seen_mlp.fill(0);
    llama_memory_clear(llama_get_memory(ctx), true);
    auto batch = llama_batch_get_one(ids, n);
    std::vector<int8_t> output(n, all_outputs ? 1 : 0);
    output.back() = 1;
    batch.logits = output.data();
    const int status = llama_decode(ctx, batch);
    assert(status == 0);
    llama_synchronize(ctx);
    for (int i=0;i<layers;++i) {
        assert(seen_attn[i] == int((attn_mask>>i)&1u));
        assert(seen_mlp[i] == int((mlp_mask>>i)&1u));
    }
    const float *logits = llama_get_logits_ith(ctx, all_outputs ? n-1 : 0);
    return int(std::max_element(logits, logits+llama_vocab_n_tokens(llama_model_get_vocab(model))) - logits);
}
extern "C" int pool_forward(int *ids, int n, uint32_t a, uint32_t f, int record, int h) {
    assert(h >= -1 && h < heads);
    head_masks.fill(h < 0 ? UINT16_MAX : uint16_t(1u << h));
    return forward(ids, n, a, f, record, n-1, true);
}
extern "C" int subset_forward(int *ids, int n, const uint16_t *masks, int record, int position) {
    std::copy_n(masks, layers, head_masks.begin());
    uint32_t occupied = 0;
    for (int i = 0; i < layers; ++i) {
        if (head_masks[i]) occupied |= 1u << i;
    }
    return forward(ids, n, occupied, 0, record, position, false);
}
extern "C" int vocabulary_size() {
    return llama_vocab_n_tokens(llama_model_get_vocab(model));
}
extern "C" const float *last_logits() {
    return llama_get_logits_ith(ctx, 0);
}
extern "C" int detokenize(const int *ids, int n, char *out, int capacity) {
    return llama_detokenize(llama_model_get_vocab(model), ids, n, out, capacity, true, false);
}
extern "C" bool is_eog(int id) {
    return llama_vocab_is_eog(llama_model_get_vocab(model), id);
}
