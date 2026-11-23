#include "llama.h"
#include "ggml-backend.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

const std::string prompt = "A scientist measured the temperature of the water at sunrise and again at noon.";
struct Probe {
    bool suffix;
    int sign, tokens, patches=0, reads=0;
    std::vector<float> source, target;
};
static bool inspect(ggml_tensor *t, bool ask, void *data) {
    auto &p = *static_cast<Probe *>(data);
    std::string name(t->name);
    bool patch = name == (p.suffix ? "model.input_embed" : "l_out-15");
    bool read = name == (p.suffix ? "l_out-15" : "l_out-31");
    if (ask) return patch || read;
    assert((patch || read) && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    assert(t->ne[0]==2560 && ggml_nelements(t)==2560*p.tokens);
    std::vector<float> h(ggml_nelements(t));
    ggml_backend_tensor_get(t,h.data(),0,ggml_nbytes(t));
    if (patch) {
        ++p.patches;
        if (!p.suffix && p.sign==0) p.source=h;
        if (!p.suffix) assert(h==p.source);
        else h=p.source;
        // Two prompt positions give a fixed direction for this replay diagnostic.
        int pos=p.tokens/2;
        for(int j=0;j<2560;++j)
            h[pos*2560+j]+=p.sign*0.05f*(p.source[2560+j]-p.source[pos*2560+j]);
        if(p.suffix || p.sign) ggml_backend_tensor_set(t,h.data(),0,ggml_nbytes(t));
    } else { ++p.reads; p.target=std::move(h); }
    return true;
}
int main(int argc,char **argv) {
    assert(argc==3);
    ggml_backend_load_all_from_path(std::getenv("LLAMA_BACKEND_DIR")); llama_backend_init();
    Probe p{};
    std::vector<llama_token> ids;
    for(int mode=0;mode<=1;++mode) {
        p.suffix=mode;
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        auto *model=llama_model_load_from_file(argv[1+mode],mp);assert(model);
        auto *vocab=llama_model_get_vocab(model);
        int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,false);
        std::vector<llama_token> local(n);
        assert(llama_tokenize(vocab,prompt.data(),prompt.size(),local.data(),n,true,false)==n);
        if(!mode){ids=local; for(int i:ids)std::cout<<i<<" ";std::cout<<"\n";} else assert(local==ids);
        p.tokens=n;
        for(int sign : {0,1,-1}) {
            p.sign=sign; p.patches=p.reads=0;
            auto cp=llama_context_default_params();
            cp.n_ctx=128;cp.n_batch=128;cp.n_ubatch=128;cp.n_threads=4;cp.n_threads_batch=4;
            cp.cb_eval=inspect;cp.cb_eval_user_data=&p;
            auto *ctx=llama_init_from_model(model,cp);assert(ctx);
            auto b=llama_batch_get_one(ids.data(),n);
            std::vector<int8_t> all(n,1);b.logits=all.data();
            assert(llama_decode(ctx,b)==0);llama_synchronize(ctx);
            std::cerr<<"observed "<<p.patches<<" patches "<<p.reads<<" outputs\n";
            assert(p.patches==1 && p.reads==1);
            std::ofstream out(std::string(mode?"suffix":"full")+std::to_string(sign)+".f32",std::ios::binary);
            out.write((const char*)p.target.data(),p.target.size()*sizeof(float));assert(out.good());
            llama_free(ctx);
        }
        llama_model_free(model);
    }
    llama_backend_free();
}
