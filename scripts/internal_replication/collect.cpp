// Direct antithetic interventions on complete prompt forwards; no generation.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using Vector=std::vector<float>;
constexpr int tokens=64;
const std::array<int,5> layers={7,15,23,30,31};
struct Capture {
    int d=0,site=0,layer=-1,sign=0;
    std::map<int,Vector> sites;
    Vector delta,final;
    bool patched=false;
};

bool inspect(ggml_tensor *node,bool ask,void *user) {
    auto &c=*static_cast<Capture*>(user);
    std::string name=node->name;
    if(name.rfind("l_out-",0)!=0)return false;
    int layer=std::stoi(name.substr(6));
    bool clean=c.sign==0;
    bool wanted=clean ? std::find(layers.begin(),layers.end(),layer)!=layers.end() : layer==31 || layer==c.layer;
    if(ask)return wanted;
    assert(wanted && ggml_is_contiguous(node) && node->type==GGML_TYPE_F32 && ggml_nelements(node)==tokens*c.d);
    size_t offset=size_t(c.site)*c.d*sizeof(float);
    if(clean || layer==c.layer) {
        Vector h(c.d);
        ggml_backend_tensor_get(node,h.data(),offset,c.d*sizeof(float));
        if(clean)c.sites[layer]=h;
        else {
            for(int j=0;j<c.d;++j)h[j]+=c.sign*c.delta[j];
            ggml_backend_tensor_set(node,h.data(),offset,c.d*sizeof(float));
            c.patched=true;
        }
    }
    if(!clean && layer==31){
        c.final.resize(tokens*c.d);
        ggml_backend_tensor_get(node,c.final.data(),0,ggml_nbytes(node));
    }
    return true;
}
std::vector<llama_token> tokenize(const llama_vocab *v,const std::string &text){
    int n=-llama_tokenize(v,text.data(),text.size(),nullptr,0,false,false);
    assert(n>=tokens);
    std::vector<llama_token> ids(n);
    assert(llama_tokenize(v,text.data(),text.size(),ids.data(),n,false,false)==n);
    ids.resize(tokens);return ids;
}
void forward(llama_context *ctx,std::vector<llama_token> &ids,Capture &c){
    llama_memory_clear(llama_get_memory(ctx),true);
    c.patched=false;c.final.clear();c.sites.clear();
    auto b=llama_batch_get_one(ids.data(),tokens);
    std::array<int8_t,tokens> all;all.fill(1);b.logits=all.data();
    assert(llama_decode(ctx,b)==0);llama_synchronize(ctx);
    assert(c.sign ? c.patched && !c.final.empty() : c.sites.size()==layers.size());
}
int main(int argc,char **argv){
    assert(argc==5);
    ggml_backend_load_all_from_path(std::getenv("LLAMA_BACKEND_DIR"));llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;
    auto *model=llama_model_load_from_file(argv[1],mp);assert(model && llama_model_n_layer(model)>=32);
    Capture c;c.d=llama_model_n_embd(model);
    auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=tokens;cp.n_ubatch=tokens;
    cp.n_threads=4;cp.n_threads_batch=4;cp.cb_eval=inspect;cp.cb_eval_user_data=&c;
    auto *ctx=llama_init_from_model(model,cp);assert(ctx);
    std::ifstream input(argv[2]);const fs::path root(argv[3]);fs::create_directories(root);
    int limit=std::stoi(argv[4]);std::string line,source,donor;
    for(int i=0;i<limit && std::getline(input,line);++i){
        int t,u;std::istringstream loc(line);assert(loc>>t>>u);
        assert(std::getline(input,source)&&std::getline(input,donor));
        auto path=root/(std::to_string(i)+".f32");
        if(fs::exists(path)){assert(fs::file_size(path)==25ULL*c.d*sizeof(float));continue;}
        auto a=tokenize(llama_model_get_vocab(model),source),b=tokenize(llama_model_get_vocab(model),donor);
        c.sign=0;c.site=u;forward(ctx,b,c);auto donors=c.sites;
        c.site=t;forward(ctx,a,c);auto sources=c.sites;
        std::ofstream file(path.string()+".tmp",std::ios::binary);
        for(int layer:layers){
            c.layer=layer;c.delta.resize(c.d);
            for(int j=0;j<c.d;++j)c.delta[j]=0.05f*(donors.at(layer)[j]-sources.at(layer)[j]);
            file.write((char*)c.delta.data(),c.d*sizeof(float));
            c.sign=1;forward(ctx,a,c);auto plus=c.final;
            c.sign=-1;forward(ctx,a,c);
            for(int k=0;k<t*c.d;++k)assert(std::abs(plus[k]-c.final[k])<=1e-6f);
            for(double gamma:{0.,0.5,0.9,1.}){
                std::vector<double> sum(c.d);double total=0,w=1;
                for(int pos=t;pos<tokens-1;++pos,w*=gamma){
                    total+=w;
                    for(int j=0;j<c.d;++j){int k=pos*c.d+j;sum[j]+=w*(plus[k]-c.final[k])*0.5;}
                }
                Vector y(c.d);
                for(int j=0;j<c.d;++j)y[j]=float(sum[j]/total);
                file.write((char*)y.data(),c.d*sizeof(float));
            }
        }
        file.close();assert(file.good());fs::rename(path.string()+".tmp",path);
        std::cout<<"completed prompt "<<i<<std::endl;
    }
    llama_free(ctx);llama_model_free(model);llama_backend_free();
}
