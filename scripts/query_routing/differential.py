"""Matched prompt-local and training-averaged derivatives on the differentiable checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ilslens.suffix import QwenSuffix
from followup import rows
from run import save_json

LAYER = 11

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def ids(self,text):
        return np.asarray(self.tokenizer.encode(text,add_special_tokens=False),np.int32)
    def chat(self,facts,question):
        content=f"Answer with only the lowercase color word.\n{facts}\n{question}"
        return self.tokenizer.apply_chat_template([{"role":"user","content":content}],
                                                  tokenize=False,add_generation_prompt=True)


def response_function(replay,site):
    h = replay.states[LAYER]
    mask = torch.zeros_like(h[:,:,:1])
    mask[:,site] = 1
    return lambda d: replay.forward(LAYER,h + mask * d)[0,-1]


def main():
    p=argparse.ArgumentParser()
    for key in ["model","output"]: p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.model/"download-manifest.json").read_text())
    for name,meta in manifest["files"].items():
        if (args.model/name).is_file():
            with (args.model/name).open("rb") as f:
                assert hashlib.file_digest(f,"sha256").hexdigest()==meta["sha256"]
    save_json(args.output/"checkpoint.json",manifest)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float32,
          attn_implementation="eager").eval().requires_grad_(False).cuda()
    vocab=Tokenizer(AutoTokenizer.from_pretrained(args.model,local_files_only=True))
    refs=[]
    for split in ["train","confirm"]:
        info=rows(vocab,split,render_chat=vocab.chat)
        arrays={k:[] for k in ["x","y","small","local","mean_j"]}; numerical=[]
        for i,r in enumerate(info):
            replay=[QwenSuffix(model,torch.tensor([t],device="cuda")) for t in r["tokens"]]
            site=r["site"]
            source=replay[0].states[LAYER][0,site]
            torch.testing.assert_close(source,replay[1].states[LAYER][0,site],rtol=0,atol=1e-6)
            delta=replay[2].states[LAYER][0,site]-source
            zero=torch.zeros_like(delta)
            arrays["x"].append(delta.cpu().numpy())
            targets=[]; small=[]; local=[]; shifts=[]; tops=[]
            for side in [0,1]:
                rep=replay[side]
                f=response_function(rep,site)
                with torch.no_grad():
                    clean=f(zero)
                    torch.testing.assert_close(clean,rep.states[23][0,-1],rtol=1e-5,atol=1e-5)
                    full=f(delta)
                    short=f(.05*delta)
                    logit=model.lm_head(model.model.norm(torch.stack((clean,full))))
                    margin=logit[:,r["donor_token"]]-logit[:,r["correct"][0]]
                    shifts.append(float(margin[1]-margin[0]))
                    tops.append(logit.argmax(1).tolist())
                    targets.append((full-clean).cpu().numpy());small.append((short-clean).cpu().numpy())
                if split=="train" and i<8: refs.append(f)
                if split=="confirm" or (split=="train" and i<8):
                    tangent=torch.autograd.functional.jvp(f,zero,delta)[1]
                    local.append(tangent.cpu().numpy())
                    if split=="train":
                        err={}
                        for eps in (.01,.005):
                            fd=(f(eps*delta)-f(-eps*delta))/(2*eps)
                            err[str(eps)]=float((fd-tangent).norm()/tangent.norm().clamp_min(1e-12))
                        numerical.append(err)
            if split=="confirm":
                avg=torch.stack([torch.autograd.functional.jvp(fn,zero,delta)[1] for fn in refs]).mean(0)
                arrays["local"].append(local);arrays["mean_j"].append(avg.cpu().numpy())
            r.update(shifts=shifts,top=tops)
            arrays["y"].append(targets); arrays["small"].append(small)
            if (i+1)%16==0: print(split,i+1,flush=True)
        save_json(args.output/(split+".json"),info)
        np.savez(args.output/(split+".npz"),**{k:np.array(v) for k,v in arrays.items() if len(v)})
        if numerical: save_json(args.output/"numerical.json",numerical)

if __name__=="__main__": main()
