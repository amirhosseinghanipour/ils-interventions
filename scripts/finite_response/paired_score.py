"""Recompute error, answer fidelity, and paired uncertainty on unseen articles."""
import argparse
import json
from pathlib import Path

import numpy as np


def score(out,paired,original):
    read=lambda path:json.loads(path.read_text())
    rows=read(paired)
    old=set(r["article"] for r in read(original))
    assert old.isdisjoint(r["article"] for r in rows)
    test=[r for r in rows if r["split"]=="test"]
    assert len(test)==len({r["article"] for r in test})==64
    expected=[(r["article"],r[q]) for r in test for q in ("qid","other_qid")]
    indices=np.random.default_rng(747103).integers(64,size=(10000,64))
    bands=[]
    summary={}
    for name in ("qwen","smol"):
        records=read(out/(name+"-rows.json"))
        assert [(r["article"],r["qid"]) for r in records]==expected
        for r in records:
            seq=r["tokens"]; yes=seq["exact"]!=seq["no_change"]
            assert r["changed"]==yes and r["first"]==seq["exact"][0]
            for k,s in seq.items():
                if k!="exact":
                    assert r["methods"][k]["changed"]==(s!=seq["no_change"]) and r["methods"][k]["first"]==s[0]
        report={}
        for condition in ("all","edited_answer","other_answer"):
            relevant=[r for r in records if condition=="all" or r["kind"]==condition]
            def by_article(v):
                return np.array(v).reshape(64,-1).sum(axis=1)
            energy=by_article([r["energy"] for r in relevant])
            residual={k:by_article([r["methods"][k]["sse"] for r in relevant]) for k in relevant[0]["methods"]}
            table={}
            for k,err in residual.items():
                diff=err-residual["two_anchor"]
                value=diff.sum()/energy.sum()
                boot=diff[indices].sum(axis=1)/energy[indices].sum(axis=1)
                item=dict(normalized_error=float(err.sum()/energy.sum()),gain_with_two_anchor=float(value),
                          first_token_agreement=sum(r["methods"][k]["first"]==r["first"] for r in relevant),
                          correct_switch_prediction=sum(r["methods"][k]["changed"]==r["changed"] for r in relevant))
                if k in relevant[0]["text_agreement"]:
                    item.update(text_matches=sum(r["text_agreement"][k] for r in relevant),
                        text_matches_on_changed=sum(r["text_changed"] and r["text_agreement"][k] for r in relevant),
                        token_matches=sum(r["tokens"][k]==r["tokens"]["exact"] for r in relevant))
                if condition=="all" and k!="two_anchor":
                    bands.append((value,boot,item))
                table[k]=item
            report[condition]=dict(questions=len(relevant),text_changed=sum(r["text_changed"] for r in relevant),
                exact_ended=sum(r["exact_ended"] for r in relevant),methods=table)
        summary[name]=report
    se=np.array([b.std(ddof=1) for _,b,_ in bands])
    nonzero=se>1e-12
    standardized=np.stack([b-v for v,b,_ in bands])[nonzero]/se[nonzero,None]
    critical=float(np.quantile(abs(standardized).max(axis=0),.95))
    for (v,_,item),s in zip(bands,se,strict=True):
        item["simultaneous_ci95"]=[float(v-critical*s),float(v+critical*s)]
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("out","paired","original","summary"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    a.summary.write_text(json.dumps(score(a.out,a.paired,a.original),indent=2,allow_nan=False)+"\n")
