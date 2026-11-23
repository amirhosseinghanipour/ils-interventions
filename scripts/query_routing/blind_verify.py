"""Recalculate the blinded readouts using Gram differences and nearest unit vectors."""
from pathlib import Path
import json
import numpy as np
import argparse

def audit(d,results,scale_file):
    expected=json.loads(results.read_text())
    scales=json.loads(scale_file.read_text())['scales']
    inputs=np.load(d/'inputs.npz');dic=np.load(d/'dictionary.npz')
    s=dic['source'].astype(float); w=dic['query'].astype(float)
    g=s@s.T
    lengths=np.sqrt(np.maximum(g.diagonal()[:,None]+g.diagonal()[None,:]-2*g,0))
    np.fill_diagonal(lengths,1)
    vec=inputs['delta'].astype(float); h=inputs['clean_query'].astype(float)
    projection=vec@s.T
    cos=(projection[:,None,:]-projection[:,:,None])/lengths
    for k in range(len(g)): cos[:,k,k]=-np.inf
    old,new=np.unravel_index(cos.reshape(len(vec),-1).argmax(1),g.shape)
    unit=lambda a:a/np.linalg.norm(a,axis=-1,keepdims=True)
    y=np.load(d/'targets.npz')['y'].astype(float)
    ss=((y-y.mean((0,1)))**2).sum()
    delta=w[new,-1]-w[old,-1]
    for j,layer in enumerate(('15','23','27','31')):
        center=w[:,j].mean(0)
        bank=unit(w[:,j]-center)
        # Squared Euclidean distance of unit vectors is monotone in cosine.
        chosen=np.stack([np.square(bank-unit(answer-center)).sum(-1).argmin() for answer in h[:,:,j].reshape(-1,w.shape[-1])]).reshape(-1,2)
        slopes=np.where(chosen==old[:,None],*scales)
        pred=slopes[:,:,None]*delta[:,None]
        val=1-((pred-y)**2).sum()/ss
        np.testing.assert_allclose(val,expected['r2'][layer]['estimate'],atol=1e-10)
    print('Independent endpoint search, query decoding and R² verified.')

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("data","results","color_scales"): p.add_argument(arg,type=Path)
    args=p.parse_args()
    audit(args.data,args.results,args.color_scales)
