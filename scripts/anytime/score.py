"""Score only states that had already reached the caller at each deadline."""
import argparse
import json
from pathlib import Path

import numpy as np


NAMES = dict(midpoint='Midpoint + verify', hvp2='2 HVP', hvp4='4 HVP',
             hvp8='8 HVP', complement='Exact-only complement', dense='Dense ridge',
             proxyspex_fourier='ProxySPEX adaptation', cached_exact='Batched exact')


def at(row, arm, t):
    events = row['traces'][arm]['events']
    return next((e for e in reversed(events) if e['ms'] <= t),
                dict(oracle_effect=0., i=0, exact=0, mae=None, stage='no-op'))


def interval(v, rows):
    by = {}
    for x, row in zip(v, rows):
        by.setdefault(row['template'], []).append(x)
    unit = np.array([np.mean(xs) for xs in by.values()])
    rng = np.random.default_rng(20654)
    draws = rng.choice(unit, size=(10000,len(unit)),replace=True).mean(axis=1)
    return [float(np.mean(v)), *map(float,np.quantile(draws,[.025,.975]))]


def cell(rows, profile):
    cap = profile['budget_ms']
    times = np.unique(np.append(np.linspace(0,cap,41), [cap*.5,cap*.75,cap]))
    curves, paired = {}, {}
    for arm in NAMES:
        curve = []
        for t in times:
            ev = [at(row,arm,t) for row in rows]
            regrets = [row['pool_best']-e['oracle_effect'] for row,e in zip(rows, ev)]
            maes = [e['mae'] for e in ev if e['mae'] is not None]
            curve.append(dict(ms=t, regret=float(np.mean(regrets)),
                              effect=float(np.mean([e['oracle_effect'] for e in ev])),
                              mae=float(np.mean(maes)) if maes else None,
                              n_mae=len(maes), exact=float(np.mean([e['exact'] for e in ev]))))
        curves[arm] = curve
    for arm in ['hvp2','hvp4','hvp8']:
        paired[arm] = {}
        for frac in [.5,.75,1.]:
            a = [at(row,arm,frac*cap) for row in rows]
            b = [at(row,'complement',frac*cap) for row in rows]
            good = [i for i,(x,y) in enumerate(zip(a,b)) if x['mae'] is not None and y['mae'] is not None]
            paired[arm][frac] = dict(
                effect_gain=interval([x['oracle_effect']-y['oracle_effect'] for x,y in zip(a,b)],rows),
                mae_reduction=interval([b[i]['mae']-a[i]['mae'] for i in good],
                                      [rows[i] for i in good]) if good else None,n_mae=len(good))
    gate = {}
    for tau in [.02,.1]:
        hard = [r for r in rows if r['signal']['midpoint_mae'] > tau]
        flags = lambda r:r['signal']['first8_mae'] <= tau and r['signal']['sketch_energy'] <= tau
        gate[tau] = dict(abstain=sum(map(flags,rows)), hard=len(hard), false_abstain=sum(map(flags,hard)))
    halves = np.array([r['signal']['boolean_halves'] for r in rows])
    truth = halves.mean(1)
    def centered(x):
        return (x-x.mean(axis=-1,keepdims=True)).ravel()
    faith = {}
    for key in ['local','sketch']:
        h = np.array([r['signal'][key] for r in rows])
        h,y = centered(h),centered(truth)
        faith[key] = dict(r=float(np.corrcoef(h,y)[0,1]), r2_unit_scale=float(1-np.sum((h-y)**2)/np.sum(y**2)))
    faith['boolean_split_half_r'] = float(np.corrcoef(centered(halves[:,0]),centered(halves[:,1]))[0,1])
    drifts = [abs(e['paid_effect']-e['oracle_effect']) for r in rows for x in r['traces'].values() for e in x['events']]
    return dict(n=len(rows),templates=len({r['template'] for r in rows}),curves=curves,paired=paired,
                gate=gate,faithfulness=faith,paid_vs_canonical_max=max(drifts),pool_best=np.mean([r['pool_best'] for r in rows]))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args = p.parse_args()
    summary = {}
    colors = ['#41454a','#70a6cb','#136fa7','#3e4b9d','#d06624','#00977d','#b04a84','#696a16']
    cells = sorted(args.root.glob('*/*/profile.json'))
    fig, axes = plt.subplots(len(cells),2,figsize=(10,2.8*len(cells)),squeeze=False)
    for ax,path in zip(axes,cells):
        prof = json.loads(path.read_text())
        rows = [json.loads(p.read_text()) for p in sorted(path.parent.glob('*-*-*.json'))]
        if len(rows) != 24:
            raise ValueError(f'{path.parent}: expected 24 held-out prompts, found {len(rows)}')
        label = ('SmolLM2-1.7B routes' if prof['task']=='route' else 'Qwen2.5-0.5B gifts') + ' · K=' + path.parent.name
        s = summary[label] = cell(rows,prof)
        for (arm,name),color in zip(NAMES.items(),colors):
            curve=s['curves'][arm]
            for a,metric in zip(ax,['mae','regret']):
                if metric=='mae' and arm=='cached_exact':
                    continue
                keep=[p for p in curve if metric!='mae' or p['n_mae']==len(rows)]
                a.plot([p['ms'] for p in keep],[p[metric] for p in keep],label=name,color=color,
                       linewidth=2 if arm in ['hvp4','complement'] else 1.4,
                       linestyle='--' if arm in ['hvp2','hvp8','cached_exact'] else '-')
                a.grid(alpha=.18)
        ax[0].set_title(label+' | unseen-mask MAE',loc='left',fontsize=11)
        ax[1].set_title(label+' | verified pool regret',loc='left',fontsize=11)
        for a in ax:
            a.set(xlabel='Elapsed wall time (ms)',ylabel='Logits, lower is better')
    handles, labels = axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(args.root/'peer-curves.png',dpi=170)
    (args.root/'summary.json').write_text(json.dumps(summary,indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
