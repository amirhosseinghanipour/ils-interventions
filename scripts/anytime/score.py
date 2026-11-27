"""Score only states that had already reached the caller at each deadline."""
import argparse
import json
from pathlib import Path

import numpy as np


NAMES = dict(midpoint='Midpoint + verify', hvp2='2 HVP', hvp4='4 HVP',
             hvp8='8 HVP', complement='Exact-only complement', dense='Dense ridge',
             proxyspex_fourier='ProxySPEX adaptation', cached_exact='Batched exact',
             cached_exact_stream='Exact, shorter batches')


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
    curves, paired, delivery = {}, {}, {}
    for r in rows:
        assert len(r['gold']) == len(r['test_y']) == 512
        decision_labels = r['gold'][2::2] + [0.] if r.get('compact') else r['gold']
        assert r['pool_best'] == max(decision_labels)
        assert set(r['traces']) == set(NAMES)
        for trace in r['traces'].values():
            prev = dict(ms=0, paid_effect=0, exact=0)
            for e in trace['events']:
                assert e['ms'] >= prev['ms'] and e['exact'] >= prev['exact']
                assert e['paid_effect'] >= prev['paid_effect']
                assert e['oracle_effect'] == r['gold'][e['i']]
                assert e['beta'] is None or 0 <= e['beta'] <= 1
                assert e['mae'] is None or e['mae'] >= 0
                prev = e
            assert trace['return_ms'] >= prev['ms']
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
        delivery[arm] = dict(
            late_final_calls=sum(r['traces'][arm]['return_ms'] > cap for r in rows),
            withheld_late_events=sum(e['ms'] > cap for r in rows for e in r['traces'][arm]['events']),
            initial_delivery_median_ms=float(np.median([r['traces'][arm]['events'][0]['ms'] for r in rows if r['traces'][arm]['events']])),
            paid_labels_mean_at_deadline=curve[-1]['exact'],
            exact_pool_optima=sum(at(r, arm, cap)['oracle_effect'] == r['pool_best'] for r in rows))
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
        corr = float(np.corrcoef(h,y)[0,1])
        faith[key] = dict(r=corr, r_squared=corr**2, r2_unit_scale=float(1-np.sum((h-y)**2)/np.sum(y**2)))
    faith['boolean_split_half_r'] = float(np.corrcoef(centered(halves[:,0]),centered(halves[:,1]))[0,1])
    drifts = [abs(e['paid_effect']-e['oracle_effect']) for r in rows for x in r['traces'].values() for e in x['events']]
    return dict(n=len(rows),templates=len({r['template'] for r in rows}),curves=curves,paired=paired,
                delivery=delivery, full_patch_optimal=sum(r['gold'][1]==r['pool_best'] for r in rows),
                noop_optimal=sum(r['pool_best']==0 for r in rows),
                initial_midpoint_optimal=sum(r['traces']['midpoint']['events'][0]['oracle_effect']==r['pool_best'] for r in rows),
                initial_midpoint_regret=float(np.mean([r['pool_best']-r['traces']['midpoint']['events'][0]['oracle_effect'] for r in rows])),
                gate=gate,faithfulness=faith,paid_vs_canonical_max=max(drifts),pool_best=np.mean([r['pool_best'] for r in rows]))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args = p.parse_args()
    summary = {}
    colors = ['#41454a','#70a6cb','#136fa7','#3e4b9d','#d06624','#00977d','#b04a84','#696a16','#97653f']
    cells = sorted(args.root.glob('*/*/profile.json'))
    fig, axes = plt.subplots(len(cells),2,figsize=(10,2.8*len(cells)),squeeze=False)
    for ax,path in zip(axes,cells):
        prof = json.loads(path.read_text())
        rows = [json.loads(p.read_text()) for p in sorted(path.parent.glob('*-*-*.json'))]
        if len(rows) != 24:
            raise ValueError(f'{path.parent}: expected 24 held-out prompts, found {len(rows)}')
        supplemental = json.loads((path.parent/'streamed_exact.json').read_text())
        assert [r['id'] for r in rows] == [r['id'] for r in supplemental]
        for r, trace in zip(rows,supplemental):
            for e in trace['events']:
                e.update(oracle_effect=r['gold'][e['i']], mae=None, beta=None, stage='exact')
            r['traces']['cached_exact_stream'] = trace
        label = ('SmolLM2-1.7B' if path.parent.parent.name=='large' else 'Qwen2.5-0.5B') + ' ' + prof['task'] + ' · K=' + path.parent.name
        s = summary[label] = cell(rows,prof)
        for (arm,name),color in zip(NAMES.items(),colors):
            curve=s['curves'][arm]
            for a,metric in zip(ax,['mae','regret']):
                if metric=='mae' and arm.startswith('cached_exact'):
                    continue
                keep=[p for p in curve if metric!='mae' or p['n_mae']==len(rows)]
                a.plot([p['ms'] for p in keep],[p[metric] for p in keep],label=name,color=color,
                       linewidth=2 if arm in ['hvp4','complement'] else 1.4,
                       linestyle='--' if arm in ['hvp2','hvp8','cached_exact','cached_exact_stream'] else '-')
                a.grid(alpha=.18)
        ax[0].set_title(label+' | unseen-mask MAE',loc='left',fontsize=11)
        ax[1].set_title(label+' | verified pool regret',loc='left',fontsize=11)
        for a in ax:
            a.set(xlabel='Elapsed wall time (ms)',ylabel='Logits, lower is better')
    handles, labels = axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(args.root/'peer-curves.png',dpi=170)
    (args.root/'summary.json').write_text(json.dumps(summary,indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
