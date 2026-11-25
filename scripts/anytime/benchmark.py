"""Profile and run a synchronized, resident-model, prompt-specific intervention search."""
import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import engine


def clock():
    return perf_counter() * 1000


def measured(call):
    start = clock()
    value = call()
    return value, clock() - start


def masks_for(k, seed):
    rng = np.random.default_rng(seed)
    # Independent complement pairs. Include no-op and full patch only in the search pool.
    half = rng.integers(2, size=(511, k), dtype=np.int8)
    search = np.vstack([np.zeros((1, k), np.int8), half[:255]])
    def paired(x):
        return np.stack((x, 1 - x), axis=1).reshape(-1, k)
    search, test = paired(search), paired(half[255:])
    if len(np.unique(np.vstack((search, test)), axis=0)) != 1024:
        return masks_for(k, seed + 100000)
    return search, test


def directions(k, seed, r):
    return np.random.default_rng(seed).choice([-1., 1.], (r, k)).astype('float32')


def fd(oracle, v, eta, g=None):
    plus = .5 + eta * v
    if g is not None:
        return (oracle.gradients(plus)[1] - g) / eta
    both = oracle.gradients(np.vstack((plus, .5 - eta * v)))[1]
    return (both[:len(v)] - both[len(v):]) / (2 * eta)


def chunks(oracle, x, batch):
    return np.concatenate([oracle.exact(x[i:i + batch]) for i in range(0, len(x), batch)])


def profile_cell(model, tok, task, k):
    search, _ = masks_for(k, 19011)
    mid = np.full((1, k), .5, dtype=np.float32)
    v = directions(k, 91, 8)
    data = []
    for row in engine.rows(task, 'development')[:3]:
        o = engine.prepare(model, tok, row, k)
        _, g = o.gradients(mid)
        item = dict(id=row['id'], common=o.common_ms, exact={}, hvp={}, sensitivity={})
        item['gradient'] = float(np.median([measured(lambda: o.gradients(mid))[1] for _ in range(3)]))
        for b in (1, 4, 8, 16, 32, 64, 128, 256, 512):
            # Exercise the shape once before measuring. Search deadlines use resident kernels.
            try:
                o.exact(search[:b])
                item['exact'][b] = float(np.median([measured(lambda: o.exact(search[:b]))[1] for _ in range(3)]))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        for r in (1, 2, 4, 8):
            for side, base in [('central', None), ('one', g)]:
                vals = [measured(lambda: fd(o, v[:r], .125, base)) for _ in range(4)]
                item['hvp'][f'{side}{r}'] = float(np.median([t for _, t in vals[1:]]))
        ref = fd(o, v, .0625)
        for eta in (.03125, .0625, .125, .25):
            for side, base in [('central', None), ('one', g)]:
                h = fd(o, v, eta, base)
                item['sensitivity'][f'{side}:{eta}'] = float(np.linalg.norm(h-ref)/max(np.linalg.norm(ref), 1e-12))
        item['native_audit'] = o.audit(search[:8])
        data.append(item)
    exact = {n: float(np.median([d['exact'][n] for d in data])) for n in data[0]['exact'] if all(n in d['exact'] for d in data)}
    best = min(exact, key=lambda n: exact[n] / n)
    rate = best / exact[best]
    hv = {s:float(np.median([d['hvp'][s] for d in data])) for s in data[0]['hvp']}
    # Freeze two operating points from development timings only. Evaluate the entire live stream.
    setup = float(np.median([d['common'] + d['gradient'] for d in data])) + exact[1]
    window = max(1.6 * exact[8], 8.)
    budget = round(setup + max(5 * (exact[32] + window), 2 * hv['central8']) + 2 * window)
    return dict(rows=data, sites=o.sites, kernels=o.kernel_profile(), exact_batch=best,
                masks_per_ms=rate, beq={s:rate*t/int(s.removeprefix('central').removeprefix('one')) for s,t in hv.items()},
                hvp_ms=hv, batch_ms=exact, setup_ms=setup, reserve_ms=window, budget_ms=budget)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.tmp').write_text(json.dumps(obj, allow_nan=False) + '\n')
    path.with_suffix('.tmp').replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['profile','search'])
    p.add_argument('checkpoint', type=Path)
    p.add_argument('task', choices=['gift','route'])
    p.add_argument('out', type=Path)
    p.add_argument('--limit', type=int, default=24)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--k', type=int, nargs='+', default=[32,64])
    args = p.parse_args()
    model, tok = engine.load(args.checkpoint)
    for k in args.k:
        dest = args.out / str(k)
        if args.stage == 'profile':
            results = profile_cell(model, tok, args.task, k)
            results.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(), checkpoint=model.config._name_or_path.split('/')[-1], task=args.task)
            write(dest / 'profile.json', results)
            print(args.task, k, 'B_eq', results['beq'], 'budget', results['budget_ms'], flush=True)
        else:
            study(model, tok, args.task, k, dest, args.start, args.limit)


def search_stream(model, tok, row, pool, arm, seed, prof, start):
    """Yield the incumbent to the caller before starting any further blocking operation."""
    from .estimators import fit, hvp_prior
    k = pool.shape[1]
    oracle = engine.prepare(model, tok, row, k)
    elapsed = lambda: clock() - start
    deadline = prof['budget_ms']
    reserve = prof['reserve_ms']
    predictions = None
    fitted = None
    seen = {0: 0.}
    incumbent = 0

    def acquire(ids):
        nonlocal incumbent
        new = [int(j) for j in ids if j not in seen]
        if new:
            values = oracle.exact(pool[new])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite native intervention')
            seen.update(zip(new, map(float, values)))
            incumbent = max([incumbent] + new, key=seen.__getitem__)

    def event(stage):
        return (dict(stage=stage, i=incumbent, paid_effect=seen[incumbent],
                     exact=len(seen)-1), fitted)

    if arm == 'cached_exact':
        batch = prof['exact_batch']
        # Prefer the highest-throughput batch that fits the remaining time.
        while len(seen) < len(pool):
            choices = [int(n) for n, t in prof['batch_ms'].items() if int(n) <= batch and t*1.15 < deadline-elapsed()]
            if not choices:
                break
            ids = [j for j in range(1, len(pool)) if j not in seen][:max(choices)]
            acquire(ids)
            yield event('exact')
        return

    empty = pool[:0]
    if arm == 'proxyspex_fourier':
        # The tree baseline is purely zeroth order and can spend the gradient time on labels.
        mid, gradient = 0., np.zeros(k)
        predictions = np.zeros(len(pool))
    else:
        mid, gradient = oracle.gradients(np.full((1,k), .5, np.float32))
        mid, gradient = float(mid[0]), gradient[0]
        fitted = fit('midpoint', empty, [], mid, gradient)
        predictions = fitted(pool)
        acquire([np.argmax(predictions)])
        yield event('midpoint')
    prior = None
    if arm.startswith('hvp'):
        r = int(arm[3:])
        if elapsed() + prof['hvp_ms'][f'central{r}'] + reserve >= deadline:
            return
        v = directions(k, seed, r)
        prior = hvp_prior(v, fd(oracle, v, .125))
        fitted = fit('complement', empty, [], mid, gradient, prior=prior)
        predictions = fitted(pool)
        acquire([np.argmax(predictions)])
        yield event('prior')
    method = 'complement' if prior is not None else arm
    selected = []
    stages = [4, 8, 16, 32, 48, 64, 80]  # new pairs at each stage
    fit_ms = 0.
    rng = np.random.default_rng(seed + 14)
    pair_order = rng.permutation(np.arange(1,len(pool)//2))
    for n_pairs in stages:
        if method == 'midpoint':
            break
        estimate = prof['batch_ms'][str(min(128, 2**int(np.ceil(np.log2(2*n_pairs)))))]
        if elapsed() + estimate * 1.25 + fit_ms + reserve >= deadline:
            break
        available = [int(j) for j in pair_order if j not in selected]
        if not available:
            break
        n_pairs = min(n_pairs, len(available))
        take = available[:n_pairs]
        if method == 'complement' and selected:
            # Half uniform, half most promising complement pairs, identical rule in both arms.
            uniform = take[:n_pairs//2]
            remaining = [j for j in available if j not in uniform]
            exploit = sorted(remaining, key=lambda j:max(predictions[2*j:2*j+2]), reverse=True)
            take = uniform + exploit[:n_pairs-len(uniform)]
        selected.extend(take)
        pair_ids = np.array(selected)[:, None]*2 + [0,1]
        acquire(pair_ids[-len(take):].ravel())
        yield event('acquire')
        # Preserve the latest delivered predictor and incumbent when an operation overruns.
        if elapsed() + fit_ms + reserve >= deadline:
            break
        fit_start = clock()
        idx = pair_ids.ravel()
        fitted = fit(method, pool[idx], [seen[int(j)] for j in idx], mid, gradient, prior=prior)
        predictions = fitted(pool)
        fit_ms = clock() - fit_start
        yield event('fit')
        if elapsed() + prof['batch_ms']['1'] >= deadline:
            break
        ranked = [int(j) for j in np.argsort(-predictions) if j not in seen]
        acquire(ranked[:8])
        yield event('verify')

    # Spend remaining time verifying the strongest unseen proposals, including midpoint's.
    while len(seen) < len(pool) and elapsed() + prof['batch_ms']['8'] * 1.15 < deadline:
        ranked = [int(j) for j in np.argsort(-predictions) if j not in seen]
        acquire(ranked[:8])
        yield event('final_verify')


def signal(o, pool, test, y, g, mid, seed):
    from .estimators import hvp_prior
    k = pool.shape[1]
    v = directions(k, seed, 8)
    h = hvp_prior(v, fd(o, v, .125))
    # Conditional mixed finite differences averaged over independent uniform Boolean contexts
    # estimate 4 times the degree-two Walsh coefficient, in gate-Hessian units.
    rng = np.random.default_rng(seed + 113)
    edges = list(combinations(range(k), 2))
    indices = rng.choice(len(edges), 24, replace=False)
    batches, native = [], []
    for ix in indices:
        i, j = edges[ix]
        b = np.repeat(rng.integers(2, size=(8,k),dtype=np.int8), 4, axis=0)
        b[:,i] = np.tile([0,0,1,1], 8)
        b[:,j] = np.tile([0,1,0,1], 8)
        batches.append(b)
        native.append((i,j))
    diffs = chunks(o, np.vstack(batches), 64).reshape(-1,8,4) @ np.array([1.,-1.,-1.,1.])
    ii,jj = np.array(native).T
    # Reference columnwise finite differences separate finite-step mismatch from sketch error.
    unit = np.eye(k, dtype=np.float32)[np.unique(jj)]
    deriv = np.concatenate([fd(o, unit[i:i+8], .125) for i in range(0,len(unit),8)])
    local = deriv[np.searchsorted(np.unique(jj),jj), ii]
    first = o.exact(pool[2:10])
    pilot = float(np.mean(np.abs(first - (mid + (pool[2:10]-.5) @ g))))
    miderr = float(np.mean(np.abs(y - (mid + (test-.5) @ g))))
    return dict(sampled_pairs=native, boolean_halves=np.stack((diffs[:,:4].mean(1),diffs[:,4:].mean(1))).tolist(),
                local=local.tolist(), sketch=h[ii,jj].tolist(), midpoint_mae=miderr, first8_mae=pilot,
                sketch_energy=float(np.linalg.norm(h)/np.sqrt(32)))


ARMS = ['midpoint','hvp2','hvp4','hvp8','complement','dense','proxyspex_fourier','cached_exact']


def study(model, tok, task, k, dest, start_index, limit):
    from .estimators import prepare_tree_backend
    prepare_tree_backend()
    # Shared resident-process startup only, on unrelated synthetic masks.
    from .estimators import fit
    warm, _ = masks_for(k, 12001)
    fit('proxyspex_fourier', warm[:32], warm[:32, :4].sum(1), 0., np.zeros(k))
    prof = json.loads((dest/'profile.json').read_text())
    for ix, row in enumerate(engine.rows(task)[start_index:start_index+limit], start_index):
        seed = 38756 + ix
        out = dest / (row['id']+'.json')
        if out.exists():
            continue
        pool, test = masks_for(k, seed)
        oracle = engine.prepare(model, tok, row, k)
        # Evaluation-only labels, never passed to the search stream.
        gold = chunks(oracle, pool, prof['exact_batch'])
        gold[0] = 0.
        test_y = chunks(oracle, test, prof['exact_batch'])
        mid, g = oracle.gradients(np.full((1,k),.5,np.float32))
        sig = signal(oracle, pool, test, test_y, g[0], mid[0], seed)
        del oracle
        traces = {}
        for arm in np.random.default_rng(seed+205).permutation(ARMS).tolist():
            start = clock()
            events, predictors = [], []
            for event, fitted in search_stream(model,tok,row,pool,arm,seed,prof,start):
                event['ms'] = clock() - start
                events.append(event)
                predictors.append(fitted)
            return_ms = clock() - start
            # Retrospective evaluation after all live deliveries. Each new predictor is scored once.
            last, mae, beta = None, None, None
            for event, predictor in zip(events, predictors):
                if predictor is not last:
                    mae = float(np.mean(np.abs(predictor(test) - test_y)))
                    beta = float(predictor.beta)
                    last = predictor
                event.update(mae=mae, beta=beta, oracle_effect=float(gold[event['i']]))
            traces[arm] = dict(events=events, return_ms=return_ms)
        write(out,dict(id=row['id'],template=row['template'],seed=seed, k=k, task=task,
                       pool_best=float(max(gold)), gold=gold.tolist(), test_y=test_y.tolist(),
                       signal=sig, traces=traces))
        print(row['id'],k,'complete',flush=True)


if __name__ == '__main__':
    main()
