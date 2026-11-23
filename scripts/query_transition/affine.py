"""Separate reversible and shared responses to opposite edits in two different questions."""
import argparse
from pathlib import Path

import numpy as np
from transport import paths
from run import save_json


def predict(x, y, xt, scalar_prior):
    # Each paired training set, including every CV fold, has zero input mean.
    losses = np.zeros(5)
    for test in np.array_split(np.arange(len(x)), 4):
        train = np.setdiff1d(np.arange(len(x)), test)
        mu = y[train].mean((0,1))
        for i, pred in enumerate(paths(x[train], y[train]-mu, x[test], scalar_prior)):
            losses[i] += np.square(y[test]-mu-pred).sum()
    mu = y.mean((0,1))
    return mu + list(paths(x, y-mu, xt, scalar_prior))[int(np.argmin(losses))]


def analyze(raw):
    x, y, xt, yt = (raw[k].astype(float) for k in ("x","y","xt","yt"))
    for a in (x,xt):
        np.testing.assert_array_equal(a[:,1], -a[:,0])
    average = (yt[:,0]+yt[:,1])/2
    odd = (yt[:,0]-yt[:,1])/2
    mu = y.mean((0,1))
    ss = np.square(yt-yt.mean((0,1))).sum()
    even_residual = 2*np.square(average-average.mean(0)).sum()
    odd_energy = 2*np.square(odd).sum()
    np.testing.assert_allclose(ss, even_residual+odd_energy)
    alpha = (x*y).sum()/np.square(x).sum()
    def r2(pred):
        return float(1 - np.square(yt-pred).sum()/ss)
    return dict(train_mean_r2=r2(mu), affine_scalar_r2=r2(mu+alpha*xt),
                affine_ridge_r2=r2(predict(x,y,xt,False)),
                affine_corrected_r2=r2(predict(x,y,xt,True)),
                oracle_odd_r2_max=float(1-2*np.square(average).sum()/ss),
                oracle_odd_with_optimal_constant_r2_max=float(odd_energy/ss),
                heldout_odd_fraction_explained_by_scalar=float(1-np.square(odd-alpha*xt[:,0]).sum()/np.square(odd).sum()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("raw",type=Path)
    p.add_argument("result",type=Path)
    a=p.parse_args()
    with np.load(a.raw) as f:
        save_json(a.result, analyze(f))
