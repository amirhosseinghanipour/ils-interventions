"""Check the archived-data bound against explicit per-token response vectors."""
import numpy as np
from lag_dispersion import pooled_energy_bound


def verify():
    rng = np.random.default_rng(112315)
    checks = 0
    for positions in range(1, 47):
        k = np.array([g ** np.arange(positions) for g in [.5,.9,1.]])
        k /= k.sum(1, keepdims=True)
        for full in (np.ones((positions,6)), rng.normal(size=(positions,6)),
                     ((-1.)**np.arange(positions))[:,None] * np.arange(1,7)[None,:]):
            means = k @ full
            lower = pooled_energy_bound(means,k)
            exact = np.sum(k[1,:,None] * full**2)
            signed_mean = np.sum(means[1]**2)
            assert signed_mean <= lower + 1e-8 and lower <= exact + 1e-8
            if positions <= 3 or np.all(full == 1):
                np.testing.assert_allclose(lower,exact,rtol=1e-9,atol=1e-9)
            checks += 1
    print(f"{checks} explicit per-position cases verified")


if __name__ == "__main__":
    verify()
