"""Synthetic signal generators, mirroring `ruptures.datasets`.

Ported verbatim so that a given `seed` produces the same signal as `ruptures`
does — which is what makes the differential test suite meaningful.
"""

from itertools import cycle

import numpy as np

from .utils import draw_bkps


def pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=None, delta=(1, 10), seed=None):
    """Piecewise-constant signal."""
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    signal = np.empty((n_samples, n_features), dtype=float)
    tt_ = np.arange(n_samples)
    delta_min, delta_max = delta
    center = np.zeros(n_features)
    rng = np.random.default_rng(seed=seed)
    for ind in np.split(tt_, bkps):
        if ind.size > 0:
            jump = rng.uniform(delta_min, delta_max, size=n_features)
            spin = rng.choice([-1, 1], n_features)
            center += jump * spin
            signal[ind] = center
    if noise_std is not None:
        signal = signal + rng.normal(size=signal.shape) * noise_std
    return signal, bkps


def pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=None, seed=None):
    """Piecewise-linear regression signal (response in column 0)."""
    rng = np.random.default_rng(seed=seed)
    covar = rng.normal(size=(n_samples, n_features))
    linear_coeff, bkps = pw_constant(
        n_samples=n_samples, n_bkps=n_bkps, n_features=n_features, noise_std=None, seed=seed
    )
    var = np.sum(linear_coeff * covar, axis=1)
    if noise_std is not None:
        var += rng.normal(scale=noise_std, size=var.shape)
    return np.c_[var, covar], bkps


def pw_normal(n_samples=200, n_bkps=3, seed=None):
    """Bivariate Gaussian signal with alternating correlation sign."""
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    signal = np.zeros((n_samples, 2), dtype=float)
    cov1 = np.array([[1, 0.9], [0.9, 1]])
    cov2 = np.array([[1, -0.9], [-0.9, 1]])
    rng = np.random.default_rng(seed=seed)
    for sub, cov in zip(np.split(signal, bkps), cycle((cov1, cov2))):
        n_sub, _ = sub.shape
        sub += rng.multivariate_normal([0, 0], cov, size=n_sub)
    return signal, bkps


def pw_wavy(n_samples=200, n_bkps=3, noise_std=None, seed=None):
    """Piecewise sinusoidal signal with alternating frequency."""
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    f1 = np.array([0.075, 0.1])
    f2 = np.array([0.1, 0.125])
    freqs = np.zeros((n_samples, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n_samples)
    signal = np.sum([np.sin(2 * np.pi * tt * f) for f in freqs.T], axis=0)
    if noise_std is not None:
        rng = np.random.default_rng(seed=seed)
        signal += rng.normal(scale=noise_std, size=signal.shape)
    return signal, bkps


__all__ = ["pw_constant", "pw_linear", "pw_normal", "pw_wavy"]
