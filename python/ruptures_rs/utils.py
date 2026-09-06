"""Small helpers, mirroring `ruptures.utils`."""

from itertools import tee
from math import ceil

from . import _ruptures_rs


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unzip(seq):
    return zip(*seq)


def sanity_check(n_samples, n_bkps, jump, min_size):
    """Whether any segmentation exists for these parameters."""
    n_adm_bkps = n_samples // jump
    if n_bkps > n_adm_bkps:
        return False
    if n_bkps * ceil(min_size / jump) * jump + min_size > n_samples:
        return False
    return True


def draw_bkps(n_samples=100, n_bkps=3, seed=None):
    import numpy as np

    rng = np.random.default_rng(seed=seed)
    alpha = np.ones(n_bkps + 1) / (n_bkps + 1) * 2000
    bkps = np.cumsum(rng.dirichlet(alpha) * n_samples).astype(int).tolist()
    bkps[-1] = n_samples
    return bkps


__all__ = ["pairwise", "unzip", "sanity_check", "draw_bkps"]
