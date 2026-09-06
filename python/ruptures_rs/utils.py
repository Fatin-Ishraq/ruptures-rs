"""Small helpers, mirroring `ruptures.utils`."""

import functools
import math
import operator
from itertools import tee
from math import ceil


def as_index(value, name):
    """Accept what `ruptures` accepts for an integer parameter.

    The reference stores `jump`, `min_size` and `width` unconverted and lets
    Python's arithmetic cope, so ``jump=5.0`` works there. Here they cross into
    Rust, which needs a real integer — so a whole-valued float is converted
    rather than rejected, and anything else raises the way it would there.
    """
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("{} must be a whole number, got {!r}".format(name, value))
        return int(value)
    return operator.index(value)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unzip(seq):
    return zip(*seq)


def sanity_check(n_samples, n_bkps, jump, min_size):
    """Whether any segmentation exists for these parameters."""
    if jump == 0:
        # `ruptures` divides by `jump` here and raises ZeroDivisionError. The
        # answer is still "no segmentation exists", and reporting that beats
        # propagating an arithmetic error from three frames down.
        return False
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


@functools.total_ordering
class Bnode:
    """A node of the `BottomUp` merge tree, mirroring `ruptures.utils.Bnode`.

    Present for compatibility: the accelerated `BottomUp` builds its tree in
    Rust and never instantiates this, but user code and subclasses reach for it.
    """

    def __init__(self, start, end, val, left=None, right=None, parent=None):
        self.start = start
        self.end = end
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

    @property
    def gain(self):
        if self.left is None or self.right is None:
            return 0
        if self.val == float("-inf"):
            return 0
        return self.val - (self.left.val + self.right.val)

    def __lt__(self, other):
        return self.start < other.start

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        return hash((self.__class__, self.start, self.end))

    def __repr__(self):
        return "Bnode({}, {}, {})".format(self.start, self.end, self.val)


def from_path_matrix_to_bkps_list(path_matrix_flat, n_bkps, n_samples, n_bkps_max, jump):
    """Walk a `KernelCPD` path matrix back into a breakpoint list.

    A port of `ruptures.utils.from_path_matrix_to_bkps_list`, which is a Cython
    wrapper around a few lines of C. Nothing in this package produces such a
    matrix — the dynamic program backtracks in Rust — but the function is part
    of the public surface a drop-in has to carry.
    """
    q = int(math.ceil(n_samples / jump))
    bkps = [0] * (n_bkps + 1)
    bkps[n_bkps] = q
    for k in range(1, n_bkps + 1):
        idx = bkps[n_bkps - k + 1] * (n_bkps_max + 1) + (n_bkps - k + 1)
        bkps[n_bkps - k] = int(path_matrix_flat[idx])
    bkps = [b * jump for b in bkps]
    bkps[n_bkps] = n_samples
    return bkps


__all__ = [
    "as_index",
    "pairwise",
    "unzip",
    "sanity_check",
    "draw_bkps",
    "Bnode",
    "from_path_matrix_to_bkps_list",
]
