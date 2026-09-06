"""Cost functions, mirroring `ruptures.costs`.

Each class is a thin handle onto a Rust `CostEngine`. The engine builds prefix
sums once in ``fit()``; after that every ``error(start, end)`` is O(1) in the
segment length for every model except ``l1``, whose per-segment median is not a
prefix-summable statistic.

The Python-visible attributes (``signal``, ``covar``, ``min_size``, ``model``,
``metric`` and ``gamma``) are kept because `ruptures`' own estimators read them,
and so does user code.

Validation lives here rather than in Rust wherever `ruptures` raises a specific
Python exception. ``sum_of_costs`` in particular loops over ``self.error`` for
exactly that reason: it has to raise `NotEnoughPoints` on a short segment the
way the reference does, instead of quietly computing a number for it.
"""

import operator
import warnings

import numpy as np

from . import _ruptures_rs
from .base import BaseCost
from .exceptions import NotEnoughPoints
from .utils import pairwise


class _RustCost(BaseCost):
    """Shared plumbing for the compiled costs."""

    model = None
    _default_min_size = 1
    #: Which of `ruptures`' two implementations of this cost to reproduce.
    #: `None` is the Python class; ``"extension"`` is the C code behind
    #: `KernelCPD`, which evaluates its kernels differently.
    _variant = None

    def __init__(self, **params):
        self.signal = None
        self.min_size = self._default_min_size
        self._params = {k: v for k, v in params.items() if v is not None}
        self._engine = None

    # -- subclasses may override to reshape / split the input ---------------
    def _prepare(self, signal):
        return signal.reshape(-1, 1) if signal.ndim == 1 else signal

    def _engine_params(self):
        return self._params or None

    def _engine_input(self, signal):
        return signal

    def fit(self, signal):
        signal = np.ascontiguousarray(np.asarray(signal, dtype=float))
        self.signal = self._prepare(signal)
        self._engine = _ruptures_rs.CostEngine(
            np.ascontiguousarray(self._engine_input(signal), dtype=float),
            self.model,
            self._engine_params(),
            self._variant,
        )
        self.min_size = max(self._default_min_size, self._engine.min_size)
        return self

    def error(self, start, end):
        if self._engine is None:
            raise RuntimeError("call fit() before error()")
        start, end = int(start), int(end)
        if end - start < self.min_size:
            raise NotEnoughPoints
        return self._engine.error(start, end)

    def sum_of_costs(self, bkps):
        if self._engine is None:
            raise RuntimeError("call fit() before sum_of_costs()")
        return sum(self.error(start, end) for start, end in pairwise([0] + list(bkps)))


class CostL2(_RustCost):
    """Least-squared-deviation cost."""

    model = "l2"
    _default_min_size = 1


class CostL1(_RustCost):
    """Least-absolute-deviation cost."""

    model = "l1"
    _default_min_size = 2


class CostNormal(_RustCost):
    """Gaussian maximum-likelihood cost."""

    model = "normal"
    _default_min_size = 2

    def __init__(self, add_small_diag=True):
        super().__init__()
        self.add_small_diag = add_small_diag
        # `ruptures` writes `if self.add_small_diag:`, so the flag is whatever
        # Python considers truthy — `0` and `None` both mean "no bias". The
        # engine is handed a real bool so that stays true across the boundary
        # instead of an unparseable value falling back to the default.
        self._params["add_small_diag"] = bool(add_small_diag)


class CostRbf(_RustCost):
    """Kernelised mean-shift cost with a Gaussian (RBF) kernel."""

    model = "rbf"
    _default_min_size = 1

    def __init__(self, gamma=None):
        super().__init__(gamma=gamma)
        self.gamma = gamma
        self._gamma_arg = gamma

    def fit(self, signal):
        # Reset before refitting: `gamma` is derived from the signal when the
        # caller did not supply one, so carrying the previous signal's value
        # over would silently score the new signal with the old bandwidth.
        self.gamma = self._gamma_arg
        self._params = {} if self._gamma_arg is None else {"gamma": self._gamma_arg}
        super().fit(signal)
        # `gamma` defaults to the reciprocal median pairwise distance, computed
        # inside the engine; surface it so `KernelCPD` can read it back.
        if self.gamma is None:
            self.gamma = self._engine.gamma
        return self


class CostCosine(_RustCost):
    """Kernelised mean-shift cost with a cosine kernel."""

    model = "cosine"
    _default_min_size = 1


class CostRank(_RustCost):
    """Rank-based cost, robust to marginal distributions."""

    model = "rank"
    _default_min_size = 2


class CostMl(_RustCost):
    """Mahalanobis-metric kernel cost."""

    model = "mahalanobis"
    _default_min_size = 2

    def __init__(self, metric=None):
        super().__init__()
        self.metric = metric
        self.has_custom_metric = metric is not None

    def fit(self, signal):
        signal = np.ascontiguousarray(np.asarray(signal, dtype=float))
        s_ = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        # Checked before `np.cov`, not after: the covariance of a wide signal is
        # itself a d x d matrix, so leaving this to the engine would mean the
        # refusal arrived after NumPy had already tried the allocation.
        _ruptures_rs.check_outer_memory(s_.shape[0], s_.shape[1], "mahalanobis")
        if not self.has_custom_metric:
            # Computed here, with NumPy, rather than in Rust. A Cholesky-based
            # inverse disagrees with `numpy.linalg.inv` about which covariance
            # matrices are singular, so it refused signals the reference
            # happily segments — and raised a different exception when it did
            # refuse. `inv` raises `LinAlgError`, exactly as `ruptures` does.
            covar = np.cov(s_.T)
            mat = covar.reshape(1, 1) if covar.size == 1 else covar
            self.metric = np.linalg.inv(mat)
            # `inv` only raises when it detects *exact* singularity, so a
            # covariance one rounding step away from singular passes through
            # and produces a metric whose entries are pure cancellation. Every
            # cost computed from it is then meaningless — in `ruptures` too,
            # which simply says nothing. Say something.
            #
            # The threshold is where the answer stops being usable rather than
            # where it stops existing. A relative error of about `eps * cond`
            # survives the inversion, so 1e12 is roughly four correct digits;
            # `1 / eps` would be none at all, and every disagreement observed
            # between this package and the reference on this cost sat just
            # below that line. Ordinary correlated columns are nowhere near:
            # this fires when two columns agree to one part in a trillion.
            condition = float(np.linalg.cond(mat))
            if condition > 1e12:
                warnings.warn(
                    "the covariance of this signal is numerically singular "
                    "(condition number {:.2e}), so the default Mahalanobis "
                    "metric is dominated by rounding error and the costs "
                    "computed from it are unreliable. Pass an explicit "
                    "`metric`, or drop the redundant columns.".format(condition),
                    RuntimeWarning,
                    stacklevel=2,
                )
        # `ruptures` accepts a nested list or an integer array here, because it
        # only ever feeds the metric to `numpy.dot`.
        self._params = {
            "metric": np.ascontiguousarray(np.asarray(self.metric, dtype=float))
        }
        return super().fit(signal)


class CostCLinear(_RustCost):
    """Continuous piecewise-linear cost."""

    model = "clinear"
    _default_min_size = 3


class CostLinear(_RustCost):
    """Linear-regression residual cost.

    The first column of the signal is the response, the remainder are
    covariates — the same convention as `ruptures`.
    """

    model = "linear"
    _default_min_size = 2

    def _prepare(self, signal):
        assert signal.ndim > 1, "Not enough dimensions"
        self.covar = signal[:, 1:]
        return signal[:, 0].reshape(-1, 1)


class CostAR(_RustCost):
    """Autoregressive-model residual cost."""

    model = "ar"
    _default_min_size = 5

    def __init__(self, order=4):
        # `operator.index` rejects a float the way `numpy.pad` does in the
        # reference, instead of silently truncating it.
        order = operator.index(order)
        if order < 0:
            raise ValueError("`order` must not be negative, got {}".format(order))
        super().__init__(order=order)
        self.order = order
        self._params["order"] = order
        self._default_min_size = max(5, order + 1)
        self.min_size = self._default_min_size

    def _prepare(self, signal):
        sig = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        n_samples = sig.shape[0]
        if self.order >= n_samples:
            raise ValueError(
                "the `ar` model needs more samples than its order ({}); "
                "the signal has {}".format(self.order, n_samples)
            )
        # Reproduce the reference's lagged design and its edge padding, because
        # user code reads `.covar` and `.signal` off the fitted cost.
        flat = np.ascontiguousarray(sig, dtype=float).reshape(-1)
        lagged = np.empty((n_samples, self.order), dtype=float)
        for i in range(n_samples):
            base = max(i - self.order, 0)
            lagged[i] = flat[base : base + self.order]
        self.covar = np.c_[lagged, np.ones(n_samples)]
        out = sig.copy()
        out[: self.order] = out[self.order]
        return out


_REGISTRY = {
    cls.model: cls
    for cls in (
        CostL1,
        CostL2,
        CostNormal,
        CostRbf,
        CostCosine,
        CostRank,
        CostMl,
        CostCLinear,
        CostLinear,
        CostAR,
    )
}


def _iter_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def cost_factory(model, *args, **kwargs):
    """Instantiate a cost by model name, mirroring `ruptures.costs.cost_factory`.

    Like the reference, this also finds cost classes the *caller* defined, so
    ``Pelt(model="my_cost")`` works for any `BaseCost` subclass that declares
    that ``model``. Built-in names are resolved first, so a user class cannot
    shadow one by accident.
    """
    if model in _REGISTRY:
        return _REGISTRY[model](*args, **kwargs)
    for cls in _iter_subclasses(BaseCost):
        if getattr(cls, "model", None) == model:
            return cls(*args, **kwargs)
    raise ValueError("Not such model: {}".format(model))


__all__ = [
    "CostL1",
    "CostL2",
    "CostNormal",
    "CostRbf",
    "CostCosine",
    "CostRank",
    "CostMl",
    "CostCLinear",
    "CostLinear",
    "CostAR",
    "cost_factory",
    "NotEnoughPoints",
]
