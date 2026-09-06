"""Cost functions, mirroring `ruptures.costs`.

Each class is a thin handle onto a Rust `CostEngine`. The engine builds prefix
sums once in ``fit()``; after that every ``error(start, end)`` is O(1) in the
segment length for every model except ``l1``, whose per-segment median is not a
prefix-summable statistic.

The Python-visible attributes (``signal``, ``min_size``, ``model``, and
``gamma`` on ``CostRbf``) are kept because `ruptures`' own estimators read them,
and so does user code.
"""

import numpy as np

from . import _ruptures_rs
from .base import BaseCost
from .exceptions import NotEnoughPoints


class _RustCost(BaseCost):
    """Shared plumbing for the compiled costs."""

    model = None
    _default_min_size = 1

    def __init__(self, **params):
        self.signal = None
        self.min_size = self._default_min_size
        self._params = {k: v for k, v in params.items() if v is not None}
        self._engine = None

    # -- subclasses may override to reshape / split the input ---------------
    def _prepare(self, signal):
        return signal.reshape(-1, 1) if signal.ndim == 1 else signal

    def fit(self, signal):
        signal = np.ascontiguousarray(np.asarray(signal, dtype=float))
        self.signal = self._prepare(signal)
        self._engine = _ruptures_rs.CostEngine(
            np.ascontiguousarray(self._engine_input(signal), dtype=float),
            self.model,
            self._params or None,
        )
        self.min_size = max(self.min_size, self._engine.min_size)
        return self

    def _engine_input(self, signal):
        return signal

    def error(self, start, end):
        if self._engine is None:
            raise RuntimeError("call fit() before error()")
        if end - start < self.min_size:
            raise NotEnoughPoints
        return self._engine.error(int(start), int(end))

    def sum_of_costs(self, bkps):
        if self._engine is None:
            raise RuntimeError("call fit() before sum_of_costs()")
        return self._engine.sum_of_costs([int(b) for b in bkps])


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
        super().__init__(add_small_diag=add_small_diag)
        self.add_small_diag = add_small_diag


class CostRbf(_RustCost):
    """Kernelised mean-shift cost with a Gaussian (RBF) kernel."""

    model = "rbf"
    _default_min_size = 1

    def __init__(self, gamma=None):
        super().__init__(gamma=gamma)
        self.gamma = gamma

    def fit(self, signal):
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
        super().__init__(metric=metric)
        self.metric = metric


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
        super().__init__(order=order)
        self.order = order
        self._default_min_size = max(5, order + 1)
        self.min_size = self._default_min_size


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


def cost_factory(model, *args, **kwargs):
    """Instantiate a cost by model name, mirroring `ruptures.costs.cost_factory`."""
    try:
        return _REGISTRY[model](*args, **kwargs)
    except KeyError:
        raise ValueError("Not such model: {}".format(model)) from None


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
