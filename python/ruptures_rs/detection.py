"""Change point detectors, mirroring `ruptures.detection`.

Each estimator keeps the `ruptures` constructor signature and the
``fit`` / ``predict`` / ``fit_predict`` protocol. When the cost is one of the
built-in models the whole search runs in Rust; when the caller supplies their
own ``custom_cost`` the search falls back to the pure-Python implementation in
`_fallback`, because a user's Python callable inside the inner loop cannot be
accelerated by moving the loop across an FFI boundary.

``.accelerated`` reports which path an instance is on.
"""

import operator

import numpy as np

from . import _fallback, _ruptures_rs
from .base import BaseEstimator
from .costs import _RustCost, cost_factory
from .exceptions import BadSegmentationParameters, NotEnoughPoints
from .utils import as_index, sanity_check


def _is_custom_cost(obj):
    """Duck-typed: anything exposing `fit` and `error` acts as a cost.

    Deliberately looser than `ruptures`' `isinstance(custom_cost, BaseCost)`,
    which would silently ignore a cost subclassing *ruptures*' `BaseCost` when
    the estimator comes from this package.
    """
    return obj is not None and hasattr(obj, "error") and hasattr(obj, "fit")


class _Detector(BaseEstimator):
    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        if _is_custom_cost(custom_cost):
            self.cost = custom_cost
        else:
            self.model_name = model
            self.cost = (
                cost_factory(model=model)
                if params is None
                else cost_factory(model=model, **params)
            )
        # Decided by what the cost *is*, not by how it was named. `model=` can
        # resolve to a user-defined `BaseCost` subclass, which has no engine
        # behind it and has to take the pure-Python path like any other
        # `custom_cost`.
        self.accelerated = isinstance(self.cost, _RustCost)
        self.min_size = max(as_index(min_size, "min_size"), self.cost.min_size)
        self.jump = as_index(jump, "jump")
        if self.jump < 1:
            raise ValueError("jump must be at least 1, got {}".format(self.jump))
        self.n_samples = None

    def _fit(self, signal):
        signal = np.asarray(signal, dtype=float)
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def _engine(self):
        return self.cost._engine

    def _check(self, n_bkps):
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters


class Dynp(_Detector):
    """Exact segmentation by dynamic programming, for a known number of changes.

    This is the estimator with the largest speedup: `ruptures` evaluates its
    cost function once per dynamic-programming cell through NumPy, which is
    millions of O(len) calls; here each cell is a handful of scalar operations
    on prefix sums, and the levels are evaluated in parallel.
    """

    def fit(self, signal) -> "Dynp":
        return self._fit(signal)

    def predict(self, n_bkps):
        n_bkps = operator.index(n_bkps)
        if n_bkps < 0:
            raise ValueError("n_bkps must not be negative, got {}".format(n_bkps))
        self._check(n_bkps)
        if self.accelerated:
            return _ruptures_rs.dynp(self._engine(), n_bkps, self.jump, self.min_size)
        return _fallback.dynp(self.cost, self.n_samples, n_bkps, self.jump, self.min_size)

    def fit_predict(self, signal, n_bkps):
        self.fit(signal)
        return self.predict(n_bkps)


class Pelt(_Detector):
    """Penalised segmentation with pruning, for an unknown number of changes."""

    def fit(self, signal) -> "Pelt":
        return self._fit(signal)

    def predict(self, pen):
        self._check(0)
        if self.accelerated:
            return _ruptures_rs.pelt(self._engine(), float(pen), self.jump, self.min_size)
        return _fallback.pelt(
            self.cost, self.n_samples, float(pen), self.jump, self.min_size
        )

    def fit_predict(self, signal, pen):
        self.fit(signal)
        return self.predict(pen)


class Binseg(_Detector):
    """Binary segmentation: greedy recursive splitting."""

    def fit(self, signal) -> "Binseg":
        signal = np.asarray(signal, dtype=float)
        self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        return self._fit(signal)

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        assert any(p is not None for p in (n_bkps, pen, epsilon)), "Give a parameter."
        self._check(0 if n_bkps is None else n_bkps)
        if self.accelerated:
            return _ruptures_rs.binseg(
                self._engine(),
                self.jump,
                self.min_size,
                None if n_bkps is None else int(n_bkps),
                None if pen is None else float(pen),
                None if epsilon is None else float(epsilon),
            )
        return _fallback.binseg(
            self.cost, self.n_samples, self.jump, self.min_size, n_bkps, pen, epsilon
        )

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)


class BottomUp(_Detector):
    """Bottom-up segmentation: start over-segmented, merge cheapest first."""

    def fit(self, signal) -> "BottomUp":
        return self._fit(signal)

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        assert any(p is not None for p in (n_bkps, pen, epsilon)), "Give a parameter."
        self._check(0 if n_bkps is None else n_bkps)
        if self.accelerated:
            return _ruptures_rs.bottomup(
                self._engine(),
                self.jump,
                self.min_size,
                None if n_bkps is None else int(n_bkps),
                None if pen is None else float(pen),
                None if epsilon is None else float(epsilon),
            )
        return _fallback.bottomup(
            self.cost, self.n_samples, self.jump, self.min_size, n_bkps, pen, epsilon
        )

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)


class Window(_Detector):
    """Sliding-window segmentation: peak-picking on a discrepancy curve."""

    def __init__(
        self, width=100, model="l2", custom_cost=None, min_size=2, jump=5, params=None
    ):
        super().__init__(
            model=model,
            custom_cost=custom_cost,
            min_size=min_size,
            jump=jump,
            params=params,
        )
        # `Window` does not raise `min_size` to the cost's own minimum.
        self.min_size = as_index(min_size, "min_size")
        self.width = 2 * (as_index(width, "width") // 2)
        self.inds = None
        self.score = []

    def fit(self, signal) -> "Window":
        signal = np.asarray(signal, dtype=float)
        self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        self._fit(signal)
        if self.accelerated:
            w2 = self.width // 2
            n = self.n_samples
            has_windows = any(w2 <= k < n - w2 for k in range(0, n, self.jump))
            # Each half-window is scored on its own, so a window narrower than
            # twice the cost's minimum segment asks the cost for a segment it
            # refuses to score. Every built-in `ruptures` cost raises
            # `NotEnoughPoints` there; the compiled path would happily have
            # computed a median of one point.
            #
            # Only on this path. A `custom_cost` scores through `_fallback`,
            # where the user's own `error` decides whether a short segment is
            # a problem — which is exactly what happens in `ruptures`, and it
            # is not this layer's place to overrule it.
            if has_windows and w2 < self.cost.min_size:
                raise NotEnoughPoints
            inds, score = _ruptures_rs.window_fit(self._engine(), self.width, self.jump)
        else:
            inds, score = _fallback.window_score(
                self.cost, self.n_samples, self.width, self.jump
            )
        self.inds = np.asarray(inds, dtype=int)
        self.score = np.asarray(score, dtype=float)
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        self._check(0 if n_bkps is None else n_bkps)
        assert any(p is not None for p in (n_bkps, pen, epsilon)), "Give a parameter."
        if not self.accelerated:
            return _fallback.window_seg(
                self.cost,
                self.n_samples,
                [int(i) for i in self.inds],
                [float(s) for s in self.score],
                self.width,
                self.jump,
                self.min_size,
                n_bkps,
                pen,
                epsilon,
            )
        return _ruptures_rs.window_predict(
            self._engine(),
            [int(i) for i in self.inds],
            [float(s) for s in self.score],
            self.width,
            self.jump,
            self.min_size,
            None if n_bkps is None else int(n_bkps),
            None if pen is None else float(pen),
            None if epsilon is None else float(epsilon),
        )

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)


class KernelCPD(BaseEstimator):
    """Exact kernel change point detection (`jump` is fixed at 1).

    In `ruptures` this is the one detector already backed by a C extension —
    and it is 1,277x faster than the pure-Python `Dynp` on the same input,
    which is the clearest possible evidence for compiling the rest.

    That C extension is *not* the same code as `ruptures`' own `CostRbf` and
    `CostCosine`, and the two disagree. It evaluates the kernel on the diagonal
    instead of taking the zero diagonal `scipy`'s `squareform` leaves behind, and
    it clips the Gaussian exponent in single precision. A unit cosine diagonal
    is worth exactly one unit of penalty per segment, so routing this detector
    through the Python-flavoured cost changed which segmentation won. Here the
    engine is built with the extension's kernel, and the penalised search is a
    transcription of the extension's own PELT rather than of `ruptures`' Python
    one, which associates its penalty differently and prunes differently.
    """

    _KERNEL_TO_MODEL = {"linear": "l2", "rbf": "rbf", "cosine": "cosine"}

    def __init__(self, kernel="linear", min_size=2, jump=1, params=None):
        assert kernel in self._KERNEL_TO_MODEL, "Kernel not found: {}.".format(kernel)
        self.kernel_name = kernel
        self.model_name = self._KERNEL_TO_MODEL[kernel]
        self.params = params
        self.cost = (
            cost_factory(model=self.model_name)
            if params is None
            else cost_factory(model=self.model_name, **params)
        )
        self.cost._variant = "extension"
        self.min_size = max(as_index(min_size, "min_size"), self.cost.min_size)
        self.jump = 1
        self.n_samples = None
        self.segmentations_dict = {}

    def fit(self, signal) -> "KernelCPD":
        self.segmentations_dict = {}
        signal = np.asarray(signal, dtype=float)
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def predict(self, n_bkps=None, pen=None):
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=1 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        if n_bkps is not None:
            n_bkps = int(n_bkps)
            assert n_bkps > 0, "The number of changes must be positive: {}".format(n_bkps)
            if n_bkps in self.segmentations_dict:
                return self.segmentations_dict[n_bkps]
            # One dynamic program answers every k up to `n_bkps`, which is what
            # the reference's path matrix gives it too — so the cache it
            # advertises is actually populated.
            out = _ruptures_rs.dynp_all(self.cost._engine, n_bkps, 1, self.min_size)
            for k, bkps in enumerate(out, start=1):
                if bkps is not None:
                    self.segmentations_dict[k] = bkps
            if n_bkps not in self.segmentations_dict:
                raise BadSegmentationParameters
            return self.segmentations_dict[n_bkps]
        if pen is not None:
            assert pen > 0, "The penalty must be positive: {}".format(pen)
            return _ruptures_rs.kernel_pelt(self.cost._engine, float(pen), self.min_size)
        raise AssertionError("Give a parameter.")

    def fit_predict(self, signal, n_bkps=None, pen=None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen)


__all__ = ["Dynp", "Pelt", "Binseg", "BottomUp", "Window", "KernelCPD"]
