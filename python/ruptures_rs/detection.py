"""Change point detectors, mirroring `ruptures.detection`.

Each estimator keeps the `ruptures` constructor signature and the
``fit`` / ``predict`` / ``fit_predict`` protocol. When the cost is one of the
built-in models *and* has not been subclassed into something else, the whole
search runs in Rust; otherwise it falls back to the pure-Python implementation
in `_fallback`, because a user's Python callable inside the inner loop cannot be
accelerated by moving the loop across an FFI boundary.

``.accelerated`` reports which path an instance is on.

Argument checking is centralised in the helpers at the top rather than repeated
per detector, and it raises rather than asserting. `ruptures` uses bare
``assert`` for its required-argument checks, which `python -O` removes — turning
"you must pass one of these" into an unhandled ``TypeError`` three frames down.
Where the reference raises `AssertionError` this raises it too, explicitly, so
that a caller's ``except`` still catches what it always caught.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from . import _fallback, _ruptures_rs
from .base import BaseEstimator
from .costs import cost_factory, is_accelerable
from .exceptions import BadSegmentationParameters, NotEnoughPoints
from .utils import Bnode, as_index, sanity_check


def _is_custom_cost(obj: Any) -> bool:
    """Duck-typed: anything exposing `fit` and `error` acts as a cost.

    Deliberately looser than `ruptures`' `isinstance(custom_cost, BaseCost)`,
    which would silently ignore a cost subclassing *ruptures*' `BaseCost` when
    the estimator comes from this package.
    """
    return obj is not None and hasattr(obj, "error") and hasattr(obj, "fit")


def _require_one(n_bkps: Any, pen: Any, epsilon: Any) -> None:
    if all(p is None for p in (n_bkps, pen, epsilon)):
        raise AssertionError("Give a parameter.")


def _as_n_bkps(value: Any) -> int:
    n_bkps = as_index(value, "n_bkps")
    if n_bkps < 0:
        raise ValueError("n_bkps must not be negative, got {}".format(n_bkps))
    return n_bkps


def _as_threshold(value: Any, name: str) -> float:
    """A penalty or an epsilon, as a float that a comparison can order.

    `inf` is allowed — it is a meaningful, if extreme, request. NaN is not:
    every comparison against it is false, so the search would neither stop nor
    prune and the result would depend on which candidate happened to be first.
    """
    out = float(value)
    if math.isnan(out):
        raise ValueError("{} must not be NaN".format(name))
    return out


class _Detector(BaseEstimator):
    def __init__(
        self,
        model: str = "l2",
        custom_cost: Any = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict[str, Any] | None = None,
    ) -> None:
        if _is_custom_cost(custom_cost):
            self.cost = custom_cost
        else:
            self.model_name = model
            self.cost = (
                cost_factory(model=model)
                if params is None
                else cost_factory(model=model, **params)
            )
        # Decided by what the cost *does*, not by what it is an instance of.
        # `model=` can resolve to a user-defined `BaseCost` subclass with no
        # engine behind it, and a subclass of a built-in that overrides `error`
        # has an engine that does not implement the override — accelerating
        # that would optimise one function while reporting another.
        self.accelerated = is_accelerable(self.cost)
        self.min_size = max(as_index(min_size, "min_size"), self.cost.min_size)
        if self.min_size < 1:
            raise ValueError(
                "min_size must be at least 1, got {}".format(self.min_size)
            )
        self.jump = as_index(jump, "jump")
        if self.jump < 1:
            raise ValueError("jump must be at least 1, got {}".format(self.jump))
        self.n_samples: int | None = None

    def _fit(self, signal: Any) -> "_Detector":
        signal = np.asarray(signal, dtype=float)
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def _engine(self) -> Any:
        return self.cost._engine

    def _fitted_samples(self) -> int:
        if self.n_samples is None or getattr(self.cost, "signal", None) is None:
            raise RuntimeError("call fit() before predict()")
        return self.cost.signal.shape[0]

    def _check(self, n_bkps: int) -> None:
        if not sanity_check(
            n_samples=self._fitted_samples(),
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

    def fit(self, signal: Any) -> "Dynp":
        self._seg_cache: dict[tuple[int, int, int], dict] = {}
        return self._fit(signal)

    def predict(self, n_bkps: Any) -> list[int]:
        n_bkps = _as_n_bkps(n_bkps)
        self._check(n_bkps)
        if self.accelerated:
            return _ruptures_rs.dynp(self._engine(), n_bkps, self.jump, self.min_size)
        return _fallback.dynp(self.cost, self.n_samples, n_bkps, self.jump, self.min_size)

    def fit_predict(self, signal: Any, n_bkps: Any) -> list[int]:
        self.fit(signal)
        return self.predict(n_bkps)

    def seg(self, start: Any, end: Any, n_bkps: Any) -> dict[tuple[int, int], float]:
        """The optimal partition of ``signal[start:end]``, as `ruptures` returns it.

        A ``{(start, end): cost}`` dict, memoised per fitted instance. The
        accelerated `predict` does not go through this — it fills a table in
        Rust — but the method is part of the surface a drop-in has to carry, and
        subclasses in the wild override it.
        """
        key = (
            as_index(start, "start"),
            as_index(end, "end"),
            _as_n_bkps(n_bkps),
        )
        cache = getattr(self, "_seg_cache", None)
        if cache is None:
            cache = self._seg_cache = {}
        if key not in cache:
            cache[key] = _fallback.dynp_seg(
                self.cost, self.jump, self.min_size, *key
            )
        return cache[key]


class Pelt(_Detector):
    """Penalised segmentation with pruning, for an unknown number of changes.

    Exact: unlike `ruptures`, this does not discard a candidate before the point
    at which discarding it is justified, so it cannot return a segmentation that
    a different one beats. See `docs/differences.md`.
    """

    def fit(self, signal: Any) -> "Pelt":
        return self._fit(signal)

    def predict(self, pen: Any) -> list[int]:
        self._check(0)
        pen = _as_threshold(pen, "pen")
        if self.accelerated:
            return _ruptures_rs.pelt(self._engine(), pen, self.jump, self.min_size)
        return _fallback.pelt(self.cost, self.n_samples, pen, self.jump, self.min_size)

    def fit_predict(self, signal: Any, pen: Any) -> list[int]:
        self.fit(signal)
        return self.predict(pen)


class Binseg(_Detector):
    """Binary segmentation: greedy recursive splitting."""

    def fit(self, signal: Any) -> "Binseg":
        signal = np.asarray(signal, dtype=float)
        self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        self._single_bkp_cache: dict[tuple[int, int], tuple[int | None, float]] = {}
        return self._fit(signal)

    def predict(
        self, n_bkps: Any = None, pen: Any = None, epsilon: Any = None
    ) -> list[int]:
        _require_one(n_bkps, pen, epsilon)
        n_bkps = None if n_bkps is None else _as_n_bkps(n_bkps)
        pen = None if pen is None else _as_threshold(pen, "pen")
        epsilon = None if epsilon is None else _as_threshold(epsilon, "epsilon")
        self._check(0 if n_bkps is None else n_bkps)
        if self.accelerated:
            return _ruptures_rs.binseg(
                self._engine(), self.jump, self.min_size, n_bkps, pen, epsilon
            )
        return _fallback.binseg(
            self.cost, self.n_samples, self.jump, self.min_size, n_bkps, pen, epsilon
        )

    def fit_predict(
        self, signal: Any, n_bkps: Any = None, pen: Any = None, epsilon: Any = None
    ) -> list[int]:
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)

    def single_bkp(self, start: Any, end: Any) -> tuple[int | None, float]:
        """Best split of ``[start:end]`` and its gain, as `ruptures` returns it."""
        key = (as_index(start, "start"), as_index(end, "end"))
        cache = getattr(self, "_single_bkp_cache", None)
        if cache is None:
            cache = self._single_bkp_cache = {}
        if key not in cache:
            cache[key] = _fallback.binseg_single_bkp(
                self.cost, self.jump, self.min_size, *key
            )
        return cache[key]


class BottomUp(_Detector):
    """Bottom-up segmentation: start over-segmented, merge cheapest first."""

    def fit(self, signal: Any) -> "BottomUp":
        signal = np.asarray(signal, dtype=float)
        self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        self._merge_cache: dict[tuple[int, int, int, int], float] = {}
        self._leaves: list[Bnode] | None = None
        return self._fit(signal)

    @property
    def leaves(self) -> list[Bnode]:
        """The over-segmented starting partition, as a list of `Bnode`.

        Grown on first access. The accelerated search builds its own tree in
        Rust and never touches this, so growing it during ``fit`` would be
        O(n / min_size) cost evaluations spent on an attribute most callers
        never read.
        """
        if getattr(self, "_leaves", None) is None:
            self._leaves = _fallback.grow_tree(
                self.cost, self._fitted_samples(), self.jump, self.min_size
            )
        return self._leaves

    def merge(self, left: Bnode, right: Bnode) -> Bnode:
        """Merge two adjacent nodes, memoised on their bounds."""
        key = (left.start, left.end, right.start, right.end)
        cache = getattr(self, "_merge_cache", None)
        if cache is None:
            cache = self._merge_cache = {}
        if key not in cache:
            cache[key] = self.cost.error(left.start, right.end)
        return Bnode(left.start, right.end, cache[key], left=left, right=right)

    def predict(
        self, n_bkps: Any = None, pen: Any = None, epsilon: Any = None
    ) -> list[int]:
        _require_one(n_bkps, pen, epsilon)
        n_bkps = None if n_bkps is None else _as_n_bkps(n_bkps)
        pen = None if pen is None else _as_threshold(pen, "pen")
        epsilon = None if epsilon is None else _as_threshold(epsilon, "epsilon")
        self._check(0 if n_bkps is None else n_bkps)
        if self.accelerated:
            return _ruptures_rs.bottomup(
                self._engine(), self.jump, self.min_size, n_bkps, pen, epsilon
            )
        return _fallback.bottomup(
            self.cost, self.n_samples, self.jump, self.min_size, n_bkps, pen, epsilon
        )

    def fit_predict(
        self, signal: Any, n_bkps: Any = None, pen: Any = None, epsilon: Any = None
    ) -> list[int]:
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)


class Window(_Detector):
    """Sliding-window segmentation: peak-picking on a discrepancy curve."""

    def __init__(
        self,
        width: int = 100,
        model: str = "l2",
        custom_cost: Any = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            custom_cost=custom_cost,
            min_size=min_size,
            jump=jump,
            params=params,
        )
        # `Window` does not raise `min_size` to the cost's own minimum.
        self.min_size = as_index(min_size, "min_size")
        if self.min_size < 1:
            raise ValueError("min_size must be at least 1, got {}".format(self.min_size))
        # Rounded down to even, as `ruptures` rounds it, and *not* rejected when
        # that leaves zero: the reference lets a zero-width window through to the
        # cost, which then raises `NotEnoughPoints` on a segment of no samples.
        # Refusing earlier with a different exception would be less compatible,
        # not more careful. `fit` catches it below, before the engine sees it.
        self.width = 2 * (as_index(width, "width") // 2)
        self.inds: np.ndarray | None = None
        self.score: Any = []

    def fit(self, signal: Any) -> "Window":
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

    def predict(
        self, n_bkps: Any = None, pen: Any = None, epsilon: Any = None
    ) -> list[int]:
        _require_one(n_bkps, pen, epsilon)
        n_bkps = None if n_bkps is None else _as_n_bkps(n_bkps)
        pen = None if pen is None else _as_threshold(pen, "pen")
        epsilon = None if epsilon is None else _as_threshold(epsilon, "epsilon")
        self._check(0 if n_bkps is None else n_bkps)
        if self.inds is None:
            raise RuntimeError("call fit() before predict()")
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
            n_bkps,
            pen,
            epsilon,
        )

    def fit_predict(
        self, signal: Any, n_bkps: Any = None, pen: Any = None, epsilon: Any = None
    ) -> list[int]:
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

    That transcription is faithful including the extension's pruning, which — in
    common with `ruptures`' Python `Pelt` before it was repaired here — can
    discard a candidate from a point at which discarding it is not yet
    justified. `predict(n_bkps=...)` is exact; `predict(pen=...)` reproduces the
    extension. `docs/differences.md` has the detail.
    """

    _KERNEL_TO_MODEL = {"linear": "l2", "rbf": "rbf", "cosine": "cosine"}

    def __init__(
        self,
        kernel: str = "linear",
        min_size: int = 2,
        jump: int = 1,
        params: dict[str, Any] | None = None,
    ) -> None:
        if kernel not in self._KERNEL_TO_MODEL:
            raise AssertionError("Kernel not found: {}.".format(kernel))
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
        self.n_samples: int | None = None
        self.segmentations_dict: dict[int, list[int]] = {}

    def fit(self, signal: Any) -> "KernelCPD":
        self.segmentations_dict = {}
        signal = np.asarray(signal, dtype=float)
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def predict(self, n_bkps: Any = None, pen: Any = None) -> list[int]:
        if self.n_samples is None:
            raise RuntimeError("call fit() before predict()")
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=1 if n_bkps is None else _as_n_bkps(n_bkps),
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        if n_bkps is not None:
            n_bkps = _as_n_bkps(n_bkps)
            if n_bkps <= 0:
                raise AssertionError(
                    "The number of changes must be positive: {}".format(n_bkps)
                )
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
            pen = _as_threshold(pen, "pen")
            if not pen > 0:
                raise AssertionError("The penalty must be positive: {}".format(pen))
            return _ruptures_rs.kernel_pelt(self.cost._engine, pen, self.min_size)
        raise AssertionError("Give a parameter.")

    def fit_predict(self, signal: Any, n_bkps: Any = None, pen: Any = None) -> list[int]:
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen)


__all__: Sequence[str] = [
    "Dynp",
    "Pelt",
    "Binseg",
    "BottomUp",
    "Window",
    "KernelCPD",
]
