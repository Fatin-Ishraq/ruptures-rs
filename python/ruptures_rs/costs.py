"""Cost functions, mirroring `ruptures.costs`.

Each class is a thin handle onto a Rust `CostEngine`. The engine builds prefix
sums once in ``fit()``; after that every ``error(start, end)`` is O(1) in the
segment length for every model except ``l1``, whose per-segment median is not a
prefix-summable statistic.

The Python-visible attributes (``signal``, ``covar``, ``min_size``, ``model``,
``metric``, ``gamma``, ``gram``, ``ranks``, ``inv_cov``) are kept because
`ruptures`' own estimators read them, and so does user code. The ones that cost
O(n^2) to materialise are computed on first access rather than during ``fit``,
so that merely fitting a long signal does not pay for a matrix nobody asked for.

Validation lives here rather than in Rust wherever `ruptures` raises a specific
Python exception. ``sum_of_costs`` in particular loops over ``self.error`` for
exactly that reason: it has to raise `NotEnoughPoints` on a short segment the
way the reference does, instead of quietly computing a number for it.

Fitting is atomic. ``fit()`` builds every piece of new state in locals and
assigns none of it until the last one has succeeded, so a rejected signal leaves
the instance exactly as it was rather than half replaced — with a new
``.signal`` and an engine still answering for the old one.
"""

from __future__ import annotations

import operator
import warnings
from typing import Any, Iterable, Sequence

import numpy as np

from . import _ruptures_rs
from .base import BaseCost
from .exceptions import NotEnoughPoints
from .utils import as_index, pairwise


def _as_bound(value: Any, name: str) -> int:
    """A segment bound, accepted the way `ruptures` accepts one.

    The reference slices with whatever it is given, so a NumPy integer works and
    a float does not. ``int()`` accepted both by silently truncating, which
    turned ``error(0, 19.9)`` into a different question than the one asked.
    """
    return as_index(value, name)


class _RustCost(BaseCost):
    """Shared plumbing for the compiled costs."""

    model: str | None = None
    _default_min_size = 1
    #: Which of `ruptures`' two implementations of this cost to reproduce.
    #: `None` is the Python class; ``"extension"`` is the C code behind
    #: `KernelCPD`, which evaluates its kernels differently.
    _variant: str | None = None

    def __init__(self, **params: Any) -> None:
        self.signal: np.ndarray | None = None
        self.min_size = self._default_min_size
        self._params = {k: v for k, v in params.items() if v is not None}
        self._engine = None
        self._gram_cache: np.ndarray | None = None

    # -- subclasses may override to reshape / split the input ---------------
    def _prepare(self, signal: np.ndarray) -> dict[str, Any]:
        """Attributes to publish once the fit has succeeded.

        Returning them rather than assigning them is what makes ``fit`` atomic:
        nothing here is visible until the engine has been built.
        """
        return {"signal": signal.reshape(-1, 1) if signal.ndim == 1 else signal}

    def _engine_params(self, signal: np.ndarray) -> dict[str, Any]:
        return dict(self._params)

    def _engine_input(self, signal: np.ndarray) -> np.ndarray:
        return signal

    def _post_fit(self, engine: Any) -> None:
        """Read back anything the engine derived from the signal."""

    def fit(self, signal: Any) -> "_RustCost":
        raw = np.ascontiguousarray(np.asarray(signal, dtype=float))
        attrs = self._prepare(raw)
        params = self._engine_params(raw)
        engine = _ruptures_rs.CostEngine(
            np.ascontiguousarray(self._engine_input(raw), dtype=float),
            self.model,
            params or None,
            self._variant,
        )
        # ---- everything above can raise; nothing below can --------------
        self._params = params
        for name, value in attrs.items():
            setattr(self, name, value)
        self._engine = engine
        self.min_size = max(self._default_min_size, engine.min_size)
        self._gram_cache = None
        self._post_fit(engine)
        return self

    def error(self, start: Any, end: Any) -> float:
        if self._engine is None:
            raise RuntimeError("call fit() before error()")
        start, end = _as_bound(start, "start"), _as_bound(end, "end")
        if end - start < self.min_size:
            raise NotEnoughPoints
        return self._engine.error(start, end)

    def sum_of_costs(self, bkps: Iterable[Any]) -> float:
        if self._engine is None:
            raise RuntimeError("call fit() before sum_of_costs()")
        return sum(self.error(start, end) for start, end in pairwise([0] + list(bkps)))

    # -- lazily materialised compatibility attributes ----------------------
    def _require_fitted(self) -> np.ndarray:
        if self.signal is None:
            raise RuntimeError("call fit() first")
        return self.signal

    def _check_square_memory(self, what: str) -> None:
        n = self._require_fitted().shape[0]
        _ruptures_rs.check_square_memory(n, what)


def _pairwise_sqeuclidean(signal: np.ndarray) -> np.ndarray:
    """`squareform(pdist(signal, "sqeuclidean"))`, without needing SciPy.

    Written as a difference of norms rather than by subtracting every pair, so
    it is one matrix multiply; the diagonal is forced to exactly zero, which the
    algebraic form does not guarantee and `pdist` does.
    """
    sq = np.einsum("ij,ij->i", signal, signal)
    out = sq[:, None] + sq[None, :] - 2.0 * (signal @ signal.T)
    np.maximum(out, 0.0, out=out)
    np.fill_diagonal(out, 0.0)
    return out


class CostL2(_RustCost):
    """Least-squared-deviation cost."""

    model = "l2"
    _default_min_size = 1

    # Declared explicitly, rather than inheriting `**params`, so that the
    # signature matches the reference's. Without it `CostL2(gamma=3)` was
    # accepted here and rejected there — and the argument silently did nothing.
    def __init__(self) -> None:
        super().__init__()


class CostL1(_RustCost):
    """Least-absolute-deviation cost."""

    model = "l1"
    _default_min_size = 2

    # Declared explicitly, rather than inheriting `**params`, so that the
    # signature matches the reference's. Without it `CostL2(gamma=3)` was
    # accepted here and rejected there — and the argument silently did nothing.
    def __init__(self) -> None:
        super().__init__()


class CostNormal(_RustCost):
    """Gaussian maximum-likelihood cost."""

    model = "normal"
    _default_min_size = 2

    def _prepare(self, signal: np.ndarray) -> dict[str, Any]:
        sig = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        # `ruptures`' `CostNormal` publishes these; nothing here needs them.
        return {
            "signal": sig,
            "n_samples": sig.shape[0],
            "n_dims": sig.shape[1],
        }

    def __init__(self, add_small_diag: bool = True) -> None:
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

    def __init__(self, gamma: float | None = None) -> None:
        super().__init__(gamma=gamma)
        self.gamma = gamma
        self._gamma_arg = gamma

    def _engine_params(self, signal: np.ndarray) -> dict[str, Any]:
        # Rebuilt from the constructor argument, not from the previous fit:
        # `gamma` is derived from the signal when the caller did not supply one,
        # so carrying the last signal's value over would score this one with the
        # wrong bandwidth.
        return {} if self._gamma_arg is None else {"gamma": self._gamma_arg}

    def _post_fit(self, engine: Any) -> None:
        # `gamma` defaults to the reciprocal median pairwise distance, computed
        # inside the engine; surface it so `KernelCPD` can read it back.
        self.gamma = self._gamma_arg if self._gamma_arg is not None else engine.gamma

    @property
    def gram(self) -> np.ndarray:
        """The dense kernel matrix, as `ruptures` exposes it.

        Materialised on first access and then cached. The engine does not need
        it — it keeps a cumulative sum instead, which answers a block query in
        O(1) — so building it during ``fit`` would be n^2 doubles spent on
        nobody's behalf.
        """
        if self._gram_cache is None:
            self._check_square_memory("the `rbf` model's gram matrix")
            k = _pairwise_sqeuclidean(self._require_fitted()) * self.gamma
            np.clip(k, 1e-2, 1e2, out=k)
            out = np.exp(-k)
            # `squareform` reassembles a *condensed* distance vector, so the
            # diagonal it writes is a structural zero that never went through
            # the clip — leaving `exp(0) = 1` rather than `exp(-0.01)`. The C
            # extension behind `KernelCPD` does clip it, which is one of the two
            # reasons these kernels disagree with each other. This attribute is
            # the Python one.
            np.fill_diagonal(out, 1.0)
            self._gram_cache = out
        return self._gram_cache


class CostCosine(_RustCost):
    """Kernelised mean-shift cost with a cosine kernel."""

    model = "cosine"
    _default_min_size = 1

    # Declared explicitly, rather than inheriting `**params`, so that the
    # signature matches the reference's. Without it `CostL2(gamma=3)` was
    # accepted here and rejected there — and the argument silently did nothing.
    def __init__(self) -> None:
        super().__init__()

    @property
    def gram(self) -> np.ndarray:
        """`squareform(1 - pdist(signal, "cosine"))`, diagonal included.

        `squareform` leaves a zero diagonal, which is not what the cosine
        similarity of a row with itself is — and that difference is worth
        exactly one unit of penalty per segment. It is reproduced here because
        this attribute exists to be the reference's, not to be right.
        """
        if self._gram_cache is None:
            self._check_square_memory("the `cosine` model's gram matrix")
            signal = self._require_fitted()
            norms = np.linalg.norm(signal, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (signal @ signal.T) / (norms[:, None] * norms[None, :])
            np.fill_diagonal(out, 0.0)
            self._gram_cache = out
        return self._gram_cache


class CostRank(_RustCost):
    """Rank-based cost, robust to marginal distributions."""

    model = "rank"
    _default_min_size = 2

    # Declared explicitly, rather than inheriting `**params`, so that the
    # signature matches the reference's. Without it `CostL2(gamma=3)` was
    # accepted here and rejected there — and the argument silently did nothing.
    def __init__(self) -> None:
        super().__init__()

    def _prepare(self, signal: np.ndarray) -> dict[str, Any]:
        sig = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        n_samples, n_features = sig.shape
        # Average ranks, matching `scipy.stats.rankdata` on ties, then centred
        # into [-(n+1)/2, (n+1)/2] as the reference centres them. Computed with
        # NumPy rather than read back out of the engine deliberately: these are
        # compatibility attributes, and `ruptures` computes them with NumPy and
        # SciPy, so this is the value user code comparing the two will expect.
        ranks = np.empty((n_samples, n_features), dtype=float)
        for j in range(n_features):
            _, inverse, counts = np.unique(
                sig[:, j], return_inverse=True, return_counts=True
            )
            starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
            ranks[:, j] = (starts + (counts + 1) / 2.0)[inverse.ravel()]
        centered = ranks - ((n_samples + 1) / 2)
        cov = np.cov(centered, rowvar=False, bias=True).reshape(
            n_features, n_features
        )
        return {
            "signal": sig,
            "ranks": centered,
            "inv_cov": np.linalg.pinv(cov),
        }


class CostMl(_RustCost):
    """Mahalanobis-metric kernel cost."""

    model = "mahalanobis"
    _default_min_size = 2

    def __init__(self, metric: Any = None) -> None:
        super().__init__()
        self.metric = metric
        self.has_custom_metric = metric is not None

    def _engine_params(self, signal: np.ndarray) -> dict[str, Any]:
        s_ = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        # Checked before `np.cov`, not after: the covariance of a wide signal is
        # itself a d x d matrix, so leaving this to the engine would mean the
        # refusal arrived after NumPy had already tried the allocation.
        _ruptures_rs.check_outer_memory(s_.shape[0], s_.shape[1], "mahalanobis")
        metric = self.metric
        if not self.has_custom_metric:
            # Computed here, with NumPy, rather than in Rust. A Cholesky-based
            # inverse disagrees with `numpy.linalg.inv` about which covariance
            # matrices are singular, so it refused signals the reference
            # happily segments — and raised a different exception when it did
            # refuse. `inv` raises `LinAlgError`, exactly as `ruptures` does.
            covar = np.cov(s_.T)
            mat = covar.reshape(1, 1) if covar.size == 1 else covar
            metric = np.linalg.inv(mat)
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
        return {"metric": np.ascontiguousarray(np.asarray(metric, dtype=float))}

    def _post_fit(self, engine: Any) -> None:
        self.metric = self._params["metric"]

    @property
    def gram(self) -> np.ndarray:
        """`signal @ metric @ signal.T`, as `ruptures` builds it eagerly.

        Lazy here: the engine reaches the same numbers from prefix sums of the
        outer products, in O(d^2) per query and no quadratic memory at all, so
        this exists only for callers that read the attribute.
        """
        if self._gram_cache is None:
            self._check_square_memory("the `mahalanobis` model's gram matrix")
            signal = self._require_fitted()
            self._gram_cache = signal.dot(self.metric).dot(signal.T)
        return self._gram_cache


class CostCLinear(_RustCost):
    """Continuous piecewise-linear cost."""

    model = "clinear"
    _default_min_size = 3

    # Declared explicitly, rather than inheriting `**params`, so that the
    # signature matches the reference's. Without it `CostL2(gamma=3)` was
    # accepted here and rejected there — and the argument silently did nothing.
    def __init__(self) -> None:
        super().__init__()


class CostLinear(_RustCost):
    """Linear-regression residual cost.

    The first column of the signal is the response, the remainder are
    covariates — the same convention as `ruptures`.
    """

    model = "linear"
    _default_min_size = 2

    # Declared explicitly, rather than inheriting `**params`, so that the
    # signature matches the reference's. Without it `CostL2(gamma=3)` was
    # accepted here and rejected there — and the argument silently did nothing.
    def __init__(self) -> None:
        super().__init__()

    def _prepare(self, signal: np.ndarray) -> dict[str, Any]:
        if signal.ndim < 2:
            # `ruptures` writes `assert signal.ndim > 1, "Not enough dimensions"`.
            # Raised rather than asserted, because `python -O` strips an assert
            # and the next line then indexes a 1-D array by two axes — but with
            # the reference's exception type, so a caller's `except` still works.
            raise AssertionError("Not enough dimensions")
        return {"covar": signal[:, 1:], "signal": signal[:, 0].reshape(-1, 1)}


class CostAR(_RustCost):
    """Autoregressive-model residual cost."""

    model = "ar"
    _default_min_size = 5

    def __init__(self, order: int = 4) -> None:
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

    def _prepare(self, signal: np.ndarray) -> dict[str, Any]:
        sig = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        n_samples = sig.shape[0]
        if self.order >= n_samples:
            raise ValueError(
                "the `ar` model needs more samples than its order ({}); "
                "the signal has {}".format(self.order, n_samples)
            )
        # Budgeted before the arrays exist, not after: the lagged design is
        # `n x order` and the engine's covariance prefix is `n x (order+1)^2`,
        # and leaving the check to the engine meant NumPy had already tried the
        # first allocation by the time the second was refused.
        _ruptures_rs.check_outer_memory(n_samples, self.order + 1, "ar")
        # Reproduce the reference's lagged design and its edge padding, because
        # user code reads `.covar` and `.signal` off the fitted cost.
        flat = np.ascontiguousarray(sig, dtype=float).reshape(-1)
        lagged = np.empty((n_samples, self.order), dtype=float)
        for i in range(n_samples):
            base = max(i - self.order, 0)
            lagged[i] = flat[base : base + self.order]
        out = sig.copy()
        out[: self.order] = out[self.order]
        return {"covar": np.c_[lagged, np.ones(n_samples)], "signal": out}


#: The cost classes that have a compiled engine behind them, by model name.
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

#: Methods whose meaning the compiled path assumes. A subclass that redefines
#: any of them is asking for its own arithmetic, and gets it.
_SEMANTIC_METHODS = (
    "fit",
    "error",
    "sum_of_costs",
    "_prepare",
    "_engine_input",
    "_engine_params",
)


def is_accelerable(cost: Any) -> bool:
    """Whether the compiled search may be used for this cost object.

    Being *an instance of* a built-in cost is not enough, and treating it as
    though it were is a way to compute the wrong answer in silence. A subclass
    of `CostL2` that overrides ``error`` still passes ``isinstance``, still has
    an engine attached — and the engine is a plain L2 that knows nothing about
    the override. The estimator would then optimise one function and report the
    cost of another, with ``sum_of_costs`` disagreeing with the segmentation it
    was handed.

    So the test is on the methods rather than on the type: an exact built-in
    always qualifies, and a subclass qualifies only while it has left the
    semantics alone.
    """
    if not isinstance(cost, _RustCost):
        return False
    kind = type(cost)
    if kind in _REGISTRY.values():
        return True
    # The nearest built-in ancestor is what the engine actually implements.
    base = next((c for c in kind.__mro__ if c in _REGISTRY.values()), None)
    if base is None:
        return False
    return all(
        getattr(kind, name, None) is getattr(base, name, None)
        for name in _SEMANTIC_METHODS
    )


def _iter_subclasses(cls: type) -> Iterable[type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def cost_factory(model: str, *args: Any, **kwargs: Any) -> BaseCost:
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


__all__: Sequence[str] = [
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
    "is_accelerable",
    "NotEnoughPoints",
]
