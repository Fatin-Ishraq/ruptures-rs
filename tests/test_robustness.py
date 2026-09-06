"""Hostile input, resource limits and the parts of the API surface that are not
about breakpoints.

Nothing here compares against `ruptures`. These are properties this package owes
its own users: that a malformed argument raises something catchable, that an
impossible allocation is refused rather than attempted, and that the pieces of
the `ruptures` surface a drop-in has to carry are actually present.

The recurring assertion is subtle enough to spell out. A Rust panic reaches
Python as `pyo3_runtime.PanicException`, which inherits from `BaseException`,
*not* from `Exception` — so a caller's `except Exception` does not catch it and
the traceback unwinds through their program. Every "does not panic" test below
therefore guards with `except Exception` and lets anything else fail the test.
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import ruptures_rs as rpt

warnings.filterwarnings("ignore", category=UserWarning)

MODELS = ["l1", "l2", "normal", "rbf", "cosine", "rank", "clinear"]
DETECTORS = ["Dynp", "Pelt", "Binseg", "BottomUp", "Window"]


def call_catching_normal_exceptions(fn):
    """Run `fn`, tolerating any ordinary exception but no `BaseException`.

    A `PanicException` is not an `Exception`, so it escapes this and fails the
    calling test — which is exactly the point.
    """
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - the type is the assertion
        return None, exc


def signal(n=120, d=1, seed=1):
    return rpt.pw_constant(n, d, 3, noise_std=2, seed=seed)[0]


# --------------------------------------------------------------- no panics


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_never_panics(model, bad):
    """A NaN in the signal used to abort through the FFI boundary.

    Three separate routes did it: an `unwrap()` on the median's comparator, a
    `sort_by` whose comparator stopped being a total order, and the covariance
    inverse. All of them are reachable from a single bad sample.
    """
    sig = signal()
    sig[50, 0] = bad
    _, exc = call_catching_normal_exceptions(
        lambda: rpt.cost_factory(model=model).fit(sig).error(0, 120)
    )
    assert exc is None or isinstance(exc, Exception)


@pytest.mark.parametrize("detector", DETECTORS)
def test_detectors_never_panic_on_nan(detector):
    sig = signal()
    sig[50, 0] = np.nan
    kwargs = {"pen": 50.0} if detector == "Pelt" else {"n_bkps": 3}
    build = dict(model="l2", min_size=5, jump=5)
    if detector == "Window":
        build["width"] = 20
    _, exc = call_catching_normal_exceptions(
        lambda: getattr(rpt, detector)(**build).fit(sig).predict(**kwargs)
    )
    assert exc is None or isinstance(exc, Exception)


@pytest.mark.parametrize("model", ["l1", "l2", "normal", "clinear", "rbf"])
def test_nan_never_reads_as_a_zero_cost(model):
    """A NaN cost must stay a NaN.

    Clamping a cost at zero — which the prefix-sum forms need, because they can
    land a few ulp below zero on a constant segment — silently turned NaN into
    `0.0` through `f64::max`. A zero cost is the *most* attractive segment
    there is, so corrupt data would have pulled breakpoints towards itself.
    """
    sig = signal()
    sig[50, 0] = np.nan
    value = rpt.cost_factory(model=model).fit(sig).error(40, 60)
    assert np.isnan(value), "NaN input produced {!r}".format(value)


def test_sum_of_costs_past_the_end_is_refused():
    cost = rpt.CostL2().fit(signal())
    with pytest.raises(ValueError):
        cost.sum_of_costs([60, 500])


def test_sum_of_costs_raises_not_enough_points_like_the_reference():
    cost = rpt.CostL1().fit(signal())
    with pytest.raises(rpt.NotEnoughPoints):
        cost.sum_of_costs([1, 120])


def test_negative_segment_bounds_are_refused():
    cost = rpt.CostL2().fit(signal())
    with pytest.raises(ValueError):
        cost.error(-5, 10)
    with pytest.raises(ValueError):
        cost.error(0, 500)


@pytest.mark.parametrize("cls,kwargs", [("Pelt", {}), ("Window", {"width": 20})])
def test_zero_jump_is_refused(cls, kwargs):
    with pytest.raises(ValueError):
        getattr(rpt, cls)(model="l2", jump=0, **kwargs)


def test_zero_jump_is_refused_by_crops():
    with pytest.raises(ValueError):
        rpt.Crops(model="l2", jump=0)


def test_negative_n_bkps_is_refused():
    with pytest.raises(ValueError):
        rpt.Dynp(model="l2").fit(signal()).predict(-1)


# ------------------------------------------------------------ memory guards


def test_kernel_memory_guard():
    """A clear refusal beats an OOM kill.

    Kernel costs need an (n+1)^2 table - the same quadratic memory `ruptures`
    spends on its dense Gram matrix, where the failure mode is the allocator
    killing the process.
    """
    huge = np.zeros((30_000, 1))
    with pytest.raises(ValueError, match="kernel table"):
        rpt.CostRbf().fit(huge)
    with pytest.raises(ValueError, match="kernel table"):
        rpt.CostCosine().fit(huge)
    assert rpt.CostL2().fit(huge).error(0, 30_000) == 0.0


@pytest.mark.parametrize("model", ["normal", "mahalanobis"])
def test_wide_signal_memory_guard(model):
    """The covariance costs keep `(n+1) * d^2` doubles.

    `ruptures` handles a wide signal in O(n * d^2) memory, so a shape it copes
    with fine used to take this process down: an allocation failure in Rust
    aborts rather than raising, and the user loses their interpreter.
    """
    wide = np.zeros((100, 20_000))
    with pytest.raises(ValueError, match="covariance blocks"):
        rpt.cost_factory(model=model).fit(wide)


# ------------------------------------------------------- parameter handling


def test_add_small_diag_respects_python_truthiness():
    """`ruptures` writes `if self.add_small_diag:`, so `0` means no bias.

    A strict bool extraction silently fell back to the default here, which
    turned the flag off-by-request into on-by-accident.
    """
    sig = signal()
    off = rpt.CostNormal(add_small_diag=False).fit(sig).error(0, 30)
    zero = rpt.CostNormal(add_small_diag=0).fit(sig).error(0, 30)
    on = rpt.CostNormal(add_small_diag=True).fit(sig).error(0, 30)
    assert zero == off
    assert on != off


@pytest.mark.parametrize("order", [-1, -5])
def test_negative_ar_order_is_refused(order):
    with pytest.raises(ValueError):
        rpt.CostAR(order=order)


def test_float_ar_order_is_refused_like_the_reference():
    with pytest.raises(TypeError):
        rpt.CostAR(order=2.0)


def test_ar_order_larger_than_the_signal_is_refused():
    with pytest.raises(ValueError):
        rpt.CostAR(order=200).fit(signal())


def test_whole_valued_float_jump_is_accepted():
    """`ruptures` never converts `jump`, so `jump=5.0` works there."""
    a = rpt.Dynp(model="l2", min_size=5, jump=5).fit(signal()).predict(3)
    b = rpt.Dynp(model="l2", min_size=5.0, jump=5.0).fit(signal()).predict(3)
    assert a == b


def test_mahalanobis_accepts_a_list_metric():
    sig = signal(d=2)
    from_list = rpt.CostMl(metric=[[1, 0], [0, 1]]).fit(sig).error(0, 50)
    from_array = rpt.CostMl(metric=np.eye(2)).fit(sig).error(0, 50)
    assert from_list == from_array


def test_singular_covariance_warns_instead_of_returning_silent_noise():
    """The default Mahalanobis metric is an inverse, and `numpy.linalg.inv`
    only raises on *exact* singularity.

    One rounding step short of that it returns a matrix of ~1e15 entries, and
    every cost computed from it is cancellation. `ruptures` says nothing; every
    disagreement between the two libraries on this cost was on a covariance in
    that state.
    """
    base = rpt.pw_constant(120, 1, 3, seed=5)[0]
    degenerate = np.c_[base, base * 0.5 + 1.0]
    with pytest.warns(RuntimeWarning, match="numerically singular"):
        rpt.CostMl().fit(degenerate)


def test_ordinary_correlated_columns_do_not_warn():
    """The other half of the claim: the warning has to be quiet on real data,
    or it is noise people learn to filter out.

    Correlated is not collinear — this fires only when two columns agree to
    about one part in a trillion.
    """
    rng = np.random.default_rng(4)
    for seed in range(12):
        sig = rpt.pw_normal(200, 3, seed=seed)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            rpt.CostMl().fit(sig)
    # Strongly but not degenerately correlated columns are fine too.
    base = rng.normal(size=(200, 1))
    correlated = np.c_[base, base * 2.0 + rng.normal(scale=1e-3, size=(200, 1))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rpt.CostMl().fit(correlated)


# ------------------------------------------------------------------ Window


@pytest.mark.parametrize(
    "model,width", [("l1", 2), ("clinear", 4), ("ar", 6), ("l2", 1)]
)
def test_window_narrower_than_the_cost_minimum_is_refused(model, width):
    """Each half-window is scored on its own.

    A window narrower than twice the cost's `min_size` asks for a segment the
    cost refuses to score, and `ruptures` raises. The compiled path went ahead
    and took a median of one point.
    """
    with pytest.raises(rpt.NotEnoughPoints):
        rpt.Window(width=width, model=model, min_size=2, jump=1).fit(signal())


class RangeCost(rpt.base.BaseCost):
    """A user cost the compiled path cannot know about."""

    model = "range_cost"
    min_size = 3

    def fit(self, sig):
        self.signal = sig.reshape(-1, 1) if sig.ndim == 1 else sig
        return self

    def error(self, start, end):
        return float(np.ptp(self.signal[start:end], axis=0).sum()) * (end - start)


def test_window_defers_the_width_check_to_a_custom_cost():
    """The narrow-window guard belongs to the compiled path only.

    `ruptures` has no such check: the cost's own `error` raises, or does not.
    A `custom_cost` that is happy to score two points should keep working, and
    one that refuses should still refuse.
    """

    class TolerantCost(RangeCost):
        model = "tolerant_range"
        min_size = 5  # advertised, but `error` does not enforce it

    est = rpt.Window(width=4, custom_cost=TolerantCost(), min_size=2, jump=2)
    bkps = est.fit(signal(n=120)).predict(n_bkps=2)
    assert bkps[-1] == 120


def test_window_supports_custom_cost():
    """This used to raise `NotImplementedError`, contradicting the promise that
    `custom_cost` keeps working on the pure-Python path."""
    est = rpt.Window(width=20, custom_cost=RangeCost(), min_size=5, jump=5)
    assert est.accelerated is False
    bkps = est.fit(signal(n=150)).predict(n_bkps=2)
    assert bkps[-1] == 150
    assert bkps == sorted(bkps)


def test_cost_factory_finds_user_defined_costs():
    """`ruptures` discovers `BaseCost` subclasses by their `model` name."""
    assert isinstance(rpt.cost_factory("range_cost"), RangeCost)
    est = rpt.Pelt(model="range_cost", min_size=5, jump=5)
    assert est.fit(signal(n=150)).predict(50)[-1] == 150


def test_cost_factory_still_rejects_unknown_models():
    with pytest.raises(ValueError):
        rpt.cost_factory(model="definitely_not_a_model")


# ----------------------------------------------------------- API surface


def test_metrics_additions():
    a, b = [50, 100, 150, 200], [48, 103, 152, 200]
    assert rpt.hamming(a, b) == pytest.approx(1 - rpt.randindex(a, b))
    assert rpt.metrics.hamming is rpt.hamming


def test_metrics_reject_incomparable_partitions_the_way_ruptures_does():
    with pytest.raises(rpt.metrics.BadPartitions):
        rpt.hausdorff([50, 100], [50, 120])
    with pytest.raises(rpt.metrics.BadPartitions):
        rpt.hausdorff([50, 50, 100], [50, 100])
    with pytest.raises(rpt.metrics.BadPartitions):
        rpt.hausdorff([], [])


def test_utils_additions():
    node = rpt.utils.Bnode(0, 10, 1.5)
    assert node.gain == 0
    assert rpt.utils.Bnode(0, 10, 9.0) == node
    # A three-breakpoint path matrix over 4 rows, stride (n_bkps_max + 1).
    matrix = [0] * 40
    matrix[3 * 4 + 1] = 1
    out = rpt.utils.from_path_matrix_to_bkps_list(matrix, 1, 3, 3, 1)
    assert out == [1, 3]


def test_public_surface_covers_the_reference():
    rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
    missing = {
        name
        for name in dir(rpt_py)
        if not name.startswith("_") and not hasattr(rpt, name)
    }
    assert not missing, "missing from ruptures_rs: {}".format(sorted(missing))


@pytest.mark.parametrize(
    "statement",
    [
        "from ruptures.detection.pelt import Pelt",
        "from ruptures.detection.kernelcpd import KernelCPD",
        "from ruptures.costs.factory import cost_factory",
        "from ruptures.costs.costl2 import CostL2",
        "from ruptures.metrics.hausdorff import hausdorff",
        "from ruptures.utils.bnode import Bnode",
        "from ruptures.datasets.pw_constant import pw_constant",
        "from ruptures.show.display import display",
    ],
)
def test_install_covers_leaf_modules(statement):
    """`ruptures` splits its API across one module per class.

    A dependency that writes `from ruptures.costs.costl2 import CostL2` used to
    get `ModuleNotFoundError` from the shim, because only the package level was
    aliased.
    """
    code = "import ruptures_rs\nruptures_rs.install()\n{}\nprint('ok')".format(statement)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_kernelcpd_fills_every_smaller_segmentation():
    """`ruptures` gets all of them from one path matrix, and caches them."""
    est = rpt.KernelCPD(kernel="linear", min_size=5).fit(signal(n=200, seed=4))
    top = est.predict(n_bkps=4)
    assert sorted(est.segmentations_dict) == [1, 2, 3, 4]
    assert est.segmentations_dict[4] == top
    for k, bkps in est.segmentations_dict.items():
        assert len(bkps) == k + 1
        assert bkps[-1] == 200
