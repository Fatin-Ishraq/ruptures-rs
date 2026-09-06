"""Differential tests: `ruptures_rs` must agree with `ruptures`.

This is the correctness engine for the whole project. A drop-in replacement is
only worth anything if it returns the *same* answer, so almost every test here
runs both implementations on the same input and demands identical breakpoints —
not "close", identical.

Costs are compared with a tolerance, because the Rust path evaluates them by a
different (prefix-sum) identity than NumPy's two-pass reductions. Breakpoints
are compared exactly, because they are what users actually consume.
"""

import warnings

import numpy as np
import pytest

# Every test here compares against `ruptures`.
# On Python versions the reference itself does not support - it declares
# `requires_python = "<3.14"` - skip rather than fail. This package works
# on those versions even though the reference does not.
rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
import ruptures.metrics as rpt_py_metrics
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)

# (model, params, signal kind) triples covering every built-in cost.
MODELS = [
    ("l1", {}, "constant"),
    ("l2", {}, "constant"),
    ("normal", {}, "constant"),
    ("rbf", {}, "constant"),
    ("rbf", {"gamma": 0.5}, "constant"),
    ("cosine", {}, "wavy2d"),
    ("rank", {}, "constant"),
    ("mahalanobis", {}, "normal2d"),
    ("clinear", {}, "constant"),
    ("linear", {}, "linear"),
    ("ar", {"order": 3}, "wavy"),
]

DETECTORS = ["Dynp", "Binseg", "BottomUp", "Pelt", "Window"]


def make_signal(kind, n=200, seed=0):
    if kind == "constant":
        return rpt_rs.pw_constant(n, 1, 3, noise_std=2, seed=seed)[0]
    if kind == "normal2d":
        return rpt_rs.pw_normal(n, 3, seed=seed)[0]
    if kind == "wavy":
        return rpt_rs.pw_wavy(n, 3, noise_std=0.2, seed=seed)[0]
    if kind == "wavy2d":
        base = rpt_rs.pw_wavy(n, 3, noise_std=0.2, seed=seed)[0]
        return np.c_[base, np.roll(base, 5) + 3.0]
    if kind == "linear":
        return rpt_rs.pw_linear(n, 2, 3, noise_std=1, seed=seed)[0]
    raise ValueError(kind)


def cost_pair(model, params, signal):
    a = rpt_py.costs.cost_factory(model=model, **params).fit(signal)
    b = rpt_rs.cost_factory(model=model, **params).fit(signal)
    return a, b


# --------------------------------------------------------------- cost values


@pytest.mark.parametrize("model,params,kind", MODELS)
def test_cost_values_agree(model, params, kind):
    """Every segment cost must match NumPy's value to near machine precision."""
    signal = make_signal(kind, n=160, seed=7)
    ref, got = cost_pair(model, params, signal)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        start = int(rng.integers(0, 140))
        end = int(rng.integers(start + max(got.min_size, 5), 161))
        expected = float(np.asarray(ref.error(start, end)).sum())
        actual = got.error(start, end)
        assert np.isclose(actual, expected, rtol=1e-8, atol=1e-8), (
            f"{model} error({start},{end}): {actual!r} != {expected!r}"
        )
        checked += 1
    assert checked == 300


# ----------------------------------------------------------------- detectors


@pytest.mark.parametrize("model,params,kind", MODELS)
@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detectors_agree(detector, model, params, kind, seed):
    """Breakpoints must be identical, not merely similar."""
    signal = make_signal(kind, n=200, seed=seed)
    min_size = 5
    jump = 5

    def build(mod):
        if detector == "Window":
            return getattr(mod, detector)(
                width=40, model=model, min_size=min_size, jump=jump, params=params or None
            )
        return getattr(mod, detector)(
            model=model, min_size=min_size, jump=jump, params=params or None
        )

    kwargs = {"pen": 100.0} if detector == "Pelt" else {"n_bkps": 3}
    expected = [int(b) for b in build(rpt_py).fit(signal).predict(**kwargs)]
    actual = [int(b) for b in build(rpt_rs).fit(signal).predict(**kwargs)]
    assert actual == expected


@pytest.mark.parametrize("jump", [1, 2, 5, 10])
@pytest.mark.parametrize("min_size", [2, 5, 12])
def test_jump_and_min_size_grid(jump, min_size):
    """The admissibility arithmetic is fiddly; cover the parameter grid."""
    signal = make_signal("constant", n=180, seed=11)
    for detector, kwargs in (
        ("Dynp", {"n_bkps": 3}),
        ("Binseg", {"n_bkps": 3}),
        ("BottomUp", {"n_bkps": 3}),
        ("Pelt", {"pen": 150.0}),
    ):
        a = getattr(rpt_py, detector)(model="l2", min_size=min_size, jump=jump)
        b = getattr(rpt_rs, detector)(model="l2", min_size=min_size, jump=jump)
        expected = [int(x) for x in a.fit(signal).predict(**kwargs)]
        actual = [int(x) for x in b.fit(signal).predict(**kwargs)]
        assert actual == expected, f"{detector} jump={jump} min_size={min_size}"


@pytest.mark.parametrize("n_bkps", [1, 2, 3, 5, 8])
def test_dynp_breakpoint_counts(n_bkps):
    signal = make_signal("constant", n=240, seed=5)
    expected = rpt_py.Dynp(model="l2", min_size=4, jump=2).fit(signal).predict(n_bkps)
    actual = rpt_rs.Dynp(model="l2", min_size=4, jump=2).fit(signal).predict(n_bkps)
    assert [int(x) for x in actual] == [int(x) for x in expected]
    assert len(actual) == n_bkps + 1


@pytest.mark.parametrize("pen", [10.0, 50.0, 100.0, 500.0, 2000.0])
def test_pelt_penalties(pen):
    signal = make_signal("constant", n=240, seed=6)
    expected = rpt_py.Pelt(model="l2", min_size=4, jump=2).fit(signal).predict(pen)
    actual = rpt_rs.Pelt(model="l2", min_size=4, jump=2).fit(signal).predict(pen)
    assert [int(x) for x in actual] == [int(x) for x in expected]


@pytest.mark.parametrize("stop", ["pen", "epsilon"])
def test_binseg_bottomup_alternate_stopping(stop):
    signal = make_signal("constant", n=200, seed=9)
    kwargs = {"pen": 200.0} if stop == "pen" else {"epsilon": 5000.0}
    for detector in ("Binseg", "BottomUp"):
        expected = (
            getattr(rpt_py, detector)(model="l2", min_size=5, jump=5)
            .fit(signal)
            .predict(**kwargs)
        )
        actual = (
            getattr(rpt_rs, detector)(model="l2", min_size=5, jump=5)
            .fit(signal)
            .predict(**kwargs)
        )
        assert [int(x) for x in actual] == [int(x) for x in expected], detector


def test_multivariate():
    signal = rpt_rs.pw_constant(220, 4, 3, noise_std=1.5, seed=13)[0]
    for model in ("l1", "l2", "normal", "rank", "mahalanobis"):
        expected = rpt_py.Dynp(model=model, min_size=5, jump=5).fit(signal).predict(3)
        actual = rpt_rs.Dynp(model=model, min_size=5, jump=5).fit(signal).predict(3)
        assert [int(x) for x in actual] == [int(x) for x in expected], model


def test_kernelcpd():
    signal = make_signal("constant", n=200, seed=4)
    for kernel in ("linear", "rbf", "cosine"):
        expected = rpt_py.KernelCPD(kernel=kernel, min_size=5).fit(signal).predict(n_bkps=3)
        actual = rpt_rs.KernelCPD(kernel=kernel, min_size=5).fit(signal).predict(n_bkps=3)
        assert [int(x) for x in actual] == [int(x) for x in expected], kernel


# ------------------------------------------------------- numerical stability


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e9])
def test_large_offsets_do_not_cancel(offset):
    """The trap this project was written to avoid.

    `sum(x^2) - sum(x)^2 / n` is the textbook one-pass variance and loses
    catastrophic precision when the mean dwarfs the spread. Centring the signal
    before building prefix sums is what keeps these cases exact; without it the
    1e9 case disagrees with NumPy, or returns a negative variance.
    """
    signal = make_signal("constant", n=200, seed=3) + offset
    expected = rpt_py.Dynp(model="l2", min_size=5, jump=5).fit(signal).predict(3)
    actual = rpt_rs.Dynp(model="l2", min_size=5, jump=5).fit(signal).predict(3)
    assert [int(x) for x in actual] == [int(x) for x in expected]

    ref = rpt_py.costs.CostL2().fit(signal)
    got = rpt_rs.CostL2().fit(signal)
    for start, end in ((0, 200), (7, 143), (100, 105)):
        assert got.error(start, end) >= 0.0
        assert np.isclose(got.error(start, end), ref.error(start, end), rtol=1e-7)


def test_constant_signal():
    """Degenerate input: zero variance everywhere."""
    signal = np.ones((120, 1))
    for model in ("l1", "l2"):
        expected = rpt_py.Dynp(model=model, min_size=5, jump=5).fit(signal).predict(2)
        actual = rpt_rs.Dynp(model=model, min_size=5, jump=5).fit(signal).predict(2)
        assert [int(x) for x in actual] == [int(x) for x in expected], model


# ------------------------------------------------------------- error handling


def test_bad_segmentation_parameters():
    signal = make_signal("constant", n=50, seed=0)
    with pytest.raises(rpt_rs.BadSegmentationParameters):
        rpt_rs.Dynp(model="l2", min_size=10, jump=5).fit(signal).predict(20)


def test_not_enough_points():
    signal = make_signal("constant", n=50, seed=0)
    cost = rpt_rs.CostL1().fit(signal)
    with pytest.raises(rpt_rs.NotEnoughPoints):
        cost.error(3, 4)


def test_unknown_model():
    with pytest.raises(ValueError):
        rpt_rs.cost_factory(model="nope")


def test_kernel_memory_guard():
    """A clear refusal beats an OOM kill.

    Kernel costs need an (n+1)^2 table - the same quadratic memory `ruptures`
    spends on its dense Gram matrix, where the failure mode is the allocator
    killing the process.
    """
    huge = np.zeros((30_000, 1))
    with pytest.raises(ValueError, match="kernel table"):
        rpt_rs.CostRbf().fit(huge)
    with pytest.raises(ValueError, match="kernel table"):
        rpt_rs.CostCosine().fit(huge)
    # non-kernel models have no such limit
    assert rpt_rs.CostL2().fit(huge).error(0, 30_000) == 0.0


# ---------------------------------------------------------------- custom cost


class SignedRangeCost(rpt_py.base.BaseCost):
    """A user cost that the compiled path cannot know about."""

    model = "custom_range"
    min_size = 3
    jump = 1

    def fit(self, signal):
        self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        return self

    def error(self, start, end):
        sub = self.signal[start:end]
        return float(np.ptp(sub, axis=0).sum() * (end - start))


@pytest.mark.parametrize("detector", ["Dynp", "Binseg", "BottomUp", "Pelt"])
def test_custom_cost_falls_back_and_matches(detector):
    """`custom_cost` must still work, and still give ruptures' answer.

    The compiled path cannot own a loop whose inner call is user Python, so
    these route to the pure-Python implementation. Coverage stays complete;
    only the speedup is (honestly) absent.
    """
    signal = make_signal("constant", n=150, seed=2)
    kwargs = {"pen": 500.0} if detector == "Pelt" else {"n_bkps": 3}
    a = getattr(rpt_py, detector)(custom_cost=SignedRangeCost(), min_size=5, jump=5)
    b = getattr(rpt_rs, detector)(custom_cost=SignedRangeCost(), min_size=5, jump=5)
    assert b.accelerated is False
    expected = [int(x) for x in a.fit(signal).predict(**kwargs)]
    actual = [int(x) for x in b.fit(signal).predict(**kwargs)]
    assert actual == expected


# ----------------------------------------------------------------- utilities


def test_datasets_are_bit_identical():
    """Same seed, same signal — otherwise the differential tests prove nothing."""
    for fn, kwargs in (
        ("pw_constant", dict(n_samples=150, n_features=2, n_bkps=3, noise_std=1, seed=42)),
        ("pw_normal", dict(n_samples=150, n_bkps=3, seed=42)),
        ("pw_wavy", dict(n_samples=150, n_bkps=3, noise_std=1, seed=42)),
        ("pw_linear", dict(n_samples=150, n_features=2, n_bkps=3, noise_std=1, seed=42)),
    ):
        a, ab = getattr(rpt_py, fn)(**kwargs)
        b, bb = getattr(rpt_rs, fn)(**kwargs)
        assert np.array_equal(a, b), fn
        assert list(ab) == list(bb), fn


def test_metrics_agree():
    true_bkps = [50, 100, 150, 200]
    my_bkps = [48, 103, 152, 200]
    assert rpt_rs.hausdorff(true_bkps, my_bkps) == rpt_py_metrics.hausdorff(true_bkps, my_bkps)
    assert rpt_rs.precision_recall(true_bkps, my_bkps) == rpt_py_metrics.precision_recall(
        true_bkps, my_bkps
    )
    assert np.isclose(
        rpt_rs.randindex(true_bkps, my_bkps), rpt_py_metrics.randindex(true_bkps, my_bkps)
    )
    assert np.isclose(
        rpt_rs.meantime(true_bkps, my_bkps), rpt_py_metrics.meantime(true_bkps, my_bkps)
    )


def test_sum_of_costs_agrees():
    signal = make_signal("constant", n=180, seed=8)
    bkps = [40, 90, 130, 180]
    ref = rpt_py.costs.CostL2().fit(signal)
    got = rpt_rs.CostL2().fit(signal)
    assert np.isclose(got.sum_of_costs(bkps), ref.sum_of_costs(bkps), rtol=1e-9)
