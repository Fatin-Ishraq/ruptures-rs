"""Randomised differential fuzzing against `ruptures`.

The hand-written cases in `test_equivalence.py` cover the parameters a person
thinks to try. This file covers the ones they don't: random signal lengths,
dimensions, `jump`/`min_size` combinations, penalties and seeds, checked for
exact breakpoint equality.

Tie-breaking bugs and off-by-one admissibility errors hide in exactly these
corners, so the case count is deliberately high and the assertion is exact.
"""

import warnings

import numpy as np
import pytest

# Every test here compares against `ruptures`.
# On Python versions the reference itself does not support - it declares
# `requires_python = "<3.14"` - skip rather than fail. This package works
# on those versions even though the reference does not.
rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)

MODELS = ["l1", "l2", "normal", "rbf", "rank", "mahalanobis", "clinear"]


def random_case(rng):
    n = int(rng.integers(60, 320))
    n_features = int(rng.integers(1, 4))
    model = str(rng.choice(MODELS))
    # `mahalanobis` needs a non-singular covariance, so keep it multivariate
    if model == "mahalanobis" and n_features == 1:
        n_features = 2
    jump = int(rng.integers(1, 11))
    min_size = int(rng.integers(2, 15))
    seed = int(rng.integers(0, 10_000))
    sig = rpt_rs.pw_constant(n, n_features, 3, noise_std=float(rng.uniform(0.5, 4)), seed=seed)[0]
    return sig, model, jump, min_size


def _skip_if_infeasible(mod, detector, sig, model, jump, min_size, kwargs):
    """Run the reference; if it refuses these parameters, so should we."""
    est = getattr(mod, detector)(model=model, min_size=min_size, jump=jump)
    try:
        return [int(b) for b in est.fit(sig).predict(**kwargs)], None
    except Exception as exc:  # noqa: BLE001 - we compare the exception type
        return None, type(exc).__name__


@pytest.mark.parametrize("trial", range(120))
def test_fuzz_dynp(trial):
    rng = np.random.default_rng(1000 + trial)
    sig, model, jump, min_size = random_case(rng)
    n_bkps = int(rng.integers(1, 5))
    kwargs = {"n_bkps": n_bkps}
    expected, exc_ref = _skip_if_infeasible(rpt_py, "Dynp", sig, model, jump, min_size, kwargs)
    actual, exc_got = _skip_if_infeasible(rpt_rs, "Dynp", sig, model, jump, min_size, kwargs)
    assert exc_got == exc_ref, f"{model} jump={jump} min_size={min_size}"
    assert actual == expected, f"{model} jump={jump} min_size={min_size} n_bkps={n_bkps}"


@pytest.mark.parametrize("trial", range(120))
def test_fuzz_pelt(trial):
    rng = np.random.default_rng(2000 + trial)
    sig, model, jump, min_size = random_case(rng)
    pen = float(rng.uniform(1, 1500))
    kwargs = {"pen": pen}
    expected, exc_ref = _skip_if_infeasible(rpt_py, "Pelt", sig, model, jump, min_size, kwargs)
    actual, exc_got = _skip_if_infeasible(rpt_rs, "Pelt", sig, model, jump, min_size, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{model} jump={jump} min_size={min_size} pen={pen:.3f}"


@pytest.mark.parametrize("trial", range(120))
def test_fuzz_binseg(trial):
    rng = np.random.default_rng(3000 + trial)
    sig, model, jump, min_size = random_case(rng)
    kwargs = {"n_bkps": int(rng.integers(1, 5))}
    expected, exc_ref = _skip_if_infeasible(rpt_py, "Binseg", sig, model, jump, min_size, kwargs)
    actual, exc_got = _skip_if_infeasible(rpt_rs, "Binseg", sig, model, jump, min_size, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{model} jump={jump} min_size={min_size}"


@pytest.mark.parametrize("trial", range(80))
def test_fuzz_bottomup(trial):
    rng = np.random.default_rng(4000 + trial)
    sig, model, jump, min_size = random_case(rng)
    kwargs = {"n_bkps": int(rng.integers(1, 5))}
    expected, exc_ref = _skip_if_infeasible(rpt_py, "BottomUp", sig, model, jump, min_size, kwargs)
    actual, exc_got = _skip_if_infeasible(rpt_rs, "BottomUp", sig, model, jump, min_size, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{model} jump={jump} min_size={min_size}"


@pytest.mark.parametrize("trial", range(60))
def test_fuzz_cost_values(trial):
    """Segment costs across random models, shapes and offsets."""
    rng = np.random.default_rng(5000 + trial)
    sig, model, _, _ = random_case(rng)
    # Moderate offsets only: at 1e5 and beyond `ruptures` itself loses
    # precision, so it stops being a valid oracle. Those cases are covered
    # against exact arithmetic in `test_precision.py` instead.
    sig = sig + float(rng.choice([0.0, 1e2]))
    ref = rpt_py.costs.cost_factory(model=model).fit(sig)
    got = rpt_rs.cost_factory(model=model).fit(sig)
    n = sig.shape[0]
    for _ in range(40):
        start = int(rng.integers(0, n - 20))
        end = int(rng.integers(start + max(got.min_size, 4), n + 1))
        expected = float(np.asarray(ref.error(start, end)).sum())
        actual = got.error(start, end)
        scale = max(1.0, abs(expected))
        assert abs(actual - expected) <= 1e-6 * scale, (
            f"{model} error({start},{end}): {actual!r} != {expected!r}"
        )


# The categories below were absent when this file was written, and each one hid
# a real defect: `Window` never ran under the fuzzer at all, `KernelCPD` was
# only ever asked for a fixed number of breakpoints, and `epsilon` was never a
# stopping rule. All three are differential like the rest.


def _pair(detector, sig, est, kwargs):
    def run(mod):
        try:
            return [int(b) for b in getattr(mod, detector)(**est).fit(sig).predict(**kwargs)], None
        except Exception as exc:  # noqa: BLE001 - the type is part of the assertion
            return None, type(exc).__name__

    return run(rpt_py), run(rpt_rs)


@pytest.mark.parametrize("trial", range(80))
def test_fuzz_window(trial):
    """`Window` across every model and all three stopping rules."""
    rng = np.random.default_rng(6000 + trial)
    sig, model, jump, min_size = random_case(rng)
    kwargs = [
        {"n_bkps": int(rng.integers(1, 5))},
        {"pen": float(rng.uniform(1, 500))},
        {"epsilon": float(rng.uniform(10, 5000))},
    ][trial % 3]
    est = dict(
        width=int(rng.integers(6, 60)), model=model, min_size=min_size, jump=jump
    )
    (expected, exc_ref), (actual, exc_got) = _pair("Window", sig, est, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{model} {est} {kwargs}"


@pytest.mark.parametrize("trial", range(60))
def test_fuzz_epsilon_stopping(trial):
    """`epsilon` reaches a different branch of `Binseg` and `BottomUp`."""
    rng = np.random.default_rng(7500 + trial)
    sig, model, jump, min_size = random_case(rng)
    detector = ["Binseg", "BottomUp"][trial % 2]
    kwargs = {"epsilon": float(rng.uniform(10, 8000))}
    est = dict(model=model, min_size=min_size, jump=jump)
    (expected, exc_ref), (actual, exc_got) = _pair(detector, sig, est, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{detector} {model} {est} {kwargs}"


@pytest.mark.parametrize("trial", range(60))
def test_fuzz_regression_models(trial):
    """`linear`, `ar` and `cosine` were covered by one hand-written case each."""
    rng = np.random.default_rng(8500 + trial)
    n = int(rng.integers(60, 250))
    jump = int(rng.integers(1, 8))
    min_size = int(rng.integers(2, 10))
    detector = ["Dynp", "Pelt", "Binseg", "BottomUp"][trial % 4]
    kwargs = (
        {"pen": float(rng.uniform(1, 300))}
        if detector == "Pelt"
        else {"n_bkps": int(rng.integers(1, 5))}
    )
    kind = trial % 3
    if kind == 0:
        sig = rpt_rs.pw_linear(
            n, int(rng.integers(1, 4)), 3, noise_std=1, seed=int(rng.integers(0, 9999))
        )[0]
        est = dict(model="linear", min_size=min_size, jump=jump)
    elif kind == 1:
        sig = rpt_rs.pw_wavy(n, 3, noise_std=0.3, seed=int(rng.integers(0, 9999)))[0]
        est = dict(
            model="ar",
            min_size=min_size,
            jump=jump,
            params={"order": int(rng.integers(1, 6))},
        )
    else:
        base = rpt_rs.pw_wavy(n, 3, noise_std=0.3, seed=int(rng.integers(0, 9999)))[0]
        sig = np.c_[base, np.roll(base, 3) + 2.0]
        est = dict(model="cosine", min_size=min_size, jump=jump)
    (expected, exc_ref), (actual, exc_got) = _pair(detector, sig, est, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{detector} {est} {kwargs}"


@pytest.mark.parametrize("trial", range(60))
def test_fuzz_kernelcpd(trial):
    """`KernelCPD` is the one detector `ruptures` implements in C, and that C
    does not agree with `ruptures`' own Python cost classes.

    Its cosine kernel has a unit diagonal where `scipy`'s `squareform` leaves a
    zero one, which is worth exactly one unit of penalty per segment; its
    Gaussian exponent is clipped in single precision; and its PELT folds the
    penalty in and prunes differently. Reproducing the Python cost here instead
    gave a different segmentation whenever a penalty was involved.
    """
    rng = np.random.default_rng(9500 + trial)
    n = int(rng.integers(40, 220))
    d = int(rng.integers(1, 3))
    sig = rpt_rs.pw_constant(
        n, d, 3, noise_std=float(rng.uniform(0.5, 4)), seed=int(rng.integers(0, 9999))
    )[0]
    kernel = ["linear", "rbf", "cosine"][trial % 3]
    est = dict(kernel=kernel, min_size=int(rng.integers(1, 6)))
    kwargs = (
        {"n_bkps": int(rng.integers(1, 5))}
        if trial % 2
        else {"pen": float(rng.uniform(0.5, 50))}
    )
    (expected, exc_ref), (actual, exc_got) = _pair("KernelCPD", sig, est, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{kernel} {est} {kwargs}"


@pytest.mark.parametrize("trial", range(40))
def test_fuzz_offset_signals(trial):
    """Offsets up to 1e4, where `ruptures` is still an accurate oracle."""
    rng = np.random.default_rng(10500 + trial)
    n = int(rng.integers(60, 250))
    d = int(rng.integers(1, 3))
    model = ["l2", "normal", "clinear", "l1", "rbf"][trial % 5]
    sig = (
        rpt_rs.pw_constant(n, d, 3, noise_std=2, seed=int(rng.integers(0, 9999)))[0]
        + float(rng.choice([1e3, 1e4]))
    )
    detector = ["Dynp", "Pelt", "Binseg", "BottomUp"][trial % 4]
    kwargs = (
        {"pen": float(rng.uniform(1, 300))}
        if detector == "Pelt"
        else {"n_bkps": int(rng.integers(1, 5))}
    )
    est = dict(model=model, min_size=5, jump=int(rng.integers(1, 6)))
    (expected, exc_ref), (actual, exc_got) = _pair(detector, sig, est, kwargs)
    assert exc_got == exc_ref
    assert actual == expected, f"{detector} {model} {est} {kwargs}"
