"""Accuracy on ill-conditioned input, judged against exact arithmetic.

Elsewhere `ruptures` is the oracle. Here it cannot be: on signals whose offset
dwarfs their spread, several `ruptures` cost functions lose most of their
significant digits, so agreeing with them would mean being wrong.

The oracle instead is `decimal.Decimal` at 60 digits — slow, but exact enough to
settle who is right. These tests assert that this package stays accurate where
the reference does not, which is a property worth defending against regressions
in either direction.
"""

import warnings
from decimal import Decimal, getcontext

import numpy as np
import pytest

import ruptures as rpt_py
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)
getcontext().prec = 60

OFFSETS = [0.0, 1e3, 1e6, 1e9]


def _D(v):
    return Decimal(float(v))


def exact_l2(sig, start, end):
    """Sum of squared deviations from the segment mean, in exact arithmetic."""
    n = end - start
    total = Decimal(0)
    for j in range(sig.shape[1]):
        mu = sum((_D(sig[i, j]) for i in range(start, end)), Decimal(0)) / _D(n)
        total += sum(((_D(sig[i, j]) - mu) ** 2 for i in range(start, end)), Decimal(0))
    return total


def exact_mahalanobis(sig, metric, start, end):
    n = end - start
    d = sig.shape[1]
    mu = [
        sum((_D(sig[i, j]) for i in range(start, end)), Decimal(0)) / _D(n)
        for j in range(d)
    ]
    total = Decimal(0)
    for i in range(start, end):
        for a in range(d):
            for b in range(d):
                total += (_D(sig[i, a]) - mu[a]) * _D(metric[a, b]) * (_D(sig[i, b]) - mu[b])
    return total


def exact_clinear(sig, start, end):
    if start == 0:
        start = 1
    m = end - start
    total = Decimal(0)
    for j in range(sig.shape[1]):
        slope = (_D(sig[end - 1, j]) - _D(sig[start - 1, j])) / _D(m)
        intercept = _D(sig[start - 1, j])
        for k in range(m):
            approx = slope * _D(k + 1) + intercept
            total += (_D(sig[start + k, j]) - approx) ** 2
    return total


def rel_err(actual, expected):
    expected = Decimal(expected)
    if expected == 0:
        return abs(Decimal(actual))
    return abs((Decimal(actual) - expected) / expected)


@pytest.mark.parametrize("offset", OFFSETS)
def test_l2_stays_accurate(offset):
    sig = rpt_rs.pw_constant(120, 2, 3, noise_std=2, seed=1)[0] + offset
    got = rpt_rs.CostL2().fit(sig)
    for start, end in ((0, 120), (13, 77), (40, 46)):
        truth = exact_l2(sig, start, end)
        assert rel_err(got.error(start, end), truth) < Decimal("1e-9"), (
            f"offset={offset:g} segment=({start},{end})"
        )


@pytest.mark.parametrize("offset", OFFSETS)
def test_clinear_stays_accurate(offset):
    """This one regressed once: the O(1) expansion carries an `m*b^2` term."""
    sig = rpt_rs.pw_constant(120, 2, 3, noise_std=2, seed=4)[0] + offset
    got = rpt_rs.CostCLinear().fit(sig)
    for start, end in ((5, 120), (17, 90), (40, 51)):
        truth = exact_clinear(sig, start, end)
        assert rel_err(got.error(start, end), truth) < Decimal("1e-8"), (
            f"offset={offset:g} segment=({start},{end})"
        )


@pytest.mark.parametrize("offset", OFFSETS)
def test_mahalanobis_stays_accurate(offset):
    sig = rpt_rs.pw_constant(120, 2, 3, noise_std=2, seed=6)[0] + offset
    got = rpt_rs.CostMl().fit(sig)
    metric = np.linalg.inv(np.cov(sig.T))
    for start, end in ((0, 120), (23, 88), (40, 47)):
        truth = exact_mahalanobis(sig, metric, start, end)
        assert rel_err(got.error(start, end), truth) < Decimal("1e-8"), (
            f"offset={offset:g} segment=({start},{end})"
        )


def test_beats_ruptures_on_extreme_offsets():
    """Documents the improvement rather than just asserting parity.

    At an offset of 1e9, `ruptures`' Gram-matrix Mahalanobis and its explicit
    `clinear` residual have almost no correct digits left, because both form a
    small difference of large quantities. Working from centred prefix sums
    avoids that. If this test ever fails, either the reference improved or this
    package regressed — both worth knowing.
    """
    sig = rpt_rs.pw_constant(120, 2, 3, noise_std=2, seed=6)[0] + 1e9
    metric = np.linalg.inv(np.cov(sig.T))
    start, end = 23, 88

    truth = exact_mahalanobis(sig, metric, start, end)
    ours = rpt_rs.CostMl().fit(sig).error(start, end)
    theirs = float(np.asarray(rpt_py.costs.CostMl().fit(sig).error(start, end)).sum())

    assert rel_err(ours, truth) < Decimal("1e-8")
    assert rel_err(ours, truth) < rel_err(theirs, truth)


@pytest.mark.parametrize("offset", OFFSETS)
def test_segmentation_is_offset_invariant(offset):
    """A constant shift must not move the breakpoints."""
    base = rpt_rs.pw_constant(200, 1, 3, noise_std=2, seed=8)[0]
    reference = rpt_rs.Dynp(model="l2", min_size=5, jump=5).fit(base).predict(3)
    shifted = rpt_rs.Dynp(model="l2", min_size=5, jump=5).fit(base + offset).predict(3)
    assert shifted == reference


def test_variance_never_negative():
    """The classic one-pass-variance failure mode, asserted away."""
    rng = np.random.default_rng(0)
    sig = rng.normal(scale=1e-6, size=(500, 1)) + 1e9
    cost = rpt_rs.CostL2().fit(sig)
    for start in range(0, 440, 17):
        for length in (2, 5, 50):
            assert cost.error(start, start + length) >= 0.0
