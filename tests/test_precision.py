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
    rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
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


def exact_ar_style_rss(y, x):
    """Residual sum of squares of `y ~ x`, in exact arithmetic."""
    n, p = x.shape

    def dot(col_a, col_b):
        return sum((col_a(i) * col_b(i) for i in range(n)), Decimal(0))

    xtx = [
        [dot(lambda i, a=a: _D(x[i, a]), lambda i, b=b: _D(x[i, b])) for b in range(p)]
        for a in range(p)
    ]
    xty = [dot(lambda i, a=a: _D(x[i, a]), lambda i: _D(y[i])) for a in range(p)]
    # Gaussian elimination in Decimal
    aug = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(p):
            if r == col or aug[col][col] == 0:
                continue
            f = aug[r][col] / aug[col][col]
            for c in range(col, p + 1):
                aug[r][c] -= f * aug[col][c]
    beta = [aug[i][p] / aug[i][i] if aug[i][i] != 0 else Decimal(0) for i in range(p)]
    total = Decimal(0)
    for i in range(n):
        pred = sum((beta[a] * _D(x[i, a]) for a in range(p)), Decimal(0))
        total += (_D(y[i]) - pred) ** 2
    return total


@pytest.mark.parametrize("slope", [0.1, 1.0, 100.0])
def test_clinear_is_accurate_on_a_trend(slope):
    """The cost is *for* trending signals, and it used to be worst on them.

    Centring removes an offset but not a trend, so on a ramp the expanded O(1)
    form still cancelled a large `m * intercept^2` term and lost most of its
    significant digits — 6e-3 relative error on a 20,000-point ramp, against
    1e-11 for the reference. The fix is that the cost is invariant under
    subtracting *any* affine function of the sample index, so the signal is
    detrended by its global least-squares line before the prefix sums are built.
    """
    n = 4000
    tt = np.arange(n)
    rng = np.random.default_rng(3)
    sig = (slope * tt + rng.normal(size=n) * 0.5).reshape(-1, 1)
    got = rpt_rs.CostCLinear().fit(sig)
    for start, end in ((1, n), (100, 3000), (2000, 2100)):
        truth = exact_clinear(sig, start, end)
        assert rel_err(got.error(start, end), truth) < Decimal("1e-9"), (
            f"slope={slope:g} segment=({start},{end})"
        )


def test_clinear_is_exact_on_a_pure_ramp():
    """A straight line is its own continuous linear approximation."""
    n = 5000
    sig = (np.arange(n) * 7.0 - 3.0).reshape(-1, 1)
    got = rpt_rs.CostCLinear().fit(sig)
    for start, end in ((1, n), (10, 400), (3000, 3100)):
        assert got.error(start, end) < 1e-6, f"({start},{end})"


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e4])
def test_ar_is_accurate_under_an_offset(offset):
    """`ar` carries an intercept column, so the residual is unchanged by shifting
    the signal — which means the signal can be centred first, exactly.

    Without that the least-squares system was solved on raw values and lost
    two significant digits per decade of offset.
    """
    sig = rpt_rs.pw_wavy(600, 3, noise_std=0.2, seed=1)[0] + offset
    got = rpt_rs.CostAR(order=3).fit(sig)
    x = got.covar
    y = np.asarray(got.signal).reshape(-1)
    for start, end in ((0, 200), (137, 400), (500, 560)):
        truth = exact_ar_style_rss(y[start:end], x[start:end])
        assert rel_err(got.error(start, end), truth) < Decimal("1e-6"), (
            f"offset={offset:g} segment=({start},{end})"
        )


def test_the_rank_decision_agrees_with_numpy_across_conditioning():
    """Where a design stops being resolvable, and that it is where NumPy says.

    The fast path solves through `X^T X`, which squares the condition number of
    `X` and so cannot tell a rank deficiency from a design it merely cannot
    resolve. Those have different right answers and one of them is `0.0` — a
    perfect fit, and the most attractive segment a search can find — so the Gram
    is no longer allowed to guess. When it cannot decide, the segment's design
    is factorised directly and NumPy's own rule is applied to the singular
    values NumPy would have seen.

    The result is agreement all the way to the genuine floor: a design at
    condition 1e13 is fitted by both, and one at 1e15 is called deficient by
    both. This test is the one to read if the factorisation ever changes.
    """
    rng = np.random.default_rng(0)
    n, seg = 400, 60
    y = rng.normal(size=n)
    c1 = rng.normal(size=n)
    c2 = rng.normal(size=n)

    def pair(scale):
        sig = np.c_[y, c1, c1 * 3.0 + scale * c2]
        ours = rpt_rs.CostLinear().fit(sig).error(0, seg)
        design = sig[:seg, 1:]
        _, residual, _, _ = np.linalg.lstsq(design, sig[:seg, 0], rcond=None)
        return ours, float(np.sum(residual)), float(np.linalg.cond(design))

    for scale in (1e-3, 1e-5, 1e-8, 1e-10, 1e-12):
        ours, theirs, cond = pair(scale)
        assert theirs > 1.0, "the reference should still be fitting this design"
        # Two different factorisations of the same ill-conditioned design agree
        # to about `eps * cond`, and no better: that is what conditioning means.
        # Demanding more would be asserting that one of them is exact.
        tol = max(1e-9, 20 * np.finfo(float).eps * cond)
        assert abs(ours - theirs) <= tol * theirs, (
            "at scale {:g} (cond {:.1e}) we report {!r} where NumPy reports {!r}".format(
                scale, cond, ours, theirs
            )
        )

    # Past the floor both give up, and give up the same way.
    for scale in (1e-14, 1e-16, 0.0):
        ours, theirs, _ = pair(scale)
        assert ours == 0.0 and theirs == 0.0, (
            "at scale {:g}: {!r} vs {!r}".format(scale, ours, theirs)
        )


def test_rank_deficient_design_reports_no_residual():
    """NumPy's `lstsq` returns an empty residual for a rank-deficient design,
    which `ruptures` sums to `0.0`. An exactly duplicated covariate column has
    to land there rather than producing rounding noise.
    """
    lin = rpt_rs.pw_linear(200, 2, 3, noise_std=1, seed=2)[0]
    collinear = np.c_[lin[:, 0], lin[:, 1], lin[:, 1] * 3.0]
    assert rpt_rs.CostLinear().fit(collinear).error(0, 40) == 0.0
    # A well-conditioned design of the same shape still reports a real residual.
    assert rpt_rs.CostLinear().fit(lin).error(0, 40) > 1.0


def test_variance_is_never_negative_or_a_silent_zero():
    """The clamp that keeps a sum of squared deviations non-negative must not
    also swallow a NaN, which `f64::max` does."""
    rng = np.random.default_rng(0)
    sig = rng.normal(scale=1e-6, size=(500, 1)) + 1e9
    cost = rpt_rs.CostL2().fit(sig)
    for start in range(0, 440, 17):
        for length in (2, 5, 50):
            assert cost.error(start, start + length) >= 0.0
    poisoned = sig.copy()
    poisoned[100, 0] = np.nan
    assert np.isnan(rpt_rs.CostL2().fit(poisoned).error(90, 120))


# ------------------------------------------------- cancellation, not centring


@pytest.mark.parametrize(
    "model,amplitude,expected", [("l2", 1e10, 82.5), ("l1", 1e18, 25.0)]
)
def test_local_detail_survives_an_unrelated_large_regime(model, amplitude, expected):
    """A small regime beside a huge one must still be scored on its own terms.

    Centring the signal by its global mean is what makes the prefix-sum form
    accurate, and it only works when there *is* one level to centre on. Here
    there are two, sixteen orders of magnitude apart, and the second regime's
    cost is smaller than the last representable bit of the prefix that contains
    it: `l2` returned `0.0` for a segment whose cost is 82.5, which reads as a
    perfect fit and pulls breakpoints towards it.

    `l1` failed one step earlier — the centring itself rounded the local detail
    away, because at 1e18 the samples 0..49 all map to the same double. It no
    longer centres, having nothing to gain from it: a median and a sum of
    absolute deviations are exact at any magnitude.
    """
    signal = np.r_[np.full(50, amplitude), np.arange(50.0)]
    got = rpt_rs.cost_factory(model).fit(signal).error(50, 60)
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e9])
@pytest.mark.parametrize("model", ["l2", "l1", "normal", "mahalanobis"])
def test_costs_are_invariant_under_a_shift(model, offset):
    """Every one of these costs is defined on deviations, so an offset is not
    supposed to be visible in the answer.

    The offsets stop at 1e9 because past that the *input* stops carrying the
    answer: at 1e12 the spacing of doubles is 2.4e-4, so unit-scale deviations
    are quantised on the way in and `base + offset` is a genuinely different
    signal from `base`. No cost function can recover what the array no longer
    holds, and demanding that it does would be testing floating point rather
    than this package. What the offsets here do check is that nothing *else*
    goes wrong on the way — which is where `clinear`, `ar` and `linear` were
    each losing digits before, and where `l1` was losing all of them.
    """
    rng = np.random.default_rng(4)
    base = rng.normal(size=(120, 2))
    plain = rpt_rs.cost_factory(model).fit(base).error(20, 80)
    shifted = rpt_rs.cost_factory(model).fit(base + offset).error(20, 80)
    assert shifted == pytest.approx(plain, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("scale", [1e80, 1e-100])
def test_linear_residual_is_invariant_under_predictor_scaling(scale):
    """Multiplying a design by a constant does not make it ill conditioned.

    The eigensolver's convergence test sums squared entries, and at these scales
    that sum overflowed at one end and underflowed to zero at the other. Both
    stopped the sweep before anything had been diagonalised, and the residual
    came back wrong — 1.29 where NumPy says 0.544 — with nothing to indicate it.
    """
    rng = np.random.default_rng(42)
    design = rng.normal(size=(60, 2))
    response = 2 * design[:, 0] - design[:, 1] + rng.normal(size=60) * 0.1
    _, residual, _, _ = np.linalg.lstsq(design * scale, response, rcond=None)
    got = rpt_rs.CostLinear().fit(np.c_[response, design * scale]).error(0, 60)
    assert got == pytest.approx(float(residual.sum()), rel=1e-9)


def test_multivariate_ar_matches_the_reference():
    """`ar` builds its lagged design by walking the flattened signal.

    With more than one column a row of lags therefore spans a column boundary,
    and subtracting a *per-column* mean first — which is exact for one column,
    because the intercept absorbs it — silently changed the model. One scalar
    mean over the whole buffer is absorbed for any number of columns.
    """
    rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
    signal = np.random.default_rng(5).normal(size=(40, 2))
    ours = rpt_rs.CostAR(order=2).fit(signal).error(5, 35)
    theirs = rpt_py.costs.CostAR(order=2).fit(signal.copy()).error(5, 35)
    assert ours == pytest.approx(theirs, rel=1e-9)
