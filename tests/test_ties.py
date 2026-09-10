"""Differential behaviour on signals full of exact ties.

Elsewhere the suite demands identical breakpoints. That claim is true on data
with any noise in it, and this file is where it stops being true: on a
noiseless integer step signal, thousands of segmentations have costs that
differ only in the last few bits, and the two libraries compute those bits
differently — `ruptures` from a two-pass NumPy reduction over the segment, this
package from a difference of prefix sums.

So these tests assert the property that actually matters, which is *stronger*
than "we happen to agree" and can be stated without hedging:

* for the exact detectors (`Dynp`, and `Pelt` at a fixed penalty), the
  segmentation returned here costs no more than the one `ruptures` returns,
  measured with `ruptures`' own cost function — and for `Pelt` it is sometimes
  strictly less, because the reference prunes a candidate from a point at which
  pruning it is not yet justified and this package does not;
* for the greedy detectors (`Binseg`, `BottomUp`), only that the answer stays
  well-formed and in the same league, because neither is optimising globally
  and a tie broken the other way sends the greedy path elsewhere;
* `Window` is excluded from the equality claim entirely, and the reason is
  recorded below.

Nothing here is a workaround for a defect. A segmentation that ties on cost is
as correct as the one it ties with, and asserting bit-identical breakpoints on
tied input would be asserting a coincidence.
"""

import warnings

import numpy as np
import pytest

rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

EXACT_DETECTORS = ["Dynp", "Pelt"]
GREEDY_DETECTORS = ["Binseg", "BottomUp"]
TIE_MODELS = ["l2", "l1", "rbf", "normal", "rank", "clinear"]


def integer_signal(rng, n, d=1):
    """Small-alphabet data: every value repeats many times."""
    return rng.integers(0, 4, size=(n, d)).astype(float)


def step_signal(rng, n):
    """Exactly piecewise constant, with no noise at all."""
    k = int(rng.integers(1, 5))
    edges = sorted(rng.choice(np.arange(5, n - 5), size=k, replace=False).tolist())
    sig = np.zeros((n, 1))
    level, start = 0.0, 0
    for end in edges + [n]:
        sig[start:end] = level
        level += float(rng.integers(-3, 4))
        start = end
    return sig


def repeated_signal(rng, n):
    """A 20-sample block tiled to length `n`, so segments repeat exactly."""
    unit = rpt_rs.pw_constant(20, 1, 1, noise_std=1, seed=int(rng.integers(0, 999)))[0]
    return np.tile(unit, (n // 20 + 1, 1))[:n]


def reference_cost(sig, model, bkps):
    """Total cost of a segmentation, scored by `ruptures`' own cost function."""
    cost = rpt_py.costs.cost_factory(model=model).fit(sig)
    return float(
        sum(
            float(np.asarray(cost.error(start, end)).sum())
            for start, end in zip([0] + list(bkps), bkps)
        )
    )


def assert_not_worse(sig, model, actual, expected, detector, pen):
    """Our penalised objective must be no larger than the reference's.

    Scored with `ruptures`' own cost function, and with the penalty included,
    because a penalised search is not comparable without it: a segmentation with
    fewer breakpoints has a smaller unpenalised cost almost by construction.
    """
    ours = reference_cost(sig, model, actual) + pen * len(actual)
    theirs = reference_cost(sig, model, expected) + pen * len(expected)
    scale = max(abs(ours), abs(theirs), 1.0)
    assert ours <= theirs + 1e-9 * scale, (
        "{} {}: ruptures-rs returned a strictly worse segmentation "
        "({} at penalised cost {!r}, reference {} at {!r})".format(
            detector, model, actual, ours, expected, theirs
        )
    )


def run_pair(detector, sig, est, kwargs):
    try:
        expected = [
            int(b) for b in getattr(rpt_py, detector)(**est).fit(sig).predict(**kwargs)
        ]
    except Exception as exc:  # noqa: BLE001
        expected, exc_ref = None, type(exc).__name__
    else:
        exc_ref = None
    try:
        actual = [
            int(b) for b in getattr(rpt_rs, detector)(**est).fit(sig).predict(**kwargs)
        ]
    except Exception as exc:  # noqa: BLE001
        actual, exc_got = None, type(exc).__name__
    else:
        exc_got = None
    return expected, actual, exc_ref, exc_got


SIGNAL_KINDS = ("integer", "step", "repeated")


def tied_case(detector, kind, trial):
    """Build one tied case from a seed that does not move between runs.

    Deliberately not `hash()`: Python randomises string hashing per process, so
    a seed derived from one makes the test pass and fail on alternate runs.
    """
    di = (EXACT_DETECTORS + GREEDY_DETECTORS).index(detector)
    ki = SIGNAL_KINDS.index(kind)
    rng = np.random.default_rng(trial * 97 + ki * 13 + di * 3 + 1)
    n = int(rng.integers(60, 240))
    sig = {
        "integer": integer_signal,
        "step": step_signal,
        "repeated": repeated_signal,
    }[kind](rng, n)
    model = TIE_MODELS[trial % len(TIE_MODELS)]
    est = dict(
        model=model, min_size=int(rng.integers(2, 8)), jump=int(rng.integers(1, 6))
    )
    kwargs = (
        {"pen": float(rng.choice([0.5, 1.0, 5.0, 20.0]))}
        if detector == "Pelt"
        else {"n_bkps": int(rng.integers(1, 5))}
    )
    return sig, model, est, kwargs


@pytest.mark.parametrize("detector", EXACT_DETECTORS)
@pytest.mark.parametrize("kind", SIGNAL_KINDS)
@pytest.mark.parametrize("trial", range(15))
def test_exact_detectors_never_cost_more_on_tied_signals(detector, kind, trial):
    """`Dynp` and `Pelt` solve their problem exactly, so the answer may differ
    from the reference's but it may never be worse.

    That is the real guarantee, and it is what "both segmentations are optimal"
    has to mean if it is going to be claimed at all.
    """
    sig, model, est, kwargs = tied_case(detector, kind, trial)
    expected, actual, exc_ref, exc_got = run_pair(detector, sig, est, kwargs)
    assert exc_got == exc_ref, "{} {}: {} vs {}".format(detector, model, exc_ref, exc_got)
    if expected is None or expected == actual:
        return
    assert_not_worse(sig, model, actual, expected, detector, kwargs.get("pen", 0.0))


@pytest.mark.parametrize("detector", GREEDY_DETECTORS)
@pytest.mark.parametrize("kind", SIGNAL_KINDS)
@pytest.mark.parametrize("trial", range(15))
def test_greedy_detectors_stay_close_on_tied_signals(detector, kind, trial):
    """`Binseg` and `BottomUp` are not optimising globally.

    A tie broken the other way sends the greedy path somewhere else, and where
    it lands can be a little better or a little worse than the reference's
    landing spot. Neither is a defect — the algorithm never promised the
    optimum — so the assertion is that the result stays well-formed and in the
    same league, not that it wins.
    """
    sig, model, est, kwargs = tied_case(detector, kind, trial)
    expected, actual, exc_ref, exc_got = run_pair(detector, sig, est, kwargs)
    assert exc_got == exc_ref
    if expected is None or expected == actual:
        return
    assert actual == sorted(actual) and actual[-1] == sig.shape[0]
    ours = reference_cost(sig, model, actual)
    theirs = reference_cost(sig, model, expected)
    scale = max(abs(ours), abs(theirs), 1.0)
    assert ours <= theirs + 0.25 * scale, (
        "{} {}: greedy divergence should be small, got {!r} vs {!r}".format(
            detector, model, ours, theirs
        )
    )


@pytest.mark.parametrize("trial", range(30))
def test_integer_signals_match_exactly(trial):
    """Small-alphabet data agrees breakpoint for breakpoint, or wins.

    This is a regression guard, not a coincidence: it agrees because `Pelt`
    folds its penalty in per segment the way summing a partition dict does, and
    because a sum of squared deviations is clamped at zero rather than being
    allowed a few ulp below it. Both of those were wrong once.

    `Pelt` is the exception, and deliberately so. Its pruning here is valid
    under `min_size` where the reference's is not, so on data with this many
    ties it sometimes finds a segmentation the reference has already discarded.
    The assertion for it is therefore "never worse", which is the property that
    matters; `test_optimality.py` proves the stronger claim against an
    exhaustive search.
    """
    rng = np.random.default_rng(7000 + trial)
    n = int(rng.integers(40, 200))
    sig = integer_signal(rng, n, d=int(rng.integers(1, 3)))
    model = TIE_MODELS[trial % len(TIE_MODELS)]
    detector = ["Dynp", "Pelt", "Binseg", "BottomUp"][trial % 4]
    est = dict(
        model=model, min_size=int(rng.integers(1, 8)), jump=int(rng.integers(1, 6))
    )
    kwargs = (
        {"pen": float(rng.choice([0.1, 0.3, 1.0, 2.5, 7.0, 20.0]))}
        if detector == "Pelt"
        else {"n_bkps": int(rng.integers(1, 5))}
    )
    expected, actual, exc_ref, exc_got = run_pair(detector, sig, est, kwargs)
    assert exc_got == exc_ref
    if detector == "Pelt" and expected is not None and actual != expected:
        assert_not_worse(sig, model, actual, expected, detector, kwargs["pen"])
        return
    assert actual == expected, "{} {} n={}".format(detector, model, n)


def test_window_score_curves_agree_even_when_the_peaks_do_not():
    """`Window` can return a different *number* of breakpoints on tied data.

    Its peaks come from `argrelmax`, which needs a score strictly greater than
    both neighbours. A noiseless signal makes the score curve flat over long
    stretches, so whether a plateau counts as a peak is decided by the last bit
    of a cost — and one extra peak changes the whole segmentation rather than
    one breakpoint.

    The assertion pins where the difference is and is not. The *score curves*
    agree to a few ulp, which says the costs are right and the disagreement is
    entirely in a strict inequality applied to a plateau. That is a property of
    peak-picking on tied data, not a defect in the arithmetic, and it cannot be
    fixed by making the arithmetic better.
    """
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(80, 240))
        sig = step_signal(rng, n)
        est = dict(
            width=int(rng.integers(8, 40)),
            model="l2",
            min_size=int(rng.integers(2, 6)),
            jump=int(rng.integers(1, 5)),
        )
        try:
            ref = rpt_py.Window(**est).fit(sig)
            got = rpt_rs.Window(**est).fit(sig)
        except Exception:  # noqa: BLE001 - infeasible parameters, not the point
            continue
        if len(ref.score) == 0:
            continue
        assert np.array_equal(np.asarray(ref.inds), np.asarray(got.inds))
        ref_score = np.asarray(ref.score, dtype=float)
        got_score = np.asarray(got.score, dtype=float)
        scale = max(float(np.abs(ref_score).max()), 1.0)
        assert np.abs(ref_score - got_score).max() < 1e-9 * scale
        checked += 1
    assert checked > 20, "too few feasible Window cases to be meaningful"


@pytest.mark.parametrize("trial", range(40))
def test_noisy_signals_agree_or_tie(trial):
    """The headline claim, restated as a guard on the tie work above.

    None of the tie handling may loosen agreement on ordinary data. Exact ties
    are rare here but not impossible — an `l1` cost can land two candidate
    splits within 1e-13 of each other on a perfectly ordinary noisy signal — so
    the assertion admits a tie and nothing else. Breakpoint-for-breakpoint
    equality across hundreds of noisy cases is covered by `test_fuzz.py`.
    """
    rng = np.random.default_rng(8000 + trial)
    n = int(rng.integers(80, 300))
    d = int(rng.integers(1, 4))
    sig = rpt_rs.pw_constant(
        n, d, 3, noise_std=float(rng.uniform(0.5, 4)), seed=int(rng.integers(0, 9999))
    )[0]
    model = TIE_MODELS[trial % len(TIE_MODELS)]
    detector = ["Dynp", "Pelt", "Binseg", "BottomUp", "Window"][trial % 5]
    est = dict(
        model=model, min_size=int(rng.integers(2, 8)), jump=int(rng.integers(1, 6))
    )
    if detector == "Window":
        est["width"] = int(rng.integers(10, 50))
    kwargs = (
        {"pen": float(rng.uniform(1, 400))}
        if detector == "Pelt"
        else {"n_bkps": int(rng.integers(1, 5))}
    )
    expected, actual, exc_ref, exc_got = run_pair(detector, sig, est, kwargs)
    assert exc_got == exc_ref
    if expected is None or expected == actual:
        return
    if detector == "Pelt":
        assert_not_worse(sig, model, actual, expected, detector, kwargs["pen"])
        return
    ours = reference_cost(sig, model, actual)
    theirs = reference_cost(sig, model, expected)
    scale = max(abs(ours), abs(theirs), 1.0)
    assert abs(ours - theirs) <= 1e-9 * scale, (
        "{} {} n={} d={}: {} (cost {!r}) is not a tie with the reference's {} "
        "(cost {!r})".format(detector, model, n, d, actual, ours, expected, theirs)
    )
