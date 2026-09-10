"""Optimality, checked against exhaustive search rather than against `ruptures`.

Agreeing with the reference is evidence about compatibility. It is not evidence
about correctness, and treating it as though it were is how a defect survives a
thousand passing tests: `ruptures`' PELT discards a candidate from a point at
which the pruning inequality does not yet hold, so on some inputs it returns a
segmentation that a feasible alternative beats — and a differential test sees
two libraries agreeing.

So this file does not ask what `ruptures` returns. It searches every legal
segmentation of a small signal, scoring each with a plain two-pass NumPy
reduction written here, and asserts that the detectors return an optimum.

The oracle is an unpruned dynamic program — every admissible predecessor of
every position, no shortcuts — and it is itself checked against literal
enumeration on signals small enough to enumerate. The cases are numerous and
randomly parameterised because the failure this exists to catch depended on the
interaction of `min_size`, `jump` and the penalty.
"""

import itertools

import numpy as np
import pytest

import ruptures_rs as rpt


# ------------------------------------------------------------------ oracle


def segment_cost(signal, start, end):
    """Sum of squared deviations, two-pass, with no prefix sums anywhere.

    Deliberately the slowest possible spelling of the `l2` cost. It shares no
    code with the thing it is checking — that is the entire point of it.
    """
    sub = signal[start:end]
    return float(np.sum((sub - sub.mean(axis=0)) ** 2))


def all_segmentations(n, min_size, jump):
    """Every breakpoint list legal under these parameters, ending at `n`.

    Interior breakpoints are multiples of `jump`, every segment is at least
    `min_size` long, and the unsplit signal `[n]` is included. Combinatorial, so
    it is only usable on tiny inputs — it exists to check the oracle below, not
    to be the oracle.
    """
    candidates = [k for k in range(jump, n, jump) if k >= min_size and n - k >= min_size]
    for size in range(0, len(candidates) + 1):
        for combo in itertools.combinations(candidates, size):
            bounds = (0,) + combo + (n,)
            if all(b - a >= min_size for a, b in zip(bounds, bounds[1:])):
                yield list(combo) + [n]


def penalised(signal, bkps, pen):
    start, total = 0, 0.0
    for end in bkps:
        total += segment_cost(signal, start, end) + pen
        start = end
    return total


def unpenalised(signal, bkps):
    return penalised(signal, bkps, 0.0)


def _grid(n, min_size, jump):
    """The positions a breakpoint may take, plus the terminal `n`."""
    return [k for k in range(0, n, jump)] + [n]


def best_penalised(signal, n, min_size, jump, pen):
    """Optimal penalised cost, by an unpruned dynamic program.

    No pruning of any kind: every admissible predecessor of every position is
    tried. That makes it O(n^2) rather than combinatorial, so it runs on signals
    long enough to be interesting, while still considering exactly the same set
    of segmentations exhaustive search would — which
    `test_the_oracle_agrees_with_brute_force` checks directly.
    """
    positions = _grid(n, min_size, jump)
    best = {0: 0.0}
    for end in positions:
        if end == 0:
            continue
        options = [
            best[start] + segment_cost(signal, start, end) + pen
            for start in positions
            if start < end and start in best and end - start >= min_size
        ]
        if options:
            best[end] = min(options)
    return best.get(n)


def best_with_k(signal, n, min_size, jump, k):
    """Cheapest segmentation with exactly `k` breakpoints, unpruned, or None."""
    positions = _grid(n, min_size, jump)
    # level[j][end] = best cost of segmenting [0, end) with j breakpoints
    level = {0: 0.0}
    for _ in range(k + 1):
        nxt = {}
        for end in positions:
            if end == 0:
                continue
            options = [
                level[start] + segment_cost(signal, start, end)
                for start in positions
                if start < end and start in level and end - start >= min_size
            ]
            if options:
                nxt[end] = min(options)
        level = nxt
    return level.get(n)


@pytest.mark.parametrize("trial", range(8))
def test_the_oracle_agrees_with_brute_force(trial):
    """The unpruned dynamic program is checked against literal enumeration.

    An oracle nobody has verified is just a second implementation of the same
    guess. On signals small enough to enumerate completely, the two must return
    the same number.
    """
    rng = np.random.default_rng(900 + trial)
    n = int(rng.integers(12, 19))
    min_size = int(rng.integers(2, 5))
    jump = int(rng.integers(1, 3))
    pen = float(rng.choice([0.3, 1.0, 3.0]))
    signal = rng.normal(size=n)
    brute = min(
        penalised(signal, bkps, pen) for bkps in all_segmentations(n, min_size, jump)
    )
    assert best_penalised(signal, n, min_size, jump, pen) == pytest.approx(brute)
    for k in (1, 2):
        exact = [
            unpenalised(signal, bkps)
            for bkps in all_segmentations(n, min_size, jump)
            if len(bkps) - 1 == k
        ]
        got = best_with_k(signal, n, min_size, jump, k)
        if exact:
            assert got == pytest.approx(min(exact))
        else:
            assert got is None


# ------------------------------------------------------------------- Pelt


@pytest.mark.parametrize("trial", range(60))
def test_pelt_attains_the_exhaustive_optimum(trial):
    """PELT is an exact algorithm, and this is what that has to mean.

    The parameters cover the region where `ruptures` gets this wrong: a
    `min_size` large enough that a pruned candidate stays useful for several
    more targets, and a penalty small enough that the search is genuinely
    choosing.
    """
    rng = np.random.default_rng(1000 + trial)
    n = int(rng.integers(14, 45))
    min_size = int(rng.integers(2, min(8, n // 2 + 1)))
    jump = int(rng.integers(1, 4))
    pen = float(rng.choice([0.2, 0.5, 1.0, 2.0, 5.0]))
    signal = rng.normal(size=n)

    bkps = rpt.Pelt(model="l2", min_size=min_size, jump=jump).fit_predict(signal, pen)
    ours = penalised(signal, bkps, pen)
    best = best_penalised(signal, n, min_size, jump, pen)
    assert ours <= best + 1e-9 * max(abs(best), 1.0), (
        "n={} min_size={} jump={} pen={}: {} costs {!r}, the optimum is {!r}".format(
            n, min_size, jump, pen, bkps, ours, best
        )
    )


def test_pelt_finds_the_case_the_reference_prunes_away():
    """The specific counterexample, kept as a named regression.

    24 standard normal samples, `min_size=6`, `jump=1`, `pen=2`. `ruptures`
    discards `t = 0` at target 21 — from which the pruning inequality does not
    reach 24, because 24 is only three samples further on and no segment may be
    that short — and then cannot see the unsplit signal at all.
    """
    signal = np.random.default_rng(86).normal(size=24)
    bkps = rpt.Pelt(model="l2", min_size=6, jump=1).fit_predict(signal, pen=2.0)
    assert penalised(signal, bkps, 2.0) <= penalised(signal, [24], 2.0) + 1e-12
    # And it is the unsplit signal that wins here, by a clear margin.
    assert bkps == [24]


@pytest.mark.parametrize("trial", range(20))
def test_pelt_never_loses_to_the_reference(trial):
    """Against `ruptures` directly: different answers are allowed, worse ones are not."""
    rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
    rng = np.random.default_rng(2000 + trial)
    n = int(rng.integers(20, 60))
    min_size = int(rng.integers(2, 9))
    jump = int(rng.integers(1, 4))
    pen = float(rng.choice([0.5, 1.0, 2.0, 5.0]))
    signal = rng.normal(size=n)
    est = dict(model="l2", min_size=min_size, jump=jump)
    theirs = rpt_py.Pelt(**est).fit_predict(signal, pen)
    ours = rpt.Pelt(**est).fit_predict(signal, pen)
    assert penalised(signal, ours, pen) <= penalised(signal, theirs, pen) + 1e-9


# ------------------------------------------------------------------- Dynp


@pytest.mark.parametrize("trial", range(40))
def test_dynp_attains_the_exhaustive_optimum(trial):
    rng = np.random.default_rng(3000 + trial)
    n = int(rng.integers(14, 45))
    min_size = int(rng.integers(2, 6))
    jump = int(rng.integers(1, 4))
    signal = rng.normal(size=n)
    for k in (1, 2, 3):
        best = best_with_k(signal, n, min_size, jump, k)
        est = rpt.Dynp(model="l2", min_size=min_size, jump=jump).fit(signal)
        try:
            bkps = est.predict(k)
        except Exception:  # noqa: BLE001 - infeasible is a legitimate answer
            assert best is None, "refused a feasible k={} segmentation".format(k)
            continue
        assert best is not None
        ours = unpenalised(signal, bkps)
        assert ours <= best + 1e-9 * max(abs(best), 1.0), (
            "k={} n={} min_size={} jump={}: {} costs {!r}, optimum {!r}".format(
                k, n, min_size, jump, bkps, ours, best
            )
        )


# ------------------------------------------------------------------ Crops


@pytest.mark.parametrize("trial", range(25))
def test_crops_intervals_really_are_optimal(trial):
    """Every regime must win everywhere it claims to win.

    `Crops` advertises the exact penalty interval over which each segmentation
    is optimal. That is a much stronger claim than "PELT returned it", and it is
    only true if PELT is exact — which is why this test and the PELT one belong
    together. Each returned regime is checked at three penalties inside its own
    interval, against exhaustive search at that penalty.
    """
    rng = np.random.default_rng(4000 + trial)
    n = int(rng.integers(16, 40))
    min_size = int(rng.integers(2, 5))
    jump = int(rng.integers(1, 3))
    signal = rng.normal(size=n)
    pen_min, pen_max = 0.05, 8.0

    regimes = rpt.Crops(model="l2", min_size=min_size, jump=jump).fit_predict(
        signal, pen_min, pen_max
    )
    assert regimes, "Crops returned no segmentations at all"
    for regime in regimes:
        lo, hi = regime.pen_min, regime.pen_max
        assert lo <= hi
        for frac in (0.25, 0.5, 0.75):
            pen = lo + frac * (hi - lo)
            ours = penalised(signal, list(regime.bkps), pen)
            best = best_penalised(signal, n, min_size, jump, pen)
            assert ours <= best + 1e-9 * max(abs(best), 1.0), (
                "n={} min_size={} jump={}: regime {} claims [{:g}, {:g}] but at "
                "pen={:g} it costs {!r} against an optimum of {!r}".format(
                    n, min_size, jump, list(regime.bkps), lo, hi, pen, ours, best
                )
            )


def test_crops_covers_the_whole_range_without_gaps():
    """The intervals must tile `[pen_min, pen_max]`, in order and without holes.

    A missing interval is a segmentation that is optimal somewhere and was never
    reported, which is the one thing `Crops` exists not to do.
    """
    signal = np.random.default_rng(4).normal(size=40)
    pen_min, pen_max = 0.1, 20.0
    regimes = rpt.Crops(model="l2", min_size=3, jump=1).fit_predict(
        signal, pen_min, pen_max
    )
    # Reported high-penalty (fewest breakpoints) first.
    assert regimes[0].pen_max == pytest.approx(pen_max)
    assert regimes[-1].pen_min == pytest.approx(pen_min)
    for above, below in zip(regimes, regimes[1:]):
        assert above.pen_min == pytest.approx(below.pen_max), "gap in the penalty path"
        assert len(below.bkps) > len(above.bkps), "segment counts must increase"


def test_crops_interval_beats_the_named_competitor():
    """The counterexample from the review, kept by name.

    At penalty 0.1613 the reported regime used to be beaten by an ordinary
    feasible alternative — a direct contradiction of the interval optimality the
    feature advertises, inherited from PELT.
    """
    signal = np.random.default_rng(4).normal(size=24)
    pen = 0.161256173012789
    regimes = rpt.Crops(model="l2", min_size=3, jump=1).fit_predict(signal, 0.1, 5.0)
    regime = next(r for r in regimes if r.pen_min <= pen <= r.pen_max)
    assert penalised(signal, list(regime.bkps), pen) <= penalised(
        signal, [4, 9, 16, 21, 24], pen
    ) + 1e-12


# ------------------------------------------------- the two paths must agree


@pytest.mark.parametrize("trial", range(15))
def test_the_fallback_and_the_engine_reach_the_same_answer(trial):
    """`accelerated` describes how an answer was reached, not which answer.

    A `custom_cost` runs the pure-Python search; a built-in runs the compiled
    one. Given the same cost function they must agree, or `accelerated` is a
    correctness flag pretending to be a performance one.
    """

    class PlainL2(rpt.base.BaseCost):
        """An L2 cost with no engine behind it, so the fallback has to run."""

        model = "plain_l2"
        min_size = 1

        def fit(self, signal):
            self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
            return self

        def error(self, start, end):
            return segment_cost(self.signal, start, end)

    rng = np.random.default_rng(5000 + trial)
    n = int(rng.integers(30, 90))
    min_size = int(rng.integers(2, 8))
    jump = int(rng.integers(1, 4))
    pen = float(rng.choice([0.5, 1.0, 3.0, 10.0]))
    signal = rng.normal(size=n)

    slow = rpt.Pelt(custom_cost=PlainL2(), min_size=min_size, jump=jump)
    fast = rpt.Pelt(model="l2", min_size=min_size, jump=jump)
    assert slow.accelerated is False
    assert fast.accelerated is True
    assert slow.fit_predict(signal, pen) == fast.fit_predict(signal, pen)
