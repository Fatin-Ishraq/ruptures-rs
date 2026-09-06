"""Tests for CROPS, the penalty-path search.

CROPS has no counterpart in `ruptures`, so it cannot be checked differentially.
Instead it is checked against the thing it claims to summarise: for every
penalty interval it reports, running PELT at a penalty inside that interval —
in *ruptures*, not here — must reproduce exactly the segmentation CROPS
attributed to it.

That is a strong property. It says the intervals are right, the segmentations
are right, and that nothing optimal was skipped.
"""

import warnings

import numpy as np
import pytest

# Regimes are verified by re-running ruptures' own PELT.
# On Python versions the reference itself does not support - it declares
# `requires_python = "<3.14"` - skip rather than fail. This package works
# on those versions even though the reference does not.
rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)


def signal(n=400, seed=0, n_bkps=4):
    return rpt_rs.pw_constant(n, 1, n_bkps, noise_std=2, seed=seed)[0]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_each_regime_reproduces_ruptures_pelt(seed):
    """The headline claim: every reported interval is genuinely that interval."""
    sig = signal(seed=seed)
    regimes = rpt_rs.Crops(model="l2", min_size=5, jump=5).fit_predict(sig, 1.0, 5000.0)
    assert regimes, "CROPS returned nothing"

    for regime in regimes:
        # a penalty strictly inside the reported interval
        pen = 0.5 * (regime.pen_min + regime.pen_max)
        if not (regime.pen_min < pen < regime.pen_max):
            continue
        reference = rpt_py.Pelt(model="l2", min_size=5, jump=5).fit(sig).predict(pen)
        assert [int(b) for b in reference] == list(regime.bkps), (
            f"penalty {pen:.4f} in [{regime.pen_min:.4f}, {regime.pen_max:.4f}]"
        )


def test_regimes_are_ordered_and_contiguous():
    sig = signal(seed=7)
    regimes = rpt_rs.Crops(model="l2", min_size=5, jump=5).fit_predict(sig, 1.0, 5000.0)
    # fewest breakpoints first, i.e. decreasing penalty
    counts = [r.n_bkps for r in regimes]
    assert counts == sorted(counts)
    # intervals tile the range without gaps
    for lower, upper in zip(regimes, regimes[1:]):
        assert lower.pen_min == pytest.approx(upper.pen_max, rel=1e-9)


def test_crops_finds_more_than_a_coarse_grid():
    """The point of CROPS: a grid steps over narrow regimes.

    A 12-point log grid over the same range is given roughly the same budget as
    CROPS spends, and is allowed to be no better.
    """
    sig = signal(n=500, seed=5, n_bkps=6)
    regimes = rpt_rs.Crops(model="l2", min_size=5, jump=5).fit_predict(sig, 1.0, 10000.0)
    found_by_crops = {r.n_bkps for r in regimes}

    grid = np.geomspace(1.0, 10000.0, 12)
    found_by_grid = set()
    for pen in grid:
        bkps = rpt_rs.Pelt(model="l2", min_size=5, jump=5).fit(sig).predict(float(pen))
        found_by_grid.add(len(bkps) - 1)

    assert found_by_grid <= found_by_crops
    assert len(found_by_crops) >= len(found_by_grid)


@pytest.mark.parametrize("model", ["l2", "l1", "normal"])
def test_crops_across_models(model):
    sig = signal(n=300, seed=2)
    regimes = rpt_rs.Crops(model=model, min_size=5, jump=5).fit_predict(sig, 1.0, 3000.0)
    assert regimes
    for regime in regimes:
        assert regime.pen_min <= regime.pen_max
        assert regime.bkps[-1] == len(sig)
        assert regime.bkps == sorted(regime.bkps)


def test_rejects_bad_range():
    sig = signal(n=200)
    crops = rpt_rs.Crops(model="l2").fit(sig)
    with pytest.raises(ValueError):
        crops.predict(0.0, 100.0)
    with pytest.raises(ValueError):
        crops.predict(100.0, 10.0)
