"""The compatibility matrix, as a test rather than as a claim.

"Same API" is easy to say and easy to be wrong about. The old check for it
compared `dir(ruptures)` against `dir(ruptures_rs)` — top-level names only,
which is the part nobody gets wrong. It said nothing about the objects those
names produce, and so missed that a fitted `CostRbf` here had no `.gram`, a
fitted `CostRank` no `.ranks` or `.inv_cov`, and the estimators none of the
public internals (`Dynp.seg`, `Binseg.single_bkp`, `BottomUp.leaves`) that
subclasses in the wild override.

What is checked here:

* every public attribute of a *fitted* reference object exists here too;
* the constructor signatures match parameter for parameter;
* the members that are meant to carry a value carry the reference's value;
* the deliberate exclusions are listed, with a reason, instead of being
  discovered later by somebody's traceback.

Where a value cannot match exactly it is compared with a tolerance, because
these are floating-point quantities computed by two different routes. Where it
is not compared at all, the exclusion is named below.
"""

import inspect
import warnings

import numpy as np
import pytest

rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#: Members that exist in `ruptures` and are deliberately absent or different
#: here, with the reason. Anything not on this list has to match.
DOCUMENTED_EXCLUSIONS = {
    # `ruptures` decorates these with `functools.lru_cache`, which publishes
    # `cache_clear` and `cache_info` on the bound method. Here they are ordinary
    # methods backed by a per-instance dict that `fit` resets, so the cache is
    # cleared by refitting rather than by calling into the decorator.
    ("Binseg", "single_bkp.cache_clear"),
    ("BottomUp", "merge.cache_clear"),
    ("Dynp", "seg.cache_clear"),
}


def signals():
    rng = np.random.default_rng(11)
    plain = rng.normal(size=(80, 2))
    return {
        "plain": plain,
        "univariate": plain[:, 0].copy(),
        "linear": np.c_[rng.normal(size=80), plain],
    }


COST_INPUT = {
    "CostLinear": "linear",
    "CostAR": "univariate",
}

COSTS = [
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
]

DETECTORS = ["Dynp", "Pelt", "Binseg", "BottomUp", "Window", "KernelCPD"]


def public(obj):
    return {name for name in dir(obj) if not name.startswith("_")}


def fitted_pair(kind, name):
    data = signals()[COST_INPUT.get(name, "plain")]
    if kind == "cost":
        theirs = getattr(rpt_py.costs, name)().fit(data.copy())
        ours = getattr(rpt_rs, name)().fit(data.copy())
    else:
        theirs = getattr(rpt_py, name)().fit(data.copy())
        ours = getattr(rpt_rs, name)().fit(data.copy())
    return theirs, ours


@pytest.mark.parametrize("name", COSTS)
def test_fitted_costs_expose_everything_the_reference_does(name):
    theirs, ours = fitted_pair("cost", name)
    missing = sorted(public(theirs) - public(ours))
    assert not missing, "{} is missing {}".format(name, missing)


@pytest.mark.parametrize("name", DETECTORS)
def test_fitted_detectors_expose_everything_the_reference_does(name):
    theirs, ours = fitted_pair("detector", name)
    missing = sorted(public(theirs) - public(ours))
    assert not missing, "{} is missing {}".format(name, missing)


@pytest.mark.parametrize("name", COSTS + DETECTORS)
def test_constructor_signatures_match(name):
    """A drop-in that renames a keyword is not a drop-in."""
    theirs = getattr(rpt_py, name, None) or getattr(rpt_py.costs, name)
    ours = getattr(rpt_rs, name)
    want = inspect.signature(theirs.__init__).parameters
    got = inspect.signature(ours.__init__).parameters
    assert list(want) == list(got), "{}: {} vs {}".format(name, list(want), list(got))
    for pname, param in want.items():
        if param.default is inspect.Parameter.empty:
            continue
        assert got[pname].default == param.default, "{}.{}".format(name, pname)


def test_kernel_gram_matrices_match_the_reference():
    data = signals()["plain"]
    for name, rtol in (("CostRbf", 1e-11), ("CostCosine", 1e-11), ("CostMl", 1e-11)):
        theirs = getattr(rpt_py.costs, name)().fit(data.copy()).gram
        ours = getattr(rpt_rs, name)().fit(data.copy()).gram
        assert np.allclose(theirs, ours, rtol=rtol, atol=1e-13), name


def test_rank_state_matches_the_reference():
    data = signals()["plain"]
    theirs = rpt_py.costs.CostRank().fit(data.copy())
    ours = rpt_rs.CostRank().fit(data.copy())
    assert np.allclose(theirs.ranks, ours.ranks)
    assert np.allclose(theirs.inv_cov, ours.inv_cov)


def test_gram_is_not_built_until_it_is_asked_for():
    """The whole point of the engine is that it never needs this matrix.

    Building it during `fit` would make every kernel fit pay n^2 doubles for an
    attribute that only exists for compatibility.
    """
    data = signals()["plain"]
    cost = rpt_rs.CostRbf().fit(data)
    assert cost._gram_cache is None
    assert cost.gram.shape == (80, 80)
    assert cost._gram_cache is not None
    # And refitting must not serve the old signal's matrix.
    cost.fit(data[:40])
    assert cost._gram_cache is None
    assert cost.gram.shape == (40, 40)


def test_estimator_internals_agree_with_the_reference():
    data = signals()["plain"]
    est = dict(model="l2", jump=5, min_size=3)

    theirs = rpt_py.Dynp(**est).fit(data)
    ours = rpt_rs.Dynp(**est).fit(data)
    left, right = theirs.seg(0, 80, 2), ours.seg(0, 80, 2)
    assert sorted(left) == sorted(right)
    for key in left:
        assert left[key] == pytest.approx(right[key])

    theirs = rpt_py.Binseg(**est).fit(data)
    ours = rpt_rs.Binseg(**est).fit(data)
    assert theirs.single_bkp(0, 80)[0] == ours.single_bkp(0, 80)[0]
    assert theirs.single_bkp(0, 80)[1] == pytest.approx(ours.single_bkp(0, 80)[1])

    theirs = rpt_py.BottomUp(**est).fit(data)
    ours = rpt_rs.BottomUp(**est).fit(data)
    assert [(leaf.start, leaf.end) for leaf in theirs.leaves] == [
        (leaf.start, leaf.end) for leaf in ours.leaves
    ]
    for a, b in zip(theirs.leaves, ours.leaves):
        assert a.val == pytest.approx(b.val)
    merged_theirs = theirs.merge(theirs.leaves[0], theirs.leaves[1])
    merged_ours = ours.merge(ours.leaves[0], ours.leaves[1])
    assert merged_theirs.val == pytest.approx(merged_ours.val)
    assert merged_theirs.gain == pytest.approx(merged_ours.gain)


def test_the_exclusions_are_the_only_exclusions():
    """The list above must stay honest: nothing on it that is actually present.

    A stale exclusion is worse than none, because it excuses a gap that has
    since been filled and hides one that appears later in the same place.
    """
    for name, member in DOCUMENTED_EXCLUSIONS:
        attr, _, sub = member.partition(".")
        _, ours = fitted_pair("detector", name)
        target = getattr(ours, attr, None)
        assert target is not None, "{}.{} is gone entirely".format(name, attr)
        assert not hasattr(target, sub), "{}.{} exists; drop the exclusion".format(
            name, member
        )
