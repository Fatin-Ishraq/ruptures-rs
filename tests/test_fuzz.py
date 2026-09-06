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
