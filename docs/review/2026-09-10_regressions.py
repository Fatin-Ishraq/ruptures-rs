"""Independent review probes; intentionally fail on reviewed version 0.1.1.

Run against a wheel built from the checkout:
    python -m pytest docs/review/2026-09-10_regressions.py -q

These assert desired behavior, not preservation of current bugs. No probe
attempts a large allocation. Malformed native calls use tiny arrays.
"""

import numpy as np
import pytest
import ruptures_rs as rpt


@pytest.mark.parametrize("model,amplitude,expected", [("l2", 1e10, 82.5), ("l1", 1e18, 25.0)])
def test_local_detail_survives_unrelated_large_regime(model, amplitude, expected):
    signal = np.r_[np.full(50, amplitude), np.arange(50.0)]
    assert rpt.cost_factory(model).fit(signal).error(50, 60) == pytest.approx(expected)


def l2_objective(signal, bkps, penalty):
    """Independent two-pass segment variance; no compiled cost reuse."""
    start, total = 0, 0.0
    for end in bkps:
        segment = signal[start:end]
        total += float(np.sum((segment - segment.mean()) ** 2)) + penalty
        start = end
    return total


def test_pelt_does_not_lose_to_unsplit_signal():
    signal = np.random.default_rng(86).normal(size=24)
    actual = rpt.Pelt(jump=1, min_size=6).fit_predict(signal, pen=2.0)
    assert l2_objective(signal, actual, 2.0) <= l2_objective(signal, [24], 2.0) + 1e-10


def test_crops_interval_does_not_lose_to_feasible_competitor():
    signal = np.random.default_rng(4).normal(size=24)
    penalty = 0.161256173012789
    regimes = rpt.Crops(jump=1, min_size=3).fit_predict(signal, 0.1, 5.0)
    regime = next(r for r in regimes if r.pen_min < penalty < r.pen_max)
    alternative = [4, 9, 16, 21, 24]
    assert l2_objective(signal, regime.bkps, penalty) <= l2_objective(signal, alternative, penalty) + 1e-10


def test_custom_builtin_subclass_error_is_honored():
    class ZeroCost(rpt.CostL2):
        def error(self, start, end):
            return 0.0

    signal = np.r_[np.zeros(20), np.full(20, 10.0)]
    estimator = rpt.Pelt(custom_cost=ZeroCost(), jump=1).fit(signal)
    assert estimator.predict(1.0) == [40]


def test_public_window_large_jump_raises_ordinary_exception_or_returns():
    try:
        rpt.Window(width=4, jump=2**63).fit(np.arange(10.0)).predict(0)
    except Exception:
        pass  # PanicException inherits BaseException and intentionally escapes.


def test_native_window_checks_array_lengths():
    engine = rpt.CostL2().fit(np.arange(10.0))._engine
    with pytest.raises(Exception):
        rpt._ruptures_rs.window_predict(engine, [], [0.0, 1.0, 0.0], 2, 1, 1, 1)


def test_native_linear_rejects_zero_features():
    with pytest.raises(Exception):
        rpt._ruptures_rs.CostEngine(np.empty((10, 0)), "linear")


def test_multivariate_normal_does_not_reward_nan_as_negative_infinity():
    signal = np.arange(20.0).reshape(10, 2)
    signal[2, 0] = np.nan
    try:
        value = rpt.CostNormal().fit(signal).error(0, 5)
    except ValueError:
        return  # Explicit refusal is also acceptable.
    assert np.isnan(value)


@pytest.mark.parametrize("scale", [1e80, 1e-100])
def test_linear_predictor_scaling_preserves_residual(scale):
    rng = np.random.default_rng(42)
    design = rng.normal(size=(60, 2))
    response = 2 * design[:, 0] - design[:, 1] + rng.normal(size=60) * 0.1
    _, residual, _, _ = np.linalg.lstsq(design * scale, response, rcond=None)
    value = rpt.CostLinear().fit(np.c_[response, design * scale]).error(0, 60)
    assert value == pytest.approx(float(residual.sum()), rel=1e-9)


def test_failed_refit_preserves_old_state_or_invalidates_engine():
    cost = rpt.CostL2().fit(np.arange(20.0))
    old_signal = cost.signal
    with pytest.raises(ValueError):
        cost.fit(np.zeros((2, 2, 2)))
    assert cost._engine is None or cost.signal is old_signal


@pytest.mark.parametrize("name,attribute", [("CostRbf", "gram"), ("CostRank", "ranks")])
def test_reference_fitted_attributes_exist(name, attribute):
    cost = getattr(rpt, name)().fit(np.random.default_rng(1).normal(size=(20, 2)))
    assert hasattr(cost, attribute)


def test_fractional_breakpoint_count_is_rejected():
    estimator = rpt.Binseg().fit(np.arange(100.0))
    with pytest.raises((TypeError, ValueError)):
        estimator.predict(n_bkps=1.9)
