"""CROPS — the whole penalty path in one call.

Penalised change point detection has an awkward practical problem: the penalty
is a free parameter, and nobody knows theirs in advance. The usual workaround is
to run PELT over a grid, which wastes work (many penalties give the same answer)
and still misses segmentations that are optimal only on a narrow interval.

CROPS (Haynes, Eckley & Fearnhead, 2017) returns *every* segmentation that is
optimal for some penalty in ``[pen_min, pen_max]``, plus the exact interval each
one owns, in a number of PELT runs proportional to the number of distinct
segmentations rather than the size of a grid.

`ruptures` has no equivalent. It is practical here because PELT itself became
cheap: running it a dozen times is only worth doing once each run is fast.

    >>> import ruptures_rs as rpt
    >>> signal, _ = rpt.pw_constant(500, 1, 4, noise_std=2, seed=0)
    >>> for lo, hi, bkps in rpt.Crops(model="l2").fit_predict(signal, 1, 500):
    ...     print(f"{len(bkps) - 1} changes for penalty in [{lo:.1f}, {hi:.1f}]")
"""

from typing import List, NamedTuple, Tuple

import numpy as np

from . import _ruptures_rs
from .costs import cost_factory
from .exceptions import BadSegmentationParameters
from .utils import as_index, sanity_check


class PenaltyRegime(NamedTuple):
    """One segmentation and the penalty interval over which it is optimal."""

    pen_min: float
    pen_max: float
    bkps: List[int]

    @property
    def n_bkps(self) -> int:
        return len(self.bkps) - 1


class Crops:
    """Segmentations across a range of penalties.

    Parameters mirror the other estimators, so an existing `Pelt` call site can
    be switched over by changing the class and passing a penalty *range* instead
    of a single value.
    """

    def __init__(self, model="l2", min_size=2, jump=5, params=None):
        self.model_name = model
        self.cost = (
            cost_factory(model=model) if params is None else cost_factory(model=model, **params)
        )
        self.min_size = max(as_index(min_size, "min_size"), self.cost.min_size)
        self.jump = as_index(jump, "jump")
        if self.jump < 1:
            raise ValueError("jump must be at least 1, got {}".format(self.jump))
        self.n_samples = None

    def fit(self, signal) -> "Crops":
        signal = np.asarray(signal, dtype=float)
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def predict(self, pen_min: float, pen_max: float) -> List[PenaltyRegime]:
        """Every optimal segmentation for a penalty in ``[pen_min, pen_max]``.

        Returned fewest-breakpoints first (i.e. by decreasing penalty).
        """
        if not pen_min > 0:
            raise ValueError("pen_min must be positive")
        if not pen_max > pen_min:
            raise ValueError("pen_max must exceed pen_min")
        # Same guard the penalised detectors apply. Without it a signal shorter
        # than `min_size` came back as a single regime covering the whole
        # penalty range, which reads as a real answer rather than as "these
        # parameters admit no segmentation".
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        raw: List[Tuple[float, float, List[int]]] = _ruptures_rs.crops(
            self.cost._engine, float(pen_min), float(pen_max), self.jump, self.min_size
        )
        return [PenaltyRegime(lo, hi, list(bkps)) for lo, hi, bkps in raw]

    def fit_predict(self, signal, pen_min: float, pen_max: float) -> List[PenaltyRegime]:
        self.fit(signal)
        return self.predict(pen_min, pen_max)


__all__ = ["Crops", "PenaltyRegime"]
