"""ruptures-rs: fast, drop-in change point detection.

A reimplementation of the `ruptures` API in Rust. Replace

    import ruptures as rpt

with

    import ruptures_rs as rpt

and nothing else changes — same classes, same arguments, same breakpoints.

Beyond parity it adds :class:`Crops`, which recovers *every* segmentation that
is optimal for some penalty in a range, together with the exact penalty
interval each one owns. `ruptures` has no equivalent, and it removes the need
to guess a penalty or sweep a grid.
"""

from . import base, costs, datasets, detection, exceptions, metrics, utils
from ._ruptures_rs import __version__
from .costs import (
    CostAR,
    CostCLinear,
    CostCosine,
    CostL1,
    CostL2,
    CostLinear,
    CostMl,
    CostNormal,
    CostRank,
    CostRbf,
    cost_factory,
)
from .crops import Crops
from .datasets import pw_constant, pw_linear, pw_normal, pw_wavy
from .detection import Binseg, BottomUp, Dynp, KernelCPD, Pelt, Window
from .exceptions import BadSegmentationParameters, NotEnoughPoints
from .metrics import hausdorff, meantime, precision_recall, randindex


def install():
    """Alias this package into :data:`sys.modules` as ``ruptures``.

    For code you cannot edit. After ``ruptures_rs.install()``, an
    ``import ruptures`` anywhere in the process resolves here. Call it before
    the first ``import ruptures``; if the real package is already imported this
    raises rather than leaving a half-patched module graph.
    """
    import sys

    if "ruptures" in sys.modules and sys.modules["ruptures"] is not sys.modules[__name__]:
        raise RuntimeError(
            "`ruptures` is already imported; call install() before importing it"
        )
    for name in (
        "",
        ".base",
        ".costs",
        ".datasets",
        ".detection",
        ".exceptions",
        ".metrics",
        ".utils",
    ):
        sys.modules["ruptures" + name] = sys.modules[__name__ + name]
    return sys.modules[__name__]


__all__ = [
    # detection
    "Dynp",
    "Pelt",
    "Binseg",
    "BottomUp",
    "Window",
    "KernelCPD",
    "Crops",
    # costs
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
    "cost_factory",
    # datasets
    "pw_constant",
    "pw_linear",
    "pw_normal",
    "pw_wavy",
    # metrics
    "precision_recall",
    "hausdorff",
    "meantime",
    "randindex",
    # exceptions
    "NotEnoughPoints",
    "BadSegmentationParameters",
    # submodules
    "base",
    "costs",
    "datasets",
    "detection",
    "exceptions",
    "metrics",
    "utils",
    "install",
    "__version__",
]
