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

A failed allocation in Rust aborts the process rather than raising, so every
structure whose size depends on the input is measured against a ceiling first
and refused with an ordinary exception. :func:`set_memory_limit` and
:func:`get_memory_limit` move that ceiling, as does the
``RUPTURES_RS_MEMORY_LIMIT_GIB`` environment variable; lowering it is how a
service that takes signal dimensions from a request declines hostile ones at its
own boundary.
"""

from . import base, costs, crops, datasets, detection, exceptions, metrics, show, utils, version
from ._ruptures_rs import __version__, get_memory_limit, set_memory_limit
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
from .metrics import hamming, hausdorff, meantime, precision_recall, randindex
from .show import display

#: Modules `ruptures` exposes at the top level, and the leaf modules inside
#: each of its packages. `install()` needs both: a dependency may well write
#: ``from ruptures.detection.pelt import Pelt``, and that import fails unless
#: the full dotted path resolves.
_TOP_LEVEL = (
    "",
    ".base",
    ".costs",
    ".datasets",
    ".detection",
    ".exceptions",
    ".metrics",
    ".show",
    ".utils",
    ".version",
)

_LEAVES = {
    "costs": (
        "factory",
        "costl1",
        "costl2",
        "costlinear",
        "costclinear",
        "costrbf",
        "costnormal",
        "costautoregressive",
        "costml",
        "costrank",
        "costcosine",
    ),
    "detection": ("binseg", "bottomup", "dynp", "kernelcpd", "pelt", "window"),
    "metrics": (
        "hausdorff",
        "timeerror",
        "precisionrecall",
        "hamming",
        "randindex",
        "sanity_check",
    ),
    "utils": ("utils", "bnode", "drawbkps"),
    "show": ("display",),
    "datasets": ("pw_constant", "pw_linear", "pw_normal", "pw_wavy"),
}


def install():
    """Alias this package into :data:`sys.modules` as ``ruptures``.

    For code you cannot edit. After ``ruptures_rs.install()``, an
    ``import ruptures`` anywhere in the process resolves here. Call it before
    the first ``import ruptures``; if the real package is already imported this
    raises rather than leaving a half-patched module graph.

    `ruptures` splits its API across one module per class, so the alias covers
    the leaf paths too: ``from ruptures.costs.costl2 import CostL2`` resolves,
    as does ``ruptures.detection.pelt``. What it cannot fake is distribution
    metadata — ``importlib.metadata.version("ruptures")`` still reports that no
    such distribution is installed, because nothing was installed.
    """
    import sys
    import types

    if "ruptures" in sys.modules and sys.modules["ruptures"] is not sys.modules[__name__]:
        raise RuntimeError(
            "`ruptures` is already imported; call install() before importing it"
        )
    for name in _TOP_LEVEL:
        sys.modules["ruptures" + name] = sys.modules[__name__ + name]

    for parent, leaves in _LEAVES.items():
        target = sys.modules[__name__ + "." + parent]
        for leaf in leaves:
            full = "ruptures.{}.{}".format(parent, leaf)
            if full in sys.modules:
                continue
            proxy = types.ModuleType(full)
            proxy.__dict__.update(vars(target))
            proxy.__name__ = full
            proxy.__doc__ = "Compatibility alias for {}.".format(target.__name__)
            sys.modules[full] = proxy
            # Only bind the attribute when the parent has nothing by that name.
            # `show.display` and `metrics.hamming` are functions in `ruptures`
            # too, and replacing them with a module would break every caller.
            if not hasattr(target, leaf):
                setattr(target, leaf, proxy)
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
    "hamming",
    # plotting
    "display",
    # exceptions
    "NotEnoughPoints",
    "BadSegmentationParameters",
    # submodules
    "base",
    "costs",
    "crops",
    "datasets",
    "detection",
    "exceptions",
    "metrics",
    "show",
    "utils",
    "version",
    "install",
    "get_memory_limit",
    "set_memory_limit",
    "__version__",
]
