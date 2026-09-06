"""Plotting, mirroring `ruptures.show`.

Ported so that `rpt.display(signal, true_bkps, my_bkps)` keeps working after the
import swap. matplotlib stays optional, exactly as in `ruptures`.
"""

from itertools import cycle

import numpy as np

from .utils import pairwise

COLOR_CYCLE = ["#4286f4", "#f44174"]


class MatplotlibMissingError(RuntimeError):
    """Raised when `display` is called without matplotlib installed."""


def display(
    signal,
    true_chg_pts,
    computed_chg_pts=None,
    computed_chg_pts_color="k",
    computed_chg_pts_linewidth=3,
    computed_chg_pts_linestyle="--",
    computed_chg_pts_alpha=1.0,
    **kwargs,
):
    """Plot a signal with its true regimes shaded and detected changes marked."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MatplotlibMissingError(
            "This feature requires the optional dependency matpotlib, you can install "
            "it using `pip install matplotlib`."
        ) from None

    if not isinstance(signal, np.ndarray):
        # e.g. a pandas DataFrame
        signal = signal.values
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    matplotlib_options = {"figsize": (10, 2 * n_features)}
    matplotlib_options.update(kwargs)

    fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
    if n_features == 1:
        axarr = [axarr]

    for axe, sig in zip(axarr, signal.T):
        color_cycle = cycle(COLOR_CYCLE)
        axe.plot(range(n_samples), sig)
        bkps = [0] + sorted(true_chg_pts)
        for (start, end), col in zip(pairwise(bkps), color_cycle):
            axe.axvspan(max(0, start - 0.5), end - 0.5, facecolor=col, alpha=0.2)
        if computed_chg_pts is not None:
            for bkp in computed_chg_pts:
                if bkp != 0 and bkp < n_samples:
                    axe.axvline(
                        x=bkp - 0.5,
                        color=computed_chg_pts_color,
                        linewidth=computed_chg_pts_linewidth,
                        linestyle=computed_chg_pts_linestyle,
                        alpha=computed_chg_pts_alpha,
                    )
    fig.tight_layout()
    return fig, axarr


__all__ = ["display", "MatplotlibMissingError", "COLOR_CYCLE"]
