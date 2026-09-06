"""Segmentation metrics, mirroring `ruptures.metrics`.

The validation here is `ruptures`' own, not a stricter version of it: a
partition is rejected only when it is empty, when the two partitions end at
different samples, or when one repeats an index. In particular the reference
does not require sorted or strictly positive breakpoints, and a drop-in that
did would reject inputs its users already have.

`assert` is deliberately absent for the same reason `ruptures` should not use
it: `python -O` strips assertions, and a validation that disappears under a
common flag is not validation.
"""

from itertools import product

import numpy as np


class BadPartitions(Exception):
    """Raised when two segmentations cannot be compared."""


def sanity_check(bkps1, bkps2):
    """Reject partitions that cannot be compared, as `ruptures` does."""
    for nom, bkps in zip(("first", "second"), (bkps1, bkps2)):
        if len(bkps) == 0:
            raise BadPartitions("The {} partition is empty.".format(nom))
    if max(bkps1) != max(bkps2):
        raise BadPartitions(
            "The end of the last regime is not the same for each of the "
            "partitions:\n{}\n{}".format(bkps1, bkps2)
        )
    for bkps in (bkps1, bkps2):
        if len(set(bkps)) != len(bkps):
            raise BadPartitions("Some indexes are repeated: {}".format(bkps))


# Bound at definition time so these keep working even after `install()` shadows
# the module attribute with the `ruptures.metrics.sanity_check` submodule.
_check = sanity_check


def _pairwise_distances(bkps1, bkps2):
    a = np.asarray(bkps1[:-1], dtype=float).reshape(-1, 1)
    b = np.asarray(bkps2[:-1], dtype=float).reshape(-1, 1)
    return np.abs(a - b.T)


def precision_recall(true_bkps, my_bkps, margin=10):
    """Precision and recall of detected changes, within `margin` samples."""
    _check(true_bkps, my_bkps)
    if margin <= 0:
        raise ValueError(
            "Margin of error must be positive (margin = {})".format(margin)
        )
    if len(my_bkps) == 1:
        return 0, 0
    used = set()
    true_pos = set(
        true_b
        for true_b, my_b in product(true_bkps[:-1], my_bkps[:-1])
        if my_b - margin < true_b < my_b + margin
        and not (my_b in used or used.add(my_b))
    )
    tp_ = len(true_pos)
    return tp_ / (len(my_bkps) - 1), tp_ / (len(true_bkps) - 1)


def hausdorff(bkps1, bkps2):
    """Hausdorff distance between two sets of change points."""
    _check(bkps1, bkps2)
    pw = _pairwise_distances(bkps1, bkps2)
    return max(pw.min(axis=0).max(), pw.min(axis=1).max())


def meantime(true_bkps, my_bkps):
    """Mean distance from each detected change to the nearest true change."""
    _check(true_bkps, my_bkps)
    pw = _pairwise_distances(true_bkps, my_bkps)
    dist_from_true = pw.min(axis=0)
    if len(dist_from_true) != len(my_bkps) - 1:
        raise BadPartitions("Unexpected number of detected changes.")
    return dist_from_true.mean()


def randindex(bkps1, bkps2):
    """Rand index between two segmentations."""
    _check(bkps1, bkps2)
    n_samples = bkps1[-1]
    b1 = [0] + list(bkps1)
    b2 = [0] + list(bkps2)
    disagreement = 0
    beginj = 0
    for i in range(len(bkps1)):
        start1, end1 = b1[i], b1[i + 1]
        for j in range(beginj, len(bkps2)):
            start2, end2 = b2[j], b2[j + 1]
            nij = max(min(end1, end2) - max(start1, start2), 0)
            disagreement += nij * abs(end1 - end2)
            if end1 < end2:
                break
            beginj = j + 1
    disagreement /= n_samples * (n_samples - 1) / 2
    return 1.0 - disagreement


def hamming(bkps1, bkps2):
    """Normalised Hamming distance between two segmentations."""
    return 1 - randindex(bkps1=bkps1, bkps2=bkps2)


__all__ = [
    "precision_recall",
    "hausdorff",
    "meantime",
    "randindex",
    "hamming",
    "sanity_check",
    "BadPartitions",
]
