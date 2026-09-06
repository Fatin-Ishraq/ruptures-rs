"""Segmentation metrics, mirroring `ruptures.metrics`."""

from itertools import product

import numpy as np


def _sanity_check(bkps1, bkps2):
    assert bkps1[-1] == bkps2[-1], "The last index must be identical."
    for bkps in (bkps1, bkps2):
        assert all(b > 0 for b in bkps), "Breakpoints must be positive."
        assert list(bkps) == sorted(bkps), "Breakpoints must be sorted."
        assert len(set(bkps)) == len(bkps), "Breakpoints must be unique."


def precision_recall(true_bkps, my_bkps, margin=10):
    """Precision and recall of detected changes, within `margin` samples."""
    _sanity_check(true_bkps, my_bkps)
    assert margin > 0, "Margin of error must be positive (margin = {})".format(margin)
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
    _sanity_check(bkps1, bkps2)
    a = np.array(bkps1[:-1]).reshape(-1, 1)
    b = np.array(bkps2[:-1]).reshape(-1, 1)
    pw = np.abs(a - b.T)
    return max(pw.min(axis=0).max(), pw.min(axis=1).max())


def meantime(true_bkps, my_bkps):
    """Mean distance from each detected change to the nearest true change."""
    _sanity_check(true_bkps, my_bkps)
    a = np.array(true_bkps[:-1]).reshape(-1, 1)
    b = np.array(my_bkps[:-1]).reshape(-1, 1)
    pw = np.abs(a - b.T)
    return pw.min(axis=0).mean()


def randindex(bkps1, bkps2):
    """Rand index between two segmentations."""
    _sanity_check(bkps1, bkps2)
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


__all__ = ["precision_recall", "hausdorff", "meantime", "randindex"]
