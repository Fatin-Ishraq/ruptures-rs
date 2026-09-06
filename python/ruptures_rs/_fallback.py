"""Pure-Python algorithms for user-supplied cost functions.

The compiled path owns the whole hot loop, which is what makes it fast — but it
can only do that for costs it implements. When a caller passes ``custom_cost``,
their Python object *is* the inner loop, and crossing the FFI boundary millions
of times to reach it would be slower than staying in Python.

So `custom_cost` runs here instead: same algorithms, same results, original
speed. Coverage stays at 100% of the `ruptures` API rather than "100% of the
parts that were convenient", and the acceleration is simply not claimed where it
cannot be delivered.

These are ports of the corresponding `ruptures` routines (Charles Truong et al.,
BSD-2-Clause), which this package is also licensed under.
"""

import heapq
from bisect import bisect_left
from functools import lru_cache
from math import floor

from .utils import pairwise, sanity_check


def dynp(cost, n_samples, n_bkps, jump, min_size):
    @lru_cache(maxsize=None)
    def seg(start, end, k):
        if k == 0:
            return ((start, end),), cost.error(start, end)
        best = None
        bkp = start
        while bkp < end:
            if bkp % jump == 0:
                if sanity_check(bkp - start, k - 1, jump, min_size) and end - bkp >= min_size:
                    left_keys, left_val = seg(start, bkp, k - 1)
                    total = left_val + cost.error(bkp, end)
                    if best is None or total < best[1]:
                        best = (left_keys + ((bkp, end),), total)
            bkp += 1
        if best is None:
            raise AssertionError("No admissible last breakpoints found.")
        return best

    keys, _ = seg(0, n_samples, n_bkps)
    return sorted(e for _, e in keys)


def pelt(cost, n_samples, pen, jump, min_size):
    partitions = {0: {(0, 0): 0.0}}
    admissible = []
    ind = [k for k in range(0, n_samples, jump) if k >= min_size]
    ind += [n_samples]
    for bkp in ind:
        new_adm_pt = floor((bkp - min_size) / jump) * jump
        admissible.append(new_adm_pt)
        subproblems = []
        for t in admissible:
            try:
                tmp = partitions[t].copy()
            except KeyError:
                continue
            tmp[(t, bkp)] = cost.error(t, bkp) + pen
            subproblems.append(tmp)
        if not subproblems:
            continue
        partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
        best = sum(partitions[bkp].values())
        admissible = [
            t
            for t, part in zip(admissible, subproblems)
            if sum(part.values()) <= best + pen
        ]
    best_partition = partitions[n_samples].copy()
    best_partition.pop((0, 0), None)
    return sorted(e for _, e in best_partition.keys())


def binseg(cost, n_samples, jump, min_size, n_bkps=None, pen=None, epsilon=None):
    @lru_cache(maxsize=None)
    def single_bkp(start, end):
        segment_cost = cost.error(start, end)
        if segment_cost == float("-inf"):
            return None, 0
        gain_list = []
        for bkp in range(start, end, jump):
            if bkp - start >= min_size and end - bkp >= min_size:
                gain = segment_cost - cost.error(start, bkp) - cost.error(bkp, end)
                gain_list.append((gain, bkp))
        if not gain_list:
            return None, 0
        gain, bkp = max(gain_list)
        return bkp, gain

    bkps = [n_samples]
    stop = False
    while not stop:
        stop = True
        new_bkps = [single_bkp(start, end) for start, end in pairwise([0] + bkps)]
        bkp, gain = max(new_bkps, key=lambda x: x[1])
        if bkp is None:
            break
        if n_bkps is not None:
            if len(bkps) - 1 < n_bkps:
                stop = False
        elif pen is not None:
            if gain > pen:
                stop = False
        elif epsilon is not None:
            if cost.sum_of_costs(bkps) > epsilon:
                stop = False
        if not stop:
            bkps.append(bkp)
            bkps.sort()
    return sorted(bkps)


class _Bnode:
    __slots__ = ("start", "end", "val", "left", "right")

    def __init__(self, start, end, val, left=None, right=None):
        self.start, self.end, self.val = start, end, val
        self.left, self.right = left, right

    @property
    def gain(self):
        if self.left is None or self.right is None:
            return 0
        if self.val == float("-inf"):
            return 0
        return self.val - (self.left.val + self.right.val)

    def __lt__(self, other):
        return self.start < other.start

    def __eq__(self, other):
        return (
            isinstance(other, _Bnode)
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        return hash((_Bnode, self.start, self.end))


def bottomup(cost, n_samples, jump, min_size, n_bkps=None, pen=None, epsilon=None):
    partition = [(-n_samples, (0, n_samples))]
    stop = False
    while not stop:
        stop = True
        _, (start, end) = partition[0]
        mid = (start + end) * 0.5
        candidates = [
            bkp
            for bkp in range(start, end)
            if bkp % jump == 0 and bkp - start >= min_size and end - bkp >= min_size
        ]
        if candidates:
            bkp = min(candidates, key=lambda x: abs(x - mid))
            heapq.heappop(partition)
            heapq.heappush(partition, (-bkp + start, (start, bkp)))
            heapq.heappush(partition, (-end + bkp, (bkp, end)))
            stop = False
    partition.sort(key=lambda x: x[1])
    leaves = [_Bnode(s, e, cost.error(s, e)) for _, (s, e) in partition]

    cache = {}

    def merge(left, right):
        key = (left.start, left.end, right.start, right.end)
        if key not in cache:
            cache[key] = cost.error(left.start, right.end)
        return _Bnode(left.start, right.end, cache[key], left=left, right=right)

    leaves = sorted(leaves)
    keys = [leaf.start for leaf in leaves]
    removed = set()
    merged = []
    for left, right in pairwise(leaves):
        cand = merge(left, right)
        heapq.heappush(merged, (cand.gain, cand))

    stop = False
    while not stop:
        stop = True
        try:
            gain, leaf = heapq.heappop(merged)
            while leaf.left in removed or leaf.right in removed:
                gain, leaf = heapq.heappop(merged)
        except IndexError:
            break
        if n_bkps is not None:
            if len(leaves) > n_bkps + 1:
                stop = False
        elif pen is not None:
            if gain < pen:
                stop = False
        elif epsilon is not None:
            if sum(x.val for x in leaves) < epsilon:
                stop = False
        if not stop:
            left_idx = bisect_left(keys, leaf.left.start)
            leaves[left_idx] = leaf
            keys[left_idx] = leaf.start
            del leaves[left_idx + 1]
            del keys[left_idx + 1]
            removed.add(leaf.left)
            removed.add(leaf.right)
            if left_idx > 0:
                cand = merge(leaves[left_idx - 1], leaf)
                heapq.heappush(merged, (cand.gain, cand))
            if left_idx < len(leaves) - 1:
                cand = merge(leaf, leaves[left_idx + 1])
                heapq.heappush(merged, (cand.gain, cand))
    return sorted(leaf.end for leaf in leaves)
