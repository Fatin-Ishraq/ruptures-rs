# Benchmarks

AMD Ryzen 5 5600G (6 cores), Python 3.14, `ruptures` 1.1.9, from the single run
recorded in [`bench/bench.log`](../bench/bench.log). Every row was checked for
identical breakpoints, and each timing is the best of as many runs as fit a
twenty-second budget — which stabilises the fast side, where a single sample
swings by more than a factor of two. The rows where `ruptures` takes minutes
still get one run each, so treat the leading digit as the claim and not the
third.

`python bench/bench.py` reproduces the whole table; `--section` rebuilds one
piece of it.

### Detectors

| workload | `ruptures` | `ruptures-rs` | speedup |
|---|---:|---:|---:|
| `Dynp` l2, n=500, K=5, jump=1 | 4.33 s | 0.0016 s | **2,694x** |
| `Dynp` l2, n=1,000 | 21.00 s | 0.0056 s | **3,725x** |
| `Dynp` l2, n=2,000 | 88.20 s | 0.0181 s | **4,863x** |
| `Dynp` l2, n=4,000 | 221.14 s | 0.0608 s | **3,637x** |
| `Pelt` l2, n=1,000, pen=200, jump=1 | 1.11 s | 0.0015 s | 723x |
| `Pelt` l2, n=5,000 | 32.44 s | 0.0308 s | **1,052x** |
| `Pelt` l2, n=20,000 | 542.08 s | 0.5386 s | **1,007x** |
| `Binseg` l2, n=5,000, K=5, jump=1 | 0.476 s | 0.0002 s | **2,018x** |
| `Window` l2, n=5,000, width=100 | 0.189 s | 0.0013 s | 146x |
| `BottomUp` l2, n=5,000, K=5, jump=1 | 0.034 s | 0.0015 s | 23x |

`BottomUp` gains least, and that is the expected result rather than a
disappointment: most of its work is building the initial tree, which was never
the part dominated by cost evaluations.

### Cost models

`Dynp`, n=1,500, K=4, jump=5:

| model | `ruptures` | `ruptures-rs` | speedup |
|---|---:|---:|---:|
| `normal` | 1.18 s | 0.0005 s | **2,524x** |
| `mahalanobis` | 6.73 s | 0.0027 s | **2,447x** |
| `rank` | 1.10 s | 0.0015 s | 737x |
| `rbf` | 7.05 s | 0.0267 s | 264x |
| `linear` | 1.01 s | 0.0091 s | 112x |
| `ar` | 1.48 s | 0.0397 s | 37x |
| `l1` | 1.33 s | 0.0461 s | 29x |

`l1` is last, as expected: it is the one cost that cannot become O(1). `linear`
and `ar` are near it for a different reason — each segment cost is a small
least-squares solve, and the rank-revealing factorisation that keeps them
agreeing with NumPy costs more than a plain Cholesky would.

### Against C, not against NumPy

`KernelCPD` is the one detector `ruptures` already implements as a C extension,
so it is the honest measure of what the rewrite itself buys once the Python
overhead is gone from both sides:

| workload | `ruptures` (C) | `ruptures-rs` | speedup |
|---|---:|---:|---:|
| `KernelCPD` linear, n=2,000, K=5 | 0.020 s | 0.0077 s | 3x |
| `KernelCPD` linear, n=10,000, K=5 | 0.571 s | 0.1971 s | 3x |

A factor of three, which is roughly what one compiled implementation should
beat another by. Everywhere else in this table the reference is paying NumPy
dispatch, and that is where the thousands come from.

### Sizes the reference cannot reach

| workload | `ruptures` | `ruptures-rs` |
|---|---:|---:|
| `Dynp` l2, n=20,000, K=5, jump=10 | infeasible | 0.010 s |
| `Dynp` l2, n=50,000, K=5, jump=10 | infeasible | 0.064 s |

Exact dynamic programming on a 50,000-point signal is not a workload `ruptures`
can run — extrapolating its own curve puts it in the range of days, and its
`lru_cache` of partition dicts was already holding 3.3 GB at n=4,000. This is
the part that is a new capability rather than a faster one.

### CROPS

| | time | segmentations found |
|---|---:|---:|
| `Crops`, n=2,000, penalties [1, 10000] | 0.017 s | 79 |
| 50-point `ruptures` PELT grid, same range | 10.42 s | 22 |

632x faster, and it finds the 59 regimes the grid steps over.