# ruptures-rs

[![CI](https://github.com/Fatin-Ishraq/ruptures-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatin-Ishraq/ruptures-rs/actions/workflows/ci.yml)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg)](https://pypi.org/project/ruptures-rs/)

Fast, drop-in change point detection for Python, powered by Rust.

```diff
- import ruptures as rpt
+ import ruptures_rs as rpt
```

That is the whole migration. Same classes, same arguments, same breakpoints —
verified by 1,391 tests that run both libraries on the same input and demand
*identical* output, not merely similar output.

```bash
pip install ruptures-rs
```

Supports **Python 3.10 through 3.14** from a single `abi3` wheel per platform,
including musl. That matters more than it sounds: `ruptures` 1.1.9 ships no
cp314 wheel, so on Python 3.14 `pip` compiles it from source — which needs a C
toolchain, and fails without one. This installs as a prebuilt wheel.

## Why it is faster

`ruptures` is a well-designed library with one structural performance problem:
its cost functions are O(len) NumPy calls, and its dynamic program evaluates
them millions of times. `CostL2.error` is

```python
return self.signal[start:end].var(axis=0).sum() * (end - start)
```

which is a NumPy call whose fixed dispatch overhead dwarfs its own arithmetic,
and the dynamic program makes one at every cell of an O(n²K) table. On a
4,000-point signal that is several minutes of wall clock, essentially all of it
inside those calls.

Nearly every one of these costs is a difference of prefix sums in disguise. With
prefix sums of `x` and `x²`, the same value is a handful of scalar operations,
independent of segment length:

```
cost(a, b) = Σ_d [ S2[b] − S2[a] − (S1[b] − S1[a])² / (b − a) ]
```

| model | `ruptures` | here |
|---|---|---|
| `l2` | O(len·d) | **O(d)** |
| `normal` | O(len·d²) | **O(d³)** |
| `mahalanobis` | O(len²) | **O(d²)** |
| `rank` | O(len·d) | **O(d²)** |
| `clinear` | O(len·d) | **O(d)** |
| `linear`, `ar` | O(len·d²) | **O(d³)** |
| `rbf`, `cosine` | O(len²) | **O(1)** |
| `l1` | O(len·d) | O(len·d) |

So the win is not only the constant factor from leaving Python. `Dynp` in
`ruptures` is O(n²K) cells × O(len) per cell; here it is O(n²K) cells × O(1).

In practice the measured speedup climbs with signal size into the thousands and
then flattens out. Below a few thousand samples NumPy's fixed per-call overhead
is what dominates the reference, and that part is a constant factor; the O(len)
term only starts to bite once segments get long.

`l1` is the honest exception: a per-segment median is not a prefix-summable
statistic, so it stays O(len) and gains only the constant factor.

## Benchmarks

AMD Ryzen 5 5600G (6 cores), Python 3.14, `ruptures` 1.1.9, from the single run
recorded in `bench/bench.log`. Every row was checked for identical breakpoints,
and each timing is the best of as many runs as fit a twenty-second budget —
which stabilises the fast side, where a single sample swings by more than a
factor of two. The rows where `ruptures` takes minutes still get one run each,
so treat the leading digit as the claim and not the third.

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

## What `Crops` adds

Penalised detection asks for a penalty, and nobody knows theirs in advance.
`Crops` implements CROPS (Haynes, Eckley & Fearnhead, 2017) and returns *every*
segmentation that is optimal somewhere in a penalty range, with the exact
interval each one owns. `ruptures` has no equivalent.

```python
import ruptures_rs as rpt

signal, _ = rpt.pw_constant(2_000, 1, 6, noise_std=3, seed=11)

for regime in rpt.Crops(model="l2", min_size=10, jump=5).fit_predict(signal, 1, 20_000):
    print(f"{regime.n_bkps:2d} changes for penalty in "
          f"[{regime.pen_min:,.0f}, {regime.pen_max:,.0f}]")
```

A segmentation that stays optimal across a wide span of penalties is the
defensible choice, because it is the one least sensitive to the parameter you
could not justify picking — ignoring the two regimes at the ends of the range,
whose width is an artefact of where you cut the range rather than evidence of
stability. `examples/penalty_path.py` works through this.

This is practical here only because PELT became cheap: running it a dozen times
is worth doing once each run is fast.

## Accuracy

Prefix sums are what buys the speed, and the naive form is also a classic
numerical trap: `Σx² − (Σx)²/n` is the textbook one-pass variance, which cancels
catastrophically when the mean dwarfs the spread. NumPy's `.var()` is two-pass
and therefore safe, so a careless port silently disagrees with the reference —
or returns negative variances.

Four defences, applied per cost according to what that cost is invariant under:

- **Centring.** Every cost built on sums of squared deviations is invariant
  under a per-column shift, so the signal is centred before any prefix array is
  built. This covers `l2`, `l1`, `normal`, `mahalanobis` and `rank`.
- **Detrending.** `clinear` compares a signal to a straight line, so it is
  invariant under subtracting *any* affine function of the sample index.
  Centring is not enough there — a trending signal is exactly what that cost is
  for — so it is detrended by its global least-squares line.
- **Pre-fitting.** `linear` and `ar` are invariant under removing any fixed
  linear fit from the response, so the whole-signal fit is removed first. That
  turns the final `yᵀy − β·Xᵀy` from a cancellation of two large numbers into
  arithmetic on the residual itself.
- **Compensated accumulation.** Prefix arrays use Neumaier summation, bounding
  accumulation error at roughly one ulp regardless of `n`.

The result is that this package is *more* accurate than the reference on
ill-conditioned input for the costs above. On a two-column signal offset by 1e9,
checked against 60-digit exact arithmetic:

| | `CostMl.error(23, 88)` |
|---|---:|
| exact | 108.92015511 |
| `ruptures` | 0.00000000 |
| `ruptures-rs` | 108.92015511 |

`ruptures` builds this cost from a Gram matrix of raw values, and at that offset
the subtraction that follows removes every significant digit it had.

`tests/test_precision.py` asserts all of this against `decimal.Decimal`, so a
regression in either direction is caught. It also pins the two costs that used
to be worst here: `clinear` on a 20,000-point ramp went from 6e-3 relative error
to 1e-11, and `ar` at an offset of 1e5 from 1.7e-2 to 2.8e-10.

## Correctness

The whole project is only worth anything if the answers match, so that is what
the suite tests.

- **1,391 tests**, most of them differential against `ruptures` on the same
  input.
- **800 randomised fuzz cases** over signal length, dimension, `jump`,
  `min_size`, penalty, model, stopping rule and seed, asserting *exact*
  breakpoint equality — including `Window`, `epsilon` stopping, `linear`, `ar`,
  `cosine` and `KernelCPD` in both of its modes.
- **API coverage** is asserted mechanically: every public name in
  `dir(ruptures)` must exist here.
- **CROPS** is validated against `ruptures`' own PELT — for every interval it
  reports, running the reference at a penalty inside that interval must
  reproduce exactly the segmentation CROPS attributed to it.
- **Crash safety.** A Rust panic reaches Python as `PanicException`, which
  inherits from `BaseException` and so slips past `except Exception`. Malformed
  input used to trigger several. `tests/test_robustness.py` fires NaN, infinity,
  out-of-range segments, `jump=0` and impossible allocations at every model and
  detector, and fails if anything but an ordinary exception comes back.

Tie-breaking is reproduced deliberately. `ruptures` relies on Python's `min` and
`max` returning the *first* extremum, so a Rust port using `Iterator::min_by`
(which returns the *last*) would quietly return different — though equally
optimal — breakpoints. Python's `min` also keeps its first element when nothing
compares less than it, which matters once a NaN is in play, so every search here
seeds from the first candidate rather than from infinity. Several `ruptures`
quirks are matched bug-for-bug, including PELT's positional `zip` of
`admissible` against `subproblems`, because a drop-in that is merely defensible
is not a drop-in.

`KernelCPD` needed more than that. It is the one detector `ruptures` implements
in C, and that C does not agree with `ruptures`' own Python cost classes: its
cosine kernel has a unit diagonal where `scipy`'s `squareform` leaves a zero one
— worth exactly one unit of penalty per segment, so asking for penalty `p` was
solving the problem for `p − 1` — and its Gaussian exponent is clipped in single
precision. Its penalised search also folds the penalty in and prunes differently
from the Python `Pelt`. All of that is reproduced separately, so each detector
here agrees with the `ruptures` code path it actually corresponds to.

## Where the answers differ

Stated precisely, because a drop-in that hides its gaps is worse than one that
does not have them. Each of these is covered by a test that asserts the
behaviour rather than hoping for it.

- **Signals with exactly repeated values admit many equally-optimal
  segmentations.** On a noiseless integer step signal, thousands of
  segmentations have costs that differ only in the last few bits, and the two
  libraries compute those bits differently — `ruptures` from a two-pass NumPy
  reduction, this package from a difference of prefix sums. `tests/test_ties.py`
  asserts the guarantee that survives: across every tied case it generates, the
  exact detectors (`Dynp`, `Pelt`) never return a segmentation that costs more
  than the reference's, scored with `ruptures`' own cost function. `Binseg` and
  `BottomUp` are greedy and can land slightly either side. On data with any
  noise in it, agreement is exact.
- **`Window` can return a different number of breakpoints on such signals.**
  Its peaks come from `argrelmax`, which needs a score strictly greater than
  both neighbours; a noiseless signal makes the score curve flat over long
  stretches, and whether a plateau counts as a peak is decided by the last bit
  of a cost. The score *curves* agree to a few ulp — the test asserts that — so
  this is peak-picking on tied data, not a disagreement about the costs.
- **`linear` calls a design rank-deficient sooner than NumPy does.** Solving
  through `XᵀX` squares the condition number, so a deficiency that shows in `X`
  at 1e-17 shows in the normal equations at 1e-34, far below the floor at which
  an eigenvalue means anything. The cutoff is therefore applied where it can be
  resolved, which catches exact collinearity — where NumPy also reports no
  residual — at the cost of also reporting no residual for a design whose
  condition number exceeds roughly 1e7. Measured: the two agree at 1.4e6 and
  this package reports no residual from 1.4e7 up. `ar` is unaffected — its
  intercept column lets the design be centred first.
- **The default Mahalanobis metric is an inverted covariance**, and
  `numpy.linalg.inv` only raises on *exact* singularity. One rounding step short
  of that it returns a matrix whose entries are pure cancellation, and every
  cost built from it is noise — in `ruptures` too, which says nothing about it.
  Every disagreement observed between the two libraries on this cost was on a
  covariance in that state, so this package warns instead of staying quiet. On
  well-conditioned data the two agree exactly across every case tried.
- **`normal` with `add_small_diag=False` is meaningless on segments that are
  constant to machine precision**, in both libraries, and they are meaningless
  in different ways. `ruptures` added the small bias in v1.1.5 for exactly this
  reason, and it is the default.
- **`CostRank.error` returns a float**, where `ruptures` returns a 1x1 NumPy
  array because it never collapses its matrix product. Arithmetic and
  comparisons behave the same either way; code that indexed the result would
  notice.
- **`custom_cost` is not accelerated.** If you pass your own cost object, your
  Python callable *is* the inner loop, and crossing an FFI boundary millions of
  times to reach it would be slower than staying in Python. Those calls run on a
  pure-Python implementation instead: same answers, original speed. Check
  `estimator.accelerated` to see which path you are on.
- **`l1` is only constant-factor faster.** The median is not prefix-summable.
- **`rbf` and `cosine` need O(n²) memory**, the same as the dense Gram matrix
  `ruptures` builds. Past ~23,000 samples this package raises an actionable
  error instead of letting the allocator kill the process. The covariance costs
  (`normal`, `mahalanobis`, `linear`, `ar`) keep `(n+1)·d²` doubles and are
  guarded the same way, which is a limit `ruptures` does not have — it computes
  those in O(n·d²).

Where this package is deliberately stricter than the reference, it is because
the alternative is a wrong answer rather than a different one: a segment index
past the end of the signal raises instead of being silently clamped by NumPy
slicing, `jump=0` is refused instead of dividing by zero three frames down, and
`Dynp` raises rather than returning fewer breakpoints than you asked for.

## Coverage

Everything in `ruptures` 1.1.9:

| | |
|---|---|
| **Detectors** | `Dynp`, `Pelt`, `Binseg`, `BottomUp`, `Window`, `KernelCPD` |
| **Costs** | `l1`, `l2`, `normal`, `rbf`, `cosine`, `rank`, `mahalanobis`, `clinear`, `linear`, `ar` |
| **Datasets** | `pw_constant`, `pw_linear`, `pw_normal`, `pw_wavy` |
| **Metrics** | `precision_recall`, `hausdorff`, `randindex`, `meantime`, `hamming` |
| **Also** | `display`, `cost_factory`, `BaseCost`, `BaseEstimator`, `Bnode`, exceptions |
| **New** | `Crops` |

Datasets are ported verbatim, so the same `seed` gives the same signal — which
is what makes the differential tests meaningful.

## For code you cannot edit

When a dependency deep in the stack imports `ruptures` by name:

```python
import ruptures_rs
ruptures_rs.install()   # before anything imports `ruptures`

import ruptures as rpt  # this is now ruptures_rs
```

It refuses rather than half-patching the module graph if the real `ruptures` has
already been imported. Leaf modules are aliased too, so
`from ruptures.costs.costl2 import CostL2` resolves. What it cannot fake is
distribution metadata: `importlib.metadata.version("ruptures")` still reports
that no such distribution is installed, because none is.

## Development

```bash
pip install maturin pytest numpy scipy ruptures
maturin develop --release
pytest tests/ -q
cargo test
```

The full benchmark is about half an hour, nearly all of it spent inside
`ruptures`. It can be rebuilt a piece at a time, which is what you want after
changing one cost:

```bash
python bench/bench.py --section models --json bench/results.json --append
python bench/bench.py --render bench/results.json
```

## Licence and credit

BSD-2-Clause, the same licence as `ruptures`.

This is a reimplementation of [`ruptures`](https://github.com/deepcharles/ruptures)
by Charles Truong, Laurent Oudre and Nicolas Vayatis, whose API, algorithms and
semantics it deliberately reproduces. The pure-Python fallback in
`_fallback.py` is a direct port of their code. If you use this in research,
cite their paper:

> C. Truong, L. Oudre, N. Vayatis. *Selective review of offline change point
> detection methods.* Signal Processing, 167:107299, 2020.
