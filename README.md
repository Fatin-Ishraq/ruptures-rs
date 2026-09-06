# ruptures-rs

Fast, drop-in change point detection for Python, powered by Rust.

```diff
- import ruptures as rpt
+ import ruptures_rs as rpt
```

That is the whole migration. Same classes, same arguments, same breakpoints —
verified by 758 tests that run both libraries on the same input and demand
*identical* output, not merely similar output.

```bash
pip install ruptures-rs
```

## Why it is faster

`ruptures` is a well-designed library with one structural performance problem:
its cost functions are O(len) NumPy calls, and its dynamic program evaluates
them millions of times. `CostL2.error` is

```python
return self.signal[start:end].var(axis=0).sum() * (end - start)
```

which is a NumPy call whose fixed dispatch overhead dwarfs its own arithmetic.
A profile of `Dynp` on a 4,000-point signal spends 273 seconds making 7.9
million of them.

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

So the speedup is not only a constant factor from leaving Python. `Dynp` in
`ruptures` is O(n²K) cells × O(n) per cell; here it is O(n²K) cells × O(1). That
is a whole factor of `n`, which is why the measured speedup *grows* with signal
size rather than flattening out.

`l1` is the honest exception: a per-segment median is not a prefix-summable
statistic, so it stays O(len) and gains only the constant factor.

## Benchmarks

AMD Ryzen 5 5600G (6 cores), Python 3.14, `ruptures` 1.1.9. Every row was
checked for identical breakpoints; `python bench/bench.py` reproduces the table.

<!--BENCH-->

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

The widest interval is the most defensible choice: it is the segmentation least
sensitive to the parameter you could not justify picking. See
`examples/penalty_path.py`.

This is practical here only because PELT became cheap — running it a dozen times
is worth doing once each run is fast.

## Accuracy

Prefix sums are what buys the speed, and the naive form is also a classic
numerical trap: `Σx² − (Σx)²/n` is the textbook one-pass variance, which cancels
catastrophically when the mean dwarfs the spread. NumPy's `.var()` is two-pass
and therefore safe, so a careless port silently disagrees with the reference —
or returns negative variances.

Two defences, applied together: the signal is centred by its per-dimension mean
before any prefix array is built (every affected cost is invariant under that
shift), and the arrays are accumulated with Neumaier compensation.

The result is that this package is *more* accurate than the reference on
ill-conditioned input, not less. On a signal offset by 1e9, checked against
60-digit exact arithmetic:

| | exact | `ruptures` | `ruptures-rs` |
|---|---|---|---|
| `CostMl.error(23, 88)` | 21.30473705 | 24.0 | 21.30473705 |

`tests/test_precision.py` asserts this against `decimal.Decimal`, so a
regression in either direction is caught.

## Correctness

The whole project is only worth anything if the answers match, so that is what
the suite tests.

- **758 tests**, almost all differential against `ruptures` on the same input.
- **440 randomised fuzz cases** over signal length, dimension, `jump`,
  `min_size`, penalty, model and seed, asserting *exact* breakpoint equality.
  This found two real defects during development.
- **API coverage** is asserted mechanically: every public name in
  `dir(ruptures)` must exist here.
- **CROPS** is validated against `ruptures`' own PELT — for every interval it
  reports, running the reference at a penalty inside that interval must
  reproduce exactly the segmentation CROPS attributed to it.

Tie-breaking is reproduced deliberately. `ruptures` relies on Python's `min`
and `max` returning the *first* extremum; a Rust port using `Iterator::min_by`
(which returns the *last*) would quietly return different — though equally
optimal — breakpoints. Two `ruptures` quirks are matched bug-for-bug, including
PELT's positional `zip` of `admissible` against `subproblems`, because a
drop-in that is merely defensible is not a drop-in.

## Coverage

Everything in `ruptures` 1.1.9:

| | |
|---|---|
| **Detectors** | `Dynp`, `Pelt`, `Binseg`, `BottomUp`, `Window`, `KernelCPD` |
| **Costs** | `l1`, `l2`, `normal`, `rbf`, `cosine`, `rank`, `mahalanobis`, `clinear`, `linear`, `ar` |
| **Datasets** | `pw_constant`, `pw_linear`, `pw_normal`, `pw_wavy` |
| **Metrics** | `precision_recall`, `hausdorff`, `randindex`, `meantime` |
| **Also** | `display`, `cost_factory`, `BaseCost`, `BaseEstimator`, exceptions |
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
already been imported.

## Limitations

Stated plainly, because a drop-in that hides its gaps is worse than one that
does not have them.

- **`custom_cost` is not accelerated.** If you pass your own cost object, your
  Python callable *is* the inner loop, and crossing an FFI boundary millions of
  times to reach it would be slower than staying in Python. Those calls run on a
  pure-Python implementation instead: same answers, original speed. Check
  `estimator.accelerated` to see which path you are on.
- **`l1` is only constant-factor faster.** The median is not prefix-summable.
- **`rbf` and `cosine` need O(n²) memory**, the same as the dense Gram matrix
  `ruptures` builds. Past ~23,000 samples this package raises an actionable
  error instead of letting the allocator kill the process.
- **Ties may resolve differently in principle.** Tie-breaking is reproduced
  exactly, but where two segmentations have costs that differ only in the last
  ulp, a different summation order could pick the other one. Both are optimal.
  No case has been observed across 758 tests.

## Development

```bash
pip install maturin pytest numpy scipy ruptures
maturin develop --release
pytest tests/ -q
python bench/bench.py
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
