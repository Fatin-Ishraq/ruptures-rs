# Accuracy and correctness

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

- **1,392 tests**, most of them differential against `ruptures` on the same
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
