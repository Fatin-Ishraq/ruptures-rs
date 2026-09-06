# Changelog

This project follows [Semantic Versioning](https://semver.org/). Dates are
ISO 8601.

## 0.1.0 — 2026-09-06

First public release: a drop-in reimplementation of `ruptures` 1.1.9 in Rust,
with the same classes, arguments and breakpoints, plus `Crops`, which the
reference does not have.

Everything below is development history rather than news to a new user. It is
recorded because the defects were real, several of them were the kind that a
test suite passing 759 cases still did not see, and someone comparing this
package to the reference deserves to know what was found and what was decided.

### Fixed — crashes

Every one of these took the interpreter with it rather than raising something a
caller could catch. A Rust panic crosses `pyo3` as `PanicException`, which
inherits from `BaseException`, so `except Exception` does not see it.

- A `NaN` anywhere in the signal panicked `l1` (an `unwrap` on the median's
  comparator), `rank` (a `sort_by` that stopped being a total order) and `rbf`
  (the same median, on pairwise distances). All ordering now goes through
  `f64::total_cmp`, and `NaN` propagates into the cost the way it does in
  `ruptures`.
- `cost.sum_of_costs([60, 500])` on a 120-sample signal indexed off the end of a
  prefix array. Segment bounds are validated and raise `ValueError`.
- `jump=0` panicked inside `step_by`. It is refused at construction.
- A wide signal aborted the process: the covariance-shaped costs (`normal`,
  `mahalanobis`, `linear`, `ar`) allocate `(n+1) * d^2` doubles with no guard,
  and a failed allocation in Rust aborts rather than raising. A 50,000 x 400
  signal asked for 60 GiB. There is now a budget check with an actionable
  message, applied before NumPy builds a covariance of its own.

### Fixed — silently wrong answers

- `Dynp` returned *fewer* breakpoints than requested when no admissible
  partition existed, instead of raising. It now raises, as `ruptures` does.
- A cost clamped at zero — needed because the prefix-sum forms can land a few
  ulp below zero on a constant segment — turned `NaN` into `0.0` through
  `f64::max`. A zero cost is the most attractive segment there is, so corrupt
  data pulled breakpoints towards itself.
- `CostNormal(add_small_diag=0)` and `CostAR(order=-1)` silently ran with the
  default, because a failed parameter extraction fell back instead of
  propagating. `add_small_diag` now follows Python truthiness, matching
  `ruptures`; a negative `order` is refused.
- `linear` and `ar` decided rank by whether a Cholesky succeeded, so a merely
  ill-conditioned design reported a residual of `0.0` where NumPy returns a real
  one. Rank is now decided from the eigenvalues of the normal-equations matrix.
- `Window` scored half-windows narrower than the cost's own `min_size` —
  computing a median of one point — where `ruptures` raises `NotEnoughPoints`.

### Fixed — agreement with `ruptures`

- `KernelCPD` was routed through `ruptures`' *Python* cost classes, which are
  not what its C extension computes. The C cosine kernel has a unit diagonal
  where `scipy`'s `squareform` leaves a zero one — worth exactly one unit of
  penalty per segment, so `predict(pen=p)` was solving the problem for `p - 1` —
  and its Gaussian exponent is clipped in single precision. Its penalised search
  also folds the penalty in and prunes differently from the Python `Pelt`. All
  three are now reproduced, and `KernelCPD` agrees across 360 differential cases
  where it previously disagreed on cosine penalties.
- `Pelt` accumulated `(total + cost) + pen`; summing a `ruptures` partition dict
  associates as `total + (cost + pen)`. The difference is one ulp, and it moved
  breakpoints on signals with exact ties.
- The default Mahalanobis metric was inverted with a Cholesky, which disagrees
  with `numpy.linalg.inv` about which covariances are singular — refusing
  signals the reference segments, and raising a different exception when it did
  refuse. It is now computed in NumPy, so `LinAlgError` matches too.
- `cosine` returned a finite cost for zero-norm rows where `scipy` returns
  `NaN`.
- `Dynp` discarded partitions whose cost was `-inf`, which `ruptures` admits and
  frequently prefers — reachable with `normal` and `add_small_diag=False`.

### Fixed — accuracy

- `clinear` lost most of its significant digits on exactly the signals it is
  for. Centring removes an offset but not a trend, so on a 20,000-point ramp the
  relative error was 6e-3 against the reference's 1e-11. The cost is invariant
  under subtracting *any* affine function of the sample index, so the signal is
  now detrended by its global least-squares line first: 1e-11, on par with the
  reference.
- `ar` was solved on raw values; the intercept column makes the residual
  invariant under a shift, so the signal is now centred first. Relative error at
  an offset of 1e5 went from 1.7e-2 to 2.8e-10.
- `linear` now removes a global least-squares fit from the response and
  accumulates its design centred. Relative error at an offset of 1e5 went from
  7.4e-3 to 3.8e-5.
- A Jacobi eigendecomposition tested convergence against an absolute `1e-300`,
  which a Gram matrix of any realistic scale never reaches, so every call ran all
  one hundred sweeps. The threshold is now relative.

### Added

- `metrics.hamming`, `metrics.sanity_check`, `metrics.BadPartitions`,
  `utils.Bnode` and `utils.from_path_matrix_to_bkps_list`, completing the
  `ruptures` surface.
- `Window` supports `custom_cost`, which previously raised `NotImplementedError`
  in contradiction of the documented fallback.
- `cost_factory` discovers user-defined `BaseCost` subclasses by their `model`
  name, so `Pelt(model="my_cost")` works as it does in `ruptures`.
- `install()` aliases leaf modules, so `from ruptures.costs.costl2 import CostL2`
  and `from ruptures.detection.pelt import Pelt` resolve.
- `KernelCPD.segmentations_dict` is filled for every `k` up to the requested
  number, from a single dynamic program, as the reference's path matrix does.
- A `RuntimeWarning` when the default Mahalanobis metric comes from a covariance
  that is numerically singular, where every cost computed from it is dominated
  by rounding error. `ruptures` says nothing in this case.
- `CostMl` accepts a nested list or an integer array as `metric`; `jump` and
  `min_size` accept whole-valued floats. All three work in `ruptures`.
- `CostLinear` accepts a single-column signal, where NumPy reports rank 0 of 0
  and the residual is the whole of `y^T y`.

### Changed

- `hausdorff` and `meantime` return floats, as `ruptures` does through
  `scipy.spatial.distance.cdist`. `hausdorff` previously returned an integer.
- Metric validation uses explicit exceptions rather than `assert`, which
  `python -O` strips, and no longer requires sorted breakpoints — `ruptures`
  does not.
- Segment bounds outside the signal raise instead of being silently clamped by
  NumPy slicing.
- `CostRank.error` returns a float rather than the 1x1 array `ruptures` returns
  from an uncollapsed matrix product.

### Known differences

Recorded in the README under "Where the answers differ". In short: signals with
exactly repeated values admit many equally-optimal segmentations, and `Window`
can pick a different number of peaks on them; `linear` on a design whose
condition number exceeds about 1e8 is reported as rank deficient; the default
Mahalanobis metric is noise once the covariance is near-singular, which now
warns; and `normal` with `add_small_diag=False` on a segment that is constant to
machine precision is meaningless in both libraries.

### Testing

1,392 tests, up from 759, with 800 randomised differential fuzz cases up from
440. New: fuzz coverage for `Window`, `epsilon` stopping, `linear`/`ar`/`cosine`
and `KernelCPD` in both of its modes; robustness tests for every crash above;
and tie-behaviour tests that assert the exact detectors never return a worse
segmentation than the reference on tied input, rather than asserting a
coincidence. `cargo test` runs 24 Rust unit tests in CI, and a new job installs
the built sdist and imports it, so a release cannot ship sources that do not
compile.
