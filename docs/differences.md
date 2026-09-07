# Where the answers differ

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
