//! Segment cost functions.
//!
//! Every cost here mirrors the corresponding `ruptures` class exactly,
//! including its arithmetic shape, so that segmentations agree.
//!
//! The important difference is complexity. In `ruptures`, `CostL2.error` is
//! `signal[start:end].var(axis=0).sum() * (end - start)`: an O(len) NumPy call
//! whose dispatch overhead dominates its own arithmetic, evaluated millions of
//! times inside the dynamic program. Here it is a difference of prefix sums:
//! O(1), no allocation, no dispatch.
//!
//! | model         | ruptures   | here      |
//! |---------------|------------|-----------|
//! | l2            | O(len*d)   | O(d)      |
//! | normal        | O(len*d^2) | O(d^3)    |
//! | mahalanobis   | O(len^2)   | O(d^2)    |
//! | rank          | O(len*d)   | O(d^2)    |
//! | clinear       | O(len*d)   | O(d)      |
//! | linear / ar   | O(len*d^2) | O(d^3)    |
//! | rbf / cosine  | O(len^2)   | O(1)      |
//! | l1            | O(len*d)   | O(len*d)  |
//!
//! `l1` is the one that resists: it needs a per-segment median, which is not a
//! prefix-summable statistic. It still gains a large constant factor from
//! dropping NumPy dispatch.

use crate::linalg;
use crate::prefix::{
    center, column_means, detrend_affine, subtract_means, Cumsum2, Neumaier, Prefix1, PrefixOuter,
};
use std::cell::RefCell;

/// The two O(1) block quantities a kernel dynamic program needs: the trace of
/// the kernel sub-matrix and the sum of all its entries.
///
/// `KernelCPD`'s penalised mode is a transcription of `ruptures`' C extension,
/// which works directly with these rather than with a segment cost, so they are
/// exposed separately from [`Cost`].
/// Clamp a quantity that is mathematically non-negative, without swallowing a
/// NaN.
///
/// `f64::max` returns the *other* operand when one is NaN, so `x.max(0.0)`
/// turns a NaN cost into a clean `0.0` — which reads as a perfect segment and
/// pulls breakpoints towards corrupt data. The comparison below is false for
/// NaN, so a NaN passes straight through, exactly as it does in NumPy.
#[inline]
fn clamp_nonneg(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else {
        v
    }
}

pub trait KernelBlocks: Send + Sync {
    /// Sum of `K(i, i)` for `i` in `start..end`.
    fn diag_sum(&self, start: usize, end: usize) -> f64;
    /// Sum of the `[start, end) x [start, end)` block of `K`.
    fn block_sum(&self, start: usize, end: usize) -> f64;
}

pub trait Cost: Send + Sync {
    fn min_size(&self) -> usize;
    fn error(&self, start: usize, end: usize) -> f64;
    fn model(&self) -> &'static str;

    /// The kernel block sums, when this cost is backed by a kernel table.
    fn as_kernel_blocks(&self) -> Option<&dyn KernelBlocks> {
        None
    }

    fn sum_of_costs(&self, bkps: &[usize]) -> f64 {
        let mut acc = 0.0;
        let mut start = 0usize;
        for &end in bkps {
            acc += self.error(start, end);
            start = end;
        }
        acc
    }
}

// ---------------------------------------------------------------- l2

pub struct CostL2 {
    s1: Prefix1,
    s2: Prefix1,
    d: usize,
}

impl CostL2 {
    pub fn new(sig: &[f64], n: usize, d: usize) -> Self {
        let y = center(sig, n, d);
        let sq: Vec<f64> = y.iter().map(|v| v * v).collect();
        Self {
            s1: Prefix1::build(&y, n, d),
            s2: Prefix1::build(&sq, n, d),
            d,
        }
    }
}

impl Cost for CostL2 {
    fn min_size(&self) -> usize {
        1
    }
    fn model(&self) -> &'static str {
        "l2"
    }
    #[inline]
    fn error(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        // Mirrors `sub.var(axis=0).sum() * (end - start)`: per-dimension
        // variance first, summed, then scaled - same association order as
        // NumPy so the two agree to the last few ulp.
        let mut var_sum = 0.0;
        for j in 0..self.d {
            let s1 = self.s1.seg(start, end, j);
            let s2 = self.s2.seg(start, end, j);
            // A sum of squared deviations cannot be negative; the prefix-sum
            // identity can still land a few ulp below zero on a segment that
            // is exactly constant, where NumPy's two-pass reduction returns a
            // clean zero. Clamping restores the invariant `error >= 0`.
            let ssd = clamp_nonneg(s2 - s1 * s1 / n);
            var_sum += ssd / n;
        }
        var_sum * n
    }
}

// ---------------------------------------------------------------- l1

thread_local! {
    static L1_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

pub struct CostL1 {
    sig: Vec<f64>,
    d: usize,
}

impl CostL1 {
    pub fn new(sig: &[f64], n: usize, d: usize) -> Self {
        Self {
            sig: center(sig, n, d),
            d,
        }
    }
}

impl Cost for CostL1 {
    fn min_size(&self) -> usize {
        2
    }
    fn model(&self) -> &'static str {
        "l1"
    }
    fn error(&self, start: usize, end: usize) -> f64 {
        let len = end - start;
        L1_SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.clear();
            buf.resize(len, 0.0);
            let mut total = 0.0;
            for j in 0..self.d {
                for (k, slot) in buf.iter_mut().enumerate() {
                    *slot = self.sig[(start + k) * self.d + j];
                }
                let med = linalg::median_inplace(&mut buf);
                for k in 0..len {
                    total += (self.sig[(start + k) * self.d + j] - med).abs();
                }
            }
            total
        })
    }
}

// ---------------------------------------------------------------- normal

pub struct CostNormal {
    s1: Prefix1,
    outer: PrefixOuter,
    d: usize,
    add_small_diag: bool,
}

impl CostNormal {
    pub fn new(sig: &[f64], n: usize, d: usize, add_small_diag: bool) -> Self {
        let y = center(sig, n, d);
        Self {
            s1: Prefix1::build(&y, n, d),
            outer: PrefixOuter::build(&y, n, d),
            d,
            add_small_diag,
        }
    }
}

impl Cost for CostNormal {
    fn min_size(&self) -> usize {
        2
    }
    fn model(&self) -> &'static str {
        "normal"
    }
    fn error(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        let d = self.d;
        // ruptures is deliberately asymmetric here: 1-D uses `sub.var()`
        // (ddof=0) while multivariate uses `np.cov` (ddof=1). Reproduced.
        if d == 1 {
            let s1 = self.s1.seg(start, end, 0);
            let mut sq = [0.0f64; 1];
            self.outer.seg_into(start, end, &mut sq);
            // Clamped for the same reason as `CostL2`: on an exactly constant
            // segment the one-pass identity can go a few ulp negative, and the
            // logarithm turns that into a NaN where `ruptures` sees a clean
            // zero variance.
            let mut var = clamp_nonneg((sq[0] - s1 * s1 / n) / n);
            if self.add_small_diag {
                var += 1e-6;
            }
            return var.ln() * n;
        }
        let mut cov = vec![0.0; d * d];
        self.outer.seg_into(start, end, &mut cov);
        let mut mean = vec![0.0; d];
        for (j, m) in mean.iter_mut().enumerate() {
            *m = self.s1.seg(start, end, j) / n;
        }
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] = (cov[a * d + b] - n * mean[a] * mean[b]) / (n - 1.0);
            }
        }
        if self.add_small_diag {
            for i in 0..d {
                cov[i * d + i] += 1e-6;
            }
        }
        match linalg::logdet_spd(&mut cov, d) {
            Some(v) => v * n,
            None => f64::NEG_INFINITY,
        }
    }
}

// ---------------------------------------------------------------- mahalanobis

/// `CostMl`, but without the O(n^2) Gram matrix.
///
/// `ruptures` materialises `S M S^T` and sums a sub-block, which is O(n^2)
/// memory and O(len^2) per query. The cost is algebraically
/// `sum_i (x_i - mu)^T M (x_i - mu)`, so prefix sums of `x` and `x x^T` give the
/// same number in O(d^2) with no quadratic memory at all.
pub struct CostMl {
    s1: Prefix1,
    outer: PrefixOuter,
    metric: Vec<f64>,
    d: usize,
}

impl CostMl {
    pub fn new(sig: &[f64], n: usize, d: usize, metric: Vec<f64>) -> Self {
        let y = center(sig, n, d);
        Self {
            s1: Prefix1::build(&y, n, d),
            outer: PrefixOuter::build(&y, n, d),
            metric,
            d,
        }
    }
}

impl Cost for CostMl {
    fn min_size(&self) -> usize {
        2
    }
    fn model(&self) -> &'static str {
        "mahalanobis"
    }
    fn error(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        let d = self.d;
        let mut scatter = vec![0.0; d * d];
        self.outer.seg_into(start, end, &mut scatter);
        // trace(M * sum_i x_i x_i^T)
        let mut trace = 0.0;
        for a in 0..d {
            for b in 0..d {
                trace += self.metric[a * d + b] * scatter[b * d + a];
            }
        }
        // (sum x)^T M (sum x) / n
        let mut s = vec![0.0; d];
        for (j, v) in s.iter_mut().enumerate() {
            *v = self.s1.seg(start, end, j);
        }
        let mut quad = 0.0;
        for a in 0..d {
            for b in 0..d {
                quad += s[a] * self.metric[a * d + b] * s[b];
            }
        }
        trace - quad / n
    }
}

// ---------------------------------------------------------------- kernel (rbf, cosine)

/// Which kernel table to build.
///
/// `ruptures` ships *two* implementations of every kernel cost, and they do not
/// agree with each other. The Python `CostRbf` / `CostCosine` classes build a
/// dense Gram matrix through `scipy.spatial.distance`, whose `squareform` step
/// zeroes the diagonal. The C extension behind `KernelCPD` evaluates the kernel
/// directly, including on the diagonal, and clips the Gaussian exponent in
/// *single* precision.
///
/// Those differences are not cosmetic. A cosine diagonal of 1 rather than 0
/// shifts the penalised objective by exactly one unit per segment, so
/// `KernelCPD(kernel="cosine").predict(pen=p)` is the Python cost at penalty
/// `p - 1`. Reproducing both variants is what makes each detector agree with
/// the `ruptures` code path it actually corresponds to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelFlavor {
    /// `scipy` + `squareform`: zero diagonal for cosine, unit diagonal for rbf,
    /// f64 clipping of the Gaussian exponent.
    Python,
    /// The `ekcpd` C extension: the kernel is evaluated on the diagonal too,
    /// and the Gaussian exponent is clipped as a `float`.
    Extension,
}

/// Kernel costs: `trace(K_block) - sum(K_block) / len`.
///
/// Backed by a 2-D cumulative sum, which is the same O(n^2) memory `ruptures`
/// already spends on the dense Gram matrix but answers each query in O(1)
/// instead of O(len^2).
pub struct CostKernel {
    cum: Cumsum2,
    diag: Prefix1,
    model: &'static str,
}

/// The C extension's `clip(float n, float lower, float upper)` truncates its
/// argument to single precision on the way in and back to double on the way
/// out. That rounding decides ties on signals where most kernel entries land on
/// the clip boundary, so it has to be reproduced exactly rather than tidied up.
#[inline]
fn clip_f32(v: f64, lo: f32, hi: f32) -> f64 {
    let n = v as f32;
    let clipped = if n > hi {
        hi
    } else if lo > n {
        lo
    } else {
        n
    };
    clipped as f64
}

impl CostKernel {
    fn from_fn<F: FnMut(usize, usize) -> f64, G: FnMut(usize) -> f64>(
        n: usize,
        f: F,
        mut diag_fn: G,
        model: &'static str,
    ) -> Self {
        let diag_vals: Vec<f64> = (0..n).map(&mut diag_fn).collect();
        Self {
            cum: Cumsum2::build(n, f),
            diag: Prefix1::build(&diag_vals, n, 1),
            model,
        }
    }

    /// Squared euclidean distances, and the reciprocal-median `gamma` heuristic
    /// when none was supplied.
    fn rbf_gamma(sig: &[f64], n: usize, d: usize, gamma: Option<f64>) -> f64 {
        match gamma {
            Some(g) => g,
            None => {
                // median of the condensed pairwise distance vector
                let mut cond = Vec::with_capacity(n * n.saturating_sub(1) / 2);
                for i in 0..n {
                    for j in (i + 1)..n {
                        let mut acc = 0.0;
                        for k in 0..d {
                            let t = sig[i * d + k] - sig[j * d + k];
                            acc += t * t;
                        }
                        cond.push(acc);
                    }
                }
                let med = if cond.is_empty() {
                    0.0
                } else {
                    linalg::median_inplace(&mut cond)
                };
                if med != 0.0 {
                    1.0 / med
                } else {
                    1.0
                }
            }
        }
    }

    pub fn rbf(
        sig: &[f64],
        n: usize,
        d: usize,
        gamma: Option<f64>,
        flavor: KernelFlavor,
    ) -> (Self, f64) {
        let gamma = Self::rbf_gamma(sig, n, d, gamma);
        let sq = move |i: usize, j: usize| -> f64 {
            let mut acc = 0.0;
            for k in 0..d {
                let t = sig[i * d + k] - sig[j * d + k];
                acc += t * t;
            }
            acc
        };
        let cost = match flavor {
            KernelFlavor::Python => Self::from_fn(
                n,
                move |i, j| {
                    if i == j {
                        // `squareform` zeroes the diagonal before `exp`
                        1.0
                    } else {
                        (-(gamma * sq(i, j)).clamp(1e-2, 1e2)).exp()
                    }
                },
                |_| 1.0,
                "rbf",
            ),
            KernelFlavor::Extension => Self::from_fn(
                n,
                move |i, j| (-clip_f32(gamma * sq(i, j), 1e-2, 1e2)).exp(),
                // `gaussian_kernel(x, x)` is `exp(-clip(0.0, 0.01, 100))`,
                // which is exp(-0.01f) rather than 1.
                |_| (-clip_f32(0.0, 1e-2, 1e2)).exp(),
                "rbf",
            ),
        };
        (cost, gamma)
    }

    pub fn cosine(sig: &[f64], n: usize, d: usize, flavor: KernelFlavor) -> Self {
        let norms: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += sig[i * d + k] * sig[i * d + k];
                }
                acc.sqrt()
            })
            .collect();
        let sim = move |i: usize, j: usize, norms: &[f64]| -> f64 {
            let mut dot = 0.0;
            for k in 0..d {
                dot += sig[i * d + k] * sig[j * d + k];
            }
            // `sqrt(a) * sqrt(b)`, exactly as both implementations spell it.
            // A zero-norm row yields 0/0 = NaN in `scipy` and in the C kernel
            // alike, so the NaN is propagated rather than papered over.
            dot / (norms[i] * norms[j])
        };
        match flavor {
            // `squareform(1 - pdist(..., "cosine"))` has a zero diagonal.
            KernelFlavor::Python => {
                let nrm = norms.clone();
                Self::from_fn(
                    n,
                    move |i, j| {
                        if i == j {
                            0.0
                        } else {
                            sim(i, j, &nrm)
                        }
                    },
                    |_| 0.0,
                    "cosine",
                )
            }
            KernelFlavor::Extension => {
                let nrm = norms.clone();
                let nrm2 = norms.clone();
                Self::from_fn(
                    n,
                    move |i, j| sim(i, j, &nrm),
                    move |i| {
                        let mut dot = 0.0;
                        for k in 0..d {
                            dot += sig[i * d + k] * sig[i * d + k];
                        }
                        dot / (nrm2[i] * nrm2[i])
                    },
                    "cosine",
                )
            }
        }
    }
}

impl KernelBlocks for CostKernel {
    #[inline]
    fn diag_sum(&self, start: usize, end: usize) -> f64 {
        self.diag.seg(start, end, 0)
    }
    #[inline]
    fn block_sum(&self, start: usize, end: usize) -> f64 {
        self.cum.rect(start, end)
    }
}

impl Cost for CostKernel {
    fn min_size(&self) -> usize {
        1
    }
    fn model(&self) -> &'static str {
        self.model
    }
    fn as_kernel_blocks(&self) -> Option<&dyn KernelBlocks> {
        Some(self)
    }
    #[inline]
    fn error(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        self.diag.seg(start, end, 0) - self.cum.rect(start, end) / n
    }
}

// ------------------------------------------------- linear kernel (KernelCPD)

/// The C extension's `linear` kernel, `K(x, y) = <x, y>`, without the quadratic
/// memory a dense Gram would need.
///
/// `trace - blocksum / len` expands to `sum_i |x_i|^2 - sum_d (sum_i x_id)^2 / len`,
/// which is the same number `CostL2` computes but grouped the way the C code
/// groups it. Keeping this separate from `CostL2` means `KernelCPD(kernel="linear")`
/// is arithmetically the extension it replaces, while still costing O(n*d)
/// memory rather than O(n^2).
pub struct CostKernelLinear {
    s1: Prefix1,
    sq: Prefix1,
    d: usize,
}

impl CostKernelLinear {
    pub fn new(sig: &[f64], n: usize, d: usize) -> Self {
        let y = center(sig, n, d);
        let rowsq: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += y[i * d + k] * y[i * d + k];
                }
                acc
            })
            .collect();
        Self {
            s1: Prefix1::build(&y, n, d),
            sq: Prefix1::build(&rowsq, n, 1),
            d,
        }
    }
}

impl KernelBlocks for CostKernelLinear {
    #[inline]
    fn diag_sum(&self, start: usize, end: usize) -> f64 {
        self.sq.seg(start, end, 0)
    }
    #[inline]
    fn block_sum(&self, start: usize, end: usize) -> f64 {
        let mut block = 0.0;
        for j in 0..self.d {
            let s = self.s1.seg(start, end, j);
            block += s * s;
        }
        block
    }
}

impl Cost for CostKernelLinear {
    fn min_size(&self) -> usize {
        1
    }
    fn model(&self) -> &'static str {
        "l2"
    }
    fn as_kernel_blocks(&self) -> Option<&dyn KernelBlocks> {
        Some(self)
    }
    #[inline]
    fn error(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        let trace = self.sq.seg(start, end, 0);
        let mut block = 0.0;
        for j in 0..self.d {
            let s = self.s1.seg(start, end, j);
            block += s * s;
        }
        clamp_nonneg(trace - block / n)
    }
}

// ---------------------------------------------------------------- rank

pub struct CostRank {
    s1: Prefix1,
    inv_cov: Vec<f64>,
    d: usize,
}

impl CostRank {
    pub fn new(sig: &[f64], n: usize, d: usize) -> Self {
        // average ranks (1-based), matching scipy.stats.rankdata
        let mut ranks = vec![0.0; n * d];
        let mut idx: Vec<usize> = (0..n).collect();
        for j in 0..d {
            // `total_cmp`, not `partial_cmp(..).unwrap_or(Equal)`: the latter
            // is not a total order once a NaN is present, and Rust's sort
            // detects that and panics straight through the FFI boundary.
            idx.sort_by(|&a, &b| linalg::total_cmp(&sig[a * d + j], &sig[b * d + j]));
            let mut i = 0usize;
            while i < n {
                let mut k = i + 1;
                while k < n && sig[idx[k] * d + j] == sig[idx[i] * d + j] {
                    k += 1;
                }
                let avg = ((i + 1 + k) as f64) / 2.0; // mean of ranks i+1..=k
                for t in i..k {
                    ranks[idx[t] * d + j] = avg;
                }
                i = k;
            }
        }
        let shift = (n as f64 + 1.0) / 2.0;
        for v in ranks.iter_mut() {
            *v -= shift;
        }
        // np.cov(..., rowvar=False, bias=True)
        let mut mean = vec![0.0; d];
        for (j, m) in mean.iter_mut().enumerate() {
            let mut acc = 0.0;
            for i in 0..n {
                acc += ranks[i * d + j];
            }
            *m = acc / n as f64;
        }
        let mut cov = vec![0.0; d * d];
        for i in 0..n {
            for a in 0..d {
                for b in 0..d {
                    cov[a * d + b] += (ranks[i * d + a] - mean[a]) * (ranks[i * d + b] - mean[b]);
                }
            }
        }
        for v in cov.iter_mut() {
            *v /= n as f64;
        }
        Self {
            s1: Prefix1::build(&ranks, n, d),
            inv_cov: linalg::pinv_sym(&cov, d),
            d,
        }
    }
}

impl Cost for CostRank {
    fn min_size(&self) -> usize {
        2
    }
    fn model(&self) -> &'static str {
        "rank"
    }
    fn error(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        let d = self.d;
        let mut mean = vec![0.0; d];
        for (j, m) in mean.iter_mut().enumerate() {
            *m = self.s1.seg(start, end, j) / n;
        }
        let mut quad = 0.0;
        for a in 0..d {
            for b in 0..d {
                quad += mean[a] * self.inv_cov[a * d + b] * mean[b];
            }
        }
        -n * quad
    }
}

// ---------------------------------------------------------------- least squares (linear, ar)

/// Shared engine for `CostLinear` and `CostAR`: the residual sum of squares of
/// `y ~ X` on a segment, via prefix sums of `X^T X`, `X^T y` and `y^T y`.
///
/// Two numerical defences, because the naive form of this is badly behaved:
///
/// 1. **The response is pre-fitted.** `RSS(y - X b0)` equals `RSS(y)` for any
///    fixed `b0`, so removing the whole-signal least-squares fit first is exact.
///    Without it the final subtraction `y^T y - beta . X^T y` cancels two large
///    quantities to leave a small one, and on offset data that costs most of
///    the significant digits.
/// 2. **The design is accumulated centred.** Prefix sums are built on `X` minus
///    its column means and the exact `X^T X` is reconstructed per query, so the
///    differencing happens on small numbers rather than on sums that grow with
///    the whole signal.
///
/// What remains is inherent to the approach: solving through `X^T X` squares the
/// condition number of `X`, where NumPy's `lstsq` factorises `X` itself. For a
/// design whose columns sit far from the origin, this is less accurate than the
/// reference, and `CostLinear` has no intercept term that would let the columns
/// be centred without changing the model.
///
/// The rank test is NumPy's, not Cholesky's: see [`linalg::lstsq_factor`].
pub struct CostLstsq {
    cxx: PrefixOuter,
    cx: Prefix1,
    cxe: Vec<Prefix1>,
    ce: Prefix1,
    cee: Prefix1,
    mx: Vec<f64>,
    my: Vec<f64>,
    p: usize,
    ny: usize,
    min_size: usize,
    model: &'static str,
}

impl CostLstsq {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: &[f64],
        y: &[f64],
        n: usize,
        p: usize,
        ny: usize,
        min_size: usize,
        model: &'static str,
    ) -> Self {
        // ---- global least-squares fit, removed from the response ----------
        let mut g0 = vec![0.0; p * p];
        {
            let mut acc = vec![Neumaier::new(); p * p];
            for i in 0..n {
                for a in 0..p {
                    for b in 0..p {
                        acc[a * p + b].add(x[i * p + a] * x[i * p + b]);
                    }
                }
            }
            for (slot, a) in g0.iter_mut().zip(acc.iter()) {
                *slot = a.value();
            }
        }
        let mut beta0 = vec![0.0; ny * p];
        if let Some(factor) = linalg::lstsq_factor(&g0, p, n) {
            let mut rhs = vec![0.0; p];
            let mut out = vec![0.0; p];
            for c in 0..ny {
                let mut acc = vec![Neumaier::new(); p];
                for i in 0..n {
                    let yc = y[i * ny + c];
                    for k in 0..p {
                        acc[k].add(x[i * p + k] * yc);
                    }
                }
                for k in 0..p {
                    rhs[k] = acc[k].value();
                }
                factor.solve_into(&rhs, &mut out);
                if out.iter().all(|v| v.is_finite()) {
                    beta0[c * p..(c + 1) * p].copy_from_slice(&out);
                }
            }
        }
        let mut resid = vec![0.0; n * ny];
        for i in 0..n {
            for c in 0..ny {
                let mut fitted = 0.0;
                for k in 0..p {
                    fitted += x[i * p + k] * beta0[c * p + k];
                }
                resid[i * ny + c] = y[i * ny + c] - fitted;
            }
        }

        // ---- centred accumulation -----------------------------------------
        let mx = column_means(x, n, p);
        let my = column_means(&resid, n, ny);
        let cx_vals = subtract_means(x, n, p, &mx);
        let ce_vals = subtract_means(&resid, n, ny, &my);

        let mut cxe = Vec::with_capacity(ny);
        for c in 0..ny {
            let mut prod = vec![0.0; n * p];
            for i in 0..n {
                let e = ce_vals[i * ny + c];
                for k in 0..p {
                    prod[i * p + k] = cx_vals[i * p + k] * e;
                }
            }
            cxe.push(Prefix1::build(&prod, n, p));
        }
        let esq: Vec<f64> = ce_vals.iter().map(|v| v * v).collect();

        Self {
            cxx: PrefixOuter::build(&cx_vals, n, p),
            cx: Prefix1::build(&cx_vals, n, p),
            cxe,
            ce: Prefix1::build(&ce_vals, n, ny),
            cee: Prefix1::build(&esq, n, ny),
            mx,
            my,
            p,
            ny,
            min_size,
            model,
        }
    }
}

impl Cost for CostLstsq {
    fn min_size(&self) -> usize {
        self.min_size
    }
    fn model(&self) -> &'static str {
        self.model
    }
    fn error(&self, start: usize, end: usize) -> f64 {
        let len = end - start;
        // NumPy returns no residual unless the system is overdetermined.
        if len <= self.p {
            return 0.0;
        }
        let p = self.p;
        let lf = len as f64;

        let mut sc = vec![0.0; p];
        for (k, slot) in sc.iter_mut().enumerate() {
            *slot = self.cx.seg(start, end, k);
        }
        let mut gram = vec![0.0; p * p];
        self.cxx.seg_into(start, end, &mut gram);
        for a in 0..p {
            for b in 0..p {
                gram[a * p + b] +=
                    sc[a] * self.mx[b] + self.mx[a] * sc[b] + lf * self.mx[a] * self.mx[b];
            }
        }

        // Rank is a property of the design alone: when NumPy would report a
        // deficient rank it returns an empty residual for every response
        // column at once, which `ruptures` sums to 0.0.
        let factor = match linalg::lstsq_factor(&gram, p, len) {
            Some(f) => f,
            None => return 0.0,
        };

        let mut total = 0.0;
        let mut rhs = vec![0.0; p];
        let mut beta = vec![0.0; p];
        for c in 0..self.ny {
            let se = self.ce.seg(start, end, c);
            for (k, slot) in rhs.iter_mut().enumerate() {
                *slot = self.cxe[c].seg(start, end, k)
                    + self.my[c] * sc[k]
                    + self.mx[k] * se
                    + lf * self.mx[k] * self.my[c];
            }
            let yty =
                self.cee.seg(start, end, c) + 2.0 * self.my[c] * se + lf * self.my[c] * self.my[c];
            factor.solve_into(&rhs, &mut beta);
            let mut dot = 0.0;
            for k in 0..p {
                dot += beta[k] * rhs[k];
            }
            total += clamp_nonneg(yty - dot);
        }
        total
    }
}

// ---------------------------------------------------------------- clinear

/// Continuous piecewise-linear cost.
///
/// `ruptures` builds the affine approximation explicitly and sums the squared
/// residual, O(len). Expanding the square leaves only sums of `x`, `x^2` and
/// `i * x`, so the whole thing collapses to O(d) prefix-sum arithmetic.
///
/// The expansion carries an `m * intercept^2` term, and the intercept is a raw
/// signal value. The cost is invariant under subtracting *any* affine function
/// of the sample index — both the data and its affine approximation move
/// together — so the signal is detrended by its global least-squares line
/// before the prefix arrays are built. Centring alone is not enough: on a
/// trending signal, which is exactly what this cost is for, the residual after
/// centring still grows with the trend and the expansion cancels away most of
/// the significant digits.
pub struct CostCLinear {
    sig: Vec<f64>,
    t1: Prefix1,
    t2: Prefix1,
    tw: Prefix1,
    d: usize,
}

impl CostCLinear {
    pub fn new(sig: &[f64], n: usize, d: usize) -> Self {
        let sig = detrend_affine(sig, n, d);
        let sq: Vec<f64> = sig.iter().map(|v| v * v).collect();
        let mut weighted = vec![0.0; n * d];
        for i in 0..n {
            for j in 0..d {
                weighted[i * d + j] = (i as f64) * sig[i * d + j];
            }
        }
        let t1 = Prefix1::build(&sig, n, d);
        let t2 = Prefix1::build(&sq, n, d);
        let tw = Prefix1::build(&weighted, n, d);
        Self { sig, t1, t2, tw, d }
    }
}

impl Cost for CostCLinear {
    fn min_size(&self) -> usize {
        3
    }
    fn model(&self) -> &'static str {
        "clinear"
    }
    fn error(&self, start: usize, end: usize) -> f64 {
        let start = if start == 0 { 1 } else { start };
        let m = (end - start) as f64;
        let d = self.d;
        // sum_{j=1..m} j  and  sum_{j=1..m} j^2
        let p1 = m * (m + 1.0) / 2.0;
        let p2 = m * (m + 1.0) * (2.0 * m + 1.0) / 6.0;
        let mut total = 0.0;
        for j in 0..d {
            let last = self.sig[(end - 1) * d + j];
            let prev = self.sig[(start - 1) * d + j];
            let a = (last - prev) / m; // slope
            let b = prev; // intercept
            let sx = self.t1.seg(start, end, j);
            let sxx = self.t2.seg(start, end, j);
            let swx = self.tw.seg(start, end, j);
            // sum over the segment of x_i * (i - start + 1)
            let sxj = swx - (start as f64 - 1.0) * sx;
            total += clamp_nonneg(
                sxx - 2.0 * (a * sxj + b * sx) + a * a * p2 + 2.0 * a * b * p1 + m * b * b,
            );
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 3.0 + 1.0).collect()
    }

    #[test]
    fn l2_is_never_negative_on_constant_segments() {
        let sig = vec![1e9; 64];
        let c = CostL2::new(&sig, 64, 1);
        for start in 0..60 {
            assert!(c.error(start, start + 4) >= 0.0);
        }
    }

    #[test]
    fn clinear_is_exact_on_a_pure_ramp() {
        // A straight line is its own continuous linear approximation, so every
        // segment cost is zero. Before detrending this returned values that
        // grew with the trend.
        let sig = ramp(2000);
        let c = CostCLinear::new(&sig, 2000, 1);
        for (start, end) in [(1, 2000), (5, 137), (900, 1000)] {
            assert!(c.error(start, end) < 1e-9, "({start},{end})");
        }
    }

    #[test]
    fn lstsq_reports_the_residual_of_a_well_conditioned_fit() {
        // y = 2x exactly: zero residual, full rank.
        let n = 50;
        let x: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        let c = CostLstsq::new(&x, &y, n, 1, 1, 2, "linear");
        assert!(c.error(0, n) < 1e-15);
    }

    #[test]
    fn lstsq_reports_zero_for_a_rank_deficient_design() {
        let n = 40;
        let mut x = vec![0.0; n * 2];
        for i in 0..n {
            x[i * 2] = i as f64;
            x[i * 2 + 1] = 2.0 * i as f64; // exactly collinear
        }
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 + 3.0).collect();
        let c = CostLstsq::new(&x, &y, n, 2, 1, 2, "linear");
        assert_eq!(c.error(0, n), 0.0);
    }

    #[test]
    fn kernel_flavors_differ_on_the_diagonal() {
        let sig: Vec<f64> = (0..20).map(|i| (i as f64).sin() + 1.5).collect();
        let py = CostKernel::cosine(&sig, 20, 1, KernelFlavor::Python);
        let ext = CostKernel::cosine(&sig, 20, 1, KernelFlavor::Extension);
        // Unit diagonal shifts the cost by exactly (len - 1).
        let len = 7.0;
        let diff = ext.error(3, 10) - py.error(3, 10);
        assert!((diff - (len - 1.0)).abs() < 1e-9, "{diff}");
    }

    #[test]
    fn nan_input_propagates_instead_of_reading_as_a_perfect_segment() {
        let sig = vec![1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0];
        assert!(CostL2::new(&sig, 6, 1).error(0, 6).is_nan());
        assert!(CostNormal::new(&sig, 6, 1, false).error(0, 6).is_nan());
        assert!(CostCLinear::new(&sig, 6, 1).error(1, 6).is_nan());
        assert!(CostKernelLinear::new(&sig, 6, 1).error(0, 6).is_nan());
    }

    #[test]
    fn nan_input_does_not_panic() {
        let sig = vec![1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0];
        assert!(CostL1::new(&sig, 6, 1).error(0, 6).is_nan());
        let _ = CostRank::new(&sig, 6, 1).error(0, 6);
        let _ = CostL2::new(&sig, 6, 1).error(0, 6);
        let (k, _) = CostKernel::rbf(&sig, 6, 1, None, KernelFlavor::Python);
        let _ = k.error(0, 6);
    }
}
