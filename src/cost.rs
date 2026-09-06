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
use crate::prefix::{center, Cumsum2, Prefix1, PrefixOuter};
use std::cell::RefCell;

pub trait Cost: Send + Sync {
    fn min_size(&self) -> usize;
    fn error(&self, start: usize, end: usize) -> f64;
    fn model(&self) -> &'static str;

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
            let ssd = s2 - s1 * s1 / n;
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
            let mut var = (sq[0] - s1 * s1 / n) / n;
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

/// Kernel costs: `trace(K_block) - sum(K_block) / len`.
///
/// Backed by a 2-D cumulative sum, which is the same O(n^2) memory `ruptures`
/// already spends on the dense Gram matrix but answers each query in O(1)
/// instead of O(len^2).
pub struct CostKernel {
    cum: Cumsum2,
    diag_unit: bool,
    model: &'static str,
}

impl CostKernel {
    pub fn rbf(sig: &[f64], n: usize, d: usize, gamma: Option<f64>) -> (Self, f64) {
        let sq = |i: usize, j: usize| -> f64 {
            let mut acc = 0.0;
            for k in 0..d {
                let t = sig[i * d + k] - sig[j * d + k];
                acc += t * t;
            }
            acc
        };
        let gamma = match gamma {
            Some(g) => g,
            None => {
                // median of the condensed pairwise distance vector
                let mut cond = Vec::with_capacity(n * n.saturating_sub(1) / 2);
                for i in 0..n {
                    for j in (i + 1)..n {
                        cond.push(sq(i, j));
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
        };
        let cum = Cumsum2::build(n, |i, j| {
            if i == j {
                // `squareform` zeroes the diagonal before `exp`, so exp(0) = 1
                1.0
            } else {
                (-(gamma * sq(i, j)).clamp(1e-2, 1e2)).exp()
            }
        });
        (
            Self {
                cum,
                diag_unit: true,
                model: "rbf",
            },
            gamma,
        )
    }

    pub fn cosine(sig: &[f64], n: usize, d: usize) -> Self {
        let norms: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += sig[i * d + k] * sig[i * d + k];
                }
                acc.sqrt()
            })
            .collect();
        let cum = Cumsum2::build(n, |i, j| {
            // `squareform(1 - pdist(cosine))` has a zero diagonal
            if i == j {
                return 0.0;
            }
            let denom = norms[i] * norms[j];
            if denom == 0.0 {
                return 0.0;
            }
            let mut dot = 0.0;
            for k in 0..d {
                dot += sig[i * d + k] * sig[j * d + k];
            }
            dot / denom
        });
        Self {
            cum,
            diag_unit: false,
            model: "cosine",
        }
    }
}

impl Cost for CostKernel {
    fn min_size(&self) -> usize {
        1
    }
    fn model(&self) -> &'static str {
        self.model
    }
    #[inline]
    fn error(&self, start: usize, end: usize) -> f64 {
        let n = (end - start) as f64;
        let trace = if self.diag_unit { n } else { 0.0 };
        trace - self.cum.rect(start, end) / n
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
        // average ranks (1-based), matching scipy.stats.mstats.rankdata
        let mut ranks = vec![0.0; n * d];
        let mut idx: Vec<usize> = (0..n).collect();
        for j in 0..d {
            idx.sort_by(|&a, &b| {
                sig[a * d + j]
                    .partial_cmp(&sig[b * d + j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
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
/// NumPy's `lstsq` returns an *empty* residual array - summing to 0.0 - when
/// the system is not overdetermined or `X` is rank deficient. That quirk is
/// load-bearing for matching `ruptures`, so it is reproduced.
pub struct CostLstsq {
    xtx: PrefixOuter,
    xty: Vec<Prefix1>, // one per response column
    yty: Prefix1,
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
        let xtx = PrefixOuter::build(x, n, p);
        let mut xty = Vec::with_capacity(ny);
        for c in 0..ny {
            let mut prod = vec![0.0; n * p];
            for i in 0..n {
                let yc = y[i * ny + c];
                for k in 0..p {
                    prod[i * p + k] = x[i * p + k] * yc;
                }
            }
            xty.push(Prefix1::build(&prod, n, p));
        }
        let ysq: Vec<f64> = y.iter().map(|v| v * v).collect();
        Self {
            xtx,
            xty,
            yty: Prefix1::build(&ysq, n, ny),
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
        let mut gram = vec![0.0; p * p];
        self.xtx.seg_into(start, end, &mut gram);
        let mut chol = gram;
        if !linalg::cholesky(&mut chol, p) {
            // rank deficient: NumPy yields an empty residual, summing to 0.0
            return 0.0;
        }
        let mut total = 0.0;
        let mut rhs = vec![0.0; p];
        for c in 0..self.ny {
            for (k, slot) in rhs.iter_mut().enumerate() {
                *slot = self.xty[c].seg(start, end, k);
            }
            let mut beta = rhs.clone();
            linalg::cholesky_solve(&chol, p, &mut beta);
            let mut dot = 0.0;
            for k in 0..p {
                dot += beta[k] * rhs[k];
            }
            let rss = self.yty.seg(start, end, c) - dot;
            total += if rss > 0.0 { rss } else { 0.0 };
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
pub struct CostCLinear {
    sig: Vec<f64>,
    t1: Prefix1,
    t2: Prefix1,
    tw: Prefix1,
    d: usize,
}

impl CostCLinear {
    pub fn new(sig: &[f64], n: usize, d: usize) -> Self {
        let sq: Vec<f64> = sig.iter().map(|v| v * v).collect();
        let mut weighted = vec![0.0; n * d];
        for i in 0..n {
            for j in 0..d {
                weighted[i * d + j] = (i as f64) * sig[i * d + j];
            }
        }
        Self {
            sig: sig.to_vec(),
            t1: Prefix1::build(sig, n, d),
            t2: Prefix1::build(&sq, n, d),
            tw: Prefix1::build(&weighted, n, d),
            d,
        }
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
            total += sxx - 2.0 * (a * sxj + b * sx) + a * a * p2 + 2.0 * a * b * p1 + m * b * b;
        }
        total
    }
}
