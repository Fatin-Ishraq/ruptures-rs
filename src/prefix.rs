//! Numerically stable prefix-sum accumulators.
//!
//! Every O(1) segment cost in this crate is a difference of prefix sums. That is
//! what buys the speedup, but the naive form is also the classic numerical trap:
//! `sum(x^2) - sum(x)^2 / n` is the textbook one-pass variance, which suffers
//! catastrophic cancellation when the mean is large relative to the spread.
//! NumPy's `.var()` is two-pass and therefore stable, so a naive port silently
//! disagrees with the reference — or returns negative variances.
//!
//! Two defences, applied together:
//!
//! 1. **Centering.** The signal is shifted by its per-dimension global mean
//!    before any prefix array is built. Sums of squared deviations are invariant
//!    under this shift, so the answer is unchanged while the cancellation
//!    becomes negligible for realistic data.
//! 2. **Compensated accumulation.** Prefix arrays are built with Neumaier
//!    summation, bounding accumulation error at roughly one ulp regardless of
//!    `n`, instead of the O(n * eps) drift of a plain running sum.

/// Neumaier ("improved Kahan") compensated running sum.
///
/// Unlike plain Kahan, this variant is also correct when the incoming term is
/// larger in magnitude than the accumulator, which happens routinely on signals
/// that start near zero and grow.
#[derive(Clone, Copy, Debug, Default)]
pub struct Neumaier {
    sum: f64,
    comp: f64,
}

impl Neumaier {
    #[inline]
    pub fn new() -> Self {
        Self { sum: 0.0, comp: 0.0 }
    }

    #[inline]
    pub fn add(&mut self, v: f64) {
        let t = self.sum + v;
        if self.sum.abs() >= v.abs() {
            self.comp += (self.sum - t) + v;
        } else {
            self.comp += (v - t) + self.sum;
        }
        self.sum = t;
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.sum + self.comp
    }
}

/// Column-wise prefix sums of an `n x d` row-major matrix.
///
/// `get(i, j)` is the sum of column `j` over rows `0..i`, so a segment sum is
/// `get(end, j) - get(start, j)`.
pub struct Prefix1 {
    data: Vec<f64>,
    d: usize,
}

impl Prefix1 {
    pub fn build(sig: &[f64], n: usize, d: usize) -> Self {
        let mut data = vec![0.0; (n + 1) * d];
        let mut acc = vec![Neumaier::new(); d];
        for i in 0..n {
            for j in 0..d {
                acc[j].add(sig[i * d + j]);
                data[(i + 1) * d + j] = acc[j].value();
            }
        }
        Self { data, d }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.d + j]
    }

    /// Sum of column `j` over rows `start..end`.
    #[inline]
    pub fn seg(&self, start: usize, end: usize, j: usize) -> f64 {
        self.data[end * self.d + j] - self.data[start * self.d + j]
    }
}

/// Prefix sums of the per-row outer products `x x^T`, stored as the full `d*d`
/// block per row (redundant across the diagonal, but branch-free to index).
pub struct PrefixOuter {
    data: Vec<f64>,
    dd: usize,
}

impl PrefixOuter {
    pub fn build(sig: &[f64], n: usize, d: usize) -> Self {
        let dd = d * d;
        let mut data = vec![0.0; (n + 1) * dd];
        let mut acc = vec![Neumaier::new(); dd];
        for i in 0..n {
            let row = &sig[i * d..(i + 1) * d];
            for a in 0..d {
                for b in 0..d {
                    acc[a * d + b].add(row[a] * row[b]);
                    data[(i + 1) * dd + a * d + b] = acc[a * d + b].value();
                }
            }
        }
        Self { data, dd }
    }

    /// Segment sum of outer products, written into `out` (length `d*d`).
    #[inline]
    pub fn seg_into(&self, start: usize, end: usize, out: &mut [f64]) {
        let (a, b) = (start * self.dd, end * self.dd);
        for k in 0..self.dd {
            out[k] = self.data[b + k] - self.data[a + k];
        }
    }
}

/// Centre a row-major `n x d` signal by its per-column mean.
///
/// Returns the centred copy. Sums of squared deviations, covariances and
/// least-squares residuals are all invariant under this shift, so it costs
/// nothing but removes the dominant source of cancellation.
pub fn center(sig: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut means = vec![0.0; d];
    for j in 0..d {
        let mut acc = Neumaier::new();
        for i in 0..n {
            acc.add(sig[i * d + j]);
        }
        means[j] = acc.value() / n as f64;
    }
    let mut out = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            out[i * d + j] = sig[i * d + j] - means[j];
        }
    }
    out
}

/// 2-D cumulative sum of a symmetric `n x n` Gram matrix.
///
/// `rect(a, b)` returns the sum of the `[a..b) x [a..b)` block in O(1).
///
/// This is the same O(n^2) memory that `ruptures` already spends storing the
/// Gram matrix itself, but it turns the kernel costs from O(len^2) per query
/// into O(1) — so it is strictly cheaper in time at equal memory.
pub struct Cumsum2 {
    data: Vec<f64>,
    stride: usize,
}

impl Cumsum2 {
    /// Build from a closure yielding `gram[i][j]`, consuming the matrix row by
    /// row so the dense Gram never has to exist alongside the cumulative sum.
    pub fn build<F: FnMut(usize, usize) -> f64>(n: usize, mut f: F) -> Self {
        let stride = n + 1;
        let mut data = vec![0.0; stride * stride];
        for i in 0..n {
            let mut row_acc = Neumaier::new();
            for j in 0..n {
                row_acc.add(f(i, j));
                data[(i + 1) * stride + (j + 1)] = row_acc.value() + data[i * stride + (j + 1)];
            }
        }
        Self { data, stride }
    }

    /// Sum over the square block `[a..b) x [a..b)`.
    #[inline]
    pub fn rect(&self, a: usize, b: usize) -> f64 {
        let s = self.stride;
        self.data[b * s + b] - self.data[a * s + b] - self.data[b * s + a] + self.data[a * s + a]
    }
}
