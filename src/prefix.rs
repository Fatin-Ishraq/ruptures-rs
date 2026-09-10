//! Numerically stable prefix-sum accumulators.
//!
//! Every O(1) segment cost in this crate is a difference of prefix sums. That is
//! what buys the speedup, but the naive form is also the classic numerical trap:
//! `sum(x^2) - sum(x)^2 / n` is the textbook one-pass variance, which suffers
//! catastrophic cancellation when the mean is large relative to the spread.
//! NumPy's `.var()` is two-pass and therefore stable, so a naive port silently
//! disagrees with the reference — or returns negative variances.
//!
//! Three defences, applied together:
//!
//! 1. **Centering.** The signal is shifted by its per-dimension global mean
//!    before any prefix array is built. Sums of squared deviations are invariant
//!    under this shift, so the answer is unchanged while the cancellation
//!    becomes negligible for realistic data.
//!
//!    It is a mitigation, not a bound. A segment far from the *global* mean —
//!    a small late regime after a much larger one — is still differenced out of
//!    prefix values that dwarf it, and no amount of care in the accumulation
//!    recovers digits the prefix array never had room to store. That case is
//!    detected and recomputed exactly; see [`crate::stable`].
//! 2. **Detrending**, for the costs that are invariant under it. `clinear`
//!    compares a signal to a straight line, so it is invariant under
//!    subtracting *any* affine function of the sample index — and a trending
//!    signal, which is the whole point of that cost, is not helped by centring
//!    alone.
//! 3. **Compensated accumulation.** Prefix arrays are built with Neumaier
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
        Self {
            sum: 0.0,
            comp: 0.0,
        }
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

    /// Sum of column `j` over rows `start..end`.
    #[inline]
    pub fn seg(&self, start: usize, end: usize, j: usize) -> f64 {
        self.data[end * self.d + j] - self.data[start * self.d + j]
    }

    /// The accumulated prefix value itself, `sum(column j over rows 0..i)`.
    ///
    /// A segment sum is a difference of two of these, so its absolute rounding
    /// error is set by *their* magnitude rather than by its own — which is the
    /// whole of the cancellation problem `stable` exists to detect.
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.d + j]
    }
}

/// Prefix sums of the per-row outer products `x x^T`, stored as the full `d*d`
/// block per row (redundant across the diagonal, but branch-free to index).
///
/// This is the crate's one quadratic-in-`d` structure: `(n + 1) * d^2` doubles.
/// Callers are expected to have checked that against a memory budget first —
/// see `check_outer_memory` in the binding layer.
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
        for (k, slot) in out.iter_mut().enumerate().take(self.dd) {
            *slot = self.data[b + k] - self.data[a + k];
        }
    }
}

/// Per-column means of a row-major `n x d` matrix, compensated.
pub fn column_means(sig: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0; d];
    if n == 0 {
        return out;
    }
    for (j, slot) in out.iter_mut().enumerate() {
        let mut acc = Neumaier::new();
        for i in 0..n {
            acc.add(sig[i * d + j]);
        }
        *slot = acc.value() / n as f64;
    }
    out
}

/// Subtract a per-column offset from a row-major `n x d` matrix.
pub fn subtract_means(sig: &[f64], n: usize, d: usize, means: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            out[i * d + j] = sig[i * d + j] - means[j];
        }
    }
    out
}

/// Centre a row-major `n x d` signal by its per-column mean.
///
/// Returns the centred copy. Sums of squared deviations, covariances and
/// least-squares residuals are all invariant under this shift, so it costs
/// nothing but removes the dominant source of cancellation.
pub fn center(sig: &[f64], n: usize, d: usize) -> Vec<f64> {
    subtract_means(sig, n, d, &column_means(sig, n, d))
}

/// Subtract one scalar — the mean of every value in the matrix — from all of it.
///
/// For a cost whose model carries an intercept, shifting the *whole* signal by
/// a constant is exactly absorbed and the residual is unchanged. Shifting each
/// column by its own mean is not, once the design mixes columns: `ar` builds
/// its lags by walking the flattened buffer, so a row can span a column
/// boundary, and per-column offsets do not cancel across one.
pub fn center_scalar(sig: &[f64], n: usize, d: usize) -> Vec<f64> {
    let total = n * d;
    if total == 0 {
        return Vec::new();
    }
    let mut acc = Neumaier::new();
    for &v in sig.iter().take(total) {
        acc.add(v);
    }
    let mean = acc.value() / total as f64;
    sig.iter().take(total).map(|v| v - mean).collect()
}

/// Subtract the per-column global least-squares line in the sample index.
///
/// For a cost that compares the signal to an affine function of the index —
/// `clinear` — the residual is unchanged by this, because the data and its
/// approximation move together. Centring is the special case where the fitted
/// slope is zero, and it is not enough on a trending signal: the values still
/// grow without bound along the segment, and the expanded O(1) form then
/// cancels a large `m * intercept^2` term against the rest.
pub fn detrend_affine(sig: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * d];
    if n == 0 {
        return out;
    }
    let nf = n as f64;
    let mean_i = (nf - 1.0) / 2.0;
    // sum (i - mean_i)^2 for i in 0..n
    let sii = nf * (nf * nf - 1.0) / 12.0;
    let means = column_means(sig, n, d);
    for (j, &mean_x) in means.iter().enumerate() {
        let slope = if sii > 0.0 {
            let mut cross = Neumaier::new();
            for i in 0..n {
                cross.add((i as f64 - mean_i) * (sig[i * d + j] - mean_x));
            }
            cross.value() / sii
        } else {
            0.0
        };
        let intercept = mean_x - slope * mean_i;
        for i in 0..n {
            out[i * d + j] = sig[i * d + j] - (slope * i as f64 + intercept);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_beats_naive_summation() {
        let mut acc = Neumaier::new();
        acc.add(1.0);
        for _ in 0..10_000 {
            acc.add(1e-16);
        }
        assert!(acc.value() > 1.0);
    }

    #[test]
    fn detrend_flattens_a_ramp() {
        let n = 500;
        let sig: Vec<f64> = (0..n).map(|i| 7.0 * i as f64 - 3.0).collect();
        let out = detrend_affine(&sig, n, 1);
        assert!(out.iter().all(|v| v.abs() < 1e-9), "ramp not removed");
    }

    #[test]
    fn detrend_is_centering_on_a_flat_signal() {
        let sig = vec![4.0; 32];
        let out = detrend_affine(&sig, 32, 1);
        assert!(out.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn cumsum2_block_sums() {
        // K[i][j] = i + j over a 4x4 grid.
        let c = Cumsum2::build(4, |i, j| (i + j) as f64);
        // block [1,3) x [1,3) = (1+1)+(1+2)+(2+1)+(2+2) = 12
        assert_eq!(c.rect(1, 3), 12.0);
        assert_eq!(c.rect(0, 0), 0.0);
    }
}
