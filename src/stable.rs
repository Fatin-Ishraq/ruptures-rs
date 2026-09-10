//! Cancellation detection, and exact recomputation for the segments that need
//! it.
//!
//! Every O(1) cost here is a difference of prefix sums, and a difference of
//! prefix sums can only carry the digits its operands had room for. Centring
//! the signal first (see [`crate::prefix`]) moves the operands close to zero
//! for data that lives at one level, but it does nothing for data that lives at
//! two: on `[1e10] * 50 ++ [0, 1, ..., 49]` the prefix of squares reaches
//! `2.5e21`, whose last representable bit is worth about `5e5` — and the cost
//! of the second regime's interior is `82.5`. The subtraction returns zero, and
//! clamping a negative result to zero hides the failure behind a number that
//! reads as a perfect segment.
//!
//! So the fast form is still computed, but it is *checked*. Each cost estimates
//! the rounding error of its own prefix differences from the magnitude of the
//! prefixes involved; a result that is not comfortably larger than that error
//! has no significant digits left, and the segment is recomputed directly from
//! the stored signal by the same two-pass reduction NumPy uses. The check is a
//! few flops on numbers already in hand, and on data whose regimes are within
//! `1 / sqrt(eps)` of each other in scale — which is to say all of it — it never
//! fires.
//!
//! A NaN is routed the same way, and gets a second benefit. A NaN anywhere in
//! the signal poisons every prefix value after it, so a prefix difference
//! reports NaN for segments that do not contain the NaN at all, where `ruptures`
//! (which reduces each segment on its own) returns an ordinary number. Falling
//! back on NaN restores that: the recomputation sees only the segment, so the
//! NaN propagates exactly as far as it should and no further.

use crate::prefix::Neumaier;

/// How many times its own estimated rounding error a result has to be before it
/// is believed. Four bits of margin: enough that ordinary accumulated slop
/// never trips the guard, small enough that a result which has lost everything
/// always does.
const GUARD: f64 = 16.0;

/// Whether `value` is large enough beside the rounding error `err` of the
/// prefix differences that produced it to be worth keeping.
///
/// False for a NaN `value` or a NaN `err`, which is deliberate: both mean the
/// fast path cannot be trusted and the caller should recompute.
#[inline]
pub fn trustworthy(value: f64, err: f64) -> bool {
    value >= GUARD * err
}

/// Rounding error of `s2 - s1 * s1 / len`, where `s1` and `s2` are differences
/// of prefix arrays whose values at the segment ends are `p1_lo`, `p1_hi` and
/// `p2_hi`.
///
/// Differencing two doubles of magnitude `P` costs about `eps * P` absolutely,
/// however carefully they were accumulated. `p2_hi` is a prefix of squares and
/// so is both non-negative and the larger end; `p1` can cancel, so both ends
/// are considered.
#[inline]
pub fn ssd_error(s1: f64, len: f64, p1_lo: f64, p1_hi: f64, p2_hi: f64) -> f64 {
    let p1 = p1_lo.abs().max(p1_hi.abs());
    f64::EPSILON * (p2_hi.abs() + 2.0 * s1.abs() * p1 / len)
}

/// Where each run of exactly equal values begins.
///
/// A segment that is constant has a sum of squared deviations of exactly zero,
/// and the guard above always sends it to the slow path: a true value of zero
/// is never sixteen times any positive error estimate. That is the right
/// decision for a segment that has merely *cancelled* to nothing, and pure
/// waste for one that is genuinely flat — and a signal built of flat plateaus
/// makes every interior segment genuinely flat, which turned a whole dynamic
/// program O(len) per cell.
///
/// So constancy is answered in O(1) instead, from one `u32` per value: the
/// first index of the run of equal values ending at `i`. A NaN starts a new run
/// on arrival, because `NaN == NaN` is false, so a run is always NaN-free and a
/// constant segment can be scored zero without looking at it.
pub struct ConstantRuns {
    start: Vec<u32>,
    d: usize,
}

impl ConstantRuns {
    pub fn build(raw: &[f64], n: usize, d: usize) -> Self {
        let mut start = vec![0u32; n * d];
        for j in 0..d {
            let mut run = 0u32;
            for i in 0..n {
                if i > 0 && raw[i * d + j] != raw[(i - 1) * d + j] {
                    run = i as u32;
                }
                start[i * d + j] = run;
            }
        }
        Self { start, d }
    }

    /// Whether column `j` is exactly constant over `start..end`.
    #[inline]
    pub fn is_constant(&self, start: usize, end: usize, j: usize) -> bool {
        if end <= start + 1 {
            return true;
        }
        (self.start[(end - 1) * self.d + j] as usize) <= start
    }

    /// Whether every column is exactly constant over `start..end`.
    #[inline]
    pub fn all_constant(&self, start: usize, end: usize) -> bool {
        (0..self.d).all(|j| self.is_constant(start, end, j))
    }
}

/// Exact mean of column `j` of a row-major `n x d` matrix over `start..end`.
#[inline]
pub fn segment_mean(raw: &[f64], d: usize, j: usize, start: usize, end: usize) -> f64 {
    let len = end - start;
    if len == 0 {
        return f64::NAN;
    }
    let mut acc = Neumaier::new();
    for i in start..end {
        acc.add(raw[i * d + j]);
    }
    acc.value() / len as f64
}

/// Exact sum of squared deviations from the segment mean, for column `j` of a
/// row-major `n x d` matrix over `start..end`.
///
/// This is `numpy.var(ddof=0) * len` computed the way NumPy computes it: mean
/// first, then a second pass over the deviations. It is O(len) rather than
/// O(1), which is the price of an answer at all.
pub fn segment_ssd(raw: &[f64], d: usize, j: usize, start: usize, end: usize) -> f64 {
    let len = end - start;
    if len == 0 {
        return 0.0;
    }
    let mean = segment_mean(raw, d, j, start, end);
    let mut acc = Neumaier::new();
    for i in start..end {
        let t = raw[i * d + j] - mean;
        acc.add(t * t);
    }
    let v = acc.value();
    // Two-pass leaves no room for a negative sum of squares except through a
    // NaN, which must survive.
    if v < 0.0 {
        0.0
    } else {
        v
    }
}

/// Exact scatter matrix `sum_i (x_i - mean)(x_i - mean)^T` over `start..end`,
/// written row-major into `out` (length `d * d`).
pub fn segment_scatter(raw: &[f64], d: usize, start: usize, end: usize, out: &mut [f64]) {
    for slot in out.iter_mut().take(d * d) {
        *slot = 0.0;
    }
    let len = end - start;
    if len == 0 {
        return;
    }
    let mut mean = vec![0.0; d];
    for (j, m) in mean.iter_mut().enumerate() {
        *m = segment_mean(raw, d, j, start, end);
    }
    let mut acc = vec![Neumaier::new(); d * d];
    let mut dev = vec![0.0; d];
    for i in start..end {
        for (j, slot) in dev.iter_mut().enumerate() {
            *slot = raw[i * d + j] - mean[j];
        }
        for a in 0..d {
            for b in 0..d {
                acc[a * d + b].add(dev[a] * dev[b]);
            }
        }
    }
    for (slot, a) in out.iter_mut().take(d * d).zip(acc.iter()) {
        *slot = a.value();
    }
}

/// Rounding error of a scatter entry `(a, b)` recovered by prefix differencing.
///
/// The prefix of `x_a * x_b` can cancel, so its magnitude is bounded through
/// Cauchy-Schwarz by the prefixes of the squares, which cannot.
#[inline]
pub fn scatter_error(p2_a: f64, p2_b: f64, s1_a: f64, s1_b: f64, len: f64) -> f64 {
    f64::EPSILON * ((p2_a.abs() * p2_b.abs()).sqrt() + (s1_a * s1_b).abs() / len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case that motivates the module: a small regime beside a huge one.
    #[test]
    fn exact_recomputation_recovers_a_swamped_segment() {
        let mut raw = vec![1e10; 50];
        raw.extend((0..50).map(|i| i as f64));
        // var([0..9]) * 10 = 82.5
        let v = segment_ssd(&raw, 1, 0, 50, 60);
        assert!((v - 82.5).abs() < 1e-9, "{v}");
    }

    #[test]
    fn the_guard_rejects_a_result_with_no_digits_left() {
        // s2 ~ 2.5e21: its last bit is worth about 5e5, so 82.5 is noise.
        let err = ssd_error(0.0, 10.0, 0.0, 0.0, 2.5e21);
        assert!(!trustworthy(82.5, err));
        // The same 82.5 out of prefixes of order 1e3 is entirely real.
        assert!(trustworthy(82.5, ssd_error(0.0, 10.0, 0.0, 0.0, 1.0e3)));
    }

    #[test]
    fn a_nan_is_never_trustworthy() {
        assert!(!trustworthy(f64::NAN, 0.0));
        assert!(!trustworthy(1.0, f64::NAN));
    }

    #[test]
    fn constant_runs_answer_flatness_in_constant_time() {
        let raw = [1.0, 1.0, 1.0, 2.0, 2.0, f64::NAN, f64::NAN];
        let runs = ConstantRuns::build(&raw, 7, 1);
        assert!(runs.is_constant(0, 3, 0));
        assert!(!runs.is_constant(0, 4, 0));
        assert!(runs.is_constant(3, 5, 0));
        // A NaN never joins a run, so a "constant" segment is always NaN-free.
        assert!(!runs.is_constant(5, 7, 0));
    }

    #[test]
    fn scatter_matches_a_hand_computed_block() {
        // Two columns: [1,2,3] and [4,6,8]. Deviations (-1,0,1), (-2,0,2).
        let raw = [1.0, 4.0, 2.0, 6.0, 3.0, 8.0];
        let mut out = [0.0; 4];
        segment_scatter(&raw, 2, 0, 3, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
        assert!((out[2] - 4.0).abs() < 1e-12);
        assert!((out[3] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn a_nan_outside_the_segment_does_not_reach_it() {
        let raw = [f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(segment_ssd(&raw, 1, 0, 2, 6).is_finite());
        assert!(segment_ssd(&raw, 1, 0, 0, 3).is_nan());
    }
}
