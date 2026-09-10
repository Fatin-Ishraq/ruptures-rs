//! Small dense linear algebra on `d x d` matrices, where `d` is the signal
//! dimension (typically 1-20). Everything is row-major `Vec<f64>`.
//!
//! Nothing here may panic on hostile input. A NaN in the signal reaches these
//! routines, and a comparator built from `partial_cmp().unwrap()` — or one that
//! answers `Equal` for NaN, which is worse because it is not a total order —
//! aborts the interpreter through `pyo3`'s panic bridge. Every ordering below
//! goes through `f64::total_cmp`.

/// Total order on `f64`, safe for sorting in the presence of NaN.
#[inline]
pub fn total_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.total_cmp(b)
}

/// In-place Cholesky factorisation, `a` becomes lower-triangular `L` with
/// `L L^T == a`. Returns `false` if `a` is not positive definite.
pub fn cholesky(a: &mut [f64], d: usize) -> bool {
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= a[i * d + k] * a[j * d + k];
            }
            if i == j {
                // Rejects zero, negative and NaN pivots alike.
                if s.is_nan() || s <= 0.0 {
                    return false;
                }
                a[i * d + j] = s.sqrt();
            } else {
                a[i * d + j] = s / a[j * d + j];
            }
        }
        for j in (i + 1)..d {
            a[i * d + j] = 0.0;
        }
    }
    true
}

/// `log|det(a)|` for a symmetric positive-definite matrix via Cholesky.
/// Returns `None` when `a` is not positive definite.
pub fn logdet_spd(a: &mut [f64], d: usize) -> Option<f64> {
    if !cholesky(a, d) {
        return None;
    }
    let mut acc = 0.0;
    for i in 0..d {
        acc += a[i * d + i].ln();
    }
    Some(2.0 * acc)
}

/// Jacobi eigendecomposition of a symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors stored column-wise
/// in a row-major `d x d` array. Used for the pseudo-inverse in `CostRank` and
/// for the rank-revealing least-squares solve, mirroring `numpy.linalg.pinv`
/// and `numpy.linalg.lstsq` on a symmetric matrix.
pub fn jacobi_eigh(a_in: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    // Work on the matrix divided by its largest entry, and put the scale back
    // on the eigenvalues at the end. Eigenvectors are unchanged by this and
    // eigenvalues scale linearly, so it costs nothing — but the convergence
    // test below sums *squared* entries, and without the normalisation that sum
    // overflows for a Gram matrix of order 1e160 and underflows to zero for one
    // of order 1e-200. Both make the loop stop before anything is diagonalised,
    // and the caller gets the undiagonalised diagonal as its eigenvalues. A
    // regression design multiplied by 1e80 is not ill conditioned, and it must
    // not be treated as though it were.
    let scale = a_in.iter().fold(0.0f64, |m, v| {
        let x = v.abs();
        if x > m {
            x
        } else {
            m
        }
    });
    let scale = if scale > 0.0 && scale.is_finite() {
        scale
    } else {
        1.0
    };
    let mut a: Vec<f64> = a_in.iter().map(|v| v / scale).collect();
    let mut v = vec![0.0; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    // Convergence is measured *relative* to the matrix, not against a fixed
    // 1e-300. An absolute floor is unreachable for a Gram matrix with entries
    // of order 1e6 — its converged off-diagonals still sum to ~1e-20 — so every
    // call ran the full hundred sweeps. That is on the inner loop of the
    // `linear` and `ar` costs, once per dynamic-programming cell.
    let norm_sq: f64 = a.iter().map(|v| v * v).sum();
    let tol = f64::EPSILON * f64::EPSILON * norm_sq;
    let skip = f64::EPSILON * norm_sq.sqrt();
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..d {
            for j in (i + 1)..d {
                off += a[i * d + j] * a[i * d + j];
            }
        }
        // The NaN test is not redundant: a NaN off-diagonal is never `<=`
        // anything, so without it a NaN matrix runs every sweep to no purpose.
        if off.is_nan() || off <= tol {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.is_nan() || apq.abs() <= skip {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eig = (0..d).map(|i| a[i * d + i] * scale).collect::<Vec<_>>();
    (eig, v)
}

/// Residual sum of squares of `y ~ X` on the rows `start..end`, decided from
/// the singular values of the design itself rather than from its Gram matrix.
///
/// This is the slow, honest path. The normal equations square the condition
/// number of `X`, which costs the caller its ability to tell a design it cannot
/// resolve from one that is genuinely rank deficient — and the two have very
/// different right answers: a real deficiency means NumPy returns no residual
/// at all (which `ruptures` sums to `0.0`), while an unresolvable-but-full-rank
/// design has an ordinary residual that simply needs a better factorisation.
/// Reporting the first when it is the second hands the search a segment that
/// appears to fit perfectly.
///
/// One-sided Jacobi is used because it determines small singular values to high
/// *relative* accuracy, which is exactly the property being asked for here. The
/// cost is O(len * p^2) per sweep, so this runs only for the segments whose Gram
/// could not decide — see [`lstsq_factor`].
///
/// `x` is row-major `n x p`, `y` is row-major `n x ny`. Returns `None` when the
/// design is rank deficient by NumPy's rule.
pub fn lstsq_residual_svd(
    x: &[f64],
    y: &[f64],
    p: usize,
    ny: usize,
    start: usize,
    end: usize,
) -> Option<f64> {
    let m = end - start;
    if m <= p {
        return None;
    }
    if p == 0 {
        // Rank 0 of 0 columns is full rank, and the residual is all of y.
        let mut total = 0.0;
        for c in 0..ny {
            let mut acc = crate::prefix::Neumaier::new();
            for i in start..end {
                let v = y[i * ny + c];
                acc.add(v * v);
            }
            total += acc.value();
        }
        return Some(total);
    }

    // Column-major copy of the segment's design: the rotations below touch
    // whole columns, and Jacobi wants them contiguous.
    let mut cols = vec![0.0f64; m * p];
    for k in 0..p {
        for i in 0..m {
            cols[k * m + i] = x[(start + i) * p + k];
        }
    }

    let dot = |cols: &[f64], a: usize, b: usize| -> f64 {
        let mut acc = 0.0;
        for i in 0..m {
            acc += cols[a * m + i] * cols[b * m + i];
        }
        acc
    };

    for _sweep in 0..30 {
        let mut rotated = false;
        for a in 0..p {
            for b in (a + 1)..p {
                let alpha = dot(&cols, a, a);
                let beta = dot(&cols, b, b);
                let gamma = dot(&cols, a, b);
                if gamma == 0.0 || !gamma.is_finite() {
                    continue;
                }
                if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let ca = cols[a * m + i];
                    let cb = cols[b * m + i];
                    cols[a * m + i] = c * ca - s * cb;
                    cols[b * m + i] = s * ca + c * cb;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }

    // Singular values are the norms of the now-orthogonal columns.
    let mut sigma = vec![0.0f64; p];
    for (k, slot) in sigma.iter_mut().enumerate() {
        *slot = dot(&cols, k, k).sqrt();
    }
    let smax = sigma
        .iter()
        .fold(0.0f64, |acc, &v| if v > acc { v } else { acc });
    if smax.is_nan() {
        return Some(f64::NAN);
    }
    if smax <= 0.0 {
        return None;
    }
    // `numpy.linalg.lstsq(rcond=None)`: keep singular values above
    // `max(m, p) * eps * sigma_max`, and report no residual unless every column
    // survives.
    let cutoff = (std::cmp::max(m, p) as f64) * f64::EPSILON * smax;
    for &sv in &sigma {
        if sv.is_nan() || sv <= cutoff {
            return None;
        }
    }

    let mut total = 0.0;
    for c in 0..ny {
        let mut yy = crate::prefix::Neumaier::new();
        for i in start..end {
            let v = y[i * ny + c];
            yy.add(v * v);
        }
        let mut explained = crate::prefix::Neumaier::new();
        for k in 0..p {
            let mut proj = 0.0;
            for i in 0..m {
                proj += cols[k * m + i] * y[(start + i) * ny + c];
            }
            let u = proj / sigma[k];
            explained.add(u * u);
        }
        let rss = yy.value() - explained.value();
        total += if rss < 0.0 { 0.0 } else { rss };
    }
    Some(total)
}

/// Moore-Penrose pseudo-inverse of a symmetric matrix.
///
/// Matches `numpy.linalg.pinv`'s default cutoff: singular values below
/// `rcond * max(sigma)` with `rcond = max(d, d) * eps` are treated as zero.
pub fn pinv_sym(a: &[f64], d: usize) -> Vec<f64> {
    let (eig, v) = jacobi_eigh(a, d);
    let smax = eig.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    let cutoff = (d as f64) * f64::EPSILON * smax;
    let mut out = vec![0.0; d * d];
    for k in 0..d {
        if eig[k].abs() <= cutoff {
            continue;
        }
        let inv = 1.0 / eig[k];
        for i in 0..d {
            let vik = v[i * d + k];
            if vik == 0.0 {
                continue;
            }
            for j in 0..d {
                out[i * d + j] += vik * inv * v[j * d + k];
            }
        }
    }
    out
}

/// A factorised normal-equations system, ready to solve for many right-hand
/// sides.
pub struct LstsqFactor {
    eig: Vec<f64>,
    vec: Vec<f64>,
    p: usize,
}

/// What the normal equations were able to establish about a design.
pub enum Lstsq {
    /// NumPy returns an empty residual whatever the data is: the system is not
    /// overdetermined. `ruptures` sums that to `0.0`.
    NoResidual,
    /// The Gram matrix cannot tell a rank deficiency from an ill-conditioned
    /// design, because it has squared away the difference. Ask the design.
    Unresolvable,
    /// Well enough conditioned that the fast path is also the accurate one.
    Ready(LstsqFactor),
}

/// Below this ratio of smallest to largest eigenvalue, the normal equations
/// have lost more than half their digits — `eps * cond(X)^2` is then worse than
/// `1e-4` relative — and the answer, if there is one, has to come from the
/// design itself.
///
/// `eps^(3/4)` puts the boundary at `cond(X)` of roughly `7e5`. Above that
/// ratio the Gram still delivers four correct digits or better, which is what
/// buys the O(1) query; below it, correctness is worth O(len * p^2).
const GRAM_RESOLVABLE: f64 = 1.8e-12;

/// Factorise `X^T X` for a design with `n_rows` rows and `p` columns.
///
/// NumPy returns an *empty* residual array — which `ruptures` then `.sum()`s to
/// `0.0` — unless the system is strictly overdetermined *and* the design has
/// full column rank. Its rank test is on the singular values of `X`: keep those
/// above `max(n_rows, p) * eps * sigma_max`.
///
/// This sees `X^T X`, not `X`, and that costs a decision it cannot recover.
/// The eigenvalues are the *squared* singular values, so a rank deficiency that
/// shows in `X` at `1e-17` shows in the Gram at `1e-34` — far below the `1e-16`
/// floor at which an eigenvalue of the Gram means anything at all. Testing the
/// square roots against NumPy's cutoff therefore certifies full rank for a
/// design of two identical columns, and hands back a residual that is pure
/// rounding noise.
///
/// So a Gram that cannot decide says so, rather than guessing. It used to guess
/// "deficient", and a deficient design scores `0.0` — a perfect fit, and the
/// most attractive segment there is. Now [`Lstsq::Unresolvable`] sends the
/// caller to [`lstsq_residual_svd`], which factorises the design and applies
/// NumPy's rule to the singular values NumPy would have seen.
pub fn lstsq_factor(gram: &[f64], p: usize, n_rows: usize) -> Lstsq {
    if n_rows <= p {
        return Lstsq::NoResidual;
    }
    if p == 0 {
        // A design with no columns: NumPy reports rank 0 == p, so the residual
        // is the whole of `y^T y`.
        return Lstsq::Ready(LstsqFactor {
            eig: Vec::new(),
            vec: Vec::new(),
            p: 0,
        });
    }
    let (eig, vec) = jacobi_eigh(gram, p);
    let mut lmax = 0.0f64;
    let mut lmin = f64::INFINITY;
    for &l in &eig {
        if l.is_nan() {
            return Lstsq::Unresolvable;
        }
        if l > lmax {
            lmax = l;
        }
        if l < lmin {
            lmin = l;
        }
    }
    if lmax <= 0.0 || !lmax.is_finite() {
        return Lstsq::Unresolvable;
    }
    let cutoff = (std::cmp::max(n_rows, p) as f64) * f64::EPSILON * lmax;
    if lmin <= cutoff || lmin <= GRAM_RESOLVABLE * lmax {
        return Lstsq::Unresolvable;
    }
    Lstsq::Ready(LstsqFactor { eig, vec, p })
}

impl LstsqFactor {
    /// Solve for one right-hand side, writing the coefficients into `out`.
    pub fn solve_into(&self, rhs: &[f64], out: &mut [f64]) {
        out[..self.p].iter_mut().for_each(|v| *v = 0.0);
        for k in 0..self.p {
            let mut dot = 0.0;
            for (i, &r) in rhs.iter().enumerate().take(self.p) {
                dot += self.vec[i * self.p + k] * r;
            }
            let coef = dot / self.eig[k];
            for (i, slot) in out.iter_mut().enumerate().take(self.p) {
                *slot += coef * self.vec[i * self.p + k];
            }
        }
    }
}

/// Median of a slice, matching `numpy.median` (mean of the two central order
/// statistics for even length, and NaN if any element is NaN). Reorders `buf`.
pub fn median_inplace(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    if n == 0 {
        return f64::NAN;
    }
    // `numpy.median` propagates NaN; checking up front also keeps the
    // selection below on a NaN-free slice.
    if buf.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, hi, _) = buf.select_nth_unstable_by(mid, total_cmp);
    let upper = *hi;
    if n % 2 == 1 {
        upper
    } else {
        let lower = buf[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_converges_on_a_large_scale_matrix() {
        // Entries of order 1e6: an absolute convergence floor never fires here.
        let a = [4.0e6, 2.0e6, 2.0e6, 3.0e6];
        let (eig, _) = jacobi_eigh(&a, 2);
        let mut sorted = eig.clone();
        sorted.sort_by(total_cmp);
        // trace and determinant are invariant under the rotations
        assert!((sorted[0] + sorted[1] - 7.0e6).abs() < 1.0);
        assert!((sorted[0] * sorted[1] - 8.0e12).abs() < 1.0e6);
    }

    #[test]
    fn median_matches_numpy_conventions() {
        let mut odd = [3.0, 1.0, 2.0];
        assert_eq!(median_inplace(&mut odd), 2.0);
        let mut even = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median_inplace(&mut even), 2.5);
        let mut empty: [f64; 0] = [];
        assert!(median_inplace(&mut empty).is_nan());
    }

    #[test]
    fn median_propagates_nan_without_panicking() {
        let mut buf = [1.0, f64::NAN, 2.0, 0.5];
        assert!(median_inplace(&mut buf).is_nan());
    }

    #[test]
    fn lstsq_rejects_underdetermined_and_defers_on_collinear() {
        // 1x1 gram, one row, one column: not overdetermined, and no amount of
        // looking at the design changes that.
        assert!(matches!(lstsq_factor(&[1.0], 1, 1), Lstsq::NoResidual));
        // Exactly collinear 2-column design: the Gram cannot tell this from an
        // ill-conditioned one, and says so rather than guessing.
        let gram = [1.0, 2.0, 2.0, 4.0];
        assert!(matches!(lstsq_factor(&gram, 2, 50), Lstsq::Unresolvable));
    }

    #[test]
    fn lstsq_accepts_a_design_it_can_actually_resolve() {
        // Eigenvalues a factor of 1e-7 apart: well inside what the normal
        // equations can resolve, so this is fitted rather than written off.
        let gram = [1.0, 0.0, 0.0, 1e-7];
        let f = match lstsq_factor(&gram, 2, 50) {
            Lstsq::Ready(f) => f,
            _ => panic!("full rank"),
        };
        let mut beta = [0.0; 2];
        f.solve_into(&[1.0, 1.0], &mut beta);
        assert!(beta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_eigensolver_is_invariant_under_scaling() {
        // A well-conditioned matrix at absurd scales: the eigenvalues must
        // simply scale with it. Before the normalisation, the convergence test
        // overflowed at the top and underflowed at the bottom, and the solver
        // returned the undiagonalised diagonal.
        for scale in [1e80f64, 1e-100, 1.0] {
            let a = [4.0 * scale, 2.0 * scale, 2.0 * scale, 3.0 * scale];
            let (mut eig, _) = jacobi_eigh(&a, 2);
            eig.sort_by(total_cmp);
            let trace = eig[0] + eig[1];
            assert!(
                (trace / (7.0 * scale) - 1.0).abs() < 1e-12,
                "scale {scale}: {eig:?}"
            );
            // Off-diagonal must actually have been eliminated: for this matrix
            // the eigenvalues are (7 +/- sqrt(17)) / 2 times the scale.
            let expect = (7.0 - 17.0f64.sqrt()) / 2.0 * scale;
            assert!((eig[0] / expect - 1.0).abs() < 1e-12, "scale {scale}");
        }
    }

    #[test]
    fn svd_residual_matches_a_hand_computed_fit() {
        // y = 2 x1 - x2 exactly on a well-conditioned design: zero residual.
        let n = 20;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = (i as f64).sin();
            let b = (i as f64 * 0.37).cos();
            x[i * 2] = a;
            x[i * 2 + 1] = b;
            y[i] = 2.0 * a - b;
        }
        let r = lstsq_residual_svd(&x, &y, 2, 1, 0, n).expect("full rank");
        // `||y||^2` is of order 40 here, so a residual of zero comes back as a
        // few ulp of that; the point is that it is not of order one.
        assert!(r < 1e-12, "{r}");
        // Exactly collinear columns are rank deficient at any scale.
        for i in 0..n {
            x[i * 2 + 1] = 3.0 * x[i * 2];
        }
        assert!(lstsq_residual_svd(&x, &y, 2, 1, 0, n).is_none());
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        let mut good = [4.0, 2.0, 2.0, 3.0];
        assert!(cholesky(&mut good, 2));
        let mut singular = [1.0, 1.0, 1.0, 1.0];
        assert!(!cholesky(&mut singular, 2));
        let mut nan = [f64::NAN, 0.0, 0.0, 1.0];
        assert!(!cholesky(&mut nan, 2));
    }

    #[test]
    fn logdet_matches_a_hand_computed_determinant() {
        // det([[4, 2], [2, 3]]) = 8, ln 8 = 2.0794415...
        let mut a = [4.0, 2.0, 2.0, 3.0];
        let v = logdet_spd(&mut a, 2).expect("positive definite");
        assert!((v - 8.0f64.ln()).abs() < 1e-12);
    }
}
