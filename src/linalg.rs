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
    let mut a = a_in.to_vec();
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
    let eig = (0..d).map(|i| a[i * d + i]).collect::<Vec<_>>();
    (eig, v)
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
/// sides, or `None` when NumPy's `lstsq` would report no residual at all.
pub struct LstsqFactor {
    eig: Vec<f64>,
    vec: Vec<f64>,
    p: usize,
}

/// Factorise `X^T X` for a design with `n_rows` rows and `p` columns, deciding
/// rank the way `numpy.linalg.lstsq(rcond=None)` would — as closely as the
/// normal equations allow.
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
/// So the cutoff is applied to the eigenvalues directly: anything below
/// `max(n_rows, p) * eps * lambda_max` is unresolvable, and a design that
/// cannot be certified full rank is reported as deficient. Exact collinearity
/// is then caught, and the cost is the mirror image — a design whose condition
/// number exceeds roughly `1 / sqrt(n_rows * eps)`, about `1e7`, is called
/// deficient here while NumPy still fits it. That boundary is inherent to
/// solving through the normal equations, and it is documented rather than
/// hidden.
pub fn lstsq_factor(gram: &[f64], p: usize, n_rows: usize) -> Option<LstsqFactor> {
    if n_rows <= p {
        return None;
    }
    if p == 0 {
        // A design with no columns: NumPy reports rank 0 == p, so the residual
        // is the whole of `y^T y`.
        return Some(LstsqFactor {
            eig: Vec::new(),
            vec: Vec::new(),
            p: 0,
        });
    }
    let (eig, vec) = jacobi_eigh(gram, p);
    let mut lmax = 0.0f64;
    for &l in &eig {
        if l > lmax {
            lmax = l;
        }
    }
    if lmax.is_nan() || lmax <= 0.0 || !lmax.is_finite() {
        return None;
    }
    let cutoff = (std::cmp::max(n_rows, p) as f64) * f64::EPSILON * lmax;
    for &l in &eig {
        if l.is_nan() || l <= cutoff {
            return None; // rank deficient: NumPy yields an empty residual
        }
    }
    Some(LstsqFactor { eig, vec, p })
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
    fn lstsq_rejects_underdetermined_and_rank_deficient() {
        // 1x1 gram, one row, one column: not overdetermined.
        assert!(lstsq_factor(&[1.0], 1, 1).is_none());
        // Exactly collinear 2-column design.
        let gram = [1.0, 2.0, 2.0, 4.0];
        assert!(lstsq_factor(&gram, 2, 50).is_none());
    }

    #[test]
    fn lstsq_accepts_a_design_it_can_actually_resolve() {
        // Eigenvalues a factor of 1e-7 apart: well inside what the normal
        // equations can resolve, so this is fitted rather than written off.
        let gram = [1.0, 0.0, 0.0, 1e-7];
        let f = lstsq_factor(&gram, 2, 50).expect("full rank");
        let mut beta = [0.0; 2];
        f.solve_into(&[1.0, 1.0], &mut beta);
        assert!(beta.iter().all(|v| v.is_finite()));
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
