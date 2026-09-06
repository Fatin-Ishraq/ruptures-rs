//! Small dense linear algebra on `d x d` matrices, where `d` is the signal
//! dimension (typically 1-20). Everything is row-major `Vec<f64>`.

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
                if !(s > 0.0) {
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

/// Solve `L L^T x = b` in place given the Cholesky factor `l`.
pub fn cholesky_solve(l: &[f64], d: usize, b: &mut [f64]) {
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * d + k] * b[k];
        }
        b[i] = s / l[i * d + i];
    }
    for i in (0..d).rev() {
        let mut s = b[i];
        for k in (i + 1)..d {
            s -= l[k * d + i] * b[k];
        }
        b[i] = s / l[i * d + i];
    }
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
/// in a row-major `d x d` array. Used for the pseudo-inverse in `CostRank`,
/// mirroring `numpy.linalg.pinv` on a symmetric covariance.
pub fn jacobi_eigh(a_in: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..d {
            for j in (i + 1)..d {
                off += a[i * d + j] * a[i * d + j];
            }
        }
        if off <= 1e-300 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
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

/// Median of a slice, matching `numpy.median` (mean of the two central order
/// statistics for even length). Reorders `buf` in place.
pub fn median_inplace(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, hi, _) = buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let upper = *hi;
    if n % 2 == 1 {
        upper
    } else {
        let lower = buf[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}
