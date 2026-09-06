//! Python bindings.
//!
//! The Python layer in `python/ruptures_rs/` owns the public API surface and
//! its validation; this module owns the numerics. The boundary is deliberately
//! coarse — one crossing per `predict()` call, not one per cost evaluation —
//! because paying FFI overhead inside the dynamic program would give back
//! exactly what the rewrite is meant to win.

mod cost;
mod crops;
mod detect;
mod linalg;
mod prefix;

use cost::{
    Cost, CostCLinear, CostKernel, CostL1, CostL2, CostLstsq, CostMl, CostNormal, CostRank,
};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Flatten a NumPy array into a row-major `(n, d)` buffer.
fn as_matrix(arr: &PyReadonlyArrayDyn<f64>) -> PyResult<(Vec<f64>, usize, usize)> {
    let view = arr.as_array();
    let shape = view.shape();
    let (n, d) = match shape.len() {
        1 => (shape[0], 1usize),
        2 => (shape[0], shape[1]),
        _ => return Err(PyValueError::new_err("signal must be 1-D or 2-D")),
    };
    let mut buf = Vec::with_capacity(n * d);
    for v in view.iter() {
        buf.push(*v);
    }
    Ok((buf, n, d))
}

/// Inverse of a symmetric positive-definite matrix (for the default
/// Mahalanobis metric, mirroring `numpy.linalg.inv(np.cov(...))`).
fn inv_spd(a: &[f64], d: usize) -> PyResult<Vec<f64>> {
    let mut l = a.to_vec();
    if !linalg::cholesky(&mut l, d) {
        return Err(PyRuntimeError::new_err(
            "covariance matrix is singular; pass an explicit `metric`",
        ));
    }
    let mut out = vec![0.0; d * d];
    let mut e = vec![0.0; d];
    for col in 0..d {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[col] = 1.0;
        linalg::cholesky_solve(&l, d, &mut e);
        for row in 0..d {
            out[row * d + col] = e[row];
        }
    }
    Ok(out)
}

/// Sample covariance with `ddof=1`, as `np.cov` computes it.
fn covariance(sig: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut mean = vec![0.0; d];
    for j in 0..d {
        let mut acc = 0.0;
        for i in 0..n {
            acc += sig[i * d + j];
        }
        mean[j] = acc / n as f64;
    }
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        for a in 0..d {
            for b in 0..d {
                cov[a * d + b] += (sig[i * d + a] - mean[a]) * (sig[i * d + b] - mean[b]);
            }
        }
    }
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    cov.iter_mut().for_each(|v| *v /= denom);
    cov
}

/// A fitted cost function over a fixed signal.
///
/// Holds the prefix-sum structures, so construction is O(n * d) (or O(n^2) for
/// the kernel costs, the same as the Gram matrix `ruptures` builds) and every
/// subsequent `error()` is O(1) in the segment length for all models but `l1`.
#[pyclass(module = "ruptures_rs._ruptures_rs")]
pub struct CostEngine {
    inner: Box<dyn Cost>,
    #[pyo3(get)]
    n_samples: usize,
    #[pyo3(get)]
    gamma: Option<f64>,
}

#[pymethods]
impl CostEngine {
    #[new]
    #[pyo3(signature = (signal, model, params=None))]
    fn new(
        signal: PyReadonlyArrayDyn<f64>,
        model: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (sig, n, d) = as_matrix(&signal)?;
        let getf = |k: &str| -> PyResult<Option<f64>> {
            match params {
                None => Ok(None),
                Some(p) => match p.get_item(k)? {
                    None => Ok(None),
                    Some(v) => {
                        if v.is_none() {
                            Ok(None)
                        } else {
                            Ok(Some(v.extract::<f64>()?))
                        }
                    }
                },
            }
        };
        let mut gamma_out = None;
        let inner: Box<dyn Cost> = match model {
            "l2" => Box::new(CostL2::new(&sig, n, d)),
            "l1" => Box::new(CostL1::new(&sig, n, d)),
            "normal" => {
                let add = match params {
                    None => true,
                    Some(p) => match p.get_item("add_small_diag")? {
                        None => true,
                        Some(v) => v.extract::<bool>().unwrap_or(true),
                    },
                };
                Box::new(CostNormal::new(&sig, n, d, add))
            }
            "rbf" => {
                let (c, g) = CostKernel::rbf(&sig, n, d, getf("gamma")?);
                gamma_out = Some(g);
                Box::new(c)
            }
            "cosine" => Box::new(CostKernel::cosine(&sig, n, d)),
            "rank" => Box::new(CostRank::new(&sig, n, d)),
            "clinear" => Box::new(CostCLinear::new(&sig, n, d)),
            "mahalanobis" => {
                let metric = match params.and_then(|p| p.get_item("metric").ok().flatten()) {
                    Some(v) if !v.is_none() => {
                        let m: PyReadonlyArrayDyn<f64> = v.extract()?;
                        let (mv, md, md2) = as_matrix(&m)?;
                        if md != d || md2 != d {
                            return Err(PyValueError::new_err("metric must be (d, d)"));
                        }
                        mv
                    }
                    _ => inv_spd(&covariance(&sig, n, d), d)?,
                };
                Box::new(CostMl::new(&sig, n, d, metric))
            }
            "linear" => {
                if d < 2 {
                    return Err(PyValueError::new_err(
                        "the `linear` model needs at least 2 columns (response + covariates)",
                    ));
                }
                let p = d - 1;
                let mut x = vec![0.0; n * p];
                let mut y = vec![0.0; n];
                for i in 0..n {
                    y[i] = sig[i * d];
                    for k in 0..p {
                        x[i * p + k] = sig[i * d + 1 + k];
                    }
                }
                Box::new(CostLstsq::new(&x, &y, n, p, 1, 2, "linear"))
            }
            "ar" => {
                let order = match params {
                    None => 4usize,
                    Some(p) => match p.get_item("order")? {
                        None => 4,
                        Some(v) => v.extract::<usize>().unwrap_or(4),
                    },
                };
                // ruptures builds the lagged design with `as_strided` over the
                // flattened buffer, then edge-pads the first `order` rows and
                // appends an intercept column.
                let p = order + 1;
                let mut x = vec![0.0; n * p];
                for i in 0..n {
                    let base = i.saturating_sub(order);
                    for k in 0..order {
                        x[i * p + k] = sig.get(base + k).copied().unwrap_or(0.0);
                    }
                    x[i * p + order] = 1.0;
                }
                let mut y = vec![0.0; n * d];
                for i in 0..n {
                    let src = if i < order { order.min(n - 1) } else { i };
                    for j in 0..d {
                        y[i * d + j] = sig[src * d + j];
                    }
                }
                let min_size = std::cmp::max(5, order + 1);
                Box::new(CostLstsq::new(&x, &y, n, p, d, min_size, "ar"))
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown model: {other}. Supported: l1, l2, normal, rbf, cosine, rank, \
                     mahalanobis, clinear, linear, ar"
                )))
            }
        };
        Ok(Self {
            inner,
            n_samples: n,
            gamma: gamma_out,
        })
    }

    #[getter]
    fn min_size(&self) -> usize {
        self.inner.min_size()
    }

    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model()
    }

    fn error(&self, start: usize, end: usize) -> PyResult<f64> {
        if end <= start || end - start < self.inner.min_size() {
            return Err(PyRuntimeError::new_err("NotEnoughPoints"));
        }
        if end > self.n_samples {
            return Err(PyValueError::new_err("segment end past the signal"));
        }
        Ok(self.inner.error(start, end))
    }

    fn sum_of_costs(&self, bkps: Vec<usize>) -> f64 {
        self.inner.sum_of_costs(&bkps)
    }
}

#[pyfunction]
fn dynp(
    py: Python<'_>,
    engine: &CostEngine,
    n_bkps: usize,
    jump: usize,
    min_size: usize,
) -> Vec<usize> {
    py.allow_threads(|| {
        detect::dynp(
            engine.inner.as_ref(),
            engine.n_samples,
            n_bkps,
            jump,
            min_size,
        )
    })
}

#[pyfunction]
fn pelt(py: Python<'_>, engine: &CostEngine, pen: f64, jump: usize, min_size: usize) -> Vec<usize> {
    py.allow_threads(|| detect::pelt(engine.inner.as_ref(), engine.n_samples, pen, jump, min_size))
}

#[pyfunction]
#[pyo3(signature = (engine, jump, min_size, n_bkps=None, pen=None, epsilon=None))]
fn binseg(
    py: Python<'_>,
    engine: &CostEngine,
    jump: usize,
    min_size: usize,
    n_bkps: Option<usize>,
    pen: Option<f64>,
    epsilon: Option<f64>,
) -> Vec<usize> {
    py.allow_threads(|| {
        let mut b = detect::Binseg::new(engine.inner.as_ref(), engine.n_samples, jump, min_size);
        b.run(n_bkps, pen, epsilon)
    })
}

#[pyfunction]
#[pyo3(signature = (engine, jump, min_size, n_bkps=None, pen=None, epsilon=None))]
fn bottomup(
    py: Python<'_>,
    engine: &CostEngine,
    jump: usize,
    min_size: usize,
    n_bkps: Option<usize>,
    pen: Option<f64>,
    epsilon: Option<f64>,
) -> Vec<usize> {
    py.allow_threads(|| {
        let mut b = detect::BottomUp::new(engine.inner.as_ref(), engine.n_samples, jump, min_size);
        b.run(n_bkps, pen, epsilon)
    })
}

#[pyfunction]
fn window_fit(
    py: Python<'_>,
    engine: &CostEngine,
    width: usize,
    jump: usize,
) -> (Vec<usize>, Vec<f64>) {
    py.allow_threads(|| detect::window_score(engine.inner.as_ref(), engine.n_samples, width, jump))
}

#[pyfunction]
#[pyo3(signature = (engine, inds, score, width, jump, min_size, n_bkps=None, pen=None, epsilon=None))]
#[allow(clippy::too_many_arguments)]
fn window_predict(
    py: Python<'_>,
    engine: &CostEngine,
    inds: Vec<usize>,
    score: Vec<f64>,
    width: usize,
    jump: usize,
    min_size: usize,
    n_bkps: Option<usize>,
    pen: Option<f64>,
    epsilon: Option<f64>,
) -> Vec<usize> {
    py.allow_threads(|| {
        detect::window_seg(
            engine.inner.as_ref(),
            engine.n_samples,
            &inds,
            &score,
            width,
            jump,
            min_size,
            n_bkps,
            pen,
            epsilon,
        )
    })
}

#[pyfunction]
#[pyo3(name = "crops")]
fn crops_path(
    py: Python<'_>,
    engine: &CostEngine,
    pen_min: f64,
    pen_max: f64,
    jump: usize,
    min_size: usize,
) -> Vec<(f64, f64, Vec<usize>)> {
    py.allow_threads(|| {
        crops::crops(
            engine.inner.as_ref(),
            engine.n_samples,
            pen_min,
            pen_max,
            jump,
            min_size,
        )
    })
}

#[pyfunction]
fn sanity_check(n_samples: usize, n_bkps: usize, jump: usize, min_size: usize) -> bool {
    detect::sanity_check(n_samples, n_bkps, jump, min_size)
}

#[pymodule]
fn _ruptures_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CostEngine>()?;
    m.add_function(wrap_pyfunction!(dynp, m)?)?;
    m.add_function(wrap_pyfunction!(pelt, m)?)?;
    m.add_function(wrap_pyfunction!(binseg, m)?)?;
    m.add_function(wrap_pyfunction!(bottomup, m)?)?;
    m.add_function(wrap_pyfunction!(window_fit, m)?)?;
    m.add_function(wrap_pyfunction!(window_predict, m)?)?;
    m.add_function(wrap_pyfunction!(crops_path, m)?)?;
    m.add_function(wrap_pyfunction!(sanity_check, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
