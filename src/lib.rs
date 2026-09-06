//! Python bindings.
//!
//! The Python layer in `python/ruptures_rs/` owns the public API surface and
//! its validation; this module owns the numerics. The boundary is deliberately
//! coarse — one crossing per `predict()` call, not one per cost evaluation —
//! because paying FFI overhead inside the dynamic program would give back
//! exactly what the rewrite is meant to win.
//!
//! Two rules hold everywhere below the boundary. Nothing may panic: a Rust
//! panic crosses `pyo3` as `PanicException`, which inherits from
//! `BaseException` and so slips past every `except Exception` a caller wrote.
//! And nothing may allocate unbounded: an allocation failure in Rust aborts the
//! process rather than raising, so every quadratic structure is measured
//! against a budget first and refused with an actionable message.

mod cost;
mod crops;
mod detect;
mod linalg;
mod prefix;

use cost::{
    Cost, CostCLinear, CostKernel, CostKernelLinear, CostL1, CostL2, CostLstsq, CostMl, CostNormal,
    CostRank, KernelFlavor,
};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyAssertionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Anything bigger than this is refused rather than attempted. Rust aborts the
/// whole process when an allocation fails, so "try it and see" is not an
/// option: the user would lose their interpreter, not get an exception.
const LIMIT_BYTES: usize = 4 << 30; // 4 GiB

fn check_alloc(elems: usize, what: &str, hint: &str) -> PyResult<()> {
    let bytes = elems.saturating_mul(std::mem::size_of::<f64>());
    if bytes > LIMIT_BYTES {
        return Err(PyValueError::new_err(format!(
            "{what} needs {:.1} GiB, over the {:.0} GiB limit. {hint}",
            bytes as f64 / (1u64 << 30) as f64,
            LIMIT_BYTES as f64 / (1u64 << 30) as f64,
        )));
    }
    Ok(())
}

/// Kernel costs need an `(n+1)^2` table of f64 — the same quadratic memory
/// `ruptures` spends on its dense Gram matrix.
fn check_kernel_memory(n: usize, model: &str, needs_median: bool) -> PyResult<()> {
    let table = (n + 1).saturating_mul(n + 1);
    check_alloc(
        table,
        &format!("the `{model}` model needs a kernel table of {n}x{n}, which"),
        "Subsample the signal, or use a non-kernel model such as `l2` or `normal`, \
         which need no quadratic memory.",
    )?;
    if needs_median {
        // The default `gamma` is the reciprocal median of the condensed
        // pairwise distance vector, which is another n(n-1)/2 doubles.
        let cond = n.saturating_mul(n.saturating_sub(1)) / 2;
        check_alloc(
            cond,
            &format!(
                "the `{model}` model's default `gamma` needs a pairwise distance table, which"
            ),
            "Pass an explicit `gamma` to skip it, or subsample the signal.",
        )?;
    }
    Ok(())
}

/// The covariance-shaped costs keep `(n+1) * d^2` doubles.
///
/// `ruptures` computes these in O(n * d^2) memory, so a wide signal that it
/// handles fine could take this process down with it. Refuse instead.
fn check_outer_memory(n: usize, d: usize, model: &str) -> PyResult<()> {
    let elems = (n + 1).saturating_mul(d.saturating_mul(d));
    check_alloc(
        elems,
        &format!("the `{model}` model needs prefix sums of {n}x{d}x{d} covariance blocks, which"),
        "Reduce the number of columns, subsample the signal, or use a model whose \
         cost does not depend on a covariance, such as `l1` or `l2`.",
    )
}

/// Flatten a NumPy array into a row-major `(n, d)` buffer.
fn as_matrix(arr: &PyReadonlyArrayDyn<f64>) -> PyResult<(Vec<f64>, usize, usize)> {
    let view = arr.as_array();
    let shape = view.shape();
    let (n, d) = match shape.len() {
        0 => (1usize, 1usize),
        1 => (shape[0], 1usize),
        2 => (shape[0], shape[1]),
        _ => return Err(PyValueError::new_err("signal must be 1-D or 2-D")),
    };
    let total = n
        .checked_mul(d)
        .ok_or_else(|| PyValueError::new_err("signal is too large to index"))?;
    check_alloc(total, "the signal", "Subsample it.")?;
    let mut buf = Vec::with_capacity(total);
    for v in view.iter() {
        buf.push(*v);
    }
    Ok((buf, n, d))
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
    #[pyo3(signature = (signal, model, params=None, variant=None))]
    fn new(
        signal: PyReadonlyArrayDyn<f64>,
        model: &str,
        params: Option<&Bound<'_, PyDict>>,
        variant: Option<&str>,
    ) -> PyResult<Self> {
        let (sig, n, d) = as_matrix(&signal)?;
        let flavor = match variant {
            None | Some("python") => KernelFlavor::Python,
            // The `ekcpd` C extension behind `KernelCPD` evaluates its kernels
            // differently from the `scipy` path the Python cost classes use.
            Some("extension") => KernelFlavor::Extension,
            Some(other) => return Err(PyValueError::new_err(format!("unknown variant: {other}"))),
        };

        // Parameter lookups propagate their errors. Swallowing them with
        // `unwrap_or(default)` silently ignored `add_small_diag=0` and
        // `order=-1`, so a caller's mistake changed the model instead of
        // being reported.
        let get_item = |k: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            match params {
                None => Ok(None),
                Some(p) => match p.get_item(k)? {
                    Some(v) if !v.is_none() => Ok(Some(v)),
                    _ => Ok(None),
                },
            }
        };
        let get_f64 = |k: &str| -> PyResult<Option<f64>> {
            match get_item(k)? {
                None => Ok(None),
                Some(v) => Ok(Some(v.extract::<f64>()?)),
            }
        };
        // `ruptures` writes `if self.add_small_diag:`, so any truthy object
        // works there. Matching that is why this is `is_truthy` rather than a
        // strict `bool` extraction.
        let get_truthy = |k: &str, default: bool| -> PyResult<bool> {
            match get_item(k)? {
                None => Ok(default),
                Some(v) => v.is_truthy(),
            }
        };

        let mut gamma_out = None;
        let inner: Box<dyn Cost> = match model {
            "l2" => {
                if flavor == KernelFlavor::Extension {
                    // `KernelCPD(kernel="linear")` is the C `linear_kernel`,
                    // which groups the same arithmetic as `trace - block / len`.
                    Box::new(CostKernelLinear::new(&sig, n, d))
                } else {
                    Box::new(CostL2::new(&sig, n, d))
                }
            }
            "l1" => Box::new(CostL1::new(&sig, n, d)),
            "normal" => {
                check_outer_memory(n, d, "normal")?;
                Box::new(CostNormal::new(
                    &sig,
                    n,
                    d,
                    get_truthy("add_small_diag", true)?,
                ))
            }
            "rbf" => {
                let gamma = get_f64("gamma")?;
                check_kernel_memory(n, "rbf", gamma.is_none())?;
                let (c, g) = CostKernel::rbf(&sig, n, d, gamma, flavor);
                gamma_out = Some(g);
                Box::new(c)
            }
            "cosine" => {
                check_kernel_memory(n, "cosine", false)?;
                Box::new(CostKernel::cosine(&sig, n, d, flavor))
            }
            "rank" => Box::new(CostRank::new(&sig, n, d)),
            "clinear" => Box::new(CostCLinear::new(&sig, n, d)),
            "mahalanobis" => {
                check_outer_memory(n, d, "mahalanobis")?;
                // The metric always arrives from the Python layer, which builds
                // it with `numpy.linalg.inv(numpy.cov(...))`. Computing it here
                // instead meant a Cholesky where NumPy uses an LU, and the two
                // disagree about which covariance matrices are singular.
                let metric = match get_item("metric")? {
                    Some(v) => {
                        let m: PyReadonlyArrayDyn<f64> = v.extract()?;
                        let (mv, md, md2) = as_matrix(&m)?;
                        if md != d || md2 != d {
                            return Err(PyValueError::new_err(format!(
                                "metric must be ({d}, {d}), got ({md}, {md2})"
                            )));
                        }
                        mv
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "the `mahalanobis` model requires a `metric`",
                        ))
                    }
                };
                Box::new(CostMl::new(&sig, n, d, metric))
            }
            "linear" => {
                // Column 0 is the response, the rest are covariates. A signal
                // with a single column leaves no covariates at all, which is
                // legal: NumPy reports rank 0 of 0 and the residual is the
                // whole of `y^T y`.
                let p = d.saturating_sub(1);
                check_outer_memory(n, p, "linear")?;
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
                let order = match get_item("order")? {
                    None => 4usize,
                    Some(v) => v.extract::<usize>()?,
                };
                if order >= n && n > 0 {
                    return Err(PyValueError::new_err(format!(
                        "the `ar` model needs more samples than its order ({order}); \
                         the signal has {n}"
                    )));
                }
                let p = order + 1;
                check_outer_memory(n, p, "ar")?;
                // The residual of `y ~ [lags, 1]` is unchanged by shifting the
                // signal, because the intercept column absorbs the shift. So
                // centring first is exact, and it keeps the lagged design from
                // being dominated by an offset.
                let sig = prefix::center(&sig, n, d);
                // ruptures builds the lagged design with `as_strided` over the
                // flattened buffer, then edge-pads the first `order` rows and
                // appends an intercept column.
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

    fn error(&self, start: i64, end: i64) -> PyResult<f64> {
        let (start, end) = self.checked_span(start, end)?;
        Ok(self.inner.error(start, end))
    }

    fn sum_of_costs(&self, bkps: Vec<i64>) -> PyResult<f64> {
        let mut acc = 0.0;
        let mut start: i64 = 0;
        for &end in &bkps {
            let (s, e) = self.checked_span(start, end)?;
            acc += self.inner.error(s, e);
            start = end;
        }
        Ok(acc)
    }
}

impl CostEngine {
    /// Validate a segment before it reaches an unchecked index.
    ///
    /// Every one of these used to be a panic or a silent out-of-bounds read:
    /// a negative index came back as `OverflowError` from an unsigned
    /// conversion, and an end past the signal indexed straight off the end of
    /// a prefix array.
    fn checked_span(&self, start: i64, end: i64) -> PyResult<(usize, usize)> {
        if start < 0 || end < 0 {
            return Err(PyValueError::new_err(format!(
                "segment bounds must not be negative, got ({start}, {end})"
            )));
        }
        let (s, e) = (start as usize, end as usize);
        if e > self.n_samples {
            return Err(PyValueError::new_err(format!(
                "segment end {e} is past the end of a signal with {} samples",
                self.n_samples
            )));
        }
        if e < s {
            return Err(PyValueError::new_err(format!(
                "segment end {e} is before its start {s}"
            )));
        }
        if e - s < self.inner.min_size() {
            return Err(PyRuntimeError::new_err("NotEnoughPoints"));
        }
        Ok((s, e))
    }
}

fn check_params(jump: usize, min_size: usize) -> PyResult<()> {
    if jump == 0 {
        return Err(PyValueError::new_err("jump must be at least 1"));
    }
    let _ = min_size;
    Ok(())
}

#[pyfunction]
fn dynp(
    py: Python<'_>,
    engine: &CostEngine,
    n_bkps: usize,
    jump: usize,
    min_size: usize,
) -> PyResult<Vec<usize>> {
    check_params(jump, min_size)?;
    let out = py.allow_threads(|| {
        detect::dynp(
            engine.inner.as_ref(),
            engine.n_samples,
            n_bkps,
            jump,
            min_size,
        )
    });
    // `ruptures` asserts here rather than returning a short segmentation, and
    // so does this: silently handing back fewer breakpoints than were asked
    // for is the kind of answer that gets used without being noticed.
    out.ok_or_else(|| {
        PyAssertionError::new_err(format!(
            "No admissible last breakpoints found. n_samples: {}, n_bkps: {n_bkps}.",
            engine.n_samples
        ))
    })
}

/// Every optimal segmentation from 1 to `n_bkps` breakpoints, from one pass.
#[pyfunction]
fn dynp_all(
    py: Python<'_>,
    engine: &CostEngine,
    n_bkps: usize,
    jump: usize,
    min_size: usize,
) -> PyResult<Vec<Option<Vec<usize>>>> {
    check_params(jump, min_size)?;
    Ok(py.allow_threads(|| {
        let tables = detect::dynp_tables(
            engine.inner.as_ref(),
            engine.n_samples,
            n_bkps,
            jump,
            min_size,
        );
        (1..=n_bkps).map(|k| tables.backtrack(k)).collect()
    }))
}

#[pyfunction]
fn pelt(
    py: Python<'_>,
    engine: &CostEngine,
    pen: f64,
    jump: usize,
    min_size: usize,
) -> PyResult<Vec<usize>> {
    check_params(jump, min_size)?;
    Ok(py.allow_threads(|| {
        detect::pelt(engine.inner.as_ref(), engine.n_samples, pen, jump, min_size)
    }))
}

/// `KernelCPD`'s penalised mode, transcribed from the C extension.
#[pyfunction]
fn kernel_pelt(
    py: Python<'_>,
    engine: &CostEngine,
    pen: f64,
    min_size: usize,
) -> PyResult<Vec<usize>> {
    let blocks = engine
        .inner
        .as_kernel_blocks()
        .ok_or_else(|| PyValueError::new_err("kernel_pelt requires a kernel cost engine"))?;
    if min_size == 0 {
        return Err(PyValueError::new_err("min_size must be at least 1"));
    }
    Ok(py.allow_threads(|| detect::kernel_pelt(blocks, engine.n_samples, pen, min_size)))
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
) -> PyResult<Vec<usize>> {
    check_params(jump, min_size)?;
    Ok(py.allow_threads(|| {
        let mut b = detect::Binseg::new(engine.inner.as_ref(), engine.n_samples, jump, min_size);
        b.run(n_bkps, pen, epsilon)
    }))
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
) -> PyResult<Vec<usize>> {
    check_params(jump, min_size)?;
    Ok(py.allow_threads(|| {
        let mut b = detect::BottomUp::new(engine.inner.as_ref(), engine.n_samples, jump, min_size);
        b.run(n_bkps, pen, epsilon)
    }))
}

#[pyfunction]
fn window_fit(
    py: Python<'_>,
    engine: &CostEngine,
    width: usize,
    jump: usize,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    check_params(jump, 1)?;
    Ok(py.allow_threads(|| {
        detect::window_score(engine.inner.as_ref(), engine.n_samples, width, jump)
    }))
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
) -> PyResult<Vec<usize>> {
    check_params(jump, min_size)?;
    Ok(py.allow_threads(|| {
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
    }))
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
) -> PyResult<Vec<(f64, f64, Vec<usize>)>> {
    check_params(jump, min_size)?;
    Ok(py.allow_threads(|| {
        crops::crops(
            engine.inner.as_ref(),
            engine.n_samples,
            pen_min,
            pen_max,
            jump,
            min_size,
        )
    }))
}

/// Expose the covariance-memory budget so the Python layer can check it before
/// building a covariance of its own that would be just as large.
#[pyfunction]
#[pyo3(name = "check_outer_memory")]
fn check_outer_memory_py(n: usize, d: usize, model: &str) -> PyResult<()> {
    check_outer_memory(n, d, model)
}

#[pyfunction]
fn sanity_check(n_samples: usize, n_bkps: usize, jump: usize, min_size: usize) -> bool {
    detect::sanity_check(n_samples, n_bkps, jump, min_size)
}

#[pymodule]
fn _ruptures_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CostEngine>()?;
    m.add_function(wrap_pyfunction!(dynp, m)?)?;
    m.add_function(wrap_pyfunction!(dynp_all, m)?)?;
    m.add_function(wrap_pyfunction!(pelt, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_pelt, m)?)?;
    m.add_function(wrap_pyfunction!(binseg, m)?)?;
    m.add_function(wrap_pyfunction!(bottomup, m)?)?;
    m.add_function(wrap_pyfunction!(window_fit, m)?)?;
    m.add_function(wrap_pyfunction!(window_predict, m)?)?;
    m.add_function(wrap_pyfunction!(crops_path, m)?)?;
    m.add_function(wrap_pyfunction!(sanity_check, m)?)?;
    m.add_function(wrap_pyfunction!(check_outer_memory_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
