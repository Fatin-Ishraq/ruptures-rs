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
mod stable;

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
///
/// The default is 4 GiB per structure. It is a ceiling on one allocation rather
/// than a promise about peak working memory, and a host with less than that
/// free will still run out before the check fires — so it is adjustable, with
/// `ruptures_rs.set_memory_limit()` or the `RUPTURES_RS_MEMORY_LIMIT_GIB`
/// environment variable. Downwards is the useful direction: a service that
/// accepts dimensions from a request can refuse hostile ones early.
const DEFAULT_LIMIT_BYTES: usize = 4 << 30; // 4 GiB

static LIMIT_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_LIMIT_BYTES);

fn limit_bytes() -> usize {
    LIMIT_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Refuse a structure of `bytes` bytes, whatever its element type.
fn check_bytes(bytes: usize, what: &str, hint: &str) -> PyResult<()> {
    let limit = limit_bytes();
    if bytes > limit {
        return Err(PyValueError::new_err(format!(
            "{what} needs {:.2} GiB, over the {:.2} GiB limit. {hint} The limit can be \
             changed with `ruptures_rs.set_memory_limit()`.",
            bytes as f64 / (1u64 << 30) as f64,
            limit as f64 / (1u64 << 30) as f64,
        )));
    }
    Ok(())
}

fn check_alloc(elems: usize, what: &str, hint: &str) -> PyResult<()> {
    check_bytes(elems.saturating_mul(std::mem::size_of::<f64>()), what, hint)
}

/// The dynamic program keeps one back-pointer per candidate position per level,
/// and `KernelCPD` then reads every level back out as its own breakpoint list.
///
/// Neither is quadratic in the signal, but both are unbounded in `n_bkps`,
/// which arrives from the caller: asking a long signal for ten million
/// breakpoints requests hundreds of gigabytes with nothing else to suggest that
/// anything is wrong.
fn check_dynp_memory(n: usize, n_bkps: usize, jump: usize, all_levels: bool) -> PyResult<()> {
    let cells = (n / jump.max(1)).saturating_add(2);
    let back = n_bkps
        .saturating_mul(cells)
        .saturating_mul(std::mem::size_of::<u32>());
    check_bytes(
        back,
        &format!(
            "a dynamic program over {n} samples with {n_bkps} breakpoints needs back-pointer \
             tables, which"
        ),
        "Ask for fewer breakpoints, raise `jump`, or subsample the signal.",
    )?;
    if all_levels {
        // Every k from 1 to n_bkps is returned, so the output alone grows with
        // the square of the number of breakpoints.
        let out = n_bkps
            .saturating_mul(n_bkps.saturating_add(3))
            .saturating_div(2)
            .saturating_mul(std::mem::size_of::<usize>());
        check_bytes(
            out,
            &format!("returning every segmentation up to {n_bkps} breakpoints, which"),
            "Ask for fewer breakpoints.",
        )?;
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
    // A signal with no columns is not a degenerate case to be handled, it is a
    // mistake: `linear` reads column 0 of it and every kernel takes a row norm
    // over nothing. Both used to index an empty slice and panic.
    if d == 0 {
        return Err(PyValueError::new_err(
            "signal must have at least one column, got a shape with zero features",
        ));
    }
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
            "rank" => {
                // The prefix arrays here are only O(n * d), but the rank
                // covariance and its pseudo-inverse are both `d x d`, and the
                // eigensolver wants a third. A two-row, thirty-thousand-column
                // signal is under half a megabyte and asks for 6.7 GiB apiece.
                check_alloc(
                    d.saturating_mul(d).saturating_mul(3),
                    &format!(
                        "the `rank` model needs a {d}x{d} covariance of ranks and its \
                         pseudo-inverse, which"
                    ),
                    "Reduce the number of columns, or use a model whose cost does not depend \
                     on a covariance, such as `l1` or `l2`.",
                )?;
                Box::new(CostRank::new(&sig, n, d))
            }
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
                // The design and the pre-fitted response are kept too, for the
                // segments whose Gram matrix cannot decide their rank.
                check_alloc(
                    n.saturating_mul(p.saturating_add(1)),
                    &format!("the `linear` model's {n}x{p} design, which"),
                    "Subsample the signal, or reduce the number of covariates.",
                )?;
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
                check_alloc(
                    n.saturating_mul(p.saturating_add(d)),
                    &format!("the `ar` model's {n}x{p} lagged design, which"),
                    "Subsample the signal, or lower the order.",
                )?;
                // The residual of `y ~ [lags, 1]` is unchanged by shifting the
                // signal by a constant, because the intercept column absorbs
                // the shift. So centring first is exact, and it keeps the
                // lagged design from being dominated by an offset.
                //
                // One scalar mean over the whole buffer, not one per column.
                // The lagged design is built by walking the *flattened* signal,
                // so with more than one column a row of lags mixes values from
                // different columns — and per-column offsets do not then move
                // together, which is what makes the shift absorbable. Removing
                // a single constant is invariant for any number of columns;
                // removing d of them silently changed the model.
                let sig = prefix::center_scalar(&sig, n, d);
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
    if min_size == 0 {
        return Err(PyValueError::new_err("min_size must be at least 1"));
    }
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
    check_dynp_memory(engine.n_samples, n_bkps, jump, false)?;
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
    check_dynp_memory(engine.n_samples, n_bkps, jump, true)?;
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
    if width == 0 {
        return Err(PyValueError::new_err("width must be at least 1"));
    }
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
    // `inds` and `score` come in as plain lists. The estimator that produced
    // them keeps them consistent, but nothing stops a caller reaching the
    // extension directly, and `window_seg` reads one at the other's indices.
    if inds.len() != score.len() {
        return Err(PyValueError::new_err(format!(
            "inds and score must be the same length, got {} and {}",
            inds.len(),
            score.len()
        )));
    }
    if let Some(&bad) = inds.iter().find(|&&i| i >= engine.n_samples) {
        return Err(PyValueError::new_err(format!(
            "window index {bad} is past the end of a signal with {} samples",
            engine.n_samples
        )));
    }
    if width == 0 {
        return Err(PyValueError::new_err("width must be at least 1"));
    }
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

/// Read the current per-structure allocation ceiling, in bytes.
#[pyfunction]
fn get_memory_limit() -> usize {
    limit_bytes()
}

/// Set the per-structure allocation ceiling, in bytes.
///
/// Lowering it is how a service that takes signal dimensions from a request
/// makes this library refuse hostile ones at the boundary rather than at the
/// allocator, where the failure is a process abort rather than an exception.
#[pyfunction]
fn set_memory_limit(bytes: usize) -> PyResult<()> {
    if bytes == 0 {
        return Err(PyValueError::new_err("the memory limit must be positive"));
    }
    LIMIT_BYTES.store(bytes, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Expose the quadratic budget so the Python layer can refuse an `n x n`
/// matrix of its own before NumPy attempts it.
#[pyfunction]
#[pyo3(name = "check_square_memory")]
fn check_square_memory_py(n: usize, what: &str) -> PyResult<()> {
    check_alloc(
        n.saturating_mul(n),
        &format!("{what} over {n} samples, which"),
        "Subsample the signal. The engine itself does not need this matrix.",
    )
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
    m.add_function(wrap_pyfunction!(check_square_memory_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_memory_limit, m)?)?;
    m.add_function(wrap_pyfunction!(set_memory_limit, m)?)?;
    // An environment variable so a limit can be imposed on a process that does
    // not own the code doing the importing.
    if let Ok(raw) = std::env::var("RUPTURES_RS_MEMORY_LIMIT_GIB") {
        match raw.trim().parse::<f64>() {
            Ok(gib) if gib > 0.0 && gib.is_finite() => {
                let bytes = (gib * (1u64 << 30) as f64) as usize;
                LIMIT_BYTES.store(bytes.max(1), std::sync::atomic::Ordering::Relaxed);
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "RUPTURES_RS_MEMORY_LIMIT_GIB must be a positive number, got {raw:?}"
                )))
            }
        }
    }
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
