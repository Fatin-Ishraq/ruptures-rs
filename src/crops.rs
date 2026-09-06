//! CROPS: Changepoints for a Range Of PenaltieS.
//!
//! Haynes, Eckley & Fearnhead (2017), *Computationally efficient changepoint
//! detection for a range of penalties*, JCGS 26(1).
//!
//! Nobody knows their penalty in advance. The usual workaround is to run PELT
//! on a grid of penalties, which is both wasteful (many penalties yield the
//! same segmentation) and incomplete (a grid can step straight over a
//! segmentation that is optimal on a narrow interval).
//!
//! CROPS instead recovers *every* segmentation that is optimal for some
//! penalty in `[pen_min, pen_max]`, along with the exact penalty interval over
//! which each one wins, using O(number of distinct segmentations) PELT runs
//! rather than O(grid size). `ruptures` has no equivalent.
//!
//! The idea: a segmentation with `m` segments and unpenalised cost `Q` has
//! penalised cost `Q + pen * m`, which is affine in `pen`. The optimal
//! segmentation at each penalty is therefore the lower envelope of a family of
//! lines, and the distinct optima are exactly the vertices of the lower convex
//! hull of the points `(m, Q)`.

use crate::cost::Cost;
use crate::detect::pelt;
use std::collections::HashMap;

/// One optimal segmentation and the penalty interval it owns.
pub type Segment = (f64, f64, Vec<usize>);

/// PELT runs already performed, keyed by the penalty's bit pattern:
/// `(n_segments, unpenalised cost, breakpoints)`.
type Seen = HashMap<u64, (usize, f64, Vec<usize>)>;

fn run(
    cost: &dyn Cost,
    n: usize,
    pen: f64,
    jump: usize,
    min_size: usize,
    seen: &mut Seen,
) -> (usize, f64) {
    let key = pen.to_bits();
    if let Some((m, q, _)) = seen.get(&key) {
        return (*m, *q);
    }
    let bkps = pelt(cost, n, pen, jump, min_size);
    let m = bkps.len();
    let q = cost.sum_of_costs(&bkps);
    seen.insert(key, (m, q, bkps));
    (m, q)
}

pub fn crops(
    cost: &dyn Cost,
    n: usize,
    pen_min: f64,
    pen_max: f64,
    jump: usize,
    min_size: usize,
) -> Vec<Segment> {
    let mut seen: Seen = HashMap::new();

    let (m_min, q_min) = run(cost, n, pen_min, jump, min_size, &mut seen);
    let (m_max, q_max) = run(cost, n, pen_max, jump, min_size, &mut seen);

    // Explore intervals whose endpoints differ by more than one segment; the
    // candidate penalty is where the two penalised costs cross.
    let mut stack: Vec<(f64, usize, f64, f64, usize, f64)> =
        vec![(pen_min, m_min, q_min, pen_max, m_max, q_max)];

    while let Some((p0, m0, q0, p1, m1, q1)) = stack.pop() {
        if m0 <= m1 + 1 {
            continue;
        }
        let denom = (m0 - m1) as f64;
        let p_int = (q1 - q0) / denom;
        if !p_int.is_finite() || p_int <= p0 || p_int >= p1 {
            continue;
        }
        let (mi, qi) = run(cost, n, p_int, jump, min_size, &mut seen);
        if mi != m1 {
            stack.push((p0, m0, q0, p_int, mi, qi));
            stack.push((p_int, mi, qi, p1, m1, q1));
        }
    }

    // Keep the cheapest segmentation per segment count.
    let mut best: HashMap<usize, (f64, Vec<usize>)> = HashMap::new();
    for (m, q, bkps) in seen.into_values() {
        best.entry(m)
            .and_modify(|e| {
                if q < e.0 {
                    *e = (q, bkps.clone());
                }
            })
            .or_insert((q, bkps));
    }
    let mut pts: Vec<(usize, f64, Vec<usize>)> =
        best.into_iter().map(|(m, (q, b))| (m, q, b)).collect();
    pts.sort_by_key(|p| p.0);

    // Lower convex hull of (m, Q): only these are optimal for some penalty.
    let mut hull: Vec<(usize, f64, Vec<usize>)> = Vec::new();
    for p in pts {
        while hull.len() >= 2 {
            let a = &hull[hull.len() - 2];
            let b = &hull[hull.len() - 1];
            // drop b if it sits on or above the segment a->p
            let cross =
                (b.0 as f64 - a.0 as f64) * (p.1 - a.1) - (p.0 as f64 - a.0 as f64) * (b.1 - a.1);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    // Penalty at which consecutive hull members swap places.
    let mut out: Vec<Segment> = Vec::with_capacity(hull.len());
    for (i, item) in hull.iter().enumerate() {
        // hull is ordered by increasing m, so decreasing penalty
        let hi = if i == 0 {
            pen_max
        } else {
            let prev = &hull[i - 1];
            (prev.1 - item.1) / (item.0 as f64 - prev.0 as f64)
        };
        let lo = if i + 1 == hull.len() {
            pen_min
        } else {
            let next = &hull[i + 1];
            (item.1 - next.1) / (next.0 as f64 - item.0 as f64)
        };
        out.push((lo.max(pen_min), hi.min(pen_max), item.2.clone()));
    }
    // `hull` is already ordered fewest-breakpoints (high penalty) first
    out
}
