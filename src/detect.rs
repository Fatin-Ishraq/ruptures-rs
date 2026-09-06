//! Change point detectors.
//!
//! These are faithful ports of the `ruptures` algorithms, including their
//! tie-breaking rules, which are load-bearing: `ruptures` leans on Python's
//! `min`/`max` returning the *first* extremum, so a Rust port that used
//! `Iterator::max_by` (which returns the *last*) would silently produce
//! different — though equally optimal — breakpoints.
//!
//! Where the shape of the computation is changed, it is changed without
//! changing the answer:
//!
//! * `Dynp` is a bottom-up table rather than a memoised recursion. Costs are
//!   accumulated left to right, matching `sum(partition.values())` on a dict
//!   whose insertion order is left to right, so the floating-point association
//!   order is preserved.
//! * `Pelt` keeps a scalar running total plus a back-pointer instead of copying
//!   a whole partition dict per candidate.

use crate::cost::Cost;
use rayon::prelude::*;
use std::collections::HashMap;

pub const INF: f64 = f64::INFINITY;

/// Port of `ruptures.utils.sanity_check`.
pub fn sanity_check(n_samples: usize, n_bkps: usize, jump: usize, min_size: usize) -> bool {
    let n_adm_bkps = n_samples / jump;
    if n_bkps > n_adm_bkps {
        return false;
    }
    let ceil_div = min_size.div_ceil(jump);
    if n_bkps * ceil_div * jump + min_size > n_samples {
        return false;
    }
    true
}

/// Total ordering wrapper so floats can live in a `BinaryHeap`.
#[derive(Clone, Copy, PartialEq)]
struct Ordf64(f64);
impl Eq for Ordf64 {}
impl PartialOrd for Ordf64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ordf64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ------------------------------------------------------------------ Dynp

/// Exact segmentation by dynamic programming.
///
/// `D[k][e]` is the optimal cost of segmenting `[0, e)` with `k` breakpoints.
/// With O(1) segment costs the recurrence is a flat O(K * |cand|^2) loop over
/// scalars, which is where the three-orders-of-magnitude speedup comes from:
/// `ruptures` spends the same asymptotic work but pays a NumPy call at every
/// single cell.
///
/// Levels are computed in parallel across target positions; within a level the
/// cells are independent given the previous one.
pub fn dynp(cost: &dyn Cost, n: usize, n_bkps: usize, jump: usize, min_size: usize) -> Vec<usize> {
    // Admissible breakpoint positions are the multiples of `jump` below `n`.
    let cand: Vec<usize> = (0..n).step_by(jump).collect();
    let mut targets = cand.clone();
    targets.push(n);

    let mut level: Vec<f64> = vec![INF; n + 1];
    for &e in &targets {
        if e >= min_size {
            level[e] = cost.error(0, e);
        }
    }

    let mut back: Vec<Vec<u32>> = Vec::with_capacity(n_bkps);

    for k in 1..=n_bkps {
        let prev = &level;
        // Only breakpoints that leave a feasible left subproblem are admissible.
        let adm: Vec<usize> = cand
            .iter()
            .copied()
            .filter(|&b| sanity_check(b, k - 1, jump, min_size) && prev[b].is_finite())
            .collect();

        let computed: Vec<(usize, f64, u32)> = targets
            .par_iter()
            .map(|&e| {
                let mut best = INF;
                let mut arg = u32::MAX;
                for &b in &adm {
                    if b >= e || e - b < min_size {
                        continue;
                    }
                    // strict `<` keeps the FIRST minimum, matching Python's `min`
                    let v = prev[b] + cost.error(b, e);
                    if v < best {
                        best = v;
                        arg = b as u32;
                    }
                }
                (e, best, arg)
            })
            .collect();

        let mut next = vec![INF; n + 1];
        let mut bk = vec![u32::MAX; n + 1];
        for (e, v, a) in computed {
            next[e] = v;
            bk[e] = a;
        }
        level = next;
        back.push(bk);
    }

    // Backtrack.
    let mut bkps = vec![n];
    let mut pos = n;
    for k in (0..n_bkps).rev() {
        let b = back[k][pos];
        if b == u32::MAX {
            break;
        }
        pos = b as usize;
        bkps.push(pos);
    }
    bkps.sort_unstable();
    bkps.dedup();
    bkps
}

// ------------------------------------------------------------------ Pelt

/// Penalised segmentation with pruning.
///
/// A faithful port, including one quirk: `ruptures` zips `admissible` against
/// `subproblems` even though `subproblems` skips positions with no recorded
/// partition, so the two can fall out of step. Reproducing that keeps the
/// output identical rather than merely defensible.
pub fn pelt(cost: &dyn Cost, n: usize, pen: f64, jump: usize, min_size: usize) -> Vec<usize> {
    let mut totals: HashMap<usize, f64> = HashMap::new();
    let mut prev: HashMap<usize, usize> = HashMap::new();
    totals.insert(0, 0.0);

    let mut ind: Vec<usize> = (0..n).step_by(jump).filter(|&k| k >= min_size).collect();
    ind.push(n);

    let mut admissible: Vec<usize> = Vec::new();

    for &bkp in &ind {
        let new_adm = ((bkp as isize - min_size as isize).div_euclid(jump as isize) * jump as isize)
            .max(0) as usize;
        admissible.push(new_adm);

        // (total, t) for every admissible t that has a recorded partition
        let mut subproblems: Vec<(f64, usize)> = Vec::with_capacity(admissible.len());
        for &t in &admissible {
            if let Some(&tot) = totals.get(&t) {
                subproblems.push((tot + cost.error(t, bkp) + pen, t));
            }
        }
        if subproblems.is_empty() {
            continue;
        }
        // first minimum wins, matching Python's `min`
        let mut best = subproblems[0];
        for &sp in subproblems.iter().skip(1) {
            if sp.0 < best.0 {
                best = sp;
            }
        }
        totals.insert(bkp, best.0);
        prev.insert(bkp, best.1);

        // pruning, zipped positionally exactly as ruptures does
        let cutoff = best.0 + pen;
        admissible = admissible
            .iter()
            .zip(subproblems.iter())
            .filter(|(_, sp)| sp.0 <= cutoff)
            .map(|(t, _)| *t)
            .collect();
    }

    let mut bkps = Vec::new();
    let mut pos = n;
    while pos != 0 {
        bkps.push(pos);
        match prev.get(&pos) {
            Some(&p) => pos = p,
            None => break,
        }
    }
    bkps.sort_unstable();
    bkps.dedup();
    bkps
}

// ------------------------------------------------------------------ Binseg

pub struct Binseg<'a> {
    cost: &'a dyn Cost,
    n: usize,
    jump: usize,
    min_size: usize,
    cache: HashMap<(usize, usize), (Option<usize>, f64)>,
}

impl<'a> Binseg<'a> {
    pub fn new(cost: &'a dyn Cost, n: usize, jump: usize, min_size: usize) -> Self {
        Self {
            cost,
            n,
            jump,
            min_size,
            cache: HashMap::new(),
        }
    }

    /// Best single split of `[start, end)` and the gain it yields.
    fn single_bkp(&mut self, start: usize, end: usize) -> (Option<usize>, f64) {
        if let Some(&hit) = self.cache.get(&(start, end)) {
            return hit;
        }
        let segment_cost = self.cost.error(start, end);
        let out = if segment_cost.is_infinite() && segment_cost < 0.0 {
            (None, 0.0)
        } else {
            // `max` over (gain, bkp) tuples: best gain, ties broken by LARGER bkp
            let mut best: Option<(f64, usize)> = None;
            let mut bkp = start;
            while bkp < end {
                if bkp - start >= self.min_size && end - bkp >= self.min_size {
                    let gain =
                        segment_cost - self.cost.error(start, bkp) - self.cost.error(bkp, end);
                    let cand = (gain, bkp);
                    best = match best {
                        None => Some(cand),
                        Some(b) => {
                            if cand.0 > b.0 || (cand.0 == b.0 && cand.1 > b.1) {
                                Some(cand)
                            } else {
                                Some(b)
                            }
                        }
                    };
                }
                bkp += self.jump;
            }
            match best {
                Some((gain, b)) => (Some(b), gain),
                None => (None, 0.0),
            }
        };
        self.cache.insert((start, end), out);
        out
    }

    pub fn run(
        &mut self,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> Vec<usize> {
        let mut bkps = vec![self.n];
        loop {
            let mut stop = true;
            // candidate split for each current segment
            let mut segs: Vec<(usize, usize)> = Vec::with_capacity(bkps.len());
            let mut start = 0usize;
            for &end in &bkps {
                segs.push((start, end));
                start = end;
            }
            let mut cands: Vec<(Option<usize>, f64)> = Vec::with_capacity(segs.len());
            for (s, e) in segs {
                cands.push(self.single_bkp(s, e));
            }
            // Python's `max(..., key=gain)` returns the FIRST maximum
            let mut chosen = cands[0];
            for &c in cands.iter().skip(1) {
                if c.1 > chosen.1 {
                    chosen = c;
                }
            }
            let (bkp, gain) = chosen;
            let bkp = match bkp {
                None => break,
                Some(b) => b,
            };
            if let Some(target) = n_bkps {
                if bkps.len() - 1 < target {
                    stop = false;
                }
            } else if let Some(p) = pen {
                if gain > p {
                    stop = false;
                }
            } else if let Some(eps) = epsilon {
                if self.cost.sum_of_costs(&bkps) > eps {
                    stop = false;
                }
            }
            if stop {
                break;
            }
            bkps.push(bkp);
            bkps.sort_unstable();
        }
        bkps
    }
}

// ------------------------------------------------------------------ BottomUp

/// Identity of a segment node: `(start, end)`. `ruptures` hashes `Bnode` on
/// exactly this pair, so merge candidates whose children were consumed can be
/// recognised and discarded.
type NodeId = (usize, usize);

/// A merged node's provenance: which two children it came from, and their
/// costs (needed for `gain`, which is `val - (left + right)`).
type Children = (NodeId, NodeId, f64, f64);

#[derive(Clone)]
struct BNode {
    start: usize,
    end: usize,
    val: f64,
    children: Option<Children>,
}

impl BNode {
    fn gain(&self) -> f64 {
        match self.children {
            None => 0.0,
            Some((_, _, lv, rv)) => {
                if self.val.is_infinite() && self.val < 0.0 {
                    0.0
                } else {
                    self.val - (lv + rv)
                }
            }
        }
    }
}

pub struct BottomUp<'a> {
    cost: &'a dyn Cost,
    n: usize,
    jump: usize,
    min_size: usize,
    merge_cache: HashMap<(usize, usize, usize, usize), f64>,
}

impl<'a> BottomUp<'a> {
    pub fn new(cost: &'a dyn Cost, n: usize, jump: usize, min_size: usize) -> Self {
        Self {
            cost,
            n,
            jump,
            min_size,
            merge_cache: HashMap::new(),
        }
    }

    /// Recursively halve the signal at admissible points nearest the midpoint.
    fn grow_tree(&self) -> Vec<BNode> {
        // min-heap on (-length, start, end): the head is the longest segment
        let mut part: Vec<(isize, usize, usize)> = vec![(-(self.n as isize), 0, self.n)];
        loop {
            part.sort();
            let (_, start, end) = part[0];
            let mid = (start + end) as f64 * 0.5;
            let mut best: Option<usize> = None;
            let mut bkp = start;
            while bkp < end {
                if bkp % self.jump == 0
                    && bkp - start >= self.min_size
                    && end - bkp >= self.min_size
                {
                    // first minimum of |bkp - mid| wins
                    best = match best {
                        None => Some(bkp),
                        Some(b) => {
                            if (bkp as f64 - mid).abs() < (b as f64 - mid).abs() {
                                Some(bkp)
                            } else {
                                Some(b)
                            }
                        }
                    };
                }
                bkp += 1;
            }
            match best {
                None => break,
                Some(b) => {
                    part.remove(0);
                    part.push((-((b - start) as isize), start, b));
                    part.push((-((end - b) as isize), b, end));
                }
            }
        }
        part.sort_by_key(|&(_, s, e)| (s, e));
        part.into_iter()
            .map(|(_, s, e)| BNode {
                start: s,
                end: e,
                val: self.cost.error(s, e),
                children: None,
            })
            .collect()
    }

    fn merge(&mut self, left: &BNode, right: &BNode) -> BNode {
        let key = (left.start, left.end, right.start, right.end);
        let val = match self.merge_cache.get(&key) {
            Some(&v) => v,
            None => {
                let v = self.cost.error(left.start, right.end);
                self.merge_cache.insert(key, v);
                v
            }
        };
        BNode {
            start: left.start,
            end: right.end,
            val,
            children: Some((
                (left.start, left.end),
                (right.start, right.end),
                left.val,
                right.val,
            )),
        }
    }

    pub fn run(
        &mut self,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> Vec<usize> {
        let mut leaves = self.grow_tree();
        let mut removed: std::collections::HashSet<(usize, usize)> = Default::default();
        // heap entries ordered by (gain, start), smallest first
        let mut merged: Vec<(Ordf64, usize, BNode)> = Vec::new();
        for i in 0..leaves.len().saturating_sub(1) {
            let c = self.merge(&leaves[i], &leaves[i + 1]);
            merged.push((Ordf64(c.gain()), c.start, c));
        }

        loop {
            let mut stop = true;
            merged.sort_by_key(|a| (a.0, a.1));
            // pop the cheapest merge whose children still exist
            let mut picked: Option<BNode> = None;
            while !merged.is_empty() {
                let (_, _, node) = merged.remove(0);
                if let Some((l, r, _, _)) = node.children {
                    if removed.contains(&l) || removed.contains(&r) {
                        continue;
                    }
                }
                picked = Some(node);
                break;
            }
            let leaf = match picked {
                None => break,
                Some(l) => l,
            };
            let gain = leaf.gain();

            if let Some(target) = n_bkps {
                if leaves.len() > target + 1 {
                    stop = false;
                }
            } else if let Some(p) = pen {
                if gain < p {
                    stop = false;
                }
            } else if let Some(eps) = epsilon {
                if leaves.iter().map(|l| l.val).sum::<f64>() < eps {
                    stop = false;
                }
            }
            if stop {
                break;
            }

            let (lchild, rchild, _, _) = leaf.children.unwrap();
            let left_idx = leaves.partition_point(|n| n.start < lchild.0);
            leaves[left_idx] = leaf.clone();
            leaves.remove(left_idx + 1);
            removed.insert(lchild);
            removed.insert(rchild);

            if left_idx > 0 {
                let c = self.merge(&leaves[left_idx - 1].clone(), &leaves[left_idx].clone());
                merged.push((Ordf64(c.gain()), c.start, c));
            }
            if left_idx + 1 < leaves.len() {
                let c = self.merge(&leaves[left_idx].clone(), &leaves[left_idx + 1].clone());
                merged.push((Ordf64(c.gain()), c.start, c));
            }
        }
        let mut bkps: Vec<usize> = leaves.iter().map(|l| l.end).collect();
        bkps.sort_unstable();
        bkps
    }
}

// ------------------------------------------------------------------ Window

/// `scipy.signal.argrelmax(data, order, mode="wrap")`.
fn argrelmax_wrap(data: &[f64], order: usize) -> Vec<usize> {
    let m = data.len();
    if m == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..m {
        let mut is_max = true;
        for shift in 1..=order {
            let minus = ((i + m * order) - shift) % m;
            let plus = (i + shift) % m;
            let strictly_greater = data[i] > data[minus] && data[i] > data[plus];
            if !strictly_greater {
                is_max = false;
                break;
            }
        }
        if is_max {
            out.push(i);
        }
    }
    out
}

pub fn window_score(
    cost: &dyn Cost,
    n: usize,
    width: usize,
    jump: usize,
) -> (Vec<usize>, Vec<f64>) {
    let w2 = width / 2;
    let inds: Vec<usize> = (0..n)
        .step_by(jump)
        .filter(|&k| k >= w2 && k < n.saturating_sub(w2))
        .collect();
    let score: Vec<f64> = inds
        .iter()
        .map(|&k| {
            let (start, end) = (k - w2, k + w2);
            let gain = cost.error(start, end);
            if gain.is_infinite() && gain < 0.0 {
                return 0.0;
            }
            gain - (cost.error(start, k) + cost.error(k, end))
        })
        .collect();
    (inds, score)
}

#[allow(clippy::too_many_arguments)]
pub fn window_seg(
    cost: &dyn Cost,
    n: usize,
    inds: &[usize],
    score: &[f64],
    width: usize,
    jump: usize,
    min_size: usize,
    n_bkps: Option<usize>,
    pen: Option<f64>,
    epsilon: Option<f64>,
) -> Vec<usize> {
    let mut bkps = vec![n];
    let mut error = cost.sum_of_costs(&bkps);
    let order = std::cmp::max(std::cmp::max(width, 2 * min_size) / (2 * jump), 1);
    let peaks = argrelmax_wrap(score, order);
    if peaks.is_empty() {
        return bkps;
    }
    // sort ascending by (gain, index); the loop pops from the back
    let mut ranked: Vec<(Ordf64, usize)> =
        peaks.iter().map(|&p| (Ordf64(score[p]), inds[p])).collect();
    ranked.sort_by_key(|a| (a.0, a.1));
    let mut peak_inds: Vec<usize> = ranked.into_iter().map(|(_, i)| i).collect();

    loop {
        let mut stop = true;
        let bkp = match peak_inds.pop() {
            None => break,
            Some(b) => b,
        };
        if let Some(target) = n_bkps {
            if bkps.len() - 1 < target {
                stop = false;
            }
        } else if let Some(p) = pen {
            let mut trial = bkps.clone();
            trial.push(bkp);
            trial.sort_unstable();
            if error - cost.sum_of_costs(&trial) > p {
                stop = false;
            }
        } else if let Some(eps) = epsilon {
            if error > eps {
                stop = false;
            }
        }
        if stop {
            break;
        }
        bkps.push(bkp);
        bkps.sort_unstable();
        error = cost.sum_of_costs(&bkps);
    }
    bkps
}
