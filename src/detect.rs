//! Change point detectors.
//!
//! These are faithful ports of the `ruptures` algorithms, including their
//! tie-breaking rules, which are load-bearing: `ruptures` leans on Python's
//! `min`/`max` returning the *first* extremum, so a Rust port that used
//! `Iterator::max_by` (which returns the *last*) would silently produce
//! different — though equally optimal — breakpoints. Python's `min` also keeps
//! the first element when it cannot be compared, which is why every search
//! below seeds itself from the first candidate rather than from infinity.
//!
//! Where the shape of the computation is changed, it is changed without
//! changing the answer:
//!
//! * `Dynp` is a bottom-up table rather than a memoised recursion. Costs are
//!   accumulated left to right, matching `sum(partition.values())` on a dict
//!   whose insertion order is left to right, so the floating-point association
//!   order is preserved.
//! * `Pelt` keeps a scalar running total plus a back-pointer instead of copying
//!   a whole partition dict per candidate. The penalty is folded in per segment
//!   — `total + (cost + pen)` — because that is how summing the dict's values
//!   associates, and the alternative differs in the last ulp often enough to
//!   move a breakpoint on signals with exact ties.
//!
//! One divergence is deliberate, and it is in `pelt`: the reference prunes a
//! candidate at a point from which the pruning inequality does not yet hold,
//! and so can return a segmentation that is not optimal. See the note there.

use crate::cost::{Cost, KernelBlocks};
use rayon::prelude::*;
use std::collections::HashMap;

pub const INF: f64 = f64::INFINITY;

/// Port of `ruptures.utils.sanity_check`.
pub fn sanity_check(n_samples: usize, n_bkps: usize, jump: usize, min_size: usize) -> bool {
    if jump == 0 {
        return false;
    }
    let n_adm_bkps = n_samples / jump;
    if n_bkps > n_adm_bkps {
        return false;
    }
    let ceil_div = min_size.div_ceil(jump);
    match n_bkps
        .checked_mul(ceil_div)
        .and_then(|v| v.checked_mul(jump))
        .and_then(|v| v.checked_add(min_size))
    {
        Some(need) => need <= n_samples,
        None => false,
    }
}

/// Total ordering wrapper so floats can live in a sorted list or heap.
///
/// Backed by `f64::total_cmp`: `partial_cmp(..).unwrap_or(Equal)` looks
/// harmless but is not transitive once a NaN is present, and Rust's sort
/// detects the broken order and panics — straight through the FFI boundary,
/// where it becomes a `BaseException` that user code cannot catch.
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
        self.0.total_cmp(&other.0)
    }
}

// ------------------------------------------------------------------ Dynp

/// Back-pointer tables from one bottom-up dynamic program.
///
/// Level `k - 1` holds, for every endpoint, the last breakpoint of the optimal
/// `k`-breakpoint segmentation. Keeping every level means a single O(n^2 K)
/// pass answers every `k <= n_bkps`, which is what `KernelCPD` needs to fill
/// its `segmentations_dict`.
pub struct DynpTables {
    back: Vec<Vec<u32>>,
    n: usize,
}

impl DynpTables {
    /// Breakpoints of the optimal `k`-breakpoint segmentation, or `None` when
    /// no admissible partition exists.
    pub fn backtrack(&self, k: usize) -> Option<Vec<usize>> {
        if k > self.back.len() {
            return None;
        }
        let mut bkps = vec![self.n];
        let mut pos = self.n;
        for level in (0..k).rev() {
            let b = self.back[level][pos];
            if b == u32::MAX {
                return None;
            }
            pos = b as usize;
            bkps.push(pos);
        }
        bkps.sort_unstable();
        bkps.dedup();
        Some(bkps)
    }
}

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
pub fn dynp_tables(
    cost: &dyn Cost,
    n: usize,
    n_bkps: usize,
    jump: usize,
    min_size: usize,
) -> DynpTables {
    // Admissible breakpoint positions are the multiples of `jump` below `n`.
    let cand: Vec<usize> = (0..n).step_by(jump.max(1)).collect();
    let mut targets = cand.clone();
    targets.push(n);

    let mut level: Vec<f64> = vec![INF; n + 1];
    // Reachability is tracked separately from the value. Using `is_finite` on
    // the value conflates "no partition exists here" with "the cost function
    // legitimately returned -inf or NaN here", and `ruptures` admits both of
    // those: a constant segment under `normal` with `add_small_diag=False` has
    // a cost of exactly -inf, and it is frequently the optimum.
    let mut computed: Vec<bool> = vec![false; n + 1];
    for &e in &targets {
        if e >= min_size {
            level[e] = cost.error(0, e);
            computed[e] = true;
        }
    }

    let mut back: Vec<Vec<u32>> = Vec::with_capacity(n_bkps);

    for k in 1..=n_bkps {
        let prev = &level;
        let prev_ok = &computed;
        // Only breakpoints that leave a feasible left subproblem are admissible.
        let adm: Vec<usize> = cand
            .iter()
            .copied()
            .filter(|&b| sanity_check(b, k - 1, jump, min_size) && prev_ok[b])
            .collect();

        let computed_cells: Vec<(usize, f64, u32)> = targets
            .par_iter()
            .map(|&e| {
                let mut best = INF;
                let mut arg = u32::MAX;
                for &b in &adm {
                    if b >= e || e - b < min_size {
                        continue;
                    }
                    let v = prev[b] + cost.error(b, e);
                    // Seed from the first candidate, then take a strict
                    // improvement: exactly Python's `min`, including when the
                    // running best is a NaN that nothing compares less than.
                    if arg == u32::MAX || v < best {
                        best = v;
                        arg = b as u32;
                    }
                }
                (e, best, arg)
            })
            .collect();

        let mut next = vec![INF; n + 1];
        let mut next_ok = vec![false; n + 1];
        let mut bk = vec![u32::MAX; n + 1];
        for (e, v, a) in computed_cells {
            if a != u32::MAX {
                next[e] = v;
                next_ok[e] = true;
            }
            bk[e] = a;
        }
        level = next;
        computed = next_ok;
        back.push(bk);
    }

    DynpTables { back, n }
}

/// Convenience wrapper for a single `n_bkps`.
pub fn dynp(
    cost: &dyn Cost,
    n: usize,
    n_bkps: usize,
    jump: usize,
    min_size: usize,
) -> Option<Vec<usize>> {
    dynp_tables(cost, n, n_bkps, jump, min_size).backtrack(n_bkps)
}

// ------------------------------------------------------------------ Pelt

/// Penalised segmentation with pruning.
///
/// PELT's pruning rule is that a candidate `t` can be discarded at target `s`
/// once `F(t) + C(t, s) >= F(s)`, because for any later target `u`
///
/// ```text
/// F(t) + C(t, u) + pen  >=  F(t) + C(t, s) + C(s, u) + pen
///                       >=  F(s) + C(s, u) + pen
///                       >=  F(u)
/// ```
///
/// so `t` can never beat what is already known. The last step needs `s` to be a
/// *legal* last breakpoint for `u` — and with a minimum segment length it is
/// not, for every `u` closer to `s` than `min_size`. `ruptures` prunes anyway,
/// and the consequence is not academic: on 24 standard normal samples with
/// `min_size=6, jump=1, pen=2`, it discards `t = 0` at `s = 21` and then cannot
/// see the unsplit signal at `u = 24` — returning a segmentation costing
/// `29.808` when leaving the signal alone costs `29.187`.
///
/// The repair keeps the same rule and the same amount of pruning; it only
/// delays the *removal*. A candidate that fails the test at `s` is marked, and
/// dropped at the first target `min_size` or more beyond `s`, which is exactly
/// the point from which the inequality above holds. Every candidate is still
/// removed once, so the cost of pruning is unchanged.
///
/// This makes `Pelt` exact where the reference is not, and the two can
/// therefore return different answers. When they do, this one's objective is
/// the smaller — never the other way round, which is what
/// `tests/test_optimality.py` asserts against an exhaustive search.
pub fn pelt(cost: &dyn Cost, n: usize, pen: f64, jump: usize, min_size: usize) -> Vec<usize> {
    let jump = jump.max(1);
    let mut totals: HashMap<usize, f64> = HashMap::new();
    let mut prev: HashMap<usize, usize> = HashMap::new();
    totals.insert(0, 0.0);

    let mut ind: Vec<usize> = (0..n).step_by(jump).filter(|&k| k >= min_size).collect();
    ind.push(n);

    // Admissible positions are kept signed, because `floor((bkp - min_size) /
    // jump) * jump` is negative when the signal is shorter than `min_size` and
    // clamping that to zero would invent a partition `ruptures` does not have.
    //
    // The second field is the target at which the candidate failed the pruning
    // test, if it has: it stays usable until `min_size` past that point.
    let mut admissible: Vec<(isize, Option<usize>)> = Vec::new();

    for &bkp in &ind {
        admissible.retain(|&(_, pruned_at)| match pruned_at {
            None => true,
            Some(s) => bkp - s < min_size,
        });
        let new_adm = (bkp as isize - min_size as isize).div_euclid(jump as isize) * jump as isize;
        admissible.push((new_adm, None));

        // (total, t, slot) for every admissible t that has a recorded partition.
        // The slot is carried so the pruning decision lands on the candidate it
        // was computed for; `ruptures` zips the two lists positionally even
        // though one of them skips entries, which is a misalignment waiting for
        // the right parameters.
        let mut subproblems: Vec<(f64, usize, usize)> = Vec::with_capacity(admissible.len());
        for (slot, &(t, _)) in admissible.iter().enumerate() {
            if t < 0 {
                continue;
            }
            let t = t as usize;
            if let Some(&tot) = totals.get(&t) {
                // `total + (cost + pen)`, matching how Python sums the
                // partition dict: one `cost + pen` term per segment, added
                // left to right.
                subproblems.push((tot + (cost.error(t, bkp) + pen), t, slot));
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

        // A position below `min_size` never receives a partition of its own, so
        // it can never contribute and is dropped outright. Everything else is
        // marked rather than removed.
        let cutoff = best.0 + pen;
        let mut verdict: Vec<Option<bool>> = vec![None; admissible.len()];
        for &(value, _, slot) in &subproblems {
            verdict[slot] = Some(value <= cutoff);
        }
        let mut kept: Vec<(isize, Option<usize>)> = Vec::with_capacity(admissible.len());
        for (slot, &(t, pruned_at)) in admissible.iter().enumerate() {
            match verdict[slot] {
                None => continue,
                Some(true) => kept.push((t, pruned_at)),
                // Keep the *earliest* failure: the candidate becomes droppable
                // sooner, and the inequality holds from that point on.
                Some(false) => kept.push((t, pruned_at.or(Some(bkp)))),
            }
        }
        admissible = kept;
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

// ------------------------------------------------- KernelCPD (C extension)

/// Transcription of `ekcpd_pelt_compute` from `ruptures`' C extension.
///
/// The Python `Pelt` and this share an algorithm but not their arithmetic: the
/// C version folds the penalty in as `(total + cost) + beta`, prunes a
/// monotonically advancing prefix of candidates rather than filtering the whole
/// list, and always keeps `s = 0` in play. Those choices pick different
/// members of a tie, so `KernelCPD(...).predict(pen=...)` gets its own
/// transcription instead of being routed through `pelt` above.
pub fn kernel_pelt(kernel: &dyn KernelBlocks, n: usize, beta: f64, min_size: usize) -> Vec<usize> {
    let mut m_v = vec![0.0f64; n + 1];
    let mut m_path = vec![0usize; n + 1];
    let mut m_pruning = vec![0.0f64; n + 1];
    let mut s_min: usize = 0;

    let seg_cost = |s: usize, t: usize| -> f64 {
        kernel.diag_sum(s, t) - kernel.block_sum(s, t) / (t - s) as f64
    };

    // For t < 2 * min_size there cannot be any change point.
    let head = std::cmp::min(2 * min_size, n + 1);
    for (t, slot) in m_v.iter_mut().enumerate().take(head).skip(1) {
        *slot = seg_cost(0, t) + beta;
    }

    for t in (2 * min_size)..=n {
        let mut s = s_min;
        let mut c_cost_sum = m_v[s] + seg_cost(s, t);
        m_pruning[s] = c_cost_sum;
        c_cost_sum += beta;
        m_v[t] = c_cost_sum;
        m_path[t] = s;

        let upper = t - min_size + 1;
        s = std::cmp::max(s_min + 1, min_size);
        while s < upper {
            let mut c_cost_sum = m_v[s] + seg_cost(s, t);
            m_pruning[s] = c_cost_sum;
            c_cost_sum += beta;
            if m_v[t] > c_cost_sum {
                m_v[t] = c_cost_sum;
                m_path[t] = s;
            }
            s += 1;
        }

        while m_pruning[s_min] >= m_v[t] && s_min < upper {
            if s_min == 0 {
                s_min += min_size;
            } else {
                s_min += 1;
            }
        }
    }

    let mut bkps = Vec::new();
    let mut ind = n;
    while ind > 0 {
        bkps.push(ind);
        let next = m_path[ind];
        if next >= ind {
            break;
        }
        ind = next;
    }
    bkps.reverse();
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
            jump: jump.max(1),
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
            jump: jump.max(1),
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
    // With `mode="wrap"`, a shift of `m` compares a point against itself, and
    // `data[i] > data[i]` is false — so an order that reaches all the way round
    // has no strict maxima at all. Returning that directly also keeps the loop
    // below from running `order` times per point when `order` is enormous.
    if order >= m {
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
        .step_by(jump.max(1))
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
    let jump = jump.max(1);
    let mut bkps = vec![n];
    let mut error = cost.sum_of_costs(&bkps);
    // Saturating throughout. `jump` reaches here as whatever the caller put in
    // the constructor, and `Window(width=4, jump=2**63)` overflowed `2 * jump`
    // to zero in release mode — a division by zero, which is a panic, which
    // crosses the FFI boundary as something `except Exception` cannot catch.
    let order = std::cmp::max(
        std::cmp::max(width, min_size.saturating_mul(2)) / jump.saturating_mul(2).max(1),
        1,
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::CostL2;

    fn step_signal() -> (Vec<f64>, usize) {
        let mut sig = vec![0.0; 120];
        for (i, v) in sig.iter_mut().enumerate() {
            *v = if i < 60 { 0.0 } else { 5.0 };
        }
        (sig, 120)
    }

    #[test]
    fn sanity_check_rejects_zero_jump_instead_of_dividing_by_it() {
        assert!(!sanity_check(100, 1, 0, 2));
        assert!(sanity_check(100, 1, 5, 2));
    }

    #[test]
    fn dynp_finds_the_obvious_step() {
        let (sig, n) = step_signal();
        let cost = CostL2::new(&sig, n, 1);
        let bkps = dynp(&cost, n, 1, 1, 2).expect("feasible");
        assert_eq!(bkps, vec![60, 120]);
    }

    #[test]
    fn dynp_reports_infeasibility_rather_than_a_short_answer() {
        let (sig, n) = step_signal();
        let cost = CostL2::new(&sig, n, 1);
        // 40 breakpoints with min_size 10 cannot fit in 120 samples.
        assert!(dynp(&cost, n, 40, 1, 10).is_none());
    }

    #[test]
    fn dynp_tables_answer_every_smaller_k() {
        let (sig, n) = step_signal();
        let cost = CostL2::new(&sig, n, 1);
        let tables = dynp_tables(&cost, n, 3, 1, 5);
        for k in 1..=3 {
            let bkps = tables.backtrack(k).expect("feasible");
            assert_eq!(bkps.len(), k + 1);
            assert_eq!(*bkps.last().unwrap(), n);
        }
    }

    #[test]
    fn pelt_recovers_the_step_at_a_moderate_penalty() {
        let (sig, n) = step_signal();
        let cost = CostL2::new(&sig, n, 1);
        assert_eq!(pelt(&cost, n, 10.0, 1, 5), vec![60, 120]);
    }

    #[test]
    fn argrelmax_wrap_matches_scipy_on_a_single_peak() {
        let data = [0.0, 1.0, 5.0, 1.0, 0.0];
        assert_eq!(argrelmax_wrap(&data, 1), vec![2]);
        assert_eq!(argrelmax_wrap(&data, 2), vec![2]);
        // A constant score has no strict maxima.
        assert!(argrelmax_wrap(&[1.0, 1.0, 1.0], 1).is_empty());
    }
}
