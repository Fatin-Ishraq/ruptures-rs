"""Benchmark `ruptures_rs` against `ruptures` on identical inputs.

Both libraries get the same signal and the same parameters, and the breakpoints
are compared, so a timing is only reported alongside proof that the two agree —
a speedup on a different answer would mean nothing.

Sizes are chosen so the *reference* can finish. Where it cannot, the row is
marked `n/a` and only this package's time is shown; those rows are the point,
not a dodge: they are workloads `ruptures` cannot run at all.

Every case is the best of as many runs as fit in a twenty-second budget, so a
timing is not reported from a single noisy sample unless the case is too slow
to repeat at all.

    python bench/bench.py                       # full table
    python bench/bench.py --quick               # smaller sizes
    python bench/bench.py --json results.json   # machine-readable
    python bench/bench.py --section models      # one section only
    python bench/bench.py --render results.json # re-print a saved table

The full table is about twenty minutes, nearly all of it spent inside
`ruptures`. `--section` exists so it can be rebuilt in pieces, and so a change
to one cost can be re-measured without paying for the rest; `--append` merges
the result back into an existing JSON file instead of replacing it.
"""

import argparse
import json
import re
import sys
import time
import warnings

import numpy as np

import ruptures as rpt_py
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)

HEADER = f"{'case':<38s} {'ruptures':>12s} {'ruptures-rs':>12s} {'speedup':>11s}   match"

#: Independently runnable pieces of the table. `pelt-large` is its own section
#: because that single row costs `ruptures` twelve minutes, more than every
#: other row put together.
SECTIONS = (
    "dynp",
    "huge",
    "pelt",
    "pelt-large",
    "greedy",
    "models",
    "kernelcpd",
    "crops",
)


#: Keep repeating a case until this much wall clock has gone into it, then
#: stop. A millisecond case gets seven runs, a ten-second case gets two, and a
#: six-minute case gets one — which is the best that can be done for the rows
#: where the reference is the whole cost.
REPEAT_BUDGET_SECONDS = 20.0
MAX_REPEATS = 7


def timed(fn, budget=REPEAT_BUDGET_SECONDS, max_repeats=MAX_REPEATS):
    """Best of several runs, where several runs are cheap.

    A sub-millisecond measurement on a desktop swings by more than a factor of
    two between runs — enough to move a reported speedup from 783x to 1,803x
    for the same code — and even a ten-second `ruptures` case was observed to
    vary by 60%. Taking the minimum over repeats removes most of that. The
    budget is what stops a case that already took six minutes from being run
    seven times to confirm that it still takes six minutes.
    """
    best = None
    out = None
    spent = 0.0
    for _ in range(max_repeats):
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        best = elapsed if best is None else min(best, elapsed)
        spent += elapsed
        if spent >= budget:
            break
    return best, out


def signal_for(n, d=1, n_bkps=5, seed=3):
    return rpt_rs.pw_constant(n, d, n_bkps, noise_std=2, seed=seed)[0]


def report(row):
    py = "n/a" if row["ruptures"] is None else f"{row['ruptures']:.3f}s"
    rs = f"{row['ruptures_rs']:.4f}s"
    sp = "n/a" if row["speedup"] is None else f"{row['speedup']:,.0f}x"
    ok = {True: "yes", False: "MISMATCH", None: "-"}[row["match"]]
    print(f"{row['case']:<38s} {py:>12s} {rs:>12s} {sp:>11s}   {ok}", flush=True)


def compare(label, py_fn, rs_fn, skip_py=False):
    rs_time, rs_out = timed(rs_fn)
    if skip_py:
        row = dict(case=label, ruptures=None, ruptures_rs=rs_time, speedup=None, match=None)
    else:
        py_time, py_out = timed(py_fn)
        row = dict(
            case=label,
            ruptures=py_time,
            ruptures_rs=rs_time,
            speedup=py_time / rs_time if rs_time > 0 else float("inf"),
            match=[int(x) for x in py_out] == [int(x) for x in rs_out],
        )
    report(row)
    return row


#: Canonical row order, so a table assembled from several `--section` runs
#: reads the same as one built in a single pass.
ROW_ORDER = (
    "Dynp l2",
    "Pelt l2",
    "Binseg l2",
    "BottomUp l2",
    "Window l2",
    "Dynp l1",
    "Dynp normal",
    "Dynp rank",
    "Dynp mahalanobis",
    "Dynp rbf",
    "Dynp linear",
    "Dynp ar",
    "KernelCPD",
    "Crops",
)


def render(path):
    """Print a saved results file as the table, in canonical order."""
    with open(path) as fh:
        rows = json.load(fh)

    def rank(row):
        # Sort by the signal size, not by the label: "n=20,000" sorts before
        # "n=4,000" as text, which puts the table in a nonsensical order.
        match = re.search(r"n=([\d,]+)", row["case"])
        size = int(match.group(1).replace(",", "")) if match else 0
        for i, prefix in enumerate(ROW_ORDER):
            if row["case"].startswith(prefix):
                return (i, size, row["case"])
        return (len(ROW_ORDER), size, row["case"])

    print(HEADER, flush=True)
    print("-" * len(HEADER), flush=True)
    for row in sorted(rows, key=rank):
        report(row)
    print("-" * len(HEADER), flush=True)
    compared = sum(1 for r in rows if r["match"] is True)
    mismatched = [r["case"] for r in rows if r["match"] is False]
    if mismatched:
        print(f"MISMATCHES: {mismatched}", flush=True)
        sys.exit(1)
    print(f"all {compared} compared cases returned identical breakpoints", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    ap.add_argument(
        "--section",
        default="all",
        help="comma-separated subset of: " + ", ".join(SECTIONS),
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="merge into an existing --json file instead of replacing it",
    )
    ap.add_argument(
        "--render",
        metavar="RESULTS_JSON",
        help="print the table from a saved results file and exit, without "
        "re-running anything",
    )
    args = ap.parse_args()

    if args.render:
        render(args.render)
        return

    wanted = (
        set(SECTIONS)
        if args.section == "all"
        else {part.strip() for part in args.section.split(",")}
    )
    unknown = sorted(wanted - set(SECTIONS))
    if unknown:
        ap.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    rows = []
    print(
        f"python {sys.version.split()[0]} | ruptures {rpt_py.__version__} "
        f"| ruptures-rs {rpt_rs.__version__}",
        flush=True,
    )
    print(HEADER, flush=True)
    print("-" * len(HEADER), flush=True)

    # ---------------------------------------------------------------- Dynp
    # The headline. `ruptures` evaluates an O(len) NumPy call at every cell of
    # an O(n^2 K) table; here each cell is a few scalar operations.
    if "dynp" in wanted:
        for n in [500, 1000, 2000] if args.quick else [500, 1000, 2000, 4000]:
            sig = signal_for(n)
            rows.append(
                compare(
                    f"Dynp l2  n={n:,} K=5 jump=1",
                    lambda s=sig: rpt_py.Dynp(model="l2", min_size=10, jump=1)
                    .fit(s)
                    .predict(5),
                    lambda s=sig: rpt_rs.Dynp(model="l2", min_size=10, jump=1)
                    .fit(s)
                    .predict(5),
                )
            )

    # Sizes the reference cannot reach: n=50,000 exact DP would take `ruptures`
    # weeks by extrapolation from the rows above.
    if "huge" in wanted:
        for n in [10_000] if args.quick else [20_000, 50_000]:
            sig = signal_for(n)
            rows.append(
                compare(
                    f"Dynp l2  n={n:,} K=5 jump=10",
                    None,
                    lambda s=sig: rpt_rs.Dynp(model="l2", min_size=10, jump=10)
                    .fit(s)
                    .predict(5),
                    skip_py=True,
                )
            )

    # ---------------------------------------------------------------- Pelt
    pelt_sizes = []
    if "pelt" in wanted:
        pelt_sizes += [1_000, 5_000]
    if "pelt-large" in wanted and not args.quick:
        pelt_sizes += [20_000]
    for n in pelt_sizes:
        sig = signal_for(n)
        rows.append(
            compare(
                f"Pelt l2  n={n:,} pen=200 jump=1",
                lambda s=sig: rpt_py.Pelt(model="l2", min_size=10, jump=1)
                .fit(s)
                .predict(200),
                lambda s=sig: rpt_rs.Pelt(model="l2", min_size=10, jump=1)
                .fit(s)
                .predict(200),
            )
        )

    # ------------------------------------------ Binseg / BottomUp / Window
    if "greedy" in wanted:
        n = 2_000 if args.quick else 5_000
        sig = signal_for(n)
        for name in ("Binseg", "BottomUp"):
            rows.append(
                compare(
                    f"{name} l2  n={n:,} K=5 jump=1",
                    lambda nm=name, s=sig: getattr(rpt_py, nm)(
                        model="l2", min_size=10, jump=1
                    )
                    .fit(s)
                    .predict(n_bkps=5),
                    lambda nm=name, s=sig: getattr(rpt_rs, nm)(
                        model="l2", min_size=10, jump=1
                    )
                    .fit(s)
                    .predict(n_bkps=5),
                )
            )
        rows.append(
            compare(
                f"Window l2  n={n:,} width=100 jump=1",
                lambda s=sig: rpt_py.Window(width=100, model="l2", min_size=10, jump=1)
                .fit(s)
                .predict(n_bkps=5),
                lambda s=sig: rpt_rs.Window(width=100, model="l2", min_size=10, jump=1)
                .fit(s)
                .predict(n_bkps=5),
            )
        )

    # ------------------------------------------------------- other models
    if "models" in wanted:
        n = 800 if args.quick else 1_500
        for model, d in (
            ("l1", 1),
            ("normal", 1),
            ("rank", 3),
            ("mahalanobis", 3),
            ("rbf", 1),
        ):
            sig = signal_for(n, d=d, seed=5)
            rows.append(
                compare(
                    f"Dynp {model}  n={n:,} K=4 jump=5",
                    lambda m=model, s=sig: rpt_py.Dynp(model=m, min_size=10, jump=5)
                    .fit(s)
                    .predict(4),
                    lambda m=model, s=sig: rpt_rs.Dynp(model=m, min_size=10, jump=5)
                    .fit(s)
                    .predict(4),
                )
            )

        # The regression costs need their own signals, and they are the two
        # that gain least: each segment cost is a small least-squares solve,
        # and the rank-revealing factorisation that keeps it agreeing with
        # NumPy costs more than a Cholesky would. Reported, not left out.
        lin = rpt_rs.pw_linear(n, 2, 4, noise_std=1, seed=5)[0]
        rows.append(
            compare(
                f"Dynp linear  n={n:,} K=4 jump=5",
                lambda s=lin: rpt_py.Dynp(model="linear", min_size=10, jump=5)
                .fit(s)
                .predict(4),
                lambda s=lin: rpt_rs.Dynp(model="linear", min_size=10, jump=5)
                .fit(s)
                .predict(4),
            )
        )
        wavy = rpt_rs.pw_wavy(n, 4, noise_std=0.3, seed=5)[0]
        rows.append(
            compare(
                f"Dynp ar  n={n:,} K=4 jump=5",
                lambda s=wavy: rpt_py.Dynp(model="ar", min_size=10, jump=5)
                .fit(s)
                .predict(4),
                lambda s=wavy: rpt_rs.Dynp(model="ar", min_size=10, jump=5)
                .fit(s)
                .predict(4),
            )
        )

    # ---------------------------------------------------------- KernelCPD
    # The one detector `ruptures` already compiles, so this row is the honest
    # comparison: Rust against C, not Rust against NumPy.
    if "kernelcpd" in wanted:
        for n_k in [2_000] if args.quick else [2_000, 10_000]:
            sig = signal_for(n_k, seed=7)
            rows.append(
                compare(
                    f"KernelCPD linear  n={n_k:,} K=5",
                    lambda s=sig: rpt_py.KernelCPD(kernel="linear", min_size=10)
                    .fit(s)
                    .predict(n_bkps=5),
                    lambda s=sig: rpt_rs.KernelCPD(kernel="linear", min_size=10)
                    .fit(s)
                    .predict(n_bkps=5),
                )
            )

    # -------------------------------------------------------------- CROPS
    # No `ruptures` equivalent, so the honest comparison is against what a user
    # would otherwise do: sweep a grid of penalties with PELT.
    if "crops" in wanted:
        sig = signal_for(2_000, seed=9)
        t_crops, regimes = timed(
            lambda: rpt_rs.Crops(model="l2", min_size=10, jump=5).fit_predict(
                sig, 1.0, 10_000.0
            )
        )
        grid = np.geomspace(1.0, 10_000.0, 50)
        t_grid, grid_out = timed(
            lambda: [
                rpt_py.Pelt(model="l2", min_size=10, jump=5).fit(sig).predict(float(p))
                for p in grid
            ]
        )
        grid_counts = {len(b) - 1 for b in grid_out}
        crops_counts = {r.n_bkps for r in regimes}
        print("-" * len(HEADER), flush=True)
        print(
            f"CROPS  n=2,000, penalties [1, 10000]: {len(regimes)} distinct "
            f"segmentations in {t_crops:.4f}s",
            flush=True,
        )
        print(
            f"  50-point ruptures grid over the same range: {t_grid:.3f}s, "
            f"{len(grid_counts)} distinct segmentations "
            f"({t_grid / t_crops:,.0f}x slower, "
            f"{len(crops_counts - grid_counts)} regimes missed)",
            flush=True,
        )
        rows.append(
            dict(
                case="Crops n=2,000 vs 50-point grid",
                ruptures=t_grid,
                ruptures_rs=t_crops,
                speedup=t_grid / t_crops,
                match=None,
                crops_regimes=len(crops_counts),
                grid_regimes=len(grid_counts),
            )
        )

    print("-" * len(HEADER), flush=True)
    bad = [r["case"] for r in rows if r["match"] is False]
    if bad:
        print(f"MISMATCHES: {bad}", flush=True)
        sys.exit(1)
    compared = sum(1 for r in rows if r["match"] is True)
    print(f"all {compared} compared cases returned identical breakpoints", flush=True)

    if args.json:
        merged = rows
        if args.append:
            # Keyed by case, so re-running one section replaces its own rows
            # rather than appending a second copy of them.
            try:
                with open(args.json) as fh:
                    previous = json.load(fh)
            except (OSError, json.JSONDecodeError):
                previous = []
            fresh = {r["case"] for r in rows}
            merged = [r for r in previous if r["case"] not in fresh] + rows
        with open(args.json, "w") as fh:
            json.dump(merged, fh, indent=2)


if __name__ == "__main__":
    main()
