"""Benchmark `ruptures_rs` against `ruptures` on identical inputs.

Both libraries get the same signal and the same parameters, and the breakpoints
are compared, so a timing is only reported alongside proof that the two agree —
a speedup on a different answer would mean nothing.

Sizes are chosen so the *reference* can finish. Where it cannot, the row is
marked `n/a` and only this package's time is shown; those rows are the point,
not a dodge: they are workloads `ruptures` cannot run at all.

    python bench/bench.py             # full table
    python bench/bench.py --quick     # smaller sizes
    python bench/bench.py --json results.json
"""

import argparse
import json
import sys
import time
import warnings

import numpy as np

import ruptures as rpt_py
import ruptures_rs as rpt_rs

warnings.filterwarnings("ignore", category=UserWarning)

HEADER = f"{'case':<38s} {'ruptures':>12s} {'ruptures-rs':>12s} {'speedup':>11s}   match"


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = []
    print(f"python {sys.version.split()[0]} | ruptures {rpt_py.__version__} "
          f"| ruptures-rs {rpt_rs.__version__}", flush=True)
    print(HEADER, flush=True)
    print("-" * len(HEADER), flush=True)

    # ---------------------------------------------------------------- Dynp
    # The headline. `ruptures` evaluates an O(len) NumPy call at every cell of
    # an O(n^2 K) table; here each cell is a few scalar operations.
    for n in [500, 1000, 2000] if args.quick else [500, 1000, 2000, 4000]:
        sig = signal_for(n)
        rows.append(
            compare(
                f"Dynp l2  n={n:,} K=5 jump=1",
                lambda s=sig: rpt_py.Dynp(model="l2", min_size=10, jump=1).fit(s).predict(5),
                lambda s=sig: rpt_rs.Dynp(model="l2", min_size=10, jump=1).fit(s).predict(5),
            )
        )

    # Sizes the reference cannot reach: n=50,000 exact DP would take `ruptures`
    # weeks by extrapolation from the rows above.
    for n in [10_000] if args.quick else [20_000, 50_000]:
        sig = signal_for(n)
        rows.append(
            compare(
                f"Dynp l2  n={n:,} K=5 jump=10",
                None,
                lambda s=sig: rpt_rs.Dynp(model="l2", min_size=10, jump=10).fit(s).predict(5),
                skip_py=True,
            )
        )

    # ---------------------------------------------------------------- Pelt
    for n in [1_000, 5_000] if args.quick else [1_000, 5_000, 20_000]:
        sig = signal_for(n)
        rows.append(
            compare(
                f"Pelt l2  n={n:,} pen=200 jump=1",
                lambda s=sig: rpt_py.Pelt(model="l2", min_size=10, jump=1).fit(s).predict(200),
                lambda s=sig: rpt_rs.Pelt(model="l2", min_size=10, jump=1).fit(s).predict(200),
            )
        )

    # ------------------------------------------ Binseg / BottomUp / Window
    n = 2_000 if args.quick else 5_000
    sig = signal_for(n)
    for name in ("Binseg", "BottomUp"):
        rows.append(
            compare(
                f"{name} l2  n={n:,} K=5 jump=1",
                lambda nm=name, s=sig: getattr(rpt_py, nm)(model="l2", min_size=10, jump=1)
                .fit(s)
                .predict(n_bkps=5),
                lambda nm=name, s=sig: getattr(rpt_rs, nm)(model="l2", min_size=10, jump=1)
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
    n = 800 if args.quick else 1_500
    for model, d in (("l1", 1), ("normal", 1), ("rank", 3), ("mahalanobis", 3), ("rbf", 1)):
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

    # -------------------------------------------------------------- CROPS
    # No `ruptures` equivalent, so the honest comparison is against what a user
    # would otherwise do: sweep a grid of penalties with PELT.
    sig = signal_for(2_000, seed=9)
    t_crops, regimes = timed(
        lambda: rpt_rs.Crops(model="l2", min_size=10, jump=5).fit_predict(sig, 1.0, 10_000.0)
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
        f"CROPS  n=2,000, penalties [1, 10000]: {len(regimes)} distinct segmentations "
        f"in {t_crops:.4f}s",
        flush=True,
    )
    print(
        f"  50-point ruptures grid over the same range: {t_grid:.3f}s, "
        f"{len(grid_counts)} distinct segmentations "
        f"({t_grid / t_crops:,.0f}x slower, {len(crops_counts - grid_counts)} regimes missed)",
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
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
