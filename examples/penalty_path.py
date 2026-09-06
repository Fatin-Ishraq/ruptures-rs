"""Choosing a penalty without guessing: the CROPS penalty path.

Penalised change point detection asks you for a penalty, and nobody knows theirs
in advance. The usual approaches are both unsatisfying:

* pick a number, get a segmentation, and never learn whether a slightly
  different number would have given a very different answer;
* sweep a grid, pay for many redundant PELT runs, and still step over
  segmentations that are optimal only on a narrow interval.

`Crops` answers the real question — *what are all the answers?* — by returning
every segmentation that is optimal somewhere in a penalty range, together with
the exact interval each one owns.

Run:  python examples/penalty_path.py
"""

import numpy as np

import ruptures_rs as rpt

PEN_MIN, PEN_MAX = 1.0, 20_000.0


def main():
    signal, true_bkps = rpt.pw_constant(2_000, 1, 6, noise_std=3, seed=11)
    n_true = len(true_bkps) - 1

    regimes = rpt.Crops(model="l2", min_size=10, jump=5).fit_predict(
        signal, PEN_MIN, PEN_MAX
    )

    print(f"signal: {signal.shape[0]:,} samples, {n_true} true changes")
    print(
        f"{len(regimes)} distinct segmentations are optimal somewhere "
        f"in [{PEN_MIN:,.0f}, {PEN_MAX:,.0f}]\n"
    )

    print(f"{'changes':>7}  {'penalty interval':>26}  {'width':>10}  breakpoints")
    print("-" * 88)
    for regime in regimes[:12]:
        span = f"[{regime.pen_min:>9,.1f}, {regime.pen_max:>9,.1f}]"
        width = regime.pen_max - regime.pen_min
        shown = regime.bkps[:-1]
        preview = ", ".join(str(b) for b in shown[:5])
        if len(shown) > 5:
            preview += f", +{len(shown) - 5} more"
        print(f"{regime.n_bkps:>7}  {span:>26}  {width:>10,.1f}  {preview}")
    if len(regimes) > 12:
        print(f"{'...':>7}  ({len(regimes) - 12} further regimes at lower penalties)")

    # A stable region — a segmentation that stays optimal across a wide span of
    # penalties — is the defensible choice, because it is the one least
    # sensitive to the parameter you could not justify picking.
    #
    # Regimes touching either end of the requested range are excluded: their
    # width is an artefact of where the range was cut, not evidence of
    # stability.
    interior = [
        r for r in regimes if r.pen_min > PEN_MIN * 1.001 and r.pen_max < PEN_MAX * 0.999
    ]
    if interior:
        # width relative to penalty scale, since penalty is a scale parameter
        stable = max(interior, key=lambda r: r.pen_max / r.pen_min)
        print(
            f"\nwidest interior regime: {stable.n_bkps} changes, optimal for penalty in "
            f"[{stable.pen_min:,.1f}, {stable.pen_max:,.1f}] "
            f"({stable.pen_max / stable.pen_min:.1f}x span)"
        )

    match = next((r for r in regimes if r.n_bkps == n_true), None)
    print(f"true changes: {true_bkps[:-1]}")
    if match is not None:
        print(f"found at {n_true} changes: {match.bkps[:-1]}")
        print(
            f"  any penalty in [{match.pen_min:,.1f}, {match.pen_max:,.1f}] "
            f"recovers exactly this"
        )
        # The whole claim, verified: a penalty inside the interval reproduces it.
        probe = 0.5 * (match.pen_min + match.pen_max)
        direct = rpt.Pelt(model="l2", min_size=10, jump=5).fit(signal).predict(probe)
        assert direct == match.bkps, "CROPS interval did not reproduce under Pelt"
        print(f"  verified: Pelt(pen={probe:,.1f}) reproduces it exactly")

    # What a grid of the same range would have found.
    grid = np.geomspace(PEN_MIN, PEN_MAX, 50)
    from_grid = {
        len(rpt.Pelt(model="l2", min_size=10, jump=5).fit(signal).predict(float(p))) - 1
        for p in grid
    }
    print(
        f"\na 50-point log grid over the same range finds {len(from_grid)} of "
        f"{len(regimes)} regimes"
    )
    if n_true not in from_grid and match is not None:
        print(f"  ...and misses the {n_true}-change one entirely")


if __name__ == "__main__":
    main()
