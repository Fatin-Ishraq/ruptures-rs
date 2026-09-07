"""Regenerate the images used by the README.

Committed images go stale silently, so they are generated from a script that
anyone can re-run, and the benchmark chart reads `bench/results.json` rather
than hard-coded numbers — it cannot drift from the table in the README.

Every figure is rendered twice, for GitHub's light and dark themes. The README
picks between them with a `<picture>` element, because a white-background PNG
on a dark page is a glaring rectangle.

    python assets/make_assets.py
"""

import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))
import ruptures_rs as rpt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

#: Two palettes, one per GitHub theme. The accent is a rust orange in both —
#: the obvious nod for a Rust extension, and it stays legible against the
#: blue-grey the signal is drawn in, which a blue accent would not.
THEMES = {
    "light": dict(
        bg="#FBFAF8",
        ink="#1F2933",
        muted="#7B8794",
        signal="#3E4C59",
        accent="#C2410C",
        shade="#CBD2D9",
        grid="#E4E7EB",
        reference="#B0B8C1",
    ),
    "dark": dict(
        bg="#0D1117",
        ink="#E4E7EB",
        muted="#8B949E",
        signal="#AEB8C2",
        accent="#F97316",
        shade="#21262D",
        grid="#21262D",
        reference="#4A5568",
    ),
}


def style(theme):
    c = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": c["bg"],
            "axes.facecolor": c["bg"],
            "savefig.facecolor": c["bg"],
            "text.color": c["ink"],
            "axes.labelcolor": c["ink"],
            "axes.edgecolor": c["grid"],
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return c


def save(fig, name, theme):
    out = HERE / f"{name}-{theme}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")


# ------------------------------------------------------------------ logo


def logo():
    """The mark: three regimes, two change points.

    Drawn once, with no background, in colours that hold on both a white and a
    near-black page — so one file serves both themes and there is nothing to
    keep in sync. Saved as SVG for scaling and PNG for anywhere that will not
    take an SVG.
    """
    accent = "#E8590C"
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # A connected staircase, which is the shape of what the library returns:
    # a piecewise-constant fit. Solid throughout — a dashed riser renders as a
    # row of chunky squares once the mark is scaled down to a favicon.
    # Horizontals carry the weight, risers are lighter, so the eye reads
    # "levels, with breaks" rather than a bar chart.
    levels = ((6, 26, 16), (26, 45, 40), (45, 58, 26))
    for x0, x1, y in levels:
        ax.plot([x0, x1], [y, y], color=accent, lw=8.5, solid_capstyle="round",
                zorder=3)
    for (_, x, y0), (_, _, y1) in zip(levels, levels[1:]):
        ax.plot([x, x], [y0, y1], color=accent, lw=4.2, solid_capstyle="round",
                zorder=2)

    ax.set_xlim(2, 62)
    ax.set_ylim(48, 8)  # inverted: descending steps read as a falling signal
    ax.axis("off")
    for ext in ("svg", "png"):
        out = HERE / f"logo.{ext}"
        fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05,
                    dpi=320)
        print(f"  {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")
    plt.close(fig)


# ------------------------------------------------------------------ hero


def hero(theme):
    """What the library is for: a signal, and where its regimes change.

    The bold line is the fitted piecewise-constant model, so the picture shows
    the answer rather than only the question. Where its steps line up with the
    edges of the shading, the detected change point is the true one.
    """
    c = style(theme)
    signal, true_bkps = rpt.pw_constant(2_000, 1, 6, noise_std=2.5, seed=11)
    t0 = time.perf_counter()
    found = rpt.Dynp(model="l2", min_size=20, jump=5).fit(signal).predict(6)
    elapsed = time.perf_counter() - t0

    fig, ax = plt.subplots(figsize=(11, 3.2))
    # Alternating tones: one flat colour for every regime reads as a single
    # block and hides the very thing the figure is about.
    for i, (start, end) in enumerate(zip([0] + list(true_bkps), true_bkps)):
        if i % 2:
            ax.axvspan(start, end, facecolor=c["shade"], alpha=0.85, lw=0)
    ax.plot(signal[:, 0], color=c["signal"], lw=0.6, alpha=0.55, zorder=2)

    # The fitted model: the mean of each detected segment.
    for start, end in zip([0] + list(found), found):
        level = signal[start:end, 0].mean()
        ax.plot([start, end], [level, level], color=c["accent"], lw=2.6, zorder=3)
    for bkp in found[:-1]:
        ax.axvline(bkp, color=c["accent"], lw=1.0, ls=":", alpha=0.8, zorder=3)

    ax.set_xlim(0, len(signal))
    ax.set_yticks([])
    ax.set_xlabel("sample", color=c["muted"])
    # Not "found exactly": the fit misses one boundary by 20 samples out of
    # 2,000, because the level change there is small next to the noise. That is
    # the detector being right about the data, and the figure should not claim
    # otherwise.
    ax.set_title(
        f"6 change points located in 2,000 samples — {elapsed * 1000:.0f} ms",
        color=c["ink"],
        fontsize=12,
        loc="left",
        pad=22,
    )
    # Above the axes, so it never sits on top of the data.
    ax.text(
        0.0,
        1.02,
        "shading: true regimes        line: fitted piecewise-constant model",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        color=c["muted"],
    )
    save(fig, "hero", theme)


# -------------------------------------------------------------- speedups


def speedups(theme):
    """Read the measured table so the chart cannot disagree with the README."""
    c = style(theme)
    rows = json.load(open(ROOT / "bench" / "results.json"))
    wanted = [
        ("Dynp l2  n=2,000 K=5 jump=1", "Dynp  n=2,000"),
        ("Dynp l2  n=4,000 K=5 jump=1", "Dynp  n=4,000"),
        ("Dynp l2  n=1,000 K=5 jump=1", "Dynp  n=1,000"),
        ("Dynp normal  n=1,500 K=4 jump=5", "Dynp  normal"),
        ("Dynp mahalanobis  n=1,500 K=4 jump=5", "Dynp  mahalanobis"),
        ("Binseg l2  n=5,000 K=5 jump=1", "Binseg  n=5,000"),
        ("Pelt l2  n=5,000 pen=200 jump=1", "Pelt  n=5,000"),
        ("Pelt l2  n=20,000 pen=200 jump=1", "Pelt  n=20,000"),
        ("Dynp rbf  n=1,500 K=4 jump=5", "Dynp  rbf"),
        ("Window l2  n=5,000 width=100 jump=1", "Window  n=5,000"),
        ("Dynp l1  n=1,500 K=4 jump=5", "Dynp  l1"),
        ("BottomUp l2  n=5,000 K=5 jump=1", "BottomUp  n=5,000"),
    ]
    by_case = {r["case"]: r for r in rows}
    pairs = [(lbl, by_case[k]["speedup"]) for k, lbl in wanted if k in by_case]
    pairs.sort(key=lambda p: p[1])
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=c["accent"], height=0.62)
    # `l1` is the one cost that cannot become O(1), so it is greyed out: the
    # chart should say which bars are the structural win and which are only the
    # constant factor from leaving Python.
    for label, bar in zip(labels, bars):
        if "l1" in label or "BottomUp" in label:
            bar.set_color(c["reference"])
    ax.set_xscale("log")
    ax.set_xlim(10, max(values) * 2.2)
    ax.set_xlabel("times faster than ruptures  (log scale)", color=c["muted"])
    ax.grid(axis="x", color=c["grid"], lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            value * 1.12,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}x",
            va="center",
            fontsize=9,
            color=c["ink"],
        )
    ax.set_title(
        "Same input, same breakpoints, measured",
        color=c["ink"],
        fontsize=12,
        loc="left",
        pad=22,
    )
    ax.text(
        0.0,
        1.02,
        "grey: gains only the constant factor from leaving Python",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        color=c["muted"],
    )
    save(fig, "speedup", theme)


# ------------------------------------------------------------------ CROPS


def crops(theme):
    """The feature `ruptures` has no equivalent for: the whole penalty path."""
    c = style(theme)
    signal, _ = rpt.pw_constant(2_000, 1, 6, noise_std=3, seed=11)
    t0 = time.perf_counter()
    regimes = rpt.Crops(model="l2", min_size=10, jump=5).fit_predict(
        signal, 1.0, 20_000.0
    )
    elapsed = time.perf_counter() - t0
    interior = [r for r in regimes if r.pen_min > 1.0 and r.pen_max < 20_000.0]
    widest = max(interior, key=lambda r: r.pen_max / r.pen_min)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for r in regimes:
        is_widest = r is widest
        ax.plot(
            [r.pen_min, r.pen_max],
            [r.n_bkps, r.n_bkps],
            color=c["accent"] if is_widest else c["signal"],
            lw=4.5 if is_widest else 2.0,
            solid_capstyle="butt",
            alpha=1.0 if is_widest else 0.65,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("penalty", color=c["muted"])
    ax.set_ylabel("changes found", color=c["muted"])
    ax.grid(color=c["grid"], lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.annotate(
        f"{widest.n_bkps} changes hold across a\n"
        f"{widest.pen_max / widest.pen_min:.0f}x span of penalties",
        xy=(np.sqrt(widest.pen_min * widest.pen_max), widest.n_bkps),
        xytext=(0.42, 0.68),
        textcoords="axes fraction",
        fontsize=9.5,
        color=c["ink"],
        arrowprops=dict(arrowstyle="->", color=c["accent"], lw=1.4),
    )
    ax.set_title(
        f"{len(regimes)} optimal segmentations from one call — {elapsed * 1000:.0f} ms",
        color=c["ink"],
        fontsize=12,
        loc="left",
        pad=12,
    )
    save(fig, "crops", theme)


if __name__ == "__main__":
    print("logo:")
    logo()
    for theme in THEMES:
        print(f"{theme}:")
        hero(theme)
        speedups(theme)
        crops(theme)
