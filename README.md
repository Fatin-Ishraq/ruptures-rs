<div align="center">

<img src="https://raw.githubusercontent.com/Fatin-Ishraq/ruptures-rs/main/assets/logo.png" alt="" width="84">

# ruptures-rs

**Find the moments a time series changed behaviour.**

[![PyPI](https://img.shields.io/pypi/v/ruptures-rs?color=C2410C&label=pypi)](https://pypi.org/project/ruptures-rs/)
[![Python](https://img.shields.io/badge/python-3.10%20–%203.14-blue.svg)](https://pypi.org/project/ruptures-rs/)
[![CI](https://github.com/Fatin-Ishraq/ruptures-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatin-Ishraq/ruptures-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

</div>

<!-- Two variants so the figure is not a glaring white rectangle on a dark
     page. GitHub honours the <picture>; PyPI strips it and keeps the inner
     <img>, which is the light one. -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/ruptures-rs/main/assets/hero-dark.png">
  <img alt="A 2,000-point noisy signal with six change points. Shading marks the true regimes; the bold line is the piecewise-constant model ruptures-rs fitted, whose steps land on the regime boundaries to within twenty samples out of two thousand." src="https://raw.githubusercontent.com/Fatin-Ishraq/ruptures-rs/main/assets/hero-light.png">
</picture>

## What this is for

You have a sequence of measurements, and partway through, something changed —
and stayed changed. **Change point detection** finds those moments: the
boundaries between stretches where the data behaves one way and stretches where
it behaves another.

That is a different question from anomaly detection. An anomaly is one odd
reading. A change point is where a new normal begins. If your latency rose
after a deploy and stayed up, no single request looks wrong, but there is a
moment worth finding.

It comes up wherever a process has regimes:

- **Operations** — when did the error rate shift, and does it line up with the
  release?
- **Industrial sensors** — split a run into phases, or catch where a machine's
  vibration signature changed.
- **Finance** — divide a series into volatility regimes instead of assuming one.
- **Genomics** — copy number segmentation, which is where much of this
  literature comes from.
- **Wearables** — cut an accelerometer trace into activities.

You supply the signal and either the number of changes you expect or a penalty
for adding one. You get back the breakpoints.

## Quick start

```bash
pip install ruptures-rs
```

```python
import ruptures_rs as rpt

# A synthetic signal with six changes, so the example runs anywhere.
signal, true_bkps = rpt.pw_constant(2_000, 1, 6, noise_std=3, seed=11)

# "I expect six changes. Where are they?"
bkps = rpt.Dynp(model="l2", min_size=20, jump=5).fit(signal).predict(6)
#  -> [275, 550, 840, 1120, 1415, 1710, 2000]      in 1.6 ms

# "I don't know how many. Charge me 500 for each one you add."
bkps = rpt.Pelt(model="l2", min_size=20).fit(signal).predict(pen=500)

rpt.display(signal, true_bkps, bkps)   # needs matplotlib
```

A breakpoint is the *end* of a segment, and the last one is always the length
of the signal — the same convention `ruptures` uses.

Choose a **detector** by what you know going in. `Dynp` when you know the number
of changes and want the exact optimum. `Pelt` when you do not, and would rather
set a penalty. `Binseg` and `BottomUp` for a quick approximate answer. `Window`
for a fast scan of a long signal.

Choose a **cost** by the kind of change you are looking for. `l2` for shifts in
the mean, `normal` for shifts in variance or correlation, `rbf` for changes in
distribution that are not just the mean, `linear` and `ar` for a change in a
relationship or a temporal model.

## A drop-in replacement for `ruptures`

This reimplements [`ruptures`](https://github.com/deepcharles/ruptures), the
standard Python library for this problem, with the same API.

```diff
- import ruptures as rpt
+ import ruptures_rs as rpt
```

That is the whole migration. Same classes, same arguments, same breakpoints —
verified by 1,392 tests that run both libraries on the same input and demand
*identical* output, not merely similar output.

It also installs where the reference currently cannot. One `abi3` wheel per
platform covers Python 3.10 through 3.14, including musl, so `pip install`
never has to go looking for a C compiler.

## Fast enough to change what you attempt

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/ruptures-rs/main/assets/speedup-dark.png">
  <img alt="Horizontal bar chart of measured speedups against ruptures, log scale, ranging from 23x for BottomUp to 4,863x for Dynp on 2,000 samples." src="https://raw.githubusercontent.com/Fatin-Ishraq/ruptures-rs/main/assets/speedup-light.png">
</picture>

`ruptures` evaluates its cost function once per cell of a dynamic-programming
table, and every call is an O(segment length) NumPy reduction. Nearly all of
those costs are a difference of prefix sums in disguise, which makes them O(1) —
so the gain is structural, not just the constant factor from leaving Python.

The practical effect is that exact segmentation stops being something you budget
for. A 50,000-point signal takes 0.06 s here, and is not a workload the
reference can run at all.

[Full benchmark tables →](docs/benchmarks.md)

## Choosing a penalty without guessing

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/ruptures-rs/main/assets/crops-dark.png">
  <img alt="The CROPS penalty path: number of change points against penalty, on log axes, as a staircase of 86 optimal segmentations. The widest step is highlighted, showing four changes holding across a 25x span of penalties." src="https://raw.githubusercontent.com/Fatin-Ishraq/ruptures-rs/main/assets/crops-light.png">
</picture>

Penalised detection asks you for a penalty, and nobody knows theirs in advance.
`Crops` returns *every* segmentation that is optimal somewhere in a penalty
range, with the exact interval each one owns:

```python
for regime in rpt.Crops(model="l2", min_size=10).fit_predict(signal, 1, 20_000):
    print(f"{regime.n_bkps:2d} changes for penalty in "
          f"[{regime.pen_min:,.0f}, {regime.pen_max:,.0f}]")
```

The answer that survives the widest span of penalties is the defensible one,
because it is the least sensitive to the parameter you could not justify
picking. `ruptures` has no equivalent, and it is only practical here because
PELT became cheap enough to run dozens of times.
[`examples/penalty_path.py`](examples/penalty_path.py) works through it.

## What's included

Everything in `ruptures` 1.1.9, plus `Crops`:

| | |
|---|---|
| **Detectors** | `Dynp`, `Pelt`, `Binseg`, `BottomUp`, `Window`, `KernelCPD`, `Crops` |
| **Costs** | `l1`, `l2`, `normal`, `rbf`, `cosine`, `rank`, `mahalanobis`, `clinear`, `linear`, `ar` |
| **Datasets** | `pw_constant`, `pw_linear`, `pw_normal`, `pw_wavy` |
| **Metrics** | `precision_recall`, `hausdorff`, `randindex`, `meantime`, `hamming` |
| **Also** | `display`, `cost_factory`, `BaseCost`, `BaseEstimator`, `Bnode`, exceptions |

Your own `custom_cost` still works. It runs on a pure-Python implementation,
because a Python callable inside the inner loop cannot be made faster by
crossing an FFI boundary to reach it — same answers, original speed.
`estimator.accelerated` reports which path you are on.

## Going deeper

- [Benchmarks](docs/benchmarks.md) — every measured row, and how it was measured.
- [Accuracy and correctness](docs/accuracy.md) — why prefix sums are a numerical
  trap, what is done about it, and how the test suite is built.
- [Where the answers differ](docs/differences.md) — the four places this package
  and `ruptures` can disagree, stated precisely, each pinned by a test.
- [Changelog](CHANGELOG.md).

## For code you cannot edit

When a dependency deep in the stack imports `ruptures` by name:

```python
import ruptures_rs
ruptures_rs.install()   # before anything imports `ruptures`

import ruptures as rpt  # this is now ruptures_rs
```

It refuses rather than half-patching the module graph if the real `ruptures`
has already been imported.

## Development

```bash
pip install maturin pytest numpy scipy ruptures
maturin develop --release
pytest tests/ -q
cargo test
```

## Licence and credit

BSD-2-Clause, the same licence as `ruptures`.

This is a reimplementation of [`ruptures`](https://github.com/deepcharles/ruptures)
by Charles Truong, Laurent Oudre and Nicolas Vayatis, whose API, algorithms and
semantics it deliberately reproduces. The pure-Python fallback in
`_fallback.py` is a direct port of their code. If you use this in research,
cite their paper:

> C. Truong, L. Oudre, N. Vayatis. *Selective review of offline change point
> detection methods.* Signal Processing, 167:107299, 2020.
