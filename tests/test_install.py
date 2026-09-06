"""Tests for `ruptures_rs.install()`, the zero-edit adoption path.

`import ruptures_rs as rpt` covers code you can edit. `install()` covers the
rest: a dependency deep in the stack that imports `ruptures` by name, which you
would otherwise have to fork to accelerate.

Each case runs in a fresh interpreter, because the whole point is what
`sys.modules` looks like before anything imports `ruptures`.
"""

import subprocess
import sys
import textwrap

import pytest


def run(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_install_redirects_plain_imports():
    """A module that says `import ruptures` gets this package instead."""
    out = run(
        """
        import ruptures_rs
        ruptures_rs.install()

        import ruptures as rpt
        signal, _ = rpt.pw_constant(300, 1, 4, noise_std=2, seed=1)
        bkps = rpt.Dynp(model="l2", min_size=5, jump=5).fit(signal).predict(4)
        print(rpt.__name__)
        print(bkps)
        """
    ).splitlines()
    assert out[0] == "ruptures_rs"
    assert out[1] == "[60, 120, 185, 240, 300]"


def test_install_redirects_submodules():
    """`from ruptures.costs import ...` has to work too, not just the top level."""
    out = run(
        """
        import ruptures_rs
        ruptures_rs.install()

        from ruptures.costs import CostL2
        from ruptures.detection import Pelt
        from ruptures.metrics import hausdorff
        from ruptures.datasets import pw_constant
        from ruptures.exceptions import BadSegmentationParameters

        signal, _ = pw_constant(300, 1, 4, noise_std=2, seed=1)
        print(Pelt(model="l2", min_size=5, jump=5).fit(signal).predict(200))
        print(round(CostL2().fit(signal).error(0, 100), 6))
        print(hausdorff([50, 100, 300], [52, 98, 300]))
        """
    ).splitlines()
    assert out[0] == "[120, 185, 240, 300]"
    assert out[1] == "425.508667"
    assert out[2] == "2"


def test_install_produces_accelerated_estimators():
    out = run(
        """
        import ruptures_rs
        ruptures_rs.install()
        import ruptures as rpt
        print(rpt.Dynp(model="l2").accelerated)
        """
    )
    assert out == "True"


def test_install_refuses_after_real_ruptures_imported():
    """Half-patching the module graph would be worse than not patching it."""
    pytest.importorskip("ruptures", reason="reference not installable here")
    out = run(
        """
        import ruptures  # the real one, first
        import ruptures_rs
        try:
            ruptures_rs.install()
            print("NOT REFUSED")
        except RuntimeError:
            print("refused")
        """
    )
    assert out == "refused"


def test_install_is_idempotent():
    out = run(
        """
        import ruptures_rs
        ruptures_rs.install()
        ruptures_rs.install()
        import ruptures as rpt
        print(rpt.__name__)
        """
    )
    assert out == "ruptures_rs"


@pytest.mark.parametrize(
    "name",
    ["Dynp", "Pelt", "Binseg", "BottomUp", "Window", "KernelCPD", "Crops"],
)
def test_public_surface_is_present(name):
    import ruptures_rs

    assert hasattr(ruptures_rs, name)


def test_covers_every_public_name_ruptures_exposes():
    """Coverage check: a partial drop-in is worthless, so measure it."""
    rpt_py = pytest.importorskip("ruptures", reason="reference not installable here")

    import ruptures_rs as rpt_rs

    missing = {
        name
        for name in dir(rpt_py)
        if not name.startswith("_") and not hasattr(rpt_rs, name)
    }
    assert not missing, f"missing from ruptures_rs: {sorted(missing)}"
