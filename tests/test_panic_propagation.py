"""A Rust panic must reach Python as an exception, not kill the process.

PyO3 wraps every ``#[pyfunction]`` body in ``catch_unwind`` and converts a
caught panic into ``pyo3_runtime.PanicException``. That machinery only works if
panics unwind. Under ``panic = "abort"`` -- which this project's release profile
previously set -- the conversion never happens: the panic aborts the process
with SIGABRT, no Python handler runs, and no caller can recover. Observed exit
code 134 before the change, 0 after.

These tests exercise ``_force_panic``, which only exists when the crate is built
with the ``testing`` feature::

    maturin develop --release --features testing

Without that feature the whole module skips, so a normal test run is unaffected
and released wheels never carry the hook.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import grizzly

_native = grizzly.native_module()

pytestmark = pytest.mark.skipif(
    not hasattr(_native, "_force_panic"),
    reason="built without the `testing` feature; _force_panic is unavailable",
)


def test_panic_raises_instead_of_aborting():
    """The panic is catchable from Python at all."""
    with pytest.raises(BaseException) as excinfo:
        _native._force_panic()

    assert type(excinfo.value).__name__ == "PanicException"
    assert "verifying panic propagation" in str(excinfo.value)


def test_interpreter_survives_a_panic():
    """The process stays alive and usable after a panic has been caught.

    Run in a subprocess: if the panic aborts, it takes the whole test session
    down rather than failing this one test.
    """
    script = textwrap.dedent(
        """
        import grizzly

        native = grizzly.native_module()
        try:
            native._force_panic()
        except BaseException:
            pass

        # The extension must still work after a caught panic.
        schema = grizzly.detect_schema([{"a": 1}])
        assert schema["columns"][0]["path"] == "a"
        print("ALIVE")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, (
        f"interpreter did not survive the panic (returncode={proc.returncode}, "
        f"134 means SIGABRT); stderr:\n{proc.stderr}"
    )
    assert "ALIVE" in proc.stdout
