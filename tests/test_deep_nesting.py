"""Regression tests for the nesting-depth guard in schema inference.

`flatten_value` in the Rust core recurses once per level of nesting. Before the
guard existed, a sufficiently deep input overflowed the Rust stack and killed
the interpreter with SIGSEGV -- not a Python exception, so no caller could
catch it, and nothing above the extension could recover. The observed crash
threshold was around 20,000-30,000 levels, reachable from ordinary API use on
deeply nested JSON.

These tests run the deep cases in a subprocess, because the failure mode being
guarded against terminates the process outright: an in-process assertion would
take the whole test run down with it rather than reporting a failure.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import grizzly

# Comfortably past the 512-level cap and past the historical crash threshold.
CRASHING_DEPTH = 60_000


def nest(depth: int) -> dict:
    """Build a `depth`-level nested dict iteratively.

    Built with a loop rather than recursion so that constructing the input does
    not hit Python's own recursion limit before Grizzly ever sees it.
    """
    obj: dict = {"leaf": 1}
    for _ in range(depth):
        obj = {"n": obj}
    return obj


def run_in_subprocess(depth: int) -> subprocess.CompletedProcess[str]:
    """Profile a nested structure of `depth` levels in a fresh interpreter."""
    script = textwrap.dedent(
        f"""
        import grizzly

        obj = {{"leaf": 1}}
        for _ in range({depth}):
            obj = {{"n": obj}}

        schema = grizzly.detect_schema(obj, sample_size=10_000_000)
        print("OK", schema["max_depth_exceeded"], len(schema["columns"]))
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_deeply_nested_input_does_not_crash_the_interpreter():
    """The historical repro: this used to exit with SIGSEGV (-11 / 139)."""
    proc = run_in_subprocess(CRASHING_DEPTH)

    assert proc.returncode == 0, (
        f"interpreter died on {CRASHING_DEPTH}-level nesting "
        f"(returncode={proc.returncode}); stderr:\n{proc.stderr}"
    )
    assert proc.stdout.startswith("OK ")


def test_deeply_nested_input_reports_truncation():
    """Truncation is reported rather than silently returning a partial schema."""
    proc = run_in_subprocess(CRASHING_DEPTH)
    assert proc.returncode == 0, proc.stderr

    _, exceeded, n_columns = proc.stdout.split()
    assert exceeded == "True"
    # The truncation point is still recorded, so the caller can see where the
    # schema stopped rather than receiving nothing at all.
    assert int(n_columns) == 1


def test_shallow_nesting_is_unaffected():
    """Input below the cap behaves exactly as before, examples included."""
    schema = grizzly.detect_schema(nest(100), sample_size=10_000_000)

    assert schema["max_depth_exceeded"] is False
    assert len(schema["columns"]) == 1

    column = schema["columns"][0]
    assert column["path"].endswith("leaf")
    assert column["path"].count("n.") == 100
    assert column["inferred"] == "int"
    assert column["examples"], "examples should still be collected below the cap"


def test_schema_reports_the_depth_cap():
    """The cap is discoverable, not a magic number the caller has to guess."""
    schema = grizzly.detect_schema({"a": 1})

    assert schema["max_depth"] == 512
    assert schema["max_depth_exceeded"] is False


def test_truncated_column_omits_examples():
    """No repr() is taken at the truncation point.

    `repr()` of a deeply nested object recurses inside CPython, so collecting an
    example exactly where the guard fires would reintroduce the stack overflow
    the guard exists to prevent.
    """
    schema = grizzly.detect_schema(nest(2000), sample_size=10_000_000)

    assert schema["max_depth_exceeded"] is True
    column = schema["columns"][0]
    assert column["examples"] == []
    assert column["inferred"] == "dict"


@pytest.mark.parametrize("depth", [0, 1, 10, 510])
def test_depths_below_the_cap_never_truncate(depth):
    schema = grizzly.detect_schema(nest(depth), sample_size=10_000_000)
    assert schema["max_depth_exceeded"] is False


def test_truncation_boundary_is_exact():
    """Pin the exact depth at which truncation starts.

    `nest(n)` produces n wrapper dicts around a `{"leaf": 1}`, so the scalar
    itself sits at depth n + 1. With a cap of 512 that makes nest(510) the
    deepest input processed in full, and nest(511) the first to trip the guard
    -- on the scalar, which is why the leaf path is still reported there.
    """
    assert grizzly.detect_schema(nest(510), sample_size=10_000_000)["max_depth_exceeded"] is False

    at_boundary = grizzly.detect_schema(nest(511), sample_size=10_000_000)
    assert at_boundary["max_depth_exceeded"] is True
    assert at_boundary["columns"][0]["path"].endswith("leaf")
    assert at_boundary["columns"][0]["inferred"] == "int"

    # One level deeper and the guard fires on the containing dict instead, so
    # the leaf is never reached.
    past_boundary = grizzly.detect_schema(nest(512), sample_size=10_000_000)
    assert past_boundary["max_depth_exceeded"] is True
    assert not past_boundary["columns"][0]["path"].endswith("leaf")
    assert past_boundary["columns"][0]["inferred"] == "dict"
