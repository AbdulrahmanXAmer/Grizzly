"""Behaviour of `normalize` with respect to optional third-party dependencies.

`normalize` probes for pandas, numpy, and pyarrow at call time and degrades
when they are absent. That means these tests behave differently depending on
what is installed, so each one declares which environment it applies to rather
than silently asserting whichever branch happens to be live.
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@pytest.mark.skipif(HAS_PYARROW, reason="passthrough only applies when pyarrow is absent")
def test_normalize_parquet_path_passes_through_without_pyarrow(tmp_path):
    """Without pyarrow, a .parquet path is returned unchanged for a later layer."""
    import grizzly

    p = tmp_path / "data.parquet"
    p.write_bytes(b"not really parquet")

    out = grizzly.normalize(str(p), sample_size=10)
    assert out == str(p)


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow to exercise the read path")
def test_normalize_unreadable_parquet_raises_with_pyarrow(tmp_path):
    """With pyarrow present, a corrupt file surfaces the reader's own error.

    `normalize` only catches ImportError, so a genuine read failure is
    propagated rather than being silently swallowed into a passthrough.
    """
    import pyarrow

    import grizzly

    p = tmp_path / "data.parquet"
    p.write_bytes(b"not really parquet")

    with pytest.raises(pyarrow.ArrowInvalid):
        grizzly.normalize(str(p), sample_size=10)


@pytest.mark.skipif(not HAS_PYARROW, reason="requires pyarrow")
def test_normalize_valid_parquet_returns_records(tmp_path):
    """A well-formed parquet file is sampled into list[dict] records."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import grizzly

    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "good.parquet"
    pq.write_table(table, p)

    out = grizzly.normalize(str(p), sample_size=2)
    assert out == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
def test_normalize_numpy_ndarray():
    """A 2-D ndarray becomes records with synthetic col_N names."""
    import numpy as np

    import grizzly

    arr = np.array([[1, 2], [3, 4], [5, 6]])
    out = grizzly.normalize(arr, sample_size=2)
    assert out == [{"col_0": 1, "col_1": 2}, {"col_0": 3, "col_1": 4}]
