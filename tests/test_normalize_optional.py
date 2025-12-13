from __future__ import annotations


def test_normalize_parquet_path_skips_without_pyarrow(tmp_path):
    import grizzly

    p = tmp_path / "data.parquet"
    p.write_bytes(b"not really parquet")

    # Without pyarrow installed, normalize just passes through
    out = grizzly.normalize(str(p), sample_size=10)
    assert out == str(p)


def test_normalize_numpy_ndarray_if_available():
    import grizzly

    try:
        import numpy as np  # type: ignore
    except ImportError:
        return

    arr = np.array([[1, 2], [3, 4], [5, 6]])
    out = grizzly.normalize(arr, sample_size=2)
    assert out == [{"col_0": 1, "col_1": 2}, {"col_0": 3, "col_1": 4}]


