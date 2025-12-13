from __future__ import annotations

from typing import Any, Dict, List


def _load_native():
    try:
        from . import _grizzly as native  # type: ignore

        return native
    except Exception:  # pragma: no cover
        return None


def is_native() -> bool:
    """True if the Rust extension module (grizzly._grizzly) is importable."""
    return _load_native() is not None


def native_module():
    """Return the native module object if available, else None."""
    return _load_native()


def detect_schema(
    data: Any,
    *,
    sample_size: int = 1000,
    max_examples: int = 5,
    normalize_input: bool = True,
) -> Dict[str, Any]:
    """
    Infer a flattened schema from arbitrary Python data.

    Path notation:
    - Dict keys are joined with '.':   user.name
    - List/tuple nesting uses '[]':    items[].id, matrix[][].value
    """
    if normalize_input:
        from .normalize import normalize

        data = normalize(data, sample_size=sample_size)

    native = _load_native()
    if native is not None:
        return native.detect_schema(data, sample_size=sample_size, max_examples=max_examples)

    # Fallback: minimal, slower Python implementation (kept intentionally small).
    from .fallback import detect_schema as py_detect_schema

    return py_detect_schema(data, sample_size=sample_size, max_examples=max_examples)


def detect_columns(data: Any, *, sample_size: int = 1000) -> List[str]:
    """Convenience wrapper returning just the sorted column paths."""
    schema = detect_schema(data, sample_size=sample_size)
    return [c["path"] for c in schema.get("columns", [])]


def info(
    data: Any,
    *,
    sample_size: int = 1000,
    max_examples: int = 5,
    normalize_input: bool = True,
    show_examples: bool = False,
    max_cols: int | None = None,
    file=None,
) -> None:
    """Module-level convenience wrapper: grizzly.info(data) -> prints summary."""
    from .grizzly import Grizzly

    Grizzly(
        data,
        sample_size=sample_size,
        max_examples=max_examples,
        normalize_input=normalize_input,
    ).info(file=file, show_examples=show_examples, max_cols=max_cols)


def csv_profile(
    path: str,
    *,
    sample_size: int = 1000,
    max_examples: int = 5,
    fast_csv: bool = True,
    lite: bool = False,
    track_freq: bool = True,
    collect_examples: bool = True,
) -> Dict[str, Any]:
    """
    Rust-accelerated CSV profiling: delimiter/header sniff + dtype + basic stats per column.
    
    Args:
        path: Path to CSV file (supports .csv.gz)
        sample_size: Maximum number of rows to sample
        max_examples: Maximum examples to collect per column
        fast_csv: If True, uses parallel byte chunking (assumes no quoted newlines).
                  If False, uses sequential reading (correct for any CSV).
                  Default: True for speed.
        lite: If True, only compute numeric stats (min/max/mean/std/quantiles).
              Skips type inference, examples, and frequency tracking.
              Use this for Polars-equivalent benchmarking speed.
        track_freq: If True, track value frequency for mode calculation.
        collect_examples: If True, collect example values per column.
    
    Returns: Profile dict with columns, stats, delimiter info, etc.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError("csv_profile requires the native Rust extension; build with `maturin develop`.")
    return native.csv_profile(
        path, 
        sample_size=sample_size, 
        max_examples=max_examples, 
        fast_csv=fast_csv,
        lite=lite,
        track_freq=track_freq,
        collect_examples=collect_examples,
    )


def csv_minmax_params(path: str, *, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Return min/max per numeric column (sampled), suitable for min-max scaling.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError("csv_minmax_params requires the native Rust extension; build with `maturin develop`.")
    return native.csv_minmax_params(path, sample_size=sample_size)


def csv_transform_minmax(
    input_path: str,
    output_path: str,
    params: Dict[str, Dict[str, float]],
    *,
    delimiter: str | None = None,
    has_header: bool | None = None,
) -> Dict[str, Any]:
    """
    Transform a CSV by applying min-max scaling to numeric columns.
    
    Args:
        input_path: Path to input CSV (can be .csv.gz)
        output_path: Path to output CSV
        params: Dict of {col_name: {"min": ..., "max": ...}, ...}
        delimiter: Optional delimiter (None = auto-detect)
        has_header: Whether file has header row (None = auto-detect)
    
    Returns: {
        "input_path": ...,
        "output_path": ...,
        "rows_written": ...,
        "numeric_cols_scaled": ...,
        "has_header": ...
    }
    """
    native = _load_native()
    if native is None:
        raise RuntimeError("csv_transform_minmax requires the native Rust extension; build with `maturin develop`.")
    return native.csv_transform_minmax(input_path, output_path, params, delimiter, has_header)


def csv_linear_regression(
    path: str,
    *,
    target: str,
    features: List[str] | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
    sample_size: int = 1_000_000,
    delimiter: str | None = None,
    has_header: bool | None = None,
    fast_csv: bool = True,
    shuffle: bool = True,
    ridge_lambda: float = 0.0,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """
    Rust-native linear regression on CSV/CSV.GZ (no numpy required).

    Returns:
      { "r2": ..., "coef": [...], "intercept": ..., "train_n": ..., "test_n": ... }
    """
    native = _load_native()
    if native is None:
        raise RuntimeError("csv_linear_regression requires the native Rust extension; build with `maturin develop`.")
    return native.csv_linear_regression(
        path,
        target=target,
        features=features,
        train_frac=train_frac,
        seed=seed,
        sample_size=sample_size,
        delimiter=delimiter,
        has_header=has_header,
        fast_csv=fast_csv,
        shuffle=shuffle,
        ridge_lambda=ridge_lambda,
        return_debug=return_debug,
    )


