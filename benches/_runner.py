"""Single-measurement child process for the Grizzly benchmark suite.

The parent (``benches/bench.py``) spawns one of these per repetition so that
every measurement starts from a cold interpreter: no warm import caches, no
allocator state carried over from a competing library, no chance that the
order libraries run in changes the result.

The child imports exactly one comparison library, runs one workload once, and
prints a single JSON object on stdout::

    {"seconds": ..., "peak_rss_bytes": ..., "fingerprint": {...}, ...}

The fingerprint is computed *outside* the timed region. It exists so the
parent can assert that every library actually produced the same answer --- a
benchmark where the implementations disagree is measuring different work, and
the numbers are meaningless.

Each implementation uses its own library's idiomatic vectorised API. Writing
the pandas path as a per-column Python loop and then reporting how much faster
Rust is would be benchmarking the loop, not pandas.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any

# Quantiles reported by every profile implementation. Chosen to match the set
# Grizzly's native profiler emits, so the comparison is like-for-like.
QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95)

# Values are rounded to this many decimals before entering a fingerprint.
# Loose enough to absorb legitimate float-summation-order differences between
# implementations, tight enough to catch an implementation doing less work.
FINGERPRINT_DECIMALS = 4


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, normalised to bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS/BSD.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _round(value: Any) -> Any:
    """Round a float for fingerprinting; pass through anything else.

    NaN collapses to ``None`` so that "this statistic does not apply to this
    column" compares equal across libraries that spell it differently.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, FINGERPRINT_DECIMALS)
    return value


def _maybe_float(value: Any) -> float | None:
    """Coerce to float, or None if the value is not numeric."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


# --------------------------------------------------------------------------
# profile workload
# --------------------------------------------------------------------------
# Definition: given a CSV on disk, produce per-column rows observed, null
# count, and -- for numeric columns -- min, max, mean, standard deviation, and
# the quantiles above; for non-numeric columns, the modal value. Reading the
# file is part of the workload for every library.
#
# The fingerprint compares rows observed, null count, min, max, and mean.
# Standard deviation is excluded because Grizzly reports a population std
# while the pandas and polars `describe` paths report a sample std (ddof=1) --
# a definitional difference, not a discrepancy in the data. Quantiles are
# excluded because Grizzly's come from a t-digest and are approximate by
# construction.


def profile_grizzly(path: str, sample_size: int) -> tuple[float, dict[str, Any]]:
    import grizzly

    start = time.perf_counter()
    result = grizzly.csv_profile(path, sample_size=sample_size, lite=False)
    elapsed = time.perf_counter() - start

    # NOTE: Grizzly's "count" is rows observed *including* nulls, unlike the
    # pandas/polars/SQL convention where count is the non-null tally. The
    # fingerprint normalises every library to (observed, null_count) so the
    # comparison is about the data, not the naming.
    columns = {
        col["name"]: {
            "observed": col["count"],
            "null_count": col["null_count"],
            "min": _round(col["min"]),
            "max": _round(col["max"]),
            "mean": _round(col["mean"]),
        }
        for col in result["columns"]
    }
    return elapsed, {"rows": result["rows_sampled"], "columns": columns}


def profile_pandas(path: str, sample_size: int) -> tuple[float, dict[str, Any]]:
    import pandas as pd

    # `describe` is the idiomatic one-shot profiling call: it computes count,
    # mean, std, min, max and every requested percentile across all columns in
    # vectorised C, and with include="all" it also yields the modal value
    # ("top") for non-numeric columns.
    start = time.perf_counter()
    df = pd.read_csv(path)
    described = df.describe(include="all", percentiles=list(QUANTILES))
    null_counts = df.isna().sum()
    elapsed = time.perf_counter() - start

    total_rows = len(df)
    columns: dict[str, Any] = {}
    for name in df.columns:
        stats = described[name]
        columns[str(name)] = {
            "observed": total_rows,
            "null_count": int(null_counts[name]),
            "min": _round(_maybe_float(stats.get("min"))),
            "max": _round(_maybe_float(stats.get("max"))),
            "mean": _round(_maybe_float(stats.get("mean"))),
        }
    return elapsed, {"rows": total_rows, "columns": columns}


def profile_polars(path: str, sample_size: int) -> tuple[float, dict[str, Any]]:
    import polars as pl

    # polars' own `describe` is the equivalent one-shot call. Modal values for
    # non-numeric columns come from a single vectorised select rather than a
    # Python-level loop.
    start = time.perf_counter()
    df = pl.read_csv(path)
    described = df.describe(percentiles=QUANTILES)
    non_numeric = [name for name, dtype in df.schema.items() if not dtype.is_numeric()]
    if non_numeric:
        df.select([pl.col(name).mode().first().alias(name) for name in non_numeric])
    elapsed = time.perf_counter() - start

    # `describe` returns statistics as rows, keyed by a leading "statistic"
    # column; transpose that into per-column lookups.
    by_statistic = {row[0]: row[1:] for row in described.rows()}
    names = described.columns[1:]
    numeric = {name for name, dtype in df.schema.items() if dtype.is_numeric()}

    def stat(label: str, index: int) -> Any:
        values = by_statistic.get(label)
        return None if values is None else values[index]

    columns: dict[str, Any] = {}
    for index, name in enumerate(names):
        is_numeric = name in numeric
        columns[name] = {
            "observed": df.height,
            "null_count": int(_maybe_float(stat("null_count", index)) or 0),
            "min": _round(_maybe_float(stat("min", index))) if is_numeric else None,
            "max": _round(_maybe_float(stat("max", index))) if is_numeric else None,
            "mean": _round(_maybe_float(stat("mean", index))) if is_numeric else None,
        }
    return elapsed, {"rows": df.height, "columns": columns}


# --------------------------------------------------------------------------
# transform workload
# --------------------------------------------------------------------------
# Definition: read a CSV, min-max scale every numeric column to [0, 1] using
# that column's own min and max, and write the result back out as CSV.
# Reading and writing are both part of the workload.


def transform_grizzly(path: str, out_path: str, sample_size: int) -> tuple[float, dict[str, Any]]:
    import grizzly

    start = time.perf_counter()
    params = grizzly.csv_minmax_params(path, sample_size=sample_size)["params"]
    result = grizzly.csv_transform_minmax(path, out_path, params)
    elapsed = time.perf_counter() - start
    return elapsed, {
        "rows_written": result["rows_written"],
        "cols_scaled": result["numeric_cols_scaled"],
    }


def transform_pandas(path: str, out_path: str, sample_size: int) -> tuple[float, dict[str, Any]]:
    import pandas as pd

    start = time.perf_counter()
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number").columns
    # Scale every numeric column in one vectorised block rather than assigning
    # column by column, which would repeatedly refragment the frame.
    block = df[numeric]
    lo = block.min()
    span = (block.max() - lo).replace(0, 1.0)
    df[numeric] = (block - lo).div(span)
    df.to_csv(out_path, index=False)
    elapsed = time.perf_counter() - start
    return elapsed, {"rows_written": len(df), "cols_scaled": len(numeric)}


def transform_polars(path: str, out_path: str, sample_size: int) -> tuple[float, dict[str, Any]]:
    import polars as pl

    start = time.perf_counter()
    df = pl.read_csv(path)
    numeric = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
    df = df.with_columns(
        [
            pl.when(pl.col(name).max() == pl.col(name).min())
            .then(0.0)
            .otherwise(
                (pl.col(name) - pl.col(name).min()) / (pl.col(name).max() - pl.col(name).min())
            )
            .alias(name)
            for name in numeric
        ]
    )
    df.write_csv(out_path)
    elapsed = time.perf_counter() - start
    return elapsed, {"rows_written": df.height, "cols_scaled": len(numeric)}


def verify_transform_output(out_path: str) -> dict[str, Any]:
    """Re-read a transform result and summarise it, outside the timed region.

    Confirms the scaling actually happened: every scaled column should span
    [0, 1]. Returned values feed the cross-library equivalence check.
    """
    import csv

    with open(out_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        mins = [math.inf] * len(header)
        maxs = [-math.inf] * len(header)
        rows = 0
        for row in reader:
            rows += 1
            for i, cell in enumerate(row):
                if not cell:
                    continue
                try:
                    value = float(cell)
                except ValueError:
                    continue
                if value < mins[i]:
                    mins[i] = value
                if value > maxs[i]:
                    maxs[i] = value

    scaled = {
        name: [round(lo, 3), round(hi, 3)]
        for name, lo, hi in zip(header, mins, maxs, strict=True)
        if lo != math.inf
    }
    return {"rows": rows, "ranges": scaled}


PROFILE_IMPLS = {
    "grizzly": profile_grizzly,
    "pandas": profile_pandas,
    "polars": profile_polars,
}

TRANSFORM_IMPLS = {
    "grizzly": transform_grizzly,
    "pandas": transform_pandas,
    "polars": transform_polars,
}


def library_version(library: str) -> str:
    import importlib

    module = importlib.import_module(library)
    return getattr(module, "__version__", "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, choices=["grizzly", "pandas", "polars"])
    parser.add_argument("--workload", required=True, choices=["profile", "transform"])
    parser.add_argument("--path", required=True, help="input CSV")
    parser.add_argument("--out", help="output CSV (transform only)")
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help=(
            "Rows Grizzly is permitted to read. Must exceed the file's row "
            "count for a fair comparison; the parent asserts full coverage."
        ),
    )
    args = parser.parse_args()

    if args.workload == "profile":
        elapsed, fingerprint = PROFILE_IMPLS[args.library](args.path, args.sample_size)
    else:
        if not args.out:
            parser.error("--out is required for the transform workload")
        elapsed, fingerprint = TRANSFORM_IMPLS[args.library](args.path, args.out, args.sample_size)
        fingerprint |= verify_transform_output(args.out)
        Path(args.out).unlink(missing_ok=True)

    json.dump(
        {
            "library": args.library,
            "library_version": library_version(args.library),
            "workload": args.workload,
            "seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes(),
            "fingerprint": fingerprint,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
