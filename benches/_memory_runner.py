"""One transform under a container memory limit, for the memory-ceiling study.

Runs inside the container. Prints a JSON line on success; on failure it is
expected to die by SIGKILL and print nothing, which is the signal the parent
is looking for.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path


def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (raw if sys.platform == "darwin" else raw * 1024) / 1e6


def transform_grizzly(data: str, out: str) -> None:
    import grizzly

    params = grizzly.csv_minmax_params(data, sample_size=10_000_000)["params"]
    grizzly.csv_transform_minmax(data, out, params)


def transform_polars(data: str, out: str) -> None:
    import polars as pl

    df = pl.read_csv(data)
    numeric = [c for c, dtype in df.schema.items() if dtype.is_numeric()]
    df = df.with_columns(
        [
            pl.when(pl.col(c).max() == pl.col(c).min())
            .then(0.0)
            .otherwise((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min()))
            .alias(c)
            for c in numeric
        ]
    )
    df.write_csv(out)


IMPLS = {"grizzly": transform_grizzly, "polars": transform_polars}


def main() -> int:
    library, data, out = sys.argv[1], sys.argv[2], sys.argv[3]

    start = time.perf_counter()
    IMPLS[library](data, out)
    elapsed = time.perf_counter() - start

    json.dump(
        {
            "seconds": elapsed,
            "peak_rss_mb": peak_rss_mb(),
            "input_bytes": Path(data).stat().st_size,
            "output_bytes": Path(out).stat().st_size,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
