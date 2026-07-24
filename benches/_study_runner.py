"""One profiling measurement for the sampling study, in an isolated process.

Kept separate from ``benches/_runner.py`` because that one compares libraries
on a fixed workload, while this one sweeps Grizzly's own ``sample_size`` and
reports the quantiles it produced.
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    path, sample_size = sys.argv[1], int(sys.argv[2])

    import grizzly

    start = time.perf_counter()
    profile = grizzly.csv_profile(path, sample_size=sample_size, lite=False)
    elapsed = time.perf_counter() - start

    column = profile["columns"][0]
    json.dump(
        {
            "seconds": elapsed,
            "rows_sampled": profile["rows_sampled"],
            "quantiles": {key: column.get(key) for key in ("p25", "median", "p75", "p90", "p95")},
            "min": column.get("min"),
            "max": column.get("max"),
            "mean": column.get("mean"),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
