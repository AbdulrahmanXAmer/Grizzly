"""Accuracy-vs-speed study: what does sampling actually cost you?

Grizzly is sampling-first. `sample_size` bounds how many rows it reads, which
is the lever that makes profiling a large file cheap. The obvious question --
and the one a benchmark comparing wall-clock times never answers -- is what
that lever costs in accuracy.

This study sweeps `sample_size` across a fixed dataset and measures, at each
setting, both the time taken and the error in the resulting quantiles. The
output is the tradeoff curve: how much accuracy you buy with each additional
row read.

Two error measures, because they answer different questions:

**Rank error** is what a t-digest bounds. For an estimate `v` returned for
quantile `q`, it is how far the true fraction of data at or below `v` sits from
`q`. This is the honest measure of a quantile estimator.

**Value error** is how far the returned number is from the exact quantile,
relative to the data range. This is what a user actually feels when they use a
percentile as a threshold.

The study runs the same values twice, shuffled and sorted, to separate two
effects that a single run would confuse.

The result is not what the obvious model predicts. Sampling does *not* read a
prefix: the profiler splits the file into per-thread chunks and each chunk
consumes rows from its own region, so a small sample is spread across the whole
file rather than taken from the front. Measured directly -- profiling 200,000
strictly ascending rows with ``sample_size=256`` reports a maximum of 198,566,
not 255. That matters because it means sampling stays usable on data with
meaningful row order (sorted, time-series, partitioned by date), where
``head -n`` or ``pandas.read_csv(nrows=...)`` would give a badly biased answer.

What the sorted variant does show is that a t-digest is sensitive to insertion
order: at full coverage the sorted input lands at roughly twice the rank error
of the shuffled one. Real but small, and worth knowing when profiling data that
arrives pre-sorted on the column being profiled.

Usage::

    python -m benches.study_sampling --rows 2000000
    python -m benches.study_sampling --out benches/results/sampling_study.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from benches.bench import capture_environment

REPO_ROOT = Path(__file__).resolve().parent.parent

# Quantiles to report. p99 is not in Grizzly's profile output, so the study
# covers the ones it does emit; p95 is the tail estimate in practice.
QUANTILES = (("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95))

# Sample sizes swept, as a fraction of the dataset. 1.0 and above read the
# whole file and are the accuracy ceiling.
SAMPLE_FRACTIONS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 4.0)

REPETITIONS = 3


def exact_quantile(ordered: list[float], q: float) -> float:
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def rank_error(ordered: list[float], value: float, q: float) -> float:
    """Tie-aware rank error: distance from q to the band `value` answers."""
    n = len(ordered)
    low = bisect.bisect_left(ordered, value) / n
    high = bisect.bisect_right(ordered, value) / n
    if low <= q <= high:
        return 0.0
    return low - q if q < low else q - high


def generate(path: Path, values: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["v"])
        for value in values:
            writer.writerow([f"{value:.9f}"])
    return path


def measure(path: Path, sample_size: int, repetitions: int) -> dict[str, Any]:
    """Profile in a fresh interpreter, `repetitions` times, and take the median."""
    script = REPO_ROOT / "benches" / "_study_runner.py"
    seconds: list[float] = []
    payload: dict[str, Any] = {}

    for _ in range(repetitions):
        proc = subprocess.run(
            [sys.executable, str(script), str(path), str(sample_size)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"study runner failed:\n{proc.stderr}")
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        seconds.append(payload["seconds"])

    payload["seconds"] = statistics.median(seconds)
    return payload


def study_one(
    label: str,
    values: list[float],
    data_dir: Path,
    repetitions: int,
) -> list[dict[str, Any]]:
    n_rows = len(values)
    path = generate(data_dir / f"study_{label}.csv", values)
    ordered = sorted(values)
    data_range = ordered[-1] - ordered[0]

    rows: list[dict[str, Any]] = []
    for fraction in SAMPLE_FRACTIONS:
        sample_size = max(1, int(n_rows * fraction))
        result = measure(path, sample_size, repetitions)

        per_quantile = {}
        worst_rank = 0.0
        worst_value = 0.0
        for key, q in QUANTILES:
            estimate = result["quantiles"].get(key)
            if estimate is None:
                continue
            r_err = rank_error(ordered, estimate, q)
            v_err = abs(estimate - exact_quantile(ordered, q)) / data_range
            worst_rank = max(worst_rank, r_err)
            worst_value = max(worst_value, v_err)
            per_quantile[key] = {
                "estimate": estimate,
                "exact": exact_quantile(ordered, q),
                "rank_error": r_err,
                "value_error_fraction": v_err,
            }

        rows.append(
            {
                "distribution": label,
                "sample_fraction": fraction,
                "sample_size": sample_size,
                "rows_actually_read": result["rows_sampled"],
                "coverage": result["rows_sampled"] / n_rows,
                "seconds": result["seconds"],
                "worst_rank_error": worst_rank,
                "worst_value_error_fraction": worst_value,
                "quantiles": per_quantile,
            }
        )
        print(
            f"  {label:<10} frac={fraction:<6} read={result['rows_sampled']:>9,} "
            f"({result['rows_sampled'] / n_rows:6.1%})  {result['seconds'] * 1000:8.1f} ms  "
            f"rank_err={worst_rank:.5f}  value_err={worst_value:.4%}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benches" / "results" / "sampling_study.json",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = [rng.lognormvariate(0.0, 1.0) for _ in range(args.rows)]

    print(f"Sampling study over {args.rows:,} rows\n")

    print("shuffled (rows in random order):")
    shuffled = list(base)
    rng.shuffle(shuffled)
    results = study_one("shuffled", shuffled, args.data_dir, args.repetitions)

    print("\nsorted (rows ascending -- isolates t-digest insertion-order sensitivity):")
    results += study_one("sorted", sorted(base), args.data_dir, args.repetitions)

    report = {
        "schema_version": 1,
        "environment": capture_environment(),
        "dataset": {
            "rows": args.rows,
            "distribution": "lognormal(0, 1)",
            "seed": args.seed,
            "variants": {
                "shuffled": "rows in random order; the baseline",
                "sorted": (
                    "rows ascending; isolates the t-digest's insertion-order "
                    "sensitivity, since chunked reading means the sample is "
                    "stratified across the file rather than a biased prefix"
                ),
            },
        },
        "methodology": {
            "repetitions": args.repetitions,
            "headline_statistic": "median",
            "process_isolation": "one fresh interpreter per repetition",
            "rank_error": "tie-aware distance from the requested quantile to the band the estimate answers",
            "value_error": "absolute error as a fraction of the data range",
        },
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
