"""Benchmark driver for Grizzly against pandas and polars.

Every number published in the project README is produced by this script. It is
deliberately conservative about the ways a benchmark can flatter its author:

* **Cold process per measurement.** Each repetition runs in a fresh
  interpreter (``benches/_runner.py``), so no library benefits from another's
  warm caches and the order libraries run in cannot change the result.
* **Warmup discarded.** The first repetition of each cell primes the OS page
  cache and is thrown away; the reported figure is the median of the rest.
* **Equivalence enforced.** Every library's output is fingerprinted and the
  fingerprints must agree. If one implementation quietly did less work, the
  run reports a mismatch instead of a speedup.
* **Full coverage enforced.** Grizzly is sampling-first: ``sample_size`` caps
  how many rows it reads. Left at its default it would profile ~1000 rows of a
  500,000-row file and appear arbitrarily fast. The driver passes a sample
  size above the row count and then asserts Grizzly actually saw every row.
* **Environment recorded.** Hardware, OS, interpreter, compiler, and the exact
  version of every comparison library are written into the results file.

Usage::

    python -m benches.bench --rows 100000 500000 --repetitions 5
    python -m benches.bench --strict --out benches/results/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benches.gen_data import dataset_path, ensure_datasets, sha256

REPO_ROOT = Path(__file__).resolve().parent.parent

# Which libraries participate in which workload.
LIBRARIES = ("grizzly", "pandas", "polars")

# (workload, dataset shape) pairs to measure.
#   profile   -- runs on both shapes; "mixed" is where type inference matters.
#   transform -- numeric only; min-max scaling of a categorical is undefined.
CELLS = (
    ("profile", "numeric"),
    ("profile", "mixed"),
    ("transform", "numeric"),
)

# Grizzly's sample_size is set to this multiple of the row count. Chunked
# parallel reads align to record boundaries and can stop slightly short of an
# exact request, so a margin above 1.0 is required for full coverage.
SAMPLE_SIZE_MARGIN = 4


class BenchmarkError(RuntimeError):
    """Raised when a run cannot produce trustworthy numbers."""


# --------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------


def _cpu_model() -> str:
    """Best-effort CPU model string, for the record in results.json."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover - informational only
        pass
    return platform.processor() or "unknown"


def _total_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return None


def _command_output(cmd: Sequence[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out.stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover - informational only
        return "unknown"


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return _command_output(["git", "-C", str(REPO_ROOT), *args])

    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty.stdout.strip()),
    }


def capture_environment() -> dict[str, Any]:
    """Everything a reader needs to judge whether these numbers apply to them."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _total_memory_bytes(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "rustc": _command_output(["rustc", "--version"]),
        "cargo_profile": "release (lto=true, codegen-units=1, opt-level=3)",
        "git": _git_state(),
    }


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def run_once(
    library: str,
    workload: str,
    path: Path,
    sample_size: int,
    scratch: Path,
) -> dict[str, Any]:
    """Run one repetition in a fresh interpreter and return its JSON payload."""
    cmd = [
        sys.executable,
        "-m",
        "benches._runner",
        "--library",
        library,
        "--workload",
        workload,
        "--path",
        str(path),
        "--sample-size",
        str(sample_size),
    ]
    if workload == "transform":
        cmd += ["--out", str(scratch / f"{library}_{path.stem}_scaled.csv")]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if proc.returncode != 0:
        raise BenchmarkError(f"{library}/{workload} on {path.name} failed:\n{proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise BenchmarkError(
            f"{library}/{workload} produced unparseable output: {proc.stdout!r}"
        ) from exc


def summarise(samples: Iterable[float]) -> dict[str, float]:
    values = sorted(samples)
    return {
        "median_ms": statistics.median(values) * 1000,
        "min_ms": values[0] * 1000,
        "max_ms": values[-1] * 1000,
        "mean_ms": statistics.fmean(values) * 1000,
        "stdev_ms": (statistics.stdev(values) * 1000) if len(values) > 1 else 0.0,
    }


def measure_cell(
    library: str,
    workload: str,
    path: Path,
    sample_size: int,
    repetitions: int,
    scratch: Path,
    verbose: bool,
) -> dict[str, Any]:
    """Warm up once, then take ``repetitions`` timed measurements."""
    run_once(library, workload, path, sample_size, scratch)  # discarded warmup

    seconds: list[float] = []
    peak_rss: list[int] = []
    fingerprint: dict[str, Any] | None = None
    version = "unknown"

    for _ in range(repetitions):
        payload = run_once(library, workload, path, sample_size, scratch)
        seconds.append(payload["seconds"])
        peak_rss.append(payload["peak_rss_bytes"])
        fingerprint = payload["fingerprint"]
        version = payload["library_version"]

    result: dict[str, Any] = {
        "library": library,
        "library_version": version,
        "repetitions": repetitions,
        "timing": summarise(seconds),
        "peak_rss_bytes": max(peak_rss),
        "fingerprint": fingerprint,
    }
    if verbose:
        print(
            f"    {library:<8} {result['timing']['median_ms']:9.2f} ms  "
            f"(sd {result['timing']['stdev_ms']:6.2f})  "
            f"peak RSS {result['peak_rss_bytes'] / 1e6:7.1f} MB"
        )
    return result


# --------------------------------------------------------------------------
# equivalence
# --------------------------------------------------------------------------


def check_equivalence(
    workload: str,
    results: dict[str, dict[str, Any]],
    expected_rows: int,
) -> dict[str, Any]:
    """Assert every library produced the same answer over the same rows.

    Returns a report rather than raising, so a mismatch is recorded in the
    results file and visible to anyone reading it. ``--strict`` turns a
    mismatch into a non-zero exit.
    """
    problems: list[str] = []

    for library, result in results.items():
        rows = result["fingerprint"].get("rows") or result["fingerprint"].get("rows_written")
        if rows != expected_rows:
            problems.append(
                f"{library} covered {rows} rows, expected {expected_rows} "
                "(comparison would not be like-for-like)"
            )

    if workload == "profile":
        reference_name, reference = next(iter(results.items()))
        for library, result in results.items():
            if library == reference_name:
                continue
            ref_cols = reference["fingerprint"]["columns"]
            got_cols = result["fingerprint"]["columns"]
            if set(ref_cols) != set(got_cols):
                problems.append(
                    f"{library} reported columns {sorted(got_cols)}, "
                    f"{reference_name} reported {sorted(ref_cols)}"
                )
                continue
            for name, ref_stats in ref_cols.items():
                got_stats = got_cols[name]
                for key in ("observed", "null_count", "min", "max", "mean"):
                    if ref_stats[key] != got_stats[key]:
                        problems.append(
                            f"column {name!r} {key}: {reference_name}="
                            f"{ref_stats[key]!r} vs {library}={got_stats[key]!r}"
                        )

    if workload == "transform":
        for library, result in results.items():
            ranges = result["fingerprint"].get("ranges", {})
            off = {
                name: span
                for name, span in ranges.items()
                if not (abs(span[0]) < 1e-6 and abs(span[1] - 1.0) < 1e-6)
            }
            if off:
                problems.append(f"{library} left columns unscaled or mis-scaled: {off}")
        scaled_counts = {lib: r["fingerprint"]["cols_scaled"] for lib, r in results.items()}
        if len(set(scaled_counts.values())) > 1:
            problems.append(f"libraries scaled different column counts: {scaled_counts}")

    return {"status": "ok" if not problems else "mismatch", "problems": problems}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run_benchmarks(
    data_dir: Path,
    row_counts: Sequence[int],
    repetitions: int,
    n_features: int,
    seed: int,
    scratch: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    ensure_datasets(
        data_dir,
        shapes=sorted({shape for _, shape in CELLS}),
        row_counts=row_counts,
        n_features=n_features,
        seed=seed,
    )
    scratch.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, Any] = {}
    measurements: list[dict[str, Any]] = []

    for workload, shape in CELLS:
        for n_rows in row_counts:
            path = dataset_path(data_dir, shape, n_rows)
            key = f"{shape}_{n_rows}"
            if key not in datasets:
                datasets[key] = {
                    "shape": shape,
                    "rows": n_rows,
                    "feature_columns": n_features,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }

            sample_size = n_rows * SAMPLE_SIZE_MARGIN
            if verbose:
                print(f"\n  {workload} / {shape} / {n_rows:,} rows")

            results = {
                library: measure_cell(
                    library, workload, path, sample_size, repetitions, scratch, verbose
                )
                for library in LIBRARIES
            }
            equivalence = check_equivalence(workload, results, n_rows)
            if verbose and equivalence["status"] != "ok":
                for problem in equivalence["problems"]:
                    print(f"    !! {problem}")

            measurements.append(
                {
                    "workload": workload,
                    "dataset": key,
                    "grizzly_sample_size": sample_size,
                    "equivalence": equivalence,
                    "results": results,
                }
            )

    return {
        "schema_version": 1,
        "environment": capture_environment(),
        "methodology": {
            "process_isolation": "one fresh interpreter per repetition",
            "warmup_runs_discarded": 1,
            "repetitions": repetitions,
            "headline_statistic": "median",
            "grizzly_sample_size_margin": SAMPLE_SIZE_MARGIN,
            "timed_region": (
                "profile: read CSV + compute per-column stats; "
                "transform: read CSV + min-max scale numeric columns + write CSV"
            ),
        },
        "datasets": datasets,
        "measurements": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 500_000])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benches" / "results" / "results.json",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="directory for transform outputs (default: <data-dir>/scratch)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any cross-library equivalence check fails",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    scratch = args.scratch or (args.data_dir / "scratch")
    report = run_benchmarks(
        args.data_dir,
        args.rows,
        args.repetitions,
        args.features,
        args.seed,
        scratch,
        verbose=not args.quiet,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not args.quiet:
        print(f"\nwrote {args.out}")

    mismatched = [m for m in report["measurements"] if m["equivalence"]["status"] != "ok"]
    if mismatched:
        print(f"\n{len(mismatched)} measurement(s) failed the equivalence check:")
        for m in mismatched:
            print(f"  {m['workload']}/{m['dataset']}")
            for problem in m["equivalence"]["problems"]:
                print(f"    - {problem}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
