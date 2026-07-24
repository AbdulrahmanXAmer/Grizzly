"""Model-fitting benchmark: CSV on disk to fitted linear model.

The transform and profile benchmarks measure data preparation; this one
measures the step preparation exists for. The timed workload per method is the
full practitioner journey — read the file, make an 80/20 split, fit, score on
the held-out set — because "how fast is training" only means something with
the loading included.

Methods compared, each through its own idiomatic path:

    grizzly_exact     csv_linear_regression (parallel normal equations)
    grizzly_sgd       csv_sgd_regression, 10 epochs, cached replay
    pandas_sklearn    pandas.read_csv -> sklearn LinearRegression
    polars_sklearn    polars.read_csv -> sklearn LinearRegression
    pandas_sgd        pandas.read_csv -> StandardScaler -> SGDRegressor(10)

Same discipline as the other suites: one cold process per repetition, first
repetition discarded as warmup, median reported, peak RSS alongside time, and
a correctness gate — every method's coefficients must agree with the exact-OLS
consensus (SGD methods within a looser band), so a fast wrong answer cannot be
published as a fast answer.

Usage::

    python -m benches.bench_fit --rows 500000 --repetitions 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from benches.bench import capture_environment
from benches.gen_data import dataset_path, ensure_datasets, sha256

REPO_ROOT = Path(__file__).resolve().parent.parent

EXACT_METHODS = ("grizzly_exact", "pandas_sklearn", "polars_sklearn")
SGD_METHODS = ("grizzly_sgd", "pandas_sgd")
ALL_METHODS = EXACT_METHODS + SGD_METHODS

# Exact OLS on ~400k train rows: different random splits move coefficients by
# sampling noise only. SGD adds optimizer error on top.
EXACT_REL_TOL = 0.05
SGD_REL_TOL = 0.15
R2_ABS_TOL = 0.02


class FitBenchmarkError(RuntimeError):
    pass


def run_once(method: str, path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benches._fit_runner",
            "--method",
            method,
            "--path",
            str(path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise FitBenchmarkError(f"{method} failed:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def measure(method: str, path: Path, repetitions: int, verbose: bool) -> dict[str, Any]:
    run_once(method, path)  # discarded warmup

    seconds: list[float] = []
    peak_rss: list[int] = []
    payload: dict[str, Any] = {}
    for _ in range(repetitions):
        payload = run_once(method, path)
        seconds.append(payload["seconds"])
        peak_rss.append(payload["peak_rss_bytes"])

    ordered = sorted(seconds)
    result = {
        "method": method,
        "family": "exact" if method in EXACT_METHODS else "sgd",
        "repetitions": repetitions,
        "timing": {
            "median_ms": statistics.median(ordered) * 1000,
            "min_ms": ordered[0] * 1000,
            "max_ms": ordered[-1] * 1000,
            "stdev_ms": (statistics.stdev(ordered) * 1000) if len(ordered) > 1 else 0.0,
        },
        "peak_rss_bytes": max(peak_rss),
        "r2": payload["r2"],
        "coef": payload["coef"],
    }
    if verbose:
        print(
            f"  {method:<16} {result['timing']['median_ms']:9.1f} ms  "
            f"(sd {result['timing']['stdev_ms']:6.1f})  "
            f"r2={result['r2']:.4f}  peak RSS {result['peak_rss_bytes'] / 1e6:6.1f} MB"
        )
    return result


def check_agreement(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Every method must land on the same model, up to its family's tolerance.

    The consensus is the element-wise mean of the exact methods' coefficients;
    deviations are measured relative to the coefficient vector's scale, so a
    near-zero coefficient does not manufacture a huge relative error.
    """
    exact = [r for r in results if r["family"] == "exact"]
    n_coef = len(exact[0]["coef"])
    consensus = [sum(r["coef"][i] for r in exact) / len(exact) for i in range(n_coef)]
    scale = max(max(abs(c) for c in consensus), 1e-9)
    r2_values = [r["r2"] for r in results]

    problems: list[str] = []
    for r in results:
        tolerance = EXACT_REL_TOL if r["family"] == "exact" else SGD_REL_TOL
        worst = max(abs(a - b) / scale for a, b in zip(r["coef"], consensus, strict=True))
        r["max_coef_deviation"] = worst
        if worst > tolerance:
            problems.append(f"{r['method']}: coefficient deviation {worst:.4f} exceeds {tolerance}")

    if max(r2_values) - min(r2_values) > R2_ABS_TOL:
        problems.append(
            f"r2 spread {max(r2_values) - min(r2_values):.4f} exceeds {R2_ABS_TOL}: "
            + ", ".join(f"{r['method']}={r['r2']:.4f}" for r in results)
        )

    return {"status": "ok" if not problems else "mismatch", "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benches" / "results" / "fit_results.json",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ensure_datasets(
        args.data_dir,
        shapes=["numeric"],
        row_counts=[args.rows],
        n_features=args.features,
        seed=args.seed,
    )
    path = dataset_path(args.data_dir, "numeric", args.rows)
    verbose = not args.quiet

    if verbose:
        print(f"fit benchmark: {args.rows:,} rows x {args.features} features\n")

    results = [measure(m, path, args.repetitions, verbose) for m in ALL_METHODS]
    agreement = check_agreement(results)
    if verbose and agreement["status"] != "ok":
        for problem in agreement["problems"]:
            print(f"  !! {problem}")

    report = {
        "schema_version": 1,
        "environment": capture_environment(),
        "dataset": {
            "rows": args.rows,
            "feature_columns": args.features,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        },
        "methodology": {
            "workload": (
                "read CSV from disk, 80/20 train/test split, fit a linear "
                "model, score R^2 on the held-out set; scaling inside the "
                "timed region where the method requires it"
            ),
            "sgd_epochs": 10,
            "repetitions": args.repetitions,
            "headline_statistic": "median",
            "process_isolation": "one fresh interpreter per repetition",
            "agreement_gate": (
                "coefficients must match the exact-OLS consensus within "
                f"{EXACT_REL_TOL:.0%} (exact) / {SGD_REL_TOL:.0%} (SGD) of the "
                "coefficient scale"
            ),
        },
        "agreement": agreement,
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if verbose:
        print(f"\nwrote {args.out}")

    if agreement["status"] != "ok" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
