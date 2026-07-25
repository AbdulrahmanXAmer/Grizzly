"""Classification benchmark: CSV on disk to a scored binary classifier.

The counterpart to `bench_fit` for the classification path. The timed workload
per method is the full practitioner journey -- read the file, make an 80/20
split, fit, score on the held-out set -- because "how fast is training" only
means something with the loading included.

Methods compared, each through its own idiomatic path, in two model families:

    grizzly_logistic  csv_logistic_regression, 10 epochs, cached replay
    pandas_sklearn    pandas.read_csv -> sklearn LogisticRegression (L-BFGS)
    polars_sklearn    polars.read_csv -> sklearn LogisticRegression (L-BFGS)
    pandas_sgd        pandas.read_csv -> StandardScaler -> SGDClassifier(10)

    grizzly_gnb       csv_gaussian_nb (one grouped-moments pass, no cache)
    pandas_gnb        pandas.read_csv -> sklearn GaussianNB
    polars_gnb        polars.read_csv -> sklearn GaussianNB

Same discipline as the other suites: one cold process per repetition, first
repetition discarded as warmup, median reported, peak RSS alongside time, and a
correctness gate.

The gate compares within a family only. Logistic regression and Naive Bayes
are different models that legitimately reach different held-out numbers on the
same data; cross-family agreement would gate on modelling philosophy, not
correctness.

The gate is on the *metrics*, not the coefficients, and that is deliberate.
`bench_fit` gates on coefficients because for least squares every method is
solving for the same unique closed-form answer. Logistic regression has no
closed form: L-BFGS runs to convergence while a ten-epoch SGD stops well short,
so their coefficients legitimately differ by more than sampling noise even
though both classify about equally well. Gating on coefficients here would
either fail on a correct implementation or need a tolerance so loose it could
not catch a real error. Held-out accuracy, ROC-AUC, and log-loss are what a
classifier is judged on, they converge far faster than the coefficients do, and
a genuinely broken fit moves them immediately -- so they are the honest gate.

Usage::

    python -m benches.bench_classify --rows 500000 --repetitions 5
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

FAMILIES = {
    "logistic": ("grizzly_logistic", "pandas_sklearn", "polars_sklearn", "pandas_sgd"),
    "gnb": ("grizzly_gnb", "pandas_gnb", "polars_gnb"),
}
ALL_METHODS = tuple(m for methods in FAMILIES.values() for m in methods)
FAMILY_OF = {m: fam for fam, methods in FAMILIES.items() for m in methods}

# Spread allowed across methods on the held-out set. Two fits on equally sized
# but different splits move these by well under a point; anything larger is a
# real disagreement about the model, not sampling noise.
#
# Log-loss gets the loosest band because it is the most sensitive of the three
# to how far an optimizer has converged -- it scores the calibration of every
# probability, not just which side of 0.5 it fell on. On the small datasets CI
# runs, a ten-epoch SGD is measurably less converged than an L-BFGS fit and the
# spread widens accordingly, without either being wrong. It is still a real
# gate: a genuinely broken classifier scores near log(2) = 0.69, which is an
# order of magnitude outside this band.
ACCURACY_ABS_TOL = 0.02
ROC_AUC_ABS_TOL = 0.02
LOG_LOSS_ABS_TOL = 0.05

GATED_METRICS = {
    "accuracy": ACCURACY_ABS_TOL,
    "roc_auc": ROC_AUC_ABS_TOL,
    "log_loss": LOG_LOSS_ABS_TOL,
}


class ClassifyBenchmarkError(RuntimeError):
    pass


def run_once(method: str, path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benches._classify_runner",
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
        raise ClassifyBenchmarkError(f"{method} failed:\n{proc.stderr.strip()}")
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
        "family": FAMILY_OF[method],
        "repetitions": repetitions,
        "timing": {
            "median_ms": statistics.median(ordered) * 1000,
            "min_ms": ordered[0] * 1000,
            "max_ms": ordered[-1] * 1000,
            "stdev_ms": (statistics.stdev(ordered) * 1000) if len(ordered) > 1 else 0.0,
        },
        "peak_rss_bytes": max(peak_rss),
        "accuracy": payload["accuracy"],
        "roc_auc": payload["roc_auc"],
        "log_loss": payload["log_loss"],
        "coef": payload["coef"],
    }
    if verbose:
        print(
            f"  {method:<18} {result['timing']['median_ms']:9.1f} ms  "
            f"(sd {result['timing']['stdev_ms']:6.1f})  "
            f"acc={result['accuracy']:.4f}  auc={result['roc_auc']:.4f}  "
            f"peak RSS {result['peak_rss_bytes'] / 1e6:6.1f} MB"
        )
    return result


def check_agreement(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Every method must land on an equally good classifier *within its family*.

    Compared as a spread across methods rather than against a designated
    reference: there is no privileged "correct" answer to defer to here, and any
    one method drifting shows up in the spread either way. Families are gated
    separately because different models legitimately score differently.
    """
    problems: list[str] = []
    for family in FAMILIES:
        members = [r for r in results if r["family"] == family]
        for metric, tolerance in GATED_METRICS.items():
            values = [r[metric] for r in members]
            spread = max(values) - min(values)
            if spread > tolerance:
                problems.append(
                    f"[{family}] {metric} spread {spread:.4f} exceeds {tolerance}: "
                    + ", ".join(f"{r['method']}={r[metric]:.4f}" for r in members)
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
        default=REPO_ROOT / "benches" / "results" / "classify_results.json",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ensure_datasets(
        args.data_dir,
        shapes=["binary"],
        row_counts=[args.rows],
        n_features=args.features,
        seed=args.seed,
    )
    path = dataset_path(args.data_dir, "binary", args.rows)
    verbose = not args.quiet

    if verbose:
        print(f"classification benchmark: {args.rows:,} rows x {args.features} features\n")

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
            "generator": (
                "labels sampled from Bernoulli(sigmoid(w.x + b)) rather than "
                "thresholded, so the classes overlap and the logistic "
                "likelihood has a finite maximum"
            ),
        },
        "methodology": {
            "workload": (
                "read CSV from disk, 80/20 train/test split, fit a binary "
                "classifier, score accuracy / ROC-AUC / log-loss on the "
                "held-out set; scaling inside the timed region where the "
                "method requires it"
            ),
            "sgd_epochs": 10,
            "repetitions": args.repetitions,
            "headline_statistic": "median",
            "process_isolation": "one fresh interpreter per repetition",
            "agreement_gate": (
                "held-out metrics must agree across methods of the same model "
                f"family within {ACCURACY_ABS_TOL} accuracy / "
                f"{ROC_AUC_ABS_TOL} ROC-AUC / {LOG_LOSS_ABS_TOL} log-loss; "
                "families are gated separately because different models "
                "legitimately score differently, and coefficients are not "
                "gated because logistic regression has no closed form and a "
                "converged L-BFGS fit legitimately differs from a ten-epoch "
                "SGD one"
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
