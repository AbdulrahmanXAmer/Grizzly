"""End-to-end demo: profile, standardize, train, and check for drift.

Runs the whole Grizzly surface against a real dataset -- NYC yellow taxi trip
records -- in the order a training pipeline actually uses it:

1. **Profile** the training month in one streaming pass, and save that profile
   as the reference.
2. **Standardize** the numeric features to zero mean and unit variance,
   streaming to a new file, using the moments from step 1.
3. **Train** a model on the standardized data, both closed-form and by
   streaming SGD, and compare them.
4. **Check for drift** by profiling a *later* month against the reference from
   step 1 -- the step that a faster DataFrame does not give you.

The dataset is real and the drift is real: fare amounts, trip distances, and
tip behaviour genuinely move month to month, so step 4 finds something rather
than demonstrating against synthetic noise.

If the dataset cannot be downloaded, the demo falls back to generated data with
injected drift, so it still runs offline. Which mode was used is stated in the
output rather than hidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import grizzly
from grizzly import drift

REPO_ROOT = Path(__file__).resolve().parent.parent

# NYC Taxi & Limousine Commission trip records, published as parquet.
TAXI_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"

# The columns worth profiling: continuous, meaningful, and genuinely drifting.
TAXI_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "passenger_count",
    "trip_duration_min",
]

# Predict the tip, not the total. `total_amount` is by construction
# fare + tip + surcharges, so predicting it from its own components gives a
# meaningless r2 of 1.0 -- and including it as a *feature* when predicting the
# tip leaks the answer outright. Tip amount from trip characteristics is a real
# task with a real error bar.
TARGET = "tip_amount"
LEAKING_FEATURES = ("total_amount",)

# Plausibility bounds for the raw TLC feed, which contains meter errors and
# negative amounts. Generous enough to keep genuine long trips.
MAX_TRIP_MILES = 100.0
MAX_FARE_DOLLARS = 500.0
MAX_TRIP_MINUTES = 360.0


def log(message: str = "") -> None:
    print(message, flush=True)


def rule(title: str) -> None:
    log()
    log(f"{'=' * 72}")
    log(f"  {title}")
    log(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# data acquisition
# ---------------------------------------------------------------------------


def _ssl_context() -> ssl.SSLContext:
    """A verifying SSL context that also works on stock macOS Python.

    Python installs from python.org do not use the system trust store, so
    urllib fails with CERTIFICATE_VERIFY_FAILED unless `Install Certificates`
    has been run. Preferring certifi's bundle when it is present keeps
    verification on rather than reaching for the usual (and unsafe) workaround
    of disabling it.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(month: str, dest: Path) -> Path | None:
    """Fetch one month of taxi data, or return None if unavailable."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    url = TAXI_URL.format(month=month)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  downloading {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "grizzly-demo"})
        with (
            urllib.request.urlopen(request, timeout=180, context=_ssl_context()) as response,
            open(dest, "wb") as fh,
        ):
            shutil.copyfileobj(response, fh)
    except (urllib.error.URLError, ssl.SSLError, OSError, TimeoutError) as exc:
        log(f"  download failed ({exc}); falling back to generated data")
        dest.unlink(missing_ok=True)
        return None
    log(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def parquet_to_csv(parquet_path: Path, csv_path: Path, limit: int) -> Path | None:
    """Flatten the columns of interest into CSV, which is Grizzly's input format."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("  pyarrow not available; cannot convert parquet")
        return None

    table = pq.read_table(parquet_path)
    names = set(table.column_names)

    required = {"trip_distance", "fare_amount", "tip_amount", "total_amount"}
    if not required.issubset(names):
        log(f"  unexpected schema, missing {sorted(required - names)}")
        return None

    n = min(limit, table.num_rows)
    columns = {name: table.column(name).to_pylist()[:n] for name in required & names}
    passengers = (
        table.column("passenger_count").to_pylist()[:n]
        if "passenger_count" in names
        else [None] * n
    )

    # Derive trip duration: a feature a real pipeline would engineer, and one
    # that drifts with traffic patterns.
    pickup = next((c for c in ("tpep_pickup_datetime", "pickup_datetime") if c in names), None)
    dropoff = next((c for c in ("tpep_dropoff_datetime", "dropoff_datetime") if c in names), None)
    if pickup and dropoff:
        starts = table.column(pickup).to_pylist()[:n]
        ends = table.column(dropoff).to_pylist()[:n]
        durations = [
            (e - s).total_seconds() / 60.0 if s is not None and e is not None else None
            for s, e in zip(starts, ends, strict=True)
        ]
    else:
        durations = [None] * n

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TAXI_COLUMNS)
        for i in range(n):
            row = [
                columns["trip_distance"][i],
                columns["fare_amount"][i],
                columns["tip_amount"][i],
                columns["total_amount"][i],
                passengers[i],
                durations[i],
            ]
            if any(v is None for v in row[:4]):
                continue
            distance, fare, tip, total = row[0], row[1], row[2], row[3]
            duration = durations[i]
            # Plausibility filters. The raw TLC feed contains negative fares,
            # zero-distance trips, and meter errors recording tens of thousands
            # of miles. Those are not outliers to be modelled, they are broken
            # records: left in, they dominate the standard deviation and
            # compress every real value toward zero when standardized.
            if not (0 < distance <= MAX_TRIP_MILES):
                continue
            if not (0 < fare <= MAX_FARE_DOLLARS):
                continue
            if not (0 <= tip <= MAX_FARE_DOLLARS):
                continue
            if total <= 0:
                continue
            if duration is not None and not (0 < duration <= MAX_TRIP_MINUTES):
                continue
            writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])
            written += 1

    log(f"  wrote {csv_path.name}: {written:,} rows")
    return csv_path if written else None


def synthesize(path: Path, n_rows: int, seed: int, *, drifted: bool) -> Path:
    """Generate taxi-shaped data, optionally with realistic drift injected."""
    rng = random.Random(seed)
    # Drifted month: longer trips, higher fares, more generous tipping -- the
    # kind of change a fare increase or a seasonal shift would produce.
    distance_scale = 1.35 if drifted else 1.0
    fare_per_mile = 4.10 if drifted else 3.20
    tip_rate = 0.22 if drifted else 0.15

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TAXI_COLUMNS)
        for _ in range(n_rows):
            distance = max(0.1, rng.lognormvariate(0.6, 0.8) * distance_scale)
            duration = max(1.0, distance * rng.uniform(2.5, 6.0))
            fare = 3.0 + distance * fare_per_mile * rng.uniform(0.9, 1.1)
            tip = max(0.0, fare * tip_rate * rng.uniform(0.0, 2.0))
            total = fare + tip + 1.0
            passengers = rng.choice([1, 1, 1, 2, 2, 3, 4])
            writer.writerow(
                [
                    f"{distance:.4f}",
                    f"{fare:.4f}",
                    f"{tip:.4f}",
                    f"{total:.4f}",
                    passengers,
                    f"{duration:.4f}",
                ]
            )
    return path


def acquire(
    data_dir: Path, months: tuple[str, str], rows: int, offline: bool
) -> tuple[Path, Path, str]:
    """Return (reference_csv, current_csv, source_description)."""
    if not offline:
        reference_pq = download(months[0], data_dir / f"yellow_{months[0]}.parquet")
        current_pq = download(months[1], data_dir / f"yellow_{months[1]}.parquet")
        if reference_pq and current_pq:
            reference = parquet_to_csv(reference_pq, data_dir / f"taxi_{months[0]}.csv", rows)
            current = parquet_to_csv(current_pq, data_dir / f"taxi_{months[1]}.csv", rows)
            if reference and current:
                return reference, current, f"NYC yellow taxi trips, {months[0]} vs {months[1]}"

    log("  using generated taxi-shaped data with injected drift")
    reference = synthesize(data_dir / "taxi_reference.csv", rows, seed=1, drifted=False)
    current = synthesize(data_dir / "taxi_current.csv", rows, seed=2, drifted=True)
    return reference, current, "generated taxi-shaped data (offline fallback)"


# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------


def stage_profile(path: Path, sample_size: int) -> dict[str, Any]:
    rule("1. Profile the training month in one streaming pass")
    start = time.perf_counter()
    profile = grizzly.csv_profile(str(path), sample_size=sample_size, lite=False)
    elapsed = time.perf_counter() - start

    size_mb = path.stat().st_size / 1e6
    log(f"  {path.name}: {size_mb:.1f} MB, {profile['rows_sampled']:,} rows in {elapsed:.3f}s")
    log(f"  throughput: {size_mb / elapsed:.0f} MB/s")
    log()
    log(f"  {'column':<20} {'type':>8} {'mean':>12} {'std':>12} {'p95':>12} {'nulls':>7}")
    log(f"  {'-' * 76}")
    for column in profile["columns"]:
        mean = column.get("mean")
        std = column.get("std")
        p95 = column.get("p95")
        log(
            f"  {column['name']:<20} {column['inferred']:>8} "
            f"{mean if mean is None else f'{mean:12.3f}':>12} "
            f"{std if std is None else f'{std:12.3f}':>12} "
            f"{p95 if p95 is None else f'{p95:12.3f}':>12} "
            f"{column['null_count']:>7,}"
        )
    return profile


def stage_standardize(path: Path, out_path: Path, sample_size: int) -> None:
    rule("2. Standardize features, streaming CSV to CSV")
    start = time.perf_counter()
    params = grizzly.csv_standardize_params(str(path), sample_size=sample_size)["params"]
    result = grizzly.csv_transform_standardize(str(path), str(out_path), params)
    elapsed = time.perf_counter() - start

    log(
        f"  scaled {result['numeric_cols_scaled']} columns over "
        f"{result['rows_written']:,} rows in {elapsed:.3f}s"
    )
    log("  memory is bounded by the chunk size, not the file size")

    check = grizzly.csv_profile(str(out_path), sample_size=sample_size, lite=False)
    log()
    log(f"  {'column':<20} {'mean':>12} {'std':>12}   (should be 0.0 and 1.0)")
    log(f"  {'-' * 60}")
    for column in check["columns"][:6]:
        mean, std = column.get("mean"), column.get("std")
        if mean is None or std is None:
            continue
        log(f"  {column['name']:<20} {mean:12.6f} {std:12.6f}")


def stage_train(path: Path, sample_size: int) -> None:
    rule("3. Train: closed form vs streaming SGD")
    features = [c for c in TAXI_COLUMNS if c != TARGET and c not in LEAKING_FEATURES]

    start = time.perf_counter()
    exact = grizzly.csv_linear_regression(
        str(path), target=TARGET, features=features, sample_size=sample_size, seed=0
    )
    exact_time = time.perf_counter() - start

    start = time.perf_counter()
    sgd = grizzly.csv_sgd_regression(
        str(path),
        target=TARGET,
        features=features,
        epochs=10,
        learning_rate=0.1,
        sample_size=sample_size,
        seed=0,
    )
    sgd_time = time.perf_counter() - start

    log(f"  target: {TARGET}   features: {', '.join(features)}")
    log(f"  ({', '.join(LEAKING_FEATURES)} excluded: it contains the target by construction)")
    log()
    log(
        f"  closed form : r2={exact['r2']:.5f}  train={exact['train_n']:,}  "
        f"test={exact['test_n']:,}  {exact_time:.3f}s"
    )
    log(
        f"  streaming SGD: r2={sgd['r2']:.5f}  train={sgd['train_n']:,}  "
        f"test={sgd['test_n']:,}  {sgd_time:.3f}s  ({sgd['epochs']} epochs)"
    )
    log()
    log("  The closed form accumulates an X'X matrix: O(p^2) memory.")
    log("  SGD holds only the weight vector: O(p). Same answer, different ceiling.")
    log()
    log(f"  {'feature':<20} {'closed form':>14} {'SGD':>14}")
    log(f"  {'-' * 50}")
    for name, a, b in zip(sgd["features"], exact["coef"], sgd["coef"], strict=True):
        log(f"  {name:<20} {a:14.5f} {b:14.5f}")
    log(f"  {'(intercept)':<20} {exact['intercept']:14.5f} {sgd['intercept']:14.5f}")


def stage_drift(
    reference_profile: dict[str, Any],
    current_path: Path,
    reference_json: Path,
    sample_size: int,
) -> dict[str, Any]:
    rule("4. Check a later month against the training reference")
    drift.save_reference(reference_profile, reference_json)
    log(
        f"  reference saved to {reference_json.name} "
        f"({reference_json.stat().st_size / 1024:.1f} KB)"
    )
    log("  the training data itself is not needed to make this comparison")
    log()

    start = time.perf_counter()
    report = drift.detect_drift(str(current_path), reference_json, sample_size=sample_size)
    elapsed = time.perf_counter() - start

    log(drift.format_report(report))
    log()
    log(f"  checked in {elapsed:.3f}s")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "demo")
    parser.add_argument("--rows", type=int, default=750_000, help="rows per month")
    parser.add_argument("--sample-size", type=int, default=10_000_000)
    parser.add_argument("--reference-month", default="2024-01")
    parser.add_argument("--current-month", default="2024-06")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the download and use generated data",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rule("Grizzly: streaming feature statistics for training pipelines")
    log(f"  grizzly {grizzly.__version__}  (native extension: {grizzly.is_native()})")
    log()
    log("Acquiring data")
    reference_csv, current_csv, source = acquire(
        args.data_dir, (args.reference_month, args.current_month), args.rows, args.offline
    )
    log(f"  source: {source}")

    profile = stage_profile(reference_csv, args.sample_size)
    stage_standardize(reference_csv, args.data_dir / "taxi_standardized.csv", args.sample_size)
    stage_train(reference_csv, args.sample_size)
    report = stage_drift(
        profile, current_csv, args.data_dir / "reference_profile.json", args.sample_size
    )

    rule("Summary")
    log(f"  source          : {source}")
    log(f"  drift verdict   : {report['verdict'].upper()}")
    log(
        f"  columns drifted : {report['counts']['significant']} significant, "
        f"{report['counts']['moderate']} moderate"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"source": source, "drift": report}, indent=2))
        log(f"  wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
