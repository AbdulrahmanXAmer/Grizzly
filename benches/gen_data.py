"""Deterministic synthetic dataset generation for the Grizzly benchmark suite.

Datasets are generated from a seeded ``random.Random`` and written with fixed
formatting, so the same ``(shape, n_rows, seed)`` triple produces a
byte-identical file on any platform. That property is what lets benchmark
results be compared across machines and across commits.

Two shapes are provided:

``numeric``
    All-float columns with a header. Used for the transform and regression
    workloads, and for profiling a homogeneous file. The final column is a
    noisy linear function of the others, so it doubles as a regression target.

``mixed``
    Floats, integers, low-cardinality categoricals, and injected nulls. Used
    for the profiling workload, where type inference and missing-data handling
    are the point rather than raw float parsing.

Usage::

    python -m benches.gen_data --out data/ --rows 100000 500000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

# Categorical vocabulary for the "mixed" shape. Small and fixed so that
# cardinality is a known quantity when interpreting profiling results.
CATEGORIES: Sequence[str] = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
)

# Fraction of cells blanked out in the "mixed" shape, per nullable column.
NULL_RATE = 0.05


def numeric_rows(
    n_rows: int,
    n_features: int,
    seed: int,
    noise: float = 0.1,
) -> Iterator[str]:
    """Yield CSV lines for the all-float shape, including the header.

    The last column (``target``) is ``w . x + b + noise`` so that the file can
    be used to benchmark regression as well as profiling and scaling.
    """
    rng = random.Random(seed)
    weights = [rng.uniform(-2.0, 2.0) for _ in range(n_features)]
    bias = rng.uniform(-1.0, 1.0)

    header = [f"f_{i}" for i in range(n_features)] + ["target"]
    yield ",".join(header) + "\n"

    for _ in range(n_rows):
        x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        y = sum(w * xi for w, xi in zip(weights, x, strict=True)) + bias + rng.gauss(0.0, noise)
        yield ",".join(f"{v:.6f}" for v in (*x, y)) + "\n"


def binary_rows(
    n_rows: int,
    n_features: int,
    seed: int,
    noise: float = 1.0,
) -> Iterator[str]:
    """Yield CSV lines for the binary-classification shape, header included.

    The label is drawn from a logistic model of the features -- ``target ~
    Bernoulli(sigmoid(w . x + b))`` -- rather than thresholded deterministically.
    That matters: a hard threshold produces perfectly separable classes, on
    which logistic regression's likelihood has no finite maximum, coefficients
    run away to infinity, and every implementation "agrees" only in the sense
    that they all diverge. Sampling the label gives overlapping classes, a
    well-posed optimum, and an accuracy ceiling below 1.0 -- which is what real
    data looks like and what makes a cross-library comparison mean anything.

    ``noise`` scales the logit; larger values sharpen the classes.
    """
    rng = random.Random(seed)
    weights = [rng.uniform(-1.5, 1.5) for _ in range(n_features)]
    bias = rng.uniform(-0.5, 0.5)

    header = [f"f_{i}" for i in range(n_features)] + ["target"]
    yield ",".join(header) + "\n"

    for _ in range(n_rows):
        x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        z = noise * (sum(w * xi for w, xi in zip(weights, x, strict=True)) + bias)
        # Numerically stable sigmoid; z can be large for wide feature counts.
        p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
        label = 1 if rng.random() < p else 0
        yield ",".join((*(f"{v:.6f}" for v in x), str(label))) + "\n"


def mixed_rows(n_rows: int, n_features: int, seed: int) -> Iterator[str]:
    """Yield CSV lines for the heterogeneous shape, including the header.

    Column layout repeats every four columns: float, int, categorical, and a
    nullable float. This keeps the type mix predictable while exercising the
    inference, frequency-tracking, and missing-value paths.
    """
    rng = random.Random(seed)

    names: list[str] = []
    for i in range(n_features):
        kind = i % 4
        names.append(f"{('num', 'int', 'cat', 'opt')[kind]}_{i}")
    yield ",".join(names) + "\n"

    for _ in range(n_rows):
        fields: list[str] = []
        for i in range(n_features):
            kind = i % 4
            if kind == 0:
                fields.append(f"{rng.gauss(0.0, 1.0):.6f}")
            elif kind == 1:
                fields.append(str(rng.randint(-10_000, 10_000)))
            elif kind == 2:
                fields.append(CATEGORIES[rng.randrange(len(CATEGORIES))])
            else:
                # Nullable float: empty field represents a missing value.
                fields.append("" if rng.random() < NULL_RATE else f"{rng.gauss(5.0, 2.0):.6f}")
        yield ",".join(fields) + "\n"


def write_dataset(
    out_path: Path,
    *,
    shape: str,
    n_rows: int,
    n_features: int,
    seed: int,
) -> Path:
    """Write one dataset to ``out_path``, creating parent directories."""
    if shape == "numeric":
        rows = numeric_rows(n_rows, n_features, seed)
    elif shape == "binary":
        rows = binary_rows(n_rows, n_features, seed)
    elif shape == "mixed":
        rows = mixed_rows(n_rows, n_features, seed)
    else:
        raise ValueError(f"unknown shape: {shape!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Branch rather than dispatch through a variable: gzip.open and open have
    # different signatures, and a shared alias erases the text-mode return type.
    if out_path.suffix == ".gz":
        with gzip.open(out_path, "wt", newline="") as fh:
            fh.writelines(rows)
    else:
        with open(out_path, "w", newline="") as fh:
            fh.writelines(rows)
    return out_path


def sha256(path: Path) -> str:
    """Content hash, used to assert datasets are identical across runs."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_path(out_dir: Path, shape: str, n_rows: int) -> Path:
    """Canonical on-disk name for a generated dataset."""
    return out_dir / f"{shape}_{n_rows}.csv"


def ensure_datasets(
    out_dir: Path,
    shapes: Sequence[str],
    row_counts: Sequence[int],
    *,
    n_features: int,
    seed: int,
    force: bool = False,
) -> dict[tuple[str, int], Path]:
    """Generate any missing datasets and return the full path mapping."""
    paths: dict[tuple[str, int], Path] = {}
    for shape in shapes:
        for n_rows in row_counts:
            path = dataset_path(out_dir, shape, n_rows)
            if force or not path.exists():
                write_dataset(
                    path,
                    shape=shape,
                    n_rows=n_rows,
                    n_features=n_features,
                    seed=seed,
                )
            paths[(shape, n_rows)] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[100_000, 500_000],
        help="row counts to generate",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["numeric", "mixed"],
        choices=["numeric", "binary", "mixed"],
    )
    parser.add_argument("--features", type=int, default=20, help="feature columns per row")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args()

    paths = ensure_datasets(
        args.out,
        args.shapes,
        args.rows,
        n_features=args.features,
        seed=args.seed,
        force=args.force,
    )
    for (shape, n_rows), path in sorted(paths.items()):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{shape:>8} {n_rows:>9,} rows  {size_mb:8.2f} MB  {sha256(path)[:16]}  {path}")


if __name__ == "__main__":
    main()
