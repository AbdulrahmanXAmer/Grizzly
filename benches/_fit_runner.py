"""One model-fit measurement in an isolated process, for the fit benchmark.

The timed workload is the whole journey a practitioner actually takes:
**CSV on disk → 80/20 train/test split → fitted linear model → test-set R².**
Reading and any scaling the method needs are inside the timed region, because
that is what training from a file costs; nobody gets handed a warm DataFrame.

Each method uses its own idiomatic path. Grizzly's functions do the split and
evaluation internally; the sklearn methods use pandas/polars to read and
scikit-learn to split and fit. The splits are equally sized but not identical
row-for-row — on hundreds of thousands of rows the fitted coefficients agree
to sampling noise, and the parent asserts exactly that, so a fast wrong answer
cannot pass as a fast answer.

Prints one JSON object on stdout: seconds, peak RSS, coefficients, r2.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

TRAIN_FRAC = 0.8
SEED = 0
SGD_EPOCHS = 10


def peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


# ---------------------------------------------------------------------------
# grizzly
# ---------------------------------------------------------------------------


def fit_grizzly_exact(path: str, target: str) -> tuple[float, list[float], float]:
    import grizzly

    start = time.perf_counter()
    result = grizzly.csv_linear_regression(
        path, target=target, train_frac=TRAIN_FRAC, seed=SEED, sample_size=100_000_000
    )
    elapsed = time.perf_counter() - start
    return elapsed, list(result["coef"]), float(result["r2"])


def fit_grizzly_sgd(path: str, target: str) -> tuple[float, list[float], float]:
    import grizzly

    start = time.perf_counter()
    result = grizzly.csv_sgd_regression(
        path,
        target=target,
        epochs=SGD_EPOCHS,
        train_frac=TRAIN_FRAC,
        seed=SEED,
        sample_size=100_000_000,
    )
    elapsed = time.perf_counter() - start
    return elapsed, list(result["coef"]), float(result["r2"])


# ---------------------------------------------------------------------------
# pandas / polars + scikit-learn
# ---------------------------------------------------------------------------


def _sklearn_ols(X, y) -> tuple[list[float], float]:
    # Callers pre-import sklearn before starting their clocks, so these
    # resolve from sys.modules in microseconds. Import cost is a per-process
    # constant, not part of a fit, and grizzly's import is untimed too — the
    # first draft of this harness charged sklearn's ~600 ms import to two of
    # the five methods, which is exactly the kind of thumb on the scale this
    # suite exists to avoid.
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_FRAC, random_state=SEED
    )
    model = LinearRegression().fit(X_train, y_train)
    r2 = float(r2_score(y_test, model.predict(X_test)))
    return [float(c) for c in model.coef_], r2


def fit_pandas_sklearn(path: str, target: str) -> tuple[float, list[float], float]:
    import pandas as pd
    import sklearn.linear_model  # noqa: F401  (pre-import, see _sklearn_ols)
    import sklearn.metrics  # noqa: F401
    import sklearn.model_selection  # noqa: F401

    start = time.perf_counter()
    df = pd.read_csv(path)
    y = df.pop(target).to_numpy()
    X = df.to_numpy()
    coef, r2 = _sklearn_ols(X, y)
    elapsed = time.perf_counter() - start
    return elapsed, coef, r2


def fit_polars_sklearn(path: str, target: str) -> tuple[float, list[float], float]:
    import polars as pl
    import sklearn.linear_model  # noqa: F401  (pre-import, see _sklearn_ols)
    import sklearn.metrics  # noqa: F401
    import sklearn.model_selection  # noqa: F401

    start = time.perf_counter()
    df = pl.read_csv(path)
    y = df.get_column(target).to_numpy()
    X = df.drop(target).to_numpy()
    coef, r2 = _sklearn_ols(X, y)
    elapsed = time.perf_counter() - start
    return elapsed, coef, r2


def fit_pandas_sgd(path: str, target: str) -> tuple[float, list[float], float]:
    """The idiomatic sklearn SGD workflow, standardization included.

    Grizzly's SGD standardizes internally, so the sklearn pipeline gets a
    StandardScaler inside its timed region too — leaving it out would compare
    a scaled fit against an unscaled one, and SGD without scaling is not a
    workflow anyone should be timed running. Coefficients are mapped back to
    the original feature space for the cross-method comparison.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    start = time.perf_counter()
    df = pd.read_csv(path)
    y = df.pop(target).to_numpy()
    X = df.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_FRAC, random_state=SEED
    )
    scaler = StandardScaler().fit(X_train)
    model = SGDRegressor(max_iter=SGD_EPOCHS, tol=None, random_state=SEED, penalty=None).fit(
        scaler.transform(X_train), y_train
    )
    r2 = float(r2_score(y_test, model.predict(scaler.transform(X_test))))
    coef = np.asarray(model.coef_) / scaler.scale_
    elapsed = time.perf_counter() - start
    return elapsed, [float(c) for c in coef], r2


METHODS = {
    "grizzly_exact": fit_grizzly_exact,
    "grizzly_sgd": fit_grizzly_sgd,
    "pandas_sklearn": fit_pandas_sklearn,
    "polars_sklearn": fit_polars_sklearn,
    "pandas_sgd": fit_pandas_sgd,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--path", required=True)
    parser.add_argument("--target", default="target")
    args = parser.parse_args()

    elapsed, coef, r2 = METHODS[args.method](args.path, args.target)

    json.dump(
        {
            "method": args.method,
            "seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes(),
            "coef": coef,
            "r2": r2,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
