"""One classifier measurement in an isolated process, for the classification benchmark.

The timed workload is the whole journey a practitioner actually takes:
**CSV on disk → 80/20 train/test split → fitted classifier → held-out metrics.**
Reading and any scaling the method needs are inside the timed region, because
that is what training from a file costs; nobody gets handed a warm DataFrame.

Each method uses its own idiomatic path. Grizzly's function does the split and
evaluation internally; the sklearn methods use pandas/polars to read and
scikit-learn to split, fit, and score. The splits are equally sized but not
identical row-for-row -- on hundreds of thousands of rows the held-out metrics
agree to sampling noise, and the parent asserts exactly that, so a fast wrong
answer cannot pass as a fast answer.

Prints one JSON object on stdout: seconds, peak RSS, coefficients, and metrics.
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

# Grizzly's documented default, benchmarked as-is: what a user gets without
# tuning is the honest thing to time. It is also the right order of magnitude
# for this loss -- the logistic residual is bounded in [-1, 1], so a rate large
# enough to be useful for squared error takes per-sample steps comparable to
# the size of the optimum itself and the iterate oscillates instead of
# settling. sklearn's SGDClassifier likewise runs on its own default schedule.
LEARNING_RATE = 0.05


def peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


# ---------------------------------------------------------------------------
# grizzly
# ---------------------------------------------------------------------------


def fit_grizzly_logistic(path: str, target: str) -> tuple[float, list[float], dict[str, float]]:
    import grizzly

    start = time.perf_counter()
    result = grizzly.csv_logistic_regression(
        path,
        target=target,
        epochs=SGD_EPOCHS,
        learning_rate=LEARNING_RATE,
        train_frac=TRAIN_FRAC,
        seed=SEED,
        sample_size=100_000_000,
    )
    elapsed = time.perf_counter() - start
    metrics = {
        "accuracy": float(result["accuracy"]),
        "roc_auc": float(result["roc_auc"]),
        "log_loss": float(result["log_loss"]),
    }
    return elapsed, list(result["coef"]), metrics


def fit_grizzly_gnb(path: str, target: str) -> tuple[float, list[float], dict[str, float]]:
    import grizzly

    start = time.perf_counter()
    result = grizzly.csv_gaussian_nb(
        path,
        target=target,
        train_frac=TRAIN_FRAC,
        seed=SEED,
        sample_size=100_000_000,
    )
    elapsed = time.perf_counter() - start
    metrics = {
        "accuracy": float(result["accuracy"]),
        "roc_auc": float(result["roc_auc"]),
        "log_loss": float(result["log_loss"]),
    }
    # GNB has no coefficient vector; class-1 means fill the comparison slot.
    return elapsed, [float(v) for v in result["theta"][1]], metrics


# ---------------------------------------------------------------------------
# pandas / polars + scikit-learn
# ---------------------------------------------------------------------------


def _sklearn_metrics(model, X_test, y_test) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, proba >= 0.5)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
    }


def _sklearn_lbfgs(X, y) -> tuple[list[float], dict[str, float]]:
    """Unregularised logistic regression by L-BFGS, sklearn's default solver.

    Callers pre-import sklearn before starting their clocks, so these resolve
    from sys.modules in microseconds. Import cost is a per-process constant,
    not part of a fit, and grizzly's import is untimed too -- charging
    sklearn's several-hundred-millisecond import to some methods and not
    others is exactly the thumb on the scale this suite exists to avoid.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_FRAC, random_state=SEED
    )
    # C=inf is unregularised, matching grizzly's default l2=0. Regularising one
    # side and not the other would compare two different objectives.
    model = LogisticRegression(C=np.inf, max_iter=1000).fit(X_train, y_train)
    return [float(c) for c in model.coef_[0]], _sklearn_metrics(model, X_test, y_test)


def fit_pandas_sklearn(path: str, target: str) -> tuple[float, list[float], dict[str, float]]:
    import pandas as pd
    import sklearn.linear_model  # noqa: F401  (pre-import, see _sklearn_lbfgs)
    import sklearn.metrics  # noqa: F401
    import sklearn.model_selection  # noqa: F401

    start = time.perf_counter()
    df = pd.read_csv(path)
    y = df.pop(target).to_numpy()
    X = df.to_numpy()
    coef, metrics = _sklearn_lbfgs(X, y)
    elapsed = time.perf_counter() - start
    return elapsed, coef, metrics


def fit_polars_sklearn(path: str, target: str) -> tuple[float, list[float], dict[str, float]]:
    import polars as pl
    import sklearn.linear_model  # noqa: F401  (pre-import, see _sklearn_lbfgs)
    import sklearn.metrics  # noqa: F401
    import sklearn.model_selection  # noqa: F401

    start = time.perf_counter()
    df = pl.read_csv(path)
    y = df.get_column(target).to_numpy()
    X = df.drop(target).to_numpy()
    coef, metrics = _sklearn_lbfgs(X, y)
    elapsed = time.perf_counter() - start
    return elapsed, coef, metrics


def fit_pandas_sgd(path: str, target: str) -> tuple[float, list[float], dict[str, float]]:
    """The idiomatic sklearn SGD workflow, standardization included.

    Grizzly standardizes internally, so the sklearn pipeline gets a
    StandardScaler inside its timed region too -- leaving it out would compare
    a scaled fit against an unscaled one, and SGD without scaling is not a
    workflow anyone should be timed running. Coefficients are mapped back to
    the original feature space for the cross-method comparison.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import SGDClassifier
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
    # loss="log_loss" makes this logistic regression rather than a linear SVM,
    # so it is optimising the same objective as the other methods; without it
    # SGDClassifier defaults to hinge and predict_proba is unavailable.
    model = SGDClassifier(
        loss="log_loss",
        max_iter=SGD_EPOCHS,
        tol=None,
        random_state=SEED,
        penalty=None,
    ).fit(scaler.transform(X_train), y_train)
    metrics = _sklearn_metrics(model, scaler.transform(X_test), y_test)
    coef = np.asarray(model.coef_[0]) / scaler.scale_
    elapsed = time.perf_counter() - start
    return elapsed, [float(c) for c in coef], metrics


def _sklearn_gnb(X, y) -> tuple[list[float], dict[str, float]]:
    """scikit-learn GaussianNB; defaults match grizzly's (var_smoothing 1e-9)."""
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_FRAC, random_state=SEED
    )
    model = GaussianNB().fit(X_train, y_train)
    coef = [float(v) for v in model.theta_[1]]
    return coef, _sklearn_metrics(model, X_test, y_test)


def fit_pandas_gnb(path: str, target: str) -> tuple[float, list[float], dict[str, float]]:
    import pandas as pd
    import sklearn.metrics  # noqa: F401  (pre-import, see _sklearn_lbfgs)
    import sklearn.model_selection  # noqa: F401
    import sklearn.naive_bayes  # noqa: F401

    start = time.perf_counter()
    df = pd.read_csv(path)
    y = df.pop(target).to_numpy()
    X = df.to_numpy()
    coef, metrics = _sklearn_gnb(X, y)
    elapsed = time.perf_counter() - start
    return elapsed, coef, metrics


def fit_polars_gnb(path: str, target: str) -> tuple[float, list[float], dict[str, float]]:
    import polars as pl
    import sklearn.metrics  # noqa: F401  (pre-import, see _sklearn_lbfgs)
    import sklearn.model_selection  # noqa: F401
    import sklearn.naive_bayes  # noqa: F401

    start = time.perf_counter()
    df = pl.read_csv(path)
    y = df.get_column(target).to_numpy()
    X = df.drop(target).to_numpy()
    coef, metrics = _sklearn_gnb(X, y)
    elapsed = time.perf_counter() - start
    return elapsed, coef, metrics


METHODS = {
    "grizzly_logistic": fit_grizzly_logistic,
    "pandas_sklearn": fit_pandas_sklearn,
    "polars_sklearn": fit_polars_sklearn,
    "pandas_sgd": fit_pandas_sgd,
    "grizzly_gnb": fit_grizzly_gnb,
    "pandas_gnb": fit_pandas_gnb,
    "polars_gnb": fit_polars_gnb,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--path", required=True)
    parser.add_argument("--target", default="target")
    args = parser.parse_args()

    elapsed, coef, metrics = METHODS[args.method](args.path, args.target)

    json.dump(
        {
            "method": args.method,
            "seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes(),
            "coef": coef,
            **metrics,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
