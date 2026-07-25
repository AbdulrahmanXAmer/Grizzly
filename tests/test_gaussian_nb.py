"""Tests for the Gaussian Naive Bayes classifier.

GNB's fitted parameters are summary statistics -- class priors and per-(class,
feature) means and variances -- which makes it the most sharply testable model
in the package: unlike SGD there is no optimizer error, so parameters must
match scikit-learn to floating-point accuracy, not merely to a tolerance that
forgives slow convergence.

The differential tests use `shuffle=False`, under which grizzly's split is
exactly "first 80% train, rest test" -- trivially reproducible in Python, so
both sides fit on *identical* rows and every disagreement is arithmetic, not
sampling.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import random

import pytest

import grizzly

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

FULL_COVERAGE = 10_000_000

requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")


def write_gaussian_csv(path, n_rows=20_000, n_features=5, seed=0, shift=1.2):
    """Class-conditional Gaussian data: y ~ Bernoulli(0.5), x | y ~ N(mu_y, 1).

    This is exactly the generative model GNB assumes, so the fit should be
    near-optimal on it -- and any gap to scikit-learn is implementation, not
    model mismatch. `shift` controls class separation.
    """
    rng = random.Random(seed)
    mu = [[rng.uniform(-0.5, 0.5) for _ in range(n_features)] for _ in range(2)]
    for j in range(n_features):
        mu[1][j] += shift * (1 if rng.random() < 0.5 else -1) * rng.uniform(0.3, 1.0)

    header = [f"f_{i}" for i in range(n_features)] + ["target"]
    rows = []
    for _ in range(n_rows):
        y = 1 if rng.random() < 0.5 else 0
        x = [rng.gauss(mu[y][j], 1.0) for j in range(n_features)]
        rows.append([f"{v:.6f}" for v in x] + [y])
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def gaussian_csv(tmp_path):
    return write_gaussian_csv(tmp_path / "gaussian.csv")


# ---------------------------------------------------------------------------
# basic contract
# ---------------------------------------------------------------------------


def test_reports_every_documented_field(gaussian_csv):
    result = grizzly.csv_gaussian_nb(gaussian_csv, target="target", sample_size=FULL_COVERAGE)

    for key in (
        "features",
        "priors",
        "theta",
        "var",
        "class_counts",
        "train_n",
        "test_n",
        "accuracy",
        "log_loss",
        "roc_auc",
        "positive_rate",
        "confusion_matrix",
    ):
        assert key in result, f"missing {key}"

    p = len(result["features"])
    assert len(result["theta"]) == 2 and len(result["theta"][0]) == p
    assert len(result["var"]) == 2 and len(result["var"][1]) == p
    assert result["priors"][0] + result["priors"][1] == pytest.approx(1.0)
    assert result["class_counts"][0] + result["class_counts"][1] == result["train_n"]
    assert all(v > 0 for cls in result["var"] for v in cls)

    cm = result["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == result["test_n"]
    assert result["train_n"] + result["test_n"] == 20_000


def test_learns_the_separation(gaussian_csv):
    result = grizzly.csv_gaussian_nb(gaussian_csv, target="target", sample_size=FULL_COVERAGE)
    # The generating classes overlap, so perfection is impossible; the point is
    # to be far from the coin-flip this data's balanced classes would give.
    assert result["accuracy"] > 0.75
    assert result["roc_auc"] > 0.85
    assert result["log_loss"] < 0.5


def test_is_deterministic(gaussian_csv):
    first = grizzly.csv_gaussian_nb(gaussian_csv, target="target", seed=3, sample_size=FULL_COVERAGE)
    second = grizzly.csv_gaussian_nb(
        gaussian_csv, target="target", seed=3, sample_size=FULL_COVERAGE
    )
    assert first["theta"] == second["theta"]
    assert first["var"] == second["var"]
    assert first["log_loss"] == second["log_loss"]


def test_rejects_labels_outside_zero_and_one(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f_0", "target"])
        for i in range(100):
            writer.writerow([f"{i * 0.1:.3f}", 3 if i == 70 else 1])

    with pytest.raises(ValueError, match="only 0 and 1"):
        grizzly.csv_gaussian_nb(str(path), target="target", sample_size=FULL_COVERAGE)


def test_single_class_stays_finite(tmp_path):
    """A class never observed has prior 0; predictions must saturate, not NaN."""
    path = tmp_path / "one_class.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f_0", "f_1", "target"])
        rng = random.Random(5)
        for _ in range(400):
            writer.writerow([f"{rng.gauss(0, 1):.4f}", f"{rng.gauss(0, 1):.4f}", 0])

    result = grizzly.csv_gaussian_nb(str(path), target="target", sample_size=FULL_COVERAGE)
    assert result["accuracy"] == 1.0
    assert result["roc_auc"] == 0.5  # undefined with one class; 0.5, not NaN
    assert math.isfinite(result["log_loss"])
    assert result["priors"][1] == 0.0


def test_constant_feature_is_survivable(tmp_path):
    """Zero variance meets var_smoothing, not a division by zero."""
    path = tmp_path / "constant.csv"
    rng = random.Random(9)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["informative", "constant", "target"])
        for _ in range(2_000):
            y = 1 if rng.random() < 0.5 else 0
            writer.writerow([f"{rng.gauss(2.0 * y, 1.0):.4f}", "7.0", y])

    result = grizzly.csv_gaussian_nb(str(path), target="target", sample_size=FULL_COVERAGE)
    assert math.isfinite(result["log_loss"])
    assert result["accuracy"] > 0.75  # the informative feature still carries
    # The constant column's variance is exactly the smoothing epsilon.
    assert result["var"][0][1] > 0.0


# ---------------------------------------------------------------------------
# differential: scikit-learn
# ---------------------------------------------------------------------------


def load_xy(path, feature_names):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = [[float(r[n]) for n in feature_names] for r in rows]
    y = [float(r["target"]) for r in rows]
    return X, y


@requires_sklearn
def test_parameters_match_sklearn_on_identical_rows(gaussian_csv):
    """Priors, theta, and var against sklearn's GaussianNB on the same rows.

    No optimizer stands between the data and these parameters on either side,
    so the tolerances are floating-point-sized: a real formula difference --
    sample vs population variance, smoothing applied differently -- fails this
    by orders of magnitude.
    """
    import numpy as np
    from sklearn.naive_bayes import GaussianNB

    result = grizzly.csv_gaussian_nb(
        gaussian_csv, target="target", shuffle=False, sample_size=FULL_COVERAGE
    )
    X, y = load_xy(gaussian_csv, result["features"])
    cut = result["train_n"] + result["test_n"]
    train_cut = result["train_n"]
    X_train = np.asarray(X[:train_cut])
    y_train = np.asarray(y[:train_cut])
    assert cut == len(X)

    model = GaussianNB().fit(X_train, y_train)
    # sklearn orders classes_ ascending, so index 0 is class 0.
    assert list(model.classes_) == [0.0, 1.0]

    np.testing.assert_allclose(result["priors"], model.class_prior_, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["theta"], model.theta_, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result["var"], model.var_, rtol=1e-6, atol=1e-12)


@requires_sklearn
def test_metrics_match_sklearn_on_identical_rows(gaussian_csv):
    """Held-out metrics against sklearn scoring the same rows with its own fit.

    Same unshuffled split on both sides: train on the first 80%, score the
    rest. Only ROC-AUC carries a real tolerance, for the 4096-bin histogram.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.naive_bayes import GaussianNB

    result = grizzly.csv_gaussian_nb(
        gaussian_csv, target="target", shuffle=False, sample_size=FULL_COVERAGE
    )
    X, y = load_xy(gaussian_csv, result["features"])
    train_cut = result["train_n"]
    X_np, y_np = np.asarray(X), np.asarray(y)

    model = GaussianNB().fit(X_np[:train_cut], y_np[:train_cut])
    proba = model.predict_proba(X_np[train_cut:])[:, 1]
    y_test = y_np[train_cut:]

    assert result["accuracy"] == pytest.approx(accuracy_score(y_test, proba >= 0.5), abs=1e-12)
    assert result["log_loss"] == pytest.approx(log_loss(y_test, proba), rel=1e-6)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(y_test, proba), abs=1e-3)


@requires_sklearn
def test_agreement_survives_a_shuffled_split(gaussian_csv):
    """With shuffled (different) splits, metrics still agree to sampling noise."""
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    result = grizzly.csv_gaussian_nb(
        gaussian_csv, target="target", shuffle=True, seed=0, sample_size=FULL_COVERAGE
    )
    X, y = load_xy(gaussian_csv, result["features"])
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(X), np.asarray(y), train_size=0.8, random_state=0
    )
    model = GaussianNB().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    assert result["accuracy"] == pytest.approx(accuracy_score(y_test, proba >= 0.5), abs=0.02)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(y_test, proba), abs=0.02)
