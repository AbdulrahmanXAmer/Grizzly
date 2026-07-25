"""Tests for model persistence: save, load, predict.

The invariant that matters is the full circle: fit -> save -> load -> predict
must land on exactly what the fit itself computed. A persistence layer that
round-trips the JSON but predicts something slightly different has quietly
forked the model, so the parity tests here are exact where the arithmetic
allows it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random

import pytest

import grizzly

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

FULL_COVERAGE = 10_000_000

requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def write_binary_csv(path, n_rows=6_000, n_features=4, seed=0):
    rng = random.Random(seed)
    weights = [rng.uniform(-1.5, 1.5) for _ in range(n_features)]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"f_{j}" for j in range(n_features)] + ["target"])
        for _ in range(n_rows):
            x = [rng.gauss(0, 1) for _ in range(n_features)]
            z = sum(w * v for w, v in zip(weights, x))
            writer.writerow([f"{v:.6f}" for v in x] + [1 if rng.random() < sigmoid(z) else 0])
    return str(path)


@pytest.fixture
def binary_csv(tmp_path):
    return write_binary_csv(tmp_path / "binary.csv")


def rows_of(path, feature_names):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = [[float(r[n]) for n in feature_names] for r in rows]
    y = [float(r["target"]) for r in rows]
    return X, y


# ---------------------------------------------------------------------------
# tagging and round-trip
# ---------------------------------------------------------------------------


def test_every_fit_is_tagged_with_its_model_kind(binary_csv):
    logistic = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=3, sample_size=FULL_COVERAGE
    )
    gnb = grizzly.csv_gaussian_nb(binary_csv, target="target", sample_size=FULL_COVERAGE)
    sgd = grizzly.csv_sgd_regression(
        binary_csv, target="f_0", epochs=2, sample_size=FULL_COVERAGE
    )
    exact = grizzly.csv_linear_regression(binary_csv, target="f_0", sample_size=FULL_COVERAGE)

    assert logistic["model"] == "logistic_regression"
    assert gnb["model"] == "gaussian_nb"
    assert sgd["model"] == "sgd_regression"
    assert exact["model"] == "linear_regression"


def test_round_trip_preserves_the_model(binary_csv, tmp_path):
    fit = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=5, sample_size=FULL_COVERAGE
    )
    path = grizzly.save_model(fit, tmp_path / "model.json")
    loaded = grizzly.load_model(path)

    assert loaded["model"] == fit["model"]
    assert loaded["coef"] == fit["coef"]
    assert loaded["intercept"] == fit["intercept"]
    # Provenance rides along: the metrics the model shipped with are part of
    # its paper trail, not discarded on save.
    assert loaded["accuracy"] == fit["accuracy"]

    # And the artifact is plain JSON a human can read without this library.
    raw = json.loads((tmp_path / "model.json").read_text())
    assert raw["schema_version"] == 1
    assert raw["model"] == "logistic_regression"


def test_save_refuses_a_non_model(tmp_path):
    with pytest.raises(ValueError, match="not a grizzly model"):
        grizzly.save_model({"accuracy": 0.9}, tmp_path / "nope.json")
    # A fit dict with its load-bearing keys stripped is refused too.
    with pytest.raises(ValueError, match="missing required keys"):
        grizzly.save_model({"model": "gaussian_nb", "features": ["a"]}, tmp_path / "nope.json")


def test_load_refuses_foreign_json(tmp_path):
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps({"model": "logistic_regression", "coef": [1.0]}))
    with pytest.raises(ValueError, match="schema_version"):
        grizzly.load_model(path)


def test_predict_validates_row_width(binary_csv):
    fit = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=3, sample_size=FULL_COVERAGE
    )
    with pytest.raises(ValueError, match="expects 4 features"):
        grizzly.predict(fit, [[1.0, 2.0]])


# ---------------------------------------------------------------------------
# predict parity: the loaded model computes what the fit computed
# ---------------------------------------------------------------------------


def test_logistic_predict_reproduces_the_fits_own_metrics(binary_csv, tmp_path):
    """Fit -> save -> load -> predict on the fit's own held-out rows must
    reproduce the accuracy the fit reported, exactly.

    shuffle=False makes the held-out rows the file's tail, so no split
    reconstruction is needed.
    """
    fit = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=20, shuffle=False, sample_size=FULL_COVERAGE
    )
    loaded = grizzly.load_model(grizzly.save_model(fit, tmp_path / "m.json"))

    X, y = rows_of(binary_csv, loaded["features"])
    scores = grizzly.predict(loaded, X[fit["train_n"] :])
    y_test = y[fit["train_n"] :]

    accuracy = sum((s >= 0.5) == (t > 0.5) for s, t in zip(scores, y_test)) / len(y_test)
    assert accuracy == pytest.approx(fit["accuracy"], abs=1e-12)


def test_gnb_predict_reproduces_the_fits_own_metrics(binary_csv, tmp_path):
    fit = grizzly.csv_gaussian_nb(
        binary_csv, target="target", shuffle=False, sample_size=FULL_COVERAGE
    )
    loaded = grizzly.load_model(grizzly.save_model(fit, tmp_path / "m.json"))

    X, y = rows_of(binary_csv, loaded["features"])
    scores = grizzly.predict(loaded, X[fit["train_n"] :])
    y_test = y[fit["train_n"] :]

    accuracy = sum((s >= 0.5) == (t > 0.5) for s, t in zip(scores, y_test)) / len(y_test)
    assert accuracy == pytest.approx(fit["accuracy"], abs=1e-12)


def test_regression_predict_is_the_linear_function(binary_csv, tmp_path):
    fit = grizzly.csv_linear_regression(binary_csv, target="f_0", sample_size=FULL_COVERAGE)
    loaded = grizzly.load_model(grizzly.save_model(fit, tmp_path / "m.json"))

    # target="f_0" leaves four features: f_1..f_3 plus the label column.
    rows = [[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 0.5, 1.0]]
    preds = grizzly.predict(loaded, rows)
    for row, pred in zip(rows, preds):
        expected = loaded["intercept"] + sum(c * v for c, v in zip(loaded["coef"], row))
        assert pred == pytest.approx(expected, abs=1e-15)


@requires_sklearn
def test_gnb_predict_matches_sklearn_predict_proba(binary_csv, tmp_path):
    """The saved GNB parameters, pushed through predict, give sklearn's
    probabilities -- because the parameters match and so does the formula."""
    import numpy as np
    from sklearn.naive_bayes import GaussianNB

    fit = grizzly.csv_gaussian_nb(
        binary_csv, target="target", shuffle=False, sample_size=FULL_COVERAGE
    )
    loaded = grizzly.load_model(grizzly.save_model(fit, tmp_path / "m.json"))

    X, y = rows_of(binary_csv, loaded["features"])
    train_n = fit["train_n"]
    model = GaussianNB().fit(np.asarray(X[:train_n]), np.asarray(y[:train_n]))

    ours = grizzly.predict(loaded, X[train_n:])
    theirs = model.predict_proba(np.asarray(X[train_n:]))[:, 1]
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-9)
