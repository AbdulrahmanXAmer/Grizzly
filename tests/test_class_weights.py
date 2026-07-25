"""Tests for class weighting (logistic) and prior override (Gaussian NB).

Class weights exist for one reason: on skewed labels an unweighted fit learns
to say "majority" and looks accurate doing it. So beyond agreeing with
scikit-learn, these tests assert the *effect* — balanced weighting must raise
minority-class recall relative to the unweighted fit on the same imbalanced
data. A weighting implementation that changes nothing would pass any pure
agreement test on easy data; it cannot pass that one.
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


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def write_imbalanced_csv(path, n_rows=30_000, n_features=5, seed=0, minority=0.1):
    """~10% positives, drawn from a shifted Gaussian so they are learnable."""
    rng = random.Random(seed)
    shift = [rng.uniform(0.8, 1.6) * (1 if rng.random() < 0.5 else -1) for _ in range(n_features)]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"f_{j}" for j in range(n_features)] + ["target"])
        for _ in range(n_rows):
            y = 1 if rng.random() < minority else 0
            x = [rng.gauss(shift[j] * y, 1.0) for j in range(n_features)]
            writer.writerow([f"{v:.6f}" for v in x] + [y])
    return str(path)


@pytest.fixture
def imbalanced_csv(tmp_path):
    return write_imbalanced_csv(tmp_path / "imbalanced.csv")


def recall(result):
    cm = result["confusion_matrix"]
    return cm["tp"] / max(cm["tp"] + cm["fn"], 1)


# ---------------------------------------------------------------------------
# logistic class weights
# ---------------------------------------------------------------------------


def test_balanced_raises_minority_recall(imbalanced_csv):
    """The reason the feature exists, asserted directly."""
    kwargs = dict(target="target", epochs=20, sample_size=FULL_COVERAGE)
    plain = grizzly.csv_logistic_regression(imbalanced_csv, **kwargs)
    balanced = grizzly.csv_logistic_regression(imbalanced_csv, class_weight="balanced", **kwargs)

    assert recall(balanced) > recall(plain) + 0.1, (
        f"balanced recall {recall(balanced):.3f} vs unweighted {recall(plain):.3f}"
    )
    # The usual price: some overall accuracy, paid knowingly.
    assert balanced["accuracy"] < plain["accuracy"]
    # Ranking quality is about equally good either way; weighting moves the
    # decision boundary, not the ordering.
    assert balanced["roc_auc"] == pytest.approx(plain["roc_auc"], abs=0.03)


def test_explicit_weights_move_the_same_direction(imbalanced_csv):
    kwargs = dict(target="target", epochs=20, sample_size=FULL_COVERAGE)
    plain = grizzly.csv_logistic_regression(imbalanced_csv, **kwargs)
    up = grizzly.csv_logistic_regression(imbalanced_csv, class_weight={0: 1.0, 1: 5.0}, **kwargs)
    assert recall(up) > recall(plain)

    # Dict and list forms are the same parameter.
    as_list = grizzly.csv_logistic_regression(imbalanced_csv, class_weight=[1.0, 5.0], **kwargs)
    assert as_list["coef"] == up["coef"]


def test_unit_weights_are_bitwise_neutral(imbalanced_csv):
    """[1, 1] must equal no weighting at all, to the last bit.

    Multiplying a gradient by 1.0 is exact, so any difference here means the
    weighted code path diverged from the unweighted one structurally.
    """
    kwargs = dict(target="target", epochs=8, sample_size=FULL_COVERAGE)
    plain = grizzly.csv_logistic_regression(imbalanced_csv, **kwargs)
    unit = grizzly.csv_logistic_regression(imbalanced_csv, class_weight=[1.0, 1.0], **kwargs)
    assert unit["coef"] == plain["coef"]
    assert unit["intercept"] == plain["intercept"]
    assert unit["log_loss"] == plain["log_loss"]


def test_weighted_caching_still_bit_identical(imbalanced_csv):
    """The cached/streamed invariant must survive weighting; the balanced
    counts in particular are computed by different code in the two paths."""
    kwargs = dict(
        target="target", epochs=6, class_weight="balanced", sample_size=FULL_COVERAGE
    )
    cached = grizzly.csv_logistic_regression(imbalanced_csv, cache_budget_mb=512, **kwargs)
    streamed = grizzly.csv_logistic_regression(imbalanced_csv, cache_budget_mb=0, **kwargs)
    assert cached["coef"] == streamed["coef"]
    assert cached["intercept"] == streamed["intercept"]
    assert cached["log_loss"] == streamed["log_loss"]


def test_class_weight_validation(imbalanced_csv):
    kwargs = dict(target="target", sample_size=FULL_COVERAGE)
    with pytest.raises(ValueError, match="balanced"):
        grizzly.csv_logistic_regression(imbalanced_csv, class_weight="typo", **kwargs)
    with pytest.raises(ValueError, match="positive finite"):
        grizzly.csv_logistic_regression(imbalanced_csv, class_weight=[1.0, -2.0], **kwargs)
    with pytest.raises(ValueError, match="positive finite"):
        grizzly.csv_logistic_regression(imbalanced_csv, class_weight=[1.0, 2.0, 3.0], **kwargs)
    with pytest.raises(ValueError, match="keys 0 and 1"):
        grizzly.csv_logistic_regression(imbalanced_csv, class_weight={1: 2.0}, **kwargs)


@requires_sklearn
def test_balanced_agrees_with_sklearn(imbalanced_csv):
    """Same rebalanced objective, compared on held-out metrics."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    result = grizzly.csv_logistic_regression(
        imbalanced_csv,
        target="target",
        epochs=40,
        class_weight="balanced",
        sample_size=FULL_COVERAGE,
    )

    with open(imbalanced_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.asarray([[float(r[n]) for n in result["features"]] for r in rows])
    y = np.asarray([float(r["target"]) for r in rows])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0, stratify=y
    )
    model = LogisticRegression(C=np.inf, class_weight="balanced", max_iter=2000).fit(
        X_train, y_train
    )
    proba = model.predict_proba(X_test)[:, 1]

    assert result["accuracy"] == pytest.approx(accuracy_score(y_test, proba >= 0.5), abs=0.03)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(y_test, proba), abs=0.02)

    # And the recall gain matches in kind: sklearn's balanced fit also trades
    # accuracy for minority recall; both implementations make the same trade.
    from sklearn.metrics import recall_score

    sk_recall = recall_score(y_test, proba >= 0.5)
    assert recall(result) == pytest.approx(sk_recall, abs=0.05)


# ---------------------------------------------------------------------------
# Gaussian NB priors
# ---------------------------------------------------------------------------


def test_priors_are_used_and_reported(imbalanced_csv):
    plain = grizzly.csv_gaussian_nb(imbalanced_csv, target="target", sample_size=FULL_COVERAGE)
    flat = grizzly.csv_gaussian_nb(
        imbalanced_csv, target="target", priors=[0.5, 0.5], sample_size=FULL_COVERAGE
    )
    assert flat["priors"] == [0.5, 0.5]
    # Likelihoods stay learned: only the prior belief changed.
    assert flat["theta"] == plain["theta"]
    assert flat["var"] == plain["var"]
    # Equal priors on 9:1 data push predictions toward the minority class.
    assert recall(flat) > recall(plain)
    assert flat["class_counts"] == plain["class_counts"]


def test_priors_validation(imbalanced_csv):
    kwargs = dict(target="target", sample_size=FULL_COVERAGE)
    with pytest.raises(ValueError, match="summing to 1"):
        grizzly.csv_gaussian_nb(imbalanced_csv, priors=[0.6, 0.6], **kwargs)
    with pytest.raises(ValueError, match="summing to 1"):
        grizzly.csv_gaussian_nb(imbalanced_csv, priors=[1.5, -0.5], **kwargs)
    with pytest.raises(ValueError, match="summing to 1"):
        grizzly.csv_gaussian_nb(imbalanced_csv, priors=[1.0], **kwargs)


@requires_sklearn
def test_priors_match_sklearn_exactly(imbalanced_csv):
    """Identical rows, identical priors, no optimizer: near-exact agreement."""
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.naive_bayes import GaussianNB

    result = grizzly.csv_gaussian_nb(
        imbalanced_csv,
        target="target",
        priors=[0.5, 0.5],
        shuffle=False,
        sample_size=FULL_COVERAGE,
    )

    with open(imbalanced_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.asarray([[float(r[n]) for n in result["features"]] for r in rows])
    y = np.asarray([float(r["target"]) for r in rows])
    train_n = result["train_n"]

    model = GaussianNB(priors=[0.5, 0.5]).fit(X[:train_n], y[:train_n])
    proba = model.predict_proba(X[train_n:])[:, 1]
    y_test = y[train_n:]

    np.testing.assert_allclose(result["theta"], model.theta_, rtol=1e-9, atol=1e-12)
    assert result["accuracy"] == pytest.approx(
        accuracy_score(y_test, proba >= 0.5), abs=1e-12
    )
    assert result["log_loss"] == pytest.approx(log_loss(y_test, proba), rel=1e-6)
