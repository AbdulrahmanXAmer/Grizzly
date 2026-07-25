"""Persist fitted models and apply them later.

Every fit in this package returns a plain dict tagged with a ``model`` key, so
persistence is deliberately boring: JSON with a schema version, no pickle, no
custom binary format. A saved model is inspectable with `cat`, diffable in a
pull request, and loadable a decade from now without this library — which is
what you want from an artifact that decides things in production.

`predict` applies a loaded (or freshly fitted) model to rows in pure Python.
It exists so a saved model is *usable*, not just storable; for bulk scoring of
a CSV, apply it and feed the scores to `csv_classification_metrics`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# What predict() needs per model kind. Extra keys (metrics, provenance) are
# saved as-is — they are the model's paper trail — but only these are load-
# bearing, and a file missing one is rejected at load rather than at predict.
_REQUIRED_KEYS = {
    "linear_regression": ("features", "coef", "intercept"),
    "sgd_regression": ("features", "coef", "intercept"),
    "logistic_regression": ("features", "coef", "intercept"),
    "gaussian_nb": ("features", "priors", "theta", "var"),
}


def _validate(model: dict[str, Any]) -> str:
    kind = model.get("model")
    if kind not in _REQUIRED_KEYS:
        known = ", ".join(sorted(_REQUIRED_KEYS))
        raise ValueError(
            f"not a grizzly model: 'model' key is {kind!r}, expected one of {known}. "
            "Pass the dict returned by a csv_*_regression / csv_gaussian_nb fit."
        )
    missing = [k for k in _REQUIRED_KEYS[kind] if k not in model]
    if missing:
        raise ValueError(f"{kind} model is missing required keys: {missing}")
    return kind


def save_model(model: dict[str, Any], path: str | Path) -> Path:
    """Write a fitted model to JSON.

    Accepts exactly what the fit functions return. Validation happens here,
    not at load: a refused save is a nuisance, a corrupt artifact discovered
    at 3am is an incident.
    """
    _validate(model)
    path = Path(path)
    payload = {"schema_version": SCHEMA_VERSION, **model}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_model(path: str | Path) -> dict[str, Any]:
    """Read a model saved by :func:`save_model`, validating it is one."""
    payload = json.loads(Path(path).read_text())
    version = payload.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported model schema_version {version!r} (this build reads "
            f"{SCHEMA_VERSION}); was this file written by save_model?"
        )
    _validate(payload)
    return payload


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def predict(model: dict[str, Any], rows: Sequence[Sequence[float]]) -> list[float]:
    """Apply a fitted model to feature rows.

    ``rows`` are sequences of floats ordered as ``model["features"]`` — the
    same order the fit reported. Returns one float per row: the predicted
    value for regressors, the probability of class 1 for classifiers (use
    ``>= 0.5`` for hard labels).

    Pure Python on purpose: prediction from p coefficients is O(p) arithmetic,
    and keeping it dependency-free means a saved model runs anywhere the JSON
    can be read. The arithmetic mirrors the Rust evaluation exactly — linear
    models score `intercept + coef·x`, Gaussian NB the same stable
    log-likelihood difference through the same sigmoid.
    """
    kind = _validate(model)
    p = len(model["features"])
    for i, row in enumerate(rows):
        if len(row) != p:
            raise ValueError(f"row {i} has {len(row)} values; model expects {p} features")

    if kind in ("linear_regression", "sgd_regression", "logistic_regression"):
        coef = model["coef"]
        intercept = model["intercept"]
        raw = [intercept + sum(c * v for c, v in zip(coef, row)) for row in rows]
        if kind == "logistic_regression":
            return [_sigmoid(z) for z in raw]
        return raw

    # gaussian_nb: log P(c) - 0.5 sum_j [ln(2 pi var) + (x - theta)^2 / var],
    # compared between classes through the sigmoid of the difference.
    priors, theta, var = model["priors"], model["theta"], model["var"]
    consts = []
    inv2var = []
    for cls in range(2):
        s = sum(math.log(2.0 * math.pi * v) for v in var[cls])
        prior = priors[cls]
        consts.append((math.log(prior) if prior > 0 else -math.inf) - 0.5 * s)
        inv2var.append([0.5 / v for v in var[cls]])

    out = []
    for row in rows:
        ll = [0.0, 0.0]
        for cls in range(2):
            t, iv = theta[cls], inv2var[cls]
            ll[cls] = sum((v - t[j]) * (v - t[j]) * iv[j] for j, v in enumerate(row))
        out.append(_sigmoid((consts[1] - ll[1]) - (consts[0] - ll[0])))
    return out
