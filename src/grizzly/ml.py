from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _require_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "grizzly.ml requires numpy. Install with `pip install grizzly[numpy]` or `pip install numpy`."
        ) from e


@dataclass(slots=True)
class LinearRegression:
    """
    Minimal sklearn-like Linear Regression (closed form).

    Solves: w = argmin ||y - Xb w||_2 using np.linalg.lstsq (stable).
    Where Xb = [X, 1] adds an intercept column.
    """

    fit_intercept: bool = True
    coef_: Optional["Any"] = None
    intercept_: float = 0.0

    def fit(self, X, y):
        np = _require_numpy()

        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64").reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have same number of rows as X")

        if self.fit_intercept:
            Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype="float64")], axis=1)
            w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
            self.coef_ = w[:-1]
            self.intercept_ = float(w[-1])
        else:
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.coef_ = w
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        np = _require_numpy()
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = np.asarray(X, dtype="float64")
        coef = np.asarray(self.coef_, dtype="float64")
        yhat = X @ coef
        if self.fit_intercept:
            yhat = yhat + self.intercept_
        return yhat

    def score(self, X, y) -> float:
        """
        R^2 score (matches sklearn for regression).
        """
        np = _require_numpy()
        y = np.asarray(y, dtype="float64").reshape(-1)
        yhat = np.asarray(self.predict(X), dtype="float64").reshape(-1)

        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)


@dataclass(slots=True)
class RidgeRegression:
    """
    Minimal sklearn-like Ridge Regression (L2 regularization).

    Closed form:
      w = (Xb^T Xb + alpha I)^-1 Xb^T y
    """

    alpha: float = 1.0
    fit_intercept: bool = True
    coef_: Optional["Any"] = None
    intercept_: float = 0.0

    def fit(self, X, y):
        np = _require_numpy()

        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64").reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have same number of rows as X")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")

        if self.fit_intercept:
            Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype="float64")], axis=1)
        else:
            Xb = X

        # Ridge: solve (A + alpha I) w = b
        A = Xb.T @ Xb
        b = Xb.T @ y
        I = np.eye(A.shape[0], dtype="float64")
        w = np.linalg.solve(A + self.alpha * I, b)

        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = float(w[-1])
        else:
            self.coef_ = w
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        # Same as linear regression
        return LinearRegression(fit_intercept=self.fit_intercept, coef_=self.coef_, intercept_=self.intercept_).predict(X)

    def score(self, X, y) -> float:
        return LinearRegression(fit_intercept=self.fit_intercept, coef_=self.coef_, intercept_=self.intercept_).score(X, y)


