from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from . import drift
from .api import (
    csv_linear_regression,
    csv_minmax_params,
    csv_profile,
    csv_sgd_regression,
    csv_standardize_params,
    csv_transform_minmax,
    csv_transform_standardize,
    detect_columns,
    detect_schema,
    info,
    is_native,
    native_module,
)
from .drift import compare_profiles, detect_drift, load_reference, save_reference
from .grizzly import Grizzly, MinMaxScaler
from .ml import LinearRegression, RidgeRegression
from .normalize import normalize

try:
    __version__ = _version("grizzly")
except _PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "csv_minmax_params",
    "csv_profile",
    "csv_transform_minmax",
    "csv_standardize_params",
    "csv_transform_standardize",
    "csv_linear_regression",
    "csv_sgd_regression",
    "drift",
    "detect_drift",
    "compare_profiles",
    "save_reference",
    "load_reference",
    "detect_columns",
    "detect_schema",
    "info",
    "is_native",
    "native_module",
    "normalize",
    "Grizzly",
    "MinMaxScaler",
    "LinearRegression",
    "RidgeRegression",
]
