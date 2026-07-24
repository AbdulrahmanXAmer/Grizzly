from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from .api import (
    csv_linear_regression,
    csv_minmax_params,
    csv_profile,
    csv_transform_minmax,
    detect_columns,
    detect_schema,
    info,
    is_native,
    native_module,
)
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
    "csv_linear_regression",
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
