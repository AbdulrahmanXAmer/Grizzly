from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple, Union

from .api import (
    csv_minmax_params,
    csv_profile,
    csv_transform_minmax,
    csv_linear_regression,
    detect_columns,
    detect_schema,
    is_native,
)


def _pretty_dtype(inferred: str) -> str:
    # Keep this small + stable; we can expand later.
    return {
        "int": "int",
        "float": "float",
        "bool": "bool",
        "string": "string",
        "bytes": "bytes",
        "null": "null",
        "mixed": "mixed",
        "unknown": "unknown",
    }.get(inferred, inferred)


@dataclass(slots=True)
class MinMaxScaler:
    """
    Simple transformer object (sklearn-ish):
      scaler = g.fit_minmax()
      scaler.transform("out.csv")
    """

    input_path: str
    params: Dict[str, Dict[str, float]]
    delimiter: str | None
    has_header: bool | None

    def transform(self, output_path: str) -> Dict[str, Any]:
        return csv_transform_minmax(
            self.input_path,
            output_path,
            self.params,
            delimiter=self.delimiter,
            has_header=self.has_header,
        )


@dataclass(slots=True)
class Grizzly:
    """
    Convenience object for repeatedly inspecting the same dataset.

    Example:
        g = grizzly.Grizzly("data.csv.gz", sample_size=1000)
        g.info()
        schema = g.schema()
    """

    data: Any
    sample_size: int = 1000
    max_examples: int = 5
    normalize_input: bool = True
    fast_csv: bool = True
    # Cache keyed by a richer signature to prevent accidental reuse across config changes.
    # (path is per-object, but sample_size/max_examples/fast_csv can vary per instance)
    _csv_profile_cache: Dict[Tuple[int, int, bool, bool, bool, bool, Optional[Tuple[str, ...]]], Dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _selected_cols: Optional[Tuple[str, ...]] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile(
        self,
        *,
        lite: bool,
        track_freq: bool,
        collect_examples: bool,
    ) -> Dict[str, Any]:
        """
        Internal cached profile snapshot.

        Cache key is per-object (path + sample_size + max_examples + fast_csv are fixed on the object),
        plus the toggles that affect the underlying Rust work.
        """
        if not isinstance(self.data, (str,)):
            raise TypeError("Grizzly profile is only available for filesystem paths (str).")

        selected_sig = self._selected_cols
        key = (self.sample_size, self.max_examples, self.fast_csv, lite, track_freq, collect_examples, selected_sig)
        if key not in self._csv_profile_cache:
            prof = csv_profile(
                self.data,
                sample_size=self.sample_size,
                max_examples=self.max_examples,
                fast_csv=self.fast_csv,
                lite=lite,
                track_freq=track_freq,
                collect_examples=collect_examples,
            )
            # If we are a selected view, filter columns before caching to make repeated calls cheaper
            if selected_sig:
                prof = dict(prof)
                prof["columns"] = self._filter_cols(list(prof.get("columns", [])))
                prof["num_columns"] = len(prof["columns"])
            self._csv_profile_cache[key] = prof
        return self._csv_profile_cache[key]

    def _filter_cols(self, cols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._selected_cols:
            return cols
        wanted = set(self._selected_cols)
        return [c for c in cols if str(c.get("name")) in wanted]

    def select(self, cols: Iterable[str]) -> "Grizzly":
        """
        DataFrame-like projection. Returns a new Grizzly view scoped to selected columns.

        Example:
            g.select(["col_0", "col_3"]).describe()
        """
        g2 = Grizzly(
            self.data,
            sample_size=self.sample_size,
            max_examples=self.max_examples,
            normalize_input=self.normalize_input,
            fast_csv=self.fast_csv,
        )
        # Share profile cache to avoid rescanning
        g2._csv_profile_cache = self._csv_profile_cache
        g2._selected_cols = tuple(str(c) for c in cols)
        return g2

    def schema(self) -> Dict[str, Any]:
        return detect_schema(
            self.data,
            sample_size=self.sample_size,
            max_examples=self.max_examples,
            normalize_input=self.normalize_input,
        )

    def columns(self) -> List[str]:
        # If this is a CSV path, return CSV column names (respects selection)
        if isinstance(self.data, str) and (self.data.endswith(".csv") or self.data.endswith(".csv.gz")):
            prof = self.csv_profile(lite=True, track_freq=False, collect_examples=False)
            cols = [str(c.get("name")) for c in prof.get("columns", [])]
            return cols
        return detect_columns(self.data, sample_size=self.sample_size)

    # ------------------------------------------------------------------
    # Phase-1: CSV-backed convenience accessors (fast, cache-friendly)
    # ------------------------------------------------------------------

    def shape_sampled(self, *, lite: bool = True) -> tuple[int, int]:
        """
        Return (rows_sampled, num_columns) from the cached CSV profile.
        This is NOT the full dataset shape unless sample_size >= file rows.
        """
        prof = self.csv_profile(lite=lite, track_freq=not lite, collect_examples=not lite)
        cols = self._filter_cols(list(prof.get("columns", [])))
        return int(prof.get("rows_sampled", 0)), len(cols)

    def file_size_bytes(self) -> int | None:
        if not isinstance(self.data, (str,)):
            return None
        p = Path(self.data)
        return p.stat().st_size if p.exists() else None

    def dtypes(self, *, lite: bool = False) -> Dict[str, str]:
        """
        Return inferred dtype per column (from profile).
        """
        prof = self.csv_profile(lite=lite, track_freq=not lite, collect_examples=not lite)
        out: Dict[str, str] = {}
        for c in self._filter_cols(list(prof.get("columns", []))):
            out[str(c.get("name"))] = _pretty_dtype(str(c.get("inferred", "unknown")))
        return out

    def null_counts(self, *, lite: bool = True) -> Dict[str, int]:
        prof = self.csv_profile(lite=lite, track_freq=not lite, collect_examples=not lite)
        cols = self._filter_cols(list(prof.get("columns", [])))
        return {str(c.get("name")): int(c.get("null_count", 0) or 0) for c in cols}

    def null_pct(self, *, lite: bool = True) -> Dict[str, float]:
        prof = self.csv_profile(lite=lite, track_freq=not lite, collect_examples=not lite)
        out: Dict[str, float] = {}
        for c in self._filter_cols(list(prof.get("columns", []))):
            name = str(c.get("name"))
            count = float(c.get("count", 0) or 0)
            nulls = float(c.get("null_count", 0) or 0)
            out[name] = (nulls / count) if count else 0.0
        return out

    def missingness(self, *, lite: bool = True) -> List[Dict[str, Any]]:
        """
        Per-column missingness table (list of dicts), sorted by null_pct desc.
        """
        rep = self.eda(lite=lite, return_json=True)
        assert rep is not None
        miss = list(rep.get("missing", []))
        miss.sort(key=lambda x: float(x.get("null_pct", 0.0)), reverse=True)
        return miss

    def describe(self, *, lite: bool = False) -> Dict[str, Any]:
        """
        Describe++ style summary. Returns:
          { "numeric": [...], "categorical": [...], "dataset": {...} }
        """
        rep = self.eda(lite=lite, return_json=True)
        assert rep is not None
        return {
            "dataset": rep.get("dataset", {}),
            "numeric": rep.get("numeric", []),
            "categorical": rep.get("categorical", []),
        }

    def info(
        self,
        *,
        file: Optional[TextIO] = None,
        show_examples: bool = False,
        max_cols: Optional[int] = None,
    ) -> None:
        """
        Print a compact summary (similar in spirit to pandas.DataFrame.info()).
        """
        import sys

        out = file if file is not None else sys.stdout
        # If this looks like a CSV path, prefer the native CSV profile for correctness
        if isinstance(self.data, str) and (self.data.endswith(".csv") or self.data.endswith(".csv.gz")):
            prof = self.csv_profile(lite=False, track_freq=False, collect_examples=True)
            cols = prof.get("columns", [])
            sample_size = int(prof.get("rows_sampled", self.sample_size))
            backend = "rust" if is_native() else "python-fallback"
            print(f"Grizzly info: {len(cols)} columns (sample_size={sample_size}, backend={backend})", file=out)
        else:
            schema = self.schema()
            cols = schema.get("columns", [])
            sample_size = schema.get("sample_size", self.sample_size)
            backend = "rust" if is_native() else "python-fallback"
            print(f"Grizzly info: {len(cols)} columns (sample_size={sample_size}, backend={backend})", file=out)

        if max_cols is not None:
            cols = cols[:max_cols]

        print("", file=out)

        # Header
        if show_examples:
            print(f"{'#':>3}  {'column':<30}  {'non-null':>8}  {'dtype':<10}  examples", file=out)
            print(f"{'-'*3}  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*30}", file=out)
        else:
            print(f"{'#':>3}  {'column':<30}  {'non-null':>8}  {'dtype':<10}", file=out)
            print(f"{'-'*3}  {'-'*30}  {'-'*8}  {'-'*10}", file=out)

        for i, c in enumerate(cols):
            # schema uses "path"; csv_profile uses "name"
            path = str(c.get("path", c.get("name", "")))
            count = int(c.get("count", 0) or 0)
            nulls = int(c.get("null_count", 0) or 0)
            non_null = max(0, count - nulls)
            dtype = _pretty_dtype(str(c.get("inferred", "unknown")))

            if show_examples:
                examples = c.get("examples", [])
                ex = ", ".join(str(x) for x in examples[: self.max_examples])
                print(f"{i:>3}  {path:<30.30}  {non_null:>8}  {dtype:<10}  {ex}", file=out)
            else:
                print(f"{i:>3}  {path:<30.30}  {non_null:>8}  {dtype:<10}", file=out)

    def csv_profile(
        self,
        *,
        lite: bool = False,
        track_freq: bool = True,
        collect_examples: bool = True,
    ) -> Dict[str, Any]:
        """
        Profile a CSV/CSV.GZ path using the Rust-native csv profiler.
        """
        # _profile already accounts for selected_cols in its cache key and returns filtered columns
        return self._profile(lite=lite, track_freq=track_freq, collect_examples=collect_examples)

    def to_numpy(
        self,
        *,
        sampled: bool = True,
        dtype: str = "float32",
        target: str | None = None,
        features: List[str] | None = None,
    ) -> Union["Any", Tuple["Any", "Any"]]:
        """
        Convert (sampled) CSV data to a NumPy array.

        - Respects `select()` (only selected columns become features)
        - If target is provided, returns (X, y)
        - If features is provided, it defines feature columns explicitly (target excluded automatically)
        - Requires numpy installed
        """
        if not isinstance(self.data, (str,)):
            raise TypeError("Grizzly.to_numpy() currently expects a filesystem path (str).")
        try:
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("to_numpy requires numpy. Install with `pip install grizzly[numpy]` or `pip install numpy`.") from e

        prof = self.csv_profile(lite=True, track_freq=False, collect_examples=False)
        cols = list(prof.get("columns", []))
        col_names = [str(c.get("name")) for c in cols]
        name_to_idx = {n: i for i, n in enumerate(col_names)}

        if target is not None and target not in name_to_idx:
            raise KeyError(f"target column not found: {target!r}. Available: {col_names}")

        # Determine which columns to load as features
        if features is not None:
            feature_names = [str(c) for c in features]
        else:
            # Default: all columns or selected view columns
            feature_names = col_names if self._selected_cols is None else list(self._selected_cols)
        # Always exclude target from X
        if target is not None:
            feature_names = [c for c in feature_names if c != target]

        # Validate feature names exist
        missing = [c for c in feature_names if c not in name_to_idx]
        if missing:
            raise KeyError(f"feature column(s) not found: {missing}. Available: {col_names}")

        feature_idx = [name_to_idx[n] for n in feature_names]
        target_idx = name_to_idx[target] if target is not None else None

        # Determine split mode from profile delimiter
        delim = prof.get("delimiter")
        whitespace = (delim == "whitespace" or delim is None)
        delim_char = None if whitespace else str(delim)

        # How many rows to read
        max_rows = int(prof.get("rows_sampled", 0)) if sampled else 10**18

        rows_x: List[List[float]] = []
        rows_y: List[float] = []

        import gzip

        path = Path(self.data)
        opener = gzip.open if path.suffix == ".gz" else open

        with opener(path, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                if len(rows_x) >= max_rows:
                    break

                parts = line.split() if whitespace else line.rstrip("\n").split(delim_char)  # type: ignore[arg-type]
                if len(parts) < len(col_names):
                    # skip malformed line
                    continue

                # parse features
                fx: List[float] = []
                for i in feature_idx:
                    try:
                        fx.append(float(parts[i]))
                    except Exception:
                        fx.append(float("nan"))
                rows_x.append(fx)

                # parse target
                if target_idx is not None:
                    try:
                        rows_y.append(float(parts[target_idx]))
                    except Exception:
                        rows_y.append(float("nan"))

        X = np.asarray(rows_x, dtype=dtype)
        if target_idx is None:
            return X
        y = np.asarray(rows_y, dtype=dtype)
        return X, y

    def csv_minmax_params(self) -> Dict[str, Any]:
        """
        Compute min/max parameters for numeric CSV columns (sampled) for min-max scaling.
        """
        if not isinstance(self.data, (str,)):
            raise TypeError("Grizzly.csv_minmax_params() currently expects a filesystem path (str).")
        return csv_minmax_params(self.data, sample_size=self.sample_size)

    def transform_minmax(
        self,
        output_path: str,
        *,
        delimiter: str | None = None,
        has_header: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Apply min-max scaling to numeric columns and write a new file.

        - Uses the cached profile to build params (no extra profiling scan)
        - Passes header detection through to the Rust transform for correctness
        """
        if not isinstance(self.data, (str,)):
            raise TypeError("Grizzly.transform_minmax() currently expects a filesystem path (str).")

        prof = self.csv_profile()
        cols = self._filter_cols(list(prof.get("columns", [])))

        params: Dict[str, Dict[str, float]] = {}
        for c in cols:
            name = c.get("name")
            mn = c.get("min")
            mx = c.get("max")
            if name is not None and mn is not None and mx is not None:
                params[str(name)] = {"min": float(mn), "max": float(mx)}

        header_flag = prof.get("has_header") if has_header is None else has_header

        return csv_transform_minmax(
            str(self.data),
            output_path,
            params,
            delimiter=delimiter,
            has_header=bool(header_flag) if header_flag is not None else None,
        )

    def fit_minmax(
        self,
        *,
        delimiter: str | None = None,
        has_header: bool | None = None,
    ) -> MinMaxScaler:
        """
        Fit a min-max scaler using the cached profile (no extra scan).
        """
        if not isinstance(self.data, (str,)):
            raise TypeError("Grizzly.fit_minmax() currently expects a filesystem path (str).")

        prof = self.csv_profile()
        params: Dict[str, Dict[str, float]] = {}
        for c in self._filter_cols(list(prof.get("columns", []))):
            name = c.get("name")
            mn = c.get("min")
            mx = c.get("max")
            if name is not None and mn is not None and mx is not None:
                params[str(name)] = {"min": float(mn), "max": float(mx)}

        header_flag = prof.get("has_header") if has_header is None else has_header

        return MinMaxScaler(
            input_path=str(self.data),
            params=params,
            delimiter=delimiter,
            has_header=bool(header_flag) if header_flag is not None else None,
        )

    def fit_linear_regression(
        self,
        *,
        target: str,
        features: List[str] | None = None,
        train_frac: float = 0.8,
        seed: int = 0,
        sample_size: int | None = None,
        delimiter: str | None = None,
        has_header: bool | None = None,
        shuffle: bool = True,
        ridge_lambda: float = 0.0,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Rust-native linear regression directly from the CSV (no numpy required).
        Returns a stable schema dict with r2/coef/intercept/train_n/test_n.
        """
        if not isinstance(self.data, (str,)):
            raise TypeError("Grizzly.fit_linear_regression() currently expects a filesystem path (str).")

        # Default feature set: respect select() if present, else all non-target columns.
        if features is None:
            cols = self.columns()
            feats = cols if self._selected_cols is None else list(self._selected_cols)
            features = [c for c in feats if c != target]

        prof = self.csv_profile(lite=True, track_freq=False, collect_examples=False)
        header_flag = prof.get("has_header") if has_header is None else has_header

        return csv_linear_regression(
            str(self.data),
            target=target,
            features=features,
            train_frac=train_frac,
            seed=int(seed),
            sample_size=int(sample_size if sample_size is not None else self.sample_size),
            delimiter=delimiter,
            has_header=bool(header_flag) if header_flag is not None else None,
            fast_csv=self.fast_csv,
            shuffle=shuffle,
            ridge_lambda=ridge_lambda,
            return_debug=return_debug,
        )

    def eda(
        self,
        *,
        lite: bool = False,
        max_cols: int | None = None,
        return_json: bool = True,
        file: Optional[TextIO] = None,
    ) -> Dict[str, Any] | None:
        """
        Phase-1 EDA (fast + practical):
        - Dataset-level summary
        - Missingness (per column)
        - Univariate numeric stats (describe++)
        - Basic categorical summary (mode + examples, when available)

        If return_json=False, prints a compact report and returns None.
        """
        if not isinstance(self.data, (str,)):
            raise TypeError("Grizzly.eda() currently expects a filesystem path (str).")

        # In lite mode we skip expensive parts (freq/examples) for speed.
        prof = self.csv_profile(
            lite=lite,
            track_freq=not lite,
            collect_examples=not lite,
        )

        cols = self._filter_cols(list(prof.get("columns", [])))
        if max_cols is not None:
            cols = cols[:max_cols]

        path = Path(str(self.data))
        file_size = path.stat().st_size if path.exists() else None

        dataset = {
            "path": str(self.data),
            "file_size_bytes": file_size,
            "rows_sampled": int(prof.get("rows_sampled", 0)),
            "num_columns": len(cols),
            "delimiter": prof.get("delimiter"),
            "has_header": bool(prof.get("has_header", False)),
            "backend": "rust" if is_native() else "python-fallback",
            "mode": "lite" if lite else "full",
            "selected_cols": list(self._selected_cols) if self._selected_cols else None,
        }

        missing = []
        numeric = []
        categorical = []

        for c in cols:
            name = c.get("name")
            count = int(c.get("count", 0) or 0)
            nulls = int(c.get("null_count", 0) or 0)
            non_null = max(0, count - nulls)
            null_pct = (nulls / count) if count else 0.0

            inferred = _pretty_dtype(str(c.get("inferred", "unknown")))
            types = list(c.get("types", [])) if isinstance(c.get("types", []), list) else []

            missing.append(
                {
                    "name": name,
                    "count": count,
                    "null_count": nulls,
                    "non_null_count": non_null,
                    "null_pct": null_pct,
                }
            )

            if c.get("min") is not None and c.get("max") is not None:
                mn = float(c["min"])
                mx = float(c["max"])
                numeric.append(
                    {
                        "name": name,
                        "dtype": inferred,
                        "count": count,
                        "null_count": nulls,
                        "non_null_count": non_null,
                        "min": mn,
                        "max": mx,
                        "range": mx - mn,
                        "mean": c.get("mean"),
                        "std": c.get("std"),
                        "median": c.get("median"),
                        "p25": c.get("p25"),
                        "p75": c.get("p75"),
                        "p90": c.get("p90"),
                        "p95": c.get("p95"),
                        # Only include mode in full mode (lite skips freq tracking)
                        **(
                            {"mode": c.get("mode"), "mode_count": c.get("mode_count")}
                            if not lite
                            else {}
                        ),
                    }
                )
            else:
                categorical.append(
                    {
                        "name": name,
                        "dtype": inferred,
                        "count": count,
                        "null_count": nulls,
                        "non_null_count": non_null,
                        "types": types,
                        **(
                            {"mode": c.get("mode"), "mode_count": c.get("mode_count"), "examples": c.get("examples", [])}
                            if not lite
                            else {}
                        ),
                    }
                )

        report: Dict[str, Any] = {
            "schema_version": "eda.v1",
            "dataset": dataset,
            "missing": missing,
            "numeric": numeric,
            "categorical": categorical,
        }

        if return_json:
            return report

        # Print a compact report
        import sys

        out = file if file is not None else sys.stdout
        print(f"Grizzly EDA ({dataset['mode']}): {dataset['num_columns']} cols, rows_sampled={dataset['rows_sampled']}", file=out)
        print(f"  path={dataset['path']}", file=out)
        print(f"  delimiter={dataset['delimiter']}, has_header={dataset['has_header']}, backend={dataset['backend']}", file=out)
        if dataset["file_size_bytes"] is not None:
            mb = dataset["file_size_bytes"] / (1024 * 1024)
            print(f"  file_size={mb:.2f} MB", file=out)
        print("", file=out)

        # Missingness top offenders
        miss_sorted = sorted(missing, key=lambda x: x["null_pct"], reverse=True)
        print("Missingness (top 10 by null_pct):", file=out)
        for m in miss_sorted[:10]:
            print(f"  {m['name']}: null_pct={m['null_pct']:.2%} ({m['null_count']}/{m['count']})", file=out)
        print("", file=out)

        print("Numeric columns:", file=out)
        for n in numeric[: min(10, len(numeric))]:
            print(f"  {n['name']}: min={n['min']}, p50={n['median']}, max={n['max']}, std={n['std']}", file=out)
        if len(numeric) > 10:
            print(f"  ... +{len(numeric) - 10} more", file=out)
        print("", file=out)

        print("Categorical/non-numeric columns:", file=out)
        for s in categorical[: min(10, len(categorical))]:
            if not lite:
                print(f"  {s['name']}: mode={s.get('mode')} (count={s.get('mode_count')})", file=out)
            else:
                print(f"  {s['name']}: dtype={s.get('dtype')}", file=out)
        if len(categorical) > 10:
            print(f"  ... +{len(categorical) - 10} more", file=out)
        print("", file=out)

        return None


