from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _is_seq(x: Any) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))


def _path_like_to_str(x: Any) -> str | None:
    """Return `x` as a filesystem path, or None if it should be treated as data.

    A `Path` is always taken as a path: constructing one is an explicit
    statement of intent, and a missing file should raise rather than be
    silently reinterpreted.

    A `str` is only taken as a path when the file actually exists. Strings are
    ordinary data, and plenty of real values happen to end in a data-file
    extension -- a filename stored in a column, a URL, or just the text
    "report.csv". Treating those as paths made `detect_schema` raise
    FileNotFoundError from inside the parquet reader instead of describing a
    string. When the path does not resolve, the safe reading is "this is a
    string", which shows up plainly in the schema as a string column.
    """
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, str):
        try:
            if Path(x).is_file():
                return x
        except (OSError, ValueError):
            # Embedded null bytes, over-long names, and similar: not a path.
            return None
    return None


def normalize(data: Any, *, sample_size: int = 1000) -> Any:
    """
    Normalize common DS objects into a Python-native structure suitable for Grizzly inference.

    Strategy (best practice):
    - Convert tabular objects into **sampled list[dict]** records.
    - Keep already-Python-native dict/list structures as-is.
    - Avoid full materialization: only convert up to sample_size rows where possible.
    """
    # 1) Parquet path -> pyarrow.Table -> list[dict]
    p = _path_like_to_str(data)
    if p and (p.lower().endswith(".csv") or p.lower().endswith(".csv.gz")):
        # Prefer pandas for robust CSV parsing if available, else stdlib fallback.
        try:
            import pandas as pd  # type: ignore

            # Try delimiter inference first.
            df = pd.read_csv(p, nrows=sample_size, compression="infer", sep=None, engine="python")
            # If it still looks like a single-column whitespace-delimited file, retry.
            if df.shape[1] == 1:
                df2 = pd.read_csv(
                    p,
                    nrows=sample_size,
                    compression="infer",
                    sep=r"\s+",
                    engine="python",
                    header=None,
                )
                if df2.shape[1] > 1:
                    df2.columns = [f"col_{i}" for i in range(df2.shape[1])]
                    df = df2
            return df.to_dict(orient="records")
        except ImportError:
            import csv
            import gzip

            open_fn = gzip.open if p.lower().endswith(".gz") else open
            records: list[dict] = []
            with open_fn(p, mode="rt", newline="") as f:  # type: ignore[arg-type]
                # Peek first non-empty line to decide delimiter/header handling.
                first = ""
                pos = f.tell()
                while True:
                    first = f.readline()
                    if not first:
                        break
                    if first.strip():
                        break
                f.seek(pos)

                def coerce_scalar(s: str) -> Any:
                    ss = s.strip()
                    if ss == "":
                        return None
                    try:
                        if any(c in ss for c in (".", "e", "E")):
                            return float(ss)
                        return int(ss)
                    except Exception:
                        return ss

                # Choose delimiter: comma/tab/semicolon, else whitespace.
                delim = None
                for cand in (",", "\t", ";", "|"):
                    if cand in first:
                        delim = cand
                        break

                if delim is None:
                    # Whitespace-delimited, likely no header
                    # Skip any potential header if it contains alphabetic chars.
                    def has_alpha(line: str) -> bool:
                        return any(ch.isalpha() for ch in line)

                    # Consume first line and decide
                    line = f.readline()
                    if not line:
                        return []
                    tokens = line.strip().split()
                    if has_alpha(line):
                        cols = [t.strip() or f"col_{i}" for i, t in enumerate(tokens)]
                    else:
                        cols = [f"col_{i}" for i in range(len(tokens))]
                        records.append(
                            {cols[i]: coerce_scalar(tokens[i]) for i in range(len(tokens))}
                        )

                    for line in f:
                        if len(records) >= sample_size:
                            break
                        if not line.strip():
                            continue
                        toks = line.strip().split()
                        if not toks:
                            continue
                        # pad/truncate to expected width
                        toks = (toks + [""] * len(cols))[: len(cols)]
                        records.append({cols[j]: coerce_scalar(toks[j]) for j in range(len(cols))})
                    return records

                # Delimited CSV: try DictReader first; if no header, synthesize col_0..col_n.
                reader = csv.reader(f, delimiter=delim)
                try:
                    header = next(reader)
                except StopIteration:
                    return []
                header = [h.strip() for h in header]
                if all(h == "" for h in header) or all(
                    not any(ch.isalpha() for ch in h) for h in header
                ):
                    cols = [f"col_{i}" for i in range(len(header))]
                    # Treat first row as data
                    records.append({cols[i]: coerce_scalar(header[i]) for i in range(len(cols))})
                else:
                    cols = [h or f"col_{i}" for i, h in enumerate(header)]

                for row in reader:
                    if len(records) >= sample_size:
                        break
                    row = (row + [""] * len(cols))[: len(cols)]
                    records.append({cols[j]: coerce_scalar(row[j]) for j in range(len(cols))})
            return records

    if p and p.lower().endswith(".parquet"):
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(p)
            return _pyarrow_table_to_records(table, sample_size=sample_size)
        except ImportError:
            # If pyarrow isn't installed, just pass through and let fallback/Rust handle (or error).
            return data

    # 2) pandas DataFrame/Series
    try:
        import pandas as pd  # type: ignore

        if isinstance(data, pd.DataFrame):
            head = data.head(sample_size)
            return head.to_dict(orient="records")
        if isinstance(data, pd.Series):
            # treat as a single column named after the series (or "value")
            name = data.name if data.name is not None else "value"
            head = data.head(sample_size)
            return [{str(name): v} for v in head.tolist()]
    except ImportError:
        pass

    # 3) numpy ndarray / scalar
    try:
        import numpy as np  # type: ignore

        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                return data.item()
            if data.ndim == 1:
                return data[:sample_size].tolist()
            if data.ndim == 2:
                # represent as records with synthetic column names
                n = min(int(data.shape[0]), sample_size)
                arr = data[:n]
                cols = [f"col_{i}" for i in range(int(arr.shape[1]))]
                return [dict(zip(cols, row.tolist(), strict=True)) for row in arr]
            # higher dims: keep nested lists (Rust will produce [][][] paths)
            n = min(int(data.shape[0]), sample_size)
            return data[:n].tolist()
        if isinstance(data, np.generic):
            return data.item()
    except ImportError:
        pass

    # 4) pyarrow Table / RecordBatch / Array
    try:
        import pyarrow as pa  # type: ignore

        if isinstance(data, pa.Table):
            return _pyarrow_table_to_records(data, sample_size=sample_size)
        if isinstance(data, pa.RecordBatch):
            return _pyarrow_table_to_records(pa.Table.from_batches([data]), sample_size=sample_size)
        if isinstance(data, pa.Array):
            # to_pylist already returns Python scalars
            return data.slice(0, sample_size).to_pylist()
    except ImportError:
        pass

    # 5) Already Python-native
    if isinstance(data, Mapping):
        return data
    if _is_seq(data):
        return data

    # 6) Generic iterable (generator, iterator): do not consume here; Rust can sample it
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, bytearray)):
        return data

    return data


def _pyarrow_table_to_records(table: Any, *, sample_size: int) -> list[dict]:
    # pyarrow.Table supports .slice and .to_pylist()
    try:
        return table.slice(0, sample_size).to_pylist()
    except Exception:
        # fallback: best-effort conversion
        return list(table.to_pydict().items())  # type: ignore
