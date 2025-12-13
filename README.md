<div align="center">
  <h2>Grizzly</h2>
  <p><strong>Rust-powered data profiling and schema inference for Python.</strong></p>
  <p>
    <a href="#quickstart">Quickstart</a> ·
    <a href="#ml-regression-workflows">ML</a> ·
    <a href="#performance-local-benchmark-results">Performance</a> ·
    <a href="#generate-synthetic-csvs-quick--repeatable">Synthetic data</a> ·
    <a href="#troubleshooting">Troubleshooting</a>
  </p>
</div>

<hr />

Grizzly is a Python package with a Rust (PyO3) core for:

- **Schema / column detection** from arbitrary Python data (nested dicts/lists/iterables)
- **CSV/CSV.GZ profiling** (delimiter/header sniffing, dtypes, missingness, numeric stats, percentiles, simple outliers)
- **Fast CSV transforms** (min-max scaling)
- **Rust-native linear regression from CSV** (no NumPy required)

<details>
  <summary><strong>Path notation (schema inference)</strong></summary>
  <br />
  <ul>
    <li>Dict keys join with <code>.</code>: <code>user.name</code></li>
    <li>Arrays add <code>[]</code>: <code>items[].id</code>, <code>matrix[][].value</code></li>
  </ul>
</details>

### Requirements

- **Python**: `>=3.14` (see `pyproject.toml`)
- **Rust** toolchain (for the native extension): `rustup` recommended

### Install (dev build)

From the `grizzly/` directory:

```bash
python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install maturin pytest

# Build + install the native extension into the active venv
maturin develop --release

# Quick sanity
python -c "import grizzly; print('native:', grizzly.is_native())"
pytest -q
```

Optional extras (normalization helpers):

```bash
python -m pip install ".[all]"
```

### Quickstart

#### 1) Schema inference (any Python data)

```python
import grizzly

data = [
    {"user": {"id": 1, "name": "Ada"}, "items": [{"id": 10}, {"id": 11}]},
    {"user": {"id": 2, "name": None}, "items": []},
]

schema = grizzly.detect_schema(data, sample_size=1000)
cols = grizzly.detect_columns(data)
grizzly.info(data, show_examples=True)
```

Grizzly can also normalize common DS inputs (pandas/numpy/pyarrow/CSV paths) into sampled Python-native records:

```python
import grizzly

records = grizzly.normalize("data.parquet", sample_size=1000)
schema = grizzly.detect_schema(records)
```

#### 2) CSV profiling / EDA (Rust)

```python
import grizzly

g = grizzly.Grizzly("playground/diabetes_data_raw.csv.gz", sample_size=100_000)

# Full profile (types + examples + mode) OR lite profile (numeric stats only)
prof = g.csv_profile(lite=False)
rep = g.eda(lite=True, return_json=True)  # fast report dict

print(rep["dataset"])
print(rep["missing"][:3])
print(rep["numeric"][:1])
```

#### 3) Column naming (header vs no header)

- If the file **has a header**, column names come from that header.
- If the file **does not have a header**, Grizzly uses synthetic names: **`col_0 ... col_{n-1}`**.

This is why your whitespace datasets benchmark with targets like `col_9`.

#### 4) Min-max scaling (transform to a new CSV)

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000)
scaler = g.fit_minmax()
scaler.transform("data_scaled.csv")
```

You can also call the lower-level functions:

```python
import grizzly

params = grizzly.csv_minmax_params("data.csv.gz", sample_size=100_000)["params"]
grizzly.csv_transform_minmax("data.csv.gz", "data_scaled.csv", params, delimiter=None)
```

#### 5) Train a model (Rust-native linear regression from CSV)

```python
import grizzly

g = grizzly.Grizzly("playground/large_100k.csv.gz", sample_size=1_000_000)
res = g.fit_linear_regression(target="col_9", train_frac=0.8, seed=0, ridge_lambda=0.0)
print(res["r2"], len(res["coef"]), res["intercept"])
```

### ML: regression workflows

Grizzly currently focuses on **fast linear models** with a pragmatic API:

- **Rust-native**: train directly from CSV/CSV.GZ without NumPy (`csv_linear_regression` / `Grizzly.fit_linear_regression`)
- **NumPy-based**: bring data into arrays and use lightweight sklearn-style models (`grizzly.ml`)

#### Rust-native regression (no NumPy)

Use this when you want the fastest path from a CSV file to a baseline model:

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000, fast_csv=True)

# Optional: restrict to a subset of columns (like a projection)
g = g.select(["col_0", "col_3", "col_9"])

res = g.fit_linear_regression(
    target="col_9",
    features=["col_0", "col_3"],   # optional; default = all except target (and respects select())
    train_frac=0.8,
    seed=0,
    shuffle=True,
    ridge_lambda=0.0,
    return_debug=False,
)

print("r2:", res["r2"])
print("coef_len:", len(res["coef"]), "intercept:", res["intercept"])
print("train_n:", res["train_n"], "test_n:", res["test_n"])
```

**What you get back**

- `r2`: test-set \(R^2\)
- `coef`: list of feature coefficients (same order as `features`)
- `intercept`: float
- `train_n`, `test_n`: row counts actually used for scoring (rows with parse failures are skipped)

If you pass `return_debug=True`, the result also includes:

- `test_n_assigned`: how many rows were assigned to test before skipping parse failures
- `ss_res`, `ss_tot`, `y_mean_test`: debug stats for the \(R^2\) calculation

**Notes / gotchas**

- **`sample_size` matters**: the model is trained on up to `sample_size` rows.
- **`fast_csv=True`** is faster but assumes “simple” CSVs (no quoted newlines). Use `fast_csv=False` for maximum CSV correctness.
- **Numeric-only**: non-numeric cells in numeric columns can cause a row to be skipped for training/scoring.

#### NumPy regression (arrays + lightweight models)

If you want to do preprocessing or compare with array-based baselines, convert to NumPy first:

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=200_000, fast_csv=True)
X, y = g.to_numpy(sampled=True, dtype="float32", target="col_9")

from grizzly.ml import LinearRegression, RidgeRegression

lr = LinearRegression().fit(X, y)
print("lr r2:", lr.score(X, y))

ridge = RidgeRegression(alpha=1.0).fit(X, y)
print("ridge r2:", ridge.score(X, y))
```

If you do scaling, prefer **train-only** scaling in NumPy to avoid leakage. `csv_transform_minmax` is great for speed demos and pipelines, but it scales using min/max derived from the sampled profile (not a strict train-only fit).

### Performance knobs (important)

- **`sample_size`**: Grizzly is sampling-first by design. Many operations stop after `sample_size` rows.
- **`fast_csv`**:
  - `fast_csv=True` is faster (parallel chunking) but assumes CSV is “simple”.
  - If you need maximum correctness (e.g., tricky quoting / quoted newlines), use **`fast_csv=False`**.

### Performance (local benchmark results)

Below are real-world numbers from a local benchmark run (not a shipped benchmark harness).

- **Metric**: median of **5 runs**
- **Sample size**: **1,000,000** rows (profiling/transform are sampling-first)
- **Mode**: full (type inference + examples + frequency tracking)
- **Workloads**:
  - **Profile**: read + compute stats
  - **Transform**: read + min-max scale + write
  - **Total**: end-to-end (profile + transform)

#### 100k rows (6.41 MB) 

<table>
  <thead>
    <tr>
      <th align="left">Workload</th>
      <th align="right">Pandas (ms)</th>
      <th align="right">Polars (ms)</th>
      <th align="right">Grizzly (ms)</th>
      <th align="right">Grizzly vs Pandas</th>
      <th align="right">Grizzly vs Polars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Profile</strong></td>
      <td align="right">134.03</td>
      <td align="right">41.39</td>
      <td align="right"><strong>81.00</strong></td>
      <td align="right">1.65× faster</td>
      <td align="right">1.96× slower</td>
    </tr>
    <tr>
      <td><strong>Transform</strong></td>
      <td align="right">301.22</td>
      <td align="right">91.22</td>
      <td align="right"><strong>69.92</strong></td>
      <td align="right">4.31× faster</td>
      <td align="right">1.30× faster</td>
    </tr>
    <tr>
      <td><strong>Total</strong></td>
      <td align="right">435.25</td>
      <td align="right">132.62</td>
      <td align="right"><strong>150.92</strong></td>
      <td align="right">2.88× faster</td>
      <td align="right">1.14× slower</td>
    </tr>
  </tbody>
</table>

#### Small demo (0.01 MB) 

<table>
  <thead>
    <tr>
      <th align="left">Workload</th>
      <th align="right">Pandas (ms)</th>
      <th align="right">Polars (ms)</th>
      <th align="right">Grizzly (ms)</th>
      <th align="right">Grizzly vs Pandas</th>
      <th align="right">Grizzly vs Polars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Profile</strong></td>
      <td align="right">0.82</td>
      <td align="right">0.35</td>
      <td align="right"><strong>0.46</strong></td>
      <td align="right">1.77× faster</td>
      <td align="right">1.32× slower</td>
    </tr>
    <tr>
      <td><strong>Transform</strong></td>
      <td align="right">0.99</td>
      <td align="right">0.53</td>
      <td align="right"><strong>0.27</strong></td>
      <td align="right">3.73× faster</td>
      <td align="right">1.99× faster</td>
    </tr>
    <tr>
      <td><strong>Total</strong></td>
      <td align="right">1.82</td>
      <td align="right">0.88</td>
      <td align="right"><strong>0.73</strong></td>
      <td align="right">2.48× faster</td>
      <td align="right">1.21× faster</td>
    </tr>
  </tbody>
</table>

#### 500k rows (32.02 MB) 

<table>
  <thead>
    <tr>
      <th align="left">Workload</th>
      <th align="right">Pandas (ms)</th>
      <th align="right">Polars (ms)</th>
      <th align="right">Grizzly (ms)</th>
      <th align="right">Grizzly vs Pandas</th>
      <th align="right">Grizzly vs Polars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Profile</strong></td>
      <td align="right">784.30</td>
      <td align="right">213.53</td>
      <td align="right"><strong>399.30</strong></td>
      <td align="right">1.96× faster</td>
      <td align="right">1.87× slower</td>
    </tr>
    <tr>
      <td><strong>Transform</strong></td>
      <td align="right">1605.27</td>
      <td align="right">456.70</td>
      <td align="right"><strong>352.62</strong></td>
      <td align="right">4.55× faster</td>
      <td align="right">1.30× faster</td>
    </tr>
    <tr>
      <td><strong>Total</strong></td>
      <td align="right">2389.57</td>
      <td align="right">670.23</td>
      <td align="right"><strong>751.92</strong></td>
      <td align="right">3.18× faster</td>
      <td align="right">1.12× slower</td>
    </tr>
  </tbody>
</table>

**Takeaway**: Grizzly is consistently faster than pandas end-to-end here, and the min-max transform is competitive with (and often faster than) Polars, while the full-profile workload is where Polars still leads on larger inputs.

### Generate synthetic CSVs (quick + repeatable)

If you want fast, repeatable datasets for local testing, use this snippet.
It can generate:

- **Delimited CSV** (comma/tab/etc.) optionally with a header
- **Whitespace-separated headerless** files (Grizzly will name columns `col_0..col_{n-1}`)
- **`.csv` or `.csv.gz`**

```python
from __future__ import annotations

import gzip
import random
from pathlib import Path


def make_regression_csv(
    out_path: str | Path,
    *,
    n_rows: int = 100_000,
    n_features: int = 20,
    seed: int = 0,
    delimiter: str = "whitespace",  # "whitespace" or "," or "\t" ...
    header: bool = False,
    noise: float = 0.1,
) -> Path:
    """
    Create a synthetic regression dataset with columns:
      - features: col_0..col_{n_features-1}
      - target:   col_{n_features}
    """
    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed underlying linear model
    w = [rng.uniform(-2.0, 2.0) for _ in range(n_features)]
    b = rng.uniform(-1.0, 1.0)

    cols = [f"col_{i}" for i in range(n_features + 1)]
    # Target column name is always the last one:
    #   target = f"col_{n_features}"

    opener = gzip.open if out_path.suffix == ".gz" else open
    with opener(out_path, "wt", newline="") as f:  # type: ignore[arg-type]
        if delimiter == "whitespace":
            if header:
                f.write(" ".join(cols) + "\n")
            for _ in range(n_rows):
                x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
                y = sum(w[i] * x[i] for i in range(n_features)) + b + rng.gauss(0.0, noise)
                f.write(" ".join([*(f"{v:.6f}" for v in x), f"{y:.6f}"]) + "\n")
        else:
            if header:
                f.write(delimiter.join(cols) + "\n")
            for _ in range(n_rows):
                x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
                y = sum(w[i] * x[i] for i in range(n_features)) + b + rng.gauss(0.0, noise)
                f.write(delimiter.join([*(f"{v:.6f}" for v in x), f"{y:.6f}"]) + "\n")

    return out_path


if __name__ == "__main__":
    p = make_regression_csv("data/synth_100k.csv.gz", n_rows=100_000, n_features=20, delimiter="whitespace", header=False)
    print("wrote:", p)
    print("target:", "col_20")
```

### Pipe generated CSVs into Grizzly (profile + EDA + train)

This is the simplest “end-to-end” local flow:

```python
import grizzly

path = "data/synth_100k.csv.gz"

g = grizzly.Grizzly(path, sample_size=1_000_000, fast_csv=True)

# Quick EDA snapshot (JSON)
rep = g.eda(lite=True, return_json=True)
print(rep["dataset"])
print("top missing:", rep["missing"][:3])

# Train rust-native linear regression directly from the CSV (no numpy)
res = g.fit_linear_regression(target="col_20", train_frac=0.8, seed=0, ridge_lambda=0.0)
print("r2:", res["r2"], "coef_len:", len(res["coef"]))
```

If you generated a **headerless whitespace** file, use `col_0..col_{n-1}` names as shown above.

### Tip: always run with the venv interpreter

If you ever see `ModuleNotFoundError: No module named 'grizzly'`, you’re almost certainly running with the
system Python instead of the venv.

Prefer these forms (they avoid any shell activation issues):

```bash
.venv/bin/python -c "import grizzly; print(grizzly.is_native())"
.venv/bin/python your_script.py
```

### Troubleshooting

- **`native: False` / “native extension not loaded”**:
  - Ensure you’re on **Python 3.14** (`python --version`)
  - Rebuild: `maturin develop --release`
  - Confirm: `python -c "import grizzly; print(grizzly.is_native())"`

- **`KeyError` / “target not found” for headerless CSVs**:
  - Use `col_0..col_{n-1}` naming (see “Column naming” above).

### License
(see `LICENSE`).
