<div align="center">

# Grizzly

<p align="center">
  <strong>Rust-powered data profiling and schema inference for Python</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/Python-3.14+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyO3-FFD43B?style=for-the-badge&logo=python&logoColor=black" alt="PyO3">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Transform-4.55x_faster-success?style=flat-square" alt="Transform Speed">
  <img src="https://img.shields.io/badge/Profile-1.96x_faster-blue?style=flat-square" alt="Profile Speed">
  <img src="https://img.shields.io/badge/No_NumPy-Required-orange?style=flat-square" alt="No NumPy">
  <img src="https://img.shields.io/badge/Native_ML-Linear_Regression-purple?style=flat-square" alt="Native ML">
</p>

---

<p align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-features">Features</a> •
  <a href="#-performance">Performance</a> •
  <a href="#-ml-workflows">ML Workflows</a> •
  <a href="#-api-reference">API</a>
</p>

</div>

---

## Overview

Grizzly is a Python package with a **Rust (PyO3) core** for high-performance data operations:

<table>
<tr>
<td width="50%">

### What It Does

- Schema/column detection from any Python data
- CSV/CSV.GZ profiling with stats & percentiles
- Fast min-max scaling transforms
- Rust-native linear regression (no NumPy!)

</td>
<td width="50%">

### Why It's Fast

- Rust core via PyO3 bindings
- Sampling-first design
- Parallel chunked CSV processing
- Zero-copy where possible

</td>
</tr>
</table>

---

## Features

<table>
<tr>
<td align="center" width="25%">
<br>
<strong>Schema Inference</strong>
<br><br>
<sub>Detect types from nested dicts, lists, iterables</sub>
</td>
<td align="center" width="25%">
<br>
<strong>CSV Profiling</strong>
<br><br>
<sub>Stats, percentiles, outliers, missing data</sub>
</td>
<td align="center" width="25%">
<br>
<strong>Fast Transforms</strong>
<br><br>
<sub>Min-max scaling 4.5x faster than Pandas</sub>
</td>
<td align="center" width="25%">
<br>
<strong>Native ML</strong>
<br><br>
<sub>Linear regression without NumPy</sub>
</td>
</tr>
</table>

<details>
<summary><strong>Path Notation (Schema Inference)</strong></summary>
<br>

| Pattern | Example | Description |
|---------|---------|-------------|
| Dict keys | `user.name` | Keys joined with `.` |
| Arrays | `items[].id` | Arrays add `[]` |
| Nested | `matrix[][].value` | Multi-dimensional |

</details>

---

## Quickstart

### Requirements

```
Python >= 3.14
Rust toolchain (rustup recommended)
```

### Installation

```bash
# Clone and setup
cd grizzly
python3.14 -m venv .venv
source .venv/bin/activate

# Install build tools
python -m pip install -U pip maturin pytest

# Build native extension
maturin develop --release

# Verify installation
python -c "import grizzly; print('native:', grizzly.is_native())"
```

<details>
<summary><strong>Optional: Install all extras</strong></summary>

```bash
python -m pip install ".[all]"
```

</details>

---

## API Reference

### 1. Schema Inference

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

<details>
<summary><strong>Normalize various data sources</strong></summary>

```python
import grizzly

# Works with pandas, numpy, pyarrow, CSV paths
records = grizzly.normalize("data.parquet", sample_size=1000)
schema = grizzly.detect_schema(records)
```

</details>

### 2. CSV Profiling / EDA

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=100_000)

# Full profile (types + examples + mode)
prof = g.csv_profile(lite=False)

# Fast EDA report
rep = g.eda(lite=True, return_json=True)
print(rep["dataset"])
print(rep["missing"][:3])
print(rep["numeric"][:1])
```

### 3. Min-Max Scaling

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000)
scaler = g.fit_minmax()
scaler.transform("data_scaled.csv")
```

<details>
<summary><strong>Lower-level API</strong></summary>

```python
import grizzly

params = grizzly.csv_minmax_params("data.csv.gz", sample_size=100_000)["params"]
grizzly.csv_transform_minmax("data.csv.gz", "data_scaled.csv", params, delimiter=None)
```

</details>

### 4. Linear Regression (Rust-Native)

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000)
res = g.fit_linear_regression(target="col_9", train_frac=0.8, seed=0)

print(f"R²: {res['r2']}")
print(f"Coefficients: {len(res['coef'])}")
print(f"Intercept: {res['intercept']}")
```

---

## ML Workflows

Grizzly focuses on **fast linear models** with a pragmatic API:

<table>
<tr>
<th width="50%">Rust-Native (No NumPy)</th>
<th width="50%">NumPy-Based</th>
</tr>
<tr>
<td>

Train directly from CSV/CSV.GZ
- Fastest path to baseline model
- No array conversion overhead
- Built-in train/test split

</td>
<td>

Convert to arrays first
- Sklearn-style API
- More preprocessing options
- Ridge regression included

</td>
</tr>
</table>

### Rust-Native Regression

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000, fast_csv=True)

# Optional: select specific columns
g = g.select(["col_0", "col_3", "col_9"])

res = g.fit_linear_regression(
    target="col_9",
    features=["col_0", "col_3"],  # default: all except target
    train_frac=0.8,
    seed=0,
    shuffle=True,
    ridge_lambda=0.0,
    return_debug=False,
)

print(f"R²: {res['r2']:.4f}")
print(f"Train: {res['train_n']}, Test: {res['test_n']}")
```

<details>
<summary><strong>Return Values</strong></summary>

| Key | Type | Description |
|-----|------|-------------|
| `r2` | float | Test-set R² |
| `coef` | list | Feature coefficients |
| `intercept` | float | Model intercept |
| `train_n` | int | Training rows used |
| `test_n` | int | Test rows used |

With `return_debug=True`:
- `test_n_assigned`, `ss_res`, `ss_tot`, `y_mean_test`

</details>

### NumPy Regression

```python
import grizzly
from grizzly.ml import LinearRegression, RidgeRegression

g = grizzly.Grizzly("data.csv.gz", sample_size=200_000, fast_csv=True)
X, y = g.to_numpy(sampled=True, dtype="float32", target="col_9")

lr = LinearRegression().fit(X, y)
print(f"LR R²: {lr.score(X, y):.4f}")

ridge = RidgeRegression(alpha=1.0).fit(X, y)
print(f"Ridge R²: {ridge.score(X, y):.4f}")
```

---

## Performance

<div align="center">

### Benchmark Summary

| Metric | vs Pandas | vs Polars |
|:------:|:---------:|:---------:|
| **Transform** | **4.55x faster** | **1.30x faster** |
| **Profile** | **1.96x faster** | 1.87x slower |
| **Total** | **3.18x faster** | 1.12x slower |

<sub>500K rows, 32 MB dataset, median of 5 runs</sub>

</div>

<details>
<summary><strong>100K Rows (6.41 MB)</strong></summary>
<br>

| Workload | Pandas (ms) | Polars (ms) | Grizzly (ms) | vs Pandas | vs Polars |
|----------|------------:|------------:|-------------:|----------:|----------:|
| **Profile** | 134.03 | 41.39 | **81.00** | 1.65x faster | 1.96x slower |
| **Transform** | 301.22 | 91.22 | **69.92** | 4.31x faster | 1.30x faster |
| **Total** | 435.25 | 132.62 | **150.92** | 2.88x faster | 1.14x slower |

</details>

<details>
<summary><strong>500K Rows (32.02 MB)</strong></summary>
<br>

| Workload | Pandas (ms) | Polars (ms) | Grizzly (ms) | vs Pandas | vs Polars |
|----------|------------:|------------:|-------------:|----------:|----------:|
| **Profile** | 784.30 | 213.53 | **399.30** | 1.96x faster | 1.87x slower |
| **Transform** | 1605.27 | 456.70 | **352.62** | 4.55x faster | 1.30x faster |
| **Total** | 2389.57 | 670.23 | **751.92** | 3.18x faster | 1.12x slower |

</details>

<details>
<summary><strong>Small Demo (0.01 MB)</strong></summary>
<br>

| Workload | Pandas (ms) | Polars (ms) | Grizzly (ms) | vs Pandas | vs Polars |
|----------|------------:|------------:|-------------:|----------:|----------:|
| **Profile** | 0.82 | 0.35 | **0.46** | 1.77x faster | 1.32x slower |
| **Transform** | 0.99 | 0.53 | **0.27** | 3.73x faster | 1.99x faster |
| **Total** | 1.82 | 0.88 | **0.73** | 2.48x faster | 1.21x faster |

</details>

> **Takeaway**: Grizzly consistently beats Pandas end-to-end. Transform operations are competitive with (often faster than) Polars. Full profiling is where Polars leads on larger inputs.

---

## Performance Knobs

<table>
<tr>
<td width="50%">

### `sample_size`

Grizzly is **sampling-first** by design. Many operations stop after `sample_size` rows.

```python
g = grizzly.Grizzly("big.csv", sample_size=100_000)
```

</td>
<td width="50%">

### `fast_csv`

| Mode | Speed | Compatibility |
|------|-------|---------------|
| `True` | Faster (parallel) | Simple CSVs |
| `False` | Slower | Quoted newlines, tricky CSVs |

</td>
</tr>
</table>

---

## Column Naming

| File Type | Column Names |
|-----------|--------------|
| **With header** | From header row |
| **No header** | `col_0`, `col_1`, ... `col_{n-1}` |

This is why headerless datasets use targets like `col_9`.

---

## Generate Synthetic Data

<details>
<summary><strong>Synthetic CSV Generator</strong></summary>

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
    delimiter: str = "whitespace",
    header: bool = False,
    noise: float = 0.1,
) -> Path:
    """Create synthetic regression dataset."""
    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = [rng.uniform(-2.0, 2.0) for _ in range(n_features)]
    b = rng.uniform(-1.0, 1.0)
    cols = [f"col_{i}" for i in range(n_features + 1)]

    opener = gzip.open if out_path.suffix == ".gz" else open
    with opener(out_path, "wt", newline="") as f:
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

# Usage
p = make_regression_csv("data/synth_100k.csv.gz", n_rows=100_000, n_features=20)
print(f"wrote: {p}, target: col_20")
```

</details>

<details>
<summary><strong>End-to-End Example</strong></summary>

```python
import grizzly

path = "data/synth_100k.csv.gz"
g = grizzly.Grizzly(path, sample_size=1_000_000, fast_csv=True)

# Quick EDA
rep = g.eda(lite=True, return_json=True)
print(rep["dataset"])
print("top missing:", rep["missing"][:3])

# Train model directly from CSV
res = g.fit_linear_regression(target="col_20", train_frac=0.8, seed=0)
print(f"R²: {res['r2']:.4f}, coefficients: {len(res['coef'])}")
```

</details>

---

## Troubleshooting

<details>
<summary><strong>"native: False" or extension not loaded</strong></summary>

1. Ensure Python 3.14: `python --version`
2. Rebuild: `maturin develop --release`
3. Verify: `python -c "import grizzly; print(grizzly.is_native())"`

</details>

<details>
<summary><strong>KeyError / "target not found" for headerless CSVs</strong></summary>

Use synthetic column names: `col_0`, `col_1`, ... `col_{n-1}`

```python
res = g.fit_linear_regression(target="col_9")  # Not "target" or custom name
```

</details>

<details>
<summary><strong>ModuleNotFoundError: No module named 'grizzly'</strong></summary>

You're using system Python instead of venv. Use explicit paths:

```bash
.venv/bin/python -c "import grizzly; print(grizzly.is_native())"
.venv/bin/python your_script.py
```

</details>

---

<div align="center">

## License

See `LICENSE` file.

---

<sub>Built with Rust and PyO3</sub>

</div>
