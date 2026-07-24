# Grizzly benchmarks

Every performance number in the project README is produced by this suite. If a
claim is not reproducible from this directory, it should not be in the README.

## Reproducing

The published numbers come from a container, not a laptop. Pinning the image
and the CPU/memory allocation fixes the software half of the environment --
same Python, same compiler, same library versions, same core count -- which a
developer machine cannot promise:

```bash
make docker-bench       # pinned CPU/memory, writes data/container-results.json
make docker-study       # the sampling accuracy-vs-speed sweep
```

To publish a run, copy its results over the committed ones and regenerate:

```bash
cp data/container-results.json benches/results/results.json
make render
```

On the host directly, which is faster to iterate on but noisier:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip maturin
python -m pip install -r benches/requirements.txt
maturin develop --release

python -m benches.bench --rows 100000 500000 --repetitions 5 --strict
python -m benches.render --check      # verify README tables match results.json
```

Datasets are generated on demand into `data/` (gitignored) and are byte-identical
for a given `(shape, rows, seed)`, so results are comparable across machines and
across commits.

## What is measured

| Workload | Definition (identical for all three libraries) |
|----------|------------------------------------------------|
| `profile` | Read a CSV from disk and compute, per column: inferred type, rows observed, null count, and for numeric columns min, max, mean, population std, and the p25/p50/p75/p90/p95 quantiles. |
| `transform` | Read a CSV from disk, min-max scale every numeric column to `[0, 1]` using that column's own min and max, and write the result back out as CSV. |

Reading and writing are inside the timed region for every library. That is the
operation a user actually performs; excluding I/O would measure something
nobody runs.

## Dataset shapes

| Shape | Columns | Purpose |
|-------|---------|---------|
| `numeric` | 20 float features + 1 target | Homogeneous float parsing. The target is a noisy linear function of the features, so the same file also serves regression benchmarks. |
| `mixed` | Repeating float / int / categorical / nullable-float | Type inference, mode tracking, and missing-value handling, which the all-float shape never exercises. Nullable columns are blanked at a 5% rate. |

## How the suite avoids flattering Grizzly

These are the specific ways a benchmark like this goes wrong, and what the
driver does about each.

**Sampling is capped, not assumed.** Grizzly is sampling-first: `sample_size`
bounds how many rows it reads, and it defaults to 1000. A profiling benchmark
that leaves the default in place compares Grizzly reading 1000 rows against
pandas reading all 500,000, which produces an arbitrarily large and completely
meaningless speedup. The driver passes `sample_size = rows × 4` and then
asserts from the returned `rows_sampled` that Grizzly actually saw every row.
The margin above 1.0 is required because chunked parallel reads align to record
boundaries and can otherwise stop slightly short.

**Outputs must agree.** Each library's result is fingerprinted — per column:
rows observed, null count, min, max, mean — and the fingerprints are compared
across libraries. A speedup from an implementation that quietly skipped work
shows up as a mismatch instead. `--strict` makes a mismatch a non-zero exit,
which is how CI runs it. For the transform workload the driver additionally
re-reads the output file and asserts every scaled column actually spans
`[0, 1]`.

Note that Grizzly's `count` field means *rows observed including nulls*, while
pandas, polars, and SQL all use `count` for the non-null tally. The fingerprint
normalises to an unambiguous `observed` field so the check compares data rather
than naming conventions.

**One cold process per measurement.** Every repetition runs in a fresh
interpreter. Libraries cannot inherit each other's warm import caches or
allocator state, and the order they run in cannot change the result.

**Warmup is discarded.** The first repetition of each cell primes the OS page
cache and is thrown away. The headline figure is the median of the remainder;
`results.json` also records min, max, mean, and standard deviation so a reader
can see how noisy a cell was.

**Memory is reported alongside time.** Peak RSS is captured per run. A library
that wins on wall-clock by buffering the entire file in memory should have to
show that.

**The environment is recorded.** CPU model, core count, RAM, OS, interpreter,
`rustc` version, cargo profile, the Grizzly commit, and the exact version of
every comparison library are written into `results.json`.

## Known limitations

Stated explicitly, because a benchmark that hides its caveats is not evidence.

- **Single machine.** Results reflect one CPU and one storage device. Ratios
  between libraries tend to be more portable than absolute timings, but neither
  should be assumed to transfer.
- **Synthetic data.** Generated columns are well-formed: no quoted newlines, no
  ragged rows, no exotic encodings. Grizzly's `fast_csv` path is fastest on
  exactly this kind of input, so real-world messy CSVs may narrow the gap.
- **Page cache is warm.** Warmup means files are cached in RAM. Cold-start disk
  reads are not measured.
- **Output formatting differs.** Each library writes floats using its own
  default formatting, so transform output files are not byte-identical and
  their sizes differ slightly. The equivalence check compares scaled ranges
  rather than bytes.
- **Approximate quantiles.** Grizzly's percentiles come from a t-digest, which
  is approximate by construction; pandas and polars compute them exactly. The
  fingerprint therefore compares min/max/mean rather than quantiles. The size of
  that approximation is measured separately by `study_sampling.py` and by
  `tests/test_quantile_accuracy.py`.
- **The container fixes software, not hardware.** Pinning the image and the
  CPU/memory allocation removes version and core-count variation. It does not
  make the host machine idle, so cells whose standard deviation exceeds 20% of
  their median are still flagged in the rendered output.

## The sampling study

`study_sampling.py` answers the question a speed comparison cannot: what does
reading less data cost in accuracy? It sweeps `sample_size` over a fixed file
and reports rank error, value error, and wall-clock at each setting.

Its most useful finding is structural rather than numeric. Sampling does **not**
read a prefix -- the profiler splits the file into per-thread chunks and each
consumes from its own region, so a small sample is spread across the whole file.
Profiling 200,000 strictly ascending rows with `sample_size=256` reports a
maximum of 198,566, not 255. That is what makes sampling usable on data with
meaningful row order, where `head -n` would be badly biased.

## Files

| File | Role |
|------|------|
| `gen_data.py` | Deterministic dataset generation. |
| `_runner.py` | One measurement in an isolated process. |
| `bench.py` | Driver: repetitions, equivalence checks, environment capture. |
| `_fit_runner.py` | One model-fit measurement in an isolated process. |
| `bench_fit.py` | Fit benchmark: CSV → split → model → R², with a coefficient-agreement gate. |
| `_study_runner.py` | One profiling measurement for the sampling study. |
| `study_sampling.py` | Accuracy-vs-speed sweep over `sample_size`. |
| `study_memory.py` | Memory-ceiling study under container limits. |
| `render.py` | Renders results into the README's generated sections, hero tiles included. |
| `requirements.txt` | Pinned comparison libraries. |
| `results/results.json` | Committed benchmark results. |
| `results/fit_results.json` | Committed fit-benchmark results. |
| `results/sampling_study.json` | Committed sampling-study results. |
| `results/memory_study.json` | Committed memory-ceiling results. |
