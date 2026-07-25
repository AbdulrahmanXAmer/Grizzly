//! Grizzly: High-Performance Data Profiling in Rust
//!
//! Optimizations:
//! - Bounded gzip decompression (stops after sample_size rows)
//! - Memory mapping (memmap2) for zero-copy I/O on raw files
//! - Hash-based string frequency tracking (no global interning)
//! - atoi-simd for SIMD-accelerated integer parsing (same as Polars)
//! - T-Digest streaming quantiles (O(1) query)
//! - SIMD delimiter/quote detection (memchr2/memchr3)
//! - Rayon parallelization + GIL release
//! - Per-chunk budget (no atomics in profiling hot path)
//! - Fast delimiter-only parsing for transform (bypass csv crate)
//! - maybe_float_bytes fast-reject for string columns
//! - Indexed Vec for chunk ordering (no sort)

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyMapping, PySequence, PyString, PyTuple};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;

use ahash::AHashMap;
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use memchr::memchr;
use memmap2::Mmap;
use rayon::prelude::*;

// Type tracking with a bitmask (keeps hot-path allocations low).

// Streaming numeric stats with T-Digest quantiles.

// Hash-based frequency tracking (no global interning in the hot path).

// Per-column statistics.

// Cell processing (SIMD int parsing + fast-float).

// File I/O (mmap for raw files, bounded gzip for profiling).

enum FileData {
    Mmap(Mmap),
    Buffered(Vec<u8>),
}

/// Bounded gzip decompression: stop after ~`max_rows` newlines for sampling-first workflows.
fn load_gz_bounded(path: &str, max_rows: usize) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);

    // Estimate: ~200 bytes per row on average, plus some buffer
    let estimated_size = (max_rows + 100) * 256;
    let mut out = Vec::with_capacity(estimated_size.min(64 << 20)); // Cap at 64MB initial

    let mut buf = [0u8; 64 * 1024]; // 64KB read buffer
    let mut newline_count = 0usize;

    loop {
        let n = decoder.read(&mut buf)?;
        if n == 0 {
            break;
        } // EOF

        // Count newlines in this chunk
        for &b in &buf[..n] {
            if b == b'\n' {
                newline_count += 1;
            }
        }

        out.extend_from_slice(&buf[..n]);

        // Stop if we have enough rows (+ header + some margin)
        if newline_count >= max_rows + 10 {
            break;
        }
    }

    Ok(out)
}

fn load_file_data(path: &str) -> std::io::Result<FileData> {
    if path.to_lowercase().ends_with(".gz") {
        // Transform needs full data for output, so decompress fully.
        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut data = Vec::new();
        decoder.read_to_end(&mut data)?;
        Ok(FileData::Buffered(data))
    } else {
        // For raw files, memory map (zero-copy)
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(FileData::Mmap(mmap))
    }
}

/// Load file for profiling (bounded gzip if applicable).
fn load_file_for_profile(path: &str, sample_size: usize) -> std::io::Result<FileData> {
    if path.to_lowercase().ends_with(".gz") {
        // Only decompress until we have enough rows.
        let data = load_gz_bounded(path, sample_size)?;
        Ok(FileData::Buffered(data))
    } else {
        // For raw files, memory map (zero-copy)
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(FileData::Mmap(mmap))
    }
}

impl FileData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            FileData::Mmap(m) => m.as_ref(),
            FileData::Buffered(v) => v.as_slice(),
        }
    }
}

// Parallel CSV profiling.

/// Result of scanning a CSV for profiling, before conversion into a Python dict:
/// (column names, per-column stats, rows scanned, detected delimiter, header present).
type ProfileScan = (Vec<String>, Vec<ColStats>, usize, Option<u8>, bool);

/// Profile a CSV file.
///
/// # Arguments
/// * `path` - Path to CSV file (supports .gz)
/// * `sample_size` - Max rows to sample
/// * `max_examples` - Max examples per column
/// * `fast_csv` - If true, uses parallel byte chunking (assumes no quoted newlines).
///                If false, uses sequential reading (correct for any CSV).
/// * `lite` - If true, only compute numeric stats (min/max/mean/std/quantiles).
///            Skips type inference, examples, and frequency tracking for speed.
/// * `track_freq` - If true, track frequency for mode calculation
/// * `collect_examples` - If true, collect example values
// Argument count mirrors the public Python signature, which is the API contract;
// collapsing it into a config struct would only move the arity into the caller.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (path, sample_size=1000, max_examples=5, fast_csv=true, lite=false, track_freq=true, collect_examples=true))]
fn csv_profile(
    py: Python<'_>,
    path: String,
    sample_size: usize,
    max_examples: usize,
    fast_csv: bool,
    lite: bool,
    track_freq: bool,
    collect_examples: bool,
) -> PyResult<PyObject> {
    let mode = if lite { ScanMode::Lite } else { ScanMode::Full };
    csv_profile_impl(
        py,
        path,
        sample_size,
        max_examples,
        fast_csv,
        mode,
        track_freq,
        collect_examples,
    )
}

/// The scan behind `csv_profile`, with the mode exposed.
///
/// `ScanMode::Moments` is internal-only: it is what the scaler-params
/// functions run, and it skips the t-digest entirely — the digest costs ~8x
/// the moments it accompanies, and those callers discard the quantiles.
#[allow(clippy::too_many_arguments)]
fn csv_profile_impl(
    py: Python<'_>,
    path: String,
    sample_size: usize,
    max_examples: usize,
    fast_csv: bool,
    mode: ScanMode,
    track_freq: bool,
    collect_examples: bool,
) -> PyResult<PyObject> {
    let result = py.allow_threads(|| -> Result<ProfileScan, String> {
        // Profiling is sampling-first: use bounded gzip when applicable.
        let file_data = load_file_for_profile(&path, sample_size).map_err(|e| e.to_string())?;
        let bytes = file_data.as_bytes();

        if bytes.is_empty() {
            return Ok((vec![], vec![], 0, None, false));
        }

        // Find first line for sniffing
        let first_newline = memchr(b'\n', bytes).unwrap_or(bytes.len());
        let first_line = &bytes[..first_newline];

        // Determine split mode (delimiter or whitespace)
        let delimiter = sniff_delimiter_simd(first_line);
        let split_mode = match delimiter {
            Some(d) => SplitMode::Delim(d),
            None => SplitMode::Whitespace,
        };

        // For header detection and csv crate compat, we need a delimiter byte
        let delim_byte_for_detection = match split_mode {
            SplitMode::Delim(d) => d,
            SplitMode::Whitespace => b' ', // Use space as proxy for detection
        };

        // Header detection heuristic: compare numeric rate of row 0 vs subsequent rows.
        let has_header = detect_header_smart(bytes, delim_byte_for_detection, 5);

        // Get column names from first line (fast path - no csv crate)
        let header_line = if first_line.ends_with(b"\r") {
            &first_line[..first_line.len() - 1]
        } else {
            first_line
        };

        let col_names: Vec<String> = if has_header {
            get_fields(header_line, split_mode)
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let trimmed = trim_bytes(f);
                    if trimmed.is_empty() {
                        format!("col_{i}")
                    } else {
                        String::from_utf8_lossy(trimmed).into_owned()
                    }
                })
                .collect()
        } else {
            get_fields(header_line, split_mode)
                .iter()
                .enumerate()
                .map(|(i, _)| format!("col_{i}"))
                .collect()
        };

        let ncols = col_names.len();
        if ncols == 0 {
            return Ok((vec![], vec![], 0, delimiter, has_header));
        }

        // Skip header bytes if present
        let data_start = if has_header { first_newline + 1 } else { 0 };
        let data_bytes = &bytes[data_start..];

        let (merged_stats, total_rows): (Vec<ColStats>, usize) = if fast_csv {
            // Fast path: direct byte splitting (assumes no quoted newlines in fields).
            let num_threads = rayon::current_num_threads();
            let byte_chunks = chunk_bytes_aligned(data_bytes, num_threads);
            let n_chunks = byte_chunks.len();

            // Per-chunk row budget (avoid shared atomics).
            let per_chunk_budget = sample_size
                .checked_div(n_chunks)
                .map_or(sample_size, |budget| budget + 1);

            byte_chunks
                .into_par_iter()
                .map(|chunk_bytes| {
                    let mut local_stats: Vec<ColStats> = (0..ncols)
                        .map(|_| ColStats {
                            freq: FreqTracker::new(),
                            ..Default::default()
                        })
                        .collect();

                    let mut rows_in_chunk = 0usize;

                    // Shares FastLineIter with every other reader. The
                    // hand-rolled loop this replaces treated a blank line as
                    // end-of-chunk rather than skipping it, so a stray empty
                    // line silently dropped the rest of its chunk from the
                    // profile. Fields are visited in order, so no per-row Vec
                    // is allocated either.
                    for line in FastLineIter::new(chunk_bytes) {
                        if rows_in_chunk >= per_chunk_budget {
                            break;
                        }

                        for_each_field(line, split_mode, |i, field| {
                            if i >= ncols {
                                return;
                            }
                            match mode {
                                ScanMode::Moments => {
                                    process_cell_moments(field, &mut local_stats[i])
                                }
                                ScanMode::Lite => process_cell_lite(field, &mut local_stats[i]),
                                ScanMode::Full => process_cell(
                                    field,
                                    &mut local_stats[i],
                                    max_examples,
                                    track_freq,
                                    collect_examples,
                                ),
                            }
                        });

                        rows_in_chunk += 1;
                    }

                    for stat in &mut local_stats {
                        stat.finalize();
                    }

                    (local_stats, rows_in_chunk)
                })
                .reduce(
                    || {
                        let empty_stats: Vec<ColStats> = (0..ncols)
                            .map(|_| ColStats {
                                freq: FreqTracker::new(),
                                ..Default::default()
                            })
                            .collect();
                        (empty_stats, 0)
                    },
                    |(mut a_stats, a_rows), (mut b_stats, b_rows)| {
                        for (i, b_stat) in b_stats.iter_mut().enumerate() {
                            if i < a_stats.len() {
                                a_stats[i].merge(b_stat);
                            }
                        }
                        (a_stats, a_rows + b_rows)
                    },
                )
        } else {
            // SAFE PATH: Sequential reading (correct for any CSV including quoted newlines)
            // Use csv crate when we need full CSV correctness (handles quoted newlines, etc.)
            let mut stats: Vec<ColStats> = (0..ncols)
                .map(|_| ColStats {
                    freq: FreqTracker::new(),
                    ..Default::default()
                })
                .collect();

            let mut reader = ReaderBuilder::new()
                .has_headers(has_header)
                .delimiter(delim_byte_for_detection)
                .flexible(true)
                .from_reader(bytes);

            let mut total_rows = 0usize;

            for record in reader.byte_records().take(sample_size).flatten() {
                for (i, field) in record.iter().enumerate() {
                    if i < ncols {
                        match mode {
                            ScanMode::Moments => process_cell_moments(field, &mut stats[i]),
                            ScanMode::Lite => process_cell_lite(field, &mut stats[i]),
                            ScanMode::Full => process_cell(
                                field,
                                &mut stats[i],
                                max_examples,
                                track_freq,
                                collect_examples,
                            ),
                        }
                    }
                }
                total_rows += 1;
            }

            for stat in &mut stats {
                stat.finalize();
            }

            (stats, total_rows)
        };

        Ok((col_names, merged_stats, total_rows, delimiter, has_header))
    });

    let (col_names, stats, rows_sampled, delimiter, has_header) =
        result.map_err(pyo3::exceptions::PyIOError::new_err)?;

    // Build Python output
    let out = PyDict::new(py);
    out.set_item("path", &path)?;
    out.set_item("rows_sampled", rows_sampled)?;
    // Report delimiter correctly based on split mode
    let delimiter_str = match delimiter {
        Some(d) => (d as char).to_string(),
        None => "whitespace".to_string(),
    };
    out.set_item("delimiter", delimiter_str)?;
    out.set_item("has_header", has_header)?;

    let col_list = PyList::empty(py);
    for (i, name) in col_names.iter().enumerate() {
        if i >= stats.len() {
            break;
        }
        let s = &stats[i];

        let d = PyDict::new(py);
        d.set_item("name", name)?;
        d.set_item("index", i)?;
        d.set_item("count", s.count)?;
        d.set_item("null_count", s.null_count)?;
        d.set_item("types", mask_to_types(s.type_mask))?;
        d.set_item("inferred", infer_from_mask(s.type_mask))?;

        d.set_item("examples", &s.examples)?;

        if s.num.n > 0 {
            d.set_item("min", s.num.min)?;
            d.set_item("max", s.num.max)?;
            d.set_item("mean", s.num.mean)?;
            d.set_item("std", s.num.std_pop().unwrap_or(0.0))?;
            // Quantiles are None when the scan never fed the t-digest
            // (ScanMode::Moments); emitting a number there would be inventing
            // one. Full and Lite scans always have them when n > 0.
            for (key, q) in [
                ("median", 0.5),
                ("p25", 0.25),
                ("p75", 0.75),
                ("p90", 0.90),
                ("p95", 0.95),
            ] {
                match s.num.quantile(q) {
                    Some(v) => d.set_item(key, v)?,
                    None => d.set_item(key, py.None())?,
                }
            }
            d.set_item("outliers_3sigma", 0)?;
        } else {
            d.set_item("min", py.None())?;
            d.set_item("max", py.None())?;
            d.set_item("mean", py.None())?;
            d.set_item("std", py.None())?;
            d.set_item("median", py.None())?;
            d.set_item("p25", py.None())?;
            d.set_item("p75", py.None())?;
            d.set_item("p90", py.None())?;
            d.set_item("p95", py.None())?;
            d.set_item("outliers_3sigma", 0)?;
        }

        // Mode from hash-based tracker
        if let Some((val, count)) = s.freq.mode_int() {
            d.set_item("mode", val.to_string())?;
            d.set_item("mode_count", count)?;
        } else if let Some((val, count)) = s.freq.mode_string() {
            d.set_item("mode", &val)?;
            d.set_item("mode_count", count)?;
        } else {
            d.set_item("mode", py.None())?;
            d.set_item("mode_count", 0)?;
        }

        col_list.append(d)?;
    }

    out.set_item("columns", col_list)?;
    Ok(out.into())
}

// Min-max params extraction.

#[pyfunction]
fn csv_minmax_params(py: Python<'_>, path: String, sample_size: usize) -> PyResult<PyObject> {
    // Moments-only scan: min/max is all this needs, and the t-digest a lite
    // scan would build costs ~8x the moments themselves.
    let prof_obj = csv_profile_impl(
        py,
        path.clone(),
        sample_size,
        0,
        true,
        ScanMode::Moments,
        false,
        false,
    )?;
    let prof = prof_obj.bind(py).downcast::<PyDict>()?;
    let out = PyDict::new(py);
    out.set_item("path", path)?;
    let params = PyDict::new(py);

    let cols_any = prof
        .get_item("columns")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing columns"))?;
    let cols_list = cols_any.downcast::<PyList>()?;

    for c in cols_list.iter() {
        let cd = c.downcast::<PyDict>()?;
        let name: String = cd
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing name"))?
            .extract()?;
        let min_opt = cd.get_item("min")?;
        let max_opt = cd.get_item("max")?;

        if let (Some(min_v), Some(max_v)) = (min_opt, max_opt) {
            if !min_v.is_none() && !max_v.is_none() {
                let d = PyDict::new(py);
                d.set_item("min", &min_v)?;
                d.set_item("max", &max_v)?;
                params.set_item(name, d)?;
            }
        }
    }

    out.set_item("params", params)?;
    Ok(out.into())
}

/// Write `buf` at an absolute file offset, without moving a shared cursor.
///
/// Positioned writes are what allow the output chunks to be written
/// concurrently: each knows exactly where it belongs, so no writer has to wait
/// for another to finish appending. `File::write_all_at` is Unix-only and
/// `seek_write` is the Windows equivalent, hence the split.
#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut written = 0usize;
    while written < buf.len() {
        // seek_write is not guaranteed to consume the whole buffer.
        let n = file.seek_write(&buf[written..], offset + written as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "positioned write made no progress",
            ));
        }
        written += n;
    }
    Ok(())
}

// Parallel streaming transforms.

/// Apply a per-column affine transform `(x - offset) / scale`, streaming.
///
/// Min-max scaling and standardization differ only in how `(offset, scale)` is
/// derived -- `(min, max - min)` versus `(mean, std)` -- so they share this one
/// parallel writer instead of two near-identical copies of it. Rows are read,
/// transformed, and written in chunks, so memory stays bounded by the chunk
/// size rather than the file size.
///
/// A non-finite or effectively-zero scale maps the column to 0.0: a constant
/// column has no spread to scale by, and emitting NaN or infinity would poison
/// every downstream consumer of the output file.
fn transform_affine(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    params_map: AHashMap<String, (f64, f64)>,
    delimiter: Option<String>,
    has_header: Option<bool>,
) -> PyResult<PyObject> {
    let input_clone = input_path.clone();
    let output_clone = output_path.clone();

    let (rows_written, numeric_cols_scaled, has_header_detected) = py
        .allow_threads(move || -> Result<(usize, usize, bool), String> {
            let file_data = load_file_data(&input_clone).map_err(|e| e.to_string())?;
            let bytes = file_data.as_bytes();

            if bytes.is_empty() {
                return Ok((0, 0, false));
            }

            // Determine split mode (delimiter or whitespace)
            let first_newline = memchr(b'\n', bytes).unwrap_or(bytes.len());
            let first_line = &bytes[..first_newline];

            let split_mode = if let Some(d) = delimiter.and_then(|d| d.bytes().next()) {
                SplitMode::Delim(d) // User specified delimiter
            } else {
                // Auto-detect
                match sniff_delimiter_simd(first_line) {
                    Some(d) => SplitMode::Delim(d),
                    None => SplitMode::Whitespace, // FIX: Use whitespace mode!
                }
            };

            // For header detection with csv crate
            let delim_byte_for_detection = match split_mode {
                SplitMode::Delim(d) => d,
                SplitMode::Whitespace => b' ',
            };

            // Determine if file has header (auto-detect if not provided)
            let has_header_actual = has_header
                .unwrap_or_else(|| detect_header_smart(bytes, delim_byte_for_detection, 5));

            // Get first line fields for column count and naming
            let first_line_clean = if first_line.ends_with(b"\r") {
                &first_line[..first_line.len() - 1]
            } else {
                first_line
            };
            let first_fields = get_fields(first_line_clean, split_mode);
            let ncols = first_fields.len();

            if ncols == 0 {
                return Err("No columns detected in CSV".to_string());
            }

            // Column names based on header presence.
            let col_names: Vec<String> = if has_header_actual {
                // Use header values as names.
                first_fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let trimmed = trim_bytes(f);
                        if trimmed.is_empty() {
                            format!("col_{i}")
                        } else {
                            String::from_utf8_lossy(trimmed).into_owned()
                        }
                    })
                    .collect()
            } else {
                // No header: generate col_0, col_1, ...
                (0..ncols).map(|i| format!("col_{i}")).collect()
            };

            // Build per-column params once (None for columns not being scaled).
            let col_params: Vec<Option<(f64, f64)>> = col_names
                .iter()
                .map(|name| params_map.get(name).copied())
                .collect();

            // Count how many columns will actually be scaled (for validation)
            let numeric_cols_to_scale = col_params.iter().filter(|p| p.is_some()).count();

            // Split data into chunks for parallel processing.
            let data_start = if has_header_actual {
                first_newline + 1
            } else {
                0
            };
            let data_bytes = &bytes[data_start..];

            let num_threads = rayon::current_num_threads();
            let byte_chunks = chunk_bytes_aligned(data_bytes, num_threads);
            let _n_chunks = byte_chunks.len();

            // Output delimiter.
            let output_delim = match split_mode {
                SplitMode::Delim(d) => d,
                SplitMode::Whitespace => b' ',
            };

            // Collect per-chunk output buffers.
            // Output capacity estimate. Scaling a value written with a handful
            // of decimals produces a full round-trip double -- "3.141593"
            // becomes "0.4724761615149732" -- so a scaled file is roughly twice
            // its input. Reserving the old 1.25x guaranteed a reallocation and
            // a full copy of a multi-megabyte buffer in every chunk.
            let growth_numerator = if numeric_cols_to_scale > 0 { 9 } else { 5 };
            let estimate_capacity = |len: usize| len.saturating_mul(growth_numerator) / 4 + 64;

            // Build the header first: its length is the starting file offset,
            // and every chunk's destination is derived from it.
            let mut header = Vec::new();
            for (i, name) in col_names.iter().enumerate() {
                if i > 0 {
                    header.push(output_delim);
                }
                write_field_csv(&mut header, name.as_bytes(), output_delim);
            }
            header.push(b'\n');

            let file = File::create(&output_clone).map_err(|e| e.to_string())?;
            write_at(&file, &header, 0).map_err(|e| e.to_string())?;

            // Chunks are processed in batches rather than all at once.
            //
            // Formatting every chunk up front and only then writing meant the
            // whole output existed in memory simultaneously: a 100 MB input
            // produced 200 MB of live buffers and a 327 MB peak RSS, which also
            // made the "memory bounded by chunk size" claim in the docs false.
            // Batching bounds live output to one batch, and lets the allocator
            // reuse the same blocks for every batch instead of faulting in
            // fresh pages for each of ~128 chunks.
            let batch_size = (rayon::current_num_threads() * 2).max(1);

            let mut total_rows = 0usize;
            let mut cursor = header.len() as u64;

            for batch in byte_chunks.chunks(batch_size) {
                let batch_results: Vec<(Vec<u8>, usize)> = batch
                    .par_iter()
                    .map(|&chunk_bytes| {
                        let mut output_buf =
                            Vec::with_capacity(estimate_capacity(chunk_bytes.len()));
                        let mut ryu_buf = ryu::Buffer::new();
                        let mut rows = 0usize;

                        // Fields are visited in order and written as they are seen,
                        // so there is no reason to collect them into a Vec first.
                        for line in FastLineIter::new(chunk_bytes) {
                            for_each_field(line, split_mode, |i, field| {
                                if i > 0 {
                                    output_buf.push(output_delim);
                                }

                                let trimmed = trim_bytes(field);

                                if let Some(Some((offset, scale))) = col_params.get(i) {
                                    // Only parse columns with params (known numeric).
                                    if let Ok(x) = fast_float::parse::<f64, _>(trimmed) {
                                        let scaled = if scale.is_finite() && scale.abs() > 1e-12 {
                                            (x - offset) / scale
                                        } else {
                                            0.0
                                        };
                                        // format(), not format_finite(): fast_float
                                        // parses "inf" and "nan", so `scaled` is not
                                        // guaranteed finite and format_finite's
                                        // output would be unspecified for those.
                                        let s = ryu_buf.format(scaled);
                                        output_buf.extend_from_slice(s.as_bytes());
                                    } else {
                                        // Non-numeric value in numeric column, pass through
                                        write_field_csv(&mut output_buf, trimmed, output_delim);
                                    }
                                } else {
                                    // No transform, fast path for non-numeric columns
                                    write_field_csv(&mut output_buf, trimmed, output_delim);
                                }
                            });
                            output_buf.push(b'\n');
                            rows += 1;
                        }

                        (output_buf, rows)
                    })
                    .collect();

                // Offsets within the batch are known once it is formatted, so
                // the batch is written in parallel too: each buffer knows
                // exactly where it belongs and no writer waits on another.
                let mut offsets = Vec::with_capacity(batch_results.len());
                for (buffer, rows) in &batch_results {
                    offsets.push(cursor);
                    cursor += buffer.len() as u64;
                    total_rows += rows;
                }

                batch_results
                    .par_iter()
                    .zip(offsets.par_iter())
                    .try_for_each(|((buffer, _), &offset)| {
                        write_at(&file, buffer, offset).map_err(|e| e.to_string())
                    })?;
            }

            // Deliberately no fsync. The previous implementation flushed a
            // BufWriter, which only pushes into the page cache; forcing durable
            // media writes would be a different guarantee and a much slower one,
            // and it is not what the libraries this is measured against do.
            Ok((total_rows, numeric_cols_to_scale, has_header_actual))
        })
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("input_path", input_path)?;
    out.set_item("output_path", output_path)?;
    out.set_item("rows_written", rows_written)?;
    out.set_item("numeric_cols_scaled", numeric_cols_scaled)?;
    out.set_item("has_header", has_header_detected)?;
    Ok(out.into())
}

/// Read `params` as a mapping of column name to two named f64 fields.
fn affine_params_from_dict(
    params: &Bound<'_, PyDict>,
    offset_key: &str,
    scale_key: &str,
) -> PyResult<AHashMap<String, (f64, f64)>> {
    let mut out: AHashMap<String, (f64, f64)> = AHashMap::new();
    for (k, v) in params.iter() {
        let col_name: String = k.extract()?;
        let v_dict = v.downcast::<PyDict>()?;
        let offset: f64 = v_dict
            .get_item(offset_key)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "column {col_name:?} is missing {offset_key:?}"
                ))
            })?
            .extract()?;
        let scale: f64 = v_dict
            .get_item(scale_key)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "column {col_name:?} is missing {scale_key:?}"
                ))
            })?
            .extract()?;
        out.insert(col_name, (offset, scale));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, params, delimiter=None, has_header=None))]
fn csv_transform_minmax(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    params: Bound<'_, PyDict>,
    delimiter: Option<String>,
    has_header: Option<bool>,
) -> PyResult<PyObject> {
    // (x - min) / (max - min)
    let mut params_map = affine_params_from_dict(&params, "min", "max")?;
    for value in params_map.values_mut() {
        value.1 -= value.0;
    }
    transform_affine(
        py,
        input_path,
        output_path,
        params_map,
        delimiter,
        has_header,
    )
}

/// Mean and population standard deviation per numeric column, from one pass.
#[pyfunction]
fn csv_standardize_params(py: Python<'_>, path: String, sample_size: usize) -> PyResult<PyObject> {
    // Moments-only scan: mean and std come from the streaming accumulator,
    // and the quantiles a lite scan would compute are discarded here anyway.
    let prof_obj = csv_profile_impl(
        py,
        path.clone(),
        sample_size,
        0,
        true,
        ScanMode::Moments,
        false,
        false,
    )?;
    let prof = prof_obj.bind(py).downcast::<PyDict>()?;
    let out = PyDict::new(py);
    out.set_item("path", path)?;
    let params = PyDict::new(py);

    let cols_any = prof
        .get_item("columns")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing columns"))?;
    let cols_list = cols_any.downcast::<PyList>()?;

    for c in cols_list.iter() {
        let cd = c.downcast::<PyDict>()?;
        let name: String = cd
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing name"))?
            .extract()?;

        let (Some(mean_v), Some(std_v)) = (cd.get_item("mean")?, cd.get_item("std")?) else {
            continue;
        };
        if mean_v.is_none() || std_v.is_none() {
            continue;
        }
        let d = PyDict::new(py);
        d.set_item("mean", &mean_v)?;
        d.set_item("std", &std_v)?;
        params.set_item(name, d)?;
    }

    out.set_item("params", params)?;
    Ok(out.into())
}

/// Standardize numeric columns to zero mean and unit variance, streaming.
#[pyfunction]
#[pyo3(signature = (input_path, output_path, params, delimiter=None, has_header=None))]
fn csv_transform_standardize(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    params: Bound<'_, PyDict>,
    delimiter: Option<String>,
    has_header: Option<bool>,
) -> PyResult<PyObject> {
    // (x - mean) / std
    let params_map = affine_params_from_dict(&params, "mean", "std")?;
    transform_affine(
        py,
        input_path,
        output_path,
        params_map,
        delimiter,
        has_header,
    )
}

// Python object schema detection (compatibility path).

#[derive(Default, Clone)]
struct PyColStats {
    count: u64,
    null_count: u64,
    types: BTreeSet<String>,
    examples: Vec<String>,
}

fn type_name(v: &Bound<'_, PyAny>) -> String {
    if v.is_none() {
        return "null".to_string();
    }
    if v.extract::<bool>().is_ok() {
        return "bool".to_string();
    }
    if v.extract::<i64>().is_ok() {
        return "int".to_string();
    }
    if v.extract::<f64>().is_ok() {
        return "float".to_string();
    }
    if v.downcast::<PyString>().is_ok() {
        return "string".to_string();
    }
    match v.get_type().name() {
        Ok(name) => name.to_string(),
        Err(_) => "unknown".to_string(),
    }
}

fn is_string_like(v: &Bound<'_, PyAny>) -> bool {
    v.is_instance_of::<PyString>()
}

fn is_bytes_like(v: &Bound<'_, PyAny>) -> bool {
    v.extract::<&[u8]>().is_ok() || v.downcast::<pyo3::types::PyByteArray>().is_ok()
}

fn is_sequence_like(v: &Bound<'_, PyAny>) -> bool {
    if is_string_like(v) || is_bytes_like(v) {
        return false;
    }
    v.downcast::<PyList>().is_ok()
        || v.downcast::<PyTuple>().is_ok()
        || v.downcast::<PySequence>().is_ok()
}

fn add_example(_py: Python<'_>, col: &mut PyColStats, v: &Bound<'_, PyAny>, max_examples: usize) {
    if col.examples.len() >= max_examples {
        return;
    }
    if let Ok(r) = v.repr() {
        if let Ok(s) = r.extract::<String>() {
            col.examples.push(s);
        }
    }
}

/// Maximum nesting depth `flatten_value` will descend into.
///
/// This is a stack-safety bound, not a stylistic one. `flatten_value` recurses
/// once per level of nesting, so without a cap a sufficiently deep input
/// overflows the Rust stack and takes the whole interpreter down with SIGSEGV
/// -- not a catchable Python exception. The observed crash threshold is around
/// 20k-30k levels; 512 is far above any real-world nested record and far below
/// the danger zone.
const MAX_NESTING_DEPTH: usize = 512;

#[allow(clippy::too_many_arguments)]
fn flatten_value(
    py: Python<'_>,
    cols: &mut BTreeMap<String, PyColStats>,
    path: &str,
    v: &Bound<'_, PyAny>,
    sample_budget: &mut i64,
    max_examples: usize,
    depth: usize,
    depth_exceeded: &mut bool,
) -> PyResult<()> {
    if *sample_budget <= 0 {
        return Ok(());
    }
    if depth >= MAX_NESTING_DEPTH {
        // Stop descending rather than overflow the stack. The caller reports
        // this to Python so the truncation is visible instead of silent.
        *depth_exceeded = true;
        *sample_budget -= 1;
        // Still record the node, so the caller sees where truncation happened
        // rather than an empty schema. Deliberately does NOT call add_example:
        // repr() of a deeply nested object recurses inside CPython and would
        // reintroduce the very stack overflow this guard exists to prevent.
        let col_key = if path.is_empty() {
            "value".to_string()
        } else {
            path.to_string()
        };
        let entry = cols.entry(col_key).or_default();
        entry.count += 1;
        if v.is_none() {
            entry.null_count += 1;
        }
        entry.types.insert(type_name(v));
        return Ok(());
    }
    *sample_budget -= 1;

    if let Ok(mapping) = v.downcast::<PyDict>() {
        for (k, vv) in mapping.iter() {
            let key = k.str()?.to_str()?.to_string();
            let p = if path.is_empty() {
                key
            } else {
                format!("{path}.{key}")
            };
            flatten_value(
                py,
                cols,
                &p,
                &vv,
                sample_budget,
                max_examples,
                depth + 1,
                depth_exceeded,
            )?;
        }
        return Ok(());
    }

    if let Ok(mapping) = v.downcast::<PyMapping>() {
        let items = mapping.items()?;
        for item in items.iter() {
            let tup = item.downcast::<PyTuple>()?;
            if tup.len() != 2 {
                continue;
            }
            let k = tup.get_item(0)?;
            let vv = tup.get_item(1)?;
            let key = k.str()?.to_str()?.to_string();
            let p = if path.is_empty() {
                key
            } else {
                format!("{path}.{key}")
            };
            flatten_value(
                py,
                cols,
                &p,
                &vv,
                sample_budget,
                max_examples,
                depth + 1,
                depth_exceeded,
            )?;
        }
        return Ok(());
    }

    if is_sequence_like(v) {
        let p = if path.is_empty() {
            "[]".to_string()
        } else {
            format!("{path}[]")
        };
        if let Ok(list) = v.downcast::<PyList>() {
            for i in 0..list.len().min(50) {
                let item = list.get_item(i)?;
                flatten_value(
                    py,
                    cols,
                    &p,
                    &item,
                    sample_budget,
                    max_examples,
                    depth + 1,
                    depth_exceeded,
                )?;
                if *sample_budget <= 0 {
                    break;
                }
            }
            return Ok(());
        }
        if let Ok(tup) = v.downcast::<PyTuple>() {
            for i in 0..tup.len().min(50) {
                let item = tup.get_item(i)?;
                flatten_value(
                    py,
                    cols,
                    &p,
                    &item,
                    sample_budget,
                    max_examples,
                    depth + 1,
                    depth_exceeded,
                )?;
                if *sample_budget <= 0 {
                    break;
                }
            }
            return Ok(());
        }
        if let Ok(iter) = PyIterator::from_object(v) {
            for (idx, item) in iter.enumerate() {
                if idx >= 50 {
                    break;
                }
                let item = item?;
                flatten_value(
                    py,
                    cols,
                    &p,
                    &item,
                    sample_budget,
                    max_examples,
                    depth + 1,
                    depth_exceeded,
                )?;
                if *sample_budget <= 0 {
                    break;
                }
            }
            return Ok(());
        }
    }

    let col_key = if path.is_empty() {
        "value".to_string()
    } else {
        path.to_string()
    };
    let entry = cols.entry(col_key).or_default();
    entry.count += 1;
    if v.is_none() {
        entry.null_count += 1;
    }
    entry.types.insert(type_name(v));
    add_example(py, entry, v, max_examples);
    Ok(())
}

fn infer_best_type(types: &BTreeSet<String>) -> String {
    if types.is_empty() {
        return "unknown".to_string();
    }
    let non_null: Vec<&str> = types
        .iter()
        .map(|s| s.as_str())
        .filter(|x| *x != "null")
        .collect();
    if non_null.is_empty() {
        return "null".to_string();
    }
    if non_null.len() == 1 {
        return non_null[0].to_string();
    }
    let set: BTreeSet<&str> = non_null.iter().copied().collect();
    if set.iter().all(|x| *x == "int" || *x == "float") {
        return "float".to_string();
    }
    "mixed".to_string()
}

#[pyfunction]
fn detect_schema(
    py: Python<'_>,
    data: Bound<'_, PyAny>,
    sample_size: usize,
    max_examples: usize,
) -> PyResult<PyObject> {
    let mut cols: BTreeMap<String, PyColStats> = BTreeMap::new();
    let mut budget = sample_size as i64;
    let mut depth_exceeded = false;

    if is_sequence_like(&data) {
        if let Ok(list) = data.downcast::<PyList>() {
            for i in 0..list.len().min(sample_size) {
                let item = list.get_item(i)?;
                flatten_value(
                    py,
                    &mut cols,
                    "",
                    &item,
                    &mut budget,
                    max_examples,
                    0,
                    &mut depth_exceeded,
                )?;
                if budget <= 0 {
                    break;
                }
            }
        } else if let Ok(tup) = data.downcast::<PyTuple>() {
            for i in 0..tup.len().min(sample_size) {
                let item = tup.get_item(i)?;
                flatten_value(
                    py,
                    &mut cols,
                    "",
                    &item,
                    &mut budget,
                    max_examples,
                    0,
                    &mut depth_exceeded,
                )?;
                if budget <= 0 {
                    break;
                }
            }
        } else if let Ok(iter) = PyIterator::from_object(&data) {
            for (idx, item) in iter.enumerate() {
                if idx >= sample_size {
                    break;
                }
                let item = item?;
                flatten_value(
                    py,
                    &mut cols,
                    "",
                    &item,
                    &mut budget,
                    max_examples,
                    0,
                    &mut depth_exceeded,
                )?;
                if budget <= 0 {
                    break;
                }
            }
        } else {
            flatten_value(
                py,
                &mut cols,
                "",
                &data,
                &mut budget,
                max_examples,
                0,
                &mut depth_exceeded,
            )?;
        }
    } else {
        flatten_value(
            py,
            &mut cols,
            "",
            &data,
            &mut budget,
            max_examples,
            0,
            &mut depth_exceeded,
        )?;
    }

    let out = PyDict::new(py);
    let col_list = PyList::empty(py);
    for (path, stats) in cols.iter() {
        let d = PyDict::new(py);
        d.set_item("path", path)?;
        d.set_item("count", stats.count)?;
        d.set_item("null_count", stats.null_count)?;
        d.set_item("types", stats.types.iter().cloned().collect::<Vec<_>>())?;
        d.set_item("inferred", infer_best_type(&stats.types))?;
        d.set_item("examples", &stats.examples)?;
        col_list.append(d)?;
    }
    out.set_item("columns", col_list)?;
    out.set_item("sample_size", sample_size)?;
    // Report truncation rather than silently returning a partial schema.
    out.set_item("max_depth_exceeded", depth_exceeded)?;
    out.set_item("max_depth", MAX_NESTING_DEPTH)?;
    Ok(out.into())
}

mod parse;
use parse::*;
mod stats;
use stats::*;
mod regression;
use regression::*;

// Streaming SGD regression.

#[cfg(test)]
mod tests;

// Module registration.

/// Panic on purpose, to verify how panics cross the FFI boundary.
///
/// PyO3 wraps every #[pyfunction] body in catch_unwind and converts a caught
/// panic into a Python PanicException. That only works if panics unwind: under
/// `panic = "abort"` the process dies instead, and no Python-level handler ever
/// runs. Gated behind the `testing` feature so it never ships.
#[cfg(feature = "testing")]
#[pyfunction]
fn _force_panic() {
    panic!("deliberate panic: verifying panic propagation across the FFI boundary");
}

#[pymodule]
fn _grizzly(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "testing")]
    m.add_function(wrap_pyfunction!(_force_panic, m)?)?;
    m.add_function(wrap_pyfunction!(detect_schema, m)?)?;
    m.add_function(wrap_pyfunction!(csv_profile, m)?)?;
    m.add_function(wrap_pyfunction!(csv_minmax_params, m)?)?;
    m.add_function(wrap_pyfunction!(csv_transform_minmax, m)?)?;
    m.add_function(wrap_pyfunction!(csv_standardize_params, m)?)?;
    m.add_function(wrap_pyfunction!(csv_transform_standardize, m)?)?;
    m.add_function(wrap_pyfunction!(csv_linear_regression, m)?)?;
    m.add_function(wrap_pyfunction!(csv_sgd_regression, m)?)?;
    m.add_function(wrap_pyfunction!(csv_logistic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(csv_gaussian_nb, m)?)?;
    m.add_function(wrap_pyfunction!(csv_classification_metrics, m)?)?;
    Ok(())
}

// Rust-native ML: Linear Regression on CSV (no NumPy).
