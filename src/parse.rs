//! Byte-level CSV parsing and detection primitives.
//!
//! Everything in this module operates on plain byte slices and depends only on
//! parsing crates -- no PyO3, no Python interpreter. That separation is what
//! makes the parser reachable from a fuzz target: the crate as a whole is a
//! cdylib linked against libpython and cannot be linked into a standalone
//! binary, but this module can be compiled on its own.
//!
//! It is also the code that handles untrusted input. Everything here indexes
//! into caller-supplied bytes, so it is the natural place for a fuzzer to look
//! for slice out-of-bounds panics and non-terminating loops.

use memchr::{memchr, memchr2, memchr3};

#[inline(always)]
pub fn trim_bytes(bytes: &[u8]) -> &[u8] {
    // FIX: Single-pass trim (not position + rposition)
    let mut start = 0;
    let mut end = bytes.len();

    // Trim start
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }

    // Trim end
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    &bytes[start..end]
}

pub fn sniff_delimiter_simd(bytes: &[u8]) -> Option<u8> {
    if memchr(b',', bytes).is_some() {
        return Some(b',');
    }
    if memchr(b'\t', bytes).is_some() {
        return Some(b'\t');
    }
    if memchr(b';', bytes).is_some() {
        return Some(b';');
    }
    if memchr(b'|', bytes).is_some() {
        return Some(b'|');
    }
    None
}

pub fn has_alphabetic(bytes: &[u8]) -> bool {
    bytes.iter().any(|&b| b.is_ascii_alphabetic())
}

/// Better header detection: compare numeric parse rate of row 0 vs rows 1-N.
/// If row 0 is mostly non-numeric and row 1+ are mostly numeric, it's likely a header.
pub fn detect_header_smart(bytes: &[u8], delim: u8, num_sample_rows: usize) -> bool {
    let mut lines = bytes.split(|&b| b == b'\n').take(num_sample_rows + 1);

    let first_line = match lines.next() {
        Some(l) => l,
        None => return false,
    };

    // Count numeric fields in first row
    let first_row_fields: Vec<&[u8]> = first_line.split(|&b| b == delim).collect();
    if first_row_fields.is_empty() {
        return false;
    }

    let first_numeric_rate = first_row_fields
        .iter()
        .filter(|f| {
            let trimmed = trim_bytes(f);
            !trimmed.is_empty()
                && (atoi_simd::parse::<i64>(trimmed).is_ok()
                    || fast_float::parse::<f64, _>(trimmed).is_ok())
        })
        .count() as f64
        / first_row_fields.len() as f64;

    // Count numeric fields in subsequent rows
    let mut total_fields = 0usize;
    let mut numeric_fields = 0usize;

    for line in lines {
        if line.is_empty() {
            continue;
        }
        for field in line.split(|&b| b == delim) {
            let trimmed = trim_bytes(field);
            total_fields += 1;
            if !trimmed.is_empty()
                && (atoi_simd::parse::<i64>(trimmed).is_ok()
                    || fast_float::parse::<f64, _>(trimmed).is_ok())
            {
                numeric_fields += 1;
            }
        }
    }

    if total_fields == 0 {
        return has_alphabetic(first_line);
    }

    let data_numeric_rate = numeric_fields as f64 / total_fields as f64;

    // Header heuristic:
    // - First row has low numeric rate (< 30%)
    // - Data rows have higher numeric rate (> 50%), OR
    // - First row is significantly less numeric than data rows

    (first_numeric_rate < 0.3 && data_numeric_rate > 0.3)
        || (data_numeric_rate - first_numeric_rate > 0.3)
        || (first_numeric_rate < 0.1 && has_alphabetic(first_line))
}

#[inline(always)]
pub fn is_integer_bytes(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && (bytes[0] == b'-' || bytes[0] == b'+' || bytes[0].is_ascii_digit())
        && !bytes.iter().any(|&b| b == b'.' || b == b'e' || b == b'E')
}

/// Fast-reject non-float bytes before attempting an expensive float parse.
#[inline(always)]
pub fn maybe_float_bytes(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    // Must start with digit, sign, or decimal
    let first = bytes[0];
    if !(first.is_ascii_digit() || first == b'-' || first == b'+' || first == b'.') {
        return false;
    }

    // Check if it contains float indicators (., e, E) or is all digits
    // Using memchr2 for SIMD acceleration
    memchr2(b'.', b'e', bytes).is_some()
        || memchr(b'E', bytes).is_some()
        || bytes
            .iter()
            .all(|&b| b.is_ascii_digit() || b == b'-' || b == b'+')
}

/// Splits a raw byte slice into chunks, aligning to newlines.
/// Uses Polars strategy: 16MB max, 4KB min chunks.
///
/// ⚠️ **CORRECTNESS LIMITATION**: This assumes no quoted newlines in CSV fields.
/// If your CSV contains quoted fields like `"hello\nworld",123`, this will
/// split in the middle of the field, causing parse errors.
///
/// **Fast path assumption**: "Clean" CSVs with no quoted newlines.
/// For full CSV correctness, use a producer/consumer pipeline with sequential reading.
pub fn chunk_bytes_aligned(bytes: &[u8], num_threads: usize) -> Vec<&[u8]> {
    const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024; // 16MB (Polars pattern)
    const MIN_CHUNK_SIZE: usize = 4 * 1024; // 4KB

    let len = bytes.len();
    if len == 0 || num_threads == 0 {
        return vec![];
    }

    // Use a Polars-like chunk sizing strategy.
    let optimal_chunk_size = len / (16 * num_threads);
    let chunk_size = optimal_chunk_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);

    let mut ranges = Vec::new();
    let mut start = 0;

    while start < len {
        let mut end = std::cmp::min(start + chunk_size, len);

        // Align 'end' to the next newline to avoid splitting rows
        // WARNING: This can still split quoted fields with embedded newlines
        if end < len {
            if let Some(newline_pos) = memchr(b'\n', &bytes[end..]) {
                end += newline_pos + 1;
            } else {
                end = len;
            }
        }

        if end > start {
            ranges.push(&bytes[start..end]);
        }
        start = end;
    }
    ranges
}

/// Split mode for CSV parsing
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum SplitMode {
    Delim(u8),  // Split by specific delimiter
    Whitespace, // Split by whitespace runs (like pandas sep=r"\s+")
}

/// Iterate fields in a line using a single-byte delimiter.
#[inline(always)]
pub fn iter_fields_delim(line: &[u8], delim: u8) -> impl Iterator<Item = &[u8]> {
    line.split(move |&b| b == delim)
}

/// Iterate fields in a line by splitting on whitespace runs
#[inline(always)]
pub fn iter_fields_ws(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(|b: &u8| b.is_ascii_whitespace())
        .filter(|f| !f.is_empty())
}

/// Get fields from a line using the appropriate split mode
pub fn get_fields(line: &[u8], mode: SplitMode) -> Vec<&[u8]> {
    match mode {
        SplitMode::Delim(d) => iter_fields_delim(line, d).collect(),
        SplitMode::Whitespace => iter_fields_ws(line).collect(),
    }
}

/// Check whether a field needs CSV quoting.
#[inline(always)]
pub fn needs_quote_simd(field: &[u8], delim: u8) -> bool {
    memchr3(delim, b'"', b'\n', field).is_some() || memchr(b'\r', field).is_some()
}

/// Write a field with proper CSV quoting if needed
#[inline(always)]
pub fn write_field_csv(output: &mut Vec<u8>, field: &[u8], delim: u8) {
    if !needs_quote_simd(field, delim) {
        output.extend_from_slice(field);
        return;
    }

    // Needs quoting
    output.push(b'"');
    // Use memchr to find quotes efficiently.
    let mut start = 0;
    while let Some(pos) = memchr(b'"', &field[start..]) {
        let i = start + pos;
        output.extend_from_slice(&field[start..=i]); // Include the quote
        output.push(b'"'); // Double it
        start = i + 1;
    }
    output.extend_from_slice(&field[start..]);
    output.push(b'"');
}

/// Yields raw lines, without splitting or allocating.
///
/// `FastRowIter` below is convenient when a row needs random access by column
/// index, but it allocates a `Vec` per row to provide it. Streaming work that
/// visits fields in order -- the transform writer, most obviously -- wants
/// neither the Vec nor the allocation, and pairs this with `for_each_field`.
/// At 500k rows a transform was paying half a million allocations for nothing.
///
/// ⚠️ Assumes no quoted newlines (fast-CSV mode only).
pub struct FastLineIter<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FastLineIter<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}

impl<'a> Iterator for FastLineIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // Loops rather than recursing on blank lines: a file with a long run of
        // them would otherwise blow the stack.
        while self.pos < self.bytes.len() {
            let line_end = memchr(b'\n', &self.bytes[self.pos..])
                .map(|i| self.pos + i)
                .unwrap_or(self.bytes.len());

            let line_len = if line_end > self.pos && self.bytes.get(line_end - 1) == Some(&b'\r') {
                line_end - 1
            } else {
                line_end
            };

            let line = &self.bytes[self.pos..line_len];
            self.pos = line_end + 1;

            if !line.is_empty() {
                return Some(line);
            }
        }
        None
    }
}

/// Visit each field of a line in order, without allocating.
#[inline]
pub fn for_each_field<F: FnMut(usize, &[u8])>(line: &[u8], mode: SplitMode, mut visit: F) {
    match mode {
        SplitMode::Delim(d) => {
            for (i, field) in iter_fields_delim(line, d).enumerate() {
                visit(i, field);
            }
        }
        SplitMode::Whitespace => {
            for (i, field) in iter_fields_ws(line).enumerate() {
                visit(i, field);
            }
        }
    }
}

/// Fast row iterator - splits on newlines, yields field slices.
///
/// Allocates a `Vec` per row, which buys random access by column index. Use it
/// where that access pattern is needed (selecting a target column, say) and
/// `FastLineIter` + `for_each_field` where fields are consumed in order.
///
/// Defined in terms of `FastLineIter` deliberately. It previously duplicated
/// the line-scanning logic and got it wrong: an empty line made `next()` return
/// `None`, which ends iteration rather than skipping the line, so a single
/// blank line anywhere in a file silently truncated everything after it. On a
/// 1,000-row file with one stray blank line, regression and SGD trained on 500
/// rows and reported nothing unusual. Sharing one implementation means the two
/// cannot disagree about what a row is.
///
/// ⚠️ Assumes no quoted newlines (fast-CSV mode only).
pub struct FastRowIter<'a> {
    lines: FastLineIter<'a>,
    split_mode: SplitMode,
}

impl<'a> FastRowIter<'a> {
    pub fn new(bytes: &'a [u8], split_mode: SplitMode) -> Self {
        Self {
            lines: FastLineIter::new(bytes),
            split_mode,
        }
    }
}

impl<'a> Iterator for FastRowIter<'a> {
    type Item = Vec<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lines
            .next()
            .map(|line| get_fields(line, self.split_mode))
    }
}
