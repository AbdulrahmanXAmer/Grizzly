//! Fuzz the CSV detection and row-splitting path with arbitrary bytes.
//!
//! `csv_profile` accepts a file path and mmaps it, so every byte the parser
//! sees is untrusted and attacker-influenced in any pipeline that profiles
//! uploaded data. The detection helpers all index into that buffer directly,
//! which is exactly where an out-of-bounds slice or a non-terminating loop
//! would live.
//!
//! What this asserts is simply: for *any* input, the parser returns. It must
//! not panic, must not read out of bounds, and must not hang. The fuzzer
//! supplies the adversarial inputs that a hand-written test would not think of
//! -- lone delimiters, unterminated final lines, embedded NULs, invalid UTF-8,
//! and every degenerate mixture of them.

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../src/parse.rs"]
mod parse;

use parse::{
    detect_header_smart, get_fields, has_alphabetic, is_integer_bytes, maybe_float_bytes,
    for_each_field, needs_quote_simd, sniff_delimiter_simd, trim_bytes, write_field_csv,
    FastLineIter, SplitMode,
};

fuzz_target!(|data: &[u8]| {
    // Detection runs before anything knows the file's shape, so it sees the
    // rawest input of all.
    let delimiter = sniff_delimiter_simd(data);
    let delim = delimiter.unwrap_or(b',');

    let _ = has_alphabetic(data);
    let _ = detect_header_smart(data, delim, 8);

    // Row iteration over both split strategies, through the same
    // FastLineIter + for_each_field pair every production reader uses.
    for mode in [SplitMode::Delim(delim), SplitMode::Whitespace] {
        let mut rows = 0usize;
        for line in FastLineIter::new(data) {
            // Bound the work per input so a pathological case is reported as a
            // timeout rather than running forever inside one execution.
            rows += 1;
            if rows > 512 {
                break;
            }

            for_each_field(line, mode, |i, field| {
                if i >= 64 {
                    return;
                }
                let trimmed = trim_bytes(field);

                // Trimming may only ever shrink the slice, and must stay
                // within it. A violation here means the parser could hand a
                // dangling or over-long slice to the statistics layer.
                assert!(trimmed.len() <= field.len());

                let _ = is_integer_bytes(trimmed);
                let _ = maybe_float_bytes(trimmed);
                let _ = needs_quote_simd(field, delim);

                // The writer is on the transform path, where fuzzed bytes come
                // back out to disk; quoting must not panic on any input.
                let mut out = Vec::new();
                write_field_csv(&mut out, field, delim);
            });
        }
    }

    // Line-level splitting, independent of the row iterator.
    for line in data.split(|&b| b == b'\n').take(128) {
        let fields = get_fields(line, SplitMode::Delim(delim));
        // Splitting a line cannot invent bytes.
        let total: usize = fields.iter().map(|f| f.len()).sum();
        assert!(total <= line.len());
    }
});
