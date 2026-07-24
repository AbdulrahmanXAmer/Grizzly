//! Unit tests for the pure internals of the profiler.
//!
//! These cover the parts that can be tested without a Python interpreter: byte
//! utilities, type inference, chunk alignment, and the streaming statistics.
//! Everything here is deterministic and runs under `cargo test`.
//!
//! The parallel profiling path splits input into chunks, accumulates a
//! `ColStats` per chunk, and merges them. That merge is the subtle part -- a
//! wrong merge produces plausible-looking but incorrect statistics that no
//! smoke test would catch -- so it gets the most attention below.

use super::*;

// ---------------------------------------------------------------------------
// byte utilities
// ---------------------------------------------------------------------------

#[test]
fn trim_bytes_strips_ascii_whitespace_both_ends() {
    assert_eq!(trim_bytes(b"  hello  "), b"hello");
    assert_eq!(trim_bytes(b"\t\r\n x \n\r\t"), b"x");
    assert_eq!(trim_bytes(b"none"), b"none");
}

#[test]
fn trim_bytes_handles_degenerate_input() {
    assert_eq!(trim_bytes(b""), b"");
    assert_eq!(trim_bytes(b"   "), b"");
    // Interior whitespace is preserved; only the ends are trimmed.
    assert_eq!(trim_bytes(b"  a b  "), b"a b");
}

#[test]
fn sniff_delimiter_prefers_comma_then_tab_then_semicolon_then_pipe() {
    assert_eq!(sniff_delimiter_simd(b"a,b,c"), Some(b','));
    assert_eq!(sniff_delimiter_simd(b"a\tb\tc"), Some(b'\t'));
    assert_eq!(sniff_delimiter_simd(b"a;b;c"), Some(b';'));
    assert_eq!(sniff_delimiter_simd(b"a|b|c"), Some(b'|'));

    // Precedence is fixed, not frequency-based: a single comma beats many tabs.
    assert_eq!(sniff_delimiter_simd(b"a\tb\tc,d"), Some(b','));
    assert_eq!(sniff_delimiter_simd(b"a;b|c\td"), Some(b'\t'));
}

#[test]
fn sniff_delimiter_returns_none_when_no_candidate_present() {
    assert_eq!(sniff_delimiter_simd(b"single_column"), None);
    assert_eq!(sniff_delimiter_simd(b""), None);
    // Whitespace-delimited files have no recognised delimiter byte.
    assert_eq!(sniff_delimiter_simd(b"1.0 2.0 3.0"), None);
}

// ---------------------------------------------------------------------------
// numeric fast paths
// ---------------------------------------------------------------------------

#[test]
fn is_integer_bytes_accepts_signed_digits_only() {
    assert!(is_integer_bytes(b"123"));
    assert!(is_integer_bytes(b"-123"));
    assert!(is_integer_bytes(b"+123"));

    assert!(!is_integer_bytes(b""));
    assert!(!is_integer_bytes(b"1.0"), "decimal point disqualifies");
    assert!(!is_integer_bytes(b"1e5"), "exponent disqualifies");
    assert!(!is_integer_bytes(b"1E5"), "capital exponent disqualifies");
    assert!(!is_integer_bytes(b"abc"));
}

#[test]
fn maybe_float_bytes_is_a_conservative_prefilter() {
    // It must never reject something that really is a float, or the profiler
    // would silently misclassify the column.
    for value in [
        &b"1.0"[..],
        b"-1.0",
        b"+1.0",
        b".5",
        b"1e5",
        b"1E5",
        b"1.5e-3",
        b"123",
        b"-123",
    ] {
        assert!(
            maybe_float_bytes(value),
            "rejected parseable float: {:?}",
            std::str::from_utf8(value)
        );
        assert!(
            fast_float::parse::<f64, _>(value).is_ok(),
            "test fixture is not actually a float: {:?}",
            std::str::from_utf8(value)
        );
    }
}

#[test]
fn maybe_float_bytes_rejects_obvious_non_numbers() {
    assert!(!maybe_float_bytes(b""));
    assert!(!maybe_float_bytes(b"abc"));
    assert!(!maybe_float_bytes(b"true"));
    assert!(!maybe_float_bytes(b"NaN"), "leading letter is rejected");
}

// ---------------------------------------------------------------------------
// type inference
// ---------------------------------------------------------------------------

#[test]
fn mask_to_types_lists_every_set_flag_in_declaration_order() {
    assert_eq!(mask_to_types(0), Vec::<&str>::new());
    assert_eq!(mask_to_types(DType::Int as u8), vec!["int"]);
    assert_eq!(
        mask_to_types(DType::Null as u8 | DType::Int as u8),
        vec!["null", "int"]
    );
    assert_eq!(
        mask_to_types(
            DType::Null as u8
                | DType::Bool as u8
                | DType::Int as u8
                | DType::Float as u8
                | DType::String as u8
        ),
        vec!["null", "bool", "int", "float", "string"]
    );
}

#[test]
fn infer_from_mask_ignores_nulls_when_choosing_a_type() {
    // A column of ints with some blanks is still an int column.
    assert_eq!(infer_from_mask(DType::Int as u8), "int");
    assert_eq!(infer_from_mask(DType::Null as u8 | DType::Int as u8), "int");
    assert_eq!(
        infer_from_mask(DType::Null as u8 | DType::String as u8),
        "string"
    );
}

#[test]
fn infer_from_mask_widens_int_and_float_to_float() {
    assert_eq!(
        infer_from_mask(DType::Int as u8 | DType::Float as u8),
        "float"
    );
    assert_eq!(
        infer_from_mask(DType::Null as u8 | DType::Int as u8 | DType::Float as u8),
        "float"
    );
}

#[test]
fn infer_from_mask_reports_mixed_for_incompatible_combinations() {
    assert_eq!(
        infer_from_mask(DType::Int as u8 | DType::String as u8),
        "mixed"
    );
    assert_eq!(
        infer_from_mask(DType::Bool as u8 | DType::Int as u8),
        "mixed"
    );
}

#[test]
fn infer_from_mask_reports_null_when_nothing_else_was_seen() {
    assert_eq!(infer_from_mask(0), "null");
    assert_eq!(infer_from_mask(DType::Null as u8), "null");
}

// ---------------------------------------------------------------------------
// header detection
// ---------------------------------------------------------------------------

#[test]
fn detect_header_smart_finds_a_text_header_over_numeric_rows() {
    let csv = b"alpha,beta,gamma\n1,2,3\n4,5,6\n7,8,9\n";
    assert!(detect_header_smart(csv, b',', 3));
}

#[test]
fn detect_header_smart_rejects_all_numeric_input() {
    let csv = b"1,2,3\n4,5,6\n7,8,9\n";
    assert!(!detect_header_smart(csv, b',', 3));
}

#[test]
fn detect_header_smart_handles_a_single_line() {
    // Only a header row and no data: fall back to "does it look like words".
    assert!(detect_header_smart(b"alpha,beta,gamma", b',', 3));
    assert!(!detect_header_smart(b"1,2,3", b',', 3));
}

#[test]
fn detect_header_smart_handles_empty_input() {
    assert!(!detect_header_smart(b"", b',', 3));
}

#[test]
fn detect_header_smart_recognises_a_text_column_alongside_numbers() {
    // A genuine header over data whose first column is categorical. The header
    // row is entirely non-numeric while data rows are two-thirds numeric.
    let csv = b"name,x,y\nalpha,1,2\nbravo,3,4\ncharlie,5,6\n";
    assert!(detect_header_smart(csv, b',', 3));
}

// ---------------------------------------------------------------------------
// chunk alignment
// ---------------------------------------------------------------------------

#[test]
fn chunk_bytes_aligned_covers_the_input_exactly_once() {
    // Chunks must partition the input: any gap silently drops rows and any
    // overlap double-counts them, either of which corrupts the statistics.
    let data: Vec<u8> = (0..50_000)
        .flat_map(|i| format!("{i},{i}\n").into_bytes())
        .collect();

    for threads in [1usize, 2, 4, 8] {
        let chunks = chunk_bytes_aligned(&data, threads);
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(
            total,
            data.len(),
            "chunks did not cover the input exactly (threads={threads})"
        );

        let rejoined: Vec<u8> = chunks.concat();
        assert_eq!(rejoined, data, "chunks did not rejoin to the original");
    }
}

#[test]
fn chunk_bytes_aligned_never_splits_a_line() {
    let data: Vec<u8> = (0..50_000)
        .flat_map(|i| format!("{i},{i}\n").into_bytes())
        .collect();

    let chunks = chunk_bytes_aligned(&data, 8);
    for chunk in &chunks[..chunks.len().saturating_sub(1)] {
        assert_eq!(
            *chunk.last().unwrap(),
            b'\n',
            "chunk boundary landed mid-line"
        );
    }

    // Every chunk parses into whole records, so the row count is preserved.
    let rows: usize = chunks
        .iter()
        .map(|c| c.split(|&b| b == b'\n').filter(|l| !l.is_empty()).count())
        .sum();
    assert_eq!(rows, 50_000);
}

#[test]
fn chunk_bytes_aligned_handles_degenerate_input() {
    assert!(chunk_bytes_aligned(b"", 4).is_empty());
    assert!(chunk_bytes_aligned(b"data", 0).is_empty());

    // Input without a trailing newline still round-trips.
    let data = b"a,b\n1,2";
    assert_eq!(chunk_bytes_aligned(data, 4).concat(), data);
}

// ---------------------------------------------------------------------------
// streaming statistics
// ---------------------------------------------------------------------------

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{what}: expected {expected}, got {actual}"
    );
}

#[test]
fn numstats_tracks_count_min_max_and_mean() {
    let mut stats = NumStats::default();
    for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
        stats.push(x);
    }
    stats.finalize();

    assert_eq!(stats.n, 5);
    assert_close(stats.min, 1.0, 1e-12, "min");
    assert_close(stats.max, 5.0, 1e-12, "max");
    assert_close(stats.mean, 3.0, 1e-12, "mean");
}

#[test]
fn numstats_computes_a_population_standard_deviation() {
    let mut stats = NumStats::default();
    for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
        stats.push(x);
    }
    stats.finalize();

    // Population std of this textbook set is exactly 2.0 (sample std is ~2.138),
    // which is what distinguishes the two conventions.
    assert_close(stats.std_pop().unwrap(), 2.0, 1e-12, "population std");
}

#[test]
fn numstats_reports_nothing_for_an_empty_column() {
    let stats = NumStats::default();
    assert_eq!(stats.n, 0);
    assert!(stats.std_pop().is_none());
    assert!(stats.quantile(0.5).is_none());
}

#[test]
fn numstats_survives_more_values_than_the_pending_batch_size() {
    // push() flushes into the t-digest every 1024 values; cross that boundary
    // to exercise the flush path rather than only the buffered one.
    let mut stats = NumStats::default();
    for i in 0..5_000 {
        stats.push(i as f64);
    }
    stats.finalize();

    assert_eq!(stats.n, 5_000);
    assert_close(stats.min, 0.0, 1e-9, "min");
    assert_close(stats.max, 4_999.0, 1e-9, "max");
    assert_close(stats.mean, 2_499.5, 1e-9, "mean");
}

#[test]
fn numstats_merge_matches_a_single_pass_over_the_same_values() {
    // This is the invariant the parallel profiler depends on: splitting the
    // data across chunks and merging must equal profiling it in one pass.
    let values: Vec<f64> = (0..4_000).map(|i| (i as f64) * 0.5 - 500.0).collect();

    let mut single = NumStats::default();
    for &x in &values {
        single.push(x);
    }
    single.finalize();

    let (left_values, right_values) = values.split_at(1_500);
    let mut left = NumStats::default();
    for &x in left_values {
        left.push(x);
    }
    let mut right = NumStats::default();
    for &x in right_values {
        right.push(x);
    }
    left.merge(&mut right);

    assert_eq!(left.n, single.n, "merged count");
    assert_close(left.min, single.min, 1e-9, "merged min");
    assert_close(left.max, single.max, 1e-9, "merged max");
    assert_close(left.mean, single.mean, 1e-9, "merged mean");
    assert_close(
        left.std_pop().unwrap(),
        single.std_pop().unwrap(),
        1e-6,
        "merged std",
    );
}

#[test]
fn numstats_merge_handles_an_empty_side() {
    let mut populated = NumStats::default();
    for x in [1.0, 2.0, 3.0] {
        populated.push(x);
    }
    populated.finalize();

    // Merging an empty accumulator in must not disturb the populated one.
    let mut empty = NumStats::default();
    let mut target = populated.clone();
    target.merge(&mut empty);
    assert_eq!(target.n, 3);
    assert_close(target.mean, 2.0, 1e-12, "mean after merging empty");

    // And merging into an empty accumulator must adopt the other side.
    let mut empty_target = NumStats::default();
    let mut source = populated.clone();
    empty_target.merge(&mut source);
    assert_eq!(empty_target.n, 3);
    assert_close(
        empty_target.mean,
        2.0,
        1e-12,
        "mean after merging into empty",
    );
}

#[test]
fn numstats_quantiles_bracket_the_data_range() {
    let mut stats = NumStats::default();
    for i in 0..10_000 {
        stats.push(i as f64);
    }
    stats.finalize();

    let median = stats.quantile(0.5).unwrap();
    assert_close(median, 5_000.0, 100.0, "median of a uniform ramp");

    // Quantiles are approximate (t-digest) but must stay inside the observed
    // range and remain monotonically ordered.
    let q25 = stats.quantile(0.25).unwrap();
    let q75 = stats.quantile(0.75).unwrap();
    assert!(stats.min <= q25, "p25 below observed min");
    assert!(q25 <= median, "p25 above median");
    assert!(median <= q75, "median above p75");
    assert!(q75 <= stats.max, "p75 above observed max");
}

// ---------------------------------------------------------------------------
// frequency tracking
// ---------------------------------------------------------------------------

#[test]
fn freqtracker_finds_the_modal_integer() {
    let mut tracker = FreqTracker::new();
    for value in [1i64, 2, 2, 3, 2, 1] {
        tracker.push_int(value);
    }

    let (mode, count) = tracker.mode_int().unwrap();
    assert_eq!(mode, 2);
    assert_eq!(count, 3);
}

#[test]
fn freqtracker_finds_the_modal_string() {
    let mut tracker = FreqTracker::new();
    for value in [
        &b"alpha"[..],
        b"bravo",
        b"bravo",
        b"charlie",
        b"bravo",
        b"alpha",
    ] {
        tracker.push_string_bytes(value);
    }

    let (mode, count) = tracker.mode_string().unwrap();
    assert_eq!(mode, "bravo");
    assert_eq!(count, 3);
}

#[test]
fn freqtracker_reports_no_mode_for_an_empty_column() {
    let tracker = FreqTracker::new();
    assert!(tracker.mode_int().is_none());
    assert!(tracker.mode_string().is_none());
}

#[test]
fn freqtracker_merge_matches_a_single_pass() {
    // Same parallel-correctness invariant as NumStats::merge.
    let values: Vec<i64> = vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4];

    let mut single = FreqTracker::new();
    for &v in &values {
        single.push_int(v);
    }

    let (left_values, right_values) = values.split_at(5);
    let mut left = FreqTracker::new();
    for &v in left_values {
        left.push_int(v);
    }
    let mut right = FreqTracker::new();
    for &v in right_values {
        right.push_int(v);
    }
    left.merge(&right);

    assert_eq!(left.mode_int(), single.mode_int());
    assert_eq!(left.mode_int(), Some((4, 4)));
}

#[test]
fn freqtracker_merge_combines_counts_split_across_sides() {
    // The winner is only the winner after merging: neither side alone has it.
    let mut left = FreqTracker::new();
    for &v in &[7i64, 7, 1] {
        left.push_int(v);
    }
    let mut right = FreqTracker::new();
    for &v in &[7i64, 7, 2, 2, 2] {
        right.push_int(v);
    }

    assert_eq!(left.mode_int(), Some((7, 2)));
    assert_eq!(right.mode_int(), Some((2, 3)));

    left.merge(&right);
    assert_eq!(left.mode_int(), Some((7, 4)));
}

#[test]
fn freqtracker_distinguishes_strings_with_the_same_length() {
    // Buckets are keyed by (hash, len), so same-length keys exercise the
    // collision-candidate list rather than the fast path.
    let mut tracker = FreqTracker::new();
    for value in [&b"aaa"[..], b"bbb", b"bbb", b"ccc"] {
        tracker.push_string_bytes(value);
    }

    let (mode, count) = tracker.mode_string().unwrap();
    assert_eq!(mode, "bbb");
    assert_eq!(count, 2);
}

// ---------------------------------------------------------------------------
// field splitting
// ---------------------------------------------------------------------------

#[test]
fn get_fields_splits_on_an_explicit_delimiter() {
    let fields = get_fields(b"a,b,c", SplitMode::Delim(b','));
    assert_eq!(fields, vec![&b"a"[..], b"b", b"c"]);

    // Empty fields are preserved, since they represent nulls.
    let fields = get_fields(b"a,,c", SplitMode::Delim(b','));
    assert_eq!(fields, vec![&b"a"[..], b"", b"c"]);
}

#[test]
fn get_fields_collapses_runs_of_whitespace() {
    // Whitespace mode treats any run as one separator, so empty fields cannot
    // occur and alignment padding does not create phantom columns.
    let fields = get_fields(b"a   b\tc", SplitMode::Whitespace);
    assert_eq!(fields, vec![&b"a"[..], b"b", b"c"]);

    let fields = get_fields(b"  a b  ", SplitMode::Whitespace);
    assert_eq!(fields, vec![&b"a"[..], b"b"]);
}

// ---------------------------------------------------------------------------
// linear algebra
// ---------------------------------------------------------------------------

#[test]
fn gaussian_solve_recovers_a_known_solution() {
    // 2x + y = 5, x + 3y = 10  =>  x = 1, y = 3
    let a = vec![2.0, 1.0, 1.0, 3.0];
    let b = vec![5.0, 10.0];

    let x = gaussian_solve(a, b, 2).expect("system is non-singular");
    assert_close(x[0], 1.0, 1e-9, "x");
    assert_close(x[1], 3.0, 1e-9, "y");
}

#[test]
fn gaussian_solve_uses_pivoting_for_a_zero_leading_entry() {
    // A zero in the first pivot position requires a row swap; without pivoting
    // this divides by zero.
    let a = vec![0.0, 1.0, 1.0, 0.0];
    let b = vec![2.0, 3.0];

    let x = gaussian_solve(a, b, 2).expect("system is non-singular after pivoting");
    assert_close(x[0], 3.0, 1e-9, "x");
    assert_close(x[1], 2.0, 1e-9, "y");
}

#[test]
fn gaussian_solve_rejects_a_singular_system() {
    // Second row is twice the first: no unique solution.
    let a = vec![1.0, 2.0, 2.0, 4.0];
    let b = vec![3.0, 6.0];
    assert!(gaussian_solve(a, b, 2).is_none());
}

#[test]
fn splitmix64_is_deterministic_and_mixes_adjacent_inputs() {
    // The train/test split depends on this being reproducible for a given seed.
    assert_eq!(splitmix64(0), splitmix64(0));
    assert_eq!(splitmix64(12_345), splitmix64(12_345));

    // Adjacent seeds must not produce adjacent outputs, or the split would
    // correlate with row order.
    let a = splitmix64(1);
    let b = splitmix64(2);
    assert_ne!(a, b);
    assert!(a.abs_diff(b) > 1_000_000, "adjacent seeds barely differed");
}
