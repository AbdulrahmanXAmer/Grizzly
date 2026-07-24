//! Fuzz the parallel chunk splitter for the property the profiler depends on.
//!
//! `chunk_bytes_aligned` divides the input among worker threads, aligning each
//! boundary to a newline so no row is split across two chunks. The profiler
//! then accumulates statistics per chunk and merges them.
//!
//! That makes the correctness condition sharper than "does not panic": the
//! chunks must **partition** the input exactly. A gap silently drops rows, an
//! overlap silently double-counts them, and either produces statistics that
//! look entirely plausible while being wrong -- the failure mode least likely
//! to be noticed in production and least likely to be caught by an
//! example-based test.

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../src/parse.rs"]
mod parse;

use parse::chunk_bytes_aligned;

fuzz_target!(|data: &[u8]| {
    for threads in [1usize, 2, 3, 8, 16] {
        let chunks = chunk_bytes_aligned(data, threads);

        if data.is_empty() || threads == 0 {
            assert!(chunks.is_empty(), "empty input must yield no chunks");
            continue;
        }

        // Exact partition: total length preserved, and the pieces rejoin to
        // the original bytes in order.
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(
            total,
            data.len(),
            "chunks did not cover the input exactly with {threads} threads"
        );

        let rejoined: Vec<u8> = chunks.concat();
        assert_eq!(
            rejoined, data,
            "chunks did not rejoin to the original input with {threads} threads"
        );

        // No empty chunks: they cost a thread and mean the sizing logic slipped.
        for chunk in &chunks {
            assert!(!chunk.is_empty(), "produced an empty chunk");
        }

        // Every boundary except the last must land immediately after a newline,
        // which is what guarantees a row is never split across chunks.
        for chunk in &chunks[..chunks.len() - 1] {
            assert_eq!(
                *chunk.last().unwrap(),
                b'\n',
                "chunk boundary landed mid-line with {threads} threads"
            );
        }

        // Row count is preserved across the split.
        let chunked_newlines: usize = chunks
            .iter()
            .map(|c| c.iter().filter(|&&b| b == b'\n').count())
            .sum();
        let total_newlines = data.iter().filter(|&&b| b == b'\n').count();
        assert_eq!(chunked_newlines, total_newlines, "lost or gained rows");
    }
});
