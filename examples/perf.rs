//! Rough A/B timing to separate permutation cost from transpose overhead.
#![allow(missing_docs)]

use std::time::Instant;

use keccak_batch::{Keccak256, degree, keccak256_many_into};

fn bench(label: &str, n: usize, len: usize) {
    let width = degree();
    let inputs_owned: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; len]).collect();
    let inputs: Vec<&[u8]> = inputs_owned.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![[0u8; 32]; n];

    let iters = (2_000_000 / (len + 1)).max(200);

    // Warmup.
    for _ in 0..iters / 10 {
        keccak256_many_into(&inputs, &mut out);
    }

    let t = Instant::now();
    for _ in 0..iters {
        keccak256_many_into(std::hint::black_box(&inputs), &mut out);
        std::hint::black_box(&out);
    }
    let batch_ns = t.elapsed().as_nanos() as f64 / (iters * n) as f64;

    let t = Instant::now();
    for _ in 0..iters {
        for x in &inputs {
            let mut h = Keccak256::new();
            h.update(std::hint::black_box(x));
            std::hint::black_box(h.finalize());
        }
    }
    let serial_ns = t.elapsed().as_nanos() as f64 / (iters * n) as f64;

    println!(
        "{label:<18} width={width} batch={batch_ns:6.1} ns/hash  serial={serial_ns:6.1} ns/hash  speedup={:.2}x",
        serial_ns / batch_ns
    );
}

/// Prove the active backend is correct before timing it: a known vector, and
/// batch-vs-scalar agreement over boundary lengths. This is what makes the wasm
/// run (where the proptest oracle is unavailable) also a correctness check of
/// the simd128 path.
fn self_check() {
    use keccak_batch::{keccak256, keccak256_many_into};

    assert_eq!(
        keccak256(b""),
        [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7,
            0x03, 0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04,
            0x5d, 0x85, 0xa4, 0x70,
        ],
        "keccak256(\"\") mismatch",
    );

    for &len in &[0usize, 1, 63, 64, 135, 136, 137, 272, 400] {
        let inputs: Vec<Vec<u8>> = (0..17).map(|s| vec![(s as u8).wrapping_mul(31); len]).collect();
        let slices: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut got = vec![[0u8; 32]; inputs.len()];
        keccak256_many_into(&slices, &mut got);
        for (s, input) in inputs.iter().enumerate() {
            assert_eq!(got[s], keccak256(input), "batch != scalar at len {len}, lane {s}");
        }
    }
    println!("self-check: ok");
}

fn main() {
    self_check();
    println!("degree = {}", degree());
    bench("8x64 (BMT node)", 8, 64);
    bench("64x64 (BMT level)", 64, 64);
    bench("8x4096 (perm-bound)", 8, 4096);
    bench("64x4096", 64, 4096);
}
