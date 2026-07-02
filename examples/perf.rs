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

fn main() {
    println!("degree = {}", degree());
    bench("8x64 (BMT node)", 8, 64);
    bench("64x64 (BMT level)", 64, 64);
    bench("8x4096 (perm-bound)", 8, 4096);
    bench("64x4096", 64, 4096);
}
