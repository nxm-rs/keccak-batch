//! Benchmarks: single hash, batched widths, and the BMT-shaped workload.
//!
//! The BMT case mirrors the redistribution hot path: 64 independent 64-byte
//! leaf pair-hashes, the level where batching pays off most.
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use keccak_batch::{Keccak256, degree, keccak256_many_into};

fn bench_single(c: &mut Criterion) {
    let mut g = c.benchmark_group("single");
    for &size in &[64usize, 136, 4096] {
        let data = vec![0xa5u8; size];
        g.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let mut h = Keccak256::new();
                h.update(std::hint::black_box(data));
                std::hint::black_box(h.finalize())
            });
        });
    }
    g.finish();
}

fn bench_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("batch64");
    g.throughput(criterion::Throughput::Elements(64));
    // 64 leaf pair-hashes of 64 bytes each.
    let inputs_owned: Vec<[u8; 64]> = (0..64).map(|i| [i as u8; 64]).collect();
    let inputs: Vec<&[u8]> = inputs_owned.iter().map(|x| x.as_slice()).collect();

    g.bench_function(BenchmarkId::new("width", degree()), |b| {
        let mut out = vec![[0u8; 32]; 64];
        b.iter(|| {
            keccak256_many_into(std::hint::black_box(&inputs), &mut out);
            std::hint::black_box(&out);
        });
    });

    // Baseline: the same 64 hashes done one at a time (scalar path).
    g.bench_function("serial_single", |b| {
        b.iter(|| {
            for x in &inputs {
                let mut h = Keccak256::new();
                h.update(x);
                std::hint::black_box(h.finalize());
            }
        });
    });
    g.finish();
}

criterion_group!(benches, bench_single, bench_batch);
criterion_main!(benches);
