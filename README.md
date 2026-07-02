# keccak-batch

Batched Keccak-256 with runtime-dispatched SIMD, for workloads that hash many
independent inputs: Merkle trees, redistribution sampling, proof systems.

Keccak does not vectorise by speeding up a single permutation (25 lanes, 25
distinct rotation offsets, no vector rotate on most ISAs). It vectorises by
running several **independent** hashes at once, one per SIMD lane. This crate
exposes that as a batch API and picks the widest backend the running CPU
supports.

| width | backend | targets |
|------:|---------|---------|
| 8 | AVX-512 | x86-64 with `avx512f` |
| 4 | AVX2 | x86-64 with `avx2` |
| 2 | SSE2 / NEON / wasm `simd128` | x86-64, aarch64, wasm |
| 1 | scalar | everywhere |

All hashing uses legacy Keccak padding (`0x01` domain byte), i.e. Ethereum /
Swarm Keccak-256, **not** FIPS-202 SHA3 (`0x06`).

## Usage

```rust
use keccak_batch::{keccak256, keccak256_many, Keccak256};

// Single hash: same shape as alloy's Keccak256, drop-in.
let h = keccak256(b"hello");

// Streaming.
let mut hasher = Keccak256::new();
hasher.update(b"hel");
hasher.update(b"lo");
assert_eq!(hasher.finalize(), h);

// Batched: N equal-length inputs, one digest each.
let a = [0u8; 64];
let b = [1u8; 64];
let digests = keccak256_many::<2>(&[&a, &b]);
```

`keccak256_many` (and the slice form `keccak256_many_into`) require every input
in the batch to be the same length; they absorb in lockstep. Batches of any
count are split across the available width automatically (e.g. 64 leaves at
width 8 run as eight 8-wide permutations). `degree()` reports the active width.

## Performance

Speedup over this crate's own scalar path, per batched hash. The BMT-node case
(64-byte inputs, one permutation each) is the redistribution / Merkle-leaf hot
path; measured on a Zen-class AVX-512 part with `cargo run --release --example
perf`:

| width | 64-byte inputs | 4 KiB inputs |
|------:|---------------:|-------------:|
| 8 (AVX-512) | ~5.4x | ~4-5x |
| 4 (AVX2)    | ~3.1x | ~2.7x |
| 2 (SSE2/NEON/wasm) | ~1.7x | ~1.5x |

The permutation is written once, generic over a `Lane` type; only a ~6-op lane
backend is per-instruction-set. Passing the 25-lane state by value and unrolling
rho/pi keeps every lane in a vector register (without it the SIMD widths lose to
scalar).

`KECCAK_BATCH_NO_AVX512` / `KECCAK_BATCH_NO_AVX2` (env, any value) cap the
backend at runtime, e.g. to avoid AVX-512 downclocking on some parts.

## wasm

The 2-wide backend uses `core::arch::wasm32` and is compiled only when
`simd128` is statically enabled:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

Without it, wasm builds run the scalar path.

### Benchmarking wasm

The `perf` example runs under `wasmtime` on the `wasm32-wasip1` target (the
`.cargo/config.toml` runner is wired up, and `nix develop` provides wasmtime):

```sh
# scalar wasm
cargo run --release --example perf --target wasm32-wasip1
# simd128 batch-2 wasm
RUSTFLAGS="-C target-feature=+simd128" cargo run --release --example perf --target wasm32-wasip1
```

Measured (wasmtime): the 64-byte BMT-node hash is ~379 ns scalar vs ~222 ns
with simd128, a ~1.7x batch-2 win, matching the native SSE2/NEON path. Without
simd128 the batch API falls back to scalar (no speedup).

## Testing

Correctness is pinned three ways: hardcoded known-answer vectors; a proptest
against an independent Keccak-256 (`tiny_keccak`) over lengths spanning the
rate-block boundaries; and a cross-check of every SIMD backend available on the
host against that same oracle. Run `cargo test`.

## Development

`nix develop` (or `direnv allow`) drops you into the pinned toolchain with the
wasm target. Otherwise use a Rust 1.92 toolchain with the `wasm32-unknown-unknown`
target added.

## Licence

MIT OR Apache-2.0.
