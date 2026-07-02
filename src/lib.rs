//! Batched Keccak-256 with runtime-dispatched SIMD.
//!
//! Keccak vectorises by running several *independent* hashes at once, one per
//! SIMD lane, rather than by speeding up a single permutation. This crate
//! exposes that as a batch API and picks the widest backend the running CPU
//! supports:
//!
//! | width | backend | targets |
//! |------:|---------|---------|
//! | 8 | AVX-512 | x86-64 with `avx512f` |
//! | 4 | AVX2 | x86-64 with `avx2` |
//! | 2 | SSE2 / NEON / wasm `simd128` | x86-64, aarch64, wasm |
//! | 1 | scalar | everywhere |
//!
//! One [`f1600`] permutation is written generically over a `Lane` type; only
//! the ~6-op lane backends are per-instruction-set. All hashing uses legacy
//! Keccak padding (`0x01` domain byte), matching Ethereum / Swarm Keccak-256,
//! not FIPS-202 SHA3.
//!
//! ```
//! use keccak_batch::{keccak256, keccak256_many};
//!
//! // Single hash (drop-in for alloy's Keccak256 shape).
//! let h = keccak256(b"");
//! assert_eq!(
//!     h,
//!     hex_literal::hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
//! );
//!
//! // Batched: hash several equal-length inputs at once.
//! let a = [0u8; 64];
//! let b = [1u8; 64];
//! let digests = keccak256_many::<2>(&[&a, &b]);
//! assert_eq!(digests[0], keccak256(&a));
//! assert_eq!(digests[1], keccak256(&b));
//! ```

mod dispatch;
mod f1600;
mod lane;
mod single;
mod sponge;

pub use dispatch::{degree, keccak256_many, keccak256_many_into};
pub use single::{Keccak256, keccak256};
