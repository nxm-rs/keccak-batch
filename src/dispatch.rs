//! Runtime backend selection and the public batched API.
//!
//! The widest backend the running CPU supports is chosen once (cached on x86),
//! and a batch of inputs is greedily split into 8/4/2/1-wide chunks fed to the
//! matching `#[target_feature]` entry. Narrower widths are always available
//! when a wider one is (AVX-512 implies AVX2 implies SSE2), so a chosen chunk
//! width never exceeds what the CPU provides.

use crate::lane::Scalar;
use crate::sponge::keccak256_batch;

/// Hash `N` equal-length messages, returning `N` digests.
///
/// Every input must be the same length. Panics otherwise.
#[inline]
pub fn keccak256_many<const N: usize>(inputs: &[&[u8]; N]) -> [[u8; 32]; N] {
    let mut out = [[0u8; 32]; N];
    keccak256_many_into(inputs, &mut out);
    out
}

/// Hash a slice of equal-length messages into `out` (`inputs.len() == out.len()`).
///
/// Every input must be the same length. Panics otherwise. This is the
/// variable-count entry the BMT level passes use; [`keccak256_many`] is the
/// fixed-`N` convenience over it.
pub fn keccak256_many_into(inputs: &[&[u8]], out: &mut [[u8; 32]]) {
    assert_eq!(
        inputs.len(),
        out.len(),
        "inputs and out must have equal length"
    );
    if let Some(first) = inputs.first() {
        let len = first.len();
        assert!(
            inputs.iter().all(|s| s.len() == len),
            "keccak256_many requires equal-length inputs",
        );
    }

    let w = degree();
    let n = inputs.len();
    let mut i = 0;
    while i < n {
        let cw = chunk_width(n - i, w);
        dispatch(cw, &inputs[i..i + cw], &mut out[i..i + cw]);
        i += cw;
    }
}

/// Widest batch width the running CPU supports: 8, 4, 2, or 1.
#[inline]
pub fn degree() -> usize {
    detected_degree()
}

/// Largest of `{1,2,4,8}` that is `<= remaining` and `<= max` (`max` is one of
/// those powers of two).
#[inline]
fn chunk_width(remaining: usize, max: usize) -> usize {
    let mut w = max;
    while w > remaining {
        w /= 2;
    }
    w
}

#[inline]
fn hash_x1(inputs: &[&[u8]], out: &mut [[u8; 32]]) {
    // SAFETY: the scalar backend has no CPU-feature precondition.
    unsafe { keccak256_batch::<Scalar>(inputs, out) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hash_x4_avx2(inputs: &[&[u8]], out: &mut [[u8; 32]]) {
    // SAFETY: reached only when avx2 was runtime-detected (see `dispatch`).
    unsafe { keccak256_batch::<crate::lane::U64x4>(inputs, out) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn hash_x8_avx512(inputs: &[&[u8]], out: &mut [[u8; 32]]) {
    // SAFETY: reached only when avx512f was runtime-detected (see `dispatch`).
    unsafe { keccak256_batch::<crate::lane::U64x8>(inputs, out) };
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline]
fn hash_x2(inputs: &[&[u8]], out: &mut [[u8; 32]]) {
    // SAFETY: the 128-bit backend is compiled only where its ISA is part of the
    // target baseline (SSE2 / NEON) or statically enabled (wasm simd128).
    unsafe { keccak256_batch::<crate::lane::U64x2>(inputs, out) };
}

#[inline]
fn dispatch(cw: usize, inputs: &[&[u8]], out: &mut [[u8; 32]]) {
    match cw {
        #[cfg(target_arch = "x86_64")]
        8 => unsafe { hash_x8_avx512(inputs, out) },
        #[cfg(target_arch = "x86_64")]
        4 => unsafe { hash_x4_avx2(inputs, out) },
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128"),
        ))]
        2 => hash_x2(inputs, out),
        _ => {
            for k in 0..inputs.len() {
                hash_x1(&inputs[k..k + 1], &mut out[k..k + 1]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detected_degree() -> usize {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cached = CACHE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached as usize;
    }
    let mut d = if is_x86_feature_detected!("avx512f") {
        8
    } else if is_x86_feature_detected!("avx2") {
        4
    } else {
        2 // SSE2 is part of the x86-64 baseline.
    };
    // Operational caps: force a narrower backend without a rebuild, e.g. to
    // avoid AVX-512 downclocking on some parts. Read once, then cached.
    if d > 4 && std::env::var_os("KECCAK_BATCH_NO_AVX512").is_some() {
        d = 4;
    }
    if d > 2 && std::env::var_os("KECCAK_BATCH_NO_AVX2").is_some() {
        d = 2;
    }
    CACHE.store(d as u8, Ordering::Relaxed);
    d
}

#[cfg(target_arch = "aarch64")]
fn detected_degree() -> usize {
    2 // NEON is mandatory on aarch64.
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn detected_degree() -> usize {
    2
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
fn detected_degree() -> usize {
    1
}

#[cfg(test)]
mod backend_tests {
    //! Cross-check every backend compiled and available on this host against an
    //! independent Keccak-256 (`tiny_keccak`), across message lengths that span
    //! the rate-block boundaries (135/136/137, 271/272/273).

    use super::*;

    /// Independent oracle: a different codebase's Keccak-256.
    fn oracle(input: &[u8]) -> [u8; 32] {
        use tiny_keccak::{Hasher, Keccak};
        let mut k = Keccak::v256();
        k.update(input);
        let mut out = [0u8; 32];
        k.finalize(&mut out);
        out
    }

    /// Deterministic splitmix64 fill so failures reproduce without an rng dep.
    fn fill(mut x: u64, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *b = (z ^ (z >> 31)) as u8;
        }
    }

    /// Lengths worth exercising: empties, sub-block, and every boundary.
    const LENS: &[usize] = &[
        0, 1, 8, 31, 32, 63, 64, 65, 135, 136, 137, 200, 271, 272, 273, 400, 500,
    ];

    /// Run a width-`w` backend on exactly `w` equal-length inputs and check
    /// each digest against the oracle. `w` must be available on this host.
    fn check_width(w: usize, len: usize, seed: u64) {
        let inputs: Vec<Vec<u8>> = (0..w)
            .map(|s| {
                let mut v = vec![0u8; len];
                fill(
                    seed ^ (s as u64).wrapping_mul(0x1234_5678_9abc_def1),
                    &mut v,
                );
                v
            })
            .collect();
        let slices: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut got = vec![[0u8; 32]; w];

        match w {
            1 => hash_x1(&slices, &mut got),
            #[cfg(any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                all(target_arch = "wasm32", target_feature = "simd128"),
            ))]
            2 => hash_x2(&slices, &mut got),
            #[cfg(target_arch = "x86_64")]
            4 => unsafe { hash_x4_avx2(&slices, &mut got) },
            #[cfg(target_arch = "x86_64")]
            8 => unsafe { hash_x8_avx512(&slices, &mut got) },
            _ => unreachable!("width {w} not available on this target"),
        }

        for (s, input) in inputs.iter().enumerate() {
            assert_eq!(got[s], oracle(input), "width {w}, len {len}, lane {s}");
        }
    }

    /// Widths whose backend is both compiled for this target and supported by
    /// the running CPU.
    fn available_widths() -> Vec<usize> {
        let mut w = vec![1usize];
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128"),
        ))]
        w.push(2);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                w.push(4);
            }
            if is_x86_feature_detected!("avx512f") {
                w.push(8);
            }
        }
        w
    }

    #[test]
    fn every_backend_matches_oracle() {
        for &w in &available_widths() {
            for &len in LENS {
                for seed in 0..8u64 {
                    check_width(w, len, seed.wrapping_mul(0xdead_beef));
                }
            }
        }
    }

    #[test]
    fn public_many_matches_oracle_odd_counts() {
        // Counts that force the greedy chunker to mix widths (e.g. 7 = 4+2+1).
        for n in [1usize, 2, 3, 5, 7, 11, 16, 31, 64] {
            for &len in &[0usize, 64, 136, 300] {
                let inputs: Vec<Vec<u8>> = (0..n)
                    .map(|s| {
                        let mut v = vec![0u8; len];
                        fill(0xabc ^ s as u64, &mut v);
                        v
                    })
                    .collect();
                let slices: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
                let mut got = vec![[0u8; 32]; n];
                keccak256_many_into(&slices, &mut got);
                for (s, input) in inputs.iter().enumerate() {
                    assert_eq!(got[s], oracle(input), "n {n}, len {len}, lane {s}");
                }
            }
        }
    }
}
