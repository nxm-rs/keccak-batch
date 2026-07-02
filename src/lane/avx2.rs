//! Width-4 AVX2 backend (256-bit registers, 4 states per vector).
//!
//! AVX2 has no vector rotate, so `rotl` is emulated as `shl | shr` with a
//! runtime shift count shared across all four lanes.

use super::Lane;
use core::arch::x86_64::*;

/// Four Keccak states' lanes packed into one `__m256i` (`LANES == 4`).
#[derive(Copy, Clone)]
pub(crate) struct U64x4(__m256i);

// SAFETY: every method calls AVX2 intrinsics. Soundness precondition: the CPU
// supports AVX2. Guaranteed by callers, which reach this type only from the
// `#[target_feature(enable = "avx2")]` entry in `crate::dispatch` after an
// `is_x86_feature_detected!("avx2")` check.
unsafe impl Lane for U64x4 {
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn splat(v: u64) -> Self {
        U64x4(unsafe { _mm256_set1_epi64x(v as i64) })
    }

    #[inline(always)]
    unsafe fn load(src: &[u64]) -> Self {
        U64x4(unsafe {
            _mm256_set_epi64x(src[3] as i64, src[2] as i64, src[1] as i64, src[0] as i64)
        })
    }

    #[inline(always)]
    unsafe fn store(self, dst: &mut [u64]) {
        let mut t = [0u64; 4];
        unsafe { _mm256_storeu_si256(t.as_mut_ptr().cast(), self.0) };
        dst[..4].copy_from_slice(&t);
    }

    #[inline(always)]
    unsafe fn xor(self, o: Self) -> Self {
        U64x4(unsafe { _mm256_xor_si256(self.0, o.0) })
    }

    #[inline(always)]
    unsafe fn not_and(self, o: Self) -> Self {
        // _mm256_andnot_si256(a, b) == (!a) & b, exactly chi.
        U64x4(unsafe { _mm256_andnot_si256(self.0, o.0) })
    }

    #[inline(always)]
    unsafe fn rotl(self, n: u32) -> Self {
        unsafe {
            let l = _mm_cvtsi64_si128(n as i64);
            let r = _mm_cvtsi64_si128((64 - n) as i64);
            U64x4(_mm256_or_si256(
                _mm256_sll_epi64(self.0, l),
                _mm256_srl_epi64(self.0, r),
            ))
        }
    }
}
