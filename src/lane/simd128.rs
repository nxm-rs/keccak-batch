//! Width-2 backend (128-bit registers, 2 states per vector), shared across
//! every 128-bit instruction set: x86 SSE2, ARM NEON, and wasm simd128.
//!
//! This is the only SIMD width available on wasm, and the baseline SIMD on
//! aarch64. On x86-64 it sits below AVX2/AVX-512 in the dispatch ladder. None
//! of these three has a native 64-bit vector rotate, so `rotl` is `shl | shr`.

use super::Lane;

/// Two Keccak states' lanes packed into one 128-bit vector (`LANES == 2`).
#[derive(Copy, Clone)]
pub(crate) struct U64x2(Inner);

// --- x86-64: SSE2 (baseline on the target, no runtime detection needed) ---
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128i as Inner;

#[cfg(target_arch = "x86_64")]
// SAFETY: SSE2 is part of the x86-64 baseline, so these intrinsics are always
// available on this target.
unsafe impl Lane for U64x2 {
    const LANES: usize = 2;

    #[inline(always)]
    unsafe fn splat(v: u64) -> Self {
        use core::arch::x86_64::_mm_set1_epi64x;
        U64x2(unsafe { _mm_set1_epi64x(v as i64) })
    }

    #[inline(always)]
    unsafe fn load(src: &[u64]) -> Self {
        use core::arch::x86_64::_mm_set_epi64x;
        U64x2(unsafe { _mm_set_epi64x(src[1] as i64, src[0] as i64) })
    }

    #[inline(always)]
    unsafe fn store(self, dst: &mut [u64]) {
        use core::arch::x86_64::_mm_storeu_si128;
        let mut t = [0u64; 2];
        unsafe { _mm_storeu_si128(t.as_mut_ptr().cast(), self.0) };
        dst[..2].copy_from_slice(&t);
    }

    #[inline(always)]
    unsafe fn xor(self, o: Self) -> Self {
        use core::arch::x86_64::_mm_xor_si128;
        U64x2(unsafe { _mm_xor_si128(self.0, o.0) })
    }

    #[inline(always)]
    unsafe fn not_and(self, o: Self) -> Self {
        use core::arch::x86_64::_mm_andnot_si128;
        U64x2(unsafe { _mm_andnot_si128(self.0, o.0) })
    }

    #[inline(always)]
    unsafe fn rotl(self, n: u32) -> Self {
        debug_assert!((1..64).contains(&n), "rotl amount must be in 1..=63");
        use core::arch::x86_64::{_mm_cvtsi64_si128, _mm_or_si128, _mm_sll_epi64, _mm_srl_epi64};
        unsafe {
            let l = _mm_cvtsi64_si128(n as i64);
            let r = _mm_cvtsi64_si128((64 - n) as i64);
            U64x2(_mm_or_si128(
                _mm_sll_epi64(self.0, l),
                _mm_srl_epi64(self.0, r),
            ))
        }
    }
}

// --- aarch64: NEON (baseline on the target) ---
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::uint64x2_t as Inner;

#[cfg(target_arch = "aarch64")]
// SAFETY: NEON is mandatory on aarch64, so these intrinsics are always
// available on this target.
unsafe impl Lane for U64x2 {
    const LANES: usize = 2;

    #[inline(always)]
    unsafe fn splat(v: u64) -> Self {
        use core::arch::aarch64::vdupq_n_u64;
        U64x2(unsafe { vdupq_n_u64(v) })
    }

    #[inline(always)]
    unsafe fn load(src: &[u64]) -> Self {
        use core::arch::aarch64::vld1q_u64;
        U64x2(unsafe { vld1q_u64(src.as_ptr()) })
    }

    #[inline(always)]
    unsafe fn store(self, dst: &mut [u64]) {
        use core::arch::aarch64::vst1q_u64;
        unsafe { vst1q_u64(dst.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    unsafe fn xor(self, o: Self) -> Self {
        use core::arch::aarch64::veorq_u64;
        U64x2(unsafe { veorq_u64(self.0, o.0) })
    }

    #[inline(always)]
    unsafe fn not_and(self, o: Self) -> Self {
        // vbicq_u64(a, b) == a & !b, so (o, self) gives o & !self == (!self) & o.
        use core::arch::aarch64::vbicq_u64;
        U64x2(unsafe { vbicq_u64(o.0, self.0) })
    }

    #[inline(always)]
    unsafe fn rotl(self, n: u32) -> Self {
        debug_assert!((1..64).contains(&n), "rotl amount must be in 1..=63");
        use core::arch::aarch64::{vdupq_n_s64, vorrq_u64, vshlq_u64};
        unsafe {
            let left = vshlq_u64(self.0, vdupq_n_s64(n as i64));
            let right = vshlq_u64(self.0, vdupq_n_s64(n as i64 - 64));
            U64x2(vorrq_u64(left, right))
        }
    }
}

// --- wasm32: simd128 (only when statically enabled) ---
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::v128 as Inner;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
// SAFETY: gated on `target_feature = "simd128"`, so the intrinsics are enabled.
unsafe impl Lane for U64x2 {
    const LANES: usize = 2;

    #[inline(always)]
    unsafe fn splat(v: u64) -> Self {
        use core::arch::wasm32::u64x2_splat;
        U64x2(u64x2_splat(v))
    }

    #[inline(always)]
    unsafe fn load(src: &[u64]) -> Self {
        use core::arch::wasm32::u64x2;
        U64x2(u64x2(src[0], src[1]))
    }

    #[inline(always)]
    unsafe fn store(self, dst: &mut [u64]) {
        use core::arch::wasm32::u64x2_extract_lane;
        dst[0] = u64x2_extract_lane::<0>(self.0);
        dst[1] = u64x2_extract_lane::<1>(self.0);
    }

    #[inline(always)]
    unsafe fn xor(self, o: Self) -> Self {
        use core::arch::wasm32::v128_xor;
        U64x2(v128_xor(self.0, o.0))
    }

    #[inline(always)]
    unsafe fn not_and(self, o: Self) -> Self {
        // v128_andnot(a, b) == a & !b, so (o, self) gives o & !self == (!self) & o.
        use core::arch::wasm32::v128_andnot;
        U64x2(v128_andnot(o.0, self.0))
    }

    #[inline(always)]
    unsafe fn rotl(self, n: u32) -> Self {
        debug_assert!((1..64).contains(&n), "rotl amount must be in 1..=63");
        use core::arch::wasm32::{i64x2_shl, u64x2_shr, v128_or};
        U64x2(v128_or(i64x2_shl(self.0, n), u64x2_shr(self.0, 64 - n)))
    }
}
