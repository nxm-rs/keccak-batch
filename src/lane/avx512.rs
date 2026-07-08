//! Width-8 AVX-512 backend (512-bit registers, 8 states per vector).
//!
//! AVX-512 has a native per-lane vector rotate (`vprolvq`), so `rotl` is a
//! single instruction rather than the `shl | shr` emulation the narrower
//! backends need.

use super::Lane;
use core::arch::x86_64::*;

/// Eight Keccak states' lanes packed into one `__m512i` (`LANES == 8`).
#[derive(Copy, Clone)]
pub(crate) struct U64x8(__m512i);

// SAFETY: every method calls AVX-512F intrinsics. Soundness precondition: the
// CPU supports AVX-512F. Guaranteed by callers, which reach this type only from
// the `#[target_feature(enable = "avx512f")]` entry in `crate::dispatch` after
// an `is_x86_feature_detected!("avx512f")` check.
unsafe impl Lane for U64x8 {
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn splat(v: u64) -> Self {
        U64x8(unsafe { _mm512_set1_epi64(v as i64) })
    }

    #[inline(always)]
    unsafe fn load(src: &[u64]) -> Self {
        U64x8(unsafe {
            _mm512_set_epi64(
                src[7] as i64,
                src[6] as i64,
                src[5] as i64,
                src[4] as i64,
                src[3] as i64,
                src[2] as i64,
                src[1] as i64,
                src[0] as i64,
            )
        })
    }

    #[inline(always)]
    unsafe fn store(self, dst: &mut [u64]) {
        let mut t = [0u64; 8];
        unsafe { _mm512_storeu_si512(t.as_mut_ptr().cast(), self.0) };
        dst[..8].copy_from_slice(&t);
    }

    #[inline(always)]
    unsafe fn xor(self, o: Self) -> Self {
        U64x8(unsafe { _mm512_xor_si512(self.0, o.0) })
    }

    #[inline(always)]
    unsafe fn not_and(self, o: Self) -> Self {
        // _mm512_andnot_si512(a, b) == (!a) & b, exactly chi.
        U64x8(unsafe { _mm512_andnot_si512(self.0, o.0) })
    }

    #[inline(always)]
    unsafe fn chi(self, b: Self, c: Self) -> Self {
        // vpternlogq with the truth table of f(a,b,c) = a ^ ((!b) & c) fuses the
        // andnot and xor of one chi output lane into a single instruction.
        //
        // imm8 bit i (i = 4a+2b+c, a the MSB) holds f(a,b,c):
        //   000→0 001→1 010→0 011→0 100→1 101→0 110→1 111→1  = 0b1101_0010 = 0xD2
        U64x8(unsafe { _mm512_ternarylogic_epi64::<0xD2>(self.0, b.0, c.0) })
    }

    #[inline(always)]
    unsafe fn xor3(self, b: Self, c: Self) -> Self {
        // f(a,b,c) = a ^ b ^ c: parity is 1 for odd popcount of (a,b,c).
        //   000→0 001→1 010→1 011→0 100→1 101→0 110→0 111→1  = 0b1001_0110 = 0x96
        U64x8(unsafe { _mm512_ternarylogic_epi64::<0x96>(self.0, b.0, c.0) })
    }

    #[inline(always)]
    unsafe fn rotl(self, n: u32) -> Self {
        debug_assert!((1..64).contains(&n), "rotl amount must be in 1..=63");
        U64x8(unsafe { _mm512_rolv_epi64(self.0, _mm512_set1_epi64(n as i64)) })
    }
}
