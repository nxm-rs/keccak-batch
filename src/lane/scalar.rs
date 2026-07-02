//! Width-1 scalar backend. Portable, always available, and the reference all
//! SIMD backends are proptest-checked against.

use super::Lane;

/// A single Keccak state's lane (`LANES == 1`).
#[derive(Copy, Clone)]
pub(crate) struct Scalar(pub u64);

// SAFETY: no intrinsics; every method is a plain arithmetic op that is sound
// on all targets. The `unsafe` is only to satisfy the trait's contract.
unsafe impl Lane for Scalar {
    const LANES: usize = 1;

    #[inline(always)]
    unsafe fn splat(v: u64) -> Self {
        Scalar(v)
    }

    #[inline(always)]
    unsafe fn load(src: &[u64]) -> Self {
        Scalar(src[0])
    }

    #[inline(always)]
    unsafe fn store(self, dst: &mut [u64]) {
        dst[0] = self.0;
    }

    #[inline(always)]
    unsafe fn xor(self, o: Self) -> Self {
        Scalar(self.0 ^ o.0)
    }

    #[inline(always)]
    unsafe fn not_and(self, o: Self) -> Self {
        Scalar(!self.0 & o.0)
    }

    #[inline(always)]
    unsafe fn rotl(self, n: u32) -> Self {
        Scalar(self.0.rotate_left(n))
    }
}
