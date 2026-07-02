//! The `Lane` abstraction: one Keccak lane position held across `LANES`
//! independent states.
//!
//! Batched Keccak vectorises *across* states, not within one. A `Lane` value
//! holds lane position `i` of `LANES` separate 1600-bit states, so a single
//! [`Lane::rotl`] advances that lane in every state at once. This is why the
//! permutation ([`crate::f1600`]) is written once, generic over `Lane`, and
//! only the handful of ops below are implemented per instruction set.
//!
//! # Safety
//!
//! Every method is `unsafe`: SIMD implementors call target-specific intrinsics
//! whose CPU feature is a precondition. Callers reach these methods only
//! through the `#[target_feature]` entry points in [`crate::dispatch`], which
//! run after the matching runtime feature check.

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
mod simd128;

/// Largest batch width any backend provides. Sizes the small stack scratch
/// buffers in [`crate::sponge`] so no per-generic-const array is needed.
pub(crate) const MAX_LANES: usize = 8;

/// One Keccak lane position across `LANES` independent states.
///
/// See the module docs for the batching model.
///
/// # Safety
///
/// SIMD implementors' methods call intrinsics whose CPU feature is a
/// precondition. Call them only from a context where that feature is present:
/// the `#[target_feature]` entry points in [`crate::dispatch`], reached after
/// the matching runtime detection. `load`/`store` slices must be `LANES` long.
pub(crate) unsafe trait Lane: Copy {
    /// Number of independent states packed into one value.
    const LANES: usize;

    /// Broadcast `v` into every state's lane (used for round constants).
    unsafe fn splat(v: u64) -> Self;

    /// Gather one `u64` per state. `src.len() == LANES`.
    unsafe fn load(src: &[u64]) -> Self;

    /// Scatter each state's lane back out. `dst.len() == LANES`.
    unsafe fn store(self, dst: &mut [u64]);

    /// Lane-wise XOR.
    unsafe fn xor(self, o: Self) -> Self;

    /// Lane-wise `(!self) & o` (Keccak's chi step).
    unsafe fn not_and(self, o: Self) -> Self;

    /// Lane-wise rotate-left by `n`, with `n` in `1..=63`.
    unsafe fn rotl(self, n: u32) -> Self;
}

pub(crate) use scalar::Scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) use avx2::U64x4;
#[cfg(target_arch = "x86_64")]
pub(crate) use avx512::U64x8;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
pub(crate) use simd128::U64x2;
