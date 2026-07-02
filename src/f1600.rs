//! The Keccak-f[1600] permutation, written once and generic over [`Lane`].
//!
//! Batching is across states, so every rotation offset is identical in all
//! lanes of a vector; a single [`Lane::rotl`] advances the same lane position
//! in every packed state at once. Instantiated at width 1/2/4/8 by the
//! backends, this is the only copy of the permutation in the crate.
//!
//! The state is taken and returned **by value**, and the rho/pi step is
//! unrolled to constant lane indices. Both are load-bearing for speed: they let
//! the compiler keep the 25 lanes in vector registers (scalar-replacement of
//! the aggregate) instead of spilling every lane op to memory, which is the
//! difference between the SIMD widths winning and losing against scalar.

use crate::lane::Lane;

/// Iota round constants for the 24 rounds.
const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Apply the 24-round Keccak-f[1600] permutation to `a`, where each element
/// holds one lane position across `L::LANES` independent states.
///
/// # Safety
///
/// `L`'s methods carry a CPU-feature precondition (see [`Lane`]); call only
/// from a context where that feature is present.
#[inline(always)]
pub(crate) unsafe fn keccak_f1600<L: Lane>(mut a: [L; 25]) -> [L; 25] {
    unsafe {
        for &rc in RC.iter() {
            // Theta: column parities, then fold each column's neighbours in.
            let mut c = [a[0]; 5];
            for x in 0..5 {
                c[x] = a[x]
                    .xor(a[x + 5])
                    .xor(a[x + 10])
                    .xor(a[x + 15])
                    .xor(a[x + 20]);
            }
            for x in 0..5 {
                let d = c[(x + 4) % 5].xor(c[(x + 1) % 5].rotl(1));
                for y in 0..5 {
                    a[x + 5 * y] = a[x + 5 * y].xor(d);
                }
            }

            // Rho + Pi: rotate each lane and move it to its permuted position.
            // Unrolled from the (position, offset) schedule to keep all indices
            // constant.
            let mut t = a[1];
            let cur = a[10];
            a[10] = t.rotl(1);
            t = cur;
            let cur = a[7];
            a[7] = t.rotl(3);
            t = cur;
            let cur = a[11];
            a[11] = t.rotl(6);
            t = cur;
            let cur = a[17];
            a[17] = t.rotl(10);
            t = cur;
            let cur = a[18];
            a[18] = t.rotl(15);
            t = cur;
            let cur = a[3];
            a[3] = t.rotl(21);
            t = cur;
            let cur = a[5];
            a[5] = t.rotl(28);
            t = cur;
            let cur = a[16];
            a[16] = t.rotl(36);
            t = cur;
            let cur = a[8];
            a[8] = t.rotl(45);
            t = cur;
            let cur = a[21];
            a[21] = t.rotl(55);
            t = cur;
            let cur = a[24];
            a[24] = t.rotl(2);
            t = cur;
            let cur = a[4];
            a[4] = t.rotl(14);
            t = cur;
            let cur = a[15];
            a[15] = t.rotl(27);
            t = cur;
            let cur = a[23];
            a[23] = t.rotl(41);
            t = cur;
            let cur = a[19];
            a[19] = t.rotl(56);
            t = cur;
            let cur = a[13];
            a[13] = t.rotl(8);
            t = cur;
            let cur = a[12];
            a[12] = t.rotl(25);
            t = cur;
            let cur = a[2];
            a[2] = t.rotl(43);
            t = cur;
            let cur = a[20];
            a[20] = t.rotl(62);
            t = cur;
            let cur = a[14];
            a[14] = t.rotl(18);
            t = cur;
            let cur = a[22];
            a[22] = t.rotl(39);
            t = cur;
            let cur = a[9];
            a[9] = t.rotl(61);
            t = cur;
            let cur = a[6];
            a[6] = t.rotl(20);
            t = cur;
            a[1] = t.rotl(44);

            // Chi: row-wise non-linear step.
            for y in 0..5 {
                let row = y * 5;
                let r = [a[row], a[row + 1], a[row + 2], a[row + 3], a[row + 4]];
                for x in 0..5 {
                    a[row + x] = r[x].xor(r[(x + 1) % 5].not_and(r[(x + 2) % 5]));
                }
            }

            // Iota: break the round symmetry.
            a[0] = a[0].xor(L::splat(rc));
        }
    }
    a
}
