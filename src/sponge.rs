//! Batched Keccak-256 sponge: absorb `LANES` equal-length messages in lockstep
//! and squeeze `LANES` 32-byte digests.
//!
//! Legacy Keccak padding (`pad10*1` with the `0x01` domain byte), matching
//! Ethereum / Swarm Keccak-256, not FIPS-202 SHA3 (`0x06`).

use crate::f1600::keccak_f1600;
use crate::lane::{Lane, MAX_LANES};

/// Keccak-256 rate in bytes (1600 - 2*256 bits).
const RATE: usize = 136;
/// Keccak-256 rate in 64-bit lanes.
const RATE_LANES: usize = RATE / 8;

#[inline(always)]
fn read_lane_le(bytes: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(b)
}

/// Hash `L::LANES` equal-length messages in parallel.
///
/// `inputs.len() == out.len() == L::LANES`, and every input must be the same
/// length (the batch absorbs in lockstep). The caller enforces both.
///
/// # Safety
///
/// Instantiates `L`, whose methods carry a CPU-feature precondition; call only
/// from a context where that feature is present (the `#[target_feature]`
/// entries in [`crate::dispatch`]).
#[inline(always)]
pub(crate) unsafe fn keccak256_batch<L: Lane>(inputs: &[&[u8]], out: &mut [[u8; 32]]) {
    let n = L::LANES;
    debug_assert_eq!(inputs.len(), n);
    debug_assert_eq!(out.len(), n);
    let len = inputs[0].len();
    debug_assert!(inputs.iter().all(|s| s.len() == len));

    unsafe {
        let mut state = [L::splat(0); 25];

        // Absorb whole rate blocks.
        let mut off = 0;
        while len - off >= RATE {
            for (i, st) in state.iter_mut().enumerate().take(RATE_LANES) {
                let mut lanes = [0u64; MAX_LANES];
                for (s, &input) in inputs.iter().enumerate() {
                    lanes[s] = read_lane_le(input, off + i * 8);
                }
                *st = st.xor(L::load(&lanes[..n]));
            }
            state = keccak_f1600(state);
            off += RATE;
        }

        // Final block: copy the tail, apply pad10*1 with the 0x01 domain byte.
        let rem = len - off;
        let mut blocks = [[0u8; RATE]; MAX_LANES];
        for (b, &input) in blocks.iter_mut().zip(inputs).take(n) {
            b[..rem].copy_from_slice(&input[off..off + rem]);
            b[rem] = 0x01;
            b[RATE - 1] |= 0x80;
        }
        for (i, st) in state.iter_mut().enumerate().take(RATE_LANES) {
            let mut lanes = [0u64; MAX_LANES];
            for (s, block) in blocks.iter().enumerate().take(n) {
                lanes[s] = read_lane_le(block, i * 8);
            }
            *st = st.xor(L::load(&lanes[..n]));
        }
        state = keccak_f1600(state);

        // Squeeze 32 bytes: the first four lanes of each state.
        let mut tmp = [0u64; MAX_LANES];
        for (i, st) in state.iter().enumerate().take(4) {
            st.store(&mut tmp[..n]);
            for (s, out_s) in out.iter_mut().enumerate().take(n) {
                out_s[i * 8..i * 8 + 8].copy_from_slice(&tmp[s].to_le_bytes());
            }
        }
    }
}
