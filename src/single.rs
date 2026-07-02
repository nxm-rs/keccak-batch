//! Streaming single-message Keccak-256, with an `alloy_primitives::Keccak256`
//! shaped inherent API and (behind the `digest` feature) the RustCrypto
//! `digest` traits, so it drop-in-swaps for either.

use crate::f1600::keccak_f1600;
use crate::lane::Scalar;

const RATE: usize = 136;

/// Incremental Keccak-256 hasher (legacy `0x01` padding).
#[derive(Clone)]
pub struct Keccak256 {
    state: [u64; 25],
    buf: [u8; RATE],
    buf_len: usize,
}

impl Default for Keccak256 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Keccak256 {
    /// Create an empty hasher.
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: [0u64; 25],
            buf: [0u8; RATE],
            buf_len: 0,
        }
    }

    #[inline]
    fn permute(state: &mut [u64; 25]) {
        let input: [Scalar; 25] = core::array::from_fn(|i| Scalar(state[i]));
        // SAFETY: the scalar backend uses no intrinsics and has no CPU-feature
        // precondition.
        let s = unsafe { keccak_f1600(input) };
        for (dst, lane) in state.iter_mut().zip(s) {
            *dst = lane.0;
        }
    }

    #[inline]
    fn absorb_block(state: &mut [u64; 25], block: &[u8; RATE]) {
        for i in 0..RATE / 8 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&block[i * 8..i * 8 + 8]);
            state[i] ^= u64::from_le_bytes(b);
        }
        Self::permute(state);
    }

    /// Absorb more input.
    #[inline]
    pub fn update(&mut self, data: impl AsRef<[u8]>) {
        let mut data = data.as_ref();

        if self.buf_len > 0 {
            let need = RATE - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == RATE {
                let block = self.buf;
                Self::absorb_block(&mut self.state, &block);
                self.buf_len = 0;
            }
        }

        while data.len() >= RATE {
            let mut block = [0u8; RATE];
            block.copy_from_slice(&data[..RATE]);
            Self::absorb_block(&mut self.state, &block);
            data = &data[RATE..];
        }

        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    #[inline]
    fn squeeze(&mut self) -> [u8; 32] {
        let rem = self.buf_len;
        let mut block = [0u8; RATE];
        block[..rem].copy_from_slice(&self.buf[..rem]);
        block[rem] = 0x01;
        block[RATE - 1] |= 0x80;
        Self::absorb_block(&mut self.state, &block);

        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.state[i].to_le_bytes());
        }
        out
    }

    /// Consume the hasher and return the 32-byte digest.
    #[inline]
    pub fn finalize(mut self) -> [u8; 32] {
        self.squeeze()
    }

    /// Consume the hasher and write the digest into `out` (`out.len() == 32`).
    #[inline]
    pub fn finalize_into(mut self, out: &mut [u8]) {
        assert_eq!(out.len(), 32, "keccak256 output is 32 bytes");
        out.copy_from_slice(&self.squeeze());
    }
}

/// One-shot Keccak-256 of `bytes`.
#[inline]
pub fn keccak256(bytes: impl AsRef<[u8]>) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(bytes);
    h.finalize()
}

#[cfg(feature = "digest")]
mod digest_impl {
    use super::Keccak256;
    use digest::{
        FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update,
        consts::U32,
    };

    impl HashMarker for Keccak256 {}

    impl OutputSizeUser for Keccak256 {
        type OutputSize = U32;
    }

    impl Update for Keccak256 {
        #[inline]
        fn update(&mut self, data: &[u8]) {
            Keccak256::update(self, data);
        }
    }

    impl Reset for Keccak256 {
        #[inline]
        fn reset(&mut self) {
            *self = Keccak256::new();
        }
    }

    impl FixedOutput for Keccak256 {
        #[inline]
        fn finalize_into(mut self, out: &mut Output<Self>) {
            out.copy_from_slice(&self.squeeze());
        }
    }

    impl FixedOutputReset for Keccak256 {
        #[inline]
        fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
            out.copy_from_slice(&self.squeeze());
            *self = Keccak256::new();
        }
    }
}
