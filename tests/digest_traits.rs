//! Pin the RustCrypto `digest` trait surface: the blanket `Digest` impl must
//! resolve to the inherent hashing logic (no trait-method recursion) and the
//! reset paths must restore a fresh state.
#![cfg(feature = "digest")]

use digest::Digest;
use keccak_batch::{Keccak256, keccak256};

fn hash_via<D: Digest>(data: &[u8]) -> Vec<u8> {
    let mut h = D::new();
    Digest::update(&mut h, data);
    h.finalize().to_vec()
}

#[test]
fn digest_trait_matches_inherent() {
    for data in [&b""[..], b"abc", &[0xa5; 200]] {
        assert_eq!(hash_via::<Keccak256>(data), keccak256(data).to_vec());
        assert_eq!(Keccak256::digest(data).to_vec(), keccak256(data).to_vec());
    }
}

#[test]
fn finalize_reset_restores_fresh_state() {
    let mut h = Keccak256::new();
    Digest::update(&mut h, b"hello");
    let first = h.finalize_reset();
    Digest::update(&mut h, b"hello");
    let second = h.finalize();
    assert_eq!(first, second);
    assert_eq!(first.to_vec(), keccak256(b"hello").to_vec());
}

#[test]
fn reset_discards_pending_input() {
    let mut h = Keccak256::new();
    Digest::update(&mut h, b"garbage that must not leak into the next hash");
    Digest::reset(&mut h);
    Digest::update(&mut h, b"abc");
    assert_eq!(h.finalize().to_vec(), keccak256(b"abc").to_vec());
}
