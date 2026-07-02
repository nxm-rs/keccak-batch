//! Known-answer tests pinning the scalar path to canonical Ethereum/Swarm
//! Keccak-256 constants (independent of any oracle crate).

use hex_literal::hex;
use keccak_batch::{Keccak256, keccak256};

#[test]
fn canonical_vectors() {
    assert_eq!(
        keccak256(b""),
        hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
    );
    assert_eq!(
        keccak256(b"abc"),
        hex!("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
    );
    assert_eq!(
        keccak256(b"hello"),
        hex!("1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8"),
    );
    assert_eq!(
        keccak256(b"The quick brown fox jumps over the lazy dog"),
        hex!("4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15"),
    );
}

#[test]
fn streaming_equals_oneshot() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let mut h = Keccak256::new();
    h.update(&data[..10]);
    h.update(&data[10..20]);
    h.update(&data[20..]);
    assert_eq!(h.finalize(), keccak256(data));
}

#[test]
fn finalize_into_matches() {
    let mut out = [0u8; 32];
    let mut h = Keccak256::new();
    h.update(b"hello");
    h.finalize_into(&mut out);
    assert_eq!(out, keccak256(b"hello"));
}
