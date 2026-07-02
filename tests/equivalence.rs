//! Property tests for the public API against an independent oracle
//! (`tiny_keccak`), plus batch-vs-single and streaming-split consistency.

use keccak_batch::{Keccak256, keccak256, keccak256_many_into};
use proptest::prelude::*;

fn oracle(input: &[u8]) -> [u8; 32] {
    use tiny_keccak::{Hasher, Keccak};
    let mut k = Keccak::v256();
    k.update(input);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn one_shot_matches_oracle(data in prop::collection::vec(any::<u8>(), 0..600)) {
        prop_assert_eq!(keccak256(&data), oracle(&data));
    }

    #[test]
    fn streaming_matches_one_shot(
        data in prop::collection::vec(any::<u8>(), 0..600),
        splits in prop::collection::vec(0usize..600, 0..8),
    ) {
        let mut h = Keccak256::new();
        let mut pos = 0;
        let mut cuts: Vec<usize> = splits.into_iter().filter(|&c| c <= data.len()).collect();
        cuts.sort_unstable();
        for c in cuts {
            h.update(&data[pos..c]);
            pos = c;
        }
        h.update(&data[pos..]);
        prop_assert_eq!(h.finalize(), keccak256(&data));
    }

    #[test]
    fn batch_matches_oracle(
        count in 1usize..40,
        len in 0usize..400,
        seed in any::<u64>(),
    ) {
        // Equal-length inputs, deterministically derived per lane.
        let inputs: Vec<Vec<u8>> = (0..count)
            .map(|s| {
                let mut v = vec![0u8; len];
                let mut x = seed ^ (s as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                for b in v.iter_mut() {
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    *b = x as u8;
                }
                v
            })
            .collect();
        let slices: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut got = vec![[0u8; 32]; count];
        keccak256_many_into(&slices, &mut got);
        for (s, input) in inputs.iter().enumerate() {
            prop_assert_eq!(got[s], oracle(input));
        }
    }
}
