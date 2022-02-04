//! Useful bitwise functions
//! Bits(n) iterates over set bits in n, bit_pairs(n) iterates over pairs of set bits in n,
//! plus functions for computing parity and getting and setting bits

use crate::excite::init::ExciteGenerator;
use crate::excite::Orbs;
use crate::wf::det::Config;

/// Iterate over set bits in a u128
pub fn bits(n: u128) -> impl Iterator<Item = i32> {
    let mut bits_left = n;
    std::iter::from_fn(move || {
        if bits_left == 0 {
            None
        } else {
            let res: i32 = bits_left.trailing_zeros() as i32;
            bits_left &= !(1 << res);
            Some(res)
        }
    })
}

/// Iterate over pairs of set bits in a u128
pub fn bit_pairs(n: u128) -> impl Iterator<Item = (i32, i32)> {
    let mut first_bit: i32 = n.trailing_zeros() as i32;
    let mut first_bits_left: u128 = n & !(1 << first_bit);
    let mut second_bits_left: u128 = first_bits_left;
    std::iter::from_fn(move || {
        if first_bits_left == 0 {
            None
        } else {
            if second_bits_left == 0 {
                first_bit = first_bits_left.trailing_zeros() as i32;
                first_bits_left &= !(1 << first_bit);
                if first_bits_left == 0 {
                    return None;
                };
                second_bits_left = first_bits_left;
            };
            let second_bit: i32 = second_bits_left.trailing_zeros() as i32;
            second_bits_left &= !(1 << second_bit);
            Some((first_bit, second_bit))
        }
    })
}

/// Iterate over the union of bits(det.up) and bits(det.dn), ignoring spin
pub fn det_bits(det: &Config) -> impl Iterator<Item = i32> {
    bits(det.up).chain(bits(det.dn))
}

/// Iterate over the cartesian product of bits(det.up) and bits(det.dn)
pub fn product_bits(det: &Config) -> impl Iterator<Item = (i32, i32)> {
    let dn = det.dn;
    bits(det.up).flat_map(move |i| bits(dn).map(move |j| (i, j)))
}

/// Iterate over the pairs of occupied orbs in a det, returns both the orbs (as an Orbs::Double)
/// and is_alpha, which is None of opposite-spin and Some(bool) for same spin
pub fn epairs(det: &Config) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    product_bits(det).map(|pq| (None, Orbs::Double(pq))).chain(
        bit_pairs(det.up)
            .map(|pq| (Some(true), Orbs::Double(pq)))
            .chain(bit_pairs(det.dn).map(|pq| (Some(false), Orbs::Double(pq)))),
    )
}

/// Iterate over the pairs of occupied valence orbs in a det, returns both the orbs (as an Orbs::Double)
/// and is_alpha, which is None of opposite-spin and Some(bool) for same spin
pub fn valence_epairs(det: &Config, excite_gen: &ExciteGenerator) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    let valence_det: Config = Config { up: det.up & excite_gen.valence, dn: det.dn & excite_gen.valence };
    epairs(&valence_det)
}

/// Iterate over all occupied orbs (as Orbs::Single) and orb pairs (as Orbs::Double),
/// also returns is_alpha for each
pub fn orbs(det: &Config) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    epairs(det).chain(
        bits(det.up)
            .map(|p| (Some(true), Orbs::Single(p)))
            .chain(bits(det.dn).map(|p| (Some(false), Orbs::Single(p)))),
    )
}

// Bit operations named after Fortran intrinsics...

pub fn ibset(n: u128, b: i32) -> u128 {
    n | (1 << b)
}

pub fn ibclr(n: u128, b: i32) -> u128 {
    n & !(1 << b)
}

pub fn btest(n: u128, b: i32) -> bool {
    !(n & (1 << b) == 0)
}

pub fn parity(mut n: u128) -> i32 {
    // Returns 1 if even number of bits, -1 if odd number
    n ^= n >> 64;
    n ^= n >> 32;
    n ^= n >> 16;
    n ^= n >> 8;
    n ^= n >> 4;
    n ^= n >> 2;
    n ^= n >> 1;
    1 - 2 * ((n & 1) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_iters() {
    //     let det: u128 = 19273;
    //     println!("Bits:");
    //     for i in bits(det) {
    //         println!("{}", i);
    //     }
    //     println!("Bit pairs:");
    //     for (i, j) in bit_pairs(det) {
    //         println!("{} {}", i, j);
    //     }
    //     println!("Bits and bit pairs:");
    //     for bbp in bits_and_bit_pairs(&Config{up: det, dn: det}) {
    //         match bbp.1 {
    //             None => {
    //                 match bbp.0 {
    //                     Orbs::Double((p, q)) => println!("Opposite spin: ({}, {})", p, q),
    //                     Orbs::Single(p) => println!("Should not happen")
    //                 }
    //             },
    //             Some(is_alpha) => {
    //                 match bbp.0 {
    //                     Orbs::Double((p, q)) => {
    //                         if is_alpha {
    //                             println!("Same spin, up: ({}, {})", p, q);
    //                         } else {
    //                             println!("Same spin, dn: ({}, {})", p, q);
    //                         }
    //                     },
    //                     Orbs::Single(p) => {
    //                         if is_alpha {
    //                             println!("Single, up: {}", p);
    //                         } else {
    //                             println!("Single, dn: {}", p);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     assert_eq!(1, 1);
    // }

    fn parity_brute_force(n: u128) -> i32 {
        let mut out: i32 = 0;
        for _ in bits(n) {
            out ^= 1;
        }
        1 - 2 * out
    }

    #[test]
    fn test_parity() {
        for i in vec![
            14,
            15,
            27,
            1919,
            4958202,
            15 << 64,
            1 << 127,
            (1 << 126) + (1 << 65),
        ] {
            println!("Parity({}) = {} = {}", i, parity(i), parity_brute_force(i));
            assert_eq!(parity(i), parity_brute_force(i));
        }
    }
}
