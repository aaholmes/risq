//! # Bit Manipulation Utilities (`utils::bits`)
//!
//! This module provides helper functions and iterators for common bitwise operations,
//! particularly useful for manipulating Slater determinant configurations (`Config`)
//! represented as `u128` bitstrings.

use crate::excite::init::ExciteGenerator;
use crate::excite::Orbs;
use crate::wf::det::Config;

/// Creates an iterator that yields the indices (0-based) of the set bits in `n`.
///
/// Example: `bits(6)` (binary `110`) would yield `1`, then `2`.
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

/// Creates an iterator that yields unique pairs `(i, j)` where `i < j` and both
/// bits `i` and `j` are set in `n`.
///
/// Example: `bits(7)` (binary `111`) would yield `(0, 1)`, `(0, 2)`, `(1, 2)`.
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

/// Creates an iterator over all occupied spatial orbital indices in a determinant `det`,
/// irrespective of spin. Combines the results of `bits(det.up)` and `bits(det.dn)`.
pub fn det_bits(det: &Config) -> impl Iterator<Item = i32> {
    bits(det.up).chain(bits(det.dn))
}

/// Creates an iterator over all pairs `(i, j)` where `i` is an occupied alpha orbital
/// and `j` is an occupied beta orbital in the determinant `det`.
/// Represents opposite-spin electron pairs.
pub fn product_bits(det: &Config) -> impl Iterator<Item = (i32, i32)> {
    let dn = det.dn;
    bits(det.up).flat_map(move |i| bits(dn).map(move |j| (i, j)))
}

/// Creates an iterator yielding all single occupied valence orbitals and pairs of occupied
/// valence orbitals within a determinant `det`.
///
/// Filters orbitals based on the `valence` mask in `excite_gen`.
/// Yields tuples `(is_alpha, orbs)`, where `is_alpha` indicates the spin channel
/// (Some(true) for alpha, Some(false) for beta, None for opposite-spin pairs) and
/// `orbs` is the `Orbs::Single` or `Orbs::Double` representing the electron(s).
/// This is useful for iterating through all possible starting points for excitations.
pub fn valence_elecs_and_epairs(
    det: &Config,
    excite_gen: &ExciteGenerator,
) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    let valence_det: Config = Config {
        up: det.up & excite_gen.valence,
        dn: det.dn & excite_gen.valence,
    };
    epairs(&valence_det).chain(elecs(&valence_det))
}

/// Creates an iterator yielding all single occupied orbitals in `det`.
///
/// Yields tuples `(Some(is_alpha), Orbs::Single(p))`, where `is_alpha` is true for alpha
/// electrons and false for beta electrons.
pub fn elecs(det: &Config) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    bits(det.up)
        .map(|p| (Some(true), Orbs::Single(p)))
        .chain(bits(det.dn).map(|p| (Some(false), Orbs::Single(p))))
}

/// Creates an iterator yielding single occupied *valence* orbitals in `det`.
///
/// Filters the output of `elecs` using the `valence` mask from `excite_gen`.
/// Yields tuples `(Some(is_alpha), Orbs::Single(p))`.
pub fn valence_elecs(
    det: &Config,
    excite_gen: &ExciteGenerator,
) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    let valence_det: Config = Config {
        up: det.up & excite_gen.valence,
        dn: det.dn & excite_gen.valence,
    };
    elecs(&valence_det)
}

/// Creates an iterator yielding all pairs of occupied orbitals in `det`.
///
/// Includes both opposite-spin pairs (`(None, Orbs::Double(p_up, q_dn))`) generated
/// via `product_bits`, and same-spin pairs (`(Some(spin), Orbs::Double(p, q))`)
/// generated via `bit_pairs` for both alpha and beta electrons.
pub fn epairs(det: &Config) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    product_bits(det).map(|pq| (None, Orbs::Double(pq))).chain(
        bit_pairs(det.up)
            .map(|pq| (Some(true), Orbs::Double(pq)))
            .chain(bit_pairs(det.dn).map(|pq| (Some(false), Orbs::Double(pq)))),
    )
}

/// Creates an iterator yielding pairs of occupied *valence* orbitals in `det`.
///
/// Filters the output of `epairs` using the `valence` mask from `excite_gen`.
/// Yields tuples `(is_alpha, Orbs::Double(p, q))`.
pub fn valence_epairs(
    det: &Config,
    excite_gen: &ExciteGenerator,
) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    let valence_det: Config = Config {
        up: det.up & excite_gen.valence,
        dn: det.dn & excite_gen.valence,
    };
    epairs(&valence_det)
}

/// Creates an iterator yielding all single occupied orbitals and all pairs of occupied orbitals.
/// Combines the output of `epairs` and `elecs`.
pub fn orbs(det: &Config) -> impl Iterator<Item = (Option<bool>, Orbs)> {
    epairs(det).chain(
        bits(det.up)
            .map(|p| (Some(true), Orbs::Single(p)))
            .chain(bits(det.dn).map(|p| (Some(false), Orbs::Single(p)))),
    )
}

// Bit operations named after Fortran intrinsics...

/// Sets the `b`-th bit (0-based) of `n` to 1. Equivalent to Fortran's `IBSET`.
pub fn ibset(n: u128, b: i32) -> u128 {
    n | (1 << b)
}

/// Clears the `b`-th bit (0-based) of `n` to 0. Equivalent to Fortran's `IBCLR`.
pub fn ibclr(n: u128, b: i32) -> u128 {
    n & !(1 << b)
}

/// Tests the `b`-th bit (0-based) of `n`. Returns `true` if the bit is 1, `false` otherwise.
/// Equivalent to Fortran's `BTEST`.
pub fn btest(n: u128, b: i32) -> bool {
    !(n & (1 << b) == 0)
}

/// Calculates the parity of the number of set bits in `n` using a fast parallel algorithm.
/// Returns `1` if the number of set bits is even, `-1` if it is odd.
/// Useful for determining the sign factor when applying fermionic operators.
pub fn parity(mut n: u128) -> i32 {
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

    // Removed commented-out test function `test_iters`

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
