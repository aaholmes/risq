//! # Variational Stage Utilities (`var::utils`)
//!
//! This module provides specialized utility functions and iterators primarily used
//! during the construction of the variational Hamiltonian matrix in `var::ham_gen`.

use crate::utils::bits::ibclr;
use crate::wf::det::Config;

/// Creates an iterator that yields all possible configurations (`u128` bitstrings)
/// obtained by removing exactly one set bit (electron) from the input `config`.
pub fn remove_1e(config: u128) -> impl Iterator<Item = u128> {
    Remove1::new(config).into_iter()
}

/// Creates an iterator that yields all possible configurations (`u128` bitstrings)
/// obtained by removing exactly two set bits (electrons) from the input `config`.
pub fn remove_2e(config: u128) -> impl Iterator<Item = u128> {
    Remove2::new(config).into_iter()
}

/// Creates an iterator that finds the intersection of two sorted slices based on the `Config` key.
///
/// Given two slices `v1` and `v2`, where each element is a tuple `(Config, usize)`,
/// and assuming both slices are sorted lexicographically by `Config` (first `up`, then `dn`),
/// this iterator yields pairs `(index1, index2)` such that `v1[index1].0 == v2[index2].0`.
///
/// It adaptively chooses between a linear merge-like algorithm (O(N+M)) and a binary
/// search based algorithm (O(N log M) or O(M log N)) depending on the estimated cost.
/// This is used to efficiently find pairs of determinants connected by double excitations
/// in `var::ham_gen`.
pub fn intersection<'a>(
    v1: &'a [(Config, usize)], // First sorted slice of (Determinant Config, Original Index)
    v2: &'a [(Config, usize)], // Second sorted slice
) -> impl Iterator<Item = (usize, usize)> + 'a { // Yields pairs of original indices
    Intersection::new(v1, v2).into_iter()
}

/// Internal struct holding the configuration for the `remove_1e` iterator.
struct Remove1 {
    config: u128,
}

impl Remove1 {
    fn new(config: u128) -> Remove1 {
        Remove1 { config }
    }
}

impl IntoIterator for Remove1 {
    type Item = u128;
    type IntoIter = Remove1IntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        Remove1IntoIterator {
            config: self.config,
            bits_left: self.config,
        }
    }
}

/// The iterator state for `remove_1e`.
#[derive(Clone, Copy)]
struct Remove1IntoIterator {
    /// The original configuration.
    config: u128,
    /// A copy of the configuration used to track remaining bits to process.
    bits_left: u128,
}

impl Iterator for Remove1IntoIterator {
    type Item = u128;

    /// Yields the next configuration with one bit removed.
    /// Finds the lowest set bit in `bits_left`, removes it from `bits_left`,
    /// and returns the original `config` with that bit cleared.
    fn next(&mut self) -> Option<u128> {
        if self.bits_left == 0 {
            return None;
        };
        let next_bit: i32 = self.bits_left.trailing_zeros() as i32;
        self.bits_left &= !(1 << next_bit);
        Some(ibclr(self.config, next_bit))
    }
}

/// Internal struct holding the configuration for the `remove_2e` iterator.
struct Remove2 {
    config: u128,
}

impl Remove2 {
    fn new(config: u128) -> Remove2 {
        Remove2 { config }
    }
}

impl IntoIterator for Remove2 {
    type Item = u128;
    type IntoIter = Remove2IntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let bit: i32 = self.config.trailing_zeros() as i32;
        let init: u128 = self.config & !(1 << bit);
        Remove2IntoIterator {
            config: self.config,
            first_bits_left: init,
            second_bits_left: init,
            first_bit: bit,
        }
    }
}

/// The iterator state for `remove_2e`.
struct Remove2IntoIterator {
    /// The original configuration.
    config: u128,
    /// Tracks the remaining bits for the outer loop (first bit removal).
    first_bits_left: u128,
    /// Tracks the remaining bits for the inner loop (second bit removal).
    second_bits_left: u128,
    /// The index of the bit removed by the outer loop in the current iteration.
    first_bit: i32,
}

impl Iterator for Remove2IntoIterator {
    type Item = u128;

    /// Yields the next configuration with two bits removed.
    /// Iterates through pairs of set bits `(first_bit, second_bit)` where `first_bit < second_bit`,
    /// and returns the original `config` with both bits cleared.
    fn next(&mut self) -> Option<u128> {
        if self.first_bits_left == 0 {
            return None;
        };
        if self.second_bits_left == 0 {
            let res: i32 = self.first_bits_left.trailing_zeros() as i32;
            self.first_bits_left &= !(1 << res);
            if self.first_bits_left == 0 {
                return None;
            };
            self.second_bits_left = self.first_bits_left;
            self.first_bit = res;
        };
        let second_bit: i32 = self.second_bits_left.trailing_zeros() as i32;
        self.second_bits_left &= !(1 << second_bit);
        Some(ibclr(ibclr(self.config, self.first_bit), second_bit))
    }
}

/// Internal struct holding references to the slices for the `intersection` iterator.
struct Intersection<'a> {
    v1: &'a [(Config, usize)],
    v2: &'a [(Config, usize)],
}

impl Intersection<'_> {
    fn new<'a>(v1: &'a [(Config, usize)], v2: &'a [(Config, usize)]) -> Intersection<'a> {
        Intersection { v1, v2 }
    }
}

impl<'a> IntoIterator for Intersection<'a> {
    type Item = (usize, usize);
    type IntoIter = IntersectionIntoIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let n1 = self.v1.len();
        let n2 = self.v2.len();
        if n1 <= n2 {
            IntersectionIntoIterator {
                linear: n1 + n2 <= n1 * ((n2 as f32).log2()).ceil() as usize,
                v1: self.v1,
                v2: self.v2,
                n1,
                n2,
                ind1: 0,
                ind2: 0,
            }
        } else {
            IntersectionIntoIterator {
                linear: n1 + n2 <= n2 * ((n1 as f32).log2()).ceil() as usize,
                v1: self.v2,
                v2: self.v1,
                n1: n2,
                n2: n1,
                ind1: 0,
                ind2: 0,
            }
        }
    }
}

/// The iterator state for `intersection`.
struct IntersectionIntoIterator<'a> {
    /// Flag indicating whether to use the linear (O(N+M)) or binary search (O(N log M)) algorithm.
    linear: bool,
    /// Reference to the first slice (potentially swapped so v1 is shorter or equal).
    v1: &'a [(Config, usize)],
    /// Reference to the second slice.
    v2: &'a [(Config, usize)],
    /// Length of v1.
    n1: usize,
    /// Length of v2.
    n2: usize,
    /// Current index in v1.
    ind1: usize,
    /// Current index in v2 (used only by the linear algorithm).
    ind2: usize,
}

impl<'a> Iterator for IntersectionIntoIterator<'a> {
    type Item = (usize, usize);

    /// Yields the next pair of indices `(index1, index2)` corresponding to matching `Config` keys.
    /// Implements both the linear merge-style comparison and the binary search approach based
    /// on the `linear` flag determined during initialization.
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.linear {
            while self.ind1 != self.n1 && self.ind2 != self.n2 {
                if self.v1[self.ind1].0 == self.v2[self.ind2].0 {
                    let res = Some((self.v1[self.ind1].1, self.v2[self.ind2].1));
                    self.ind1 += 1;
                    self.ind2 += 1;
                    return res;
                } else if self.v1[self.ind1].0.up < self.v2[self.ind2].0.up
                    || (self.v1[self.ind1].0.up == self.v2[self.ind2].0.up
                        && self.v1[self.ind1].0.dn < self.v2[self.ind2].0.dn)
                {
                    self.ind1 += 1;
                } else {
                    self.ind2 += 1;
                }
            }
            None
        } else {
            while self.ind1 != self.n1 {
                // search for ind1 in v2
                let ind2 = self.v2.partition_point(|&(det, _)| {
                    det.up < self.v1[self.ind1].0.up
                        || (det.up == self.v1[self.ind1].0.up && det.dn < self.v1[self.ind1].0.dn)
                });
                if ind2 < self.n2 {
                    if self.v1[self.ind1].0 == self.v2[ind2].0 {
                        // Found by binary search
                        let res = Some((self.v1[self.ind1].1, self.v2[ind2].1));
                        self.ind1 += 1;
                        return res;
                    } else {
                        // Nothing found; go to next
                        self.ind1 += 1;
                    }
                } else {
                    // Nothing found; go to next
                    self.ind1 += 1;
                }
            }
            None
        }
    }
}
