//! # Variational Stage Utilities (`var::utils`)
//!
//! This module provides specialized utility functions and iterators primarily used
//! during the construction of the variational Hamiltonian matrix in `var::ham_gen`.

use crate::utils::bits::ibclr;

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

