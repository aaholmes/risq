// Module for utility functions useful for fast variational Hamiltonian generation

use crate::utils::bits::ibclr;

// Iterate over configs with 1 electron moved:
pub fn remove_1e(config: u128) -> impl Iterator<Item = u128> {
    Remove1::new(config).into_iter()
}

// Iterate over configs with 2 electrons moved:
pub fn remove_2e(config: u128) -> impl Iterator<Item = u128> {
    Remove1::new(config).into_iter()
}


// Backend for remove_1e

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

#[derive(Clone, Copy)]
struct Remove1IntoIterator {
    config: u128,
    bits_left: u128,
}

impl Iterator for Remove1IntoIterator {
    type Item = u128;

    fn next(&mut self) -> Option<u128> {
        if self.bits_left == 0 {
            return None;
        };
        let next_bit: i32 = self.bits_left.trailing_zeros() as i32;
        self.bits_left &= !(1 << next_bit);
        Some(ibclr(self.config, next_bit))
    }
}


// Backend for remove_2e

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
        let bit: i32 = self.det.trailing_zeros() as i32;
        let init: u128 = self.det & !(1 << bit);
        BitPairsIntoIterator {
            config: self.config,
            first_bits_left: init,
            second_bits_left: init,
            first_bit: bit,
        }
    }
}

struct BitPairsIntoIterator {
    config: u128,
    first_bits_left: u128,
    second_bits_left: u128,
    first_bit: i32,
}

impl Iterator for BitPairsIntoIterator {
    type Item = u128;

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
