// Useful bitwise functions:
// Bits(n) iterates over set bits in n, bit_pairs(n) iterates over pairs of set bits in n,
// plus functions for computing parity and getting and setting bits

use crate::wf::det::Config;

// Iterate over set bits in a u128
// Syntax: for i in bits(det: u128): loops over the set bits in det
pub fn bits(det: u128) -> impl Iterator<Item = i32> {
    Bits::new(det).into_iter()
}

// Iterate over pairs of set bits in a u128
// Syntax: for i in bit_pairs(det: u128): loops over the unique pairs of set bits in det
pub fn bit_pairs(det: u128) -> impl Iterator<Item = (i32, i32)> {
    BitPairs::new(det).into_iter()
}

// Iterate over set bits in a Config
pub fn det_bits(det: &Config) -> impl Iterator<Item = i32> {
    bits(det.up).chain(bits(det.dn))
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


// Backend for bits()

struct Bits {
    det: u128,
}

impl Bits {
    fn new(d: u128) -> Bits {
        Bits { det: d }
    }
}

impl IntoIterator for Bits {
    type Item = i32;
    type IntoIter = BitsIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        BitsIntoIterator {
            bits_left: self.det,
        }
    }
}

struct BitsIntoIterator {
    bits_left: u128,
}

impl Iterator for BitsIntoIterator {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.bits_left == 0 {
            return None;
        };
        let res: i32 = self.bits_left.trailing_zeros() as i32;
        self.bits_left &= !(1 << res);
        Some(res)
    }
}


// Backend for bit_pairs()

struct BitPairs {
    det: u128,
}

impl BitPairs {
    fn new(d: u128) -> BitPairs {
        BitPairs { det: d }
    }
}

impl IntoIterator for BitPairs {
    type Item = (i32, i32);
    type IntoIter = BitPairsIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let bit: i32 = self.det.trailing_zeros() as i32;
        let init: u128 = self.det & !(1 << bit);
        BitPairsIntoIterator {
            first_bits_left: init,
            second_bits_left: init,
            first_bit: bit,
        }
    }
}

struct BitPairsIntoIterator {
    first_bits_left: u128,
    second_bits_left: u128,
    first_bit: i32,
}

impl Iterator for BitPairsIntoIterator {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<(i32, i32)> {
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
        let res: i32 = self.second_bits_left.trailing_zeros() as i32;
        self.second_bits_left &= !(1 << res);
        Some((self.first_bit, res))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iters() {
        let det: u128 = 19273;
        for i in bits(det) {
            println!("{}", i);
        }
        for (i, j) in bit_pairs(det) {
            println!("{} {}", i, j);
        }
        assert_eq!(1, 1);
    }

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
