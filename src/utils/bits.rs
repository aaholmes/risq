// Useful bitwise functions:
// Bits(n) iterates over set bits in n, bit_pairs(n) iterates over pairs of set bits in n,
// plus functions for computing parity and getting and setting bits

use crate::wf::det::Config;
// use crate::excite::Orbs;

// Iterate over set bits in a u128
// Syntax: for i in bits(det: u128): loops over the set bits in det
pub fn bits(det: u128) -> impl Iterator<Item = i32> {
    Bits::new(det).into_iter()
}

// Iterate over pairs of set bits in a u128
// Syntax: for (i, j) in bit_pairs(det: u128): loops over the unique pairs of set bits in det
pub fn bit_pairs(det: u128) -> impl Iterator<Item = (i32, i32)> {
    BitPairs::new(det).into_iter()
}

// // Single iterator (same as bits, but return type is Orbs)
// pub fn sing_iter(det: u128) -> impl Iterator<Item = Orbs> {
//     SingIter::new(det).into_iter()
// }
//
// // Same-spin iterator (same as bit_pairs, but return type is Orbs)
// pub fn same_iter(det: u128) -> impl Iterator<Item = Orbs> {
//     SameIter::new(det).into_iter()
// }
//
// // Opposite-spin iterator
// pub fn opp_iter(det: &Config) -> Box<dyn Iterator<Item = Orbs>> {
//     //pub fn opp_iter(det: &Config) -> impl Iterator<Item = Orbs> {
//     OppIter::new(det).into_iter()
// }

// Iterate over set bits in a Config
pub fn det_bits(det: &Config) -> impl Iterator<Item = i32> {
    bits(det.up).chain(bits(det.dn))
}

// Iterate over all bits and bit pairs in a Config
// Output is (orbs, is_alpha), where orbs can be either one or two orbs, and is_alpha is either
// None (for opposite doubles) or Some(p), where p is true for alpha, false for beta
// pub fn bits_and_bit_pairs(det: &Config) -> impl Iterator<Item = (Orbs, Option<bool>)> {
//     BitsAndBitPairs::new(det).into_iter()
// }

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

#[derive(Clone, Copy)]
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

// Backend for sing_iter (same as bits but return type is Orbs)

// struct SingIter {
//     det: u128,
// }
//
// impl SingIter {
//     fn new(d: u128) -> SingIter {
//         SingIter { det: d }
//     }
// }
//
// impl IntoIterator for SingIter {
//     type Item = Orbs;
//     type IntoIter = SingIterIntoIterator;
//
//     fn into_iter(self) -> Self::IntoIter {
//         SingIterIntoIterator {
//             bits_left: self.det,
//         }
//     }
// }
//
// struct SingIterIntoIterator {
//     bits_left: u128,
// }
//
// impl Iterator for SingIterIntoIterator {
//     type Item = Orbs;
//
//     fn next(&mut self) -> Option<Orbs> {
//         if self.bits_left == 0 {
//             return None;
//         };
//         let res: i32 = self.bits_left.trailing_zeros() as i32;
//         self.bits_left &= !(1 << res);
//         Some(Orbs::Single(res))
//     }
// }

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

// struct SameIter {
//     det: u128,
// }
//
// impl SameIter {
//     fn new(d: u128) -> SameIter {
//         SameIter { det: d }
//     }
// }
//
// impl IntoIterator for SameIter {
//     type Item = Orbs;
//     type IntoIter = SameIterIntoIterator;
//
//     fn into_iter(self) -> Self::IntoIter {
//         let bit: i32 = self.det.trailing_zeros() as i32;
//         let init: u128 = self.det & !(1 << bit);
//         SameIterIntoIterator {
//             first_bits_left: init,
//             second_bits_left: init,
//             first_bit: bit,
//         }
//     }
// }
//
// struct SameIterIntoIterator {
//     first_bits_left: u128,
//     second_bits_left: u128,
//     first_bit: i32,
// }
//
// impl Iterator for SameIterIntoIterator {
//     type Item = Orbs;
//
//     fn next(&mut self) -> Option<Orbs> {
//         if self.first_bits_left == 0 {
//             return None;
//         };
//         if self.second_bits_left == 0 {
//             let res: i32 = self.first_bits_left.trailing_zeros() as i32;
//             self.first_bits_left &= !(1 << res);
//             if self.first_bits_left == 0 {
//                 return None;
//             };
//             self.second_bits_left = self.first_bits_left;
//             self.first_bit = res;
//         };
//         let res: i32 = self.second_bits_left.trailing_zeros() as i32;
//         self.second_bits_left &= !(1 << res);
//         Some(Orbs::Double((self.first_bit, res)))
//     }
// }
//
// struct OppIter {
//     det: &'static Config,
// }
//
// impl OppIter {
//     fn new(d: &'static Config) -> OppIter {
//         OppIter { det: d }
//     }
// }
//
// impl IntoIterator for OppIter {
//     type Item = Orbs;
//     type IntoIter = OppIterIntoIterator;
//
//     fn into_iter(self) -> Self::IntoIter {
//         OppIterIntoIterator {
//             iter: iproduct!(bits(self.det.up), bits(self.det.dn))
//         }
//     }
// }
//
// struct OppIterIntoIterator {
//     iter: dyn Iterator<Item=(i32, i32)>
// }
//
// impl Iterator for OppIterIntoIterator {
//     type Item = Orbs;
//
//     fn next(&mut self) -> Option<Orbs> {
//         match self.iter.next() {
//             None => None,
//             Some(pq) => Some(Orbs::Double(pq))
//         }
//     }
// }

// Backend for bit_and_bit_pairs()

// struct BitsAndBitPairs {
//     det: &'static Config
// }
//
// impl BitsAndBitPairs {
//     fn new(det: &Config) -> BitsAndBitPairs {
//         BitsAndBitPairs{
//             det: det
//         }
//     }
// }
//
// impl IntoIterator for BitsAndBitPairs {
//     type Item = i32;
//     type IntoIter = BitsAndBitPairsIntoIterator;
//
//     fn into_iter(self) -> Self::IntoIter {
//         let mut into_iter = ();
//         into_iter.iters = [
//             (iproduct!(bits(self.up), bits(self.dn)), None), // Opposite spin double
//             (bit_pairs(self.up).iter(), Some(true)), // Same spin, up
//             (bit_pairs(self.dn).iter(), Some(false)), // Same spin, dn
//             (bits(self.up).iter(), Some(true)), // Single, up
//             (bits(self.dn).iter(), Some(false)) // Single, dn
//         ].iter();
//         into_iter.curr = into_iter.iters.next();
//         into_iter
//     }
// }
//
// struct BitsAndBitPairsIntoIterator {
//     iters: dyn Iterator,
//     curr: dyn Iterator,
// }
//
// impl Iterator for BitsAndBitPairsIntoIterator {
//     type Item = i32;
//
//     fn next(&mut self) -> Option<(Orbs, Option<bool>)> {
//         loop {
//             match self.curr.next() {
//                 None => {
//                     // Go to next iter
//                     match self.into_iter.next() {
//                         None => {
//                             // Done with all iters
//                             return None;
//                         },
//                         Some(i) => self.curr = i
//                     }
//                 },
//                 Some(i) => {
//                     // Return this one in a unified form
//                     match i.0 {
//                         Orbs::Double(pq) => return Some((Orbs::Double(pq), i.1)),
//                         Orbs::Single(p) => return Some((Orbs::Single(p), i.1))
//                     }
//                 }
//             }
//         }
//     }
// }

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
