//! Utility functions useful for fast variational Hamiltonian generation

use crate::utils::bits::ibclr;
use crate::wf::det::Config;

/// Iterate over configs with 1 electron removed:
pub fn remove_1e(config: u128) -> impl Iterator<Item = u128> {
    Remove1::new(config).into_iter()
}

/// Iterate over configs with 2 electrons removed:
pub fn remove_2e(config: u128) -> impl Iterator<Item = u128> {
    Remove2::new(config).into_iter()
}

/// Iterate over intersection of 2 sorted lists:
pub fn intersection<'a>(
    v1: &'a Vec<(Config, usize)>,
    v2: &'a Vec<(Config, usize)>,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    Intersection::new(v1, v2).into_iter()
}

/// Backend for remove_1e
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

/// Backend for remove_2e
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

struct Remove2IntoIterator {
    config: u128,
    first_bits_left: u128,
    second_bits_left: u128,
    first_bit: i32,
}

impl Iterator for Remove2IntoIterator {
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

/// Backend for intersection
struct Intersection<'a> {
    v1: &'a Vec<(Config, usize)>,
    v2: &'a Vec<(Config, usize)>,
}

impl Intersection<'_> {
    fn new<'a>(v1: &'a Vec<(Config, usize)>, v2: &'a Vec<(Config, usize)>) -> Intersection<'a> {
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
                n1: n1,
                n2: n2,
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

struct IntersectionIntoIterator<'a> {
    linear: bool, // if true, use N+M algorithm; else, use N log M algorithm
    v1: &'a Vec<(Config, usize)>,
    v2: &'a Vec<(Config, usize)>,
    n1: usize,
    n2: usize,
    ind1: usize,
    ind2: usize,
}

impl<'a> Iterator for IntersectionIntoIterator<'a> {
    type Item = (usize, usize);

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
