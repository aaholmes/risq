// Iterator over excitations

// Eventually want something like
// for excite in truncated_excites(det, excite_gen, eps) {}
// which should return an iterator of Excites (singles and doubles)
// where singles are candidates whose matrix elements must be computed separately
// (because we want to check if they're new first before computing their matrix element)

// use itertools::Itertools;
use std::collections::HashMap;
use std::iter::empty;

use crate::excite::init::ExciteGenerator;
use crate::excite::{Doub, Excite, OPair, Sing, StoredDoub, StoredSing};
use crate::utils::bits::{bit_pairs, bits};
use crate::wf::det::{Config, Det};

pub fn truncated_excites(
    det: Det,
    excite_gen: &ExciteGenerator,
    eps: f64,
) -> impl Iterator<Item = Excite> {
    // Returns an iterator over all double excitations that exceed eps
    // and all *candidate* single excitations that *may* exceed eps
    // The single excitation matrix elements must still be compared to eps
    let local_eps = eps / det.coeff.abs();
    {
        // Opposite spin double
        if excite_gen.max_opp_spin_doub >= local_eps {
            new_opp(det.config, &excite_gen.opp_spin_doub_generator, local_eps).into_iter()
        } else {
            empty()
        }
    }
    .chain({
        // Same spin double
        if excite_gen.max_same_spin_doub >= local_eps {
            new_same(det.config, &excite_gen.same_spin_doub_generator, local_eps).into_iter()
        } else {
            empty()
        }
    })
    .chain({
        // Single excitations
        if excite_gen.max_sing >= local_eps {
            new_sing(det.config, &excite_gen.sing_generator, local_eps).into_iter()
        } else {
            empty()
        }
    })
}

// Backend for truncated_excites

fn new_opp(
    det: Config,
    excite_gen: &HashMap<OPair, Vec<StoredDoub>>,
    eps: f64,
) -> impl Iterator<Item = Excite> {
    // Generate iterators over all valid opposite-spin excitations
    let mut excite_iter: dyn Iterator = empty();
    for i in bits(det.config.up) {
        for j in bits(det.config.dn) {
            excite_iter = excite_iter.chain(
                DoubTruncator {
                    det: det,
                    init: OPair(i, j),
                    excite_iter: excite_gen.get(&OPair(i, j)).unwrap(),
                    eps: eps,
                    is_alpha: None,
                }
                .into_iter(),
            );
        }
    }
    excite_iter
}

fn new_same(
    det: Config,
    excite_gen: &HashMap<OPair, Vec<StoredDoub>>,
    eps: f64,
) -> impl Iterator<Item = Excite> {
    // Generate iterators over all valid same-spin double excitations
    let mut excite_iter: dyn Iterator = empty();
    for (config, is_alpha) in [(det.config.up, true), (det.config.dn, false)] {
        for (i, j) in bit_pairs(config) {
            excite_iter = excite_iter.chain(
                DoubTruncator {
                    det: det,
                    init: OPair(i, j),
                    excite_iter: excite_gen.get(&OPair(i, j)).unwrap(),
                    eps: eps,
                    is_alpha: Some(is_alpha),
                }
                .into_iter(),
            );
        }
    }
    excite_iter
}

fn new_sing(
    det: Config,
    excite_gen: &Vec<Vec<StoredSing>>,
    eps: f64,
) -> impl Iterator<Item = Excite> {
    // Generate iterators over all potential single excitations
    let mut excite_iter: dyn Iterator = empty();
    for (config, is_alpha) in [(det.config.up, true), (det.config.dn, false)] {
        for i in bits(config) {
            excite_iter = excite_iter.chain(
                SingTruncator {
                    det: det,
                    init: i,
                    excite_iter: excite_gen[i],
                    eps: eps,
                    is_alpha: is_alpha,
                }
                .into_iter(),
            );
        }
    }
    excite_iter
}

struct DoubTruncator {
    det: Config,               // Needed to check if excitation is valid
    init: OPair,               // Exciting electron pair
    excite_iter: Box<dyn Iterator<Item=StoredDoub>>, // Iterates over stored excitations
    eps: f64,
    is_alpha: Option<bool>,
}

impl IntoIterator for DoubTruncator {
    type Item = Excite;
    type IntoIter = DoubTruncatorIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        DoubTruncatorIntoIterator {
            det: self.det,
            init: OPair,
            excite_iter: self.excite_iter,
            eps: self.eps,
            is_alpha: self.is_alpha,
        }
    }
}

struct DoubTruncatorIntoIterator {
    det: Config,               // Needed to check if excitation is valid
    init: OPair,               // Exciting electron pair
    excite_iter: Box<dyn Iterator<Item=StoredSing>>, // Iterates over stored excitations
    eps: f64,
    is_alpha: Option<bool>,
}

impl Iterator for DoubTruncatorIntoIterator {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<StoredDoub>;
        loop {
            excite = self.sorted_excites.next();
            match excite {
                None => {
                    // No more excitations left; go to next electron pair
                    return None;
                }
                Some(exc) => {
                    // Check whether it meets threshold; if not, quit this sorted excitations list
                    if exc.abs_h >= self.eps {
                        // Only return this excitation if it is a valid excite for this det
                        let out_exc = Excite::Double {
                            0: Doub {
                                init: self.init.copy(),
                                target: exc.target,
                                abs_h: exc.abs_h,
                                is_alpha: self.is_alpha,
                            },
                        };
                        if self.det.is_valid(&out_exc) {
                            // Found valid excitation; return it
                            return Some(out_exc);
                        } // Else, this excitation was not valid; go to next excitation
                    } else {
                        // Remaining excitations are smaller than eps; go to next electron pair
                        return None;
                    }
                }
            }
        }
    }
}

struct SingTruncator {
    det: Config,               // Needed to check if excitation is valid
    init: i32,                 // Exciting electron pair
    excite_iter: Box<dyn Iterator<Item=StoredSing>>, // Iterates over stored excitations
    eps: f64,
    is_alpha: bool,
}

impl IntoIterator for SingTruncator {
    type Item = Excite;
    type IntoIter = SingTruncatorIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        SingTruncatorIntoIterator {
            det: self.det,
            init: self.init,
            excite_iter: self.excite_iter,
            eps: self.eps,
            is_alpha: self.is_alpha,
        }
    }
}

struct SingTruncatorIntoIterator {
    det: Config,               // Needed to check if excitation is valid
    init: i32,                 // Exciting electron pair
    excite_iter: Box<dyn Iterator<Item=StoredSing>>, // Iterates over stored excitations
    eps: f64,
    is_alpha: bool,
}

impl Iterator for SingTruncatorIntoIterator {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<StoredSing>;
        loop {
            excite = self.sorted_excites.next();
            match excite {
                None => {
                    // No more excitations left; go to next electron pair
                    return None;
                }
                Some(exc) => {
                    // Check whether it meets threshold; if not, quit this sorted excitations list
                    if exc.abs_h >= self.eps {
                        // Only return this excitation if it is a valid excite for this det
                        let out_exc = Excite::Single {
                            0: Sing {
                                init: self.init.copy(),
                                target: exc.target,
                                max_abs_h: exc.max_abs_h,
                                is_alpha: self.is_alpha,
                            },
                        };
                        if self.det.is_valid(&out_exc) {
                            // Found valid excitation; return it
                            return Some(out_exc);
                        } // Else, this excitation was not valid; go to next excitation
                    } else {
                        // Remaining excitations are smaller than eps; go to next electron pair
                        return None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::excite::init::init_excite_generator;
    use crate::ham::read_ints::read_ints;
    use crate::ham::Ham;
    use crate::utils::read_input::{read_input, Global};

    #[test]
    fn test_iter() {
        println!("Reading input file");
        let ref global: Global = read_input("in.json").unwrap();

        println!("Reading integrals");
        let ham: Ham = read_ints(&global, "FCIDUMP");

        println!("Initializing excitation generator");
        let excite_gen: ExciteGenerator = init_excite_generator(&global, &ham);

        let det = Det {
            config: Config { up: 3, dn: 3 },
            coeff: 1.0,
            diag: 0.0,
        };

        let eps = 0.1;
        for excite in truncated_excites(det, &excite_gen, eps) {
            println!("Got here");
        }
    }
}
