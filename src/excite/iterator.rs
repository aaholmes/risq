// Iterator over excitations

// Eventually want something like
// for excite in EXCITE_GEN.truncated_excites(det, eps) {}
// which should return an iterator of Excites (singles and doubles)
// where singles are candidates whose matrix elements must be computed separately
// (because we want to check if they're new first before computing their matrix element)

use std::slice::Iter;
use std::collections::HashMap;

use crate::excite::init::ExciteGenerator;
use crate::excite::{Doub, Excite, OPair, Sing, StoredDoub, StoredSing};
use crate::utils::bits::{bit_pairs, bits};
use crate::wf::det::{Config, Det};

impl ExciteGenerator {
    pub fn truncated_excites(
        &'static self,
        det: Det,
        //excite_gen: &ExciteGenerator,
        eps: f64,
    ) -> impl Iterator<Item=Excite> {
        // Returns an iterator over all double excitations that exceed eps
        // and all *candidate* single excitations that *may* exceed eps
        // The single excitation matrix elements must still be compared to eps
        let local_eps = eps / det.coeff.abs();
        // Opposite spin double excitations
        new_opp(det.config, self.max_opp_spin_doub, &self.opp_spin_doub_generator, local_eps)
        .chain(
            // Same spin double excitations
            new_same(det.config, self.max_same_spin_doub, &self.same_spin_doub_generator, local_eps)
        ).chain(
            // Single excitations
            new_sing(det.config, self.max_sing, &self.sing_generator, local_eps)
        )
    }
}

// Backend for EXCITE_GEN.truncated_excites()

fn new_opp(
    det: Config,
    max_opp_spin_doub: f64,
    excite_gen: &'static HashMap<OPair, Vec<StoredDoub>>,
    eps: f64,
) -> impl Iterator<Item=Excite> {
    // Generate iterators over all valid opposite-spin excitations
    if max_opp_spin_doub < eps {
        return None.iter();
    }
    let mut excite_iter: dyn Iterator<Item=Excite> = vec![].iter();
    for i in bits(det.up) {
        for j in bits(det.dn) {
            excite_iter = excite_iter.chain(
                DoubTruncator {
                    det: det,
                    init: OPair(i, j),
                    excite_iter: excite_gen.get(&OPair(i, j)).unwrap().iter(),
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
    max_same_spin_doub: f64,
    excite_gen: &'static HashMap<OPair, Vec<StoredDoub>>,
    eps: f64,
) -> impl Iterator<Item=Excite> {
    // Generate iterators over all valid same-spin double excitations
    if max_same_spin_doub < eps {
        return None.iter();
    }
    let mut excite_iter: dyn Iterator<Item=Excite> = vec![].iter();
    for (config, is_alpha) in [(det.up, true), (det.dn, false)].iter() {
        for (i, j) in bit_pairs(*config) {
            excite_iter = excite_iter.chain(
                DoubTruncator {
                    det: det,
                    init: OPair(i, j),
                    excite_iter: excite_gen.get(&OPair(i, j)).unwrap().iter(),
                    eps: eps,
                    is_alpha: Some(*is_alpha),
                }
                .into_iter(),
            );
        }
    }
    excite_iter
}

fn new_sing(
    det: Config,
    max_sing: f64,
    excite_gen: &'static Vec<Vec<StoredSing>>,
    eps: f64,
) -> impl Iterator<Item=Excite> {
    // Generate iterators over all potential single excitations
    if max_sing < eps {
        return None.iter();
    }
    let mut excite_iter: dyn Iterator<Item=Excite> = vec![].iter();
    for (config, is_alpha) in [(det.up, true), (det.dn, false)].iter() {
        for i in bits(*config) {
            excite_iter = excite_iter.chain(
                SingTruncator {
                    det: det,
                    init: i,
                    excite_iter: excite_gen[i as usize].iter(),
                    eps: eps,
                    is_alpha: *is_alpha,
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
    excite_iter: Iter<'static, StoredDoub>, // Iterates over stored excitation
    eps: f64,
    is_alpha: Option<bool>,
}

impl IntoIterator for DoubTruncator {
    type Item = Excite;
    type IntoIter = DoubTruncatorIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        DoubTruncatorIntoIterator {
            det: self.det,
            init: self.init,
            excite_iter: self.excite_iter,
            eps: self.eps,
            is_alpha: self.is_alpha,
        }
    }
}

struct DoubTruncatorIntoIterator {
    det: Config,               // Needed to check if excitation is valid
    init: OPair,               // Exciting electron pair
    excite_iter: Iter<'static, StoredDoub>, // Iterates over stored excitations
    eps: f64,
    is_alpha: Option<bool>,
}

impl Iterator for DoubTruncatorIntoIterator {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<&StoredDoub>;
        loop {
            excite = self.excite_iter.next();
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
                                init: self.init.clone(),
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
    excite_iter: Iter<'static, StoredSing>, // Iterates over stored excitations
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
    excite_iter: Iter<'static, StoredSing>, // Iterates over stored excitations
    eps: f64,
    is_alpha: bool,
}

impl Iterator for SingTruncatorIntoIterator {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<&StoredSing>;
        loop {
            excite = self.excite_iter.next();
            match excite {
                None => {
                    // No more excitations left; go to next electron pair
                    return None;
                }
                Some(exc) => {
                    // Check whether it meets threshold; if not, quit this sorted excitations list
                    if exc.max_abs_h >= self.eps {
                        // Only return this excitation if it is a valid excite for this det
                        let out_exc = Excite::Single {
                            0: Sing {
                                init: self.init,
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
        lazy_static! {
            static ref GLOBAL: Global = read_input("in.json").unwrap();
        }
    
        println!("Reading integrals");
        lazy_static! {
            static ref HAM: Ham = read_ints(&GLOBAL, "FCIDUMP");
        }
    
        println!("Initializing excitation generator");
        lazy_static! {
            static ref EXCITE_GEN: ExciteGenerator = init_excite_generator(&GLOBAL, &HAM);
        }

        let det = Det {
            config: Config { up: 3, dn: 3 },
            coeff: 1.0,
            diag: 0.0,
        };

        let eps = 0.1;
        println!("About to iterate!");
        for excite in EXCITE_GEN.truncated_excites(det, eps) {
            println!("Got here");
        }
    }
}
