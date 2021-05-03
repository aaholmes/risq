// Iterator over excitations

// Eventually want something like
// for excite in EXCITE_GEN.truncated_excites(det, eps) {}
// which should return an iterator of Excites (singles and doubles)
// where singles are candidates whose matrix elements must be computed separately
// (because we want to check if they're new first before computing their matrix element)

use std::slice::Iter;
use std::collections::HashMap;

use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::utils::bits::{bit_pairs, bits, bits_and_bit_pairs};
use crate::wf::det::{Config, Det};
use crate::utils::iter::empty;

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

struct Exciter {
    det: Config,               // Needed to check if excitation is valid
    init: Orbs,               // Exciting electron pair
    excite_iter: Iter<'static, StoredExcite>, // Iterates over stored excitation
    eps: f64,
    is_alpha: Option<bool>,
}

impl IntoIterator for Exciter {
    type Item = Excite;
    type IntoIter = ExciterIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let mut out = ();
        out.det = self.det;
        out.excite_gen = self.excite_gen;
        out.epair_iter = bits_and_bit_pairs(&self.det);
        out.epair = self.epair_iter.next();
        out.target_iter = self.excite_gen.get(&out.epair);
        out.eps = self.eps;
        out
    }
}

struct ExciterIntoIterator {
    det: Config,               // Needed to check if excitation is valid
    excite_gen: HashMap<Orbs, Vec<StoredExcite>>, // Lookup table of all sorted excites
    epair_iter: dyn Iterator<Item=Orbs>, // Iterator over pairs of electrons in det to excite
    epair: Orbs,               // Current exciting electron pair
    target_iter: Iter<'static, StoredExcite>, // Iterates over sorted target orbs to excite to
    eps: f64,
}

impl Iterator for ExciterIntoIterator {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<&StoredExcite>;
        loop {
            excite = self.target_iter.next();
            match excite {
                None => {
                    // No more excitations left; done with this electron pair
                    let epair = self.epair_iter.next();
                    match epair {
                        // If no more electron pairs left to excite from, return None
                        None => return None,
                        // Otherwise, go to next epair
                        Some(e) => {
                            self.epair = e;
                            self.target_iter = self.excite_gen.get(&self.epair).unwrap().iter();
                        }
                    }
                }
                Some(exc) => {
                    // Check whether it meets threshold; if not, quit this sorted excitations list
                    if exc.abs_h >= self.eps {
                        // Only return this excitation if it is a valid excite for this det
                        let out_exc = Excite::Double {
                            0: Doub {
                                init: self.epair.clone(),
                                target: exc.target,
                                abs_h: exc.abs_h,
                                is_alpha: self.is_alpha,
                            },
                        };
                        if self.det.is_valid(&out_exc) {
                            // Found valid excitation; return it
                            return Some(out_exc);
                        } // Else, this excitation was not valid; go to next excitation (i.e., continue loop)
                    } else {
                        // Remaining excitations are smaller than eps; done with this electron pair
                        let epair = self.epair_iter.next();
                        match epair {
                            // If no more electron pairs left to excite from, return None
                            None => return None,
                            // Otherwise, go to next epair
                            Some(e) => {
                                self.epair = e;
                                self.target_iter = self.excite_gen.get(&self.epair).unwrap().iter();
                            }
                        }
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
