// Iterator over excitations

// Eventually want something like
// for excite in EXCITE_GEN.truncated_excites(det, eps) {}
// which should return an iterator of Excites (singles and doubles)
// where singles are candidates whose matrix elements must be computed separately
// (because we want to check if they're new first before computing their matrix element)

// use std::slice::Iter;
use std::collections::HashMap;

use crate::excite::init::ExciteGenerator;
use crate::excite::{Excite, Orbs, StoredExcite};
use crate::utils::bits::{opp_iter, same_iter, sing_iter};
use crate::wf::det::{Config, Det};
// use crate::utils::iter::empty;

impl ExciteGenerator {
    pub fn truncated_excites(
        &'static self,
        det: Det,
        eps: f64,
    ) -> impl Iterator<Item=Excite> {
        // Returns an iterator over all double excitations that exceed eps
        // and all *candidate* single excitations that *may* exceed eps
        // The single excitation matrix elements must still be compared to eps
        // TODO: Put in max_doub, etc
        let local_eps: f64 = eps / det.coeff.abs();
        Exciter {
            det: &det.config,
            epair_iter: opp_iter(&det.config),
            sorted_excites: &self.opp_doub_generator,
            eps: local_eps,
            is_alpha: None
        }.into_iter().chain(
            Exciter {
                det: &det.config,
                epair_iter: same_iter(det.config.up),
                sorted_excites: &self.same_doub_generator,
                eps: local_eps,
                is_alpha: Some(true)
            }.into_iter()
        ).chain(
            Exciter {
                det: &det.config,
                epair_iter: same_iter(det.config.dn),
                sorted_excites: &self.same_doub_generator,
                eps: local_eps,
                is_alpha: Some(false)
            }.into_iter()
        ).chain(
            Exciter {
                det: &det.config,
                epair_iter: sing_iter(det.config.up),
                sorted_excites: &self.sing_generator,
                eps: local_eps,
                is_alpha: Some(true)
            }.into_iter()
        ).chain(
            Exciter {
                det: &det.config,
                epair_iter: sing_iter(det.config.dn),
                sorted_excites: &self.sing_generator,
                eps: local_eps,
                is_alpha: Some(false)
            }.into_iter()
        )
    }
}

// Backend for EXCITE_GEN.truncated_excites()

#[derive(Default)]
struct Exciter {
    det: &'static Config,               // Needed to check if excitation is valid
    epair_iter: Box<dyn Iterator<Item=Orbs>>,
    sorted_excites: &'static HashMap<Orbs, Vec<StoredExcite>>,
    eps: f64,
    is_alpha: Option<bool>
}

impl IntoIterator for Exciter {
    type Item = Excite;
    type IntoIter = ExciterIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let mut out = ExciterIntoIterator::default();
        out.det = self.det;
        out.epair_iter = self.epair_iter;
        // Initialize to first electron or pair
        out.epair = out.epair_iter.next().unwrap();
        out.sorted_excites = self.sorted_excites;
        out.target_iter = out.sorted_excites.get_key(&self.epair).unwrap().iter();
        out.eps = self.eps;
        out.is_alpha = self.is_alpha;
        out
    }
}

struct ExciterIntoIterator {
    det: &'static Config,               // Needed to check if excitation is valid
    epair_iter: Box<dyn Iterator<Item=Orbs>>, // Iterator over electrons or pairs of electrons in det to excite
    epair: Orbs,               // Current exciting electron pair
    sorted_excites: &'static HashMap<Orbs, Vec<StoredExcite>>,
    target_iter: Box<dyn Iterator<Item=StoredExcite>>,
    eps: f64,
    is_alpha: Option<bool>
}

impl Iterator for ExciterIntoIterator {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<StoredExcite>;
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
                        if self.det.is_valid_stored(&exc) {
                            // Found valid excitation; return it
                            match exc.target {
                                Orbs::Double(target) => {
                                    return Some(
                                        Excite::Double {
                                            init: self.epair.clone(),
                                            target,
                                            abs_h: exc.abs_h,
                                            is_alpha: self.is_alpha
                                        }
                                    );
                                },
                                Orbs::Single(target) => {
                                    return Some(
                                        Excite::Single {
                                            init: self.epair,
                                            target,
                                            abs_h: exc.abs_h,
                                            is_alpha: self.is_alpha
                                        }
                                    );
                                }
                            }
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

// pub fn chain_iters(iters: &[dyn Iterator]) {}


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
