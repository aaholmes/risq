// Iterator over excitations

// Eventually want something like
// for excite in get_excites_exceeding_eps(det, excite_gen, eps) {}
// which should return an iterator of Excites (singles and doubles)
// where singles are candidates whose matrix elements must be computed separately
// (because we want to check if they're new first before computing their matrix element)

// use itertools::Itertools;
use std::iter::empty;
use std::collections::HashMap;

use crate::wf::det::{Config, Det};
use crate::excite::{Excite, OPair, StoredSing, StoredDoub};
use crate::excite::init::ExciteGenerator;
use crate::utils::bits::bits;

pub fn get_excites_exceeding_eps(det: Det, excite_gen: &ExciteGenerator, eps: f64) -> impl Iterator<Item = Excite> {
    // Returns an iterator over all *candidate* excitations from the given determinant that may exceed eps
    // The single excitation matrix elements must still be compared to eps
    // TODO: Chain these together for the 3 different excitation types
    if excite_gen.max_opp_spin_doub >= eps / det.coeff.abs() {
        Exciter::new_opp(det, &excite_gen.opp_spin_doub_generator, eps).into_iter()
    } else {
        empty()
    }
}


// Backend for get_excites_exceeding_eps

enum Exciter<'a> {
    Double(DoubExciter<'a>),
    Single(SingExciter<'a>)
}

struct DoubExciter<'a> {
    det: Det,
    excite_gen: &'a HashMap<OPair, Vec<StoredDoub>>,
    eps: f64
}

struct SingExciter<'a> {
    det: Det,
    excite_gen: &'a Vec<Vec<StoredSing>>,
    eps: f64
}

impl Exciter<'_> {
    fn new_opp(det: Det, excite_gen: &HashMap<OPair, Vec<StoredDoub>>, eps: f64) -> Exciter {
         Exciter::Double(
             DoubExciter {
                 det: det,
                 excite_gen: &excite_gen,
                 eps: eps
             }
         )
    }
}

impl IntoIterator for Exciter<'_> {
    type Item = Excite;
    type IntoIter = ExciterIntoIterator<'_>;

    fn into_iter(self) -> Self::IntoIter {
        let mut out = ExciterIntoIterator {
            det: self.det.config,
            e_pair_iter: iproduct!(bits(self.det.config.up), bits(self.det.config.dn)),
            excite_gen: self.excite_gen,
            eps: self.eps / self.det.coeff.abs(),
            sorted_excites: empty()
        };
        let p = self.e_pair_iter.next();
        out.sorted_excites = self.excite_gen.get_key(&OPair(p[0], p[1])).to_iter();
        out
    }
}

struct ExciterIntoIterator<'a> {
    det: Config,
    e_pair_iter: dyn Iterator<Item=(i32, i32)>,
    excite_gen: &'a HashMap<OPair, Vec<StoredDoub>>,
    eps: f64, // This is eps / |c_i|
    sorted_excites: dyn Iterator<Item=StoredDoub>,
}

impl Iterator for ExciterIntoIterator<'_> {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<Excite>;
        loop {
            excite = self.sorted_excites.next();
            match excite {
                None => {
                    // Go to next electron pair
                    if self.next_e_pair() { return None; }
                }
                Some(exc) => {
                    // Check whether it meets threshold; if not, quit this sorted_excites list
                    if exc.abs_h >= self.eps {
                        // Only return this excitation if it is a valid excite for this det
                        if self.det.is_valid(&exc) {
                            return Some(exc);
                        }
                    } else {
                        // Go to next electron pair
                        if self.next_e_pair() { return None; }
                    }
                }
            }
        }
    }
}

impl ExciterIntoIterator<'_> {
    fn next_e_pair(&mut self) -> bool {
        // Go to next electron pair
        // Returns true if no more electron pairs left
        let next_pair = self.e_pair_iter.next();
        match next_pair {
            None => return true,
            Some(p) => {
                self.sorted_excites = self.excite_gen.get_key(&OPair(p[0], p[1])).to_iter();
                return false;
            }
        }
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::excite::init::init_excite_generator;
    use crate::ham::Ham;
    use crate::ham::read_ints::read_ints;
    use crate::utils::read_input::{Global, read_input};

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

        let eps  = 0.1;
        for excite in get_excites_exceeding_eps(det, &excite_gen, eps) {
            println!("Got here");
        }
    }
}
