// Iterator over excitations

// Eventually want something like
// for excite in get_excites_exceeding_eps(det, excite_gen, eps) {}
// which should return an iterator of Excites (singles and doubles)
// where singles are candidates whose matrix elements must be computed separately
// (because we want to check if they're new first before computing their matrix element)

use core::option::Option::{None, Some};
use core::option::Option;

use crate::excite::{Excite, OPair};
use crate::wf::det::{Config, Det};
use crate::excite::init::ExciteGenerator;

pub fn get_excites_exceeding_eps(det: Det, excite_gen: &ExciteGenerator, eps: f64) -> impl Iterator<Item = Excite> {
    // Returns an iterator over all *candidate* excitations from the given determinant that may exceed eps
    // The single excitation matrix elements must still be compared to eps
    OppDoubExciter::new(det, &excite_gen, eps).into_iter()
}


// Backend for get_excites_exceeding_eps

struct OppDoubExciter {
    det: Det,
    excite_gen: &ExciteGenerator,
    eps: f64
}

impl OppDoubExciter {
    fn new(det: Det, excite_gen: &ExciteGenerator, eps: f64) -> OppDoubExciter {
         OppDoubExciter {
            det: det,
            excite_gen: &excite_gen,
            eps: eps
        }
    }
}

impl IntoIterator for OppDoubExciter {
    type Item = Excite;
    type IntoIter = ExciterIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let out = ExciterIntoIterator {
            det: self.config,
            e_pair_iter: iproduct!(bits(self.config.up), bits(self.config.dn)),
            excite_gen: &excite_gen,
            eps: eps / self.coeff.abs(),
        };
        p = up_iter.next();
        out.sorted_excites = self.excite_gen.get_key(&OPair(p[0], p[1]));
        out
    }
}

struct ExciterIntoIterator {
    det: Config,
    e_pair_iter: BitPairsIterator,
    excite_gen: &ExciteGenerator,
    eps: f64, // This is eps / |c_i|
}

impl Iterator for ExciterIntoIterator {
    type Item = Excite;

    fn next(&mut self) -> Option<Excite> {
        let excite: Option<Excite> = self.sorted_excites.next();
        loop {
            match excite {
                None => {
                    // Go to next electron pair
                    if self.next_e_pair() { return None; }
                }
                Some(exc) => {
                    // Check whether it meets threshold; if not, quit this sorted_excites list
                    if exc.abs_h >= self.eps {
                        // Only return this excitation if it is a valid excite for this det
                        if (self.det.is_valid_opp_excite(&exc)) {
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

impl ExciterIntoIterator {
    fn next_e_pair(&mut self) -> bool {
        // Go to next electron pair
        // Returns true if no more electron pairs left
        next_pair = self.up_iter.next();
        match next_pair {
            None => return true,
            Some(p) => {
                self.sorted_excites = self.excite_gen.get_key(OPair(p[0], p[1]));
                return false;
            }
        }
    }
}
