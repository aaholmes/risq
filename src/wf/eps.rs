// Variational epsilon iterator (to attach to wf):
// Epsilon starts at the largest value that allows at least one double excitation from the initial
// wf, then drops by a factor of 2 every iteration until it reaches the target value set in the
// input file

use crate::wf::Wf;
use crate::utils::bits::{bits, btest, bit_pairs};
use crate::utils::read_input::Global;
use crate::excite::Orbs;
use crate::excite::init::ExciteGenerator;


#[derive(Clone, Copy)]
pub struct Eps {
    next: f64,
    target: f64,
}

impl Iterator for Eps {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let curr: f64 = self.next;
        self.next = if self.next / 2.0 > self.target { self.next / 2.0 } else { self.target };
        // self.next = if self.next * 0.9 > self.target { self.next * 0.9 } else { self.target };
        Some(curr)
    }
}

impl Default for Eps {
    fn default() -> Eps {
        Eps{next: 0.0, target: 0.0}
    }
}

pub fn init_eps(wf: &Wf, global: &Global, excite_gen: &ExciteGenerator) -> Eps {
    // Initialize epsilon iterator
    // max_doub is the min of the largest symmetrical and largest asymmetrical double excitation magnitudes coming from the wavefunction
    // Can't just use excite_gen.max_(same/opp)_spin_doub because we want to only consider
    // excitations coming from initial wf (usually HF det)
    // We use this initial eps so that when we do excited states, there will be at least two closed
    // shell and at least two open shell determinants

    let mut max_sym: f64 = global.eps_var;
    let mut max_asym: f64 = global.eps_var;
    let mut this_doub: f64;
    for det in &wf.dets {
        // Opposite spin
        for i in bits(excite_gen.valence & det.config.up) {
            for j in bits(excite_gen.valence & det.config.dn) {
                let mut found_sym = false;
                let mut found_asym = false;
                for excite in excite_gen.opp_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                    match excite.target {
                        Orbs::Double(t) => {
                            if !btest(det.config.up, t.0) && !btest(det.config.dn, t.1) {
                                this_doub = excite.abs_h;
                                if i == j && t.0 == t.1 {
                                    // Symmetric
                                    found_sym = true;
                                    if this_doub > max_sym {
                                        max_sym = this_doub;
                                    }
                                } else {
                                    // Asymmetric
                                    found_asym = true;
                                    if this_doub > max_asym {
                                        max_asym = this_doub;
                                    }
                                }
                                if found_sym && found_asym { break; };
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
        // Same spin
        for config in &[det.config.up, det.config.dn] {
            for (i, j) in bit_pairs(excite_gen.valence & *config) {
                for excite in excite_gen.same_doub_sorted_list.get(&Orbs::Double((i, j))).unwrap() {
                    match excite.target {
                        Orbs::Double(t) => {
                            if !btest(*config, t.0) && !btest(*config, t.1) {
                                this_doub = excite.abs_h;
                                if this_doub > max_asym {
                                    max_asym = this_doub;
                                }
                                break;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }
    } // det

    let max_doub = { if max_sym < max_asym { max_sym } else { max_asym } };
    Eps {
        next: max_doub - 1e-9, // Slightly less than max_doub in case there are two or more elements that are off by machine precision
        target: global.eps_var,
    }
}
