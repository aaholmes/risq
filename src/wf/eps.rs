// Variational epsilon iterator (to attach to wf):
// Epsilon starts at the largest value that allows at least one double excitation from the initial
// wf, then drops by a factor of 2 every iteration until it reaches the target value set in the
// input file

use crate::wf::Wf;
use crate::utils::bits::{bits, btest, bit_pairs};
use crate::utils::read_input::Global;
use crate::excite::{Orbs, StoredExcite};
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
        self.next = if self.next * 0.9 > self.target { self.next * 0.9 } else { self.target };
        // self.next = if self.next / 2.0 > self.target { self.next / 2.0 } else { self.target };
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
    // max_doub is the largest double excitation magnitude coming from the wavefunction
    // Can't just use excite_gen.max_(same/opp)_spin_doub because we want to only consider
    // excitations coming from initial wf (usually HF det)
    let mut excite: &StoredExcite;
    let mut max_doub: f64 = global.eps;
    let mut this_doub: f64;
    for det in &wf.dets {
        // Opposite spin
        for i in bits(det.config.up) {
            for j in bits(det.config.dn) {
                excite = &excite_gen.opp_doub_generator.get(&Orbs::Double((i, j))).unwrap()[0];
                match excite.target {
                    Orbs::Double(t) => {
                        if !btest(det.config.up, t.0) && !btest(det.config.dn, t.1) {
                            this_doub = excite.abs_h;
                            if this_doub > max_doub {
                                max_doub = this_doub;
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        // Same spin
        for config in &[det.config.up, det.config.dn] {
            for (i, j) in bit_pairs(*config) {
                excite = &excite_gen.same_doub_generator.get(&Orbs::Double((i, j))).unwrap()[0];
                match excite.target {
                    Orbs::Double(t) => {
                        if !btest(*config, t.0) && !btest(*config, t.1) {
                            this_doub = excite.abs_h;
                            if this_doub > max_doub {
                                max_doub = this_doub;
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
    } // det

    println!("Setting initial eps = {:.4}", max_doub);
    Eps {
        next: max_doub - 1e-9, // Slightly less than max_doub in case there are two or more elements that are off by machine precision
        target: global.eps,
    }
}
