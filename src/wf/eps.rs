//! Variational epsilon iterator (to attach to variational wf)
//! Epsilon starts at the largest value that allows at least one double excitation from the initial
//! wf, then drops by a factor of 2 every iteration until it reaches the target value set in the
//! input file

use crate::excite::init::ExciteGenerator;
use crate::excite::Orbs;
use crate::utils::bits::{bit_pairs, bits, btest};
use crate::utils::read_input::Global;
use crate::wf::Wf;

/// Variational epsilon iterator
#[derive(Clone, Copy)]
pub struct Eps {
    next_one: f64,
    target: f64,
}

impl Iterator for Eps {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let curr: f64 = self.next_one;
        let new_next_one: f64 = self.next_one * 0.1;
        self.next_one = if new_next_one > self.target {
            new_next_one
        } else {
            self.target
        };
        Some(curr)
    }
}

impl Default for Eps {
    fn default() -> Eps {
        Eps {
            next_one: 0.0,
            target: 0.0,
        }
    }
}

/// Initialize epsilon iterator
/// max_doub is the min of the largest symmetrical and largest asymmetrical double excitation magnitudes coming from the wavefunction
/// Can't just use excite_gen.max_(same/opp)_spin_doub because we want to only consider
/// excitations coming from initial wf (usually HF det)
/// We use this initial eps so that when we do excited states, there will be at least two closed
/// shell and at least two open shell determinants
pub fn init_eps(wf: &Wf, global: &Global, excite_gen: &ExciteGenerator) -> Eps {
    let mut max_sym: f64 = global.eps_var;
    let mut max_asym: f64 = global.eps_var;
    let mut max_asym_connectable_to_spin_flipped: f64 = global.eps_var; // Asymmetrical excitation that is connected to its spin-flipped counterpart, i.e., up: i->j, dn: i->k, or up: i->k, dn: j->k
    let mut this_doub: f64;
    for det in &wf.dets {
        // Opposite spin
        for i in bits(excite_gen.valence & det.config.up) {
            for j in bits(excite_gen.valence & det.config.dn) {
                let mut found_sym = false;
                let mut found_asym = false;
                for excite in excite_gen
                    .opp_doub_sorted_list
                    .get(&Orbs::Double((i, j)))
                    .unwrap()
                {
                    if let Orbs::Double(t) = excite.target {
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
                                if i == j || t.0 == t.1 {
                                    // Asymmetric and connected to spin-flipped counterpart
                                    if this_doub > max_asym_connectable_to_spin_flipped {
                                        max_asym_connectable_to_spin_flipped = this_doub;
                                    }
                                }
                            }
                            if found_sym && found_asym {
                                break;
                            };
                        }
                    }
                }
            }
        }
        // Same spin
        for config in &[det.config.up, det.config.dn] {
            for (i, j) in bit_pairs(excite_gen.valence & *config) {
                for excite in excite_gen
                    .same_doub_sorted_list
                    .get(&Orbs::Double((i, j)))
                    .unwrap()
                {
                    if let Orbs::Double(t) = excite.target {
                        if !btest(*config, t.0) && !btest(*config, t.1) {
                            this_doub = excite.abs_h;
                            if this_doub > max_asym {
                                max_asym = this_doub;
                            }
                            break;
                        }
                    }
                }
            }
        }
    } // det

    println!("\nFrom HF det: Largest magnitude symmetric double excite magnitude: {:.4}", max_sym);
    println!("             Largest magnitude asymmetric double excite magnitude: {:.4}", max_asym);
    println!("             Largest magnitude asymmetric (connected to spin-flipped counterpart) double excite magnitude: {:.4}", max_asym_connectable_to_spin_flipped);

    let max_doub = {
        if global.z_sym == 1 {
            // If targeting a symmetric state, just need at least one double excite of any kind
            if max_sym > max_asym {
                max_sym
            } else {
                max_asym
            }
        } else {
            // If targeting an anti-symmetric state, need at least one asymmetric pair that are connected to each other
            max_asym_connectable_to_spin_flipped
        }
    };
    println!("Setting initial eps_var = {:.4}", max_doub);
    Eps {
        next_one: &max_doub - 1e-9, // Slightly less than max_doub in case there are two or more elements that are off by machine precision
        target: global.eps_var, //{if global.eps_var < &max_doub - 1e-9 {global.eps_var} else {&max_doub - 1e-9}},
    }
}
