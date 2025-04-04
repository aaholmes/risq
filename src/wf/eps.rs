//! # Variational Epsilon Iterator (`wf::eps`)
//!
//! This module provides an iterator (`Eps`) that controls the variational screening
//! threshold (epsilon_1 or `eps_var`) used during the Heat-bath Configuration Interaction (HCI)
//! iterations.
//!
//! The typical strategy is to start with a relatively large epsilon and gradually decrease
//! it over iterations until a target value (specified in the input) is reached. This allows
//! the wavefunction to grow progressively while managing computational cost.

use crate::excite::init::ExciteGenerator;
use crate::excite::Orbs;
use crate::utils::bits::{bit_pairs, bits, btest};
use crate::utils::read_input::Global;
use crate::wf::Wf;

/// An iterator that yields the variational screening threshold (`epsilon_1`) for each HCI iteration.
///
/// Stores the next epsilon value to be yielded (`next_one`) and the final target
/// epsilon (`target`) specified by the user.
#[derive(Clone, Copy, Debug)]
pub struct Eps {
    /// The epsilon value that will be returned by the next call to `next()`.
    next_one: f64,
    /// The final target value for epsilon, read from the input file (`global.eps_var`).
    target: f64,
}

impl Iterator for Eps {
    type Item = f64;

    /// Returns the current epsilon value and updates the next value for the subsequent iteration.
    ///
    /// The update strategy implemented here decreases epsilon by a factor of 10 (`* 0.1`)
    /// in each step, until the `target` value is reached or passed. Once the target is
    /// reached, subsequent calls will keep returning the `target` value.
    ///
    /// Note: The original comment mentioned a factor of 2 decrease, but the code uses 0.1.
    /// This implementation always returns `Some(value)`, effectively creating an infinite
    /// iterator once the target is reached.
    fn next(&mut self) -> Option<f64> {
        let curr = self.next_one;
        // Calculate the proposed next epsilon (decrease by factor of 10)
        let proposed_next = self.next_one * 0.1;
        // Set the actual next epsilon: either the proposed value or the target, whichever is larger.
        self.next_one = if proposed_next > self.target {
            proposed_next
        } else {
            self.target
        };
        // Return the current epsilon value for this iteration.
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

/// Initializes the `Eps` iterator.
///
/// Determines the starting epsilon value based on the largest magnitude double excitations
/// originating from the initial reference wavefunction (`wf`, typically Hartree-Fock).
/// The goal is to set an initial `eps_var` that is just large enough to include at least
/// one significant double excitation, ensuring the variational space grows meaningfully
/// in the first iteration. Special consideration is given based on whether a symmetric
/// (`global.z_sym == 1`) or anti-symmetric state is targeted, influencing which type
/// of double excitation magnitude determines the starting point.
///
/// # Arguments
/// * `wf`: The initial reference wavefunction (often just the HF determinant).
/// * `global`: Global calculation parameters, including the target `eps_var`.
/// * `excite_gen`: The pre-computed excitation generator data.
///
/// # Returns
/// An `Eps` iterator initialized with the calculated starting epsilon and the target epsilon.
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
