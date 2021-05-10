// Wavefunction data structure:
// Includes functions for initializing, printing, and adding new determinants

pub mod det;
mod eps;

use std::collections::HashMap;

use super::utils::read_input::Global;
use super::ham::Ham;
use crate::excite::{Excite, Orbs};
use crate::excite::init::ExciteGenerator;
use det::{Det, Config};
use eps::{Eps, init_eps};
use crate::utils::bits::{bit_pairs, bits};

#[derive(Default)]
pub struct Wf {
    pub n_states: i32, // number of states - same as in input file, but want it attached to the wf
    pub converged: bool, // whether variational wf is converged. Update at end of each HCI iteration
    pub eps_iter: Eps, // iterator that produces the variational epsilon for each HCI iteration
    pub n: usize,                         // number of dets
    pub energy: f64,                    // variational energy
    inds: HashMap<Config, usize>,            // hashtable : det -> u64 for looking up index by det
    pub dets: Vec<Det>,                     // for looking up augmented det by index
}

impl Wf {
    pub fn print(&self) {
        println!("\nWavefunction has {} dets with energy {}", self.n, self.energy);
        println!("Coeff     Det_up     Det_dn    <D|H|D>");
        for d in self.dets.iter() {
            println!("{:.4}   {}   {}   {:.3}", d.coeff, fmt_det(d.config.up), fmt_det(d.config.dn), d.diag);
        }
        println!("\n");
    }

    pub fn push(&mut self, d: Det) {
        // Just adds a new det to the wf
        if !self.inds.contains_key(&d.config) {
            self.inds.insert(d.config, self.n);
            self.n += 1;
            self.dets.push(d);
        }
    }

    pub fn add_det_with_coeff(&mut self, exciting_det: &Det, ham: &Ham, excite: &Excite, new_det: Config, coeff: f64) {
        // Add det with its coefficient
        // If det already exists in wf, add its coefficient to that det
        // Also, computes diagonal element if necessary (that's what exciting_det, ham and excite are needed for)
        let ind = self.inds.get(&new_det);
        match ind {
            Some(k) => self.dets[*k].coeff += coeff,
            None => {
                self.inds.insert(new_det, self.n);
                self.n += 1;
                self.dets.push(
                    Det {
                        config: new_det,
                        coeff: coeff,
                        diag: exciting_det.new_diag(ham, excite)
                    }
                );
            }
        }
    }

    pub fn approx_matmul(&self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> (Wf, Alias) {
        // Approximate matrix-vector multiplication
        // Uses eps as a cutoff for doubles, but uses additional singles (since checking whether
        // they meet the cutoff is as expensive as actually calculating the matrix element)

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // add to output wf
        // TODO: Set up for sampling of remaining
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;

        // Diagonal component
        let mut out: Wf = Wf::default();
        for det in &self.dets {
            out.push(Det{config: det.config, coeff: det.diag * det.coeff, diag: det.diag});
        }

        // Off-diagonal component
        for det in &self.dets {
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(det.config.up) {
                    for j in bits(det.config.dn) {
                        for stored_excite in excite_gen.opp_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(*config) {
                        for stored_excite in excite_gen.same_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out.add_det_with_coeff(det, ham, &excite, d, ham.ham_doub(&det.config, &d) * det.coeff);
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(*config) {
                        for stored_excite in excite_gen.sing_generator.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    // Compute matrix element and add to H*psi
                                    // TODO: Do this in a cache efficient way
                                    out.add_det_with_coeff(det, ham, &excite, d, ham.ham_sing(&det.config, &d) * det.coeff);
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        } // for det in self.dets

        out
    }

    pub fn approx_matmul_variational(&self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64) -> Wf {
        // Approximate matrix-vector multiplication within variational space only
        // Uses eps as a cutoff for doubles, but uses additional singles (since checking whether
        // they meet the cutoff is as expensive as actually calculating the matrix element)
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;
        let mut ind: usize;

        // Diagonal component
        let mut out: Wf = Wf::default();
        for det in &self.dets {
            out.push(Det{config: det.config, coeff: det.diag * det.coeff, diag: det.diag});
        }

        // Off-diagonal component

        // Iterate over all dets; for each, use eps to truncate the excitations; for each excitation,
        // only add if it is already in variational wf
        for det in &self.dets {
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(det.config.up) {
                    for j in bits(det.config.dn) {
                        for stored_excite in excite_gen.opp_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    match self.inds.get(&d) {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        Some(ind) => {
                                            out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
                                        },
                                        _ => {}
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(*config) {
                        for stored_excite in excite_gen.same_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    // Valid excite: add to H*psi
                                    match self.inds.get(&d) {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        Some(ind) => {
                                            out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
                                        },
                                        _ => {}
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(*config) {
                        for stored_excite in excite_gen.sing_generator.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: add to H*psi
                                    match self.inds.get(&d) {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        Some(ind) => {
                                            out.dets[*ind].coeff += ham.ham_doub(&det.config, &d) * det.coeff
                                        },
                                        _ => {}
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        } // for det in self.dets

        out
    }

    pub fn get_new_dets(&mut self, ham: &Ham, excite_gen: &ExciteGenerator) {
        // Get new dets: iterate over all dets; for each, propose all excitations; for each, check if new;
        // if new, add to wf

        let eps: f64 = self.eps_iter.next().unwrap();

        let new_dets: Wf = self.iterate_excites(ham, excite_gen, eps, false);

        // Add all new dets to the wf
        for det in new_dets.dets {
            self.push(det);
        }
    }

    fn iterate_excites(&mut self, ham: &Ham, excite_gen: &ExciteGenerator, eps: f64, matmul: bool) -> Wf {
        // Iterate over excitations using heat-bath cutoff eps
        // Used internally by both approx_matmul and get_new_dets
        // If matmul, then return H*psi; else, return a wf composed of new dets
        println!("Getting new dets with epsilon = {:.1e}", eps);
        let mut local_eps: f64;
        let mut excite: Excite;
        let mut new_det: Option<Config>;
        // We can't just iterate over dets because we are adding new dets to the same dets data structure
        let mut out: Wf = Wf::default();
        for det in &self.dets {
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_doub >= local_eps {
                for i in bits(det.config.up) {
                    for j in bits(det.config.dn) {
                        for stored_excite in excite_gen.opp_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: None
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    // Valid excite: either add to H*psi or add this det to out
                                    if matmul {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        // out.add_det_with_coeff(det, ham, excite, d,
                                        //                        ham.ham_doub(&det.config, &d) * det.coeff);
                                        todo!()
                                    } else {
                                        // If not already in input or output, compute diagonal element and add to output
                                        if !self.inds.contains_key(&d) {
                                            if !out.inds.contains_key(&d) {
                                                match excite.target {
                                                    Orbs::Double(rs) => {
                                                        out.push(
                                                            Det {
                                                                config: d,
                                                                coeff: 0.0,
                                                                diag: det.new_diag_opp(ham, (i, j), rs)
                                                            }
                                                        );
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_doub >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for (i, j) in bit_pairs(*config) {
                        for stored_excite in excite_gen.same_doub_generator.get(&Orbs::Double((i, j))).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Double((i, j)),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if matmul {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        // out.add_det_with_coeff(det, ham, excite, d,
                                        //                       ham.ham_doub(&det.config, &d) * det.coeff);
                                        todo!()
                                    } else {
                                        if !self.inds.contains_key(&d) {
                                            if !out.inds.contains_key(&d) {
                                                match excite.target {
                                                    Orbs::Double(rs) => {
                                                        out.push(
                                                            Det {
                                                                config: d,
                                                                coeff: 0.0,
                                                                diag: det.new_diag_same(ham, (i, j), rs, *is_alpha)
                                                            }
                                                        );
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for (config, is_alpha) in &[(det.config.up, true), (det.config.dn, false)] {
                    for i in bits(*config) {
                        for stored_excite in excite_gen.sing_generator.get(&Orbs::Single(i)).unwrap() {
                            if stored_excite.abs_h < local_eps { break; }
                            excite = Excite {
                                init: Orbs::Single(i),
                                target: stored_excite.target,
                                abs_h: stored_excite.abs_h,
                                is_alpha: Some(*is_alpha)
                            };
                            new_det = det.config.safe_excite_det(&excite);
                            match new_det {
                                Some(d) => {
                                    if matmul {
                                        // Compute matrix element and add to H*psi
                                        // TODO: Do this in a cache efficient way
                                        // out.add_det_with_coeff(det, ham, excite, d,
                                        //                       ham.ham_sing(&det.config, &d) * det.coeff);
                                        todo!()
                                    } else {
                                        if !self.inds.contains_key(&d) {
                                            if !out.inds.contains_key(&d) {
                                                match excite.target {
                                                    Orbs::Single(r) => {
                                                        // Compute whether single excitation actually exceeds eps!
                                                        if ham.ham_sing(&det.config, &d).abs() > local_eps {
                                                            out.push(
                                                                Det {
                                                                    config: d,
                                                                    coeff: 0.0,
                                                                    diag: det.new_diag_sing(ham, i, r, *is_alpha)
                                                                }
                                                            );
                                                        }
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        } // for det in self.dets

        out
    }
}


fn fmt_det(d: u128) -> String {
    let mut s = format!("{:#10b}", d);
    s = str::replace(&s, "0", "_");
    str::replace(&s, "_b", "")
}

// Initialize variational wf to the HF det (only needs to be called once)
pub fn init_var_wf(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator) -> Wf {
    let mut wf: Wf = Wf::default();
    wf.n_states = global.n_states;
    wf.converged = false;
    wf.n = 1;
    let one: u128 = 1;
    let mut hf = Det {
        config: Config {
            up: ((one << global.nup) - 1),
            dn: ((one << global.ndn) - 1),
        },
        coeff: 1.0,
        diag: 0.0,
    };
    let h: f64 = ham.ham_diag(&hf.config);
    hf.diag = h;
    wf.inds = HashMap::new();
    wf.inds.insert(hf.config, 0);
    wf.dets.push(hf);
    wf.energy = wf.dets[0].diag;
    wf.eps_iter = init_eps(&wf, &global, &excite_gen);
    wf
}
