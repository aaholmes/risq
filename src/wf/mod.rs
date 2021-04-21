// Wavefunction data structure:
// Includes functions for initializing, printing, and adding new determinants

pub mod det;
mod eps;

use std::collections::HashMap;
use std::cmp::max;

use super::utils::read_input::Global;
use super::ham::Ham;
use crate::excite::{ExciteGenerator, Doub, OPair, Sing};
use crate::utils::bits::{bits, btest, ibset, ibclr};
use det::{AugDet, Det};
use eps::{Eps, init_eps};

#[derive(Default)]
pub struct Wf {
    pub n_states: i32, // number of states - same as in input file, but want it attached to the wf
    pub converged: bool, // whether variational wf is converged. Update at end of each HCI iteration
    pub eps_iter: Eps, // iterator that produces the variational epsilon for each HCI iteration
    pub n: usize,                         // number of dets
    pub energy: f64,                    // variational energy
    inds: HashMap<Det, usize>,            // hashtable : det -> u64 for looking up index by det
    dets: Vec<AugDet>,                     // for looking up augmented det by index
}

impl Wf {
    pub fn print(&self) {
        println!(
            "Wavefunction has {} dets with energy {}",
            self.n, self.energy
        );
        for d in self.dets.iter() {
            println!("{} {}   {}", fmt_det(d.det.up), fmt_det(d.det.dn), d.coeff);
        }
    }

    pub fn add_det(&mut self, d: AugDet) {
        if !self.inds.contains_key(&d.det) {
            self.inds.insert(d.det, self.n);
            self.n += 1;
            self.dets.push(d);
        }
    }

    pub fn get_new_dets(&mut self, ham: &Ham, excite_gen: &ExciteGenerator) {
        // Get new dets: iterate over all dets; for each, propose all excitations; for each, check if new;
        // if new, add to wf
        let eps: f64 = self.eps_iter.next().unwrap();
        println!("Getting new dets with epsilon = {}", eps);
        let mut local_eps: f64;
        let mut new_det: Option<Det>;
        // We can't just iterate over dets because we are adding new dets to the same dets data structure
        let mut new_dets: Vec<AugDet> = vec![];
        for det in &self.dets {
            local_eps = eps / det.coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_spin_doub >= local_eps {
                for i in bits(det.det.up) {
                    for j in bits(det.det.dn) {
                        for excite in excite_gen.opp_spin_doub_generator.get(&OPair(i, j)).unwrap() {
                            if excite.abs_h < local_eps { break; }
                            new_det = det.det.excite_det_opp_doub(excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        new_dets.push(
                                            AugDet {
                                                det: d,
                                                coeff: 0.0,
                                                diag: det.new_diag_opp(ham, det.diag, excite)
                                            }
                                        );
                                    }
                                }
                                None => break
                            }
                        }
                    }
                }
            }

            // Same spin
            if excite_gen.max_same_spin_doub >= local_eps {
                // TODO: Make this a new iterator over pairs of set bits!
                for i in bits(det.det.up) {
                    for j in bits(det.det.up) {
                        if i >= j { continue; }
                        for excite in excite_gen.same_spin_doub_generator.get(&OPair(i, j)).unwrap() {
                            if excite.abs_h < local_eps { break; }
                            new_det = det.det.excite_det_same_doub(excite, true);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        new_dets.push(
                                            AugDet {
                                                det: d,
                                                coeff: 0.0,
                                                diag: det.new_diag_same(ham, det.diag, excite, true)
                                            }
                                        );
                                    }
                                }
                                None => break
                            }
                        }
                    }
                }
                // TODO: Make this a new iterator over pairs of set bits!
                for i in bits(det.det.dn) {
                    for j in bits(det.det.dn) {
                        if i >= j { continue; }
                        for excite in excite_gen.same_spin_doub_generator.get(&OPair(i, j)).unwrap() {
                            if excite.abs_h < local_eps { break; }
                            new_det = det.det.excite_det_same_doub(excite, false);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        new_dets.push(
                                            AugDet {
                                                det: d,
                                                coeff: 0.0,
                                                diag: det.new_diag_same(ham, det.diag, excite, false)
                                            }
                                        );
                                    }
                                }
                                None => break
                            }
                        }
                    }
                }
            }

            // Single excitations
            if excite_gen.max_sing >= local_eps {
                for i in bits(det.det.up) {
                    for excite in &excite_gen.sing_generator[i as usize] {
                        if excite.max_abs_h < local_eps { break; }
                        // TODO: compute single excitation mat elem here, check if it exceeds eps
                        new_det = det.det.excite_det_sing(&excite, true);
                        match new_det {
                            Some(d) => {
                                if !self.inds.contains_key(&d) {
                                    new_dets.push(
                                        AugDet {
                                            det: d,
                                            coeff: 0.0,
                                            diag: det.new_diag_sing(ham, det.diag, &excite, true)
                                        }
                                    );
                                }
                            }
                            None => break
                        }
                    }
                }
                for i in bits(det.det.dn) {
                    for excite in &excite_gen.sing_generator[i as usize] {
                        if excite.max_abs_h < local_eps { break; }
                        // TODO: compute single excitation mat elem here, check if it exceeds eps
                        new_det = det.det.excite_det_sing(&excite, false);
                        match new_det {
                            Some(d) => {
                                if !self.inds.contains_key(&d) {
                                    new_dets.push(
                                        AugDet {
                                            det: d,
                                            coeff: 0.0,
                                            diag: det.new_diag_sing(ham, det.diag, &excite, false)
                                        }
                                    );
                                }
                            }
                            None => break
                        }
                    }
                }
            }

        } // for det in self.dets

        // Finally, add all new dets to the wf
        for det in new_dets {
            self.add_det(det);
        }
    }
}

fn fmt_det(d: u128) -> String {
    let mut s = format!("{:#10b}", d);
    s = str::replace(&s, "0", "_");
    str::replace(&s, "_b", "")
}

// Init wf to the HF det (only needs to be called once)
pub fn init_wf(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator) -> Wf {
    let mut wf: Wf = Wf::default();
    wf.n_states = global.n_states;
    wf.converged = false;
    wf.n = 1;
    let one: u128 = 1;
    let mut hf = AugDet {
        det: Det {
            up: ((one << global.nup) - 1),
            dn: ((one << global.ndn) - 1),
        },
        coeff: 1.0,
        diag: 0.0,
    };
    let h: f64 = ham.ham_diag(&hf.det);
    hf.diag = h;
    wf.inds = HashMap::new();
    wf.inds.insert(hf.det, 0);
    wf.dets.push(hf);
    wf.energy = wf.dets[0].diag;
    wf.eps_iter = init_eps(&wf, &global, &excite_gen);
    wf
}
