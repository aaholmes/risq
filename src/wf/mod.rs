use std::collections::HashMap;

use super::ham::Ham;
use super::utils::read_input::Global;
use crate::excite::{ExciteGenerator, Doub, OPair, Sing};
use crate::utils::bits::bits;
use std::cmp::max;

// Determinant
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Det {
    pub up: u128,
    pub dn: u128,
}

// Wavefunction
#[derive(Default)]
pub struct Wf {
    pub n_states: i32, // number of states - same as in input file, but want it attached to the wf
    pub converged: bool, // whether variational wf is converged. Update at end of each HCI iteration
    pub eps_iter: Eps, // iterator that produces the variational epsilon for each HCI iteration
    pub n: u64,                         // number of dets
    pub energy: f64,                    // variational energy
    inds: HashMap<Det, u64>,            // hashtable : det -> u64 for looking up index by det
    dets: Vec<Det>,                     // for looking up det by index
    coeffs: Vec<f64>,                   // coefficients
    diags: Vec<f64>, // diagonal elements of Hamiltonian (so new diagonal elements can be computed quickly)
}

fn fmt_det(d: u128) -> String {
    let mut s = format!("{:#10b}", d);
    s = str::replace(&s, "0", "_");
    str::replace(&s, "_b", "")
}

impl Wf {
    pub fn print(&self) {
        println!(
            "Wavefunction has {} dets with energy {}",
            self.n, self.energy
        );
        for (d, c) in self.dets.iter().zip(self.coeffs.iter()) {
            println!("{} {}   {}", fmt_det(d.up), fmt_det(d.dn), c);
        }
    }

    pub fn add_det(&mut self, d: Det, diag: f64) {
        if !self.inds.contains_key(&d) {
            self.n += 1;
            self.inds.insert(d, self.n);
            self.dets.push(d);
            self.coeffs.push(0.0);
            self.diags.push(diag);
        }
    }

    pub fn init_eps(&mut self, global: &Global, excite_gen: &ExciteGenerator) {
        // Initialize epsilon iterator
        // max_doub is the largest double excitation magnitude coming from the wavefunction
        // Can't just use excite_gen.max_(same/opp)_spin_doub because we want to only consider
        // excitations coming from current wf
        let mut max_doub: f64 = global.eps;
        let mut this_doub: f64 = 0.0;
        for det in self.dets {
            for i in bits(det.up) {
                for j in bits(det.dn) {
                    for excite in excite_gen.opp_spin_doub_generator.get(&OPair(i, j)) {
                        this_doub: f64 = excite.next().unwrap().abs_h;
                        if this_doub > max_doub {
                            max_doub = this_doub;
                        }
                        break;
                    }
                }
            }
            for i in bits(det.up) {
                for j in bits(det.up) {
                    if i >= j { continue; }
                    for excite in excite_gen.same_spin_doub_generator.get(&OPair(i, j)) {
                        this_doub: f64 = excite.next().unwrap().abs_h;
                        if this_doub > max_doub {
                            max_doub = this_doub;
                        }
                        break;
                    }
                }
            }
            for i in bits(det.dn) {
                for j in bits(det.dn) {
                    if i >= j { continue; }
                    for excite in excite_gen.opp_spin_doub_generator.get(&OPair(i, j)) {
                        this_doub: f64 = excite.next().unwrap().abs_h;
                        if this_doub > max_doub {
                            max_doub = this_doub;
                        }
                        break;
                    }
                }
            }
        }
        self.eps_iter = Eps {
            next: max_doub - 1e-9,
            target: global.eps,
        };
    }

    pub fn get_new_dets(&mut self, excite_gen: &ExciteGenerator) {
        // Get new dets: iterate over all dets; for each, propose all excitations; for each, check if new;
        // if new, add to wf
        let eps: f64 = self.eps_iter.next().unwrap();
        let mut local_eps: f64;
        let mut new_det: Option<Det>;
        for (det, coeff) in self.dets.zip(&self.coeffs) {
            local_eps = eps / coeff.abs();
            // Double excitations
            // Opposite spin
            if excite_gen.max_opp_spin_doub >= local_eps {
                for i in bits(det.up) {
                    for j in bits(det.dn) {
                        for excite in excite_gen.opp_spin_doub_generator.get(&OPair(i, j)) {
                            if excite.abs_h < local_eps { break; }
                            new_det = det.excite_det_opp_doub(excite);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        self.add_det(d, det.new_diag_opp(excite))
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
                for i in bits(det.up) {
                    for j in bits(det.up) {
                        if i >= j { continue; }
                        for excite in excite_gen.same_spin_doub_generator.get(&OPair(i, j)) {
                            if excite.abs_h < local_eps { break; }
                            new_det = det.excite_det_same_doub(excite, true);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        self.add_det(d, det.new_diag_same(excite, true))
                                    }
                                }
                                None => break
                            }
                        }
                    }
                }
                // TODO: Make this a new iterator over pairs of set bits!
                for i in bits(det.dn) {
                    for j in bits(det.dn) {
                        if i >= j { continue; }
                        for excite in excite_gen.same_spin_doub_generator.get(&OPair(i, j)) {
                            if excite.abs_h < local_eps { break; }
                            new_det = det.excite_det_same_doub(excite, false);
                            match new_det {
                                Some(d) => {
                                    if !self.inds.contains_key(&d) {
                                        self.add_det(d, det.new_diag_same(excite, false))
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
                for i in bits(det.up) {
                    for excite in excite_gen.sing_generator[i] {
                        if excite.max_abs_h < local_eps { break; }
                        // TODO: compute single excitation mat elem here, check if it exceeds eps
                        new_det = det.excite_sing(excite, true);
                        match new_det {
                            Some(d) => {
                                if !self.inds.contains_key(&d) {
                                    self.add_det(d, det.new_diag_sing(excite, true))
                                }
                            }
                            None => break
                        }
                    }
                }
                for i in bits(det.dn) {
                    for excite in excite_gen.sing_generator[i] {
                        if excite.max_abs_h < local_eps { break; }
                        // TODO: compute single excitation mat elem here, check if it exceeds eps
                        new_det = det.excite_sing(excite, false);
                        match new_det {
                            Some(d) => {
                                if !self.inds.contains_key(&d) {
                                    self.add_det(d, det.new_diag_sing(excite, false))
                                }
                            }
                            None => break
                        }
                    }
                }
            }

        } // for (det, coeff) in self.dets.zip(&self.coeffs)
    }
}

impl Det {
    pub fn new_diag_opp(&self, &excite: Doub) {
        // Compute new diagonal element given the old one
    }

    pub fn new_diag_same(&self, &excite: Doub) {
        // Compute new diagonal element given the old one
    }

    pub fn new_diag_sing(&self, &excite: Sing) {
        // Compute new diagonal element given the old one
    }
}

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
        Some(curr)
    }
}

impl Default for Eps {
    fn default() -> Self {
        Eps{ next: 0.0, target: 0.0}
    }
}

// Init wf to the HF det (only needs to be called once)
pub fn init_wf(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator) -> Wf {
    let mut wf: Wf = Wf::default();
    wf.n_states = global.n_states;
    wf.converged = false;
    wf.n = 1;
    let one: u128 = 1;
    let hf = Det {
        up: ((one << global.nup) - 1),
        dn: ((one << global.ndn) - 1),
    };
    let h: f64 = ham.ham_diag(&hf);
    wf.inds = HashMap::new();
    wf.inds.insert(hf, 0);
    wf.dets.push(hf);
    wf.coeffs.push(1.0);
    wf.diags.push(h);
    wf.energy = wf.diags[0];
    wf.init_eps(global, excite_gen);
    wf
}
