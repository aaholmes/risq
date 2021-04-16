use std::collections::HashMap;

use super::ham::Ham;
use super::utils::read_input::Global;
use crate::excite::{ExciteGenerator, Doub, OPair};
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
    pub n: u64,                         // number of dets
    pub energy: f64,                    // variational energy
    inds: HashMap<Det, u64>,            // hashtable : det -> u64 for looking up index by det
    dets: Vec<Det>,                     // for looking up det by index
    coeffs: Vec<f64>,                   // coefficients
    diags: Vec<f64>, // diagonal elements of Hamiltonian (so new diagonal elements can be computed quickly)
    pub eps_iter: Eps, // iterator that produces the variational epsilon for each HCI iteration
    //pub eps_iter: &'a dyn Iterator<Item=f64>, // iterator that produces the variational epsilon for each HCI iteration
}

impl Det {
    fn print(&self) {
        println!("{} {}", format!("{:b}", self.up), format!("{:b}", self.dn));
    }
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
        // can't just use excite_gen.max_doub because we want to only consider
        // excitations coming from current wf
        let mut max_doub: f64 = 1.0;
        // for det in self.dets {
        //     for mut excite in excite_gen.iter(det) {
        //         let this_doub: f64 = excite.next().unwrap().abs_h;
        //         if this_doub > max_doub {
        //             max_doub = this_doub;
        //         }
        //     }
        // }
        self.eps_iter = Eps {
            next: max_doub - 1e-8,
            target: global.eps,
        };
    }

    // pub fn get_new_dets(&mut self, excite_gen: &ExciteGenerator) {
    //     let eps: f64 = self.eps_iter.next().unwrap();
    //     let local_eps: f64;
    //     for (det, coeff) in self.dets.zip(self.coeffs) {
    //         local_eps = eps / coeff.abs();
    //         // Double excitations
    //         for excite_list in excite_gen.iter(det) {
    //             for excite in excite_list {
    //                 println!("New excite to orbitals ({}, {}) with |H| = {}", excite.target.0, excite.target.1, excite.abs_h);
    //                 if excite.abs_h < local_eps { continue; }
    //                 // Apply this excitation to det
    //                 // let excited_det: Det = excite_det(&det, &epair, &excite);
    //                 // // See if resulting det is in wf
    //                 // // If not, compute its diagonal element and add it to the wf
    //                 // if !self.inds.contains(excited_det) {
    //                 //     // Compute its diagonal element
    //                 //     let diag: f64 = new_diag();
    //                 //     // Add it to the wf
    //                 //     self.add_det(excited_det, diag);
    //                 // }
    //             }
    //         }
    //         // TODO: Single excitations
    //     }
    // }
}

pub fn excite_det(det: &Det, epair: &OPair, excite: &Doub) -> Det {
    // Create new det given by the excitation applied to the old det
    todo!();
}

pub fn new_diag() {
    // Compute new diagonal element given the old one
    todo!();
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
