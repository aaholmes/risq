use std::collections::HashMap;

use super::ham::Ham;
use super::utils::read_input::Global;
use crate::excite::ExciteGenerator;
use crate::utils::bits::bits;
use std::cmp::max;

// Determinant
#[derive(Copy)]
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
    pub eps_iter: Iterator<Item = f64>, // iterator that produces the variational epsilon for each HCI iteration
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

    pub fn add_det(&mut self, d: Det) {
        if !self.inds.contains_key(&d) {
            self.n += 1;
            self.inds.insert(d, self.n);
            self.dets.push(d);
            self.coeffs.push(0.0);
            //TODO: implement diag elem delta
            self.diags.push(1.0);
        }
    }

    pub fn init_eps(&mut self, excite_gen: &ExciteGenerator) {
        // Initialize epsilon iterator
        let mut max_doub: f64 = 0.0;
        for det in self.dets {
            for mut excite in excite_gen.iter(det) {
                let this_doub: f64 = excite.next().unwrap().abs_h;
                if this_doub > max_doub {
                    max_doub = this_doub;
                }
            }
        }
        // max_doub is now the largest double excitation magnitude coming from the wavefunction
        self.eps_iter = Eps {
            next: max_doub,
            target: global.eps,
        };
    }
}

struct Eps {
    next: f64,
    target: f64,
}

impl Iterator for Eps {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let curr: f64 = self.next;
        self.next = max(self.next / 2.0, self.target);
        Some(curr)
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
    wf.init_eps(excite_gen);
    wf
}
