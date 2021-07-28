// Variational stage

mod davidson;
mod ham_gen;
mod sparse;
mod utils;

use super::ham::Ham;
use super::wf::Wf;
use crate::excite::init::ExciteGenerator;
use crate::var::davidson::{dense_optimize, sparse_optimize};
use crate::utils::read_input::Global;
use std::time::Instant;


pub fn variational(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator, wf: &mut Wf) {

    let mut iter: i32 = 0;

    println!("Start of variational stage: Wavefunction has {} det with energy {:.4}", wf.n, wf.energy);

    let eps_energy_converged: f64 = 2.5e-4;
    let mut last_energy: Option<f64> = None;

    while !wf.converged {

        iter += 1;

        let start_find_new_dets: Instant = Instant::now();
        if wf.get_new_dets(&ham, &excite_gen) {
            println!("No new dets added; wf converged");
            wf.converged = true;
            break;
        }
        println!("Time to find new dets: {:?}", start_find_new_dets.elapsed());

        last_energy = Some(wf.energy);

        let coeff_eps: f64 = 1e-3; // Davidson convergence epsilon for coefficients
        let energy_eps: f64 = 1e-6; // Davidson convergence epsilon for energy

        let start_optimize_coeffs: Instant = Instant::now();
        sparse_optimize(&global, wf, coeff_eps, energy_eps, &ham);
        // dense_optimize(wf, coeff_eps, energy_eps, &ham, &excite_gen);
        println!("Time to optimize wf coefficients: {:?}", start_optimize_coeffs.elapsed());

        println!("End of iteration {} (eps = {:.1e}): Wavefunction has {} determinants with energy {:.6}", iter, wf.eps, wf.n, wf.energy);
        if wf.n <= 20 {
            wf.print();
        }

        match last_energy {
            None => {},
            Some(e) => {
                if (e - wf.energy).abs() < eps_energy_converged {
                    println!("Variational energy did not change much; wf converged");
                    wf.converged = true;
                    break;
                }
            }
        }
    }

}
