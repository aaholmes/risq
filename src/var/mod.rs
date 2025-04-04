//! Variational stage of Heat-bath Configuration Interaction

mod davidson;
pub mod eigenvalues;
mod ham_gen;
pub mod off_diag;
pub(crate) mod sparse;
mod utils;

use super::ham::Ham;
use crate::excite::init::ExciteGenerator;
use crate::utils::read_input::Global;
use crate::var::davidson::sparse_optimize;
use crate::wf::VarWf;
use std::time::Instant;
use crate::excite::iterator::dets_excites_and_excited_dets;

/// Perform variational selected CI
pub fn variational(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator, var_wf: &mut VarWf) {
    let mut iter: i32 = 0;

    println!(
        "Start of variational stage: Wavefunction has {} det with energy {:.14}",
        var_wf.wf.n, var_wf.wf.energy
    );

    // let eps_energy_converged: f64 = 2.5e-4;
    let mut last_energy: Option<f64>;

    while !var_wf.converged {
        iter += 1;

        if iter > 20 {
            println!("Too many iterations! Stopping");
            break;
        }

        let start_find_new_dets: Instant = Instant::now();
        if (var_wf.eps == global.eps_var) & var_wf.find_new_dets(global, ham, excite_gen) {
            println!("No new dets added; wf converged");
            var_wf.converged = true;
            break;
        }
        println!("Time to find new dets: {:?}", start_find_new_dets.elapsed());

        last_energy = Some(var_wf.wf.energy);

        let coeff_eps: f64 = 1e-4; // Davidson convergence epsilon for coefficients
        let energy_eps: f64 = 1e-8; // Davidson convergence epsilon for energy

        println!("\nOptimizing coefficients of wf with {} dets", var_wf.wf.n);
        let start_optimize_coeffs: Instant = Instant::now();
        sparse_optimize(
            global,
            ham,
            excite_gen,
            var_wf,
            coeff_eps,
            energy_eps,
            iter > 1,
        );
        // dense_optimize(wf, coeff_eps, energy_eps, &ham, &excite_gen);
        println!(
            "Time to optimize wf coefficients: {:?}",
            start_optimize_coeffs.elapsed()
        );

        println!("End of iteration {} (eps = {:.1e}): Wavefunction has {} determinants with energy {:.14}", iter, var_wf.eps, var_wf.wf.n, var_wf.wf.energy);
        if var_wf.wf.n <= 10 {
            var_wf.print();
        } else {
            var_wf.print_largest(10);
        }

        // if iter == 2 { panic!("Debug!") }

        if var_wf.eps == global.eps_var {
            match last_energy {
                None => {}
                Some(_) => {
                    // if (e - wf.energy).abs() < eps_energy_converged {
                    //     println!("Variational energy did not change much; wf converged");
                    //     wf.converged = true;
                    //     break;
                    // }
                }
            }
        }
    }
}
