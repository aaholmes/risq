// Variational stage

mod davidson;
mod ham_gen;
mod sparse;

use super::ham::Ham;
use super::wf::Wf;
use crate::excite::init::ExciteGenerator;
use crate::var::davidson::dense_optimize;
// use crate::var::davidson::optimize;


pub fn variational(ham: &Ham, excite_gen: &ExciteGenerator, wf: &mut Wf) {

    let mut iter: i32 = 0;

    println!("Start of variational stage: Wavefunction has {} det with energy {:.4}", wf.n, wf.energy);

    while !wf.converged {

        iter += 1;

        if wf.get_new_dets(&ham, &excite_gen) {
            println!("No new dets added; wf converged");
            wf.converged = true;
            break;
        }

        if wf.n <= 20 {
            println!("Wf after add_new_dets:");
            wf.print();
        }

        let coeff_eps: f64 = 1e-3; // Davidson convergence epsilon for coefficients
        let energy_eps: f64 = 1e-6; // Davidson convergence epsilon for energy

        dense_optimize(wf, coeff_eps, energy_eps, &ham, &excite_gen);

        println!("End of iteration {} (eps = {:.1e}): Wavefunction has {} determinants with energy {:.4}", iter, wf.eps, wf.n, wf.energy);

        if wf.n <= 20 {
            wf.print();
        }

        // if wf.n > 1000 {
        //     break;
        // }
    }

}
