// Variational stage

mod davidson;

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

        wf.get_new_dets(&ham, &excite_gen);

        println!("Wf after add_new_dets:");
        wf.print();

        let dav_eps: f64 = 1e-9; // Davidson convergence epsilon
        dense_optimize(wf, dav_eps, &ham, &excite_gen);

        println!("End of iteration {}: Wavefunction has {} determinants with energy {:.4}", iter, wf.n, wf.energy);

        if wf.n <= 200 {
            wf.print();
        }

        if wf.n > 10 {
            break;
        }
    }

}
