// Variational stage

use super::utils::read_input::Global;
use super::ham::Ham;
use super::wf::Wf;
use crate::excite::ExciteGenerator;


pub fn variational(ham: &Ham, excite_gen: &ExciteGenerator, wf: &mut Wf) {

    let mut iter: i32 = 0;

    println!("Start of variational stage: Wavefunction has {} determinants with energy {}", wf.n, wf.energy);

    while !wf.converged {

        wf.add_new_dets(&ham, &excite_gen);

        wf.optimize(&ham, &excite_gen);

        iter += 1;

        println!("End of iteration {}: Wavefunction has {} determinants with energy {}", iter, wf.n, wf.energy);

        if wf.n <= 20 {
            wf.print();
        }

    }

}
