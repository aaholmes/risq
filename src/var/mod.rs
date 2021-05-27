// Variational stage

mod davidson;

use super::ham::Ham;
use super::wf::Wf;
use crate::excite::init::ExciteGenerator;
// use crate::var::davidson::optimize;


pub fn variational(ham: &Ham, excite_gen: &ExciteGenerator, wf: &mut Wf) {

    let mut iter: i32 = 0;

    println!("Start of variational stage: Wavefunction has {} det with energy {:.4}", wf.n, wf.energy);

    while !wf.converged {

        iter += 1;

        wf.get_new_dets(&ham, &excite_gen);

        println!("Wf after add_new_dets:");
        wf.print();

        // optimize(wf, &ham, &excite_gen);

        println!("End of iteration {}: Wavefunction has {} determinants with energy {:.4}", iter, wf.n, wf.energy);

        if wf.n <= 20 {
            wf.print();
        }

        break;
    }

}
