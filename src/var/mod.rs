// Variational stage

use super::utils::read_input::Global;
use super::ham::Ham;
use super::wf::Wf;
use crate::excite::ExciteGenerator;

pub fn variational(global: &Global, ham: &Ham, excite_gen: &ExciteGenerator, wf: &mut Wf) {
    for (iter, eps) in wf.eps_iter.enumerate() {
        println!("Initial wf:");
        wf.print();

        println!("Starting variation {} with epsilon {}", iter, eps);
        // Get new dets
        // wf.get_new_dets(excite_gen);

        println!("After first iteration of getting new dets:");
        wf.print();

        // Compute variational H
        //var_h = compute_var_h(wf);

        // Minimize energy using davidson
        //davidson(var_h, wf);
        //println!("End of iteration {}: Wavefunction has {} determinants with energy {}", iter, wf.n, wf.energy);
    }
}