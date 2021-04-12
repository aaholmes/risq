// Variational stage

// Here, put in the logic for performing the variational stage

use super::utils::read_input::Global;
use super::ham::Ham;
use super::wf::Wf;

pub fn variational(global: &Global, ham: &Ham, wf: &Wf) {
    for (iter, eps) in wf.eps_iter.enumerate() {




        println!("End of variation {}: Wavefunction has {} determinants with energy {}", iter, wf.n, wf.energy);
    }
}